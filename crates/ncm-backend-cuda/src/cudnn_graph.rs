//! Bounded cuDNN graph forward plans for the experimental LM1B CUDA backend.
//! Descriptor contracts: NVIDIA cuDNN Backend API, cudnn_graph library.
#![allow(unsafe_code)]

use candle_core::cuda_backend::cudarc::{
    cudnn::sys::{
        self, cudnnBackendAttributeName_t::*, cudnnBackendAttributeType_t::*,
        cudnnBackendDescriptorType_t::*, cudnnBackendHeurMode_t::*, cudnnStatus_t::*,
    },
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
};
use std::{
    collections::VecDeque,
    error::Error,
    ffi::c_void,
    ptr::null_mut,
    sync::{Arc, Mutex},
};

const MAX_PLANS: usize = 16;
const MAX_WORKSPACE: usize = 128 * 1024 * 1024;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn check(status: sys::cudnnStatus_t) -> Result<()> {
    if status == CUDNN_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{status:?}").into())
    }
}

struct Desc(sys::cudnnBackendDescriptor_t);

impl Desc {
    fn new(kind: sys::cudnnBackendDescriptorType_t) -> Result<Self> {
        let mut raw = null_mut();
        check(unsafe { sys::cudnnBackendCreateDescriptor(kind, &mut raw) })?;
        Ok(Self(raw))
    }

    fn set<T>(
        &self,
        name: sys::cudnnBackendAttributeName_t,
        ty: sys::cudnnBackendAttributeType_t,
        values: &[T],
    ) -> Result<()> {
        check(unsafe {
            sys::cudnnBackendSetAttribute(
                self.0,
                name,
                ty,
                values.len() as i64,
                values.as_ptr().cast(),
            )
        })
        .map_err(|err| format!("setting {name:?}: {err}").into())
    }

    fn get<T>(
        &self,
        name: sys::cudnnBackendAttributeName_t,
        ty: sys::cudnnBackendAttributeType_t,
        values: &mut [T],
    ) -> Result<i64> {
        let mut count = 0;
        let ptr = if values.is_empty() {
            null_mut()
        } else {
            values.as_mut_ptr().cast()
        };
        check(unsafe {
            sys::cudnnBackendGetAttribute(self.0, name, ty, values.len() as i64, &mut count, ptr)
        })
        .map_err(|err| format!("getting {name:?}: {err}"))?;
        Ok(count)
    }

    fn finalize(&self) -> Result<()> {
        check(unsafe { sys::cudnnBackendFinalize(self.0) })
    }
}

impl Drop for Desc {
    fn drop(&mut self) {
        unsafe { sys::cudnnBackendDestroyDescriptor(self.0) };
    }
}

struct Handle(sys::cudnnHandle_t);
impl Drop for Handle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: only successful initialization transfers a handle here.
            unsafe { sys::cudnnDestroy(self.0) };
        }
    }
}

fn tensor(id: i64, shape: [i64; 4], alignment: i64) -> Result<Desc> {
    let desc = Desc::new(CUDNN_BACKEND_TENSOR_DESCRIPTOR)?;
    let [_, c, h, w] = shape;
    let strides = [c * h * w, h * w, w, 1];
    desc.set(CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, &[id])?;
    desc.set(CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, &shape)?;
    desc.set(CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, &strides)?;
    desc.set(
        CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
        CUDNN_TYPE_INT64,
        &[alignment],
    )?;
    desc.set(
        CUDNN_ATTR_TENSOR_DATA_TYPE,
        CUDNN_TYPE_DATA_TYPE,
        &[sys::cudnnDataType_t::CUDNN_DATA_FLOAT],
    )?;
    desc.finalize()?;
    Ok(desc)
}

struct Plan {
    execution: Desc,
    _descriptors: [Desc; 7],
    workspace_bytes: usize,
}

impl Plan {
    fn new(handle: &Handle, batch: i64, width: i64) -> Result<Self> {
        let x_desc = tensor(1, [batch, 384, 1, width], 16)?;
        let w_desc = tensor(2, [384, 384, 1, 3], 16)?;
        let y_desc = tensor(3, [batch, 384, 1, width], 16)?;
        let conv = Desc::new(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR)?;
        conv.set(
            CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
            CUDNN_TYPE_INT64,
            &[2i64],
        )?;
        conv.set(
            CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
            CUDNN_TYPE_DATA_TYPE,
            &[sys::cudnnDataType_t::CUDNN_DATA_FLOAT],
        )?;
        conv.set(
            CUDNN_ATTR_CONVOLUTION_CONV_MODE,
            CUDNN_TYPE_CONVOLUTION_MODE,
            &[sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION],
        )?;
        conv.set(
            CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
            CUDNN_TYPE_INT64,
            &[0i64, 1],
        )?;
        conv.set(
            CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
            CUDNN_TYPE_INT64,
            &[0i64, 1],
        )?;
        conv.set(
            CUDNN_ATTR_CONVOLUTION_DILATIONS,
            CUDNN_TYPE_INT64,
            &[1i64, 1],
        )?;
        conv.set(
            CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
            CUDNN_TYPE_INT64,
            &[1i64, 1],
        )?;
        conv.finalize()?;

        let op = Desc::new(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)?;
        op.set(
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[x_desc.0],
        )?;
        op.set(
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[w_desc.0],
        )?;
        op.set(
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[y_desc.0],
        )?;
        op.set(
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[conv.0],
        )?;
        op.set(
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
            CUDNN_TYPE_FLOAT,
            &[1f32],
        )?;
        op.set(
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
            CUDNN_TYPE_FLOAT,
            &[0f32],
        )?;
        op.finalize()?;

        let graph = Desc::new(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)?;
        graph.set(
            CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
            CUDNN_TYPE_HANDLE,
            &[handle.0],
        )?;
        graph.set(
            CUDNN_ATTR_OPERATIONGRAPH_OPS,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[op.0],
        )?;
        graph.finalize()?;

        let heur = Desc::new(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)?;
        heur.set(
            CUDNN_ATTR_ENGINEHEUR_MODE,
            CUDNN_TYPE_HEUR_MODE,
            &[CUDNN_HEUR_MODE_INSTANT],
        )?;
        heur.set(
            CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[graph.0],
        )?;
        heur.finalize()?;
        let config = Desc::new(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)?;
        let mut found = [config.0];
        if heur.get(
            CUDNN_ATTR_ENGINEHEUR_RESULTS,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &mut found,
        )? != 1
            || found[0] != config.0
        {
            return Err("no graph convolution engine".into());
        }
        let plan = Desc::new(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)?;
        plan.set(
            CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
            CUDNN_TYPE_HANDLE,
            &[handle.0],
        )?;
        plan.set(
            CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &[config.0],
        )?;
        plan.finalize()?;
        let mut bytes = [0i64];
        plan.get(
            CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
            CUDNN_TYPE_INT64,
            &mut bytes,
        )?;
        let workspace_bytes = usize::try_from(bytes[0])?;
        if workspace_bytes > MAX_WORKSPACE {
            return Err("graph convolution workspace exceeds limit".into());
        }
        Ok(Self {
            execution: plan,
            _descriptors: [config, graph, op, conv, y_desc, w_desc, x_desc],
            workspace_bytes,
        })
    }

    fn launch(
        &self,
        handle: &Handle,
        stream: &CudaStream,
        x: &impl DevicePtr<f32>,
        weights: &impl DevicePtr<f32>,
        out: &mut CudaSlice<f32>,
        workspace: &mut CudaSlice<u8>,
    ) -> Result<()> {
        let (x_ptr, _input_access) = x.device_ptr(stream);
        let (weight_ptr, _weight_access) = weights.device_ptr(stream);
        let (out_ptr, _output_access) = out.device_ptr_mut(stream);
        let (workspace_ptr, _workspace_access) = workspace.device_ptr_mut(stream);
        if [x_ptr, weight_ptr, out_ptr].iter().any(|ptr| ptr % 16 != 0) {
            return Err("unaligned graph convolution input or output".into());
        }
        let data = [x_ptr, weight_ptr, out_ptr].map(|ptr| ptr as *mut c_void);
        let scratch = if self.workspace_bytes == 0 {
            null_mut()
        } else {
            workspace_ptr as *mut c_void
        };
        let pack = Desc::new(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)?;
        pack.set(
            CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
            CUDNN_TYPE_INT64,
            &[1i64, 2, 3],
        )?;
        pack.set(
            CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
            CUDNN_TYPE_VOID_PTR,
            &data,
        )?;
        pack.set(
            CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
            CUDNN_TYPE_VOID_PTR,
            &[scratch],
        )?;
        pack.finalize()?;
        // SAFETY: the Candle adapter verifies slice bounds against the finalized
        // plan. The CUDA guards record this stream's access to all device slices.
        // The variant pack describes the launch. The plan itself stays alive
        // until the stream is drained before eviction or cache destruction.
        check(unsafe { sys::cudnnBackendExecute(handle.0, self.execution.0, pack.0) })
    }
}

struct State {
    plans: VecDeque<((i64, i64), Plan)>,
    workspace: CudaSlice<u8>,
    handle: Handle,
    stream: Arc<CudaStream>,
}

// SAFETY: every cuDNN call on this handle and its descriptors is serialized by
// the cache mutex, and each use binds the owning CUDA context to the thread.
// Destruction has exclusive ownership and binds the same context.
unsafe impl Send for State {}

impl State {
    fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let mut raw = null_mut();
        // SAFETY: the owning context is current and the output slot is live.
        check(unsafe { sys::cudnnCreate(&mut raw) })?;
        let handle = Handle(raw);
        // SAFETY: the handle and stream belong to the same current context.
        check(unsafe { sys::cudnnSetStream(handle.0, stream.cu_stream().cast()) })?;
        Ok(Self {
            plans: VecDeque::new(),
            workspace: stream.alloc_zeros::<u8>(1)?,
            handle,
            stream: stream.clone(),
        })
    }

    fn execute(
        &mut self,
        shape: (i64, i64),
        x: &impl DevicePtr<f32>,
        weights: &impl DevicePtr<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let plan = if let Some(index) = self.plans.iter().position(|(key, _)| *key == shape) {
            self.plans.remove(index).expect("located graph plan").1
        } else {
            let plan = Plan::new(&self.handle, shape.0, shape.1)?;
            if self.plans.len() == MAX_PLANS {
                self.stream.synchronize()?;
                self.plans.pop_front();
            }
            plan
        };
        self.plans.push_back((shape, plan));
        let plan = &self.plans.back().expect("inserted graph plan").1;
        if self.workspace.len() < plan.workspace_bytes {
            // The old allocation was used on this same stream. Cudarc orders
            // its free behind those uses when replacing the slice.
            self.workspace = self.stream.alloc_zeros::<u8>(plan.workspace_bytes)?;
        }
        plan.launch(
            &self.handle,
            &self.stream,
            x,
            weights,
            out,
            &mut self.workspace,
        )
    }
}

impl Drop for State {
    fn drop(&mut self) {
        // The stream outlives this drop, and fields drop in declaration order.
        // A failed stream is already unusable by the caller.
        let _ = self.stream.synchronize();
    }
}

#[derive(Default)]
pub(crate) struct ForwardCache {
    state: Mutex<Option<State>>,
}
impl ForwardCache {
    pub(crate) fn execute(
        &self,
        stream: &Arc<CudaStream>,
        x: &impl DevicePtr<f32>,
        weights: &impl DevicePtr<f32>,
        out: &mut CudaSlice<f32>,
        batch: i64,
        width: i64,
    ) -> candle_core::Result<()> {
        let mut run = || -> Result<()> {
            stream.context().bind_to_thread()?;
            let mut state = self
                .state
                .lock()
                .map_err(|_| "cuDNN graph cache lock poisoned")?;
            if state.is_none() {
                *state = Some(State::new(stream)?);
            }
            let state = state.as_mut().expect("initialized graph state");
            if !Arc::ptr_eq(stream, &state.stream) {
                return Err("cuDNN graph cache stream changed".into());
            }
            state.execute((batch, width), x, weights, out)
        };
        run().map_err(|err| candle_core::Error::Msg(format!("LM1B cuDNN graph: {err}")))
    }
}
