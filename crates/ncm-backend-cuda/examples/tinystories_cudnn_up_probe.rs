//! Diagnostic upstream-layout cuDNN transposed convolution, with strict FP32 math.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::{
        cudnn::{
            safe::{ConvBackwardData, Cudnn},
            sys,
        },
        driver::CudaSlice,
    },
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Shape};
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::Artifact;
use ncm_model::Codec;
use safetensors::{Dtype, SafeTensors};
struct Up;
impl CustomOp2 for Up {
    fn name(&self) -> &'static str {
        "cudnn-upstream-transpose-probe"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA only")
    }
    #[allow(unsafe_code)]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let [batch, channels, length] = xl.dims() else {
            candle_core::bail!("invalid input")
        };
        let [cin, cout, kernel] = wl.dims() else {
            candle_core::bail!("invalid weights")
        };
        if *batch != 2
            || *channels != 384
            || *cin != 384
            || *cout != 384
            || ![2, 4].contains(kernel)
            || ![1, 2, 8, 32].contains(length)
        {
            candle_core::bail!("invalid diagnostic shape")
        }
        let (a, b) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("input must be contiguous".into()))?;
        let (c, d) = wl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("weight must be contiguous".into()))?;
        let xs = x.as_cuda_slice::<f32>()?;
        let ws = w.as_cuda_slice::<f32>()?;
        if b - a != batch * channels * length
            || d - c != cin * cout * kernel
            || b > xs.len()
            || d > ws.len()
        {
            candle_core::bail!("storage extent mismatch")
        }
        let xs = xs.slice(a..b);
        let ws = ws.slice(c..d);
        let device = &x.device;
        let output_len = length * kernel;
        let handle = Cudnn::new(device.cuda_stream()).map_err(candle_core::Error::wrap)?;
        let mut desc = handle
            .create_conv2d::<f32>(
                [0, 0],
                [1, *kernel as i32],
                [1, 1],
                sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .map_err(candle_core::Error::wrap)?;
        desc.set_math_type(sys::cudnnMathType_t::CUDNN_FMA_MATH)
            .map_err(candle_core::Error::wrap)?;
        let dx = handle
            .create_4d_tensor::<f32>(
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [*batch as i32, 384, 1, output_len as i32],
            )
            .map_err(candle_core::Error::wrap)?;
        let wd = handle
            .create_4d_filter::<f32>(
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [384, 384, 1, *kernel as i32],
            )
            .map_err(candle_core::Error::wrap)?;
        let dy = handle
            .create_4d_tensor::<f32>(
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [*batch as i32, 384, 1, *length as i32],
            )
            .map_err(candle_core::Error::wrap)?;
        let op = ConvBackwardData {
            conv: &desc,
            dx: &dx,
            w: &wd,
            dy: &dy,
        };
        let algorithm = op.pick_algorithm().map_err(candle_core::Error::wrap)?;
        let size = op
            .get_workspace_size(algorithm)
            .map_err(candle_core::Error::wrap)?;
        eprintln!("length={length} stride={kernel} algorithm={algorithm:?} workspace_bytes={size}");
        let mut workspace = device
            .cuda_stream()
            .alloc_zeros::<u8>(size)
            .map_err(candle_core::Error::wrap)?;
        // SAFETY: the FP32 output length and all cuDNN descriptors are checked above.
        let mut out = unsafe { device.alloc::<f32>(batch * 384 * output_len) }
            .map_err(candle_core::Error::wrap)?;
        // SAFETY: input, filter, output and workspace have the exact described extents.
        unsafe {
            op.launch::<CudaSlice<u8>, _, _, _>(
                algorithm,
                Some(&mut workspace),
                (1.0, 0.0),
                &mut out,
                &ws,
                &xs,
            )
        }
        .map_err(candle_core::Error::wrap)?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: device.clone(),
            },
            vec![*batch, 384, output_len].into(),
        ))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: tinystories_cudnn_up_probe ARTIFACT TRACE".into());
    }
    let bytes = std::fs::read(&args[2])?;
    let reference = SafeTensors::deserialize(&bytes)?;
    let emb = reference.tensor("embedding")?;
    if emb.dtype() != Dtype::F32 || emb.shape() != [2, 1, 384] {
        return Err("invalid trace".into());
    }
    let values: Vec<f32> = emb
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let backend = CandleBackend::cuda(0)?;
    let artifact = Artifact::load(&args[1])?;
    let codec = Codec::from_artifact(backend.clone(), &artifact)?;
    let mut x = backend.upload_f32(&values, &[2, 1, 384])?;
    let mut results = std::collections::BTreeMap::new();
    for i in 0..4 {
        let w = codec
            .weights()
            .get(&format!("vq.learned_up_l1.{i}.weight"))
            .ok_or("missing weight")?;
        let b = codec
            .weights()
            .get(&format!("vq.learned_up_l1.{i}.bias"))
            .ok_or("missing bias")?;
        x = x
            .transpose(1, 2)?
            .contiguous()?
            .apply_op2_no_bwd(w, &Up)?
            .transpose(1, 2)?
            .broadcast_add(b)?;
        let expected = reference.tensor(&format!("cascade.{i}"))?;
        let actual = backend.read_f32(&x)?;
        if expected.shape() != backend.shape(&x) {
            return Err("shape mismatch".into());
        }
        let mut max = 0f64;
        let mut sq = 0f64;
        let mut diff = 0usize;
        for (a, b) in actual.iter().zip(expected.data().as_chunks::<4>().0) {
            let b = f32::from_le_bytes(*b);
            let d = (f64::from(*a) - f64::from(b)).abs();
            max = max.max(d);
            sq += d * d;
            diff += usize::from(*a != b)
        }
        results.insert(i,serde_json::json!({"elements":actual.len(),"different":diff,"max_abs":max,"rms":(sq/actual.len() as f64).sqrt()}));
    }
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
