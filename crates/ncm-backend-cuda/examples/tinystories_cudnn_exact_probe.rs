//! Experimental upstream-layout cuDNN convolution; not a production backend.
use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::{
        cudnn::{
            safe::{ConvForward, Cudnn},
            sys,
        },
        driver::CudaSlice,
    },
};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Layout, Shape, Tensor};
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::Artifact;
use ncm_model::Codec;
use safetensors::{Dtype, SafeTensors};
struct Conv;
impl CustomOp2 for Conv {
    fn name(&self) -> &'static str {
        "cudnn-upstream-layout-probe"
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
        if xl.dims() != [128, 384, 128] || wl.dims() != [384, 384, 3] {
            candle_core::bail!("unexpected diagnostic shapes")
        }
        let (a, b) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("input must be contiguous".into()))?;
        let (c, d) = wl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("weights must be contiguous".into()))?;
        let xs = x.as_cuda_slice::<f32>()?;
        let ws = w.as_cuda_slice::<f32>()?;
        if b - a != 128 * 384 * 128 || d - c != 384 * 384 * 3 || b > xs.len() || d > ws.len() {
            candle_core::bail!("storage extent mismatch")
        }
        let xs = xs.slice(a..b);
        let ws = ws.slice(c..d);
        let device = &x.device;
        let handle = Cudnn::new(device.cuda_stream()).map_err(candle_core::Error::wrap)?;
        let mut desc = handle
            .create_conv2d::<f32>(
                [0, 1],
                [1, 1],
                [1, 1],
                sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .map_err(candle_core::Error::wrap)?;
        desc.set_math_type(sys::cudnnMathType_t::CUDNN_FMA_MATH)
            .map_err(candle_core::Error::wrap)?;
        let xd = handle
            .create_4d_tensor::<f32>(
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [128, 384, 1, 128],
            )
            .map_err(candle_core::Error::wrap)?;
        let wd = handle
            .create_4d_filter::<f32>(
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [384, 384, 1, 3],
            )
            .map_err(candle_core::Error::wrap)?;
        let yd = handle
            .create_4d_tensor::<f32>(
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [128, 384, 1, 128],
            )
            .map_err(candle_core::Error::wrap)?;
        let op = ConvForward {
            conv: &desc,
            x: &xd,
            w: &wd,
            y: &yd,
        };
        let algorithm = op.pick_algorithm().map_err(candle_core::Error::wrap)?;
        let bytes = op
            .get_workspace_size(algorithm)
            .map_err(candle_core::Error::wrap)?;
        eprintln!("algorithm={algorithm:?} workspace_bytes={bytes}");
        let mut workspace = device
            .cuda_stream()
            .alloc_zeros::<u8>(bytes)
            .map_err(candle_core::Error::wrap)?;
        // SAFETY: descriptors and device slices have exactly the checked FP32 shapes above.
        let mut out =
            unsafe { device.alloc::<f32>(128 * 384 * 128) }.map_err(candle_core::Error::wrap)?;
        // SAFETY: checked allocations, descriptor dimensions, dtype and workspace match this fixed diagnostic.
        unsafe {
            op.launch::<CudaSlice<u8>, _, _, _>(
                algorithm,
                Some(&mut workspace),
                (1.0, 0.0),
                &xs,
                &ws,
                &mut out,
            )
        }
        .map_err(candle_core::Error::wrap)?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: device.clone(),
            },
            vec![128, 384, 128].into(),
        ))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: tinystories_cudnn_exact_probe ARTIFACT TRACE".into());
    }
    let bytes = std::fs::read(&args[2])?;
    let trace = SafeTensors::deserialize(&bytes)?;
    let x = trace.tensor("cascade.3")?;
    let expected = trace.tensor("residual.conv")?;
    if x.dtype() != Dtype::F32 || x.shape() != [128, 128, 384] || expected.shape() != x.shape() {
        return Err("invalid trace".into());
    }
    let values: Vec<f32> = x
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let backend = CandleBackend::cuda(0)?;
    let artifact = Artifact::load(&args[1])?;
    let model = Codec::from_artifact(backend.clone(), &artifact)?;
    let weight = model
        .weights()
        .get("vq.quant_resi.0.weight")
        .ok_or("missing weight")?;
    let bias = model
        .weights()
        .get("vq.quant_resi.0.bias")
        .ok_or("missing bias")?;
    let sequence = backend.upload_f32(&values, &[128, 128, 384])?;
    let input = sequence.transpose(1, 2)?.contiguous()?;
    let out: Tensor = input
        .apply_op2_no_bwd(weight, &Conv)?
        .transpose(1, 2)?
        .broadcast_add(bias)?;
    let actual = backend.read_f32(&out)?;
    let mut max = 0f64;
    let mut sq = 0f64;
    let mut different = 0usize;
    for (a, b) in actual.iter().zip(expected.data().as_chunks::<4>().0) {
        let b = f32::from_le_bytes(*b);
        let d = (f64::from(*a) - f64::from(b)).abs();
        max = max.max(d);
        sq += d * d;
        different += usize::from(*a != b);
    }
    let mut timings = std::collections::BTreeMap::new();
    for mode in ["current_layout", "upstream_layout"] {
        let mut samples = Vec::new();
        for trial in 0..35 {
            backend.device().synchronize()?;
            let start = std::time::Instant::now();
            let output = if mode == "current_layout" {
                backend.conv1d(&sequence, weight, bias, 1, 1, false)?
            } else {
                input
                    .apply_op2_no_bwd(weight, &Conv)?
                    .transpose(1, 2)?
                    .broadcast_add(bias)?
            };
            backend.evaluate(&[output])?;
            if trial >= 5 {
                samples.push(start.elapsed().as_secs_f64());
            }
        }
        timings.insert(mode, samples);
    }
    println!(
        "{}",
        serde_json::json!({"elements":actual.len(),"max_abs":max,"rms":(sq/actual.len() as f64).sqrt(),"different":different,"math":"CUDNN_FMA_MATH","layout":"B,C,1,L with padding 0,1","timings_seconds":timings})
    );
    Ok(())
}
