//! Offline resampling experiment, NOT the production codec or a performance claim.
//! Fixed-shape metadata plus gathers replace per-position dispatch. CPU validation
//! does not qualify CUDA numerics, trained traces or end-to-end performance.
use candle_core::{Device, Result, Tensor};
use ncm_backend_cuda::ops;
use serde_json::json;
use std::{error::Error, fs, path::Path};

struct Linear {
    left: Tensor,
    right: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
}
impl Linear {
    fn new(input: usize, output: usize, fp32_coordinates: bool, device: &Device) -> Result<Self> {
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut weight = Vec::new();
        let mut complement = Vec::new();
        for i in 0..output {
            let (l, w, other) = if fp32_coordinates {
                let scale = input as f32 / output as f32;
                let source = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
                let l = (source.floor() as usize).min(input - 1);
                let w = (source - l as f32).clamp(0.0, 1.0);
                (l, w, 1.0 - w)
            } else {
                let source = ((i as f64 + 0.5) * input as f64 / output as f64 - 0.5).max(0.0);
                let l = (source.floor() as usize).min(input - 1);
                let w = source - l as f64;
                (l, w as f32, (1.0 - w) as f32)
            };
            left.push(l as u32);
            right.push((l + 1).min(input - 1) as u32);
            weight.push(w);
            complement.push(other);
        }
        Ok(Self {
            left: Tensor::from_vec(left, output, device)?,
            right: Tensor::from_vec(right, output, device)?,
            left_weight: Tensor::from_vec(complement, (1, output, 1), device)?,
            right_weight: Tensor::from_vec(weight, (1, output, 1), device)?,
        })
    }
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        x.index_select(&self.left, 1)?
            .broadcast_mul(&self.left_weight)?
            + x.index_select(&self.right, 1)?
                .broadcast_mul(&self.right_weight)?
    }
}

struct Area {
    indices: Tensor,
    mask: Tensor,
    divisor: Tensor,
    output: usize,
    width: usize,
}
impl Area {
    fn new(input: usize, output: usize, device: &Device) -> Result<Self> {
        let bins: Vec<_> = (0..output)
            .map(|i| (i * input / output, ((i + 1) * input).div_ceil(output)))
            .collect();
        let width = bins.iter().map(|(a, b)| b - a).max().unwrap();
        let mut indices = Vec::new();
        let mut mask = Vec::new();
        let mut divisor = Vec::new();
        for (start, end) in bins {
            divisor.push((end - start) as f32);
            for i in 0..width {
                indices.push((start + i).min(end - 1) as u32);
                mask.push(if start + i < end { 1f32 } else { 0f32 });
            }
        }
        Ok(Self {
            indices: Tensor::from_vec(indices, output * width, device)?,
            mask: Tensor::from_vec(mask, (1, output, width, 1), device)?,
            divisor: Tensor::from_vec(divisor, (1, output, 1), device)?,
            output,
            width,
        })
    }
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _, channels) = x.dims3()?;
        let x = x.contiguous()?;
        x.index_select(&self.indices, 1)?
            .reshape((batch, self.output, self.width, channels))?
            .broadcast_mul(&self.mask)?
            .sum(2)?
            .broadcast_div(&self.divisor)
    }
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || Path::new(&args[3]).exists() {
        return Err("usage: resample_poc cpu|cuda f32|f64 NEW_REPORT (synthetic cases)".into());
    }
    let fp32_coordinates = match args[2].as_str() {
        "f32" => true,
        "f64" => false,
        _ => return Err("unsupported coordinate mode".into()),
    };
    let device = match args[1].as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)?,
        _ => return Err("unsupported device; no fallback".into()),
    };
    let mut rows = Vec::new();
    // All trained level lengths, tiny uneven ratios, identity and one-position axes.
    let lengths = [1, 2, 3, 4, 6, 7, 9, 13, 16, 18, 26, 37, 50, 67, 89];
    for batch in [1, 2, 4] {
        for length in lengths {
            for (input, output) in [(length, 89), (89, length)] {
                let channels = 16;
                let data: Vec<_> = (0..batch * input * channels)
                    .map(|i| ((i * 73 % 1021) as f32 - 510.) / 64.)
                    .collect();
                let x = Tensor::from_vec(data, (batch, input, channels), &device)?;
                // Include non-contiguous input; these plans must not assume BLC strides.
                for strided in [false, true] {
                    let x = if strided {
                        x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?
                    } else {
                        x.clone()
                    };
                    let linear = Linear::new(input, output, fp32_coordinates, &device)?;
                    let area = Area::new(input, output, &device)?;
                    for (kind, candidate, reference) in [
                        ("linear", linear.apply(&x)?, ops::linear1d(&x, output)?),
                        ("area", area.apply(&x)?, ops::area1d(&x, output)?),
                    ] {
                        let candidate = candidate.flatten_all()?.to_vec1::<f32>()?;
                        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
                        let mut max_abs = 0f32;
                        let mut mismatches = 0usize;
                        for (&actual, &expected) in candidate.iter().zip(&reference) {
                            if !actual.is_finite() {
                                return Err("nonfinite candidate".into());
                            }
                            let error = (actual - expected).abs();
                            mismatches += usize::from(error > 1e-5 + 1e-4 * expected.abs());
                            max_abs = max_abs.max(error);
                        }
                        rows.push(
                            json!({"batch":batch,"input_length":input,"output_length":output,
                            "channels":channels,"strided":strided,"kind":kind,
                            "max_abs_vs_composition":max_abs,"mismatches_vs_composition":mismatches,"output":candidate}),
                        );
                    }
                }
            }
        }
    }
    let report = json!({"schema_version":1,"device":args[1],"precision":"fp32","coordinate_math":args[2],
        "scope":"finite synthetic resampling PoC only; no timing or trained codec qualification",
        "source_sha256":ncm_io::sha256(include_bytes!("resample_poc.rs")),
        "rows":rows});
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    serde_json::to_writer(file, &report)?;
    println!(
        "{} synthetic rows captured; independent Torch check required",
        rows.len()
    );
    Ok(())
}
