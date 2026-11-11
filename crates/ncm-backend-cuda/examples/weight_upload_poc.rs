//! Isolated checked-weight upload comparison; not full checked-load acceptance.
use candle_core::{DType, Device, Tensor};
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use safetensors::SafeTensors;
use serde_json::json;
use std::{error::Error, fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn raw_upload(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    // The pinned native conversion assumes integral element bytes, including in
    // its unaligned-copy branch. Check before calling it, not after allocation.
    let count = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if !cfg!(target_endian = "little")
        || shape.is_empty()
        || shape.contains(&0)
        || count.and_then(|n| n.checked_mul(4)) != Some(bytes.len())
    {
        return Err("PoC requires a bounded nonempty little-endian FP32 shape/buffer".into());
    }
    Ok(Tensor::from_raw_buffer(bytes, DType::F32, shape, device)?)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || cfg!(debug_assertions) {
        return Err("usage (release): weight_upload_poc CANONICAL_BUNDLE NEW_REPORT".into());
    }
    if Path::new(&args[2]).exists() {
        return Err("output exists".into());
    }
    let manifest = fs::read(Path::new(&args[1]).join("manifest.json"))?;
    if manifest
        != include_bytes!(
            "../../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
        )
    {
        return Err("unpinned artifact manifest".into());
    }
    let artifact = Artifact::load(&args[1])?;
    let backend = CandleBackend::cuda(0)?;
    // Full existing validation authenticates/preflights all weights before the
    // isolated experiment. Keep this reference model for byte-exact readback.
    let model = Codec::from_artifact(backend.clone(), &artifact)?;
    let source = SafeTensors::deserialize(artifact.file("codec.safetensors")?)?;
    let views: Vec<_> = model
        .weights()
        .keys()
        .map(|name| source.tensor(name))
        .collect::<std::result::Result<_, _>>()?;
    let upload = |raw: bool| -> Result<Vec<Tensor>> {
        views
            .iter()
            .map(|view| {
                if raw {
                    raw_upload(view.data(), view.shape(), backend.device())
                } else {
                    let values: Vec<f32> = view
                        .data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    Ok(backend.upload_f32(&values, view.shape())?)
                }
            })
            .collect()
    };
    for raw in [false, true] {
        let outputs = upload(raw)?;
        backend.evaluate(&outputs)?;
        for (actual, expected) in outputs.iter().zip(model.weights().values()) {
            if backend
                .read_f32(actual)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
                != backend
                    .read_f32(expected)?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            {
                return Err("upload readback differs".into());
            }
        }
    }
    let mut old = Vec::new();
    let mut raw = Vec::new();
    for trial in 0..35 {
        for mode in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            backend.evaluate(&[])?;
            let start = Instant::now();
            let tensors = upload(mode)?;
            backend.evaluate(&tensors)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            if trial >= 5 {
                if mode {
                    raw.push(elapsed);
                } else {
                    old.push(elapsed);
                }
            }
            drop(tensors);
        }
    }
    let report = json!({"schema_version":1,"scope":"Isolated upload of all validated runtime tensors; excludes authentication/preflight/model construction and destruction. Not full-load acceptance.",
        "runtime":"rust-candle-cuda-upload-poc","precision":"fp32","warmups":5,"trials":30,
        "tensor_count":views.len(),"tensor_bytes":views.iter().map(|v|v.data().len()).sum::<usize>(),
        "readback_exact":true,"decoded_vec_upload_ms":old,"native_raw_upload_ms":raw,
        "manifest_sha256":sha256(&manifest),"source_sha256":sha256(&fs::read("crates/ncm-backend-cuda/examples/weight_upload_poc.rs")?),
        "limitations":["Only native tensor readback is qualified, not a new production loader API.","No unsafe code added; raw shape/length guard is mandatory for pinned native conversion.","No claim of zero-copy GPU storage or reduced RSS from timings alone."]});
    fs::write(
        &args[2],
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_partial_elements_and_overflow_before_native_conversion() {
        for (bytes, shape) in [
            (&[0u8; 7][..], vec![1]),
            (&[0; 4][..], vec![2]),
            (&[0; 4][..], vec![usize::MAX, 2]),
            (&[][..], vec![0]),
        ] {
            assert!(raw_upload(bytes, &shape, &Device::Cpu).is_err());
        }
    }
    #[test]
    fn aligned_and_unaligned_uploads_own_identical_bits() {
        let bits = [1.25f32.to_bits(), (-0.0f32).to_bits()];
        let bytes: Vec<_> = bits.into_iter().flat_map(u32::to_le_bytes).collect();
        for alignment in 0..4 {
            let mut storage = vec![0u8; bytes.len() + 4];
            let offset = (0..4)
                .find(|&i| (storage.as_ptr() as usize + i) % 4 == alignment)
                .unwrap();
            storage[offset..offset + bytes.len()].copy_from_slice(&bytes);
            let tensor =
                raw_upload(&storage[offset..offset + bytes.len()], &[2], &Device::Cpu).unwrap();
            storage.fill(0);
            assert_eq!(
                tensor
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                bits
            );
        }
    }
}
