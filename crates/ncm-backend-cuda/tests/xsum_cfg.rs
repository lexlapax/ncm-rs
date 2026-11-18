//! Exercise explicit CPU and device final selection on the same guided model.
#![cfg(feature = "cuda-cudnn")]

use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::XsumTeacher;
use safetensors::{Dtype, SafeTensors};

fn reference_ids(
    data: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let tensor = data.tensor(name)?;
    if tensor.dtype() != Dtype::I64 || tensor.shape() != shape {
        return Err(format!("unexpected {name} shape or dtype").into());
    }
    Ok(tensor
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| u32::try_from(i64::from_le_bytes(*bytes)))
        .collect::<Result<_, _>>()?)
}

#[test]
#[ignore = "requires the pinned XSum artifact and retained scale-one CUDA trace"]
fn guided_cpu_and_device_final_selection_match_reference() -> Result<(), Box<dyn std::error::Error>>
{
    let artifact = Artifact::load_inference(&std::env::var("NCM_XSUM_CFG_ARTIFACT")?)?;
    let trace = std::fs::read(std::env::var("NCM_XSUM_CFG_TRACE")?)?;
    let reference = SafeTensors::deserialize(&trace)?;
    let batch = 8;
    let prefixes = reference_ids(&reference, "generator.prefix_ids", &[batch, 512])?;
    let codes = reference_ids(&reference, "generator.selected_codes", &[batch, 200])?;
    let ids = reference_ids(&reference, "generator.penalized_ids", &[batch, 48])?;
    let backend = CandleBackend::cuda_cudnn(0)?;
    let model = XsumTeacher::from_owned_artifact(backend.clone(), artifact)?;
    let cpu = model.generate_top1_cfg_cpu(&prefixes, batch, 1.0)?;
    let device = model.generate_top1_cfg_device(&prefixes, batch, 1.0)?;
    assert_eq!(backend.read_indices(&cpu.codes)?, codes);
    assert_eq!(backend.read_indices(&device.codes)?, codes);
    assert_eq!(cpu.ids, ids);
    assert_eq!(device.ids, ids);
    assert!(
        model
            .generate_top1_cfg_cpu(&prefixes, batch, f64::NAN)
            .is_err()
    );
    assert!(model.generate_top1_cfg_cpu(&prefixes, batch, -1.0).is_err());
    assert!(model.generate_top1_cfg_cpu(&prefixes, 65, 1.0).is_err());
    Ok(())
}
