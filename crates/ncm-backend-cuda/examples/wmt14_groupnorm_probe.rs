//! Check reference-order WMT14 group normalization against pinned CUDA inputs.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::sha256;
use safetensors::{Dtype, SafeTensors};
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: wmt14_groupnorm_probe PYTHON_TRACE".into());
    }
    let bytes = fs::read(&args[1])?;
    if sha256(&bytes) != "b8795991563f379eb2e6c64e6d7fb886d24bff45fcce6b6f6bcacda5ee0a0a94" {
        return Err("unexpected reference trace".into());
    }
    let tensors = SafeTensors::deserialize(&bytes)?;
    let backend = CandleBackend::cuda_wmt14_reference(0)?;
    let invalid = backend.upload_f32(&[0.0], &[1, 1, 1])?;
    if backend
        .wmt14_initial_refinement(&invalid, &invalid, &invalid)
        .is_ok()
    {
        return Err("invalid initial refinement shape was accepted".into());
    }
    let mut cases = 0;
    for (stage, length) in [(1, 4), (3, 16)] {
        let input = tensors.tensor(&format!("diagnostic.cascade_{stage:02}.input"))?;
        let expected = tensors.tensor(&format!("diagnostic.cascade_{stage:02}.output"))?;
        if input.dtype() != Dtype::F32
            || expected.dtype() != Dtype::F32
            || input.shape() != [8, length, 384]
            || expected.shape() != input.shape()
        {
            return Err("unexpected reference shape".into());
        }
        let values: Vec<f32> = input
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let expected_values: Vec<f32> = expected
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        for batch in [1, 4, 8, 32] {
            let width = length * 384;
            let input_rows: Vec<f32> = (0..batch)
                .flat_map(|i| values[(i % 8) * width..(i % 8 + 1) * width].iter().copied())
                .collect();
            let expected_rows: Vec<f32> = (0..batch)
                .flat_map(|i| {
                    expected_values[(i % 8) * width..(i % 8 + 1) * width]
                        .iter()
                        .copied()
                })
                .collect();
            let actual = backend.group_norm(
                &backend.upload_f32(&input_rows, &[batch, length, 384])?,
                8,
                1e-5,
            )?;
            if backend
                .read_f32(&actual)?
                .iter()
                .map(|x| x.to_bits())
                .ne(expected_rows.iter().map(|x| x.to_bits()))
            {
                return Err(format!("groupnorm mismatch: length {length}, batch {batch}").into());
            }
            cases += 1;
        }
    }
    println!("{cases} exact WMT14 reference groupnorm cases");
    Ok(())
}
