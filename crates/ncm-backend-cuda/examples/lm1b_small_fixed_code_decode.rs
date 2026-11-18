//! Decode retained LM1B-small codes without running the generator or sampler.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, GeneratorBackend, WeightBackend};
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use safetensors::{Dtype, SafeTensors};
use std::{error::Error, fs, path::Path};

const BATCH: usize = 8;
const CODE_WIDTH: usize = 641;
const IDS_WIDTH: usize = 128;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact, input, input_sha, options @ ..] = args.as_slice() else {
        return Err(
            "usage: lm1b_small_fixed_code_decode ARTIFACT CODES INPUT_SHA256 [cudnn-default-math|cudnn-graph|cudnn-graph-efficient]"
                .into(),
        );
    };
    let mode = match options {
        [] => "",
        [option]
            if ["cudnn-default-math", "cudnn-graph", "cudnn-graph-efficient"]
                .contains(&option.as_str()) =>
        {
            option
        }
        _ => return Err("invalid fixed-code decode option".into()),
    };
    if sha256(&fs::read(Path::new(artifact).join("manifest.json"))?)
        != "58969517838438b12811f48a6d34bf81cd09a56c6eaa1c8ab5fd65993b6ecd76"
    {
        return Err("unexpected LM1B-small artifact manifest".into());
    }
    let data = fs::read(input)?;
    if sha256(&data) != *input_sha {
        return Err("fixed-code input differs".into());
    }
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&data)?;
    if rows.is_empty()
        || rows.len() > 2560
        || !rows.len().is_multiple_of(BATCH)
        || rows
            .iter()
            .any(|row| row.len() != CODE_WIDTH || row.iter().any(|&code| code >= 8192))
    {
        return Err("invalid fixed-code rows".into());
    }
    let artifact = Artifact::load_inference(artifact)?;
    let tensors = SafeTensors::deserialize(artifact.file("inference.safetensors")?)?;
    let view = tensors.tensor("token_unigram")?;
    if view.dtype() != Dtype::F32 || view.shape() != [30528] {
        return Err("invalid LM1B-small allowance".into());
    }
    let allowance: Vec<_> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    let backend = match mode {
        "cudnn-graph-efficient" => CandleBackend::cuda_lm1b_efficient_experimental(0)?,
        "cudnn-graph" => CandleBackend::cuda_lm1b_cudnn_graph_experimental(0)?,
        "cudnn-default-math" => CandleBackend::cuda_lm1b_cudnn_default_math_experimental(0)?,
        _ => CandleBackend::cuda_lm1b_qkv_reference_experimental(0)?,
    };
    let allowance = backend.upload_f32(&allowance, &[30528])?;
    let codec = Codec::from_lm1b_small_artifact_experimental(backend.clone(), &artifact)?;
    let mut output = Vec::with_capacity(rows.len());
    for chunk in rows.as_chunks::<BATCH>().0 {
        let codes: Vec<_> = chunk.iter().flatten().copied().collect();
        let decoded = codec.decode(&codes, BATCH)?;
        let selected = backend.lm1b_penalty_fp64(&decoded.logits, &allowance)?;
        let ids = backend.read_indices(&selected)?;
        if ids.len() != BATCH * IDS_WIDTH || ids.iter().any(|&id| id >= 30528) {
            return Err("invalid replayed LM1B-small IDs".into());
        }
        output.extend(
            ids.as_chunks::<IDS_WIDTH>()
                .0
                .iter()
                .map(|row| row.to_vec()),
        );
    }
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
