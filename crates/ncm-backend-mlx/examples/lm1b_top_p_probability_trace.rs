//! Same-input experimental Metal probability trace; no qualification claim.
use ncm_backend_mlx::MlxBackend;
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::sha256;
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, serialize},
};
use std::{collections::BTreeMap, error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: lm1b_top_p_probability_trace FIXTURE OUTPUT".into());
    }
    if Path::new(&args[2]).exists() {
        return Err("output exists".into());
    }
    let raw = fs::read(&args[1])?;
    if sha256(&raw) != "8175af98005430f1aebd35f1e1b4965cb3b9889aafe5fd47853d49bdbaa5ccd9" {
        return Err("unexpected fixture".into());
    }
    let fixture = SafeTensors::deserialize(&raw)?;
    let backend = MlxBackend::gpu()?;
    let mut all = BTreeMap::new();
    for name in ["cuda", "metal", "ties"] {
        let input = fixture.tensor(&format!("{name}_logits"))?;
        if input.dtype() != Dtype::F32
            || input.shape().len() != 2
            || input.shape()[1] != 8192
            || !(1..=128).contains(&input.shape()[0])
        {
            return Err("unexpected logits".into());
        }
        let values: Vec<f32> = input
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let logits = backend.upload_f32(&values, &[1, input.shape()[0], 8192])?;
        let probabilities = backend.lm1b_top_p_one_probabilities_experimental(&logits)?;
        let bytes: Vec<u8> = backend
            .read_f32(&probabilities)?
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        all.insert(format!("{name}_probs"), (input.shape().to_vec(), bytes));
    }
    let views = all
        .iter()
        .map(|(name, (shape, bytes))| {
            Ok((name, TensorView::new(Dtype::F32, shape.clone(), bytes)?))
        })
        .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
    fs::write(&args[2], serialize(views, None)?)?;
    Ok(())
}
