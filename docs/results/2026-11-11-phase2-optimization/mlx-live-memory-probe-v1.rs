//! Paired retained-root MLX memory observation; not a latency benchmark.
use mlx_rs::memory;
use ncm_backend_mlx::{MlxBackend, MlxTensor};
use ncm_core::{CodecBackend, WeightBackend};
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, thread::sleep, time::Duration};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn snap() -> Result<Value> {
    Ok(
        json!({"active":memory::active_memory()?,"cache":memory::cache_memory()?,"peak":memory::peak_memory()?}),
    )
}
fn data(batch: usize, levels: &[usize]) -> (Vec<u32>, Vec<u32>) {
    let mut tokens = Vec::new();
    let mut codes = Vec::new();
    for i in 0..batch {
        for row in 0..9 {
            if row > 0 {
                tokens.push(10);
            }
            for col in 0..9 {
                tokens.push(((row * 3 + row / 3 + col + i % 2) % 9 + 1) as u32);
            }
        }
        for (level, &length) in levels.iter().enumerate() {
            codes.extend((0..length).map(|p| ((i * length + p + 3 * level) % 16) as u32));
        }
    }
    (tokens, codes)
}
fn hash_ids(ids: &[u32]) -> String {
    sha256(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
fn call(
    model: &Codec<MlxBackend>,
    scope: &str,
    tokens: &[u32],
    codes: &[u32],
    batch: usize,
    latent: &MlxTensor,
) -> Result<(Vec<MlxTensor>, Vec<u32>, Option<MlxTensor>)> {
    let b = model.backend();
    match scope {
        "encode" => {
            let v = model.encode(tokens, batch)?;
            b.evaluate(&v.snapshots)?;
            Ok((v.snapshots, b.read_indices(&v.codes)?, None))
        }
        "decode" => {
            let v = model.decode(codes, batch)?;
            let mut f = v.snapshots;
            f.push(v.logits.clone());
            b.evaluate(&f)?;
            Ok((f, b.read_indices(&v.ids)?, Some(v.logits)))
        }
        "reconstruct" => {
            let (a, v) = model.reconstruct(tokens, batch)?;
            let mut f = a.snapshots;
            f.extend(v.snapshots);
            f.push(v.logits.clone());
            b.evaluate(&f)?;
            let mut ids = b.read_indices(&a.codes)?;
            ids.extend(b.read_indices(&v.ids)?);
            Ok((f, ids, Some(v.logits)))
        }
        "resident_decoder_block" => {
            let v = model.decoder_block(latent)?;
            b.evaluate(std::slice::from_ref(&v))?;
            Ok((vec![v], Vec::new(), None))
        }
        _ => Err("unsupported scope".into()),
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || cfg!(debug_assertions) || Path::new(&args[2]).exists() {
        return Err("usage (release): probe BUNDLE NEW_REPORT".into());
    }
    let artifact = Artifact::load(&args[1])?;
    let manifest_sha256 = sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?);
    let b = MlxBackend::gpu()?;
    let one = b.upload_f32(&[1.], &[1])?;
    b.evaluate(std::slice::from_ref(&one))?;
    let model = Codec::from_artifact(b.clone(), &artifact)?;
    b.evaluate(&model.weights().values().cloned().collect::<Vec<_>>())?;
    let mut rows = Vec::new();
    for batch in [1_usize, 2, 4] {
        let (tokens, codes) = data(batch, model.config().levels());
        let encoded = model.encode(&tokens, batch)?;
        let latent = encoded.snapshots.last().ok_or("missing latent")?.clone();
        b.evaluate(std::slice::from_ref(&latent))?;
        drop(encoded);
        let vals = b.read_f32(&latent)?;
        let latent_sha256 = sha256(
            &vals
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        drop(vals);
        for scope in ["encode", "decode", "reconstruct", "resident_decoder_block"] {
            memory::reset_peak_memory()?;
            let first = call(&model, scope, &tokens, &codes, batch, &latent)?;
            let hash = hash_ids(&first.1);
            if let Some(x) = &first.2 {
                drop(b.read_f32(x)?)
            }
            drop(first);
            for _ in 0..35 {
                let out = call(&model, scope, &tokens, &codes, batch, &latent)?;
                if hash_ids(&out.1) != hash {
                    return Err("nonrepeatable IDs".into());
                }
                drop(out);
            }
            let immediate = snap()?;
            let barrier = b.add(&one, &one)?;
            b.evaluate(std::slice::from_ref(&barrier))?;
            drop(barrier);
            let synchronized = snap()?;
            sleep(Duration::from_millis(10));
            let after_10ms = snap()?;
            sleep(Duration::from_millis(90));
            let after_100ms = snap()?;
            rows.push(json!({"batch":batch,"scope":scope,"input_ids_sha256":hash_ids(&tokens),"supplied_codes_sha256":hash_ids(&codes),"output_ids_sha256":hash,"latent_sha256":latent_sha256,"immediate":immediate,"synchronized":synchronized,"after_10ms":after_10ms,"after_100ms":after_100ms}));
        }
        drop(latent);
    }
    drop(model);
    drop(one);
    let input = b.upload_f32(&[1.], &[1])?;
    let barrier = b.add(&input, &input)?;
    b.evaluate(std::slice::from_ref(&barrier))?;
    drop(barrier);
    drop(input);
    sleep(Duration::from_millis(100));
    let after_model_drop = snap()?;
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?,
        &json!({"schema_version":1,"runtime":"rust-mlx","scope":"finite retained-root memory diagnostic; not acceptance","manifest_sha256":manifest_sha256,"warmups_and_trials":35,"rows":rows,"after_model_drop":after_model_drop}),
    )?;
    Ok(())
}
