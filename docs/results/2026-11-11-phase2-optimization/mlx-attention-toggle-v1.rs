//! Isolated same-process masked/unmasked attention codec comparison; not acceptance.
use ncm_backend_mlx::{MlxBackend, MlxTensor};
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use serde_json::json;
use std::{error::Error, fs::OpenOptions, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn tokens(batch: usize) -> Vec<u32> {
    let mut result = Vec::new();
    for i in 0..batch {
        for row in 0..9 {
            if row > 0 {
                result.push(10);
            }
            for col in 0..9 {
                result.push(((row * 3 + row / 3 + col + i % 2) % 9 + 1) as u32);
            }
        }
    }
    result
}
fn run_scope(
    model: &Codec<MlxBackend>,
    scope: &str,
    input: &[u32],
    codes: &[u32],
    batch: usize,
    latent: &MlxTensor,
) -> Result<(MlxTensor, Vec<u32>)> {
    let b = model.backend();
    if scope == "resident_decoder_block" {
        let y = model.decoder_block(latent)?;
        b.evaluate(std::slice::from_ref(&y))?;
        Ok((y, Vec::new()))
    } else if scope == "decode" {
        let decoded = model.decode(codes, batch)?;
        let mut tensors = decoded.snapshots;
        tensors.push(decoded.logits.clone());
        b.evaluate(&tensors)?;
        let ids = b.read_indices(&decoded.ids)?;
        Ok((decoded.logits, ids))
    } else if scope == "reconstruct" {
        let (encoded, decoded) = model.reconstruct(input, batch)?;
        let mut tensors = encoded.snapshots;
        tensors.extend(decoded.snapshots);
        tensors.push(decoded.logits.clone());
        b.evaluate(&tensors)?;
        let mut ids = b.read_indices(&encoded.codes)?;
        ids.extend(b.read_indices(&decoded.ids)?);
        Ok((decoded.logits, ids))
    } else {
        Err("unknown scope".into())
    }
}
fn hash_ids(ids: &[u32]) -> String {
    sha256(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || cfg!(debug_assertions) || Path::new(&args[2]).exists() {
        return Err("usage (release): diagnostic BUNDLE NEW_REPORT".into());
    }
    let b = MlxBackend::gpu()?;
    let artifact = Artifact::load(&args[1])?;
    let model = Codec::from_artifact(b.clone(), &artifact)?;
    b.evaluate(&model.weights().values().cloned().collect::<Vec<_>>())?;
    let mut rows = Vec::new();
    for batch in [1_usize, 2, 4] {
        let input = tokens(batch);
        let mut codes = Vec::new();
        for i in 0..batch {
            for (level, &length) in model.config().levels().iter().enumerate() {
                codes.extend((0..length).map(|p| ((i * length + p + 3 * level) % 16) as u32));
            }
        }
        let encoded = model.encode(&input, batch)?;
        let latent = encoded.snapshots.last().ok_or("missing latent")?.clone();
        b.evaluate(std::slice::from_ref(&latent))?;
        drop(encoded);
        for scope in ["decode", "reconstruct", "resident_decoder_block"] {
            b.set_unmasked_attention_for_diagnostic(false);
            let (plain, plain_ids) = run_scope(&model, scope, &input, &codes, batch, &latent)?;
            let expected = b.read_f32(&plain)?;
            let start = Instant::now();
            b.set_unmasked_attention_for_diagnostic(true);
            let (unmasked, unmasked_ids) =
                run_scope(&model, scope, &input, &codes, batch, &latent)?;
            let first_unmasked_ms = start.elapsed().as_secs_f64() * 1000.;
            let actual = b.read_f32(&unmasked)?;
            if plain_ids != unmasked_ids
                || expected.len() != actual.len()
                || expected.iter().zip(&actual).any(|(&a, &v)| {
                    !a.is_finite() || !v.is_finite() || (a - v).abs() > 1e-4 + 2e-4 * a.abs()
                })
            {
                return Err("output mismatch or nonfinite value".into());
            }
            let expected_hash = hash_ids(&plain_ids);
            drop((plain, unmasked, expected, actual));
            let mut eager = Vec::new();
            let mut optimized = Vec::new();
            for i in 0..35 {
                for enabled in if i % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    b.set_unmasked_attention_for_diagnostic(enabled);
                    let start = Instant::now();
                    let (out, ids) = run_scope(&model, scope, &input, &codes, batch, &latent)?;
                    let ms = start.elapsed().as_secs_f64() * 1000.;
                    if hash_ids(&ids) != expected_hash {
                        return Err("nonrepeatable IDs".into());
                    }
                    if i >= 5 {
                        if enabled {
                            optimized.push(ms);
                        } else {
                            eager.push(ms);
                        }
                    }
                    drop(out);
                }
            }
            rows.push(json!({"batch":batch,"scope":scope,"first_unmasked_ms":first_unmasked_ms,"zero_mask_ms":eager,"unmasked_ms":optimized,"ids_sha256":expected_hash,"finite_output_within_component_tolerance":true}));
        }
    }
    serde_json::to_writer_pretty(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?,
        &json!({"schema_version":1,"scope":"same model, alternating zero-mask and unmasked attention; diagnostic only","warmups":5,"trials":30,"cases":rows}),
    )?;
    Ok(())
}
