//! Compare prefix-cache OFF vs ON for XSum (CFG 0, CFG 1, seeded) and WMT14 (CFG 0) on Metal.
use ncm_backend_mlx::MlxBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::{Wmt14Teacher, XsumTeacher};
use serde_json::Value;
use std::{error::Error, fs, time::Instant};

fn load_xsum(path: &str, count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    let doc: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = doc["inputs"].as_array().ok_or("missing xsum inputs")?;
    let mut out = Vec::with_capacity(count * 512);
    for row in &rows[..count] {
        for x in &row.as_array().ok_or("bad row")?[..512] {
            out.push(u32::try_from(x.as_u64().ok_or("bad id")?)?);
        }
    }
    Ok(out)
}

fn load_wmt14(path: &str, count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    let doc: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = doc["prefixes"].as_array().ok_or("missing wmt14 prefixes")?;
    let mut out = Vec::with_capacity(count * 96);
    for row in &rows[..count] {
        for x in row.as_array().ok_or("bad row")? {
            out.push(u32::try_from(x.as_u64().ok_or("bad id")?)?);
        }
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: prefix_cache_equiv_probe xsum|wmt14 ARTIFACT INPUTS COUNT".into());
    }
    let profile = args[1].as_str();
    let artifact_path = args[2].as_str();
    let input_path = args[3].as_str();
    let count: usize = args[4].parse()?;
    let backend = MlxBackend::gpu()?;

    match profile {
        "xsum" => {
            let prefixes = load_xsum(input_path, count)?;
            let b4 = count.min(4);
            let s4 = &prefixes[..b4 * 512];

            let (u_codes, u_ids, u_cfg, u_seed, u_cfg_seed, u_ms1, u_ms4) = {
                let model = XsumTeacher::from_owned_artifact(
                    backend.clone(),
                    Artifact::load_inference_streaming(artifact_path)?,
                )?;
                let mut codes = Vec::new();
                let mut ids = Vec::new();
                for start in (0..count).step_by(8) {
                    let batch = (count - start).min(8);
                    let out = model
                        .generate_top1_cpu(&prefixes[start * 512..(start + batch) * 512], batch)?;
                    codes.extend(backend.read_indices(&out.codes)?);
                    ids.extend(out.ids);
                }
                let cfg = model.generate_top1_cfg_cpu(s4, b4, 1.0)?.ids;
                let seed = model.sample_seeded_cfg_cpu(s4, b4, 0.0, 42)?.ids;
                let cfg_seed = model.sample_seeded_cfg_cpu(s4, b4, 1.0, 42)?.ids;
                let mut ms = [0.0; 2];
                for (slot, tb) in [(0, 1usize), (1, b4)] {
                    let sl = &prefixes[..tb * 512];
                    for _ in 0..2 {
                        let out = model.generate_top1_cpu(sl, tb)?;
                        let _ = backend.read_indices(&out.codes)?;
                    }
                    let t0 = Instant::now();
                    for _ in 0..5 {
                        let out = model.generate_top1_cpu(sl, tb)?;
                        let _ = backend.read_indices(&out.codes)?;
                    }
                    ms[slot] = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
                }
                (codes, ids, cfg, seed, cfg_seed, ms[0], ms[1])
            };

            let (c_codes, c_ids, c_cfg, c_seed, c_cfg_seed, c_ms1, c_ms4) = {
                let model = XsumTeacher::from_owned_artifact(
                    backend.clone(),
                    Artifact::load_inference_streaming(artifact_path)?,
                )?
                .with_prefix_cache(true);
                let mut codes = Vec::new();
                let mut ids = Vec::new();
                for start in (0..count).step_by(8) {
                    let batch = (count - start).min(8);
                    let out = model
                        .generate_top1_cpu(&prefixes[start * 512..(start + batch) * 512], batch)?;
                    codes.extend(backend.read_indices(&out.codes)?);
                    ids.extend(out.ids);
                }
                let cfg = model.generate_top1_cfg_cpu(s4, b4, 1.0)?.ids;
                let seed = model.sample_seeded_cfg_cpu(s4, b4, 0.0, 42)?.ids;
                let cfg_seed = model.sample_seeded_cfg_cpu(s4, b4, 1.0, 42)?.ids;
                let mut ms = [0.0; 2];
                for (slot, tb) in [(0, 1usize), (1, b4)] {
                    let sl = &prefixes[..tb * 512];
                    for _ in 0..2 {
                        let out = model.generate_top1_cpu(sl, tb)?;
                        let _ = backend.read_indices(&out.codes)?;
                    }
                    let t0 = Instant::now();
                    for _ in 0..5 {
                        let out = model.generate_top1_cpu(sl, tb)?;
                        let _ = backend.read_indices(&out.codes)?;
                    }
                    ms[slot] = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
                }
                (codes, ids, cfg, seed, cfg_seed, ms[0], ms[1])
            };

            let code_diffs = u_codes.iter().zip(&c_codes).filter(|(a, b)| a != b).count();
            let id_diffs = u_ids.iter().zip(&c_ids).filter(|(a, b)| a != b).count();
            let cfg_diffs = u_cfg.iter().zip(&c_cfg).filter(|(a, b)| a != b).count();
            let seed_diffs = u_seed.iter().zip(&c_seed).filter(|(a, b)| a != b).count();
            let cfg_seed_diffs = u_cfg_seed
                .iter()
                .zip(&c_cfg_seed)
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "xsum b=1: uncached={u_ms1:.2}ms cached={c_ms1:.2}ms speedup={:.2}x",
                u_ms1 / c_ms1
            );
            println!(
                "xsum b={b4}: uncached={u_ms4:.2}ms cached={c_ms4:.2}ms speedup={:.2}x",
                u_ms4 / c_ms4
            );
            println!(
                "xsum equiv (rows={count}): cfg0_code_diffs={code_diffs} cfg0_id_diffs={id_diffs} cfg1_id_diffs={cfg_diffs} seed42_id_diffs={seed_diffs} cfg1_seed42_id_diffs={cfg_seed_diffs}"
            );
        }
        "wmt14" => {
            let prefixes = load_wmt14(input_path, count)?;
            let b4 = count.min(4);

            let (u_codes, u_ids, u_ms1, u_ms4) = {
                let model = Wmt14Teacher::from_owned_artifact(
                    backend.clone(),
                    Artifact::load_inference_streaming(artifact_path)?,
                )?
                .with_deferred_head_check_experimental()
                .with_level_materialization_experimental(5)?;
                let mut codes = Vec::new();
                let mut ids = Vec::new();
                for start in (0..count).step_by(8) {
                    let batch = (count - start).min(8);
                    let out = model.generate_top1_with_tiled_decoder(
                        &prefixes[start * 96..(start + batch) * 96],
                        batch,
                        0.0,
                        2,
                    )?;
                    codes.extend(out.codes);
                    ids.extend(out.ids);
                }
                let mut ms = [0.0; 2];
                for (slot, tb) in [(0, 1usize), (1, b4)] {
                    let sl = &prefixes[..tb * 96];
                    for _ in 0..2 {
                        let _ = model.generate_top1_with_tiled_decoder(sl, tb, 0.0, 2)?;
                    }
                    let t0 = Instant::now();
                    for _ in 0..5 {
                        let _ = model.generate_top1_with_tiled_decoder(sl, tb, 0.0, 2)?;
                    }
                    ms[slot] = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
                }
                (codes, ids, ms[0], ms[1])
            };

            let (c_codes, c_ids, c_ms1, c_ms4) = {
                let model = Wmt14Teacher::from_owned_artifact(
                    backend.clone(),
                    Artifact::load_inference_streaming(artifact_path)?,
                )?
                .with_prefix_cache(true)
                .with_deferred_head_check_experimental()
                .with_level_materialization_experimental(5)?;
                let mut codes = Vec::new();
                let mut ids = Vec::new();
                for start in (0..count).step_by(8) {
                    let batch = (count - start).min(8);
                    let out = model.generate_top1_with_tiled_decoder(
                        &prefixes[start * 96..(start + batch) * 96],
                        batch,
                        0.0,
                        2,
                    )?;
                    codes.extend(out.codes);
                    ids.extend(out.ids);
                }
                let mut ms = [0.0; 2];
                for (slot, tb) in [(0, 1usize), (1, b4)] {
                    let sl = &prefixes[..tb * 96];
                    for _ in 0..2 {
                        let _ = model.generate_top1_with_tiled_decoder(sl, tb, 0.0, 2)?;
                    }
                    let t0 = Instant::now();
                    for _ in 0..5 {
                        let _ = model.generate_top1_with_tiled_decoder(sl, tb, 0.0, 2)?;
                    }
                    ms[slot] = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
                }
                (codes, ids, ms[0], ms[1])
            };

            let code_diffs = u_codes.iter().zip(&c_codes).filter(|(a, b)| a != b).count();
            let id_diffs = u_ids.iter().zip(&c_ids).filter(|(a, b)| a != b).count();
            println!(
                "wmt14 b=1: uncached={u_ms1:.2}ms cached={c_ms1:.2}ms speedup={:.2}x",
                u_ms1 / c_ms1
            );
            println!(
                "wmt14 b={b4}: uncached={u_ms4:.2}ms cached={c_ms4:.2}ms speedup={:.2}x",
                u_ms4 / c_ms4
            );
            println!("wmt14 equiv (rows={count}): code_diffs={code_diffs} id_diffs={id_diffs}");
        }
        _ => return Err("profile must be xsum or wmt14".into()),
    }
    Ok(())
}
