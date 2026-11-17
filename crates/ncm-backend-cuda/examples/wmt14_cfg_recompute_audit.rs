//! Compare raw cached and uncached guided heads on previously inspected rows.
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, Error, Result};
use ncm_io::{Artifact, sha256};
use ncm_model::{TraceSink, Wmt14Teacher};
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error as StdError, fs, io::Write, path::Path};

struct Capture<B: CodecBackend> {
    backend: B,
    cached: bool,
    logits: BTreeMap<usize, Vec<f32>>,
    flags: BTreeMap<usize, u32>,
    codes: Vec<u32>,
    ids: Vec<u32>,
}
impl<B: CodecBackend> Capture<B> {
    fn new(backend: B, cached: bool) -> Self {
        Self {
            backend,
            cached,
            logits: BTreeMap::new(),
            flags: BTreeMap::new(),
            codes: Vec::new(),
            ids: Vec::new(),
        }
    }
}
impl<B: CodecBackend> TraceSink<B> for Capture<B> {
    fn tensor(&mut self, name: &str, value: &B::Tensor) -> Result<()> {
        let prefix = if self.cached {
            "diagnostic.cached.level_"
        } else {
            "generator.level_"
        };
        if let Some(index) = name
            .strip_prefix(prefix)
            .and_then(|s| s.strip_suffix(".logits"))
        {
            let level: usize = index
                .parse()
                .map_err(|_| Error::Input("invalid trace level".into()))?;
            if level > 0
                && self
                    .logits
                    .insert(level, self.backend.read_f32(value)?)
                    .is_some()
            {
                return Err(Error::Input("duplicate trace level".into()));
            }
        }
        Ok(())
    }
    fn indices(&mut self, name: &str, value: &B::Indices) -> Result<()> {
        match name {
            "generator.codes" => self.codes = self.backend.read_indices(value)?,
            "generator.greedy_ids" => self.ids = self.backend.read_indices(value)?,
            _ => {
                if let Some(index) = name.strip_prefix("diagnostic.recomputed.level_") {
                    let level: usize = index
                        .parse()
                        .map_err(|_| Error::Input("invalid flag level".into()))?;
                    let flag = self.backend.read_indices(value)?;
                    if flag.len() != 1 || flag[0] > 1 || self.flags.insert(level, flag[0]).is_some()
                    {
                        return Err(Error::Input("invalid recompute flag".into()));
                    }
                }
            }
        }
        Ok(())
    }
}

fn checked_json(path: &Path, expected: &str) -> std::result::Result<Value, Box<dyn StdError>> {
    let bytes = fs::read(path)?;
    if sha256(&bytes) != expected {
        return Err("unexpected input hash".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn main() -> std::result::Result<(), Box<dyn StdError>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: wmt14_cfg_recompute_audit ARTIFACT PREFIXES PYTHON OUTPUT COUNT".into(),
        );
    }
    let count: usize = args[5].parse()?;
    if count == 0 || count > 3003 {
        return Err("invalid count".into());
    }
    let output = Path::new(&args[4]);
    if output.exists() {
        return Err("output must be new".into());
    }
    if sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected artifact".into());
    }
    let prefixes = checked_json(
        Path::new(&args[2]),
        "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f",
    )?;
    let python = checked_json(
        Path::new(&args[3]),
        "78f24bba0fecc8ce7c06abf22508516142a5331cc6df5572efd533f9211e9256",
    )?;
    let rows = prefixes["prefixes"].as_array().ok_or("missing prefixes")?;
    let expected = python["outputs"]
        .as_array()
        .ok_or("missing Python outputs")?;
    if rows.len() != 3003 || expected.len() != 3003 {
        return Err("wrong input count".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    let backend = CandleBackend::cuda_wmt14_reference(0)?;
    let plain = Wmt14Teacher::from_artifact(backend.clone(), &artifact)?;
    let cached =
        Wmt14Teacher::from_artifact_cfg_recompute_diagnostic(backend.clone(), &artifact, 1e-4)?;
    let mut positions = 0_u64;
    let mut uncertain = 0_u64;
    let mut disagreements = 0_u64;
    let mut above_margin = 0_u64;
    let mut uncovered_batch_levels = 0_u64;
    let mut recomputed = 0_u64;
    let mut max_error = 0_f32;
    for start in (0..count).step_by(4) {
        let size = (count - start).min(4);
        let mut tokens = Vec::with_capacity(size * 96);
        for row in &rows[start..start + size] {
            let row = row.as_array().ok_or("invalid prefix")?;
            if row.len() != 96 {
                return Err("invalid prefix width".into());
            }
            for token in row {
                tokens.push(u32::try_from(token.as_u64().ok_or("invalid ID")?)?);
            }
        }
        let mut a = Capture::new(backend.clone(), false);
        let mut b = Capture::new(backend.clone(), true);
        plain.generate_cfg_top1_traced(&tokens, size, 1.0, &mut a)?;
        cached.generate_cfg_top1_traced(&tokens, size, 1.0, &mut b)?;
        if a.codes != b.codes
            || a.ids != b.ids
            || a.codes.len() != size * 313
            || a.ids.len() != size * 64
        {
            return Err(format!("guided output mismatch at row {start}").into());
        }
        for (offset, row) in expected[start..start + size].iter().enumerate() {
            let codes: Vec<u32> = serde_json::from_value(row["codes"].clone())?;
            let ids: Vec<u32> = serde_json::from_value(row["raw_ids"].clone())?;
            if row["row"].as_u64() != Some((start + offset) as u64)
                || codes != a.codes[offset * 313..(offset + 1) * 313]
                || ids != a.ids[offset * 64..(offset + 1) * 64]
            {
                return Err(format!("Python mismatch at row {}", start + offset).into());
            }
        }
        if a.logits.len() != 10 || b.logits.len() != 10 || b.flags.len() != 10 {
            return Err("missing level traces".into());
        }
        for level in 1..=10 {
            let left = a.logits.get(&level).ok_or("missing uncached logits")?;
            let right = b.logits.get(&level).ok_or("missing cached logits")?;
            if left.len() != right.len() || !left.len().is_multiple_of(4096) {
                return Err("invalid logit shape".into());
            }
            let mut level_uncertain = false;
            let mut level_disagreements = 0_u64;
            for (x, y) in left
                .as_chunks::<4096>()
                .0
                .iter()
                .zip(right.as_chunks::<4096>().0)
            {
                let mut top = (f32::NEG_INFINITY, 0_usize);
                let mut second = f32::NEG_INFINITY;
                let mut ref_top = (f32::NEG_INFINITY, 0_usize);
                for (index, (&p, &q)) in x.iter().zip(y).enumerate() {
                    if !p.is_finite() || !q.is_finite() {
                        return Err("nonfinite score".into());
                    }
                    max_error = max_error.max((p - q).abs());
                    if p > ref_top.0 {
                        ref_top = (p, index);
                    }
                    if q > top.0 {
                        second = top.0;
                        top = (q, index);
                    } else if q > second {
                        second = q;
                    }
                }
                let near = top.0 - second <= 1e-4;
                positions += 1;
                uncertain += u64::from(near);
                level_uncertain |= near;
                if top.1 != ref_top.1 {
                    disagreements += 1;
                    above_margin += u64::from(!near);
                    level_disagreements += 1;
                }
            }
            if b.flags[&level] != u32::from(level_uncertain) {
                return Err("recompute flag mismatch".into());
            }
            recomputed += u64::from(level_uncertain);
            uncovered_batch_levels += u64::from(level_disagreements > 0 && !level_uncertain);
        }
        eprintln!("{}/{}", start + size, count);
    }
    let result = json!({"scope":"previously inspected WMT14 CFG CUDA corpus; no unseen-input error bound", "count":count, "batch_size":4, "compared_positions":positions, "uncertain_positions":uncertain, "raw_argmax_disagreements":disagreements, "above_margin_disagreements":above_margin, "uncovered_batch_levels":uncovered_batch_levels, "recomputed_batch_levels":recomputed, "max_absolute_logit_difference":max_error, "final_codes_and_ids_match_python":true});
    writeln!(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
        "{}",
        serde_json::to_string_pretty(&result)?
    )?;
    println!("{}", result);
    Ok(())
}
