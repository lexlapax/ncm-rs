//! Explicit real-asset parity; no downloads, ignored cases or Python runtime.
use ncm_tokenizers::TextTokenizer;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};

fn ids(case: &Value) -> Result<Vec<u32>, serde_json::Error> {
    serde_json::from_value(case["ids"].clone())
}
fn rejections(source: &Path) -> Result<usize, Box<dyn Error>> {
    let temporary = tempfile::tempdir()?;
    for name in ["manifest.json", "tokenizer.json", "LICENSE"] {
        fs::copy(source.join(name), temporary.path().join(name))?;
    }
    TextTokenizer::load(temporary.path())?;
    let mut count = 0;
    for name in ["manifest.json", "tokenizer.json", "LICENSE"] {
        let path = temporary.path().join(name);
        let original = fs::read(&path)?;
        let mut changed = original.clone();
        changed[0] ^= 1;
        fs::write(&path, changed)?;
        if TextTokenizer::load(temporary.path()).is_ok() {
            return Err(format!("accepted changed {name}").into());
        }
        fs::write(path, original)?;
        count += 1;
    }
    #[cfg(unix)]
    {
        let payload = temporary.path().join("tokenizer.json");
        fs::remove_file(&payload)?;
        std::os::unix::fs::symlink(source.join("tokenizer.json").canonicalize()?, payload)?;
        if TextTokenizer::load(temporary.path()).is_ok() {
            return Err("accepted payload symlink".into());
        }
        count += 1;
    }
    Ok(count)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: asset_probe CANONICAL_ROOT FIXTURE_JSON".into());
    }
    let fixture_bytes = fs::read(&args[2])?;
    let fixture: Value = serde_json::from_slice(&fixture_bytes)?;
    if fixture["schema_version"] != 1 {
        return Err("unknown fixture schema".into());
    }
    let mut report = serde_json::Map::new();
    for (profile, cases) in fixture["profiles"].as_object().ok_or("missing profiles")? {
        let directory = Path::new(&args[1]).join(profile);
        let tokenizer = TextTokenizer::load(&directory)?;
        let metadata = tokenizer.metadata();
        let manifest_sha256 = format!(
            "{:x}",
            Sha256::digest(fs::read(directory.join("manifest.json"))?)
        );
        if cases["manifest_sha256"] != manifest_sha256
            || fixture["upstream_commit"] != metadata.upstream_commit
            || fixture["reference_versions"] != serde_json::to_value(&metadata.reference_versions)?
        {
            return Err("fixture identity differs from runtime asset".into());
        }
        let rejection_cases = rejections(&directory)?;
        for (key, value) in [
            ("vocab_size", metadata.vocab_size),
            ("total_vocab_size", metadata.total_vocab_size),
            ("eos_token_id", metadata.eos_token_id as usize),
            ("pad_token_id", metadata.pad_token_id as usize),
        ] {
            if cases[key] != value {
                return Err(format!("{profile} metadata mismatch: {key}").into());
            }
        }
        if cases["cleanup"] != metadata.cleanup {
            return Err("cleanup metadata mismatch".into());
        }
        let encode_cases = cases["cases"].as_array().ok_or("missing encode cases")?;
        let decode_cases = cases["decode_cases"]
            .as_array()
            .ok_or("missing decode cases")?;
        if encode_cases.is_empty() || decode_cases.is_empty() {
            return Err("empty tokenizer qualification".into());
        }
        for (index, case) in encode_cases.iter().enumerate() {
            let actual = tokenizer.encode(case["text"].as_str().ok_or("missing input")?)?;
            if actual != ids(case)? {
                return Err(format!(
                    "{profile} encode case {index}: {actual:?} != {}",
                    case["ids"]
                )
                .into());
            }
        }
        for (index, case) in encode_cases.iter().chain(decode_cases).enumerate() {
            for (skip, key) in [(false, "decoded"), (true, "decoded_skip_special")] {
                let actual = tokenizer.decode(&ids(case)?, skip)?;
                if case[key] != actual {
                    return Err(format!(
                        "{profile} decode case {index}, skip={skip}: {actual:?} != {}",
                        case[key]
                    )
                    .into());
                }
            }
        }
        if tokenizer.decode(&[u32::MAX], true).is_ok() {
            return Err("invalid token ID accepted".into());
        }
        report.insert(profile.clone(),json!({"encode_cases":encode_cases.len(),"decode_cases_including_encoded":encode_cases.len()+decode_cases.len(),
            "both_skip_modes":true,"all_passed":true,"vocab_size":metadata.vocab_size,"total_vocab_size":metadata.total_vocab_size,
            "eos_token_id":metadata.eos_token_id,"pad_token_id":metadata.pad_token_id,"cleanup":metadata.cleanup,
            "manifest_sha256":manifest_sha256,"rejection_cases":rejection_cases}));
    }
    if report.len() != 4 {
        return Err("expected all four profiles".into());
    }
    let sources: std::collections::BTreeMap<_, _> = [
        "crates/ncm-tokenizers/src/lib.rs",
        "crates/ncm-tokenizers/src/bpe.rs",
        "crates/ncm-tokenizers/src/assets.rs",
        "crates/ncm-tokenizers/src/hf_bpe.rs",
        "crates/ncm-tokenizers/src/hf_wordpiece.rs",
        "vendor/tokenizers-0.22.2/src/models/bpe/model.rs",
        "vendor/tokenizers-0.22.2/src/normalizers/bert.rs",
        "crates/ncm-tokenizers/examples/asset_probe.rs",
        "Cargo.lock",
        "tools/reference/tokenizers-v1.lock.json",
        "tools/reference/tokenizer-sources.lock.json",
    ]
    .into_iter()
    .map(|path| Ok((path, format!("{:x}", Sha256::digest(fs::read(path)?)))))
    .collect::<Result<_, std::io::Error>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"schema_version":1,"runtime":"rust-tokenizers","profiles":report,"source_sha256":sources,
                "fixture_sha256":format!("{:x}",Sha256::digest(&fixture_bytes)),"reference_versions":fixture["reference_versions"]})
        )?
    );
    Ok(())
}
