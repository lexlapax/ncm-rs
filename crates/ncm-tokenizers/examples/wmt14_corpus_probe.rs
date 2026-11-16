//! Compare every WMT14 source row with the pinned offline Python tokenizer capture.
use ncm_tokenizers::TextTokenizer;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};

#[derive(Deserialize)]
struct Dataset {
    profile: String,
    inputs: Vec<String>,
}
#[derive(Deserialize)]
struct Reference {
    dataset_sha256: String,
    tokenizer_manifest_sha256: String,
    rows: Vec<Vec<u32>>,
    prefixes: Vec<Vec<u32>>,
}
fn checked(path: &Path, expected: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(format!("input hash differs: {}", path.display()).into());
    }
    Ok(bytes)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: wmt14_corpus_probe TOKENIZER_DIR DATASET_JSON REFERENCE_JSON".into());
    }
    let data_sha = "36035390ce542834f8dfeeac004c1b532aff247a4ff39910f4a16efd009ca47c";
    let manifest_sha = "e1f881e225d651271d01d4067b3308578eb836703fdd25cfb17e7a1b95fcd69a";
    let reference_sha = "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f";
    let dataset: Dataset = serde_json::from_slice(&checked(Path::new(&args[2]), data_sha)?)?;
    let reference: Reference =
        serde_json::from_slice(&checked(Path::new(&args[3]), reference_sha)?)?;
    checked(&Path::new(&args[1]).join("manifest.json"), manifest_sha)?;
    let tokenizer = TextTokenizer::load(&args[1])?;
    if dataset.profile != "wmt14"
        || dataset.inputs.len() != 3003
        || reference.rows.len() != 3003
        || reference.prefixes.len() != 3003
        || reference.dataset_sha256 != data_sha
        || reference.tokenizer_manifest_sha256 != manifest_sha
        || tokenizer.metadata().pad_token_id != 151643
    {
        return Err("unexpected dataset or tokenizer identity".into());
    }
    let mut truncated = 0;
    let mut tokens = 0;
    for (index, text) in dataset.inputs.iter().enumerate() {
        let ids = tokenizer.encode(text)?;
        if ids != reference.rows[index] {
            return Err(format!("encoded IDs differ at row {index}").into());
        }
        tokens += ids.len();
        truncated += usize::from(ids.len() > 96);
        let mut prefix = ids;
        prefix.truncate(96);
        prefix.resize(96, 151643);
        if prefix != reference.prefixes[index] {
            return Err(format!("padded prefix differs at row {index}").into());
        }
    }
    println!(
        "{}",
        serde_json::json!({"rows": dataset.inputs.len(), "tokens": tokens,
        "truncated_rows": truncated, "full_ids_match": true, "padded_prefixes_match": true,
        "scope": "source encoding and padding only; not model inference or quality"})
    );
    Ok(())
}
