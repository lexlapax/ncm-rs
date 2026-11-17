//! Compare WMT14-style EOS/PAD decoding on synthetic source-derived sequences.
use ncm_tokenizers::TextTokenizer;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};

#[derive(Deserialize)]
struct Record {
    ids: Vec<u32>,
    text: String,
}
#[derive(Deserialize)]
struct Reference {
    reference_sha256: String,
    manifest_sha256: String,
    rows: Vec<Record>,
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
    if args.len() != 3 {
        return Err("usage: wmt14_decode_probe TOKENIZER_DIR REFERENCE_JSON".into());
    }
    let manifest_sha = "e1f881e225d651271d01d4067b3308578eb836703fdd25cfb17e7a1b95fcd69a";
    let source_sha = "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f";
    let reference: Reference = serde_json::from_slice(&checked(
        Path::new(&args[2]),
        "d38eaa32c1d92e55d421f524bae456765bb64e9a06f2a75fa2ba5f0f73081c98",
    )?)?;
    checked(&Path::new(&args[1]).join("manifest.json"), manifest_sha)?;
    let tokenizer = TextTokenizer::load(&args[1])?;
    if reference.rows.len() != 6006
        || reference.reference_sha256 != source_sha
        || reference.manifest_sha256 != manifest_sha
        || tokenizer.metadata().eos_token_id != 151645
        || tokenizer.metadata().pad_token_id != 151643
    {
        return Err("unexpected reference or tokenizer identity".into());
    }
    for (index, row) in reference.rows.iter().enumerate() {
        let ids: Vec<u32> = row
            .ids
            .iter()
            .copied()
            .take_while(|&id| id != 151645)
            .filter(|&id| id != 151643)
            .collect();
        if tokenizer.decode(&ids, true)? != row.text {
            return Err(format!("decoded text differs at row {index}").into());
        }
    }
    println!(
        "{}",
        serde_json::json!({"rows": reference.rows.len(), "decoded_text_matches": true,
        "scope": "synthetic source-derived sequences; not model inference or quality"})
    );
    Ok(())
}
