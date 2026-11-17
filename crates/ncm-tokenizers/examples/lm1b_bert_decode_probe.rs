//! Compare LM1B FLM scorer decoding on development and synthetic model IDs.
use ncm_tokenizers::TextTokenizer;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};

#[derive(Deserialize)]
struct Row {
    kind: String,
    ids: Vec<u32>,
    text: String,
}
#[derive(Deserialize)]
struct Reference {
    manifest_sha256: String,
    rows: Vec<Row>,
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
        return Err("usage: lm1b_bert_decode_probe TOKENIZER_DIR REFERENCE_JSON".into());
    }
    let manifest = "8042d9e1f380867272ee1b9d296ed060f19cacfecf249239c5332cda9c934efd";
    let reference: Reference = serde_json::from_slice(&checked(
        Path::new(&args[2]),
        "fee7a02040abf97b7a64991d31e616133ace1ea37e574cdb018592d4ee19d6ee",
    )?)?;
    checked(&Path::new(&args[1]).join("manifest.json"), manifest)?;
    let tokenizer = TextTokenizer::load(&args[1])?;
    if reference.manifest_sha256 != manifest
        || reference.rows.len() != 1288
        || tokenizer.metadata().total_vocab_size != 30522
    {
        return Err("unexpected reference or tokenizer identity".into());
    }
    let mut reserved = 0;
    for (index, row) in reference.rows.iter().enumerate() {
        if row.ids.iter().any(|&id| id >= 30528) {
            return Err(format!("invalid model ID at row {index}").into());
        }
        let ids: Vec<_> = row.ids.iter().copied().filter(|&id| id < 30522).collect();
        if ids.len() != row.ids.len() {
            if row.kind != "synthetic" || tokenizer.decode(&row.ids, false).is_ok() {
                return Err(format!("unexpected reserved ID behavior at row {index}").into());
            }
            reserved += 1;
        }
        if tokenizer.decode(&ids, false)? != row.text {
            return Err(format!("decoded text differs at row {index}").into());
        }
    }
    println!(
        "{}",
        serde_json::json!({"rows": reference.rows.len(),
        "synthetic_rows_with_reserved_ids": reserved, "matches_after_explicit_reserved_id_filter": true})
    );
    Ok(())
}
