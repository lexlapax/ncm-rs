//! Allocation attribution for the unchanged tokenizer workload, not a benchmark.
#[path = "../../../tests/support/allocation_counts.rs"]
mod allocations;
#[global_allocator]
static ALLOCATOR: allocations::Counting = allocations::Counting;

use ncm_tokenizers::TextTokenizer;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    decoded: String,
    decoded_skip_special: String,
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 || cfg!(debug_assertions) {
        return Err(
            "usage (release): asset_allocations CANONICAL_ROOT FIXTURE PROFILE NEW_REPORT".into(),
        );
    }
    if std::env::var("TOKENIZERS_PARALLELISM").as_deref() != Ok("false") {
        return Err("set TOKENIZERS_PARALLELISM=false for this diagnostic".into());
    }
    if Path::new(&args[4]).exists() {
        return Err("output exists".into());
    }
    let fixture_bytes = fs::read(&args[2])?;
    if fixture_bytes != include_bytes!("../../../tests/fixtures/tokenizers-v1.json") {
        return Err("unpinned fixture".into());
    }
    let fixture: Value = serde_json::from_slice(&fixture_bytes)?;
    let profile = &args[3];
    let record = fixture["profiles"].get(profile).ok_or("unknown profile")?;
    let cases: Vec<Case> = serde_json::from_value(record["cases"].clone())?;
    if cases.len() != 89 {
        return Err("unexpected workload size".into());
    }
    let directory = Path::new(&args[1]).join(profile);
    let manifest = fs::read(directory.join("manifest.json"))?;
    if record["manifest_sha256"] != digest(&manifest) {
        return Err("unpinned manifest".into());
    }
    let mut rows = Vec::new();
    let mut first = None;
    let mut trials = Vec::new();
    for trial in 0..36 {
        let (loaded, counts) = allocations::measure(|| TextTokenizer::load(&directory));
        drop(loaded?);
        if trial == 0 {
            first = Some(counts);
        } else if trial >= 6 {
            trials.push(counts);
        }
    }
    rows.push(json!({"scope":"checked_load","first":first,"trials":trials}));
    let tokenizer = TextTokenizer::load(&directory)?;
    for scope in ["encode", "decode", "decode_skip_special"] {
        let mut first = None;
        let mut trials = Vec::new();
        let mut output_hash = String::new();
        for trial in 0..36 {
            let (bytes, counts) = if scope == "encode" {
                let (output, counts) = allocations::measure(|| {
                    cases
                        .iter()
                        .map(|c| Ok(tokenizer.encode(&c.text)?))
                        .collect::<Result<Vec<_>>>()
                });
                let output = output?;
                if output.iter().zip(&cases).any(|(o, c)| *o != c.ids) {
                    return Err("encode mismatch".into());
                }
                let mut bytes = Vec::new();
                for ids in output {
                    bytes.extend((ids.len() as u64).to_le_bytes());
                    bytes.extend(ids.into_iter().flat_map(u32::to_le_bytes));
                }
                (bytes, counts)
            } else {
                let skip = scope == "decode_skip_special";
                let (output, counts) = allocations::measure(|| {
                    cases
                        .iter()
                        .map(|c| Ok(tokenizer.decode(&c.ids, skip)?))
                        .collect::<Result<Vec<_>>>()
                });
                let output = output?;
                if output.iter().zip(&cases).any(|(o, c)| {
                    *o != if skip {
                        &c.decoded_skip_special
                    } else {
                        &c.decoded
                    }
                    .as_str()
                }) {
                    return Err("decode mismatch".into());
                }
                let mut bytes = Vec::new();
                for text in output {
                    bytes.extend((text.len() as u64).to_le_bytes());
                    bytes.extend(text.as_bytes());
                }
                (bytes, counts)
            };
            let hash = digest(&bytes);
            if trial == 0 {
                first = Some(counts);
                output_hash = hash;
            } else {
                if hash != output_hash {
                    return Err("nonrepeatable output".into());
                }
                if trial >= 6 {
                    trials.push(counts);
                }
            }
        }
        rows.push(json!({"scope":scope,"first":first,"trials":trials,"output_sha256":output_hash}));
    }
    let sources: BTreeMap<_, _> = [
        "crates/ncm-tokenizers/examples/asset_allocations.rs",
        "tests/support/allocation_counts.rs",
        "crates/ncm-tokenizers/src/assets.rs",
        "crates/ncm-tokenizers/src/bpe.rs",
        "crates/ncm-tokenizers/src/hf_bpe.rs",
        "crates/ncm-tokenizers/src/hf_wordpiece.rs",
        "Cargo.lock",
    ]
    .into_iter()
    .map(|p| Ok((p, digest(&fs::read(p)?))))
    .collect::<Result<_>>()?;
    let report = json!({"schema_version":1,"diagnostic_profiling":true,"runtime":"rust-tokenizer-allocations",
        "profile":profile,"fixture_sha256":digest(&fixture_bytes),"manifest_sha256":digest(&manifest),
        "source_sha256":sources,"warmups":5,"trials":30,"strings_per_trial":89,"measurements":rows,
        "scope":"Successful Rust GlobalAlloc/System calls and requested bytes during checked load or 89 retained-output scalar calls. Validation/hashing/returned-output destruction excluded. Includes Rust threads in the region; parallel tokenization disabled.",
        "limitations":["Native C/library allocations, allocator overhead, stack and physical memory excluded.",
        "Realloc records requested new size and released old size, not hidden malloc copying or physical peak.",
        "Counts are diagnostic only: no latency, live-memory or peak-memory claim."]});
    fs::write(
        &args[4],
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    Ok(())
}
