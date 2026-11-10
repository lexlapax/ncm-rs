//! Matched scalar encode/decode workload from the pinned parity fixture.
use ncm_tokenizers::TextTokenizer;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, fs, path::Path, time::Instant};
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
fn host() -> Result<Value> {
    let status = fs::read_to_string("/proc/self/status")?;
    let mut result = BTreeMap::new();
    for key in ["VmRSS:", "VmHWM:"] {
        let line = status
            .lines()
            .find(|l| l.starts_with(key))
            .ok_or("missing memory counter")?;
        result.insert(
            key.trim_end_matches(':'),
            line.split_whitespace()
                .nth(1)
                .ok_or("missing memory value")?
                .parse::<u64>()?
                * 1024,
        );
    }
    Ok(json!(result))
}
fn timed<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    let start = Instant::now();
    let value = f()?;
    Ok((value, start.elapsed().as_secs_f64() * 1000.))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 || cfg!(debug_assertions) {
        return Err(
            "usage (release): asset_bench CANONICAL_ROOT FIXTURE PROFILE NEW_REPORT".into(),
        );
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
    let directory = Path::new(&args[1]).join(profile);
    let record = fixture["profiles"].get(profile).ok_or("unknown profile")?;
    let cases: Vec<Case> = serde_json::from_value(record["cases"].clone())?;
    if cases.len() != 89 {
        return Err("unexpected workload size".into());
    }
    let load = || -> Result<TextTokenizer> { Ok(TextTokenizer::load(&directory)?) };
    let (tokenizer, first_load_ms) = timed(load)?;
    drop(tokenizer);
    let mut loads = Vec::new();
    for trial in 0..35 {
        let (tokenizer, ms) = timed(load)?;
        if trial >= 5 {
            loads.push(ms);
        }
        drop(tokenizer);
    }
    let load_host = host()?;
    let tokenizer = load()?;
    let mut rows = Vec::new();
    for scope in ["encode", "decode", "decode_skip_special"] {
        let mut trials = Vec::new();
        let mut first_ms = 0.;
        let mut output_hash = String::new();
        for trial in 0..36 {
            let (bytes, ms) = if scope == "encode" {
                let (output, ms) = timed(|| {
                    cases
                        .iter()
                        .map(|c| Ok(tokenizer.encode(&c.text)?))
                        .collect::<Result<Vec<_>>>()
                })?;
                if output.iter().zip(&cases).any(|(o, c)| *o != c.ids) {
                    return Err("encode mismatch".into());
                }
                let mut bytes = Vec::new();
                for ids in output {
                    bytes.extend((ids.len() as u64).to_le_bytes());
                    bytes.extend(ids.into_iter().flat_map(u32::to_le_bytes));
                }
                (bytes, ms)
            } else {
                let skip = scope == "decode_skip_special";
                let (output, ms) = timed(|| {
                    cases
                        .iter()
                        .map(|c| Ok(tokenizer.decode(&c.ids, skip)?))
                        .collect::<Result<Vec<_>>>()
                })?;
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
                (bytes, ms)
            };
            let hash = digest(&bytes);
            if trial == 0 {
                first_ms = ms;
                output_hash = hash;
            } else if hash != output_hash {
                return Err("nonrepeatable output".into());
            }
            if trial >= 6 {
                trials.push(ms);
            }
        }
        rows.push(json!({"scope":scope,"first_ms":first_ms,"trials_ms":trials,"output_sha256":output_hash,"host":host()?}));
    }
    let paths = [
        "crates/ncm-tokenizers/examples/asset_bench.rs",
        "crates/ncm-tokenizers/src/bpe.rs",
        "crates/ncm-tokenizers/src/assets.rs",
        "Cargo.lock",
    ];
    let sources: BTreeMap<_, _> = paths
        .into_iter()
        .map(|p| Ok((p, digest(&fs::read(p)?))))
        .collect::<Result<_>>()?;
    let report = json!({"schema_version":1,"runtime":"rust-tokenizers","profile":profile,"fixture_sha256":digest(&fixture_bytes),
        "manifest_sha256":digest(&fs::read(directory.join("manifest.json"))?),"source_sha256":sources,"warmups":5,"trials":30,
        "strings_per_trial":cases.len(),"input_utf8_bytes":cases.iter().map(|c|c.text.len()).sum::<usize>(),"token_positions":cases.iter().map(|c|c.ids.len()).sum::<usize>(),
        "first_load_ms":first_load_ms,"load_trials_ms":loads,"load_host":load_host,"measurements":rows,"host_final":host()?,
        "load_scope":"authenticated canonical manifest, payload and license; native tokenizer construction; destruction excluded",
        "execution_scope":"89 sequential scalar calls, outputs retained; validation, hashing and destruction excluded; encode specials enabled, decode modes separate",
        "cold_disk_cache":false,"device":"cpu"});
    fs::write(
        &args[4],
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    Ok(())
}
