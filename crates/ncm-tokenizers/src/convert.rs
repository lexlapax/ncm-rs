//! Pure-Rust canonical tokenizer asset builder reproducing the pinned
//! `tools/reference/tokenizers-v1.lock.json` manifests and payloads offline,
//! with zero Python dependencies.

use crate::assets::{AssetError, Result, TextTokenizer};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use tokenizers::AddedToken;

const UPSTREAM_COMMIT: &str = "c70274273f489abc73832b2b98584a8483b6b203";

/// Completed canonical tokenizer asset bytes (`manifest.json`, `tokenizer.json`, `LICENSE`),
/// already verified against `tools/reference/tokenizers-v1.lock.json` via `TextTokenizer::from_asset_bytes`.
#[derive(Debug, Clone)]
pub struct CanonicalTokenizerAsset {
    pub profile: String,
    pub manifest_json: Vec<u8>,
    pub tokenizer_json: Vec<u8>,
    pub license: Vec<u8>,
}

fn invalid(msg: impl Into<String>) -> AssetError {
    AssetError::Invalid(msg.into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Format a `serde_json::Value` identically to Python's
/// `json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"`.
pub fn format_python_json(value: &Value) -> Vec<u8> {
    let mut out = String::new();
    write_py_json(value, 0, &mut out);
    out.push('\n');
    out.into_bytes()
}

fn write_py_json_str(s: &str, out: &mut String) {
    use std::fmt::Write;
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if (c as u32) <= 0x7e => out.push(c),
            c if (c as u32) <= 0xffff => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => {
                let mut pair = [0u16; 2];
                for unit in c.encode_utf16(&mut pair) {
                    let _ = write!(out, "\\u{:04x}", unit);
                }
            }
        }
    }
    out.push('"');
}

fn write_indent(depth: usize, out: &mut String) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn format_py_number(n: &serde_json::Number, out: &mut String) {
    if let Some(i) = n.as_i64() {
        out.push_str(&i.to_string());
        return;
    }
    if let Some(u) = n.as_u64() {
        out.push_str(&u.to_string());
        return;
    }
    if let Some(f) = n.as_f64() {
        let abs = f.abs();
        if abs > 0.0 && !(1e-4..1e16).contains(&abs) {
            // Python repr(float): shortest significand + `e` + sign + >=2-digit exponent (e.g. `1e-05`).
            let raw = format!("{f:e}");
            if let Some((mant, exp_str)) = raw.split_once('e')
                && let Ok(exp) = exp_str.parse::<i32>()
            {
                let sign = if exp < 0 { '-' } else { '+' };
                let mag = exp.unsigned_abs();
                out.push_str(&format!("{mant}e{sign}{mag:02}"));
                return;
            }
        }
        let s = f.to_string();
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            out.push_str(&s);
            out.push_str(".0");
        } else {
            out.push_str(&s);
        }
        return;
    }
    out.push_str(&n.to_string());
}

fn write_py_json(value: &Value, depth: usize, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => format_py_number(n, out),
        Value::String(s) => write_py_json_str(s, out),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                write_indent(depth + 1, out);
                write_py_json(item, depth + 1, out);
                if i + 1 < items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            write_indent(depth, out);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (i, key) in keys.iter().enumerate() {
                write_indent(depth + 1, out);
                write_py_json_str(key, out);
                out.push_str(": ");
                write_py_json(&map[*key], depth + 1, out);
                if i + 1 < keys.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            write_indent(depth, out);
            out.push('}');
        }
    }
}

fn source_record(profile: &str) -> Result<Value> {
    let lock: Value = serde_json::from_str(include_str!(
        "../../../tools/reference/tokenizer-sources.lock.json"
    ))?;
    lock["sources"]
        .get(profile)
        .cloned()
        .ok_or_else(|| invalid(format!("unknown tokenizer source profile `{profile}`")))
}

#[allow(clippy::too_many_arguments)]
fn build_manifest(
    profile: &str,
    kind: &str,
    vocab_size: usize,
    total_vocab_size: usize,
    eos_token_id: u32,
    pad_token_id: u32,
    cleanup: bool,
    payload: &[u8],
    license: &[u8],
) -> Result<Vec<u8>> {
    let source = source_record(profile)?;
    let mut ref_versions = Map::new();
    ref_versions.insert("regex".into(), Value::String("2025.9.18".into()));
    ref_versions.insert("tokenizers".into(), Value::String("0.22.2".into()));
    ref_versions.insert("transformers".into(), Value::String("4.57.6".into()));

    let mut manifest = Map::new();
    manifest.insert("cleanup".into(), Value::Bool(cleanup));
    manifest.insert("eos_token_id".into(), Value::from(eos_token_id));
    manifest.insert("kind".into(), Value::String(kind.into()));
    manifest.insert("license_sha256".into(), Value::String(sha256_hex(license)));
    manifest.insert("pad_token_id".into(), Value::from(pad_token_id));
    manifest.insert("payload_bytes".into(), Value::from(payload.len()));
    manifest.insert("payload_sha256".into(), Value::String(sha256_hex(payload)));
    manifest.insert("profile".into(), Value::String(profile.into()));
    manifest.insert("reference_versions".into(), Value::Object(ref_versions));
    manifest.insert("schema_version".into(), Value::from(1));
    manifest.insert("source".into(), source);
    manifest.insert("total_vocab_size".into(), Value::from(total_vocab_size));
    manifest.insert(
        "upstream_commit".into(),
        Value::String(UPSTREAM_COMMIT.into()),
    );
    manifest.insert("vocab_size".into(), Value::from(vocab_size));
    Ok(format_python_json(&Value::Object(manifest)))
}

/// Build and authenticate the pinned TinyStories `byte_bpe` canonical tokenizer asset
/// directly from its decoded vocabulary (`0..8192`) and ranked merges (`7934` pairs).
pub fn build_tinystories_tokenizer_asset(
    vocab: &[Vec<u8>],
    merges: &[(Vec<u8>, Vec<u8>)],
    license: &[u8],
) -> Result<CanonicalTokenizerAsset> {
    if vocab.len() != 8192 {
        return Err(invalid(format!(
            "TinyStories BPE vocab length {} != 8192",
            vocab.len()
        )));
    }
    if vocab[0] != b"<|endoftext|>" || vocab[1] != b"<|pad|>" {
        return Err(invalid("TinyStories special token IDs 0/1 differ"));
    }
    for byte in 0u8..=255 {
        if vocab[byte as usize + 2] != [byte] {
            return Err(invalid("TinyStories base byte vocabulary differs"));
        }
    }
    if merges.len() != 7934 {
        return Err(invalid(format!(
            "TinyStories BPE merge count {} != 7934",
            merges.len()
        )));
    }
    let mut known: BTreeSet<&[u8]> = vocab[..258].iter().map(Vec::as_slice).collect();
    let mut seen_pairs: BTreeSet<(&[u8], &[u8])> = BTreeSet::new();
    for (rank, (a, b)) in merges.iter().enumerate() {
        if !seen_pairs.insert((a.as_slice(), b.as_slice()))
            || !known.contains(a.as_slice())
            || !known.contains(b.as_slice())
        {
            return Err(invalid(format!(
                "inconsistent TinyStories BPE merge at rank {rank}"
            )));
        }
        let mut merged = Vec::with_capacity(a.len() + b.len());
        merged.extend_from_slice(a);
        merged.extend_from_slice(b);
        if merged != vocab[258 + rank] {
            return Err(invalid(format!(
                "TinyStories BPE merge rank {rank} does not match vocab[{}]",
                258 + rank
            )));
        }
        known.insert(vocab[258 + rank].as_slice());
    }

    let vocab_hex: Vec<Value> = vocab.iter().map(|v| Value::String(hex_encode(v))).collect();
    let merges_hex: Vec<Value> = merges
        .iter()
        .map(|(a, b)| {
            Value::Array(vec![
                Value::String(hex_encode(a)),
                Value::String(hex_encode(b)),
            ])
        })
        .collect();

    let mut payload_map = Map::new();
    payload_map.insert("merges_hex".into(), Value::Array(merges_hex));
    payload_map.insert("schema_version".into(), Value::from(1));
    payload_map.insert("vocab_hex".into(), Value::Array(vocab_hex));
    let tokenizer_json = format_python_json(&Value::Object(payload_map));

    let manifest_json = build_manifest(
        "tinystories",
        "byte_bpe",
        8192,
        8192,
        0,
        1,
        false,
        &tokenizer_json,
        license,
    )?;

    // Authenticate against `tools/reference/tokenizers-v1.lock.json`.
    TextTokenizer::from_asset_bytes(&manifest_json, &tokenizer_json, license)?;
    Ok(CanonicalTokenizerAsset {
        profile: "tinystories".into(),
        manifest_json,
        tokenizer_json,
        license: license.to_vec(),
    })
}

/// Build and authenticate a pinned HuggingFace canonical tokenizer asset (`gpt2`, `qwen3`, `bert`)
/// directly from its raw verified upstream `tokenizer.json` and license bytes.
pub fn build_hf_tokenizer_asset(
    profile: &str,
    raw_tokenizer_json: &[u8],
    license: &[u8],
) -> Result<CanonicalTokenizerAsset> {
    let (vocab_size, total_vocab_size, eos_id, pad_id, add_pad) = match profile {
        "gpt2" => (50257usize, 50258usize, 50256u32, 50257u32, Some("<|pad|>")),
        "qwen3" => (151643usize, 151669usize, 151645u32, 151643u32, None),
        "bert" => (30522usize, 30522usize, 101u32, 0u32, None),
        other => {
            return Err(invalid(format!(
                "unsupported HuggingFace tokenizer profile `{other}`"
            )));
        }
    };

    let mut tok = tokenizers::Tokenizer::from_bytes(raw_tokenizer_json)
        .map_err(|e| invalid(format!("parse raw {profile} tokenizer.json: {e}")))?;

    if let Some(pad_tok) = add_pad {
        tok.add_special_tokens(&[AddedToken::from(pad_tok, true)]);
    }

    let serialized = tok
        .to_string(false)
        .map_err(|e| invalid(format!("serialize {profile} tokenizer: {e}")))?;
    let parsed: Value = serde_json::from_str(&serialized)?;
    let tokenizer_json = format_python_json(&parsed);

    let manifest_json = build_manifest(
        profile,
        "huggingface",
        vocab_size,
        total_vocab_size,
        eos_id,
        pad_id,
        false,
        &tokenizer_json,
        license,
    )?;

    // Authenticate against `tools/reference/tokenizers-v1.lock.json`!
    TextTokenizer::from_asset_bytes(&manifest_json, &tokenizer_json, license)?;
    Ok(CanonicalTokenizerAsset {
        profile: profile.into(),
        manifest_json,
        tokenizer_json,
        license: license.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn python_json_formatter_escapes_unicode_and_sorts_keys() {
        let val =
            serde_json::json!({"z": "café", "a": [], "b": {}, "c": "😀", "e": 1e-5, "f": 1.0});
        let text = String::from_utf8(format_python_json(&val)).unwrap();
        assert_eq!(
            text,
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": \"\\ud83d\\ude00\",\n  \"e\": 1e-05,\n  \"f\": 1.0,\n  \"z\": \"caf\\u00e9\"\n}\n"
        );
    }

    #[test]
    #[ignore = "requires local raw upstream tokenizers under artifacts/upstream/tokenizers-v2"]
    fn real_hf_tokenizers_match_pinned_lock_bit_for_bit() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let raw_dir = root.join("artifacts/upstream/tokenizers-v2");
        for (profile, lic_name) in [
            ("gpt2", "OPENAI_LICENSE.txt"),
            ("qwen3", "LICENSE"),
            ("bert", "LICENSE"),
        ] {
            let raw_tok = std::fs::read(raw_dir.join(profile).join("tokenizer.json")).unwrap();
            let raw_lic = std::fs::read(raw_dir.join(profile).join(lic_name)).unwrap();
            let asset = build_hf_tokenizer_asset(profile, &raw_tok, &raw_lic).unwrap();
            assert_eq!(asset.profile, profile);
        }
    }
}
