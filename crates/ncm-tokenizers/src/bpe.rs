//! The pinned NCM byte BPE, not GPT-2's tokenizer model/ID layout.
use crate::assets::{AssetError, Result};
use fancy_regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;

const PATTERN: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const SPECIAL: [&str; 2] = ["<|endoftext|>", "<|pad|>"];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    schema_version: u32,
    vocab_hex: Vec<String>,
    merges_hex: Vec<[String; 2]>,
}

pub struct ByteBpe {
    vocab: Vec<Vec<u8>>,
    merges: HashMap<(u32, u32), (usize, u32)>,
    pattern: Regex,
}
fn bad(message: &str) -> AssetError {
    AssetError::Invalid(message.into())
}
fn hex(text: &str) -> Result<Vec<u8>> {
    if text.is_empty() || !text.len().is_multiple_of(2) || text.len() > 8192 {
        return Err(bad("invalid bounded byte hex"));
    }
    let digit = |b| match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err(bad("expected lowercase hex")),
    };
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| Ok(digit(p[0])? * 16 + digit(p[1])?))
        .collect()
}
impl ByteBpe {
    /// Strict data-only form, also usable with synthetic byte-BPE test fixtures.
    /// This constructor is structural validation, not source authentication.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 32 * 1024 * 1024 {
            return Err(bad("byte BPE exceeds 32 MiB"));
        }
        let raw: Raw = serde_json::from_slice(bytes)?;
        if raw.schema_version != 1
            || !(258..=65536).contains(&raw.vocab_hex.len())
            || raw.merges_hex.len() != raw.vocab_hex.len() - 258
        {
            return Err(bad("invalid byte BPE schema/inventory"));
        }
        let vocab: Vec<_> = raw
            .vocab_hex
            .iter()
            .map(|v| hex(v))
            .collect::<Result<_>>()?;
        if vocab[0] != SPECIAL[0].as_bytes()
            || vocab[1] != SPECIAL[1].as_bytes()
            || (0..256).any(|i| vocab[i + 2] != [i as u8])
        {
            return Err(bad("special/byte ID layout differs"));
        }
        let reverse: HashMap<_, _> = vocab
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i as u32))
            .collect();
        if reverse.len() != vocab.len() {
            return Err(bad("duplicate vocabulary bytes"));
        }
        let mut merges = HashMap::new();
        for (rank, pair) in raw.merges_hex.iter().enumerate() {
            let left = hex(&pair[0])?;
            let right = hex(&pair[1])?;
            let (Some(&a), Some(&b)) = (reverse.get(&left), reverse.get(&right)) else {
                return Err(bad("unknown merge operand"));
            };
            let output = (258 + rank) as u32;
            if a >= output
                || b >= output
                || [left, right].concat() != vocab[output as usize]
                || merges.insert((a, b), (rank, output)).is_some()
            {
                return Err(bad("inconsistent ranked merge"));
            }
        }
        let pattern = Regex::new(PATTERN).map_err(|e| bad(&e.to_string()))?;
        Ok(Self {
            vocab,
            merges,
            pattern,
        })
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.len() > 1024 * 1024 {
            return Err(bad("text exceeds 1 MiB request bound"));
        }
        let mut output = Vec::new();
        let mut rest = text;
        // The two fixed special strings have no ambiguous prefix; earliest text
        // occurrence matches upstream's escaped reverse-lexicographic split.
        loop {
            let next = SPECIAL
                .iter()
                .enumerate()
                .filter_map(|(id, t)| rest.find(t).map(|pos| (pos, id)))
                .min();
            let (chunk, special) = match next {
                Some((pos, id)) => (&rest[..pos], Some((pos, id))),
                None => (rest, None),
            };
            let mut end = 0;
            for matched in self.pattern.find_iter(chunk) {
                let matched = matched.map_err(|e| bad(&e.to_string()))?;
                if matched.start() != end {
                    return Err(bad("pretokenizer failed to cover input"));
                }
                end = matched.end();
                let mut word: Vec<u32> =
                    matched.as_str().bytes().map(|b| u32::from(b) + 2).collect();
                loop {
                    let earliest = word
                        .windows(2)
                        .filter_map(|p| {
                            self.merges
                                .get(&(p[0], p[1]))
                                .map(|&(rank, id)| (rank, id, p[0], p[1]))
                        })
                        .min_by_key(|x| x.0);
                    let Some((_, id, a, b)) = earliest else {
                        break;
                    };
                    let mut merged = Vec::with_capacity(word.len());
                    let mut i = 0;
                    while i < word.len() {
                        if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                            merged.push(id);
                            i += 2;
                        } else {
                            merged.push(word[i]);
                            i += 1;
                        }
                    }
                    word = merged;
                }
                output.extend(word);
            }
            if end != chunk.len() {
                return Err(bad("pretokenizer failed to cover input"));
            }
            match special {
                Some((pos, id)) => {
                    output.push(id as u32);
                    rest = &rest[pos + SPECIAL[id].len()..];
                }
                None => break,
            }
        }
        Ok(output)
    }
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        if ids.len() > 1024 * 1024 {
            return Err(bad("too many input token IDs"));
        }
        let mut bytes = Vec::new();
        for &id in ids {
            let value = self
                .vocab
                .get(id as usize)
                .ok_or_else(|| bad("invalid byte-BPE token ID"))?;
            if !skip_special_tokens || id > 1 {
                if bytes.len() + value.len() > 16 * 1024 * 1024 {
                    return Err(bad("decoded bytes exceed bound"));
                }
                bytes.extend_from_slice(value);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        let mut vocab = vec![
            SPECIAL[0].as_bytes().to_vec(),
            SPECIAL[1].as_bytes().to_vec(),
        ];
        vocab.extend((0..=255).map(|b| vec![b]));
        vocab.extend([b"ab".to_vec(), b"abab".to_vec()]);
        serde_json::json!({"schema_version":1,"vocab_hex":vocab.iter().map(|v|v.iter().map(|b|format!("{b:02x}")).collect::<String>()).collect::<Vec<_>>(),"merges_hex":[["61","62"],["6162","6162"]]})
    }
    #[test]
    fn ranked_all_occurrence_merges_and_specials() {
        let tokenizer = ByteBpe::from_json(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        assert_eq!(tokenizer.encode("ababab").unwrap(), [259, 258]);
        assert_eq!(
            tokenizer.encode("<|pad|>abab<|endoftext|>").unwrap(),
            [1, 259, 0]
        );
        assert_eq!(tokenizer.decode(&[1, 259, 0], true).unwrap(), "abab");
        assert_eq!(
            tokenizer.decode(&[1, 259, 0], false).unwrap(),
            "<|pad|>abab<|endoftext|>"
        );
        assert_eq!(
            tokenizer
                .decode(&[0xf0 + 2, 0x28 + 2, 0x8c + 2, 0x28 + 2], false)
                .unwrap(),
            "�(�("
        );
        assert!(tokenizer.decode(&[u32::MAX], true).is_err());
    }
    #[test]
    fn unicode_bytes_whitespace_and_no_implicit_normalization() {
        let tokenizer = ByteBpe::from_json(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        for text in [
            "é e\u{301} 中文 🙂",
            "a\r\n\t \u{a0}b",
            "\u{1c89}\u{105c0}",
            "",
        ] {
            assert_eq!(
                tokenizer
                    .decode(&tokenizer.encode(text).unwrap(), false)
                    .unwrap(),
                text
            );
        }
    }
    #[test]
    fn invalid_metadata_merges_and_hex_fail() {
        let mut f = fixture();
        f["unknown"] = true.into();
        assert!(ByteBpe::from_json(&serde_json::to_vec(&f).unwrap()).is_err());
        let mut f = fixture();
        f["merges_hex"][0][0] = "6162".into();
        assert!(ByteBpe::from_json(&serde_json::to_vec(&f).unwrap()).is_err());
        let mut f = fixture();
        f["vocab_hex"][2] = "☃a".into();
        assert!(ByteBpe::from_json(&serde_json::to_vec(&f).unwrap()).is_err());
        assert!(hex("A0").is_err());
        assert!(hex("0").is_err());
    }
}
