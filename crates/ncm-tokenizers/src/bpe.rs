//! The pinned NCM byte BPE, not GPT-2's tokenizer model/ID layout.
use crate::assets::{AssetError, Result};
use fancy_regex::Regex;
use serde::Deserialize;
use std::{borrow::Cow, collections::HashMap};

const PATTERN: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const SPECIAL: [&str; 2] = ["<|endoftext|>", "<|pad|>"];
#[derive(Deserialize)]
struct HexString<'a>(#[serde(borrow)] Cow<'a, str>);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw<'a> {
    schema_version: u32,
    #[serde(borrow)]
    vocab_hex: Vec<HexString<'a>>,
    #[serde(borrow)]
    merges_hex: Vec<[HexString<'a>; 2]>,
}

struct Vocabulary {
    bytes: Box<[u8]>,
    offsets: Box<[u32]>,
}
impl Vocabulary {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
    fn get(&self, id: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(id)? as usize;
        let end = *self.offsets.get(id.checked_add(1)?)? as usize;
        self.bytes.get(start..end)
    }
}
pub struct ByteBpe {
    vocab: Vocabulary,
    // Output ID is 258 + rank, so storing rank separately is redundant.
    merges: HashMap<(u32, u32), u32>,
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
        // One byte arena and a dense offset table instead of one allocation and
        // three machine words per token. The bounded payload fits u32 offsets.
        let mut bytes = Vec::with_capacity(raw.vocab_hex.iter().map(|v| v.0.len() / 2).sum());
        let mut offsets = Vec::with_capacity(raw.vocab_hex.len() + 1);
        offsets.push(0);
        for token in &raw.vocab_hex {
            bytes.extend(hex(&token.0)?);
            offsets
                .push(u32::try_from(bytes.len()).map_err(|_| bad("vocabulary offset overflow"))?);
        }
        let vocab = Vocabulary {
            bytes: bytes.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
        };
        if vocab.get(0) != Some(SPECIAL[0].as_bytes())
            || vocab.get(1) != Some(SPECIAL[1].as_bytes())
            || (0..256).any(|i| vocab.get(i + 2) != Some(&[i as u8][..]))
        {
            return Err(bad("special/byte ID layout differs"));
        }
        let reverse: HashMap<_, _> = (0..vocab.len())
            .map(|i| (vocab.get(i).expect("validated offset table"), i as u32))
            .collect();
        if reverse.len() != vocab.len() {
            return Err(bad("duplicate vocabulary bytes"));
        }
        let mut merges = HashMap::with_capacity(raw.merges_hex.len());
        for (rank, pair) in raw.merges_hex.iter().enumerate() {
            let left = hex(&pair[0].0)?;
            let right = hex(&pair[1].0)?;
            let (Some(&a), Some(&b)) =
                (reverse.get(left.as_slice()), reverse.get(right.as_slice()))
            else {
                return Err(bad("unknown merge operand"));
            };
            let output = (258 + rank) as u32;
            if a >= output
                || b >= output
                || !left.iter().chain(&right).eq(vocab
                    .get(output as usize)
                    .expect("validated output ID")
                    .iter())
                || merges.insert((a, b), output).is_some()
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
                        .filter_map(|p| self.merges.get(&(p[0], p[1])).map(|&id| (id, p[0], p[1])))
                        .min_by_key(|x| x.0);
                    let Some((id, a, b)) = earliest else {
                        break;
                    };
                    // Stable left-to-right compaction preserves all-occurrence
                    // merging. The write cursor never overtakes unread symbols,
                    // so the same allocation can serve every merge round.
                    let mut read = 0;
                    let mut write = 0;
                    while read < word.len() {
                        if read + 1 < word.len() && word[read] == a && word[read + 1] == b {
                            word[write] = id;
                            read += 2;
                        } else {
                            word[write] = word[read];
                            read += 1;
                        }
                        write += 1;
                    }
                    word.truncate(write);
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
        // Bound the eager reservation for long requests or skipped specials.
        // Each ordinary vocabulary token contains at least one byte.
        let mut bytes = Vec::with_capacity(ids.len().min(4096));
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
        // Reuse the output allocation for valid UTF-8; only malformed byte
        // sequences need the pinned replacement-character conversion.
        Ok(match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(error) => String::from_utf8_lossy(error.as_bytes()).into_owned(),
        })
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
    fn packed_vocabulary_preserves_every_entry_and_bounds() {
        let f = fixture();
        let tokenizer = ByteBpe::from_json(&serde_json::to_vec(&f).unwrap()).unwrap();
        let entries = f["vocab_hex"].as_array().unwrap();
        assert_eq!(tokenizer.vocab.offsets.len(), entries.len() + 1);
        let mut total = 0;
        for (id, value) in entries.iter().enumerate() {
            let bytes = hex(value.as_str().unwrap()).unwrap();
            total += bytes.len();
            assert_eq!(tokenizer.vocab.get(id), Some(bytes.as_slice()));
        }
        assert_eq!(tokenizer.vocab.bytes.len(), total);
        assert!(tokenizer.vocab.get(entries.len()).is_none());
        assert!(tokenizer.vocab.get(usize::MAX).is_none());
        assert!(
            tokenizer.vocab.offsets.len() * std::mem::size_of::<u32>()
                < entries.len() * std::mem::size_of::<Vec<u8>>()
        );
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
    fn borrowed_hex_keeps_escaped_json_semantics() {
        let plain = serde_json::to_string(&fixture()).unwrap();
        let escaped = plain.replace("\"61\"", "\"\\u00361\"");
        assert_ne!(plain, escaped);
        let raw_plain: Raw<'_> = serde_json::from_str(&plain).unwrap();
        let raw_escaped: Raw<'_> = serde_json::from_str(&escaped).unwrap();
        assert!(matches!(raw_plain.vocab_hex[99].0, Cow::Borrowed(_)));
        assert!(matches!(raw_escaped.vocab_hex[99].0, Cow::Owned(_)));
        let a = ByteBpe::from_json(plain.as_bytes()).unwrap();
        let b = ByteBpe::from_json(escaped.as_bytes()).unwrap();
        for text in ["", "aaaa", "ababab", "é a", "<|pad|>a"] {
            let ids = a.encode(text).unwrap();
            assert_eq!(ids, b.encode(text).unwrap());
            assert_eq!(
                a.decode(&ids, false).unwrap(),
                b.decode(&ids, false).unwrap()
            );
        }
    }
    #[test]
    fn decode_long_inputs_and_skipped_specials() {
        let tokenizer = ByteBpe::from_json(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let mut ids = vec![b'a' as u32 + 2; 5000];
        ids.extend([0, 1, 0xf0 + 2, 0x28 + 2]);
        assert_eq!(
            tokenizer.decode(&ids, true).unwrap(),
            format!("{}�(", "a".repeat(5000))
        );
        assert_eq!(tokenizer.decode(&vec![0; 5000], true).unwrap(), "");
        ids.push(u32::MAX);
        assert!(tokenizer.decode(&ids, true).is_err());
    }
    #[test]
    fn in_place_merge_matches_rebuilding_for_overlaps_and_rank_order() {
        // Include self-overlap and competing pairs, with each output composed
        // only from earlier entries, as required by the canonical byte BPE.
        let mut raw = fixture();
        raw["vocab_hex"].as_array_mut().unwrap().extend([
            serde_json::json!("6161"),
            serde_json::json!("6261"),
            serde_json::json!("616162"),
            serde_json::json!("616261"),
        ]);
        raw["merges_hex"].as_array_mut().unwrap().extend([
            serde_json::json!(["61", "61"]),
            serde_json::json!(["62", "61"]),
            serde_json::json!(["6161", "62"]),
            serde_json::json!(["6162", "61"]),
        ]);
        let tokenizer = ByteBpe::from_json(&serde_json::to_vec(&raw).unwrap()).unwrap();
        let rebuild = |text: &str| {
            let mut word: Vec<u32> = text.bytes().map(|b| u32::from(b) + 2).collect();
            loop {
                let chosen = word
                    .windows(2)
                    .filter_map(|p| {
                        tokenizer
                            .merges
                            .get(&(p[0], p[1]))
                            .map(|&rank| (rank, p[0], p[1]))
                    })
                    .min_by_key(|v| v.0);
                let Some((id, a, b)) = chosen else {
                    return word;
                };
                let mut next = Vec::new();
                let mut i = 0;
                while i < word.len() {
                    if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                        next.push(id);
                        i += 2;
                    } else {
                        next.push(word[i]);
                        i += 1;
                    }
                }
                word = next;
            }
        };
        // Exhaust all binary words through length ten, plus long overlapping runs.
        for length in 0..=10 {
            for bits in 0..(1usize << length) {
                let text: String = (0..length)
                    .map(|i| if bits & (1 << i) == 0 { 'a' } else { 'b' })
                    .collect();
                assert_eq!(tokenizer.encode(&text).unwrap(), rebuild(&text), "{text:?}");
            }
        }
        for text in ["a".repeat(4097), "ab".repeat(2049), "baab".repeat(1024)] {
            assert_eq!(tokenizer.encode(&text).unwrap(), rebuild(&text));
        }
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
