//! Evaluator-compatible decoding of generated text IDs.
use crate::Result;
use ncm_io::Artifact;
use ncm_tokenizers::TextTokenizer;

struct Policy {
    tokenizer_profile: &'static str,
    eos: u32,
    pad: u32,
    model_vocab: u32,
    input_length: Option<usize>,
}

fn policy(profile: &str) -> Result<Policy> {
    match profile {
        "tinystories-inference-v1" => Ok(Policy {
            tokenizer_profile: "tinystories",
            eos: 0,
            pad: 1,
            model_vocab: 8192,
            input_length: None,
        }),
        "xsum-inference-v1" | "xsum-inference-v2" => Ok(Policy {
            tokenizer_profile: "gpt2",
            eos: 50256,
            pad: 50257,
            model_vocab: 50304,
            input_length: Some(512),
        }),
        "wmt14-inference-v1" => Ok(Policy {
            tokenizer_profile: "qwen3",
            eos: 151645,
            pad: 151643,
            model_vocab: 151680,
            input_length: Some(96),
        }),
        _ => Err("decoding is unsupported for this artifact profile".into()),
    }
}

/// The evaluators truncate at the first EOS, remove padding and ask their
/// tokenizer to skip special tokens. HF ignores model-only padded vocabulary
/// IDs absent from the tokenizer; filter only those IDs, within model bounds.
fn evaluator_ids(row: &[u32], policy: &Policy, tokenizer_vocab: usize) -> Result<Vec<u32>> {
    if row.iter().any(|&id| id >= policy.model_vocab) {
        return Err("generated ID exceeds model vocabulary".into());
    }
    let end = row
        .iter()
        .position(|&id| id == policy.eos)
        .unwrap_or(row.len());
    Ok(row[..end]
        .iter()
        .copied()
        .filter(|&id| id != policy.pad && (id as usize) < tokenizer_vocab)
        .collect())
}

pub(super) struct Decoder {
    tokenizer: TextTokenizer,
    policy: Policy,
}

impl Decoder {
    pub(super) fn from_artifact(artifact: &Artifact) -> Result<Self> {
        let policy = policy(&artifact.manifest().profile)?;
        let tokenizer = TextTokenizer::from_asset_bytes(
            artifact.file("tokenizer-manifest.json")?,
            artifact.file("tokenizer.json")?,
            artifact.file("tokenizer.LICENSE")?,
        )?;
        let meta = tokenizer.metadata();
        if meta.profile != policy.tokenizer_profile
            || meta.eos_token_id != policy.eos
            || meta.pad_token_id != policy.pad
            || meta.total_vocab_size > policy.model_vocab as usize
        {
            return Err("tokenizer does not match generation profile".into());
        }
        Ok(Self { tokenizer, policy })
    }

    pub(super) fn encode_texts(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        if texts.is_empty() || texts.len() > 128 {
            return Err("text input must contain 1..=128 strings".into());
        }
        texts
            .iter()
            .map(|text| {
                let mut ids = self.tokenizer.encode(text)?;
                if let Some(length) = self.policy.input_length {
                    ids.truncate(length);
                    ids.resize(length, self.policy.pad);
                } else {
                    validate_tinystories_prefix(&ids)?;
                }
                Ok(ids)
            })
            .collect()
    }

    pub(super) fn add_texts(&self, result: &mut serde_json::Value) -> Result<()> {
        let ids: Vec<Vec<u32>> =
            serde_json::from_value(result.get("ids").ok_or("missing generated IDs")?.clone())?;
        let texts: Vec<String> = ids
            .iter()
            .map(|row| {
                let selected = evaluator_ids(
                    row,
                    &self.policy,
                    self.tokenizer.metadata().total_vocab_size,
                )?;
                Ok(self.tokenizer.decode(&selected, true)?)
            })
            .collect::<Result<_>>()?;
        result["texts"] = serde_json::json!(texts);
        Ok(())
    }
}

fn validate_tinystories_prefix(ids: &[u32]) -> Result<()> {
    if !matches!(ids, [id] if *id < 8192 && *id != 1) {
        return Err("TinyStories text must encode to exactly one non-padding token".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Decoder, evaluator_ids, policy, validate_tinystories_prefix};

    #[test]
    fn tinystories_text_requires_exactly_one_non_padding_id() {
        for ids in [&[][..], &[1], &[8192], &[2, 3]] {
            assert!(validate_tinystories_prefix(ids).is_err());
        }
        for ids in [&[0][..], &[2], &[8191]] {
            assert!(validate_tinystories_prefix(ids).is_ok());
        }
    }
    #[test]
    #[ignore = "requires pinned local TinyStories artifact"]
    fn tinystories_text_uses_pinned_tokenizer_and_rejects_unrepresentable_prefixes() {
        let artifact =
            ncm_io::Artifact::load_inference(std::env::var("NCM_TINYSTORIES_ARTIFACT").unwrap())
                .unwrap();
        let decoder = Decoder::from_artifact(&artifact).unwrap();
        let texts = ["a", "hello", "<|endoftext|>"].map(str::to_owned);
        assert_eq!(
            decoder.encode_texts(&texts).unwrap(),
            [vec![99], vec![5663], vec![0]]
        );
        for text in ["", "<|pad|>", "hello world", "😀"] {
            assert!(
                decoder.encode_texts(&[text.to_owned()]).is_err(),
                "{text:?}"
            );
        }
        assert!(decoder.encode_texts(&[]).is_err());
        assert!(decoder.encode_texts(&vec!["a".to_owned(); 129]).is_err());
    }

    #[test]
    #[ignore = "requires pinned TinyStories artifact and generated upstream text reference"]
    fn tinystories_text_matches_upstream_vocabulary_reference() {
        let artifact =
            ncm_io::Artifact::load_inference(std::env::var("NCM_TINYSTORIES_ARTIFACT").unwrap())
                .unwrap();
        let decoder = Decoder::from_artifact(&artifact).unwrap();
        drop(artifact);
        let raw = std::fs::read(std::env::var("NCM_TINYSTORIES_TEXT_REFERENCE").unwrap()).unwrap();
        assert_eq!(
            ncm_io::sha256(&raw),
            "b693f72b52e4803b8604774f63f3edc5eaed1b609136d2d0218c76485068add3"
        );
        let data: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let rows = data["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 8199);
        let mut accepted = 0;
        for (index, row) in rows.iter().enumerate() {
            let text = row["text"].as_str().unwrap().to_owned();
            let expected: Vec<u32> = serde_json::from_value(row["ids"].clone()).unwrap();
            let actual = decoder.encode_texts(&[text]);
            if matches!(expected.as_slice(), [id] if *id < 8192 && *id != 1) {
                assert_eq!(actual.unwrap(), vec![expected], "row {index}");
                accepted += 1;
            } else {
                assert!(actual.is_err(), "row {index}");
            }
        }
        assert_eq!(accepted, 8064);
    }

    #[test]
    #[ignore = "requires pinned local artifacts and external reference corpora"]
    fn corpus_text_encoding_matches_every_reference_row() {
        for (artifact_var, reference_var, count) in [
            ("NCM_XSUM_ARTIFACT", "NCM_XSUM_TEXT_REFERENCE", 11334),
            ("NCM_WMT14_ARTIFACT", "NCM_WMT14_TEXT_REFERENCE", 3003),
        ] {
            let artifact =
                ncm_io::Artifact::load_inference(std::env::var(artifact_var).unwrap()).unwrap();
            let decoder = Decoder::from_artifact(&artifact).unwrap();
            drop(artifact);
            let data: serde_json::Value = serde_json::from_slice(
                &std::fs::read(std::env::var(reference_var).unwrap()).unwrap(),
            )
            .unwrap();
            let texts: Vec<String> = serde_json::from_value(data["texts"].clone()).unwrap();
            let ids: Vec<Vec<u32>> = serde_json::from_value(data["ids"].clone()).unwrap();
            assert_eq!(texts.len(), count);
            assert_eq!(ids.len(), count);
            for (i, (text_chunk, id_chunk)) in texts.chunks(128).zip(ids.chunks(128)).enumerate() {
                assert_eq!(
                    decoder.encode_texts(text_chunk).unwrap(),
                    id_chunk,
                    "mismatch in batch {i}"
                );
            }
        }
    }

    #[test]
    fn evaluator_stops_at_eos_and_discards_padding_and_model_only_ids() {
        let x = policy("xsum-inference-v2").unwrap();
        assert_eq!(
            evaluator_ids(&[42, 50257, 50258, 43, 50256, 44], &x, 50258).unwrap(),
            [42, 43]
        );
        assert!(evaluator_ids(&[50304], &x, 50258).is_err());
        let t = policy("tinystories-inference-v1").unwrap();
        assert_eq!(evaluator_ids(&[1, 5, 0, 9], &t, 8192).unwrap(), [5]);
        let w = policy("wmt14-inference-v1").unwrap();
        assert_eq!(
            evaluator_ids(&[151643, 42, 151669, 151645, 43], &w, 151669).unwrap(),
            [42]
        );
        assert!(policy("owt-inference-v1").is_err());
    }
}
