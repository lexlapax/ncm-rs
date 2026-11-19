//! Evaluator-compatible decoding of generated text IDs.
use crate::Result;
use ncm_io::Artifact;
use ncm_tokenizers::TextTokenizer;

struct Policy {
    tokenizer_profile: &'static str,
    eos: u32,
    pad: u32,
    model_vocab: u32,
}

fn policy(profile: &str) -> Result<Policy> {
    match profile {
        "tinystories-inference-v1" => Ok(Policy {
            tokenizer_profile: "tinystories",
            eos: 0,
            pad: 1,
            model_vocab: 8192,
        }),
        "xsum-inference-v1" | "xsum-inference-v2" => Ok(Policy {
            tokenizer_profile: "gpt2",
            eos: 50256,
            pad: 50257,
            model_vocab: 50304,
        }),
        "wmt14-inference-v1" => Ok(Policy {
            tokenizer_profile: "qwen3",
            eos: 151645,
            pad: 151643,
            model_vocab: 151680,
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

#[cfg(test)]
mod tests {
    use super::{evaluator_ids, policy};
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
