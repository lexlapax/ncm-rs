//! Direct parsing of the authenticated canonical tuple-merge BPE format.
//!
//! Tokenizers 0.22.2 accepts both tuple and legacy string merges through an
//! untagged enum, buffering a generic serde tree before constructing the pairs.
//! Our pinned assets use only tuples. Avoid that intermediate tree, but delegate
//! construction and every model operation to the same pinned BPE implementation.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tokenizers::{
    Model, Token,
    models::{
        ModelWrapper,
        bpe::{BPE, Vocab},
    },
};

#[derive(Deserialize)]
enum Kind {
    #[serde(rename = "BPE")]
    Bpe,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Canonical {
    #[serde(rename = "type")]
    _kind: Kind,
    dropout: Option<f32>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: Option<bool>,
    byte_fallback: Option<bool>,
    ignore_merges: Option<bool>,
    vocab: Vocab,
    merges: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalBpe(BPE);

impl<'de> Deserialize<'de> for CanonicalBpe {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let c = Canonical::deserialize(deserializer)?;
        let mut builder = BPE::builder().vocab_and_merges(c.vocab, c.merges);
        if let Some(v) = c.dropout {
            builder = builder.dropout(v);
        }
        if let Some(v) = c.unk_token {
            builder = builder.unk_token(v);
        }
        if let Some(v) = c.continuing_subword_prefix {
            builder = builder.continuing_subword_prefix(v);
        }
        if let Some(v) = c.end_of_word_suffix {
            builder = builder.end_of_word_suffix(v);
        }
        if let Some(v) = c.fuse_unk {
            builder = builder.fuse_unk(v);
        }
        if let Some(v) = c.byte_fallback {
            builder = builder.byte_fallback(v);
        }
        if let Some(v) = c.ignore_merges {
            builder = builder.ignore_merges(v);
        }
        builder.build().map(Self).map_err(serde::de::Error::custom)
    }
}
impl Serialize for CanonicalBpe {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}
impl From<CanonicalBpe> for ModelWrapper {
    fn from(value: CanonicalBpe) -> Self {
        Self::BPE(value.0)
    }
}
impl Model for CanonicalBpe {
    type Trainer = <BPE as Model>::Trainer;
    fn tokenize(&self, sequence: &str) -> tokenizers::Result<Vec<Token>> {
        self.0.tokenize(sequence)
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.0.get_vocab()
    }
    fn get_vocab_size(&self) -> usize {
        self.0.get_vocab_size()
    }
    fn save(&self, folder: &Path, prefix: Option<&str>) -> tokenizers::Result<Vec<PathBuf>> {
        self.0.save(folder, prefix)
    }
    fn get_trainer(&self) -> Self::Trainer {
        self.0.get_trainer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_options_and_rejects_noncanonical_models() {
        let original = BPE::builder()
            .vocab_and_merges(
                [("a".into(), 0), ("b".into(), 1), ("ab".into(), 2)]
                    .into_iter()
                    .collect::<Vocab>(),
                vec![("a".into(), "b".into())],
            )
            .dropout(0.0)
            .unk_token("a".into())
            .continuing_subword_prefix("".into())
            .end_of_word_suffix("".into())
            .fuse_unk(true)
            .byte_fallback(true)
            .ignore_merges(true)
            .build()
            .unwrap();
        let json = serde_json::to_value(&original).unwrap();
        let parsed: CanonicalBpe = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), json);
        for (key, value) in [
            ("type", serde_json::json!("WordPiece")),
            ("merges", serde_json::json!(["a b"])),
            ("unknown", serde_json::json!(true)),
            ("dropout", serde_json::json!(2.0)),
        ] {
            let mut bad = json.clone();
            bad[key] = value;
            assert!(serde_json::from_value::<CanonicalBpe>(bad).is_err());
        }
    }
}
