//! Capacity-bounded parsing for the authenticated canonical BERT vocabulary.
//!
//! Construction and every runtime operation still use the pinned WordPiece
//! implementation. The capacity comes from the fixed, hash-checked profile.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tokenizers::{
    Model, Token,
    models::{ModelWrapper, bpe::Vocab, wordpiece::WordPiece},
};

#[derive(Deserialize)]
enum Kind {
    WordPiece,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Canonical<const CAP: usize> {
    #[serde(rename = "type")]
    _kind: Kind,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
    #[serde(deserialize_with = "crate::hf_bpe::read_vocab::<_, CAP>")]
    vocab: Vocab,
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalWordPiece<const CAP: usize>(WordPiece);

impl<'de, const CAP: usize> Deserialize<'de> for CanonicalWordPiece<CAP> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let c = Canonical::<CAP>::deserialize(deserializer)?;
        WordPiece::builder()
            .vocab(c.vocab)
            .unk_token(c.unk_token)
            .continuing_subword_prefix(c.continuing_subword_prefix)
            .max_input_chars_per_word(c.max_input_chars_per_word)
            .build()
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}
impl<const CAP: usize> Serialize for CanonicalWordPiece<CAP> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}
impl<const CAP: usize> From<CanonicalWordPiece<CAP>> for ModelWrapper {
    fn from(value: CanonicalWordPiece<CAP>) -> Self {
        Self::WordPiece(value.0)
    }
}
impl<const CAP: usize> Model for CanonicalWordPiece<CAP> {
    type Trainer = <WordPiece as Model>::Trainer;
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
    fn preserves_model_and_enforces_profile_capacity() {
        let source = serde_json::json!({"type":"WordPiece", "unk_token":"[UNK]",
            "continuing_subword_prefix":"##", "max_input_chars_per_word":100,
            "vocab":{"[UNK]":0,"a":1,"##b":2}});
        let original: WordPiece = serde_json::from_str(&source.to_string()).unwrap();
        let parsed: CanonicalWordPiece<3> = serde_json::from_value(source.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(parsed).unwrap(),
            serde_json::to_value(original).unwrap()
        );
        assert!(serde_json::from_value::<CanonicalWordPiece<2>>(source.clone()).is_err());
        for (key, value) in [
            ("type", serde_json::json!("BPE")),
            ("extra", serde_json::json!(1)),
        ] {
            let mut bad = source.clone();
            bad[key] = value;
            assert!(serde_json::from_value::<CanonicalWordPiece<3>>(bad).is_err());
        }
    }
}
