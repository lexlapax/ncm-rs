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
struct Canonical<const CAP: usize> {
    #[serde(rename = "type")]
    _kind: Kind,
    dropout: Option<f32>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: Option<bool>,
    byte_fallback: Option<bool>,
    ignore_merges: Option<bool>,
    #[serde(deserialize_with = "read_vocab::<_, CAP>")]
    vocab: Vocab,
    #[serde(deserialize_with = "read_merges::<_, CAP>")]
    merges: Vec<(String, String)>,
}

// Capacities come from the authenticated profile, not untrusted size hints.
pub(crate) fn read_vocab<'de, D: Deserializer<'de>, const CAP: usize>(
    d: D,
) -> Result<Vocab, D::Error> {
    struct Visitor<const CAP: usize>;
    impl<'de, const CAP: usize> serde::de::Visitor<'de> for Visitor<CAP> {
        type Value = Vocab;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("bounded canonical vocabulary")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Vocab, A::Error> {
            let mut vocab = Vocab::with_capacity(CAP);
            while let Some((key, id)) = map.next_entry::<String, u32>()? {
                vocab.insert(key, id);
                if vocab.len() > CAP {
                    return Err(serde::de::Error::custom(
                        "vocabulary exceeds profile capacity",
                    ));
                }
            }
            Ok(vocab)
        }
    }
    d.deserialize_map(Visitor::<CAP>)
}
fn read_merges<'de, D: Deserializer<'de>, const CAP: usize>(
    d: D,
) -> Result<Vec<(String, String)>, D::Error> {
    struct Visitor<const CAP: usize>;
    impl<'de, const CAP: usize> serde::de::Visitor<'de> for Visitor<CAP> {
        type Value = Vec<(String, String)>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("bounded canonical tuple merges")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            let mut merges = Vec::with_capacity(CAP);
            while let Some(pair) = seq.next_element()? {
                if merges.len() == CAP {
                    return Err(serde::de::Error::custom("merges exceed profile capacity"));
                }
                merges.push(pair);
            }
            Ok(merges)
        }
    }
    d.deserialize_seq(Visitor::<CAP>)
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalBpe<const CAP: usize>(BPE);

impl<'de, const CAP: usize> Deserialize<'de> for CanonicalBpe<CAP> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let c = Canonical::<CAP>::deserialize(deserializer)?;
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
impl<const CAP: usize> Serialize for CanonicalBpe<CAP> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}
impl<const CAP: usize> From<CanonicalBpe<CAP>> for ModelWrapper {
    fn from(value: CanonicalBpe<CAP>) -> Self {
        Self::BPE(value.0)
    }
}
impl<const CAP: usize> Model for CanonicalBpe<CAP> {
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
    fn bpe_builder_preserves_unicode_prefix_merges_and_missing_tokens() {
        let vocab: Vocab = [("é", 0), ("##猫", 1), ("é猫", 2)]
            .into_iter()
            .map(|(token, id)| (token.to_owned(), id))
            .collect();
        let merges = vec![("é".to_owned(), "##猫".to_owned())];
        let model = BPE::builder()
            .vocab_and_merges(vocab.clone(), merges.clone())
            .continuing_subword_prefix("##".to_owned())
            .build()
            .unwrap();
        let tokens = model.tokenize("é猫").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!((tokens[0].id, tokens[0].value.as_str()), (2, "é猫"));

        let mut missing = vocab;
        missing.remove("é猫");
        assert!(
            BPE::builder()
                .vocab_and_merges(missing, merges)
                .continuing_subword_prefix("##".to_owned())
                .build()
                .is_err()
        );
    }
    #[test]
    fn profile_capacities_bound_and_preserve_collections() {
        let mut d = serde_json::Deserializer::from_str(r#"{"a":0,"b":1}"#);
        assert_eq!(read_vocab::<_, 2>(&mut d).unwrap().len(), 2);
        let mut d = serde_json::Deserializer::from_str(r#"{"a":0,"b":1}"#);
        assert!(read_vocab::<_, 1>(&mut d).is_err());
        let mut d = serde_json::Deserializer::from_str(r#"[["a","b"],["c","d"]]"#);
        let pairs = read_merges::<_, 2>(&mut d).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs.capacity(), 2);
        let mut d = serde_json::Deserializer::from_str(r#"[["a","b"],["c","d"]]"#);
        assert!(read_merges::<_, 1>(&mut d).is_err());
    }
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
        let parsed: CanonicalBpe<4> = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), json);
        for (key, value) in [
            ("type", serde_json::json!("WordPiece")),
            ("merges", serde_json::json!(["a b"])),
            ("unknown", serde_json::json!(true)),
            ("dropout", serde_json::json!(2.0)),
        ] {
            let mut bad = json.clone();
            bad[key] = value;
            assert!(serde_json::from_value::<CanonicalBpe<4>>(bad).is_err());
        }
    }
}
