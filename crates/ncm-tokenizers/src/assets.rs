//! Offline-only, immutable tokenizer asset loading. No Hub/HTTP or remote code.
use crate::bpe::ByteBpe;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("invalid or unsupported tokenizer: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, AssetError>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerMetadata {
    pub schema_version: u32,
    pub profile: String,
    pub kind: String,
    pub upstream_commit: String,
    pub source: serde_json::Value,
    pub reference_versions: BTreeMap<String, String>,
    pub vocab_size: usize,
    pub total_vocab_size: usize,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub cleanup: bool,
    pub payload_bytes: usize,
    pub payload_sha256: String,
    pub license_sha256: String,
}
enum Implementation {
    ByteBpe(Box<ByteBpe>),
    HuggingFace(Box<tokenizers::Tokenizer>),
}
pub struct TextTokenizer {
    metadata: TokenizerMetadata,
    implementation: Implementation,
}
fn invalid(message: &str) -> AssetError {
    AssetError::Invalid(message.into())
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > limit as u64 {
        return Err(invalid("invalid or oversized asset file"));
    }
    // Preserve the byte bound while avoiding geometric read-buffer reallocations.
    let capacity =
        usize::try_from(meta.len() + 1).map_err(|_| invalid("asset capacity overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("asset allocation failed"))?;
    fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(invalid("asset exceeds byte bound"));
    }
    Ok(bytes)
}
// The authenticated profiles fix the model kind. Deserialize that concrete model
// directly, avoiding ModelWrapper's intermediate generic JSON tree; move the
// complete tokenizer into the same runtime wrapper without cloning its vocabulary.
fn total_vocab_size(tokenizer: &tokenizers::Tokenizer) -> usize {
    use tokenizers::Model;
    let model = tokenizer.get_model();
    // Count the union without get_vocab_size(true)'s full vocabulary clone.
    model.get_vocab_size()
        + tokenizer
            .get_added_vocabulary()
            .get_vocab()
            .keys()
            .filter(|token| model.token_to_id(token).is_none())
            .count()
}
fn parse_hf(payload: &[u8], profile: &str) -> Result<tokenizers::Tokenizer> {
    type Typed<M> = tokenizers::TokenizerImpl<
        M,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >;
    match profile {
        "bert" => Ok(serde_json::from_slice::<
            Typed<crate::hf_wordpiece::CanonicalWordPiece<30522>>,
        >(payload)?
        .into()),
        // Pinned base-vocabulary capacities also bound the tuple-merge counts.
        "gpt2" => Ok(
            serde_json::from_slice::<Typed<crate::hf_bpe::CanonicalBpe<50257>>>(payload)?.into(),
        ),
        "qwen3" => Ok(
            serde_json::from_slice::<Typed<crate::hf_bpe::CanonicalBpe<151643>>>(payload)?.into(),
        ),
        _ => Err(invalid("unsupported HF profile")),
    }
}

impl TextTokenizer {
    /// Only the four reviewed Phase 2 manifests/payloads are supported. Trusted
    /// parent directories/no concurrent mutation are required (not a path sandbox).
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory: PathBuf = directory.as_ref().components().collect();
        let meta = fs::symlink_metadata(&directory)?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err(invalid("tokenizer root must be a real directory"));
        }
        let bytes = read(&directory.join("manifest.json"), 64 * 1024)?;
        let metadata: TokenizerMetadata = serde_json::from_slice(&bytes)?;
        let lock: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tools/reference/tokenizers-v1.lock.json"
        ))?;
        let record = lock["profiles"]
            .get(&metadata.profile)
            .ok_or_else(|| invalid("unknown tokenizer profile"))?;
        if metadata.schema_version != 1
            || record["manifest_sha256"].as_str() != Some(&digest(&bytes))
        {
            return Err(invalid("manifest differs from pinned profile"));
        }
        let payload = read(&directory.join("tokenizer.json"), 32 * 1024 * 1024)?;
        if payload.len() != metadata.payload_bytes
            || digest(&payload) != metadata.payload_sha256
            || record["payload_sha256"].as_str() != Some(&metadata.payload_sha256)
        {
            return Err(invalid("tokenizer payload integrity failure"));
        }
        if digest(&read(&directory.join("LICENSE"), 64 * 1024)?) != metadata.license_sha256 {
            return Err(invalid("license notice differs from pinned asset"));
        }
        let implementation = match metadata.kind.as_str() {
            "byte_bpe" => {
                let bpe = ByteBpe::from_json(&payload)?;
                if bpe.vocab_size() != metadata.total_vocab_size {
                    return Err(invalid("BPE vocabulary differs"));
                }
                Implementation::ByteBpe(Box::new(bpe))
            }
            "huggingface" => {
                let tokenizer = parse_hf(&payload, &metadata.profile)?;
                if tokenizer.get_vocab_size(false) != metadata.vocab_size
                    || total_vocab_size(&tokenizer) != metadata.total_vocab_size
                {
                    return Err(invalid("HF vocabulary differs"));
                }
                Implementation::HuggingFace(Box::new(tokenizer))
            }
            _ => return Err(invalid("unsupported tokenizer kind")),
        };
        Ok(Self {
            metadata,
            implementation,
        })
    }
    pub fn metadata(&self) -> &TokenizerMetadata {
        &self.metadata
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.len() > 1024 * 1024 {
            return Err(invalid("text exceeds 1 MiB request bound"));
        }
        match &self.implementation {
            Implementation::ByteBpe(bpe) => bpe.encode(text),
            // This API returns IDs only. Keep the complete tokenizer pipeline
            // and special tokens, but do not compute offsets that are discarded.
            Implementation::HuggingFace(tokenizer) => Ok(tokenizer
                .encode_fast(text, true)
                .map_err(|e| invalid(&e.to_string()))?
                .get_ids()
                .to_vec()),
        }
    }
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        if ids.len() > 1024 * 1024
            || ids
                .iter()
                .any(|&id| id as usize >= self.metadata.total_vocab_size)
        {
            return Err(invalid("invalid token ID or request size"));
        }
        let decoded = match &self.implementation {
            Implementation::ByteBpe(bpe) => bpe.decode(ids, skip_special_tokens)?,
            Implementation::HuggingFace(tokenizer) => tokenizer
                .decode(ids, skip_special_tokens)
                .map_err(|e| invalid(&e.to_string()))?,
        };
        Ok(if self.metadata.cleanup {
            cleanup(decoded)
        } else {
            decoded
        })
    }
}
fn cleanup(mut text: String) -> String {
    for (from, to) in [
        (" .", "."),
        (" ?", "?"),
        (" !", "!"),
        (" ,", ","),
        (" ' ", "'"),
        (" n't", "n't"),
        (" 'm", "'m"),
        (" 's", "'s"),
        (" 've", "'ve"),
        (" 're", "'re"),
    ] {
        text = text.replace(from, to);
    }
    text
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vocabulary_union_counts_overlapping_added_tokens_without_cloning() {
        let model = tokenizers::models::wordpiece::WordPiece::builder()
            .vocab([("[UNK]".into(), 0), ("a".into(), 1)])
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        for tokens in [vec!["a"], vec!["new", "[UNK]"], vec!["new", "other"]] {
            tokenizer.add_tokens(
                &tokens
                    .into_iter()
                    .map(|t| tokenizers::AddedToken::from(t, false))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(total_vocab_size(&tokenizer), tokenizer.get_vocab_size(true));
        }
    }
    #[test]
    fn bounded_asset_reads_reserve_once() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("payload");
        for length in [0, 1, 32768, 1024 * 1024 + 17] {
            let bytes = vec![42u8; length];
            fs::write(&path, &bytes).unwrap();
            let loaded = read(&path, length).unwrap();
            assert_eq!(loaded, bytes);
            assert_eq!(loaded.capacity(), length + 1);
            if length > 0 {
                assert!(read(&path, length - 1).is_err());
            }
        }
    }
    #[test]
    fn typed_loading_preserves_complete_runtime_configuration() {
        for (profile, model) in [
            (
                "gpt2",
                serde_json::json!({"type":"BPE", "vocab":{"[UNK]":0,"a":1,"b":2}, "merges":[], "unk_token":"[UNK]"}),
            ),
            (
                "bert",
                serde_json::json!({"type":"WordPiece", "vocab":{"[UNK]":0,"a":1,"b":2}, "unk_token":"[UNK]", "continuing_subword_prefix":"##", "max_input_chars_per_word":100}),
            ),
        ] {
            let bytes = serde_json::to_vec(&serde_json::json!({"version":"1.0", "model":model,
                "added_tokens":[{"id":3,"content":"[PAD]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
                "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"padding":null,"truncation":null})).unwrap();
            let original = tokenizers::Tokenizer::from_bytes(&bytes).unwrap();
            let typed = parse_hf(&bytes, profile).unwrap();
            assert_eq!(
                serde_json::to_value(&original).unwrap(),
                serde_json::to_value(&typed).unwrap()
            );
            for text in ["", "ab", "a", "[PAD]", "é"] {
                assert_eq!(
                    original.encode(text, true).unwrap().get_ids(),
                    typed.encode(text, true).unwrap().get_ids()
                );
            }
            for skip in [false, true] {
                assert_eq!(
                    original.decode(&[0, 1, 3], skip).unwrap(),
                    typed.decode(&[0, 1, 3], skip).unwrap()
                );
            }
            assert!(parse_hf(&bytes, "unknown").is_err());
        }
    }
    #[test]
    fn id_only_encoding_preserves_normalization_specials_padding_and_truncation() {
        use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, TruncationParams};
        let words = [
            "[UNK]", "a", "b", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "中", "##b", "c",
        ];
        let vocab: [(String, u32); 10] = std::array::from_fn(|i| (words[i].to_string(), i as u32));
        let model = tokenizers::models::wordpiece::WordPiece::builder()
            .vocab(vocab)
            .build()
            .unwrap();
        let mut reference = tokenizers::Tokenizer::new(model);
        reference.with_normalizer(Some(
            tokenizers::normalizers::bert::BertNormalizer::default(),
        ));
        reference.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::bert::BertPreTokenizer));
        reference.add_special_tokens(&[AddedToken::from("[MASK]", true).lstrip(true).rstrip(true)]);
        reference.with_post_processor(Some(
            tokenizers::processors::template::TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 4), ("[SEP]", 5)])
                .build()
                .unwrap(),
        ));
        reference
            .with_truncation(Some(TruncationParams {
                max_length: 6,
                ..Default::default()
            }))
            .unwrap();
        reference.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            pad_id: 3,
            ..Default::default()
        }));
        // Private synthetic construction exercises wrapper behavior, not asset trust.
        let tokenizer = TextTokenizer {
            metadata: serde_json::from_value(serde_json::json!({
                "schema_version":1,"profile":"bert","kind":"huggingface","upstream_commit":"synthetic",
                "source":{},"reference_versions":{},"vocab_size":10,"total_vocab_size":10,
                "eos_token_id":5,"pad_token_id":3,"cleanup":false,"payload_bytes":0,
                "payload_sha256":"","license_sha256":""
            })).unwrap(),
            implementation: Implementation::HuggingFace(Box::new(reference.clone())),
        };
        for text in [
            "",
            "A B",
            "Á a\u{301}",
            "\t中 [MASK] AB\n",
            "a b c a b c a b c",
            "é💡",
        ] {
            let expected = reference.encode(text, true).unwrap();
            let actual = tokenizer.encode(text).unwrap();
            assert_eq!(actual, expected.get_ids());
            assert_eq!(actual.len(), 8);
            assert_eq!(actual[0], 4);
            assert_eq!(actual[7], 3);
        }
        assert!(tokenizer.encode(&"a".repeat(1024 * 1024 + 1)).is_err());
    }
    #[test]
    fn cleanup_matches_transformers_order() {
        assert_eq!(cleanup("I 'm sure ! do n't .".into()), "I'm sure! don't.");
    }
    #[test]
    fn malformed_unknown_and_tampered_manifests_fail() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("manifest.json"), b"{}").unwrap();
        assert!(TextTokenizer::load(root.path()).is_err());
        fs::write(root.path().join("manifest.json"), vec![b' '; 65537]).unwrap();
        assert!(TextTokenizer::load(root.path()).is_err());
        let mut valid_shape = serde_json::json!({"schema_version":1,"profile":"unknown","kind":"byte_bpe",
            "upstream_commit":"untrusted","source":{},"reference_versions":{},"vocab_size":258,"total_vocab_size":258,
            "eos_token_id":0,"pad_token_id":1,"cleanup":false,"payload_bytes":0,"payload_sha256":"bad","license_sha256":"bad"});
        fs::write(
            root.path().join("manifest.json"),
            serde_json::to_vec(&valid_shape).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            TextTokenizer::load(root.path()),
            Err(AssetError::Invalid(_))
        ));
        valid_shape["profile"] = "tinystories".into();
        fs::write(
            root.path().join("manifest.json"),
            serde_json::to_vec(&valid_shape).unwrap(),
        )
        .unwrap();
        assert!(
            matches!(TextTokenizer::load(root.path()),Err(AssetError::Invalid(message)) if message=="manifest differs from pinned profile")
        );
        valid_shape["unknown_option"] = true.into();
        fs::write(
            root.path().join("manifest.json"),
            serde_json::to_vec(&valid_shape).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            TextTokenizer::load(root.path()),
            Err(AssetError::Json(_))
        ));
    }
    #[cfg(unix)]
    #[test]
    fn symlink_root_even_with_trailing_separator_is_rejected() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("root");
        fs::create_dir(&root).unwrap();
        let link = parent.path().join("link");
        std::os::unix::fs::symlink(&root, &link).unwrap();
        for suffix in ["", "/", "/."] {
            assert!(
                matches!(TextTokenizer::load(format!("{}{suffix}",link.display())),Err(AssetError::Invalid(message)) if message=="tokenizer root must be a real directory")
            );
        }
        let target = parent.path().join("metadata");
        fs::write(&target, b"{}").unwrap();
        std::os::unix::fs::symlink(target, root.join("manifest.json")).unwrap();
        assert!(
            matches!(TextTokenizer::load(&root),Err(AssetError::Invalid(message)) if message=="invalid or oversized asset file")
        );
    }
}
