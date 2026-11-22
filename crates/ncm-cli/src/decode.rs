//! Evaluator-compatible decoding of generated text IDs.
use crate::Result;
use ncm_core::LocalNcmConfig;
use ncm_io::Artifact;
use ncm_tokenizers::{LocalTokenizer, TextTokenizer};

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

struct LocalPolicy {
    eos: Option<u32>,
    pad: Option<u32>,
    model_vocab: u32,
    prefix_len: usize,
}

enum DecoderInner {
    Pinned {
        tokenizer: Box<TextTokenizer>,
        policy: Policy,
    },
    Local {
        tokenizer: LocalTokenizer,
        policy: LocalPolicy,
    },
}

pub(super) struct Decoder(DecoderInner);

impl Decoder {
    pub(super) fn from_artifact(artifact: &Artifact) -> Result<Self> {
        if artifact.manifest().profile == "local-ncm-v1" {
            let cfg = LocalNcmConfig::from_json(artifact.file("config.json")?)?;
            let tokenizer_bytes = artifact.file("tokenizer.json").map_err(
                |_| "local-ncm-v1 --text/--decode requires tokenizer.json in the artifact",
            )?;
            let tokenizer = LocalTokenizer::from_json_bytes(tokenizer_bytes)?;
            let model_vocab = u32::try_from(cfg.codec().raw().in_vocab_size)
                .map_err(|_| "local-ncm-v1 vocabulary exceeds u32")?;
            if tokenizer.vocab_size() > cfg.codec().raw().in_vocab_size {
                return Err("local-ncm-v1 tokenizer vocabulary exceeds codec in_vocab_size".into());
            }
            return Ok(Self(DecoderInner::Local {
                tokenizer,
                policy: LocalPolicy {
                    eos: cfg.model().eos_token_id,
                    pad: cfg.model().pad_token_id,
                    model_vocab,
                    prefix_len: cfg.model().prefix_len,
                },
            }));
        }
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
        Ok(Self(DecoderInner::Pinned {
            tokenizer: Box::new(tokenizer),
            policy,
        }))
    }

    pub(super) fn encode_texts(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        if texts.is_empty() || texts.len() > 128 {
            return Err("text input must contain 1..=128 strings".into());
        }
        match &self.0 {
            DecoderInner::Pinned { tokenizer, policy } => texts
                .iter()
                .map(|text| {
                    let mut ids = tokenizer.encode(text)?;
                    if let Some(length) = policy.input_length {
                        ids.truncate(length);
                        ids.resize(length, policy.pad);
                    } else {
                        validate_tinystories_prefix(&ids)?;
                    }
                    Ok(ids)
                })
                .collect(),
            DecoderInner::Local { tokenizer, policy } => texts
                .iter()
                .map(|text| {
                    let mut ids = tokenizer.encode(text)?;
                    if ids.is_empty() {
                        return Err("local-ncm-v1 text input encoded to zero tokens".into());
                    }
                    if let Some(pad) = policy.pad {
                        ids.truncate(policy.prefix_len);
                        ids.resize(policy.prefix_len, pad);
                        if ids.iter().all(|&id| id == pad) {
                            return Err("local-ncm-v1 text input encoded to all pad tokens".into());
                        }
                    } else if ids.len() != policy.prefix_len {
                        return Err(format!(
                            "local-ncm-v1 text input encoded to {} tokens, expected {} (no pad_token_id configured)",
                            ids.len(),
                            policy.prefix_len
                        )
                        .into());
                    }
                    Ok(ids)
                })
                .collect(),
        }
    }

    pub(super) fn add_texts(&self, result: &mut serde_json::Value) -> Result<()> {
        let ids: Vec<Vec<u32>> =
            serde_json::from_value(result.get("ids").ok_or("missing generated IDs")?.clone())?;
        let texts: Vec<String> = match &self.0 {
            DecoderInner::Pinned { tokenizer, policy } => ids
                .iter()
                .map(|row| {
                    let selected =
                        evaluator_ids(row, policy, tokenizer.metadata().total_vocab_size)?;
                    Ok(tokenizer.decode(&selected, true)?)
                })
                .collect::<Result<_>>()?,
            DecoderInner::Local { tokenizer, policy } => ids
                .iter()
                .map(|row| {
                    if row.iter().any(|&id| id >= policy.model_vocab) {
                        return Err("generated ID exceeds model vocabulary".into());
                    }
                    let end = policy
                        .eos
                        .and_then(|eos| row.iter().position(|&id| id == eos))
                        .unwrap_or(row.len());
                    let selected: Vec<u32> = row[..end]
                        .iter()
                        .copied()
                        .filter(|&id| {
                            Some(id) != policy.pad && (id as usize) < tokenizer.vocab_size()
                        })
                        .collect();
                    Ok(tokenizer.decode(&selected, true)?)
                })
                .collect::<Result<_>>()?,
        };
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
    #[ignore = "requires pinned TinyStories artifact and generated upstream text reference"]
    fn tinystories_single_token_decode_matches_upstream_vocabulary() {
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
        let ids: Vec<Vec<u32>> = (0..8192).map(|id| vec![id]).collect();
        let mut result = serde_json::json!({"ids": ids});
        decoder.add_texts(&mut result).unwrap();
        let actual: Vec<String> = serde_json::from_value(result["texts"].clone()).unwrap();
        assert_eq!(actual.len(), 8192);
        for (id, text) in actual.iter().enumerate() {
            let expected = if id < 2 {
                "" // EOS stops generation; padding is removed.
            } else {
                rows[id]["text"].as_str().unwrap()
            };
            assert_eq!(text, expected, "token {id}");
        }
    }

    #[test]
    #[ignore = "requires pinned TinyStories artifact and synthetic decode reference"]
    fn tinystories_multi_token_decode_matches_upstream() {
        let artifact =
            ncm_io::Artifact::load_inference(std::env::var("NCM_TINYSTORIES_ARTIFACT").unwrap())
                .unwrap();
        let decoder = Decoder::from_artifact(&artifact).unwrap();
        drop(artifact);
        let raw =
            std::fs::read(std::env::var("NCM_TINYSTORIES_DECODE_REFERENCE").unwrap()).unwrap();
        assert_eq!(
            ncm_io::sha256(&raw),
            "0c2361bece994e027a1db7a9e93b856cf27cf8aff9c1b1194d39f979623f737b"
        );
        let data: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let ids: Vec<Vec<u32>> = serde_json::from_value(data["ids"].clone()).unwrap();
        let texts: Vec<String> = serde_json::from_value(data["texts"].clone()).unwrap();
        assert_eq!(ids.len(), 137);
        assert_eq!(texts.len(), ids.len());
        let mut result = serde_json::json!({"ids": ids});
        decoder.add_texts(&mut result).unwrap();
        assert_eq!(result["texts"], serde_json::json!(texts));
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
    #[ignore = "requires pinned XSum/WMT14 artifacts and generated boundary references"]
    fn synthetic_text_boundaries_match_pinned_python() {
        for (artifact_var, reference_var, profile, digest, length) in [
            (
                "NCM_XSUM_ARTIFACT",
                "NCM_XSUM_BOUNDARY_REFERENCE",
                "xsum",
                "2f7c8ed36cfd669a59ecf2fc5e13fe2fb584941c61c93e069a0825a7c5ff46f6",
                512,
            ),
            (
                "NCM_WMT14_ARTIFACT",
                "NCM_WMT14_BOUNDARY_REFERENCE",
                "wmt14",
                "03886e0864b60f9fc7992311f83b3e20511b9ce73274fef5f51624d41d960fb6",
                96,
            ),
        ] {
            let artifact =
                ncm_io::Artifact::load_inference(std::env::var(artifact_var).unwrap()).unwrap();
            let decoder = Decoder::from_artifact(&artifact).unwrap();
            drop(artifact);
            let raw = std::fs::read(std::env::var(reference_var).unwrap()).unwrap();
            assert_eq!(ncm_io::sha256(&raw), digest);
            let data: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(data["profile"], profile);
            let texts: Vec<String> = serde_json::from_value(data["texts"].clone()).unwrap();
            let ids: Vec<Vec<u32>> = serde_json::from_value(data["ids"].clone()).unwrap();
            let lengths: Vec<usize> =
                serde_json::from_value(data["original_lengths"].clone()).unwrap();
            assert_eq!(texts.len(), 7);
            assert_eq!(ids.len(), texts.len());
            assert_eq!(lengths.len(), texts.len());
            assert!(lengths.iter().any(|&n| n > length));
            assert!(lengths.iter().any(|&n| n < length));
            assert!(ids.iter().all(|row| row.len() == length));
            assert_eq!(decoder.encode_texts(&texts).unwrap(), ids, "{profile}");
        }
    }

    #[test]
    #[ignore = "requires pinned XSum/WMT14 artifacts and generated decode references"]
    fn synthetic_decode_boundaries_match_pinned_python() {
        for (artifact_var, reference_var, profile, digest) in [
            (
                "NCM_XSUM_ARTIFACT",
                "NCM_XSUM_DECODE_REFERENCE",
                "xsum",
                "ce931c2380ee57da00aa84ed818b005ed27bd5c81e3265766b075f126cde377c",
            ),
            (
                "NCM_WMT14_ARTIFACT",
                "NCM_WMT14_DECODE_REFERENCE",
                "wmt14",
                "16bbc0f63d8fe2f28813b12554375e4a54d251cee420ac2602c450878bd0dc9c",
            ),
        ] {
            let artifact =
                ncm_io::Artifact::load_inference(std::env::var(artifact_var).unwrap()).unwrap();
            let decoder = Decoder::from_artifact(&artifact).unwrap();
            drop(artifact);
            let raw = std::fs::read(std::env::var(reference_var).unwrap()).unwrap();
            assert_eq!(ncm_io::sha256(&raw), digest);
            let data: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(data["profile"], profile);
            let ids: Vec<Vec<u32>> = serde_json::from_value(data["ids"].clone()).unwrap();
            let texts: Vec<String> = serde_json::from_value(data["texts"].clone()).unwrap();
            assert_eq!(ids.len(), 135);
            assert_eq!(texts.len(), ids.len());
            let mut result = serde_json::json!({"ids": ids});
            decoder.add_texts(&mut result).unwrap();
            assert_eq!(result["texts"], serde_json::json!(texts), "{profile}");
        }
    }

    #[test]
    #[ignore = "diagnostic timing requires pinned XSum/WMT14 artifacts and retained references"]
    fn text_tokenizer_stage_diagnostic() {
        let name = std::env::var("NCM_TEXT_TIMING_PROFILE").unwrap();
        let (manifest_sha, reference_sha, output_sha) = match name.as_str() {
            "xsum" => (
                "8b850bdaa5ee7ada43bfc4105e2ac99e2edc3d7e114a76567ce03b87f0b9eee4",
                "f51a1232bcd47169cd970edd61b597b5aebef12f0f202e6d51f41be6642faa2f",
                "6db35c54fece2e27ea0e96e6f4e29cebba23a42bd20e899401218658cd2a0328",
            ),
            "wmt14" => (
                "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81",
                "4c1d6b68e89b46800219e9099116b1a50d51b2c96a51bdc8abe6dec6a98add9e",
                "fd68c625bc98b4e7fccfde240db198c67ae8251e8607110b5319e8a617098d32",
            ),
            _ => panic!("unsupported diagnostic profile"),
        };
        let artifact_path =
            std::path::PathBuf::from(std::env::var("NCM_TEXT_TIMING_ARTIFACT").unwrap());
        let reference_path =
            std::path::PathBuf::from(std::env::var("NCM_TEXT_TIMING_REFERENCE").unwrap());
        let output_path =
            std::path::PathBuf::from(std::env::var("NCM_TEXT_TIMING_OUTPUT").unwrap());
        let manifest = std::fs::read(artifact_path.join("manifest.json")).unwrap();
        let reference = std::fs::read(&reference_path).unwrap();
        let output = std::fs::read(&output_path).unwrap();
        assert_eq!(ncm_io::sha256(&manifest), manifest_sha);
        assert_eq!(ncm_io::sha256(&reference), reference_sha);
        assert_eq!(ncm_io::sha256(&output), output_sha);
        let reference: serde_json::Value = serde_json::from_slice(&reference).unwrap();
        let output: serde_json::Value = serde_json::from_slice(&output).unwrap();
        let text = reference["texts"][0].as_str().unwrap().to_owned();
        let expected_ids: Vec<u32> = serde_json::from_value(reference["ids"][0].clone()).unwrap();
        let expected_texts = output["texts"].clone();
        let artifact = ncm_io::Artifact::load_inference(&artifact_path).unwrap();
        let mut construct = Vec::new();
        let mut encode = Vec::new();
        let mut decode = Vec::new();
        for round in 0..35 {
            let start = std::time::Instant::now();
            let decoder = Decoder::from_artifact(&artifact).unwrap();
            let construct_elapsed = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            let ids = decoder.encode_texts(std::slice::from_ref(&text)).unwrap();
            let encode_elapsed = start.elapsed().as_secs_f64();
            assert_eq!(ids.as_slice(), std::slice::from_ref(&expected_ids));
            let mut generated = serde_json::json!({"ids": output["ids"]});
            let start = std::time::Instant::now();
            decoder.add_texts(&mut generated).unwrap();
            let decode_elapsed = start.elapsed().as_secs_f64();
            assert_eq!(generated["texts"], expected_texts);
            if round >= 5 {
                construct.push(construct_elapsed);
                encode.push(encode_elapsed);
                decode.push(decode_elapsed);
            }
        }
        assert_eq!(
            ncm_io::sha256(&std::fs::read(artifact_path.join("manifest.json")).unwrap()),
            manifest_sha
        );
        assert_eq!(
            ncm_io::sha256(&std::fs::read(reference_path).unwrap()),
            reference_sha
        );
        assert_eq!(
            ncm_io::sha256(&std::fs::read(output_path).unwrap()),
            output_sha
        );
        println!(
            "TEXT_STAGE_TIMINGS {}",
            serde_json::json!({"profile": name, "construct_seconds": construct,
            "encode_seconds": encode, "decode_seconds": decode, "warmups": 5, "trials": 30})
        );
    }

    #[test]
    #[ignore = "requires retained TinyStories CUDA captures and pinned artifact"]
    fn tinystories_retained_corpus_decode_matches_pinned_evaluator() {
        let artifact =
            ncm_io::Artifact::load_inference(std::env::var("NCM_TINYSTORIES_ARTIFACT").unwrap())
                .unwrap();
        let decoder = Decoder::from_artifact(&artifact).unwrap();
        drop(artifact);
        let directory =
            std::path::PathBuf::from(std::env::var("NCM_TINYSTORIES_CAPTURES").unwrap());
        for (seed, hash) in [
            (
                0,
                "7130d8f638c6c60c08eb4a6e92a3aa6029576b121243b00434c583cdcb0858ea",
            ),
            (
                1,
                "c50b33df058e9d9d95d884ffcae61593f17b712c3df3f48852fdfd7d6f7cdd5c",
            ),
            (
                2,
                "337709f40d8174f247c8d30c2b62364877ca369e116ca352dc10150343f7cd00",
            ),
            (
                3,
                "59d3c85c90396bb15a4bb9276fe97180a9d33caedda21fab31ecf8d8c33320ae",
            ),
            (
                42,
                "78cb02830fedcd6725bc22213534582b646c3b94a8fe97457bb16c339927774b",
            ),
        ] {
            let raw = std::fs::read(directory.join(format!("rust-{seed}.json"))).unwrap();
            assert_eq!(ncm_io::sha256(&raw), hash);
            let capture: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(capture["seed"], seed);
            let ids: Vec<Vec<u32>> = serde_json::from_value(capture["raw_ids"].clone()).unwrap();
            let texts: Vec<String> = serde_json::from_value(capture["texts"].clone()).unwrap();
            assert_eq!(ids.len(), 256);
            assert_eq!(texts.len(), ids.len());
            assert!(ids.iter().all(|row| row.len() == 128));
            let mut result = serde_json::json!({"ids": ids});
            decoder.add_texts(&mut result).unwrap();
            assert_eq!(result["texts"], serde_json::json!(texts), "seed {seed}");
        }
    }

    #[test]
    #[ignore = "requires retained XSum CUDA scorer output and pinned artifact"]
    fn xsum_retained_corpus_decode_matches_pinned_evaluator() {
        let artifact =
            ncm_io::Artifact::load_inference(std::env::var("NCM_XSUM_ARTIFACT").unwrap()).unwrap();
        let decoder = Decoder::from_artifact(&artifact).unwrap();
        drop(artifact);
        let reference = std::fs::read(std::env::var("NCM_XSUM_SCORED_CAPTURE").unwrap()).unwrap();
        assert_eq!(
            ncm_io::sha256(&reference),
            "47302a63f8d014ae64eb7d6293ff3cbd0b41f4e94f702a42c6f870b4a93c6ed3"
        );
        let reference: serde_json::Value = serde_json::from_slice(&reference).unwrap();
        assert_eq!(reference["start"], 0);
        assert_eq!(reference["stop"], 11334);
        let rows = reference["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 11334);
        for (batch, rows) in rows.chunks(128).enumerate() {
            let mut ids = Vec::with_capacity(rows.len());
            let mut expected = Vec::with_capacity(rows.len());
            for (offset, row) in rows.iter().enumerate() {
                assert_eq!(row["index"], batch * 128 + offset);
                let raw_ids: Vec<u32> = serde_json::from_value(row["raw_ids"].clone()).unwrap();
                assert_eq!(raw_ids.len(), 48);
                ids.push(raw_ids);
                expected.push(row["hypothesis"].as_str().unwrap());
            }
            let mut result = serde_json::json!({"ids": ids});
            decoder.add_texts(&mut result).unwrap();
            assert_eq!(
                result["texts"],
                serde_json::json!(expected),
                "batch {batch}"
            );
        }
    }

    fn check_wmt14_retained_corpus_decode(capture_hash: &str, reference_hash: &str) {
        let artifact =
            ncm_io::Artifact::load_inference(std::env::var("NCM_WMT14_ARTIFACT").unwrap()).unwrap();
        let decoder = Decoder::from_artifact(&artifact).unwrap();
        drop(artifact);
        let capture = std::fs::read(std::env::var("NCM_WMT14_CAPTURE").unwrap()).unwrap();
        let reference = std::fs::read(std::env::var("NCM_WMT14_DECODED_CAPTURE").unwrap()).unwrap();
        assert_eq!(ncm_io::sha256(&capture), capture_hash);
        assert_eq!(ncm_io::sha256(&reference), reference_hash);
        let capture: serde_json::Value = serde_json::from_slice(&capture).unwrap();
        let reference: serde_json::Value = serde_json::from_slice(&reference).unwrap();
        assert_eq!(reference["capture_sha256"], capture_hash);
        assert_eq!(capture["count"], 3003);
        assert_eq!(reference["count"], 3003);
        let rows = capture["outputs"].as_array().unwrap();
        let expected = reference["hypotheses"].as_array().unwrap();
        assert_eq!(rows.len(), 3003);
        assert_eq!(expected.len(), rows.len());
        for (batch, (rows, texts)) in rows.chunks(128).zip(expected.chunks(128)).enumerate() {
            let ids: Vec<Vec<u32>> = rows
                .iter()
                .enumerate()
                .map(|(offset, row)| {
                    assert_eq!(row["row"], batch * 128 + offset);
                    let ids: Vec<u32> = serde_json::from_value(row["raw_ids"].clone()).unwrap();
                    assert_eq!(ids.len(), 64);
                    ids
                })
                .collect();
            let mut result = serde_json::json!({"ids": ids});
            decoder.add_texts(&mut result).unwrap();
            assert_eq!(result["texts"], serde_json::json!(texts), "batch {batch}");
        }
    }

    #[test]
    #[ignore = "requires retained WMT14 CUDA capture and pinned evaluator decoding"]
    fn wmt14_retained_corpus_decode_matches_pinned_evaluator() {
        check_wmt14_retained_corpus_decode(
            "d26b8a3b16a4ad07565f52d3e61f22b30e2512ea10348e674a24e9ed52fecc79",
            "0daf73dd5844b1dd832a9b9192e231d149c1be815b40cb833f55f4e81235c624",
        );
    }

    #[test]
    #[ignore = "requires retained WMT14 CFG=1 CUDA capture and pinned evaluator decoding"]
    fn wmt14_cfg1_retained_corpus_decode_matches_pinned_evaluator() {
        check_wmt14_retained_corpus_decode(
            "ff04571bfc72be02aee45ef7f64b6b4226431dca2264a7fd02d30be44294ef0f",
            "43392bd5c8011956f514a5f6bf3f8090d72a09dd3cd08b3027fc02d5c3e3b0d2",
        );
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
