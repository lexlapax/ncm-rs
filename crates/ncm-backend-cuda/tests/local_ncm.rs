//! End-to-end CPU integration tests for config-driven `local-ncm-v1` artifacts and `LocalNcmModel`.
#![cfg(feature = "cpu")]

use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, LocalNcmConfig};
use ncm_io::{Artifact, FileRecord, Manifest, TensorDtype, TensorRecord, sha256};
use ncm_model::LocalNcmModel;
use safetensors::tensor::{Dtype, TensorView};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct TempDir(PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("ncm-local-test-{tag}-{}-{id}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn deterministic_f32(name: &str, count: usize) -> Vec<u8> {
    let mut seed = 0x517cc1b727220a95u64;
    for b in name.bytes() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(b as u64 + 1);
    }
    let mut out = Vec::with_capacity(count * 4);
    let is_norm = name.contains("ln_") || name.contains("q_norm") || name.contains("k_norm");
    let is_bias = name.ends_with(".bias") || name.starts_with("head_biases.");
    for i in 0..count {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407 ^ (i as u64));
        let u = (((seed >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0;
        let val = if is_norm && !is_bias {
            1.0 + 0.05 * u
        } else if is_bias {
            0.05 * u
        } else {
            0.15 * u
        };
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

fn serialize_bundle(shapes: &BTreeMap<String, Vec<usize>>) -> (Vec<u8>, Vec<TensorRecord>) {
    let mut raw: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(shapes.len());
    let mut records = Vec::with_capacity(shapes.len());
    for (name, shape) in shapes {
        let elems: usize = shape.iter().product();
        let bytes = deterministic_f32(name, elems);
        records.push(TensorRecord {
            name: name.clone(),
            shape: shape.clone(),
            dtype: TensorDtype::F32,
            bytes: bytes.len(),
            sha256: sha256(&bytes),
        });
        raw.push((name.clone(), shape.clone(), bytes));
    }
    let views: Vec<(&str, TensorView<'_>)> = raw
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    (serialized, records)
}

fn write_local_artifact(
    dir: &std::path::Path,
    head_kind: &str,
    pad: Option<u32>,
    vocab_sizes: &str,
) {
    let pad_line = match pad {
        Some(id) => format!(r#""pad_token_id": {id},"#),
        None => String::new(),
    };
    let config_str = format!(
        r#"{{
          "tokenizer": {{
            "checkpoint_path": "local-codec.pt",
            "in_vocab_size": 32,
            "context_length": 8,
            "embed_dim": 16,
            "vqvae_vocab_size": {vocab_sizes},
            "n_layers": 2,
            "compression_factor": 1,
            "dropout": 0.0,
            "pre_quant_dropout": 0.0,
            "pre_quant_groupnorm": 4,
            "pre_quant_norm_type": "group",
            "use_qk_norm": true,
            "dec_layers": 2,
            "enc_positional": true,
            "dec_positional": false,
            "vector_quantizer_config": {{
              "clss": "multiscale_residual_vector_quantizer",
              "commitment_cost": 0.25,
              "decay": 0.99,
              "epsilon": 1e-5,
              "learned_all_sampling": false,
              "learned_l1_sampling": true,
              "learned_l1_cascade": true,
              "learned_l1_cascade_base": 2,
              "learned_l1_cascade_norm": true,
              "learned_l1_cascade_norm_final": false,
              "levels": {{
                "use_manual_levels": true,
                "manual_levels": [2, 4, 8]
              }},
              "quant_resi": {{
                "enabled": true,
                "learnable_ratio": false,
                "ratio": 0.5,
                "share_mode": 0
              }},
              "aux": {{
                "fine_drop": {{
                  "min_keep": 1,
                  "prob": 0.0
                }}
              }}
            }}
          }},
          "model": {{
            "attn_pattern": "block_diagonal",
            "bias": true,
            "context_length": 24,
            "dropout": 0.0,
            "n_embd": 32,
            "n_head": 4,
            "n_layer": 2,
            "prefix_len": 6,
            "query_level0": true,
            "raw_token_prefix": true,
            "rope_base": 10000,
            "tie_heads_to_codebook": false,
            "use_level_encoding": true,
            "use_positional_encoding": true,
            "use_qk_norm": true,
            "use_rope": true,
            "vocab_size": {vocab_sizes},
            "head_kind": "{head_kind}",
            {pad_line}
            "eos_token_id": 0
          }}
        }}"#
    );
    let cfg = LocalNcmConfig::from_json(config_str.as_bytes()).unwrap();
    let (codec_bytes, codec_records) = serialize_bundle(&cfg.codec().weight_shapes());
    let (gen_bytes, gen_records) = serialize_bundle(&cfg.model().weight_shapes(cfg.codec()));
    let prov_bytes = br#"{"source":"synthetic-unit-test"}"#.to_vec();

    let mut files = BTreeMap::new();
    for (name, data) in [
        ("config.json", config_str.as_bytes()),
        ("provenance.json", prov_bytes.as_slice()),
        ("codec.safetensors", codec_bytes.as_slice()),
        ("generator.safetensors", gen_bytes.as_slice()),
    ] {
        fs::write(dir.join(name), data).unwrap();
        files.insert(
            name.to_owned(),
            FileRecord {
                bytes: data.len() as u64,
                sha256: sha256(data),
            },
        );
    }
    let mut tensors = BTreeMap::new();
    tensors.insert("codec.safetensors".to_owned(), codec_records);
    tensors.insert("generator.safetensors".to_owned(), gen_records);
    let manifest = Manifest {
        schema_version: 1,
        profile: "local-ncm-v1".into(),
        upstream_commit: "local-test".into(),
        checkpoint_repository: "local/synthetic".into(),
        checkpoint_revision: "v1".into(),
        files,
        tensors,
    };
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn local_ncm_end_to_end_cache_equivalence_and_head_kinds() {
    let cases = [
        ("untied_affine", Some(1), "[12, 16, 20]"),
        ("untied_linear", None, "[12, 16, 20]"),
        ("shared_head_biases", Some(1), "[16, 16, 16]"),
    ];
    for (head_kind, pad, vocab_sizes) in cases {
        let dir = TempDir::new(head_kind);
        write_local_artifact(&dir.0, head_kind, pad, vocab_sizes);
        let backend = CandleBackend::cpu();

        // Both in-memory and streaming loaders must succeed and match identically.
        let art_mem = Artifact::load_inference(&dir.0).unwrap();
        let model_cached = LocalNcmModel::from_artifact(backend.clone(), &art_mem)
            .unwrap()
            .with_prefix_cache(true);
        let art_stream = Artifact::load_inference_streaming(&dir.0).unwrap();
        let model_uncached = LocalNcmModel::from_artifact(backend.clone(), &art_stream)
            .unwrap()
            .with_prefix_cache(false);

        let prefixes: Vec<u32> = if pad.is_some() {
            // Row 0 has trailing pad=1; Row 1 has no pad.
            vec![3, 7, 11, 1, 1, 1, 4, 8, 12, 16, 20, 24]
        } else {
            vec![3, 7, 11, 2, 5, 9, 4, 8, 12, 16, 20, 24]
        };

        // Greedy cache-on vs cache-off must match bit-for-bit on codes and IDs.
        let out_cached = model_cached.generate(&prefixes, 2, None).unwrap();
        let out_uncached = model_uncached.generate(&prefixes, 2, None).unwrap();
        assert_eq!(
            backend.read_indices(&out_cached.codes).unwrap(),
            backend.read_indices(&out_uncached.codes).unwrap(),
            "greedy codes mismatch for {head_kind}"
        );
        assert_eq!(
            backend.read_indices(&out_cached.ids).unwrap(),
            backend.read_indices(&out_uncached.ids).unwrap(),
            "greedy ids mismatch for {head_kind}"
        );

        // Seeded sampling must be deterministic and match cache-on vs cache-off.
        let seeded_cached = model_cached.generate(&prefixes, 2, Some(42)).unwrap();
        let seeded_uncached = model_uncached.generate(&prefixes, 2, Some(42)).unwrap();
        assert_eq!(
            backend.read_indices(&seeded_cached.codes).unwrap(),
            backend.read_indices(&seeded_uncached.codes).unwrap(),
            "seeded codes mismatch for {head_kind}"
        );
        assert_eq!(
            backend.read_indices(&seeded_cached.ids).unwrap(),
            backend.read_indices(&seeded_uncached.ids).unwrap(),
            "seeded ids mismatch for {head_kind}"
        );

        // Different seed produces a valid output of the same shape.
        let seeded_other = model_cached.generate(&prefixes, 2, Some(99)).unwrap();
        assert_eq!(backend.read_indices(&seeded_other.ids).unwrap().len(), 16);

        // All-pad row must fail closed when pad_token_id is configured.
        if let Some(p) = pad {
            let all_pad = vec![p; 6];
            assert!(model_cached.generate(&all_pad, 1, None).is_err());
        }
    }
}
