//! Explicit experimental LM1B inference (small, medium or large, from the artifact) on CUDA or Metal.
use super::Result;

#[cfg(any(
    test,
    all(feature = "lm1b-experimental", target_os = "linux"),
    all(
        feature = "lm1b-mlx-experimental",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
fn text_prefixes(bytes: &[u8]) -> Result<Vec<Vec<u32>>> {
    let texts: Vec<String> = serde_json::from_slice(bytes)?;
    if !(1..=32).contains(&texts.len()) || texts.iter().any(|text| !text.is_empty()) {
        return Err(
            "LM1B text input requires 1..=32 empty strings; prompts are unsupported".into(),
        );
    }
    Ok(vec![vec![101]; texts.len()])
}

pub(super) fn sample(
    args: &[String],
    text: bool,
    precision: ncm_core::ComputePrecision,
) -> Result<serde_json::Value> {
    const USAGE: &str = "usage: ncm sample-lm1b ARTIFACT cuda-cudnn|mlx INPUT_JSON SEED REFERENCE_ROPE [--decode] [--top-p-fp32|--host-fp64]";
    let [
        _,
        _,
        artifact,
        device,
        input,
        seed,
        reference_rope,
        options @ ..,
    ] = args
    else {
        return Err(USAGE.into());
    };
    let (mut decode, mut top_p, mut host_fp64) = (false, false, false);
    for flag in options {
        match flag.as_str() {
            "--decode" if !decode => decode = true,
            "--top-p-fp32" if !top_p && !host_fp64 => top_p = true,
            "--host-fp64" if !top_p && !host_fp64 => host_fp64 = true,
            _ => return Err(USAGE.into()),
        }
    }
    if (top_p || host_fp64) && device != "mlx" {
        return Err("--top-p-fp32 and --host-fp64 require explicit mlx device".into());
    }
    match device.as_str() {
        "cuda-cudnn" => {
            #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
            {
                backend::sample_cuda(
                    artifact,
                    (input, text),
                    seed.parse()?,
                    reference_rope,
                    decode,
                    precision,
                )
            }
            #[cfg(not(all(feature = "lm1b-experimental", target_os = "linux")))]
            {
                let _ = (
                    artifact,
                    input,
                    seed,
                    reference_rope,
                    decode,
                    text,
                    precision,
                );
                Err("experimental LM1B-small CUDA is unavailable in this build".into())
            }
        }
        "mlx" => {
            #[cfg(all(
                feature = "lm1b-mlx-experimental",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            {
                backend::sample_mlx(
                    artifact,
                    (input, text),
                    seed.parse()?,
                    reference_rope,
                    decode,
                    top_p,
                    host_fp64,
                    precision,
                )
            }
            #[cfg(not(all(
                feature = "lm1b-mlx-experimental",
                target_os = "macos",
                target_arch = "aarch64"
            )))]
            {
                let _ = (
                    artifact,
                    input,
                    seed,
                    reference_rope,
                    decode,
                    host_fp64,
                    text,
                    precision,
                );
                Err("experimental LM1B-small Metal is unavailable in this build".into())
            }
        }
        _ => Err("experimental LM1B-small requires an explicit cuda-cudnn or mlx device".into()),
    }
}

#[cfg(any(
    all(feature = "lm1b-experimental", target_os = "linux"),
    all(
        feature = "lm1b-mlx-experimental",
        target_os = "macos",
        target_arch = "aarch64"
    )
))]
mod backend {
    use super::Result;
    use ncm_core::GeneratorBackend;
    use ncm_io::Artifact;
    use ncm_model::{Lm1bSmallTeacher, lm1b_small_reference_rope};
    use ncm_tokenizers::TextTokenizer;
    use std::{fs, io::Read, path::Path};

    fn prefixes(inputs: &[Vec<u32>]) -> Result<Vec<u32>> {
        if !(1..=32).contains(&inputs.len()) || inputs.iter().any(|row| row.as_slice() != [101]) {
            return Err("experimental LM1B-small expects 1..=32 rows containing only [101]".into());
        }
        Ok(vec![101; inputs.len()])
    }

    const ROTARY_BYTES: u64 = 32_920;
    enum Sampling {
        #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
        DeviceFull,
        #[cfg(all(
            feature = "lm1b-mlx-experimental",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        HostFull,
        #[cfg(all(
            feature = "lm1b-mlx-experimental",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        DeviceFp32,
        #[cfg(all(
            feature = "lm1b-mlx-experimental",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        HostTopP,
    }

    #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
    pub(super) fn sample_cuda(
        artifact: &str,
        input: (&str, bool),
        seed: u64,
        reference_rope: &str,
        decode: bool,
        precision: ncm_core::ComputePrecision,
    ) -> Result<serde_json::Value> {
        sample(
            || {
                ncm_backend_cuda::CandleBackend::cuda_lm1b_efficient_experimental(0)?
                    .with_precision(precision)
            },
            artifact,
            input,
            seed,
            reference_rope,
            decode,
            Sampling::DeviceFull,
        )
    }

    #[cfg(all(
        feature = "lm1b-mlx-experimental",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sample_mlx(
        artifact: &str,
        input: (&str, bool),
        seed: u64,
        reference_rope: &str,
        decode: bool,
        top_p: bool,
        host_fp64: bool,
        precision: ncm_core::ComputePrecision,
    ) -> Result<serde_json::Value> {
        sample(
            || ncm_backend_mlx::MlxBackend::gpu()?.with_precision(precision),
            artifact,
            input,
            seed,
            reference_rope,
            decode,
            if top_p {
                Sampling::HostTopP
            } else if host_fp64 {
                Sampling::HostFull
            } else {
                Sampling::DeviceFp32
            },
        )
    }

    fn sample<B: GeneratorBackend + Clone>(
        make_backend: impl FnOnce() -> ncm_core::Result<B>,
        artifact: &str,
        input: (&str, bool),
        seed: u64,
        reference_rope: &str,
        decode: bool,
        sampling: Sampling,
    ) -> Result<serde_json::Value> {
        let inputs = if input.1 {
            super::text_prefixes(&crate::read_input_bytes(Path::new(input.0))?)?
        } else {
            crate::read_inputs(Path::new(input.0))?
        };
        let prefixes = prefixes(&inputs)?;
        let external_rope = if reference_rope == "--embedded-rope" {
            None
        } else {
            if !fs::metadata(reference_rope)
                .map_err(|e| format!("LM1B rotary table {reference_rope}: {e}"))?
                .is_file()
            {
                return Err("LM1B-small reference rotary table must be a regular file".into());
            }
            let mut bytes = Vec::new();
            fs::File::open(reference_rope)?
                .take(ROTARY_BYTES + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() as u64 != ROTARY_BYTES {
                return Err("unexpected LM1B-small reference rotary table size".into());
            }
            Some(lm1b_small_reference_rope(&bytes)?)
        };

        let artifact = Artifact::load_inference_streaming(artifact)?;
        // Small, medium and large (v1 and v2) are admitted; any other profile fails explicitly.
        ncm_model::Lm1bSize::from_profile(&artifact.manifest().profile)?;
        if external_rope.is_none() && !artifact.manifest().profile.ends_with("-v2") {
            return Err(
                "LM1B v1 artifacts require --reference-rope or sibling rotary sidecar".into(),
            );
        }
        let tokenizer = if decode {
            Some(TextTokenizer::from_asset_bytes(
                artifact.file("tokenizer-manifest.json")?,
                artifact.file("tokenizer.json")?,
                artifact.file("tokenizer.LICENSE")?,
            )?)
        } else {
            None
        };
        let backend = make_backend()?;
        let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?;
        let model = if let Some((cosine, sine)) = external_rope {
            model.with_reference_rope_diagnostic(&cosine, &sine)?
        } else {
            model
        };
        drop(artifact);

        let batch = prefixes.len();
        let output = match sampling {
            #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
            Sampling::DeviceFull => model.sample_seeded_fully_device(&prefixes, batch, seed)?,
            #[cfg(all(
                feature = "lm1b-mlx-experimental",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            Sampling::HostFull => model.sample_seeded(&prefixes, batch, seed)?,
            #[cfg(all(
                feature = "lm1b-mlx-experimental",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            Sampling::HostTopP => model.sample_seeded_top_p_one(&prefixes, batch, seed)?,
            #[cfg(all(
                feature = "lm1b-mlx-experimental",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            Sampling::DeviceFp32 => model.sample_seeded_device_fp32(&prefixes, batch, seed)?,
        };
        let codes = backend.read_indices(&output.codes)?;
        if codes.len() != batch * 641
            || output.ids.len() != batch * 128
            || codes.iter().any(|&code| code >= 8192)
            || output.ids.iter().any(|&id| id >= 30528)
        {
            return Err("invalid LM1B-small backend output".into());
        }
        let codes: Vec<_> = codes
            .as_chunks::<641>()
            .0
            .iter()
            .map(<[u32; 641]>::as_slice)
            .collect();
        let ids: Vec<_> = output
            .ids
            .as_chunks::<128>()
            .0
            .iter()
            .map(<[u32; 128]>::as_slice)
            .collect();
        let mut result = serde_json::json!({ "codes": codes, "ids": ids });
        if let Some(tokenizer) = tokenizer {
            let texts = ids
                .iter()
                .map(|row| tokenizer.decode_lm1b_flm(row))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            result["texts"] = serde_json::json!(texts);
        }
        Ok(result)
    }

    #[cfg(test)]
    mod tests {
        use super::prefixes;

        #[test]
        fn only_the_pinned_start_token_and_bounded_batch_are_admitted() {
            for input in [&[][..], &[vec![]], &[vec![100]], &[vec![101, 101]]] {
                assert!(prefixes(input).is_err());
            }
            assert!(prefixes(&vec![vec![101]; 33]).is_err());
            assert_eq!(prefixes(&[vec![101]]).unwrap(), [101]);
            assert_eq!(prefixes(&vec![vec![101]; 32]).unwrap(), [101; 32]);
        }
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::{sample, text_prefixes};

    #[test]
    fn empty_only_text_maps_to_fixed_prefix() {
        assert_eq!(
            text_prefixes(br#"["", ""]"#).unwrap(),
            vec![vec![101], vec![101]]
        );
        for input in [
            br#"[]"#.as_slice(),
            br#"["hello"]"#,
            br#"[" "]"#,
            br#"[[101]]"#,
        ] {
            assert!(text_prefixes(input).is_err());
        }
        assert!(text_prefixes(serde_json::to_string(&vec![""; 33]).unwrap().as_bytes()).is_err());
    }

    fn args(device: &str) -> Vec<String> {
        [
            "ncm",
            "sample-lm1b-small",
            "artifact",
            device,
            "input",
            "11",
            "rope",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn top_p_flag_requires_metal_and_rejects_duplicates() {
        let p = ncm_core::ComputePrecision::Fp32;
        for flag in ["--top-p-fp32", "--host-fp64"] {
            let mut cuda = args("cuda-cudnn");
            cuda.push(flag.into());
            assert!(
                sample(&cuda, false, p)
                    .unwrap_err()
                    .to_string()
                    .contains("require explicit mlx")
            );
        }
        let mut both = args("mlx");
        both.extend(["--top-p-fp32".into(), "--host-fp64".into()]);
        assert!(
            sample(&both, false, p)
                .unwrap_err()
                .to_string()
                .contains("usage:")
        );
        let mut duplicate = args("mlx");
        duplicate.extend(["--top-p-fp32".into(), "--top-p-fp32".into()]);
        assert!(
            sample(&duplicate, false, p)
                .unwrap_err()
                .to_string()
                .contains("usage:")
        );
    }

    #[test]
    #[ignore = "requires retained LM1B CUDA cohort and pinned evaluator decoding"]
    fn retained_corpus_decode_matches_pinned_evaluator() {
        let size = std::env::var("NCM_LM1B_SIZE").unwrap();
        let (receipt_hash, reference_hash) = match size.as_str() {
            "small" => (
                "742d4f25e6c85b2037cfd621a86e48948b54bff137e639894330702feac99506",
                "e54a4cbe7272fd6c5a92cb5d0b8e638155e1e2e4eedf2f31836b818b8f91efff",
            ),
            "medium" => (
                "57f89148f93318ba7567bf32b7270d48867b8e719b8095855a24013bab1a20da",
                "f4111039d11d7ab9142530f7e5543ff07378bce752265381197fcc2c2d5becaa",
            ),
            "large" => (
                "166c8e49c3189595c84eb952fe99584e6515ef095e2dbfd720c3628924876a3f",
                "dcbd89cdbdfac0d9fea13dd57db154655c12ec2ed13926989584ac25e186780e",
            ),
            _ => panic!("unsupported LM1B size"),
        };
        let artifact =
            ncm_io::Artifact::load_inference(std::env::var("NCM_LM1B_ARTIFACT").unwrap()).unwrap();
        assert_eq!(
            artifact.manifest().profile,
            format!("lm1b-{size}-inference-v1")
        );
        let tokenizer = ncm_tokenizers::TextTokenizer::from_asset_bytes(
            artifact.file("tokenizer-manifest.json").unwrap(),
            artifact.file("tokenizer.json").unwrap(),
            artifact.file("tokenizer.LICENSE").unwrap(),
        )
        .unwrap();
        drop(artifact);
        let directory = std::path::PathBuf::from(std::env::var("NCM_LM1B_CAPTURES").unwrap());
        let receipt = std::fs::read(directory.join("receipt.json")).unwrap();
        assert_eq!(ncm_io::sha256(&receipt), receipt_hash);
        let receipt: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        assert_eq!(receipt["runtime"], "rust");
        assert_eq!(receipt["captures"].as_object().unwrap().len(), 80);
        let reference = std::fs::read(std::env::var("NCM_LM1B_DECODE_REFERENCE").unwrap()).unwrap();
        assert_eq!(ncm_io::sha256(&reference), reference_hash);
        let reference: serde_json::Value = serde_json::from_slice(&reference).unwrap();
        assert_eq!(reference["size"], size);
        assert_eq!(reference["receipt_sha256"], receipt_hash);
        let seeds = reference["seeds"].as_array().unwrap();
        let texts = reference["texts"].as_array().unwrap();
        assert_eq!(seeds.len(), 80);
        assert_eq!(texts.len(), 80);
        for (i, seed) in (2000..2080).enumerate() {
            assert_eq!(seeds[i], seed);
            let capture = std::fs::read(directory.join(format!("rust-seed-{seed}.json"))).unwrap();
            assert_eq!(
                ncm_io::sha256(&capture),
                receipt["captures"][seed.to_string()].as_str().unwrap()
            );
            let capture: serde_json::Value = serde_json::from_slice(&capture).unwrap();
            assert_eq!(capture["size"], size);
            assert_eq!(capture["seed"], seed);
            let ids: Vec<Vec<u32>> = serde_json::from_value(capture["raw_ids"].clone()).unwrap();
            let expected = texts[i].as_array().unwrap();
            assert_eq!(ids.len(), 256);
            assert_eq!(expected.len(), ids.len());
            for (row, expected) in ids.iter().zip(expected) {
                assert_eq!(row.len(), 128);
                assert_eq!(
                    tokenizer.decode_lm1b_flm(row).unwrap(),
                    expected.as_str().unwrap(),
                    "seed {seed}"
                );
            }
        }
    }

    #[test]
    fn unsupported_device_is_rejected_before_files_are_opened() {
        let err = sample(&args("cpu"), false, ncm_core::ComputePrecision::Fp32)
            .unwrap_err()
            .to_string();
        assert!(err.contains("explicit cuda-cudnn or mlx"));
    }

    #[cfg(not(all(
        feature = "lm1b-mlx-experimental",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    #[test]
    fn unavailable_metal_is_not_silently_replaced() {
        let p = ncm_core::ComputePrecision::Fp32;
        let err = sample(&args("mlx"), false, p).unwrap_err().to_string();
        assert!(err.contains("Metal is unavailable"));
        let mut top_p = args("mlx");
        top_p.push("--top-p-fp32".into());
        let err = sample(&top_p, false, p).unwrap_err().to_string();
        assert!(err.contains("Metal is unavailable"));
    }
}
