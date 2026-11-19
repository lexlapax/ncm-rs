//! Explicit experimental LM1B-small inference on CUDA or Metal.
use super::Result;

pub(super) fn sample(args: &[String]) -> Result<serde_json::Value> {
    const USAGE: &str = "usage: ncm sample-lm1b-small ARTIFACT cuda-cudnn|mlx INPUT_JSON SEED REFERENCE_ROPE [--decode]";
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
    let decode = match options {
        [] => false,
        [flag] if flag == "--decode" => true,
        _ => return Err(USAGE.into()),
    };
    match device.as_str() {
        "cuda-cudnn" => {
            #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
            {
                backend::sample_cuda(artifact, input, seed.parse()?, reference_rope, decode)
            }
            #[cfg(not(all(feature = "lm1b-experimental", target_os = "linux")))]
            {
                let _ = (artifact, input, seed, reference_rope, decode);
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
                backend::sample_mlx(artifact, input, seed.parse()?, reference_rope, decode)
            }
            #[cfg(not(all(
                feature = "lm1b-mlx-experimental",
                target_os = "macos",
                target_arch = "aarch64"
            )))]
            {
                let _ = (artifact, input, seed, reference_rope, decode);
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

    #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
    pub(super) fn sample_cuda(
        artifact: &str,
        input: &str,
        seed: u64,
        reference_rope: &str,
        decode: bool,
    ) -> Result<serde_json::Value> {
        sample(
            || ncm_backend_cuda::CandleBackend::cuda_lm1b_efficient_experimental(0),
            artifact,
            input,
            seed,
            reference_rope,
            decode,
            true,
        )
    }

    #[cfg(all(
        feature = "lm1b-mlx-experimental",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub(super) fn sample_mlx(
        artifact: &str,
        input: &str,
        seed: u64,
        reference_rope: &str,
        decode: bool,
    ) -> Result<serde_json::Value> {
        sample(
            ncm_backend_mlx::MlxBackend::gpu,
            artifact,
            input,
            seed,
            reference_rope,
            decode,
            false,
        )
    }

    fn sample<B: GeneratorBackend + Clone>(
        make_backend: impl FnOnce() -> ncm_core::Result<B>,
        artifact: &str,
        input: &str,
        seed: u64,
        reference_rope: &str,
        decode: bool,
        device_sampling: bool,
    ) -> Result<serde_json::Value> {
        let prefixes = prefixes(&crate::read_inputs(Path::new(input))?)?;
        if !fs::metadata(reference_rope)?.is_file() {
            return Err("LM1B-small reference rotary table must be a regular file".into());
        }
        let mut bytes = Vec::new();
        fs::File::open(reference_rope)?
            .take(ROTARY_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 != ROTARY_BYTES {
            return Err("unexpected LM1B-small reference rotary table size".into());
        }
        let (cosine, sine) = lm1b_small_reference_rope(&bytes)?;

        let artifact = Artifact::load_inference(artifact)?;
        if artifact.manifest().profile != "lm1b-small-inference-v1" {
            return Err("unsupported LM1B-small inference artifact".into());
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
        let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?
            .with_reference_rope_diagnostic(&cosine, &sine)?;
        drop(artifact);

        let batch = prefixes.len();
        let output = if device_sampling {
            model.sample_seeded_fully_device(&prefixes, batch, seed)?
        } else {
            model.sample_seeded(&prefixes, batch, seed)?
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
    use super::sample;

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
    fn unsupported_device_is_rejected_before_files_are_opened() {
        let err = sample(&args("cpu")).unwrap_err().to_string();
        assert!(err.contains("explicit cuda-cudnn or mlx"));
    }

    #[cfg(not(all(
        feature = "lm1b-mlx-experimental",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    #[test]
    fn unavailable_metal_is_not_silently_replaced() {
        let err = sample(&args("mlx")).unwrap_err().to_string();
        assert!(err.contains("Metal is unavailable"));
    }
}
