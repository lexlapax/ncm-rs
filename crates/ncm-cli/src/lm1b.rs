//! Explicit experimental LM1B-small inference on Linux CUDA.
use super::Result;

pub(super) fn sample(args: &[String]) -> Result<serde_json::Value> {
    let [_, _, artifact, device, input, seed, reference_rope] = args else {
        return Err(
            "usage: ncm sample-lm1b-small ARTIFACT cuda-cudnn INPUT_JSON SEED REFERENCE_ROPE"
                .into(),
        );
    };
    if device != "cuda-cudnn" {
        return Err("experimental LM1B-small requires an explicit cuda-cudnn device".into());
    }
    #[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
    {
        cuda::sample(artifact, input, seed.parse()?, reference_rope)
    }
    #[cfg(not(all(feature = "lm1b-experimental", target_os = "linux")))]
    {
        let _ = (artifact, input, seed, reference_rope);
        Err("experimental LM1B-small CUDA is unavailable in this build".into())
    }
}

#[cfg(all(feature = "lm1b-experimental", target_os = "linux"))]
mod cuda {
    use super::Result;
    use ncm_backend_cuda::CandleBackend;
    use ncm_core::CodecBackend;
    use ncm_io::Artifact;
    use ncm_model::{Lm1bSmallTeacher, lm1b_small_reference_rope};
    use std::{fs, io::Read, path::Path};

    fn prefixes(inputs: &[Vec<u32>]) -> Result<Vec<u32>> {
        if !(1..=32).contains(&inputs.len()) || inputs.iter().any(|row| row.as_slice() != [101]) {
            return Err("experimental LM1B-small expects 1..=32 rows containing only [101]".into());
        }
        Ok(vec![101; inputs.len()])
    }

    const ROTARY_BYTES: u64 = 32_920;

    pub(super) fn sample(
        artifact: &str,
        input: &str,
        seed: u64,
        reference_rope: &str,
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
        let backend = CandleBackend::cuda_lm1b_efficient_experimental(0)?;
        let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?
            .with_reference_rope_diagnostic(&cosine, &sine)?;
        drop(artifact);

        let batch = prefixes.len();
        let output = model.sample_seeded_fully_device(&prefixes, batch, seed)?;
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
        Ok(serde_json::json!({ "codes": codes, "ids": ids }))
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
