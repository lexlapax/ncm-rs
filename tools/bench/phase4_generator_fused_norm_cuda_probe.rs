//! Isolated CUDA regression for the experimental generator normalization patch.
#![cfg(feature = "cuda-cudnn")]
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::TinyStoriesTeacher;

fn digest(values: &[u32]) -> String {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    sha256(&bytes)
}

#[test]
fn strict_cuda_matches_retained_first_tinystories_batch() {
    let root = std::env::var_os("NCM_PHASE4_ARTIFACT").expect("NCM_PHASE4_ARTIFACT required");
    let backend = CandleBackend::cuda_cudnn(0).unwrap();
    let artifact = Artifact::load_inference(root).unwrap();
    let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact).unwrap();
    drop(artifact);
    let output = model.sample_seeded(&[0; 128], 128, 42).unwrap();
    let codes = backend.read_indices(&output.codes).unwrap();
    let ids = backend.read_indices(&output.ids).unwrap();
    assert_eq!(codes.len(), 128 * 641);
    assert_eq!(ids.len(), 128 * 128);
    assert_eq!(
        digest(&codes),
        "6e5c90065e205654a632bc8c74481ebe7a63df58373a1d0575e6356828f4acaf"
    );
    assert_eq!(
        digest(&ids),
        "54481f86f43beb5ebeb34c3662abb244b3c60d0e8907225ef5df2092ff9eebda"
    );
}
