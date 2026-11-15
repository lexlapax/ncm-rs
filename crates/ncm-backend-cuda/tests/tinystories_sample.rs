//! Explicit real-artifact smoke check; this is not a quality or parity gate.
#![cfg(feature = "cuda")]
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::TinyStoriesTeacher;

#[test]
#[ignore = "requires the pinned TinyStories artifact and CUDA hardware"]
fn cuda_tinystories_seeded_generation_smoke() {
    let root = std::env::var_os("NCM_PHASE4_ARTIFACT").expect("NCM_PHASE4_ARTIFACT required");
    #[cfg(feature = "cuda-cudnn")]
    let backend = CandleBackend::cuda_cudnn(0).unwrap();
    #[cfg(not(feature = "cuda-cudnn"))]
    let backend = CandleBackend::cuda(0).unwrap();
    let artifact = Artifact::load_inference(root).unwrap();
    let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact).unwrap();
    assert!(model.sample_seeded(&[1], 1, 42).is_err());
    let first = model.sample_seeded(&[0, 0], 2, 42).unwrap();
    let a = backend.read_indices(&first.codes).unwrap();
    let ids = backend.read_indices(&first.ids).unwrap();
    assert_eq!(a.len(), 2 * 641);
    assert_eq!(ids.len(), 2 * 128);
    assert!(a.iter().all(|&x| x < 4096));
    assert!(ids.iter().all(|&x| x < 8192));
    let second = model.sample_seeded(&[0, 0], 2, 42).unwrap();
    assert_eq!(a, backend.read_indices(&second.codes).unwrap());
    assert_eq!(ids, backend.read_indices(&second.ids).unwrap());
    let third = model.sample_seeded(&[0, 0], 2, 43).unwrap();
    assert_ne!(a, backend.read_indices(&third.codes).unwrap());
    #[cfg(feature = "cuda-cudnn")]
    std::thread::scope(|scope| {
        let run = || {
            let out = model.sample_seeded(&[0], 1, 7).unwrap();
            backend.read_indices(&out.codes).unwrap()
        };
        let first = scope.spawn(run);
        let second = scope.spawn(run);
        assert_eq!(first.join().unwrap(), second.join().unwrap());
    });
}
