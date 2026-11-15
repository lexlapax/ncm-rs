//! Explicit hardware regression against pinned Python's original Sudoku trace.
#![cfg(feature = "cuda")]
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Generator;

fn digest(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .flat_map(|&x| i64::from(x).to_le_bytes())
        .collect();
    sha256(&bytes)
}
#[test]
#[ignore = "requires the pinned Sudoku artifact and CUDA hardware"]
fn cuda_matches_pinned_python_synthetic_golden() {
    let root = std::env::var_os("NCM_PHASE3_ARTIFACT").expect("NCM_PHASE3_ARTIFACT required");
    let backend = CandleBackend::cuda(0).unwrap();
    let artifact = Artifact::load(root).unwrap();
    let generator = Generator::from_artifact(backend.clone(), &artifact).unwrap();
    assert!(generator.generate(&[12; 91], 1).is_err());
    let mut prefixes = Vec::new();
    for shift in 0..2 {
        prefixes.push(11);
        for row in 0..9 {
            if row > 0 {
                prefixes.push(10);
            }
            for col in 0..9 {
                let digit = (row * 3 + row / 3 + col + shift) % 9 + 1;
                prefixes.push(if (row * 9 + col) % 3 == 0 {
                    0
                } else {
                    digit as u32
                });
            }
        }
        prefixes.push(11);
    }
    let out = generator.generate(&prefixes, 2).unwrap();
    // A later request and a rejected request must not alter prior outputs.
    let previous_codes = backend.read_indices(&out.codes).unwrap();
    let previous_ids = backend.read_indices(&out.ids).unwrap();
    generator.generate(&prefixes[..91], 1).unwrap();
    assert!(generator.generate(&[12; 91], 1).is_err());
    assert_eq!(backend.read_indices(&out.codes).unwrap(), previous_codes);
    assert_eq!(backend.read_indices(&out.ids).unwrap(), previous_ids);
    assert_eq!(
        digest(&backend.read_indices(&out.codes).unwrap()),
        "0a9e9b6a4b2ed68e8a31ad1dfe8e1bbad910b39a2452975c059e429ab50ff245"
    );
    assert_eq!(
        digest(&backend.read_indices(&out.ids).unwrap()),
        "a92e42db209774bd3a6b318e0e2f747040de73907098cbda4276ad209cd34a48"
    );
    std::thread::scope(|scope| {
        let run = || {
            let output = generator.generate(&prefixes, 2).unwrap();
            (
                backend.read_indices(&output.codes).unwrap(),
                backend.read_indices(&output.ids).unwrap(),
            )
        };
        let a = scope.spawn(run);
        let b = scope.spawn(run);
        for (codes, ids) in [a.join().unwrap(), b.join().unwrap()] {
            assert_eq!(codes, previous_codes);
            assert_eq!(ids, previous_ids);
        }
    });
}
