//! Real Metal regression against the pinned Python Sudoku generator trace.
#![cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use ncm_backend_mlx::MlxBackend;
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
#[ignore = "requires the pinned Sudoku artifact and real Metal"]
fn metal_matches_pinned_python_synthetic_golden() {
    let root = std::env::var_os("NCM_PHASE3_ARTIFACT").expect("NCM_PHASE3_ARTIFACT required");
    let backend = MlxBackend::gpu().expect("real Metal required");
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
    let output = generator.generate(&prefixes, 2).unwrap();
    let codes = backend.read_indices(&output.codes).unwrap();
    let ids = backend.read_indices(&output.ids).unwrap();
    assert_eq!(
        digest(&codes),
        "0a9e9b6a4b2ed68e8a31ad1dfe8e1bbad910b39a2452975c059e429ab50ff245"
    );
    assert_eq!(
        digest(&ids),
        "a92e42db209774bd3a6b318e0e2f747040de73907098cbda4276ad209cd34a48"
    );
    generator.generate(&prefixes[..91], 1).unwrap();
    assert!(generator.generate(&[12; 91], 1).is_err());
    assert_eq!(backend.read_indices(&output.codes).unwrap(), codes);
    assert_eq!(backend.read_indices(&output.ids).unwrap(), ids);
}
