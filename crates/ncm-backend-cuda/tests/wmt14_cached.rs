//! Explicit, local-artifact GPU regression for the unqualified WMT14 cache.
#![cfg(all(feature = "wmt14-experimental", feature = "cuda-cudnn"))]
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::Wmt14Teacher;
use serde_json::Value;

#[test]
#[ignore = "requires the pinned local WMT14 artifact, prefixes and CUDA/cuDNN"]
fn cached_matches_uncached_for_padded_base_and_guided_batches() {
    let artifact = std::env::var("NCM_WMT14_CACHE_ARTIFACT").expect("artifact path");
    let prefixes = std::env::var("NCM_WMT14_CACHE_PREFIXES").expect("prefix path");
    let value: Value = serde_json::from_slice(&std::fs::read(prefixes).unwrap()).unwrap();
    let rows = value["prefixes"].as_array().unwrap();
    let mut input = Vec::new();
    for row in &rows[..4] {
        let row = row.as_array().unwrap();
        assert_eq!(row.len(), 96);
        assert!(row.iter().any(|v| v.as_u64() == Some(151643)));
        input.extend(
            row.iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap()),
        );
    }
    let mut synthetic = input.clone();
    for token in &mut synthetic[..96] {
        if *token == 151643 {
            *token = 17;
        }
    }
    synthetic[96..192].fill(151643);
    synthetic[96] = 151645;
    for token in synthetic[192..288].iter_mut().step_by(2) {
        *token = 151643;
    }
    let backend = CandleBackend::cuda_wmt14_reference(0).unwrap();
    let mut expected = Vec::new();
    for cached in [false, true] {
        let data = Artifact::load_inference(&artifact).unwrap();
        let model = if cached {
            Wmt14Teacher::from_artifact_cached_experimental(backend.clone(), &data).unwrap()
        } else {
            Wmt14Teacher::from_artifact(backend.clone(), &data).unwrap()
        };
        for prefixes in [&input, &synthetic] {
            for batch in [1, 4] {
                for cfg in [false, true] {
                    let prefix = &prefixes[..batch * 96];
                    let out = if cfg {
                        model.generate_cfg_top1(prefix, batch, 1.0).unwrap()
                    } else {
                        model.generate_top1(prefix, batch).unwrap()
                    };
                    let result = (
                        backend.read_indices(&out.codes).unwrap(),
                        backend.read_indices(&out.ids).unwrap(),
                    );
                    if cached {
                        assert_eq!(result, expected.remove(0), "batch {batch}, CFG {cfg}");
                    } else {
                        expected.push(result);
                    }
                }
            }
        }
    }
    assert!(expected.is_empty());
}

#[path = "../../../tests/support/codec_capture.rs"]
mod capture;

#[test]
#[ignore = "requires the pinned local WMT14 artifact, prefixes and CUDA/cuDNN"]
fn cached_forced_history_preserves_existing_tensor_bands() {
    let artifact = std::env::var("NCM_WMT14_CACHE_ARTIFACT").expect("artifact path");
    let prefixes = std::env::var("NCM_WMT14_CACHE_PREFIXES").expect("prefix path");
    let value: Value = serde_json::from_slice(&std::fs::read(prefixes).unwrap()).unwrap();
    let input: Vec<u32> = value["prefixes"].as_array().unwrap()[..4]
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    let b = CandleBackend::cuda_wmt14_reference(0).unwrap();
    let mut expected = std::collections::BTreeMap::new();
    let mut expected_indices = std::collections::BTreeMap::new();
    let mut codes = Vec::new();
    for cached in [false, true] {
        let artifact = Artifact::load_inference(&artifact).unwrap();
        let model = if cached {
            Wmt14Teacher::from_artifact_cached_experimental(b.clone(), &artifact).unwrap()
        } else {
            Wmt14Teacher::from_artifact(b.clone(), &artifact).unwrap()
        };
        if !cached {
            codes = b
                .read_indices(&model.generate_top1(&input, 4).unwrap().codes)
                .unwrap();
        }
        let mut trace = capture::Captured::default();
        model.run_traced(&input, &codes, 4, &mut trace).unwrap();
        assert_eq!(trace.tensors.len(), 23, "incomplete tensor trace");
        assert_eq!(trace.indices.len(), 3, "incomplete index trace");
        for (name, indices) in trace.indices {
            let observed = (b.indices_shape(&indices), b.read_indices(&indices).unwrap());
            if !cached {
                expected_indices.insert(name, observed);
            } else {
                assert_eq!(
                    observed,
                    expected_indices.remove(&name).expect("missing index trace"),
                    "{name}"
                );
            }
        }
        for (name, tensor) in trace.tensors {
            let shape = b.shape(&tensor);
            let values = b.read_f32(&tensor).unwrap();
            if !cached {
                expected.insert(name, (shape, values));
            } else {
                let (reference_shape, reference): (Vec<usize>, Vec<f32>) =
                    expected.remove(&name).expect("missing reference tensor");
                assert_eq!(shape, reference_shape, "{name}: shape");
                assert_eq!(reference.len(), values.len(), "{name}");
                let (atol, rtol) = capture::tolerance(&name);
                for (actual, reference) in values.into_iter().zip(reference) {
                    assert!(
                        actual.is_finite() && reference.is_finite(),
                        "{name}: nonfinite"
                    );
                    let delta = (f64::from(actual) - f64::from(reference)).abs();
                    assert!(
                        delta <= atol + rtol * f64::from(reference).abs(),
                        "{name}: delta {delta}"
                    );
                }
            }
        }
    }
    assert!(expected.is_empty());
    assert!(expected_indices.is_empty());
}
