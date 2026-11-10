#![cfg(feature = "cpu")]

mod support;

use candle_core::{Device, Tensor};
use ncm_backend_cuda::ops;
use support::{Fixture, Prepared, Result, TensorData, check_output, compare};

const FIXTURE: &[u8] = include_bytes!("../../../tests/fixtures/operators-v1.json");

#[test]
fn all_operator_fixtures_match_torch_on_cpu() -> Result<()> {
    let fixture = Fixture::parse(FIXTURE)?;
    assert!(!fixture.torch_version.is_empty());
    assert_eq!(fixture.cases.len(), 18);
    for case in fixture.cases {
        let prepared = Prepared::new(&case, &Device::Cpu)?;
        let metrics = check_output(&prepared.execute()?, &case.expected)?;
        assert!(metrics.passed, "{}: {metrics:?}", case.name);
    }
    Ok(())
}

#[test]
fn invalid_shapes_values_and_schema_are_rejected() {
    for bad in [
        TensorData {
            shape: vec![usize::MAX, 2],
            data: vec![],
        },
        TensorData {
            shape: vec![0],
            data: vec![],
        },
        TensorData {
            shape: vec![2],
            data: vec![1.0],
        },
        TensorData {
            shape: vec![1],
            data: vec![f32::NAN],
        },
    ] {
        assert!(bad.validate().is_err());
    }
    let mut fixture: serde_json::Value = serde_json::from_slice(FIXTURE).unwrap();
    fixture["schema_version"] = 2.into();
    assert!(Fixture::parse(&serde_json::to_vec(&fixture).unwrap()).is_err());
    fixture["schema_version"] = 1.into();
    fixture["unexpected"] = true.into();
    assert!(Fixture::parse(&serde_json::to_vec(&fixture).unwrap()).is_err());
}

#[test]
fn comparison_rejects_nonfinite_and_detects_regressions() {
    assert!(compare(&[f32::NAN], &[0.0]).is_err());
    assert!(compare(&[0.0], &[f32::INFINITY]).is_err());
    assert!(compare(&[], &[]).is_err());
    assert!(compare(&[0.0], &[0.0, 1.0]).is_err());
    assert!(!compare(&[1.0], &[0.0]).unwrap().passed);
    assert!(compare(&[1.0], &[1.0]).unwrap().passed);
}

#[test]
fn invalid_operator_parameters_are_rejected() -> Result<()> {
    let x = Tensor::zeros((1, 3, 4), candle_core::DType::F32, &Device::Cpu)?;
    assert!(ops::area1d(&x, 0).is_err());
    assert!(ops::linear1d(&x, 0).is_err());
    assert!(ops::group_norm(&x, 0, 1e-5).is_err());
    assert!(ops::group_norm(&x, 3, 1e-5).is_err());
    assert!(ops::layer_norm(&x, f64::NAN).is_err());
    assert!(ops::rms_norm(&x, 0.0).is_err());
    let odd = Tensor::zeros((1, 3), candle_core::DType::F32, &Device::Cpu)?;
    assert!(ops::rope(&odd, &odd, &odd).is_err());
    Ok(())
}

#[test]
fn all_masked_attention_rows_are_rejected() -> Result<()> {
    let mut fixture = Fixture::parse(FIXTURE)?;
    let case = fixture
        .cases
        .iter_mut()
        .find(|case| case.name == "prefix_attention_padded")
        .unwrap();
    case.inputs.get_mut("mask").unwrap().data.fill(0.0);
    assert!(Prepared::new(case, &Device::Cpu).is_err());
    Ok(())
}
