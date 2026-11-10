#![cfg(feature = "mlx")]
mod support;
use mlx_rs::{Array, Device, with_device};
use ncm_backend_mlx::ops;
use support::{Fixture, Prepared, Result, check_output};
#[test]
fn all_fixtures_on_real_metal() -> Result<()> {
    with_device(Device::gpu(), || {
        let fixture = Fixture::parse(include_bytes!("../../../tests/fixtures/operators-v1.json"))?;
        assert!(!fixture.torch_version.is_empty());
        assert_eq!(fixture.cases.len(), 18);
        for case in &fixture.cases {
            let output = Prepared::new(case)?.execute()?;
            output.eval()?;
            let result = check_output(&output, &case.expected)?;
            assert!(result.passed, "{}: {result:?}", case.name);
        }
        Ok(())
    })
}
#[test]
fn invalid_parameters_and_masks() -> Result<()> {
    with_device(Device::gpu(), || {
        let x = Array::from_slice(&[1f32; 12], &[1, 3, 4]);
        assert!(ops::group_norm(&x, 3, 1e-5).is_err());
        assert!(ops::area1d(&x, 0).is_err());
        assert!(ops::linear1d(&x, 0).is_err());
        assert!(ops::layer_norm(&x, f64::NAN).is_err());
        let odd = Array::from_slice(&[1f32; 3], &[1, 1, 1, 3]);
        assert!(ops::rope(&odd, &odd, &odd).is_err());
        let mut fixture =
            Fixture::parse(include_bytes!("../../../tests/fixtures/operators-v1.json"))?;
        let attention = fixture
            .cases
            .iter_mut()
            .find(|c| c.inputs.contains_key("mask"))
            .unwrap();
        attention.inputs.get_mut("mask").unwrap().data.fill(0.);
        assert!(Prepared::new(attention).is_err());
        Ok(())
    })
}
