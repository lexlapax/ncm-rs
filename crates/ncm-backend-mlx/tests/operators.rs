#![cfg(feature = "mlx")]
mod support;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Device, with_device};
use ncm_backend_mlx::ops;
use support::{Fixture, Prepared, Result, check_output};

#[test]
fn linear_gather_matches_scalar_geometry_on_real_metal() -> Result<()> {
    with_device(Device::gpu(), || {
        for (batch, source, length) in [
            (1_usize, 1_usize, 89_usize),
            (1, 2, 89),
            (1, 13, 89),
            (1, 50, 89),
            (1, 89, 13),
            (4, 13, 89),
            (2, 37, 67),
            (2, 67, 37),
            (1, 5, 1),
        ] {
            let data: Vec<f32> = (0..batch * source * 64)
                .map(|i| ((i * 73 % 509) as f32 - 254.) / 37.)
                .collect();
            let x = Array::from_slice(&data, &[batch as i32, source as i32, 64]);
            let mut bins = Vec::with_capacity(length);
            for i in 0..length {
                let position = ((i as f64 + 0.5) * source as f64 / length as f64 - 0.5).max(0.);
                let lo = (position.floor() as i32).min(source as i32 - 1);
                let hi = (lo + 1).min(source as i32 - 1);
                let weight = position - f64::from(lo);
                let a = x
                    .try_index((.., lo..lo + 1, ..))?
                    .multiply(Array::from_f32((1. - weight) as f32))?;
                let b = x
                    .try_index((.., hi..hi + 1, ..))?
                    .multiply(Array::from_f32(weight as f32))?;
                bins.push(a.add(b)?);
            }
            let old = mlx_rs::ops::concatenate(&bins, 1)?;
            let new = ops::linear1d(&x, length)?;
            old.eval()?;
            new.eval()?;
            assert_eq!(old.shape(), new.shape());
            for (&expected, &actual) in old
                .try_as_slice::<f32>()?
                .iter()
                .zip(new.try_as_slice::<f32>()?)
            {
                assert!(
                    (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                    "shape {batch}/{source}/{length}: {actual} != {expected}"
                );
            }
        }
        Ok(())
    })
}
#[test]
fn accelerated_layer_norm_matches_composition_on_real_metal() -> Result<()> {
    with_device(Device::gpu(), || {
        for (batch, length, width) in [(1, 89, 256), (4, 89, 256), (2, 7, 33)] {
            let data: Vec<f32> = (0..batch * length * width)
                .map(|i| ((i * 73 % 509) as f32 - 254.) / 37.)
                .collect();
            let x = Array::from_slice(&data, &[batch as i32, length as i32, width as i32]);
            let reference = ops::layer_norm(&x, 1e-6)?;
            let accelerated = ops::layer_norm_accelerated(&x, 1e-6)?;
            reference.eval()?;
            accelerated.eval()?;
            for (&expected, &actual) in reference
                .try_as_slice::<f32>()?
                .iter()
                .zip(accelerated.try_as_slice::<f32>()?)
            {
                assert!((actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
            }
        }
        assert!(ops::layer_norm_accelerated(&Array::from_f32(1.), f64::NAN).is_err());
        Ok(())
    })
}

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
