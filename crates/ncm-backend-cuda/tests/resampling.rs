#![cfg(feature = "cpu")]
use candle_core::{Device, Tensor};
use ncm_backend_cuda::resample;

fn check(device: &Device) -> candle_core::Result<()> {
    let x = Tensor::from_vec(
        vec![1f32, 2., 3., 4., 5., -1., -2., -3., -4., -5.],
        (2, 5, 1),
        device,
    )?;
    let area = resample::area1d(&x, 3)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(area, vec![1.5, 3., 4.5, -1.5, -3., -4.5]);
    let linear = resample::linear1d(&x, 3)?.flatten_all()?.to_vec1::<f32>()?;
    for (a, e) in linear
        .iter()
        .zip([4. / 3., 3., 14. / 3., -4. / 3., -3., -14. / 3.])
    {
        assert!((a - e).abs() < 1e-5);
    }
    let x = x.broadcast_as((2, 5, 4))?;
    for f in [resample::area1d, resample::linear1d] {
        let output = f(&x, 3)?;
        let contiguous = f(&x.contiguous()?, 3)?;
        assert_eq!(output.dims(), &[2, 3, 4]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            contiguous.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            f(&x, 5)?.flatten_all()?.to_vec1::<f32>()?,
            x.flatten_all()?.to_vec1::<f32>()?
        );
        assert!(f(&x, 0).is_err());
        assert!(f(&x, usize::MAX).is_err());
        assert!(f(&x.to_dtype(candle_core::DType::F64)?, 3).is_err());
        assert!(f(&x.flatten_all()?, 3).is_err());
        let one = Tensor::from_vec(vec![3f32, -3.], (2, 1, 1), device)?;
        assert_eq!(
            f(&one, 4)?.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 3., 3., 3., -3., -3., -3., -3.]
        );
    }
    Ok(())
}

#[test]
fn cpu_resampling() -> candle_core::Result<()> {
    check(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resampling() -> candle_core::Result<()> {
    check(&Device::new_cuda(0)?)
}
