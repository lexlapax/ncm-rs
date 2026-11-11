#![cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::sys;
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, WeightBackend};

// A separate test process avoids unrelated tests changing this device's pool.
#[allow(unsafe_code)]
fn used(backend: &CandleBackend) -> Result<u64, Box<dyn std::error::Error>> {
    backend.evaluate(&[])?;
    let stream = backend.device().as_cuda_device()?.cuda_stream();
    let context = stream.context();
    context.bind_to_thread()?;
    let mut pool = std::ptr::null_mut();
    let mut bytes = 0u64;
    // Safety: initialized/current context, CUDA-owned pool, live u64 output.
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, context.cu_device()).result()?;
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&mut bytes as *mut u64).cast(),
        )
        .result()?;
    }
    Ok(bytes)
}

#[test]
fn cached_geometry_is_stable_and_drops_with_its_owner() -> Result<(), Box<dyn std::error::Error>> {
    let controller = CandleBackend::cuda(0)?;
    let values: Vec<f32> = (0..2 * 89 * 3).map(|n| n as f32 / 17.).collect();
    let x = controller.upload_f32(&values, &[2, 89, 3])?;
    let other =
        controller.upload_f32(&values.iter().map(|v| -v).collect::<Vec<_>>(), &[2, 89, 3])?;
    let baseline = used(&controller)?;
    for _ in 0..3 {
        let owner = controller.clone();
        let mut steady = None;
        for _ in 0..4 {
            for output in [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89] {
                for area in [false, true] {
                    let run = |input: &candle_core::Tensor| {
                        if area {
                            owner.area1d(input, output)
                        } else {
                            owner.linear1d(input, output)
                        }
                    };
                    let result = run(&x)?;
                    let before = result.flatten_all()?.to_vec1::<f32>()?;
                    let later = run(&other)?;
                    owner.evaluate(&[later])?;
                    assert_eq!(result.flatten_all()?.to_vec1::<f32>()?, before);
                    let expected = if area {
                        ncm_backend_cuda::resample::area1d(&x, output)?
                    } else {
                        ncm_backend_cuda::resample::linear1d(&x, output)?
                    };
                    assert_eq!(before, expected.flatten_all()?.to_vec1::<f32>()?);
                }
            }
            let current = used(&controller)?;
            assert!(current > baseline);
            if let Some(previous) = steady {
                assert_eq!(current, previous);
            }
            steady = Some(current);
        }
        let survivor = owner.linear1d(&x, 37)?;
        let before = survivor.flatten_all()?.to_vec1::<f32>()?;
        let independent = owner.clone();
        drop(owner);
        assert_eq!(survivor.flatten_all()?.to_vec1::<f32>()?, before);
        drop(survivor);
        assert_eq!(used(&controller)?, baseline);
        // Keeping the clone alive must not retain the former owner's cache.
        independent.evaluate(&[])?;
    }
    Ok(())
}
