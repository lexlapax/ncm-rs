//! Whole-model cache/drop diagnostic, not acceptance latency or process peak attribution.
#[path = "../../../tests/support/host_memory.rs"]
mod host_memory;
use candle_core::cuda_backend::cudarc::driver::sys;
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[allow(unsafe_code)]
fn observe(backend: &CandleBackend) -> Result<Value> {
    backend.evaluate(&[])?;
    let stream = backend.device().as_cuda_device()?.cuda_stream();
    let context = stream.context();
    context.bind_to_thread()?;
    let mut pool = std::ptr::null_mut();
    // Safety: initialized/current context, CUDA-owned pool, live correctly typed
    // output pointers. This reads counters only; no memory/policy is modified.
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, context.cu_device()).result()?;
    }
    let mut counters = BTreeMap::new();
    for (name, attribute) in [
        (
            "used_current",
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        ),
        (
            "reserved_current",
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
        ),
    ] {
        let mut bytes = 0u64;
        // Safety: each documented attribute writes a u64 to this live local.
        unsafe {
            sys::cuMemPoolGetAttribute(pool, attribute, (&mut bytes as *mut u64).cast())
                .result()?;
        }
        counters.insert(name, bytes);
    }
    Ok(json!({"cuda_pool":counters,"host":host_memory::host()?}))
}
fn used(row: &Value) -> Result<u64> {
    row["cuda_pool"]["used_current"]
        .as_u64()
        .ok_or_else(|| "missing pool counter".into())
}
fn load(root: &str, backend: &CandleBackend) -> Result<Codec<CandleBackend>> {
    let artifact = Artifact::load(root)?;
    Ok(Codec::from_artifact(backend.clone(), &artifact)?)
}
fn exercise(model: &Codec<CandleBackend>, batch: usize) -> Result<Value> {
    let mut board = Vec::new();
    for row in 0..9 {
        if row > 0 {
            board.push(10);
        }
        for col in 0..9 {
            board.push((row * 3 + row / 3 + col) % 9 + 1);
        }
    }
    let tokens = board.repeat(batch);
    let (encoded, decoded) = model.reconstruct(&tokens, batch)?;
    let mut floats = encoded.snapshots;
    floats.extend(decoded.snapshots);
    floats.push(decoded.logits);
    model.backend().evaluate(&floats)?;
    if model.backend().read_indices(&decoded.ids)? != tokens {
        return Err("fixed solved-board reconstruction differs".into());
    }
    let retained = observe(model.backend())?;
    drop(floats);
    Ok(retained) // No request outputs survive the caller's next observation.
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || cfg!(debug_assertions) {
        return Err("usage (release CUDA): codec_memory CANONICAL_BUNDLE NEW_REPORT".into());
    }
    if Path::new(&args[2]).exists() {
        return Err("output exists".into());
    }
    let manifest = fs::read(Path::new(&args[1]).join("manifest.json"))?;
    if manifest
        != include_bytes!(
            "../../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
        )
    {
        return Err("unpinned manifest".into());
    }
    let backend = CandleBackend::cuda(0)?;
    let initial = observe(&backend)?;
    let baseline = used(&initial)?;
    let mut cycles = Vec::new();
    for cycle in 0..10 {
        let model = load(&args[1], &backend)?;
        let loaded = observe(&backend)?;
        let mut requests = Vec::new();
        let mut steady = None;
        // First cycle also probes a long-lived model across repeated shape changes.
        let rounds = if cycle == 0 { 30 } else { 2 };
        for round in 0..rounds {
            for batch in [1, 2, 4] {
                let retained = exercise(&model, batch)?;
                let row = observe(&backend)?;
                let current = used(&row)?;
                if let Some(expected) = steady {
                    if current != expected {
                        return Err("live pool usage grows after warm preparation".into());
                    }
                } else {
                    steady = Some(current);
                }
                requests.push(
                    json!({"round":round,"batch":batch,"retained_outputs":retained,"memory":row}),
                );
            }
        }
        // Retain an independent backend clone while dropping all model state.
        let independent = model.backend().clone();
        drop(model);
        let dropped = observe(&backend)?;
        if used(&dropped)? != baseline {
            return Err("model drop retains live pool allocations".into());
        }
        independent.evaluate(&[])?;
        cycles
            .push(json!({"cycle":cycle,"loaded":loaded,"requests":requests,"after_drop":dropped}));
    }
    let sources: BTreeMap<_, _> = [
        "crates/ncm-backend-cuda/examples/codec_memory.rs",
        "tests/support/host_memory.rs",
        "crates/ncm-backend-cuda/src/runtime.rs",
        "crates/ncm-backend-cuda/src/codec.rs",
        "crates/ncm-backend-cuda/src/resample.rs",
        "crates/ncm-backend-cuda/src/ops.rs",
        "crates/ncm-model/src/lib.rs",
        "crates/ncm-model/src/codec.rs",
        "crates/ncm-model/src/blocks.rs",
        "crates/ncm-io/src/lib.rs",
        "Cargo.lock",
    ]
    .into_iter()
    .map(|p| Ok((p, sha256(&fs::read(p)?))))
    .collect::<Result<_>>()?;
    let report = json!({"schema_version":1,"diagnostic_profiling":true,"runtime":"rust-candle-cuda-model-memory",
        "manifest_sha256":sha256(&manifest),"source_sha256":sources,"initial":initial,"cycles":cycles,
        "all_checks_passed":true,"scope":"Ten full authenticated model load/execute/drop cycles; first model executes thirty B1/B2/B4 rounds. Each request has synchronized retained-output and after-output-drop observations; retained outputs include native index tensors, unlike the acceptance harness's host ID vectors. Independent backend clones remain alive across model drop.",
        "limitations":["CUDA pool used/reserved and process RSS/HWM have different coverage; no interchangeable physical peak claim.","No timing or numerical-tolerance change; fixed-board output checks supplement, not replace, trained qualification.","Zero live pool usage does not imply zero reserved/context/native-library memory or zero host RSS."]});
    fs::write(
        &args[2],
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    Ok(())
}
