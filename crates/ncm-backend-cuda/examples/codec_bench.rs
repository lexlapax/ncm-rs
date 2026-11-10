//! CUDA codec measurements. Snapshot/output scope is fixed in codec-bench-v1.json.
#[path = "support/stages.rs"]
mod stages;
use candle_core::{Tensor, cuda_backend::cudarc::driver::sys};
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::Codec;
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Read/reset only this benchmark process's CUDA default async-pool counters.
/// Safety: context is initialized/bound; device zero is the selected device;
/// CUDA documents these four attributes as u64, and each pointer lives through
/// the synchronous FFI call. No tensor pointer or allocation is changed here.
#[allow(unsafe_code)]
fn pool(b: &CandleBackend, reset: bool) -> Result<Value> {
    b.device()
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .bind_to_thread()?;
    let mut handle = std::ptr::null_mut();
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut handle, 0).result()?;
    }
    use sys::CUmemPool_attribute::*;
    let mut values = BTreeMap::new();
    for (name, attribute) in [
        ("used_current", CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
        ("used_peak", CU_MEMPOOL_ATTR_USED_MEM_HIGH),
        ("reserved_current", CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
        ("reserved_peak", CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH),
        ("release_threshold", CU_MEMPOOL_ATTR_RELEASE_THRESHOLD),
    ] {
        let mut value = 0u64;
        if reset && name.ends_with("peak") {
            unsafe {
                sys::cuMemPoolSetAttribute(handle, attribute, (&mut value as *mut u64).cast())
                    .result()?;
            }
        }
        unsafe {
            sys::cuMemPoolGetAttribute(handle, attribute, (&mut value as *mut u64).cast())
                .result()?;
        }
        values.insert(name, value);
    }
    let (free, total) = b
        .device()
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .mem_get_info()?;
    values.insert("device_global_used", (total - free) as u64);
    Ok(json!(values))
}
fn host() -> Result<Value> {
    let status = fs::read_to_string("/proc/self/status")?;
    let mut result = BTreeMap::new();
    for key in ["VmRSS:", "VmHWM:"] {
        let line = status
            .lines()
            .find(|line| line.starts_with(key))
            .ok_or("missing RSS")?;
        result.insert(
            key.trim_end_matches(':'),
            line.split_whitespace()
                .nth(1)
                .ok_or("missing RSS value")?
                .parse::<u64>()?
                * 1024,
        );
    }
    Ok(json!(result))
}
fn load(root: &str, b: &CandleBackend) -> Result<Codec<CandleBackend>> {
    let artifact = Artifact::load(root)?;
    let model = Codec::from_artifact(b.clone(), &artifact)?;
    b.evaluate(&[])?;
    Ok(model)
}
fn data(batch: usize, levels: &[usize]) -> (Vec<u32>, Vec<u32>) {
    let mut tokens = Vec::new();
    let mut codes = Vec::new();
    for index in 0..batch {
        for row in 0..9 {
            if row > 0 {
                tokens.push(10);
            }
            for col in 0..9 {
                tokens.push(((row * 3 + row / 3 + col + index % 2) % 9 + 1) as u32);
            }
        }
        for (level, &length) in levels.iter().enumerate() {
            codes.extend((0..length).map(|p| ((index * length + p + 3 * level) % 16) as u32));
        }
    }
    (tokens, codes)
}
fn ids_hash(ids: &[u32]) -> String {
    sha256(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
struct Output {
    floats: Vec<Tensor>,
    ids: Vec<u32>,
    logits: Option<Tensor>,
}
fn call(
    model: &Codec<CandleBackend>,
    scope: &str,
    tokens: &[u32],
    codes: &[u32],
    batch: usize,
    latent: &Tensor,
) -> Result<Output> {
    let b = model.backend();
    match scope {
        "encode" => {
            let encoded = model.encode(tokens, batch)?;
            b.evaluate(&encoded.snapshots)?;
            Ok(Output {
                floats: encoded.snapshots,
                ids: b.read_indices(&encoded.codes)?,
                logits: None,
            })
        }
        "decode" => {
            let decoded = model.decode(codes, batch)?;
            let mut floats = decoded.snapshots;
            floats.push(decoded.logits.clone());
            b.evaluate(&floats)?;
            Ok(Output {
                floats,
                ids: b.read_indices(&decoded.ids)?,
                logits: Some(decoded.logits),
            })
        }
        "reconstruct" => {
            let (encoded, decoded) = model.reconstruct(tokens, batch)?;
            let mut floats = encoded.snapshots;
            floats.extend(decoded.snapshots);
            floats.push(decoded.logits.clone());
            b.evaluate(&floats)?;
            let mut ids = b.read_indices(&encoded.codes)?;
            ids.extend(b.read_indices(&decoded.ids)?);
            Ok(Output {
                floats,
                ids,
                logits: Some(decoded.logits),
            })
        }
        "resident_decoder_block" => {
            let value = model.decoder_block(latent)?;
            b.evaluate(std::slice::from_ref(&value))?;
            Ok(Output {
                floats: vec![value],
                ids: Vec::new(),
                logits: None,
            })
        }
        _ => Err("unknown scope".into()),
    }
}
fn timed<T>(b: &CandleBackend, f: impl FnOnce() -> Result<T>) -> Result<(T, f64, f64)> {
    b.evaluate(&[])?;
    let stream = b.device().as_cuda_device()?.cuda_stream();
    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let clock = Instant::now();
    let output = f()?;
    let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    end.synchronize()?;
    let wall = clock.elapsed().as_secs_f64() * 1000.;
    Ok((output, wall, f64::from(start.elapsed_ms(&end)?)))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) || cfg!(debug_assertions) {
        return Err(
            "usage (release CUDA): codec_bench BUNDLE WORKLOAD NEW_REPORT [--profile-block4]"
                .into(),
        );
    }
    let profile_block4 = match args.get(4).map(String::as_str) {
        None => false,
        Some("--profile-block4") => true,
        _ => return Err("unknown profiling option".into()),
    };
    if std::path::Path::new(&args[3]).exists() {
        return Err("output exists".into());
    }
    let workload_bytes = fs::read(&args[2])?;
    let workload: Value = serde_json::from_slice(&workload_bytes)?;
    let pinned: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/codec-bench-v1.json"))?;
    if workload != pinned {
        return Err("unsupported workload protocol".into());
    }
    let manifest = fs::read(std::path::Path::new(&args[1]).join("manifest.json"))?;
    let anchor = include_bytes!(
        "../../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
    );
    if serde_json::from_slice::<Value>(&manifest)? != serde_json::from_slice::<Value>(anchor)? {
        return Err("artifact differs from pinned inventory".into());
    }
    let b = CandleBackend::cuda(0)?;
    if profile_block4 {
        candle_core::cuda_backend::cudarc::driver::profiler_stop()?;
    }
    b.evaluate(&[])?;
    let before = pool(&b, true)?;
    let clock = Instant::now();
    let model = load(&args[1], &b)?;
    let first_load_ms = clock.elapsed().as_secs_f64() * 1000.;
    let loaded = pool(&b, false)?;
    drop(model);
    b.evaluate(&[])?;
    let mut load_trials = Vec::new();
    for trial in 0..35 {
        let clock = Instant::now();
        let model = load(&args[1], &b)?;
        let elapsed = clock.elapsed().as_secs_f64() * 1000.;
        if trial >= 5 {
            load_trials.push(elapsed);
        }
        drop(model);
        b.evaluate(&[])?;
    }
    let load_memory = json!({"before":before,"first_loaded":loaded,"after_trials":pool(&b,false)?,"host":host()?});
    let model = load(&args[1], &b)?;
    let (cold_tokens, cold_codes) = data(1, model.config().levels());
    let (cold, cold_ms, cold_span) = timed(&b, || {
        call(
            &model,
            "reconstruct",
            &cold_tokens,
            &cold_codes,
            1,
            &model.weights()["encoder.wte.weight"],
        )
    })?;
    drop(cold);
    let mut rows = Vec::new();
    let mut stage_rows = Vec::new();
    for batch in serde_json::from_value::<Vec<usize>>(workload["batches"].clone())? {
        let (tokens, codes) = data(batch, model.config().levels());
        let encoded = model.encode(&tokens, batch)?;
        let latent = encoded.snapshots.last().ok_or("missing latent")?.clone();
        drop(encoded);
        for scope in ["encode", "decode", "reconstruct", "resident_decoder_block"] {
            b.evaluate(&[])?;
            pool(&b, true)?;
            let selected = profile_block4 && batch == 4 && scope == "resident_decoder_block";
            if selected {
                candle_core::cuda_backend::cudarc::driver::profiler_start()?;
            }
            let (first, first_ms, first_span) =
                timed(&b, || call(&model, scope, &tokens, &codes, batch, &latent))?;
            let expected_hash = ids_hash(&first.ids);
            let logits = first.logits.as_ref().map(|x| b.read_f32(x)).transpose()?;
            let float_elements: usize = first.floats.iter().map(Tensor::elem_count).sum();
            let output_count = first.ids.len();
            drop(first);
            let mut wall = Vec::new();
            let mut spans = Vec::new();
            for i in 0..35 {
                let (output, ms, span) =
                    timed(&b, || call(&model, scope, &tokens, &codes, batch, &latent))?;
                if ids_hash(&output.ids) != expected_hash {
                    return Err("nonrepeatable output IDs".into());
                }
                if i >= 5 {
                    wall.push(ms);
                    spans.push(span);
                }
                drop(output);
            }
            b.evaluate(&[])?;
            if selected {
                candle_core::cuda_backend::cudarc::driver::profiler_stop()?;
            }
            rows.push(json!({"batch":batch,"scope":scope,"first_ms":first_ms,"first_cuda_stream_span_ms":first_span,
                "trials_ms":wall,"cuda_stream_spans_ms":spans,"output_ids_sha256":expected_hash,"output_ids_count":output_count,
                "output_float_elements":float_elements,"input_ids_sha256":ids_hash(&tokens),"supplied_codes_sha256":ids_hash(&codes),
                "first_logits":logits,"pool_bytes":pool(&b,false)?,"host":host()?}));
        }
        stage_rows.push(
            json!({"batch":batch,"cuda_stream_spans_ms":stages::profile(&model,&tokens,batch)?}),
        );
    }
    let source_paths = [
        "crates/ncm-backend-cuda/examples/codec_bench.rs",
        "crates/ncm-backend-cuda/examples/support/stages.rs",
        "crates/ncm-backend-cuda/src/runtime.rs",
        "crates/ncm-backend-cuda/src/codec.rs",
        "crates/ncm-backend-cuda/src/ops.rs",
        "crates/ncm-backend-cuda/src/resample.rs",
        "crates/ncm-core/src/config.rs",
        "crates/ncm-model/src/lib.rs",
        "crates/ncm-model/src/blocks.rs",
        "crates/ncm-model/src/codec.rs",
        "crates/ncm-io/src/lib.rs",
        "Cargo.lock",
    ];
    let sources: BTreeMap<_, _> = source_paths
        .into_iter()
        .map(|p| Ok((p, sha256(&fs::read(p)?))))
        .collect::<Result<_>>()?;
    let report = json!({"schema_version":1,"runtime":"rust-candle","device":"cuda","diagnostic_profiling":profile_block4,"workload_sha256":sha256(&workload_bytes),
        "manifest_sha256":sha256(&fs::read(std::path::Path::new(&args[1]).join("manifest.json"))?),"source_sha256":sources,
        "precision":"fp32","index_dtype":"u32","warmups":5,"trials":30,"first_load_ms":first_load_ms,"load_trials_ms":load_trials,
        "load_memory":load_memory,"measurements":rows,"instrumented_stages":stage_rows,"host_final":host()?,"cold_disk_cache":false,"context_initialization_timed":false,
        "memory_scope":"CUDA default async pool used/reserved counters, with explicit 128 MiB release threshold; not a hard capacity limit; excludes non-pool context/cuBLAS allocations; RSS is process high-water mark",
        "first_execution_ms":cold_ms,"first_execution_cuda_stream_span_ms":cold_span,
        "first_execution_scope":"batch-one reconstruction before component preparation",
        "compile_profile":"ahead-of-time release; no runtime graph compilation"});
    fs::write(
        &args[3],
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    Ok(())
}
