//! Matched Metal codec scopes; synchronized graph construction plus execution.
#[path = "../../../tests/support/host_memory.rs"]
mod host_memory;
use mlx_rs::memory;
use ncm_backend_mlx::{MlxBackend, MlxTensor};
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Codec, TraceSink};
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn memory() -> Result<Value> {
    Ok(
        json!({"active":memory::active_memory()?,"peak":memory::peak_memory()?,"cache":memory::cache_memory()?}),
    )
}
fn load(root: &str, b: &MlxBackend) -> Result<Codec<MlxBackend>> {
    let artifact = Artifact::load(root)?;
    let model = Codec::from_artifact(b.clone(), &artifact)?;
    b.evaluate(&model.weights().values().cloned().collect::<Vec<_>>())?;
    Ok(model)
}
fn data(batch: usize, levels: &[usize]) -> (Vec<u32>, Vec<u32>) {
    let mut tokens = Vec::new();
    let mut codes = Vec::new();
    for i in 0..batch {
        for row in 0..9 {
            if row > 0 {
                tokens.push(10);
            }
            for col in 0..9 {
                tokens.push(((row * 3 + row / 3 + col + i % 2) % 9 + 1) as u32);
            }
        }
        for (level, &length) in levels.iter().enumerate() {
            codes.extend((0..length).map(|p| ((i * length + p + 3 * level) % 16) as u32));
        }
    }
    (tokens, codes)
}
fn ids_hash(ids: &[u32]) -> String {
    sha256(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
struct Output {
    floats: Vec<MlxTensor>,
    ids: Vec<u32>,
    logits: Option<MlxTensor>,
}
fn call(
    model: &Codec<MlxBackend>,
    scope: &str,
    tokens: &[u32],
    codes: &[u32],
    batch: usize,
    latent: &MlxTensor,
) -> Result<Output> {
    let b = model.backend();
    match scope {
        "encode" => {
            let value = model.encode(tokens, batch)?;
            b.evaluate(&value.snapshots)?;
            Ok(Output {
                floats: value.snapshots,
                ids: b.read_indices(&value.codes)?,
                logits: None,
            })
        }
        "decode" => {
            let value = model.decode(codes, batch)?;
            let mut floats = value.snapshots;
            floats.push(value.logits.clone());
            b.evaluate(&floats)?;
            Ok(Output {
                floats,
                ids: b.read_indices(&value.ids)?,
                logits: Some(value.logits),
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
                ids: vec![],
                logits: None,
            })
        }
        _ => Err("unsupported scope".into()),
    }
}
fn timed<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    let start = Instant::now();
    let output = f()?;
    Ok((output, start.elapsed().as_secs_f64() * 1000.))
}
struct Stages<'a> {
    backend: &'a MlxBackend,
    start: Option<(String, Instant)>,
    values: BTreeMap<String, f64>,
}
impl Stages<'_> {
    fn begin(&mut self, name: &str) -> ncm_core::Result<()> {
        if self.start.is_some() {
            return Err(ncm_core::Error::Input("nested stage".into()));
        }
        self.start = Some((name.into(), Instant::now()));
        Ok(())
    }
    fn end(&mut self, name: &str) -> ncm_core::Result<()> {
        let (actual, start) = self
            .start
            .take()
            .ok_or_else(|| ncm_core::Error::Input("missing stage".into()))?;
        if actual != name {
            return Err(ncm_core::Error::Input("mismatched stage".into()));
        }
        self.values
            .insert(actual, start.elapsed().as_secs_f64() * 1000.);
        Ok(())
    }
}
impl TraceSink<MlxBackend> for Stages<'_> {
    fn tensor(&mut self, name: &str, value: &MlxTensor) -> ncm_core::Result<()> {
        if name == "encode.pre_quant" {
            self.backend.evaluate(std::slice::from_ref(value))?;
            self.end("encoder_pre_quant")?;
        }
        if name == "decode.decoder.input" {
            self.backend.evaluate(std::slice::from_ref(value))?;
            self.begin("decoder_logits")?;
        }
        if name == "decode.logits" {
            self.backend.evaluate(std::slice::from_ref(value))?;
            self.end("decoder_logits")?;
            self.begin("output_selection_readback")?;
        }
        if let Some(level) = name
            .strip_suffix(".residual_before")
            .filter(|s| s.starts_with("encode.level_"))
        {
            self.backend.evaluate(std::slice::from_ref(value))?;
            self.begin(level)?;
        }
        if let Some(level) = name
            .strip_suffix(".contribution")
            .filter(|s| s.starts_with("encode.level_"))
        {
            self.backend.evaluate(std::slice::from_ref(value))?;
            self.end(level)?;
        }
        Ok(())
    }
    fn indices(&mut self, name: &str, value: &MlxTensor) -> ncm_core::Result<()> {
        if name == "encode.encoder.input" {
            self.backend.read_indices(value)?;
            self.begin("encoder_pre_quant")?;
        }
        if name == "decode.ids" {
            self.backend.read_indices(value)?;
            self.end("output_selection_readback")?;
        }
        Ok(())
    }
}
fn profile(
    model: &Codec<MlxBackend>,
    tokens: &[u32],
    batch: usize,
) -> Result<BTreeMap<String, Vec<f64>>> {
    let mut trials = BTreeMap::<String, Vec<f64>>::new();
    for i in 0..35 {
        let mut stages = Stages {
            backend: model.backend(),
            start: None,
            values: BTreeMap::new(),
        };
        let result = model.reconstruct_traced(tokens, batch, &mut stages)?;
        if stages.start.is_some() || stages.values.len() != 15 {
            return Err("incomplete stage profile".into());
        }
        if i >= 5 {
            for (name, ms) in stages.values {
                trials.entry(name).or_default().push(ms);
            }
        }
        drop(result);
    }
    Ok(trials)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || cfg!(debug_assertions) {
        return Err("usage (release): mlx_codec_bench BUNDLE WORKLOAD NEW_REPORT".into());
    }
    if Path::new(&args[3]).exists() {
        return Err("output exists".into());
    }
    let raw = fs::read(&args[2])?;
    let workload: Value = serde_json::from_slice(&raw)?;
    if workload
        != serde_json::from_str::<Value>(include_str!(
            "../../../tests/fixtures/codec-bench-v1.json"
        ))?
    {
        return Err("unpinned workload".into());
    }
    let manifest = fs::read(Path::new(&args[1]).join("manifest.json"))?;
    if serde_json::from_slice::<Value>(&manifest)?
        != serde_json::from_str::<Value>(include_str!(
            "../../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
        ))?
    {
        return Err("unpinned artifact".into());
    }
    let b = MlxBackend::gpu()?;
    memory::clear_cache()?;
    memory::reset_peak_memory()?;
    let before = memory()?;
    let (model, first_load) = timed(|| load(&args[1], &b))?;
    let loaded = memory()?;
    drop(model);
    let after_drop = memory()?;
    let mut loads = Vec::new();
    for i in 0..35 {
        let (model, ms) = timed(|| load(&args[1], &b))?;
        if i >= 5 {
            loads.push(ms);
        }
        drop(model);
    }
    let load_memory = json!({"before":before,"first_loaded":loaded,"first_dropped":after_drop,"after_trials":memory()?,"host":host_memory::host()?});
    let model = load(&args[1], &b)?;
    let (tokens, codes) = data(1, model.config().levels());
    let (cold, first_execution) = timed(|| {
        call(
            &model,
            "reconstruct",
            &tokens,
            &codes,
            1,
            &model.weights()["encoder.wte.weight"],
        )
    })?;
    drop(cold);
    let mut rows = Vec::new();
    let mut stages = Vec::new();
    for batch in [1, 2, 4] {
        let (tokens, codes) = data(batch, model.config().levels());
        let encoded = model.encode(&tokens, batch)?;
        let latent = encoded.snapshots.last().ok_or("missing latent")?.clone();
        b.evaluate(std::slice::from_ref(&latent))?;
        drop(encoded);
        for scope in ["encode", "decode", "reconstruct", "resident_decoder_block"] {
            memory::reset_peak_memory()?;
            let (first, first_ms) = timed(|| call(&model, scope, &tokens, &codes, batch, &latent))?;
            let hash = ids_hash(&first.ids);
            let count = first.ids.len();
            let floats: usize = first
                .floats
                .iter()
                .map(|x| b.shape(x).iter().product::<usize>())
                .sum();
            let logits = first.logits.as_ref().map(|x| b.read_f32(x)).transpose()?;
            drop(first);
            let mut wall = Vec::new();
            for i in 0..35 {
                let (output, ms) = timed(|| call(&model, scope, &tokens, &codes, batch, &latent))?;
                if ids_hash(&output.ids) != hash {
                    return Err("nonrepeatable IDs".into());
                }
                if i >= 5 {
                    wall.push(ms);
                }
                drop(output);
            }
            rows.push(json!({"batch":batch,"scope":scope,"first_ms":first_ms,"trials_ms":wall,"output_ids_sha256":hash,"output_ids_count":count,"output_float_elements":floats,
                "input_ids_sha256":ids_hash(&tokens),"supplied_codes_sha256":ids_hash(&codes),"first_logits":logits,"allocator_bytes":memory()?,"host":host_memory::host()?}));
        }
        stages.push(json!({"batch":batch,"synchronized_stage_ms":profile(&model,&tokens,batch)?}));
    }
    let paths = [
        "crates/ncm-backend-mlx/examples/mlx_codec_bench.rs",
        "crates/ncm-backend-mlx/src/codec.rs",
        "crates/ncm-backend-mlx/src/ops.rs",
        "crates/ncm-core/src/config.rs",
        "crates/ncm-model/src/lib.rs",
        "crates/ncm-model/src/codec.rs",
        "crates/ncm-model/src/blocks.rs",
        "crates/ncm-io/src/lib.rs",
        "tests/support/host_memory.rs",
        "Cargo.lock",
    ];
    let sources: BTreeMap<_, _> = paths
        .into_iter()
        .map(|p| Ok((p, sha256(&fs::read(p)?))))
        .collect::<Result<_>>()?;
    let report = json!({"schema_version":1,"runtime":"rust-mlx","device":"metal","precision":"fp32","index_dtype":"u32","warmups":5,"trials":30,"workload_sha256":sha256(&raw),"manifest_sha256":sha256(&manifest),"source_sha256":sources,
        "first_load_ms":first_load,"load_trials_ms":loads,"load_memory":load_memory,"first_execution_ms":first_execution,"measurements":rows,"instrumented_stages":stages,"host_final":host_memory::host()?,
        "timing_scope":"synchronous graph construction and evaluation; resident middle block excludes host copies, not kernel-only time; output destruction excluded",
        "stage_scope":"separate synchronized barriers perturb execution; includes final selection host readback; not additive headline timings",
        "memory_scope":"MLX active/peak/cache unified-memory allocator; process RSS via ps and HWM via Darwin getrusage; not separate VRAM",
        "cold_disk_cache":false,"context_initialization_timed":false,"compile_profile":"eager MLX compositions; no graph compilation"});
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?,
        &report,
    )?;
    Ok(())
}
