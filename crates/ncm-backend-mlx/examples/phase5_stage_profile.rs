//! Phase 5 Step 2 Apple Silicon / Metal stage-breakdown profiler (timing, host RSS, and MLX peaks).
//!
//! Synchronizing between stages forces MLX graph materialization per stage; an uninstrumented warm
//! pass in the same process reports end-to-end latency without inter-stage barriers.
#[path = "../../../tests/support/host_memory.rs"]
mod host_memory;

use mlx_rs::memory;
use ncm_backend_mlx::{MlxBackend, MlxTensor};
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::{
    Generator, Lm1bSmallTeacher, TinyStoriesTeacher, TraceSink, Wmt14Teacher, XsumTeacher,
    lm1b_small_reference_rope,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn snap() -> Result<Value> {
    Ok(json!({
        "host": host_memory::host()?,
        "mlx_memory_bytes": {
            "active": memory::active_memory()?,
            "cache": memory::cache_memory()?,
            "peak": memory::peak_memory()?,
        },
    }))
}

struct StageRecorder {
    backend: MlxBackend,
    last: Instant,
    events: Vec<(String, f64)>,
    record_mem: bool,
    mem_events: Vec<Value>,
}

impl StageRecorder {
    fn new(backend: MlxBackend, record_mem: bool) -> Self {
        Self {
            backend,
            last: Instant::now(),
            events: Vec::new(),
            record_mem,
            mem_events: Vec::new(),
        }
    }
    fn mark_eval(&mut self, name: &str, value: Option<&MlxTensor>) -> ncm_core::Result<()> {
        if let Some(t) = value {
            self.backend.eval_tensor(t)?;
        }
        let now = Instant::now();
        let ms = now.duration_since(self.last).as_secs_f64() * 1000.0;
        self.events.push((name.to_owned(), ms));
        if self.record_mem {
            let mem = snap().map_err(|e| ncm_core::Error::Backend(e.to_string()))?;
            self.mem_events.push(json!({"stage": name, "memory": mem}));
        }
        self.last = Instant::now();
        Ok(())
    }
}

impl TraceSink<MlxBackend> for StageRecorder {
    fn tensor(&mut self, _: &str, _: &MlxTensor) -> ncm_core::Result<()> {
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &MlxTensor) -> ncm_core::Result<()> {
        Ok(())
    }
    fn stage_tensor(&mut self, name: &str, value: &MlxTensor) -> ncm_core::Result<()> {
        self.mark_eval(name, Some(value))
    }
    fn wants_trace(&self) -> bool {
        false
    }
    fn stage_indices(&mut self, name: &str, value: &MlxTensor) -> ncm_core::Result<()> {
        self.mark_eval(name, Some(value))
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

fn summarize_events(trials: &[Vec<(String, f64)>]) -> Value {
    let mut prefix_ms = Vec::new();
    let mut levels: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut sampling_ms = Vec::new();
    let mut penalties_ms = Vec::new();
    let mut codec_ms = Vec::new();
    let mut readback_ms = Vec::new();
    let mut totals = Vec::new();
    for trial in trials {
        let mut p = 0.0;
        let mut s = 0.0;
        let mut pen = 0.0;
        let mut c = 0.0;
        let mut r = 0.0;
        let mut tot = 0.0;
        for (name, ms) in trial {
            tot += *ms;
            if name == "prefix_done" {
                p += *ms;
            } else if name.ends_with("_logits") {
                levels.entry(name.clone()).or_default().push(*ms);
            } else if name.ends_with("_sampled") {
                s += *ms;
            } else if name == "codec_decode_done" {
                c += *ms;
            } else if name == "penalties_done" {
                pen += *ms;
            } else if name == "host_readback_done" {
                r += *ms;
            }
        }
        prefix_ms.push(p);
        sampling_ms.push(s);
        penalties_ms.push(pen);
        codec_ms.push(c);
        readback_ms.push(r);
        totals.push(tot);
    }
    let level_medians: Vec<Value> = levels
        .into_iter()
        .map(|(k, v)| json!({"level": k, "median_ms": median(v)}))
        .collect();
    let gen_sum: f64 = level_medians
        .iter()
        .map(|x| x["median_ms"].as_f64().unwrap_or(0.0))
        .sum();
    let samp_med = median(sampling_ms);
    let pen_med = median(penalties_ms);
    json!({
        "prefix_prepare_ms": median(prefix_ms),
        "generator_levels_sum_ms": gen_sum,
        "per_level_generator_ms": level_medians,
        "per_level_sampling_ms": samp_med,
        "final_penalties_ms": pen_med,
        "sampling_and_penalties_ms": samp_med + pen_med,
        "codec_decode_ms": median(codec_ms),
        "host_readback_ms": median(readback_ms),
        "instrumented_total_ms": median(totals),
    })
}

fn load_sudoku_inputs(path: &str, batch: usize) -> Result<Vec<u32>> {
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&fs::read(path)?)?;
    Ok(rows[..batch].iter().flatten().copied().collect())
}

fn load_xsum_inputs(path: &str, batch: usize) -> Result<Vec<u32>> {
    let doc: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = doc["inputs"]
        .as_array()
        .or_else(|| doc["ids"].as_array())
        .or_else(|| doc.as_array())
        .ok_or("invalid xsum inputs")?;
    let mut out = Vec::with_capacity(batch * 512);
    for row in &rows[..batch] {
        for x in &row.as_array().ok_or("invalid row")?[..512] {
            out.push(u32::try_from(x.as_u64().ok_or("invalid id")?)?);
        }
    }
    Ok(out)
}

fn load_wmt14_inputs(path: &str, batch: usize) -> Result<Vec<u32>> {
    let doc: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = doc["prefixes"].as_array().ok_or("missing prefixes")?;
    let mut out = Vec::with_capacity(batch * 96);
    for row in &rows[..batch] {
        for x in row.as_array().ok_or("invalid row")? {
            out.push(u32::try_from(x.as_u64().ok_or("invalid id")?)?);
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: phase5_stage_profile PROFILE ARTIFACT BATCH [INPUT_OR_ROPE]".into());
    }
    let profile = args[1].as_str();
    let artifact_path = args[2].as_str();
    let batch: usize = args[3].parse()?;
    let extra = args.get(4).map(String::as_str);

    let uninst_warmups = 3usize;
    let uninst_trials = 10usize;
    let stage_warmups = 1usize;
    let stage_trials = 5usize;

    let mut uninst_ms = Vec::with_capacity(uninst_trials);
    let mut stage_runs: Vec<Vec<(String, f64)>> = Vec::with_capacity(stage_trials);
    let (
        artifact_load_ms,
        weight_upload_ms,
        mem_after_artifact,
        mem_after_upload,
        mem_after_uninst,
        stage_mem,
    );

    let backend = MlxBackend::gpu()?;
    memory::reset_peak_memory()?;

    match profile {
        "sudoku-easy" => {
            let tokens = load_sudoku_inputs(extra.ok_or("missing sudoku input")?, batch)?;
            let t0 = Instant::now();
            let artifact = Artifact::load_inference_streaming(artifact_path)?;
            artifact_load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            mem_after_artifact = snap()?;
            let t1 = Instant::now();
            let model = Generator::from_artifact(backend.clone(), &artifact)?;
            drop(artifact);
            weight_upload_ms = t1.elapsed().as_secs_f64() * 1000.0;
            mem_after_upload = snap()?;

            for i in 0..uninst_warmups + uninst_trials {
                let s = Instant::now();
                let out = model.generate(&tokens, batch)?;
                let _c = backend.read_indices(&out.codes)?;
                let _ids = backend.read_indices(&out.ids)?;
                if i >= uninst_warmups {
                    uninst_ms.push(s.elapsed().as_secs_f64() * 1000.0);
                }
            }
            mem_after_uninst = snap()?;
            for i in 0..stage_warmups + stage_trials {
                let mut rec = StageRecorder::new(backend.clone(), false);
                let out = model.generate_traced(&tokens, batch, &mut rec)?;
                let _c = backend.read_indices(&out.codes)?;
                let _ids = backend.read_indices(&out.ids)?;
                rec.mark_eval("host_readback_done", None)?;
                if i >= stage_warmups {
                    stage_runs.push(rec.events);
                }
            }
            let mut rec = StageRecorder::new(backend.clone(), true);
            let out = model.generate_traced(&tokens, batch, &mut rec)?;
            let _c = backend.read_indices(&out.codes)?;
            let _ids = backend.read_indices(&out.ids)?;
            rec.mark_eval("host_readback_done", None)?;
            stage_mem = rec.mem_events;
        }
        "tinystories" => {
            let prefixes = vec![0_u32; batch];
            let t0 = Instant::now();
            let artifact = Artifact::load_inference_streaming(artifact_path)?;
            artifact_load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            mem_after_artifact = snap()?;
            let t1 = Instant::now();
            let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
            drop(artifact);
            weight_upload_ms = t1.elapsed().as_secs_f64() * 1000.0;
            mem_after_upload = snap()?;

            for i in 0..uninst_warmups + uninst_trials {
                let s = Instant::now();
                let out = model.sample_seeded(&prefixes, batch, 42)?;
                let _c = backend.read_indices(&out.codes)?;
                let _ids = backend.read_indices(&out.ids)?;
                if i >= uninst_warmups {
                    uninst_ms.push(s.elapsed().as_secs_f64() * 1000.0);
                }
            }
            mem_after_uninst = snap()?;
            for i in 0..stage_warmups + stage_trials {
                let mut rec = StageRecorder::new(backend.clone(), false);
                let out = model.sample_seeded_traced(&prefixes, batch, 42, &mut rec)?;
                let _c = backend.read_indices(&out.codes)?;
                let _ids = backend.read_indices(&out.ids)?;
                rec.mark_eval("host_readback_done", None)?;
                if i >= stage_warmups {
                    stage_runs.push(rec.events);
                }
            }
            let mut rec = StageRecorder::new(backend.clone(), true);
            let out = model.sample_seeded_traced(&prefixes, batch, 42, &mut rec)?;
            let _c = backend.read_indices(&out.codes)?;
            let _ids = backend.read_indices(&out.ids)?;
            rec.mark_eval("host_readback_done", None)?;
            stage_mem = rec.mem_events;
        }
        "xsum" => {
            let prefixes = load_xsum_inputs(extra.ok_or("missing xsum input")?, batch)?;
            let t0 = Instant::now();
            let artifact = Artifact::load_inference_streaming(artifact_path)?;
            artifact_load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            mem_after_artifact = snap()?;
            let t1 = Instant::now();
            let model = XsumTeacher::from_owned_artifact(backend.clone(), artifact)?
                .with_prefix_cache(true);
            weight_upload_ms = t1.elapsed().as_secs_f64() * 1000.0;
            mem_after_upload = snap()?;

            for i in 0..uninst_warmups + uninst_trials {
                let s = Instant::now();
                let out = model.generate_top1_cpu(&prefixes, batch)?;
                let _c = backend.read_indices(&out.codes)?;
                if i >= uninst_warmups {
                    uninst_ms.push(s.elapsed().as_secs_f64() * 1000.0);
                }
            }
            mem_after_uninst = snap()?;
            for i in 0..stage_warmups + stage_trials {
                let mut rec = StageRecorder::new(backend.clone(), false);
                let out = model.generate_top1_cpu_profiled(&prefixes, batch, &mut rec)?;
                let _c = backend.read_indices(&out.codes)?;
                rec.mark_eval("host_readback_done", None)?;
                if i >= stage_warmups {
                    stage_runs.push(rec.events);
                }
            }
            let mut rec = StageRecorder::new(backend.clone(), true);
            let out = model.generate_top1_cpu_profiled(&prefixes, batch, &mut rec)?;
            let _c = backend.read_indices(&out.codes)?;
            rec.mark_eval("host_readback_done", None)?;
            stage_mem = rec.mem_events;
        }
        "wmt14" => {
            let prefixes = load_wmt14_inputs(extra.ok_or("missing wmt14 input")?, batch)?;
            let t0 = Instant::now();
            let artifact = Artifact::load_inference_streaming(artifact_path)?;
            artifact_load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            mem_after_artifact = snap()?;
            let t1 = Instant::now();
            let model = Wmt14Teacher::from_owned_artifact(backend.clone(), artifact)?
                .with_prefix_cache(true)
                .with_deferred_head_check_experimental()
                .with_level_materialization_experimental(5)?;
            weight_upload_ms = t1.elapsed().as_secs_f64() * 1000.0;
            mem_after_upload = snap()?;

            for i in 0..uninst_warmups + uninst_trials {
                let s = Instant::now();
                let _out = model.generate_top1_with_tiled_decoder(&prefixes, batch, 0.0, 2)?;
                if i >= uninst_warmups {
                    uninst_ms.push(s.elapsed().as_secs_f64() * 1000.0);
                }
            }
            mem_after_uninst = snap()?;
            for i in 0..stage_warmups + stage_trials {
                let mut rec = StageRecorder::new(backend.clone(), false);
                let _out = model
                    .generate_top1_with_tiled_decoder_traced(&prefixes, batch, 0.0, 2, &mut rec)?;
                rec.mark_eval("host_readback_done", None)?;
                if i >= stage_warmups {
                    stage_runs.push(rec.events);
                }
            }
            let mut rec = StageRecorder::new(backend.clone(), true);
            let _out = model
                .generate_top1_with_tiled_decoder_traced(&prefixes, batch, 0.0, 2, &mut rec)?;
            rec.mark_eval("host_readback_done", None)?;
            stage_mem = rec.mem_events;
        }
        "lm1b-small" | "lm1b-medium" | "lm1b-large" => {
            let rope_bytes = fs::read(extra.ok_or("missing rope path")?)?;
            let (cosine, sine) = lm1b_small_reference_rope(&rope_bytes)?;
            let prefixes = vec![101_u32; batch];
            let t0 = Instant::now();
            let artifact = Artifact::load_inference_streaming(artifact_path)?;
            artifact_load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            mem_after_artifact = snap()?;
            let t1 = Instant::now();
            let model = Lm1bSmallTeacher::from_artifact(backend.clone(), &artifact)?
                .with_reference_rope_diagnostic(&cosine, &sine)?;
            drop(artifact);
            weight_upload_ms = t1.elapsed().as_secs_f64() * 1000.0;
            mem_after_upload = snap()?;

            for i in 0..uninst_warmups + uninst_trials {
                let s = Instant::now();
                let out = model.sample_seeded_device_fp32(&prefixes, batch, 42)?;
                let _c = backend.read_indices(&out.codes)?;
                if i >= uninst_warmups {
                    uninst_ms.push(s.elapsed().as_secs_f64() * 1000.0);
                }
            }
            mem_after_uninst = snap()?;
            for i in 0..stage_warmups + stage_trials {
                let mut rec = StageRecorder::new(backend.clone(), false);
                let out = model.sample_seeded_device_fp32_traced(&prefixes, batch, 42, &mut rec)?;
                let _c = backend.read_indices(&out.codes)?;
                rec.mark_eval("host_readback_done", None)?;
                if i >= stage_warmups {
                    stage_runs.push(rec.events);
                }
            }
            let mut rec = StageRecorder::new(backend.clone(), true);
            let out = model.sample_seeded_device_fp32_traced(&prefixes, batch, 42, &mut rec)?;
            let _c = backend.read_indices(&out.codes)?;
            rec.mark_eval("host_readback_done", None)?;
            stage_mem = rec.mem_events;
        }
        _ => return Err(format!("unsupported profile: {profile}").into()),
    }

    let mut sorted_u = uninst_ms.clone();
    sorted_u.sort_by(f64::total_cmp);
    let p95 = sorted_u[((0.95 * sorted_u.len() as f64).ceil() as usize).saturating_sub(1)];
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "platform": "mlx",
            "profile": profile,
            "batch": batch,
            "model_load": {
                "artifact_load_ms": artifact_load_ms,
                "weight_upload_ms": weight_upload_ms,
                "total_ms": artifact_load_ms + weight_upload_ms,
                "after_artifact_load": mem_after_artifact,
                "after_weight_upload": mem_after_upload,
            },
            "uninstrumented": {
                "warmups": uninst_warmups,
                "trials": uninst_trials,
                "median_ms": median(uninst_ms.clone()),
                "p95_ms": p95,
                "trials_ms": uninst_ms,
                "after_memory": mem_after_uninst,
            },
            "stages": summarize_events(&stage_runs),
            "stage_memory_trace": stage_mem,
        }))?
    );
    Ok(())
}
