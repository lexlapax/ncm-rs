//! Offline native capture against authenticated Sudoku inputs; no Python runtime.
#[path = "codec_capture.rs"]
mod capture;
use capture::Captured;
use ncm_core::CodecBackend;
use ncm_io::{Artifact, sha256};
use ncm_model::{Codec, TraceSink};
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, serialize},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, path::Path};

struct Observation {
    shape: Vec<usize>,
    dtype: Dtype,
    bytes: Vec<u8>,
}
fn indices(t: TensorView<'_>) -> Result<Vec<u32>, Box<dyn Error>> {
    if t.dtype() != Dtype::I64 {
        return Err("reference indices must be I64".into());
    }
    t.data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| Ok(u32::try_from(i64::from_le_bytes(*b))?))
        .collect()
}
pub fn run<B: CodecBackend>(
    artifact_root: &str,
    reference: &Path,
    output: &Path,
    runtime: &str,
    device: &str,
    backend: impl FnOnce() -> ncm_core::Result<B>,
    adapter_sources: &[&str],
) -> Result<(), Box<dyn Error>> {
    if output.exists() {
        return Err("output must not exist".into());
    }
    let reference_report = fs::read(reference.join("report.json"))?;
    if reference_report
        != include_bytes!("../../docs/results/2026-11-10-phase2-oracle/raw/sudoku-cuda.json")
    {
        return Err("reference report differs from committed trained CUDA oracle".into());
    }
    let report: Value = serde_json::from_slice(&reference_report)?;
    let reference_data = fs::read(reference.join("traces.safetensors"))?;
    if sha256(&reference_data)
        != report["trace_sha256"]
            .as_str()
            .ok_or("missing reference hash")?
    {
        return Err("reference trace hash mismatch".into());
    }
    let reference_tensors = SafeTensors::deserialize(&reference_data)?;
    let artifact = Artifact::load(artifact_root)?;
    let anchor: Value = serde_json::from_str(include_str!(
        "../../docs/results/2026-11-10-sudoku-artifacts/raw/artifact-manifest.json"
    ))?;
    for name in ["config.json", "codec.safetensors"] {
        if sha256(artifact.file(name)?)
            != anchor["files"][name]["sha256"]
                .as_str()
                .ok_or("missing anchor")?
        {
            return Err(format!("untrusted artifact {name}").into());
        }
    }
    let config_bytes = fs::read(reference.join("config.json"))?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    let original: Value = serde_json::from_slice(artifact.file("config.json")?)?;
    if original["tokenizer"]["vqvae"] != config
        || sha256(&config_bytes)
            != report["config_sha256"]
                .as_str()
                .ok_or("missing config hash")?
    {
        return Err("reference/runtime config differs".into());
    }
    let codec = Codec::from_artifact(backend()?, &artifact)?;
    let b = codec.backend();
    let mut weights = BTreeMap::new();
    for (name, tensor) in codec.weights() {
        let bytes: Vec<u8> = b
            .read_f32(tensor)?
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        weights.insert(
            name.clone(),
            json!({"shape": b.shape(tensor),"sha256":sha256(&bytes)}),
        );
    }
    if serde_json::to_value(&weights)? != report["weights"] {
        return Err("native runtime weights differ".into());
    }
    let mut all = BTreeMap::<String, Observation>::new();
    for case in report["cases"].as_array().ok_or("missing cases")? {
        let name = case["name"].as_str().ok_or("missing case name")?;
        // Only IDs are read from the reference; all output/intermediate tensors
        // below are executed by the shared native codec, never copied goldens.
        let input = reference_tensors.tensor(&format!("{name}/input.ids"))?;
        let supplied = reference_tensors.tensor(&format!("{name}/supplied.codes"))?;
        let batch = input.shape()[0];
        let tokens = indices(input.clone())?;
        let codes = indices(supplied.clone())?;
        let mut captured = Captured::<B>::default();
        captured.indices("input.ids", &b.upload_indices(&tokens, input.shape())?)?;
        captured.indices(
            "supplied.codes",
            &b.upload_indices(&codes, supplied.shape())?,
        )?;
        let (encoded, decoded) = codec.reconstruct_traced(&tokens, batch, &mut captured)?;
        let forced = codec.decode_traced(&codes, batch, "forced", &mut captured)?;
        for (prefix, snapshots) in [
            ("encode", encoded.snapshots),
            ("decode", decoded.snapshots),
            ("forced", forced.snapshots),
        ] {
            for (level, snapshot) in snapshots.iter().enumerate() {
                let traced = &captured.tensors[&format!("{prefix}.level_{level:02}.cumulative")];
                if b.read_f32(snapshot)? != b.read_f32(traced)? {
                    return Err("snapshot changed after subsequent execution".into());
                }
            }
        }
        let (plain_codes, plain_decode) = codec.reconstruct(&tokens, batch)?;
        let plain_forced = codec.decode(&codes, batch)?;
        if b.read_indices(&plain_codes.codes)?
            != b.read_indices(&captured.indices["encode.codes"])?
        {
            return Err("uninstrumented code IDs differ".into());
        }
        // Optimized reconstruction reuses immutable encode storage. Check every
        // returned snapshot after another decode, not only final logits/IDs.
        for (prefix, snapshots) in [
            ("encode", &plain_codes.snapshots),
            ("decode", &plain_decode.snapshots),
            ("forced", &plain_forced.snapshots),
        ] {
            if snapshots.len() != codec.config().levels().len() {
                return Err("uninstrumented snapshot count differs".into());
            }
            for (level, snapshot) in snapshots.iter().enumerate() {
                let name = format!("{prefix}.level_{level:02}.cumulative");
                if b.read_f32(snapshot)? != b.read_f32(&captured.tensors[&name])? {
                    return Err("uninstrumented snapshot differs after subsequent decode".into());
                }
            }
        }
        for (label, plain) in [("decode", plain_decode), ("forced", plain_forced)] {
            if b.read_f32(&plain.logits)?
                != b.read_f32(&captured.tensors[&format!("{label}.logits")])?
                || b.read_indices(&plain.ids)?
                    != b.read_indices(&captured.indices[&format!("{label}.ids")])?
            {
                return Err("uninstrumented decoder output differs".into());
            }
        }
        for (key, tensor) in captured.tensors {
            let data = b.read_f32(&tensor)?;
            if data.iter().any(|x| !x.is_finite()) {
                return Err(format!("nonfinite {name}/{key}").into());
            }
            all.insert(
                format!("{name}/{key}"),
                Observation {
                    shape: b.shape(&tensor),
                    dtype: Dtype::F32,
                    bytes: data.into_iter().flat_map(f32::to_le_bytes).collect(),
                },
            );
        }
        for (key, tensor) in captured.indices {
            all.insert(
                format!("{name}/{key}"),
                Observation {
                    shape: b.indices_shape(&tensor),
                    dtype: Dtype::I64,
                    bytes: b
                        .read_indices(&tensor)?
                        .into_iter()
                        .flat_map(|x| i64::from(x).to_le_bytes())
                        .collect(),
                },
            );
        }
    }
    let serialized = serialize(
        all.iter()
            .map(|(name, o)| Ok((name, TensorView::new(o.dtype, o.shape.clone(), &o.bytes)?)))
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?,
        None,
    )?;
    let sources: BTreeMap<_, _> = [
        "crates/ncm-core/src/lib.rs",
        "crates/ncm-core/src/config.rs",
        "crates/ncm-model/src/lib.rs",
        "crates/ncm-model/src/blocks.rs",
        "crates/ncm-model/src/codec.rs",
        "tests/support/trained_codec.rs",
        "tests/support/codec_capture.rs",
        "Cargo.lock",
    ]
    .into_iter()
    .chain(adapter_sources.iter().copied())
    .map(|p| Ok((p, sha256(&fs::read(p)?))))
    .collect::<Result<_, std::io::Error>>()?;
    fs::create_dir_all(output)?;
    fs::write(output.join("traces.safetensors"), &serialized)?;
    fs::write(output.join("config.json"), config_bytes)?;
    let result = json!({"schema_version":1,"profile":"sudoku","runtime":runtime,"device":device,
        "upstream_commit": ncm_io::UPSTREAM,"policy":report["policy"],"config_sha256": report["config_sha256"],
        "weights":weights,"trace_sha256":sha256(&serialized),"trace_count":all.len(),"source_sha256":sources,
        "reference_report_sha256":sha256(&reference_report),"performance":null,
        "snapshot_check":"retained API snapshots equal independently retained observations after subsequent decode",
        "uninstrumented_matches_traced":true});
    fs::write(
        output.join("report.json"),
        format!("{}\n", serde_json::to_string_pretty(&result)?),
    )?;
    // Keep the shared comparison policy linked into this capture binary.
    assert_eq!(capture::tolerance("decode.logits"), (5e-4, 2e-4));
    println!(
        "{} traces captured on {}; parity must be checked separately",
        all.len(),
        device
    );
    Ok(())
}
