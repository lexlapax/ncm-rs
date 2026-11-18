//! Capture each canonical WMT14 batch with one loaded model and retain failures.
use ncm_io::sha256;
use std::{error::Error, fs, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const PREFIX_SHA: &str = "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f";
const ARTIFACT_SHA: &str = "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81";
const ROWS: usize = 3003;
const BATCH: usize = 4;

#[cfg(any(
    feature = "cuda-cudnn",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
fn capture<B: ncm_core::GeneratorBackend + Clone>(
    backend: B,
    artifact: &str,
    output: &str,
    rows: &[Vec<u32>],
    first: usize,
    guided: bool,
    padded_retry: bool,
) -> Result<()> {
    use std::io::{LineWriter, Write};

    let model = if padded_retry {
        let artifact = ncm_io::Artifact::load_wmt14_inference_reusable(artifact)?;
        ncm_model::Wmt14Teacher::from_reusable_artifact(backend.clone(), artifact)?
            .with_deferred_head_check_experimental()
    } else {
        let artifact = ncm_io::Artifact::load_inference(artifact)?;
        ncm_model::Wmt14Teacher::from_owned_artifact(backend.clone(), artifact)?
    };
    let mut output = LineWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    let mut failed = 0;
    for (index, batch) in rows.chunks(BATCH).enumerate() {
        let start = first + BATCH * index;
        let prefixes: Vec<_> = batch.iter().flatten().copied().collect();
        let generated = if padded_retry {
            model.generate_top1_with_padded_retry(&prefixes, batch.len(), f64::from(guided))
        } else {
            let result = if guided {
                model.generate_cfg_top1(&prefixes, batch.len(), 1.0)
            } else {
                model.generate_top1(&prefixes, batch.len())
            };
            result.map(|output| (output, false))
        };
        let record = match generated {
            Ok((generated, retried)) => {
                drop(generated.logits);
                let codes = backend.read_indices(&generated.codes)?;
                let ids = backend.read_indices(&generated.ids)?;
                if codes.len() != batch.len() * 313 || ids.len() != batch.len() * 64 {
                    return Err("unexpected WMT14 output shape".into());
                }
                let codes = codes.as_chunks::<313>().0;
                let ids = ids.as_chunks::<64>().0;
                let codes: Vec<_> = codes.iter().map(<[u32; 313]>::as_slice).collect();
                let ids: Vec<_> = ids.iter().map(<[u32; 64]>::as_slice).collect();
                let mut record = serde_json::json!({"start": start, "count": batch.len(), "codes": codes, "raw_ids": ids});
                if retried {
                    eprintln!("batch {start}: recovered rejected head with eight internal rows");
                    record["internal_retry_batch"] = serde_json::json!(8);
                }
                record
            }
            Err(error) => {
                failed += 1;
                eprintln!("batch {start}: {error}");
                serde_json::json!({"start": start, "count": batch.len(), "error": error.to_string()})
            }
        };
        serde_json::to_writer(&mut output, &record)?;
        writeln!(&mut output)?;
        if index.is_multiple_of(64) || start + batch.len() == first + rows.len() {
            eprintln!(
                "captured {}/{} rows",
                start + batch.len() - first,
                rows.len()
            );
        }
    }
    output.flush()?;
    if failed > 0 {
        return Err(format!("{failed} WMT14 batches failed; capture retained").into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact, device, prefixes, output, scale, first, count] = args.as_slice() else {
        return Err(
            "usage: wmt14_corpus_capture ARTIFACT cuda|mlx|mlx-upsample|mlx-retry-padded PREFIXES OUTPUT 0|1 START COUNT".into(),
        );
    };
    let guided = match scale.as_str() {
        "0" => false,
        "1" => true,
        _ => return Err("expected CFG 0 or 1".into()),
    };
    let first: usize = first.parse()?;
    let count: usize = count.parse()?;
    let end = first.checked_add(count).ok_or("row count overflow")?;
    if !first.is_multiple_of(BATCH)
        || count == 0
        || end > ROWS
        || (end != ROWS && !end.is_multiple_of(BATCH))
    {
        return Err("expected complete aligned WMT14 batches".into());
    }
    let data = fs::read(prefixes)?;
    if sha256(&data) != PREFIX_SHA
        || sha256(&fs::read(Path::new(artifact).join("manifest.json"))?) != ARTIFACT_SHA
    {
        return Err("unexpected WMT14 fixture or artifact".into());
    }
    let doc: serde_json::Value = serde_json::from_slice(&data)?;
    let rows: Vec<Vec<u32>> =
        serde_json::from_value(doc.get("prefixes").ok_or("missing prefixes")?.clone())?;
    if rows.len() != ROWS || rows.iter().any(|row| row.len() != 96) {
        return Err("unexpected WMT14 fixture shape".into());
    }
    let _ = (guided, output);
    match device.as_str() {
        #[cfg(feature = "cuda-cudnn")]
        "cuda" => capture(
            ncm_backend_cuda::CandleBackend::cuda_wmt14_fast_argmax_experimental(0)?,
            artifact,
            output,
            &rows[first..end],
            first,
            guided,
            false,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => capture(
            ncm_backend_mlx::MlxBackend::gpu()?,
            artifact,
            output,
            &rows[first..end],
            first,
            guided,
            false,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx-upsample" => capture(
            ncm_backend_mlx::MlxBackend::gpu_wmt14_upsample_experimental()?,
            artifact,
            output,
            &rows[first..end],
            first,
            guided,
            false,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx-retry-padded" => capture(
            ncm_backend_mlx::MlxBackend::gpu()?,
            artifact,
            output,
            &rows[first..end],
            first,
            guided,
            true,
        ),
        _ => Err("requested device is unavailable".into()),
    }
}
