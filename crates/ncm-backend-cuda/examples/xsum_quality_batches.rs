//! Experimental batch stream for evaluating XSum quality on strict CUDA.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::XsumTeacher;
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: xsum_quality_batches ARTIFACT".into());
    }
    let backend = CandleBackend::cuda_cudnn(0)?;
    let artifact = Artifact::load_inference(&args[1])?;
    let model = XsumTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.len() > 256 * 1024 {
            return Err("batch input exceeds size limit".into());
        }
        let rows: Vec<Vec<u32>> = serde_json::from_str(&line)?;
        if rows.is_empty() || rows.len() > 32 || rows.iter().any(|row| row.len() != 512) {
            return Err("expected 1..=32 rows of 512 tokens".into());
        }
        let batch = rows.len();
        let input: Vec<u32> = rows.into_iter().flatten().collect();
        let result = model.generate_top1_device(&input, batch)?;
        let codes = backend.read_indices(&result.codes)?;
        if codes.len() != batch * 200 || result.ids.len() != batch * 48 {
            return Err("unexpected output dimensions".into());
        }
        serde_json::to_writer(
            &mut stdout,
            &serde_json::json!({"codes": codes.as_chunks::<200>().0.iter().map(|row| row.as_slice()).collect::<Vec<_>>(),
                               "raw_ids": result.ids.as_chunks::<48>().0.iter().map(|row| row.as_slice()).collect::<Vec<_>>() }),
        )?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
