//! Export sampled TinyStories IDs and text for an explicit offline comparison.
use ncm_backend_cuda::CandleBackend;
use ncm_core::CodecBackend;
use ncm_io::Artifact;
use ncm_model::TinyStoriesTeacher;
use ncm_tokenizers::TextTokenizer;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: tinystories_samples ARTIFACT COUNT BATCH SEED".into());
    }
    let count: usize = args[2].parse()?;
    let batch: usize = args[3].parse()?;
    let seed: u64 = args[4].parse()?;
    if !(1..=256).contains(&count) || !(1..=128).contains(&batch) {
        return Err("invalid sample count or batch".into());
    }
    let artifact = Artifact::load(&args[1])?;
    let tokenizer = TextTokenizer::from_asset_bytes(
        artifact.file("tokenizer-manifest.json")?,
        artifact.file("tokenizer.json")?,
        artifact.file("tokenizer.LICENSE")?,
    )?;
    #[cfg(feature = "cuda-cudnn")]
    let backend = CandleBackend::cuda_cudnn(0)?;
    #[cfg(not(feature = "cuda-cudnn"))]
    let backend = CandleBackend::cuda(0)?;
    let model = TinyStoriesTeacher::from_artifact(backend.clone(), &artifact)?;
    drop(artifact);
    let mut all_codes = Vec::with_capacity(count);
    let mut all_ids = Vec::with_capacity(count);
    let mut texts = Vec::with_capacity(count);
    let mut tokenized_texts = Vec::with_capacity(count);
    let mut offset = 0;
    while offset < count {
        let size = batch.min(count - offset);
        let batch_seed = seed ^ ((offset / batch) as u64).wrapping_mul(0x9e3779b97f4a7c15);
        let result = model.sample_seeded(&vec![0; size], size, batch_seed)?;
        let codes = backend.read_indices(&result.codes)?;
        let ids = backend.read_indices(&result.ids)?;
        if codes.len() != size * 641 || ids.len() != size * 128 {
            return Err("invalid generated output shape".into());
        }
        for (c, row) in codes
            .as_chunks::<641>()
            .0
            .iter()
            .zip(ids.as_chunks::<128>().0.iter())
        {
            let end = row.iter().position(|&id| id == 0).unwrap_or(row.len());
            let text = tokenizer.decode(&row[..end], true)?;
            tokenized_texts.push(tokenizer.encode(&text)?);
            texts.push(text);
            all_codes.push(c.to_vec());
            all_ids.push(row.to_vec());
        }
        offset += size;
    }
    println!(
        "{}",
        serde_json::json!({"seed":seed,"batch":batch,"rng":"SplitMix64 inverse CDF; chunk seed = seed XOR (chunk index * 0x9e3779b97f4a7c15)", "codes":all_codes, "raw_ids":all_ids,"texts":texts,"toks":tokenized_texts})
    );
    Ok(())
}
