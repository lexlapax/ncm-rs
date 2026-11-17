//! Host-only LM1B-small preparation inspection; never creates a backend.
use ncm_io::sha256;
use ncm_model::{Lm1bSmallCodecInspection, Lm1bSmallGeneratorInspection};
use std::{error::Error, fs, io::Read, path::Path};

fn read_checked(path: &Path, maximum: u64, expected: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.file_type().is_file() || meta.len() > maximum {
        return Err("invalid file type or size".into());
    }
    let capacity = usize::try_from(meta.len() + 1)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity)?;
    fs::File::open(path)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != meta.len()
        || bytes.len() as u64 > maximum
        || sha256(&bytes) != expected
    {
        return Err("file identity differs".into());
    }
    Ok(bytes)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: lm1b_small_preparation_probe DIRECTORY".into());
    }
    let root = Path::new(&args[1]);
    if !fs::symlink_metadata(root)?.file_type().is_dir() {
        return Err("invalid root directory".into());
    }
    read_checked(
        &root.join("manifest.json"),
        4 * 1024 * 1024,
        "82ebe150ef1fc848b3dc4520c38342b9e44021e2eb219aba2b6eab5c4ce48ed1",
    )?;
    let config = read_checked(
        &root.join("config.json"),
        1024 * 1024,
        "1bc5e0c10050c8221ec72c277d8510a9a541117db5705cb12173f12b733a1475",
    )?;
    let generator = read_checked(
        &root.join("generator.safetensors"),
        448 * 1024 * 1024,
        "44dd32ca0cbf3b61e961e81f9e19af60ad7e768f900bf687cc409abfea90add2",
    )?;
    let inference = read_checked(
        &root.join("inference.safetensors"),
        1024 * 1024,
        "d6e38383177c8fcf039ceca8f7031acd11a1cf2c686028bcde4c9956dd5bc640",
    )?;
    let result =
        Lm1bSmallGeneratorInspection::from_preparation_bytes(&config, &generator, &inference)?;
    drop(generator);
    let codec = read_checked(
        &root.join("codec.safetensors"),
        448 * 1024 * 1024,
        "934248e4cdb056f0da9cbe50fb94107eef82eab6d39991e6eabb6de08a04429e",
    )?;
    let codec_result = Lm1bSmallCodecInspection::from_preparation_bytes(&config, &codec)?;
    println!(
        "{}",
        serde_json::json!({"scope":"host generator and codec preparation preflight only; no runtime or GPU qualification", "generator_tensors":result.generator_tensors, "inference_tensors":result.inference_tensors, "codec_tensors":codec_result.codec_tensors})
    );
    Ok(())
}
