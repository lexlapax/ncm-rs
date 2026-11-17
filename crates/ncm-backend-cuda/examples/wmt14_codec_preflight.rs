//! Validate the pinned WMT14 codec configuration and tensor metadata on CPU.
use ncm_core::{CodecConfig, RawCodecConfig};
use ncm_io::{Artifact, sha256};
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeSet, env, error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        return Err("usage: wmt14_codec_preflight ARTIFACT".into());
    }
    if sha256(&fs::read(Path::new(&args[1]).join("manifest.json"))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected WMT14 manifest".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    if artifact.manifest().profile != "wmt14-inference-v1" {
        return Err("unexpected artifact profile".into());
    }
    let envelope: serde_json::Value = serde_json::from_slice(artifact.file("config.json")?)?;
    let raw: RawCodecConfig = serde_json::from_value(envelope["tokenizer"]["vqvae"].clone())?;
    let config = CodecConfig::validate_wmt14(raw)?;
    let expected = config.weight_shapes();
    let tensors = SafeTensors::deserialize(artifact.file("codec.safetensors")?)?;
    if expected.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != tensors.names().into_iter().collect()
    {
        return Err("WMT14 codec tensor names differ".into());
    }
    let mut bytes = 0usize;
    for (name, shape) in &expected {
        let tensor = tensors.tensor(name)?;
        let (words, remainder) = tensor.data().as_chunks::<4>();
        if tensor.dtype() != Dtype::F32
            || tensor.shape() != shape
            || !remainder.is_empty()
            || words
                .iter()
                .any(|word| u32::from_le_bytes(*word) & 0x7f80_0000 == 0x7f80_0000)
        {
            return Err(format!("invalid WMT14 codec tensor: {name}").into());
        }
        bytes += tensor.data().len();
    }
    if expected.len() != 63 || bytes != 576_011_264 || config.code_length() != 313 {
        return Err("unexpected WMT14 codec inventory".into());
    }
    println!(
        "{}",
        serde_json::json!({"tensors":expected.len(),"bytes":bytes,"code_length":config.code_length(),
        "finite":true,"scope":"CPU config and tensor preflight only; no model or GPU execution"})
    );
    Ok(())
}
