//! Report actual FP32 payload alignment after checked XSum artifact loading.
use ncm_io::Artifact;
use safetensors::{Dtype, SafeTensors};
use std::{env, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        return Err("usage: xsum_payload_alignment ARTIFACT".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    for name in ["codec.safetensors", "generator.safetensors"] {
        let tensors = SafeTensors::deserialize(artifact.file(name)?)?;
        let (mut count, mut aligned, mut bytes, mut aligned_bytes) = (0, 0, 0, 0);
        for (_, tensor) in tensors.tensors() {
            if tensor.dtype() == Dtype::F32 {
                count += 1;
                bytes += tensor.data().len();
                if (tensor.data().as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
                    aligned += 1;
                    aligned_bytes += tensor.data().len();
                }
            }
        }
        println!(
            "{name}: {aligned}/{count} aligned F32 tensors, {aligned_bytes}/{bytes} aligned bytes"
        );
    }
    Ok(())
}
