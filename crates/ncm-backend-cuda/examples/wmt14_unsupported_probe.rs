//! Confirm that artifact inspection does not imply an implemented WMT14 model.
use ncm_backend_cuda::CandleBackend;
use ncm_io::Artifact;
use ncm_model::Codec;
use std::{env, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        return Err("usage: wmt14_unsupported_probe ARTIFACT".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    match Codec::from_artifact(CandleBackend::cpu(), &artifact) {
        Ok(_) => Err("WMT14 model unexpectedly accepted".into()),
        Err(error) => {
            println!("WMT14 model rejected explicitly: {error}");
            Ok(())
        }
    }
}
