//! Confirm that artifact loading does not imply an implemented LM1B model.
use ncm_backend_cuda::CandleBackend;
use ncm_io::Artifact;
use ncm_model::Codec;
use std::{env, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        return Err("usage: lm1b_small_unsupported_probe ARTIFACT".into());
    }
    let artifact = Artifact::load_inference(&args[1])?;
    match Codec::from_artifact(CandleBackend::cpu(), &artifact) {
        Ok(_) => Err("LM1B model unexpectedly accepted".into()),
        Err(error)
            if error
                .to_string()
                .contains("LM1B-small codec execution is unsupported") =>
        {
            println!("LM1B model rejected explicitly: {error}");
            Ok(())
        }
        Err(error) => Err(format!("unexpected rejection: {error}").into()),
    }
}
