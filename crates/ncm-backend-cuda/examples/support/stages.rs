//! Instrumented stream spans, separate from uninstrumented headline timings.
use candle_core::{
    Tensor,
    cuda_backend::cudarc::driver::{CudaEvent, sys},
};
use ncm_backend_cuda::CandleBackend;
use ncm_core::{CodecBackend, Error, Result};
use ncm_model::{Codec, TraceSink};
use std::collections::BTreeMap;

struct Events<'a> {
    backend: &'a CandleBackend,
    starts: BTreeMap<String, CudaEvent>,
    pairs: BTreeMap<String, (CudaEvent, CudaEvent)>,
}
impl Events<'_> {
    fn event(&self) -> Result<CudaEvent> {
        self.backend
            .device()
            .as_cuda_device()
            .map_err(|e| Error::Backend(e.to_string()))?
            .cuda_stream()
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| Error::Backend(e.to_string()))
    }
    fn start(&mut self, name: &str) -> Result<()> {
        let event = self.event()?;
        if self.starts.insert(name.into(), event).is_some() {
            return Err(Error::Input("duplicate stage start".into()));
        }
        Ok(())
    }
    fn end(&mut self, name: &str) -> Result<()> {
        let start = self
            .starts
            .remove(name)
            .ok_or_else(|| Error::Input("missing stage start".into()))?;
        let end = self.event()?;
        self.pairs.insert(name.into(), (start, end));
        Ok(())
    }
}
impl TraceSink<CandleBackend> for Events<'_> {
    fn tensor(&mut self, name: &str, _: &Tensor) -> Result<()> {
        if name == "encode.pre_quant" {
            self.end("encoder_pre_quant")?;
        }
        if name == "decode.decoder.input" {
            self.start("decoder_logits")?;
        }
        if name == "decode.logits" {
            self.end("decoder_logits")?;
            self.start("output_selection")?;
        }
        if let Some(level) = name
            .strip_suffix(".residual_before")
            .filter(|s| s.starts_with("encode.level_"))
        {
            self.start(level)?;
        }
        if let Some(level) = name
            .strip_suffix(".contribution")
            .filter(|s| s.starts_with("encode.level_"))
        {
            self.end(level)?;
        }
        Ok(())
    }
    fn indices(&mut self, name: &str, _: &Tensor) -> Result<()> {
        if name == "encode.encoder.input" {
            self.start("encoder_pre_quant")?;
        }
        if name == "decode.ids" {
            self.end("output_selection")?;
        }
        Ok(())
    }
}
pub fn profile(
    model: &Codec<CandleBackend>,
    tokens: &[u32],
    batch: usize,
) -> Result<BTreeMap<String, Vec<f64>>> {
    let b = model.backend();
    let mut trials = BTreeMap::<String, Vec<f64>>::new();
    for i in 0..35 {
        b.evaluate(&[])?;
        let mut events = Events {
            backend: b,
            starts: BTreeMap::new(),
            pairs: BTreeMap::new(),
        };
        let output = model.reconstruct_traced(tokens, batch, &mut events)?;
        b.evaluate(&output.1.snapshots)?;
        if !events.starts.is_empty() || events.pairs.len() != 15 {
            return Err(Error::Input("incomplete stage trace".into()));
        }
        for (name, (start, end)) in events.pairs {
            let ms = f64::from(
                start
                    .elapsed_ms(&end)
                    .map_err(|e| Error::Backend(e.to_string()))?,
            );
            if i >= 5 {
                trials.entry(name).or_default().push(ms);
            }
        }
        drop(output);
    }
    Ok(trials)
}
