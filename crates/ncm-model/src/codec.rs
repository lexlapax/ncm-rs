//! Shared, request-local FP32 codec execution. No generator or sampling loop.
use crate::{Codec, NoTrace, TraceSink, codebook_distances};
use ncm_core::{CodecBackend, Result};

pub struct Encoded<B: CodecBackend> {
    pub codes: B::Indices,
    pub snapshots: Vec<B::Tensor>,
}
pub struct Decoded<B: CodecBackend> {
    pub logits: B::Tensor,
    pub ids: B::Indices,
    pub snapshots: Vec<B::Tensor>,
}

/// Prefix observations without changing execution or duplicating model semantics.
struct Prefix<'a, T> {
    prefix: &'a str,
    trace: &'a mut T,
}
impl<B: CodecBackend, T: TraceSink<B>> TraceSink<B> for Prefix<'_, T> {
    fn tensor(&mut self, name: &str, value: &B::Tensor) -> Result<()> {
        self.trace.tensor(&format!("{}.{name}", self.prefix), value)
    }
    fn indices(&mut self, name: &str, value: &B::Indices) -> Result<()> {
        self.trace
            .indices(&format!("{}.{name}", self.prefix), value)
    }
}

impl<B: CodecBackend> Codec<B> {
    pub fn encode(&self, tokens: &[u32], batch: usize) -> Result<Encoded<B>> {
        self.encode_traced(tokens, batch, &mut NoTrace)
    }
    pub fn encode_traced(
        &self,
        tokens: &[u32],
        batch: usize,
        trace: &mut impl TraceSink<B>,
    ) -> Result<Encoded<B>> {
        self.config.check_tokens(tokens, batch)?;
        let b = &self.backend;
        let config = self.config.raw();
        let (length, width) = (config.context_length, config.embed_dim);
        let ids = b.upload_indices(tokens, &[batch, length])?;
        trace.indices("encode.encoder.input", &ids)?;
        let positions: Vec<u32> = (0..length as u32).collect();
        let positions = b.upload_indices(&positions, &[length])?;
        let encoder = b.add(
            &b.embedding(self.weight("encoder.wte.weight")?, &ids)?,
            &b.embedding(self.weight("encoder.wpe.weight")?, &positions)?,
        )?;
        trace.tensor("encode.encoder.output", &encoder)?;
        let mut residual = b.group_norm(&encoder, config.pre_quant_groupnorm, 1e-5)?;
        trace.tensor("encode.pre_quant", &residual)?;
        let mut cumulative = None;
        let mut snapshots = Vec::new();
        let mut codes = Vec::new();
        for (level, &level_length) in self.config.levels().iter().enumerate() {
            let name = format!("encode.level_{level:02}");
            trace.tensor(&format!("{name}.residual_before"), &residual)?;
            let terminal = level + 1 == self.config.levels().len();
            let down = if terminal {
                b.reshape(&residual, &[batch * length, width])?
            } else {
                let down = if level == 0 {
                    self.cascade(&residual, false)?
                } else {
                    b.area1d(&residual, level_length)?
                };
                b.reshape(&down, &[batch * level_length, width])?
            };
            trace.tensor(&format!("{name}.quant_input"), &down)?;
            let book = self.weight(&format!("vq.embeddings.{level}.weight"))?;
            let distances = codebook_distances(b, &down, book)?;
            trace.tensor(&format!("{name}.distances"), &distances)?;
            let ids =
                b.reshape_indices(&b.select_last(&distances, false)?, &[batch, level_length])?;
            let value = b.embedding(book, &ids)?;
            let up = self.upsample(&value, level)?;
            trace.tensor(&format!("{name}.upsampled"), &up)?;
            let contribution = self.refine(&up, level)?;
            trace.tensor(&format!("{name}.contribution"), &contribution)?;
            let sum = match cumulative {
                Some(ref previous) => b.add(previous, &contribution)?,
                None => contribution.clone(),
            };
            trace.tensor(&format!("{name}.cumulative"), &sum)?;
            residual = b.sub(&residual, &contribution)?;
            snapshots.push(sum.clone());
            cumulative = Some(sum);
            codes.push(ids);
        }
        let codes = b.concat_indices(&codes, 1)?;
        trace.indices("encode.codes", &codes)?;
        Ok(Encoded { codes, snapshots })
    }
    fn upsample(&self, values: &B::Tensor, level: usize) -> Result<B::Tensor> {
        if level == 0 {
            self.cascade(values, true)
        } else if level + 1 == self.config.levels().len() {
            Ok(values.clone())
        } else {
            self.backend
                .linear1d(values, self.config.raw().context_length)
        }
    }
    pub fn decode(&self, codes: &[u32], batch: usize) -> Result<Decoded<B>> {
        self.decode_traced(codes, batch, "decode", &mut NoTrace)
    }
    /// The label changes observation names only (e.g. fixed-code "forced" capture).
    pub fn decode_traced(
        &self,
        codes: &[u32],
        batch: usize,
        label: &str,
        trace: &mut impl TraceSink<B>,
    ) -> Result<Decoded<B>> {
        self.config.check_codes(codes, batch)?;
        let ids = self
            .backend
            .upload_indices(codes, &[batch, self.config.code_length()])?;
        self.decode_device(&ids, label, trace)
    }
    fn decode_device(
        &self,
        codes: &B::Indices,
        label: &str,
        trace: &mut impl TraceSink<B>,
    ) -> Result<Decoded<B>> {
        let b = &self.backend;
        let mut trace = Prefix {
            prefix: label,
            trace,
        };
        let mut offset = 0;
        let mut cumulative = None;
        let mut snapshots = Vec::new();
        for (level, &length) in self.config.levels().iter().enumerate() {
            let name = format!("level_{level:02}");
            let ids = b.narrow_indices(codes, 1, offset, length)?;
            offset += length;
            let value =
                b.embedding(self.weight(&format!("vq.embeddings.{level}.weight"))?, &ids)?;
            let up = self.upsample(&value, level)?;
            trace.tensor(&format!("{name}.upsampled"), &up)?;
            let contribution = self.refine(&up, level)?;
            trace.tensor(&format!("{name}.contribution"), &contribution)?;
            let sum = match cumulative {
                Some(ref previous) => b.add(previous, &contribution)?,
                None => contribution,
            };
            trace.tensor(&format!("{name}.cumulative"), &sum)?;
            snapshots.push(sum.clone());
            cumulative = Some(sum);
        }
        let latent = cumulative.expect("validated nonempty levels");
        self.decode_latent(&latent, snapshots, &mut trace)
    }
    fn decode_latent(
        &self,
        latent: &B::Tensor,
        snapshots: Vec<B::Tensor>,
        trace: &mut impl TraceSink<B>,
    ) -> Result<Decoded<B>> {
        let b = &self.backend;
        trace.tensor("decoder.input", latent)?;
        let hidden = self.decoder_block_traced(latent, trace)?;
        let hidden = self.norm("decoder.ln_f", &hidden, false, trace)?;
        let logits = self.linear("decoder.output_layer", &hidden, false, trace)?;
        trace.tensor("decoder.output", &logits)?;
        trace.tensor("logits", &logits)?;
        let ids = b.select_last(&logits, true)?;
        trace.indices("ids", &ids)?;
        Ok(Decoded {
            logits,
            ids,
            snapshots,
        })
    }
    /// Encode and decode without a host roundtrip of the intermediate codes.
    /// Reuse the quantized latent already accumulated by encode. Both returned
    /// snapshot sets have immutable value semantics; Clone may share storage.
    /// Decoding remains lossy; this API does not promise recovery of input tokens.
    pub fn reconstruct(&self, tokens: &[u32], batch: usize) -> Result<(Encoded<B>, Decoded<B>)> {
        let encoded = self.encode(tokens, batch)?;
        let latent = encoded.snapshots.last().expect("validated nonempty levels");
        let decoded = self.decode_latent(latent, encoded.snapshots.clone(), &mut NoTrace)?;
        Ok((encoded, decoded))
    }
    /// Retain the independent decode construction for every named observation.
    /// Regression captures compare this route with latent-reusing reconstruction;
    /// traced/profiled timing is not the optimized public API's wall time.
    pub fn reconstruct_traced(
        &self,
        tokens: &[u32],
        batch: usize,
        trace: &mut impl TraceSink<B>,
    ) -> Result<(Encoded<B>, Decoded<B>)> {
        let encoded = self.encode_traced(tokens, batch, trace)?;
        let decoded = self.decode_device(&encoded.codes, "decode", trace)?;
        Ok((encoded, decoded))
    }
}
