//! Shared codec components; device adapters contain no level or block semantics.
use crate::Codec;
use ncm_core::{CodecBackend, Error, Result};

/// Optional observations have immutable value semantics. Observers must not alter
/// tensors; cloning retains independent snapshots even on lazy native backends.
pub trait TraceSink<B: CodecBackend> {
    fn tensor(&mut self, name: &str, value: &B::Tensor) -> Result<()>;
    fn indices(&mut self, name: &str, value: &B::Indices) -> Result<()>;
}
pub struct NoTrace;
impl<B: CodecBackend> TraceSink<B> for NoTrace {
    fn tensor(&mut self, _: &str, _: &B::Tensor) -> Result<()> {
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &B::Indices) -> Result<()> {
        Ok(())
    }
}

/// Squared Euclidean distance; no cosine normalization or hidden precision change.
pub fn codebook_distances<B: CodecBackend>(
    b: &B,
    x: &B::Tensor,
    book: &B::Tensor,
) -> Result<B::Tensor> {
    let xx = b.sum_last(&b.square(x)?)?;
    let ee = b.permute(&b.sum_last(&b.square(book)?)?, &[1, 0])?;
    b.sub(&b.add(&xx, &ee)?, &b.scale(&b.project(x, book)?, 2.0)?)
}

impl<B: CodecBackend> Codec<B> {
    pub(crate) fn check_latent(&self, x: &B::Tensor, length: usize) -> Result<Vec<usize>> {
        let shape = self.backend.shape(x);
        if shape.len() != 3
            || !(1..=128).contains(&shape[0])
            || shape[1] != length
            || shape[2] != self.config.raw().embed_dim
        {
            return Err(Error::Input("invalid codec latent shape".into()));
        }
        Ok(shape)
    }
    pub(crate) fn weight(&self, name: &str) -> Result<&B::Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| Error::Weights(format!("missing runtime weight {name}")))
    }
    /// Level-zero learned cascade. Normalization precedes each stage after the
    /// first; the final down-only normalization has no affine parameters.
    pub fn cascade(&self, x: &B::Tensor, up: bool) -> Result<B::Tensor> {
        self.check_latent(
            x,
            if up {
                1
            } else {
                self.config.raw().context_length
            },
        )?;
        let b = &self.backend;
        let q = &self.config.raw().vector_quantizer_config;
        let mut strides = self.config.strides().to_vec();
        if up {
            strides.reverse();
        }
        let mut value = x.clone();
        for (i, stride) in strides.into_iter().enumerate() {
            if i > 0 && q.learned_l1_cascade_norm {
                value = b.group_norm(&value, 8, 1e-5)?;
            }
            let index = if q.learned_l1_cascade_norm { i * 2 } else { i };
            let name = format!("vq.learned_{}_l1.{index}", if up { "up" } else { "down" });
            value = b.conv1d(
                &value,
                self.weight(&format!("{name}.weight"))?,
                self.weight(&format!("{name}.bias"))?,
                stride,
                0,
                up,
            )?;
        }
        if !up && q.learned_l1_cascade_norm_final {
            value = b.group_norm(&value, 8, 1e-5)?;
        }
        Ok(value)
    }
    pub fn refine(&self, x: &B::Tensor, level: usize) -> Result<B::Tensor> {
        self.check_latent(x, self.config.raw().context_length)?;
        if level >= self.config.levels().len() {
            return Err(Error::Input("invalid refinement level".into()));
        }
        let b = &self.backend;
        let ratio = self.config.raw().vector_quantizer_config.quant_resi.ratio;
        let name = format!("vq.quant_resi.{level}");
        let conv = b.conv1d(
            x,
            self.weight(&format!("{name}.weight"))?,
            self.weight(&format!("{name}.bias"))?,
            1,
            1,
            false,
        )?;
        b.add(&b.scale(x, 1.0 - ratio)?, &b.scale(&conv, ratio)?)
    }
    pub fn decoder_block(&self, x: &B::Tensor) -> Result<B::Tensor> {
        self.decoder_block_traced(x, &mut NoTrace)
    }
    pub fn decoder_block_traced(
        &self,
        x: &B::Tensor,
        trace: &mut impl TraceSink<B>,
    ) -> Result<B::Tensor> {
        let shape = self.check_latent(x, self.config.raw().context_length)?;
        let b = &self.backend;
        trace.tensor("decoder.middle_block.input", x)?;
        let norm = self.norm("decoder.middle_block.ln_1", x, true, trace)?;
        trace.tensor("decoder.middle_block.attn.input", &norm)?;
        let qkv = self.linear("decoder.middle_block.attn.c_attn", &norm, true, trace)?;
        let (batch, length, width) = (shape[0], shape[1], shape[2]);
        let heads = self.config.raw().n_layers;
        let split = |offset| {
            b.permute(
                &b.reshape(
                    &b.narrow(&qkv, 2, offset, width)?,
                    &[batch, length, heads, width / heads],
                )?,
                &[0, 2, 1, 3],
            )
        };
        let mut q = split(0)?;
        let mut k = split(width)?;
        let v = split(2 * width)?;
        trace.tensor("decoder.middle_block.attn.q_norm.input", &q)?;
        q = b.rms_norm(
            &q,
            self.weight("decoder.middle_block.attn.q_norm.weight")?,
            1e-6,
        )?;
        trace.tensor("decoder.middle_block.attn.q_norm.output", &q)?;
        trace.tensor("decoder.middle_block.attn.k_norm.input", &k)?;
        k = b.rms_norm(
            &k,
            self.weight("decoder.middle_block.attn.k_norm.weight")?,
            1e-6,
        )?;
        trace.tensor("decoder.middle_block.attn.k_norm.output", &k)?;
        let attention = b.reshape(
            &b.permute(&b.attention(&q, &k, &v, None)?, &[0, 2, 1, 3])?,
            &[batch, length, width],
        )?;
        let attention = self.linear("decoder.middle_block.attn.c_proj", &attention, true, trace)?;
        trace.tensor("decoder.middle_block.attn.output", &attention)?;
        let residual = b.add(x, &attention)?;
        let norm = self.norm("decoder.middle_block.ln_2", &residual, true, trace)?;
        trace.tensor("decoder.middle_block.mlp.input", &norm)?;
        let hidden = self.linear("decoder.middle_block.mlp.c_fc", &norm, true, trace)?;
        trace.tensor("decoder.middle_block.mlp.gelu.input", &hidden)?;
        let hidden = b.gelu(&hidden)?;
        trace.tensor("decoder.middle_block.mlp.gelu.output", &hidden)?;
        let mlp = self.linear("decoder.middle_block.mlp.c_proj", &hidden, true, trace)?;
        trace.tensor("decoder.middle_block.mlp.output", &mlp)?;
        let output = b.add(&residual, &mlp)?;
        trace.tensor("decoder.middle_block.output", &output)?;
        Ok(output)
    }
    pub(crate) fn linear(
        &self,
        name: &str,
        x: &B::Tensor,
        bias: bool,
        trace: &mut impl TraceSink<B>,
    ) -> Result<B::Tensor> {
        trace.tensor(&format!("{name}.input"), x)?;
        let mut y = self
            .backend
            .project(x, self.weight(&format!("{name}.weight"))?)?;
        if bias {
            y = self
                .backend
                .add(&y, self.weight(&format!("{name}.bias"))?)?;
        }
        trace.tensor(&format!("{name}.output"), &y)?;
        Ok(y)
    }
    pub(crate) fn norm(
        &self,
        name: &str,
        x: &B::Tensor,
        bias: bool,
        trace: &mut impl TraceSink<B>,
    ) -> Result<B::Tensor> {
        trace.tensor(&format!("{name}.input"), x)?;
        let bias = if bias {
            Some(self.weight(&format!("{name}.bias"))?)
        } else {
            None
        };
        let y = self
            .backend
            .layer_norm(x, self.weight(&format!("{name}.weight"))?, bias, 1e-6)?;
        trace.tensor(&format!("{name}.output"), &y)?;
        Ok(y)
    }
}
