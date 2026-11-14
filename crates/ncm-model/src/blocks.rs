//! Shared codec components; device adapters contain no level or block semantics.
use crate::Codec;
use ncm_core::{CodecBackend, Error, Result};
use std::sync::OnceLock;

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
    distances_with_norm(b, x, book, || squared_book_norm(b, book))
}
fn squared_book_norm<B: CodecBackend>(b: &B, book: &B::Tensor) -> Result<B::Tensor> {
    b.permute(&b.sum_last(&b.square(book)?)?, &[1, 0])
}
fn distances_with_norm<B: CodecBackend>(
    b: &B,
    x: &B::Tensor,
    book: &B::Tensor,
    norm: impl FnOnce() -> Result<B::Tensor>,
) -> Result<B::Tensor> {
    // Preserve xx + ee - 2*dot ordering, including the row-constant xx term.
    let xx = b.sum_last(&b.square(x)?)?;
    let ee = norm()?;
    b.sub(&b.add(&xx, &ee)?, &b.scale(&b.project(x, book)?, 2.0)?)
}
fn cached<T>(cell: &OnceLock<T>, make: impl FnOnce() -> Result<T>) -> Result<&T> {
    if let Some(value) = cell.get() {
        return Ok(value);
    }
    // Errors are not cached. Simultaneous first calls may compute a duplicate;
    // set publishes one immutable value and drops the unused temporary safely.
    let _ = cell.set(make()?);
    Ok(cell
        .get()
        .expect("successful immutable cache initialization"))
}

impl<B: CodecBackend> Codec<B> {
    pub(crate) fn level_distances(
        &self,
        x: &B::Tensor,
        book: &B::Tensor,
        level: usize,
    ) -> Result<B::Tensor> {
        let cell = self
            .codebook_norms
            .get(level)
            .ok_or_else(|| Error::Input("invalid codebook level".into()))?;
        distances_with_norm(&self.backend, x, book, || {
            Ok(cached(cell, || squared_book_norm(&self.backend, book))?.clone())
        })
    }
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
        let norm = self.norm_accelerated("decoder.middle_block.ln_1", x, true, trace)?;
        trace.tensor("decoder.middle_block.attn.input", &norm)?;
        let qkv = self.linear("decoder.middle_block.attn.c_attn", &norm, true, trace)?;
        let (batch, length, width) = (shape[0], shape[1], shape[2]);
        let heads = self.config.raw().n_layers;
        // Group Q/K/V in one layout conversion. On eager backends, reshaping
        // the permuted tensor materializes it once; each subsequent slice is a
        // contiguous view. Lazy backends may defer that conversion.
        let packed_shape = [3, batch, heads, length, width / heads];
        let packed = b.reshape(
            &b.permute(
                &b.reshape(&qkv, &[batch, length, 3, heads, width / heads])?,
                &[2, 0, 3, 1, 4],
            )?,
            &packed_shape,
        )?;
        let split = |index| {
            b.reshape(
                &b.narrow(&packed, 0, index, 1)?,
                &[batch, heads, length, width / heads],
            )
        };
        let mut q = split(0)?;
        let mut k = split(1)?;
        let v = split(2)?;
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
        let norm = self.norm_accelerated("decoder.middle_block.ln_2", &residual, true, trace)?;
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
    fn norm_accelerated(
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
        let y = self.backend.layer_norm_accelerated(
            x,
            self.weight(&format!("{name}.weight"))?,
            bias,
            1e-6,
        )?;
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

#[cfg(test)]
mod prepared_tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    #[test]
    fn failed_preparation_is_retryable_and_success_is_reused() {
        let cell = OnceLock::<u32>::new();
        let calls = Cell::new(0);
        assert!(
            cached(&cell, || {
                calls.set(calls.get() + 1);
                Err(Error::Backend("preparation failure".into()))
            })
            .is_err()
        );
        assert!(cell.get().is_none());
        assert_eq!(
            *cached(&cell, || {
                calls.set(calls.get() + 1);
                Ok(7)
            })
            .unwrap(),
            7
        );
        assert_eq!(*cached(&cell, || panic!("must not recompute")).unwrap(), 7);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn separate_owners_do_not_share_values_or_extend_lifetimes() {
        // Also demonstrate that non-Send/non-Sync backend values are supported.
        let first = OnceLock::new();
        let second = OnceLock::new();
        let weak = Rc::downgrade(cached(&first, || Ok(Rc::new(7))).unwrap());
        assert_eq!(**cached(&second, || Ok(Rc::new(13))).unwrap(), 13);
        drop(first);
        assert!(weak.upgrade().is_none());
        assert_eq!(**second.get().unwrap(), 13);
    }

    #[test]
    fn concurrent_first_calls_publish_one_immutable_value() {
        let cell = OnceLock::new();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    assert_eq!(cached(&cell, || Ok(vec![1, 2, 3])).unwrap(), &[1, 2, 3]);
                });
            }
        });
        assert_eq!(cell.get().unwrap(), &[1, 2, 3]);
    }
}
