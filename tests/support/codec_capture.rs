//! Device-neutral owned observations for fixture checks and offline trace export.
use ncm_core::{CodecBackend, Result};
use ncm_model::TraceSink;
use std::collections::BTreeMap;

pub struct Captured<B: CodecBackend> {
    pub tensors: BTreeMap<String, B::Tensor>,
    pub indices: BTreeMap<String, B::Indices>,
}
impl<B: CodecBackend> Default for Captured<B> {
    fn default() -> Self {
        Self {
            tensors: BTreeMap::new(),
            indices: BTreeMap::new(),
        }
    }
}
impl<B: CodecBackend> TraceSink<B> for Captured<B> {
    fn tensor(&mut self, name: &str, value: &B::Tensor) -> Result<()> {
        assert!(!self.indices.contains_key(name));
        assert!(
            self.tensors.insert(name.into(), value.clone()).is_none(),
            "duplicate trace {name}"
        );
        Ok(())
    }
    fn indices(&mut self, name: &str, value: &B::Indices) -> Result<()> {
        assert!(!self.tensors.contains_key(name));
        assert!(
            self.indices.insert(name.into(), value.clone()).is_none(),
            "duplicate trace {name}"
        );
        Ok(())
    }
}
pub fn tolerance(name: &str) -> (f64, f64) {
    let atol = if name.ends_with("distances") {
        1e-3
    } else if [
        "cumulative",
        "logits",
        "decoder.output",
        "decoder.input",
        "decoder.middle_block.output",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
    {
        5e-4
    } else {
        1e-4
    };
    (atol, 2e-4)
}
