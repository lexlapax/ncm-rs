use ncm_core::{CodecConfig, Error, Result, WeightBackend};
use ncm_model::Codec;
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};
use serde_json::Value;
use std::{cell::Cell, collections::BTreeMap, rc::Rc};

#[derive(Clone, Default)]
struct Counter(Rc<Cell<usize>>);
impl WeightBackend for Counter {
    type Tensor = Vec<usize>;
    fn upload_f32(&self, values: &[f32], shape: &[usize]) -> Result<Self::Tensor> {
        assert_eq!(values.len(), shape.iter().product::<usize>());
        self.0.set(self.0.get() + 1);
        Ok(shape.to_vec())
    }
}
struct HostTensor {
    shape: Vec<usize>,
    dtype: Dtype,
    data: Vec<u8>,
}
fn fixture() -> (CodecConfig, BTreeMap<String, HostTensor>) {
    let f: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/codec-v1.json")).unwrap();
    let config = CodecConfig::from_json(&serde_json::to_vec(&f["config"]).unwrap()).unwrap();
    let weights = f["weights"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, key)| {
            let a = &f["arrays"][key.as_str().unwrap()];
            let hex = a["data_le_hex"].as_str().unwrap();
            let data = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            (
                name.clone(),
                HostTensor {
                    shape: serde_json::from_value(a["shape"].clone()).unwrap(),
                    dtype: Dtype::F32,
                    data,
                },
            )
        })
        .collect();
    (config, weights)
}
fn bytes(weights: &BTreeMap<String, HostTensor>) -> Vec<u8> {
    serialize(
        weights.iter().map(|(name, t)| {
            (
                name,
                TensorView::new(t.dtype, t.shape.clone(), &t.data).unwrap(),
            )
        }),
        None,
    )
    .unwrap()
}
#[test]
fn tiny_weights_construct_checked_shared_state() {
    let (config, weights) = fixture();
    let counter = Counter::default();
    let codec = Codec::from_safetensors(counter.clone(), config, &bytes(&weights)).unwrap();
    assert_eq!(counter.0.get(), weights.len());
    assert_eq!(codec.weights().len(), weights.len());
    assert_eq!(codec.config().code_length(), 27);
}
#[test]
fn invalid_inventories_and_layouts_allocate_nothing() {
    for mutation in 0..6 {
        let (config, mut weights) = fixture();
        // Last lexical tensor tests full preflight, not merely first-item checking.
        let key = weights.last_key_value().unwrap().0.clone();
        match mutation {
            0 => {
                weights.remove(&key);
            }
            1 => {
                weights.insert(
                    "unexpected".into(),
                    HostTensor {
                        shape: vec![1],
                        dtype: Dtype::F32,
                        data: vec![0; 4],
                    },
                );
            }
            2 => {
                let t = weights.get_mut(&key).unwrap();
                t.shape = vec![t.data.len() / 4];
            }
            3 => {
                weights.get_mut(&key).unwrap().dtype = Dtype::I32;
            }
            4 => {
                weights.get_mut(&key).unwrap().data[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
            5 => {
                weights.get_mut(&key).unwrap().data[0..4]
                    .copy_from_slice(&f32::INFINITY.to_le_bytes());
            }
            _ => unreachable!(),
        }
        let counter = Counter::default();
        assert!(
            Codec::from_safetensors(counter.clone(), config, &bytes(&weights)).is_err(),
            "accepted mutation {mutation}"
        );
        assert_eq!(counter.0.get(), 0);
    }
}
#[test]
fn malformed_file_and_backend_failure_propagate() {
    let (config, weights) = fixture();
    let counter = Counter::default();
    assert!(Codec::from_safetensors(counter.clone(), config.clone(), b"not safetensors").is_err());
    assert_eq!(counter.0.get(), 0);
    struct Failing;
    impl WeightBackend for Failing {
        type Tensor = ();
        fn upload_f32(&self, _: &[f32], _: &[usize]) -> Result<()> {
            Err(Error::Backend("device allocation failed".into()))
        }
    }
    assert!(matches!(
        Codec::from_safetensors(Failing, config, &bytes(&weights)),
        Err(Error::Backend(_))
    ));
}
