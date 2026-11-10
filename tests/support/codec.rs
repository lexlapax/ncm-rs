//! Shared, device-neutral execution of the fixed Phase 2 tiny reference.
#[path = "codec_capture.rs"]
mod capture;
use ncm_core::{CodecBackend, CodecConfig, Result};
use ncm_model::{Codec, TraceSink, codebook_distances};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub struct Array {
    pub shape: Vec<usize>,
    dtype: String,
    data_le_hex: String,
}
impl Array {
    pub fn bytes(&self) -> Vec<u8> {
        assert!(self.data_le_hex.len().is_multiple_of(2));
        (0..self.data_le_hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&self.data_le_hex[i..i + 2], 16).unwrap())
            .collect()
    }
    pub fn floats(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "F32");
        self.bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
    pub fn indices(&self) -> Vec<u32> {
        assert_eq!(self.dtype, "I64");
        self.bytes()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u32::try_from(i64::from_le_bytes(*b)).unwrap())
            .collect()
    }
    pub fn upload<B: CodecBackend>(&self, b: &B) -> B::Tensor {
        b.upload_f32(&self.floats(), &self.shape).unwrap()
    }
}
#[derive(Deserialize)]
pub struct Case {
    pub name: String,
    pub traces: BTreeMap<String, String>,
}
#[derive(Deserialize)]
pub struct Fixture {
    pub arrays: BTreeMap<String, Array>,
    pub cases: Vec<Case>,
    weights: BTreeMap<String, String>,
    config: serde_json::Value,
    schema_version: u32,
    profile: String,
    upstream_commit: String,
}
impl Fixture {
    pub fn load() -> Self {
        let f: Self = serde_json::from_str(include_str!("../fixtures/codec-v1.json")).unwrap();
        assert_eq!(f.schema_version, 1);
        assert_eq!(f.profile, "synthetic-codec-v1");
        assert_eq!(f.upstream_commit, ncm_io::UPSTREAM);
        for (key, a) in &f.arrays {
            let bytes = a.bytes();
            let item = match a.dtype.as_str() {
                "F32" => 4,
                "I64" => 8,
                _ => panic!("unsupported dtype"),
            };
            assert_eq!(bytes.len(), a.shape.iter().product::<usize>() * item);
            let mut hash = Sha256::new();
            hash.update(format!("[\"{}\", {:?}]", a.dtype, a.shape));
            hash.update([0]);
            hash.update(&bytes);
            assert_eq!(*key, format!("{:x}", hash.finalize()));
        }
        f
    }
    pub fn array(&self, case: &Case, name: &str) -> &Array {
        &self.arrays[&case.traces[name]]
    }
    pub fn model<B: CodecBackend>(&self, b: B) -> Codec<B> {
        let config = CodecConfig::from_json(&serde_json::to_vec(&self.config).unwrap()).unwrap();
        let data: BTreeMap<_, _> = self
            .weights
            .iter()
            .map(|(name, key)| (name, self.arrays[key].bytes()))
            .collect();
        let views = data.iter().map(|(name, bytes)| {
            (
                *name,
                TensorView::new(
                    Dtype::F32,
                    self.arrays[&self.weights[*name]].shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        });
        let bytes = serialize(views, None).unwrap();
        Codec::from_safetensors(b, config, &bytes).unwrap()
    }
}

#[derive(Default, Debug)]
pub struct Summary {
    pub tensors: usize,
    pub elements: usize,
    pub max_abs: f64,
}
impl Summary {
    pub fn check<B: CodecBackend>(
        &mut self,
        b: &B,
        x: &B::Tensor,
        reference: &Array,
        name: &str,
        atol: f64,
        rtol: f64,
    ) {
        assert_eq!(b.shape(x), reference.shape, "{name}");
        let actual = b.read_f32(x).unwrap();
        let expected = reference.floats();
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            let error = (f64::from(a) - f64::from(e)).abs();
            assert!(
                a.is_finite() && e.is_finite() && error <= atol + rtol * f64::from(e).abs(),
                "{name}[{i}]: actual {a}, reference {e}, abs error {error}"
            );
            self.max_abs = self.max_abs.max(error);
        }
        self.tensors += 1;
        self.elements += actual.len();
    }
}
struct Observer<'a, B: CodecBackend> {
    b: &'a B,
    fixture: &'a Fixture,
    case: &'a Case,
    summary: &'a mut Summary,
    prefix: &'a str,
}
impl<B: CodecBackend> TraceSink<B> for Observer<'_, B> {
    fn tensor(&mut self, name: &str, x: &B::Tensor) -> Result<()> {
        let key = format!("{}.{name}", self.prefix);
        let atol = if name == "decoder.middle_block.output" {
            5e-4
        } else {
            1e-4
        };
        self.summary.check(
            self.b,
            x,
            self.fixture.array(self.case, &key),
            &key,
            atol,
            2e-4,
        );
        Ok(())
    }
    fn indices(&mut self, _: &str, _: &B::Indices) -> Result<()> {
        panic!("unexpected block index trace")
    }
}

pub fn components<B: CodecBackend>(backend: B) -> Summary {
    let f = Fixture::load();
    let codec = f.model(backend);
    let b = codec.backend();
    let mut summary = Summary::default();
    for case in &f.cases {
        let width = codec.config().raw().embed_dim;
        let levels = codec.config().levels();
        for phase in ["decode", "forced"] {
            let input = f
                .array(case, &format!("{phase}.decoder.middle_block.input"))
                .upload(b);
            codec
                .decoder_block_traced(
                    &input,
                    &mut Observer {
                        b,
                        fixture: &f,
                        case,
                        summary: &mut summary,
                        prefix: phase,
                    },
                )
                .unwrap();
            let input = f
                .array(
                    case,
                    &format!("{phase}.decoder.middle_block.mlp.gelu.input"),
                )
                .upload(b);
            let output = b.gelu(&input).unwrap();
            summary.check(
                b,
                &output,
                f.array(
                    case,
                    &format!("{phase}.decoder.middle_block.mlp.gelu.output"),
                ),
                "isolated exact GELU",
                1e-5,
                1e-4,
            );
        }
        let expected_codes = f.array(case, "encode.codes").indices();
        let batch = f.array(case, "encode.codes").shape[0];
        let mut offset = 0;
        let mut cumulative: Option<B::Tensor> = None;
        for (level, &length) in levels.iter().enumerate() {
            let prefix = format!("encode.level_{level:02}");
            let residual = f
                .array(case, &format!("{prefix}.residual_before"))
                .upload(b);
            let down = if level == 0 {
                codec.cascade(&residual, false).unwrap()
            } else if level + 1 == levels.len() {
                residual
            } else {
                b.area1d(&residual, length).unwrap()
            };
            let down = b.reshape(&down, &[batch * length, width]).unwrap();
            summary.check(
                b,
                &down,
                f.array(case, &format!("{prefix}.quant_input")),
                "quant input",
                1e-4,
                2e-4,
            );
            let book = &codec.weights()[&format!("vq.embeddings.{level}.weight")];
            let distances = codebook_distances(b, &down, book).unwrap();
            summary.check(
                b,
                &distances,
                f.array(case, &format!("{prefix}.distances")),
                "distances",
                1e-3,
                2e-4,
            );
            let ids = b.select_last(&distances, false).unwrap();
            let expected: Vec<_> = expected_codes
                .chunks_exact(codec.config().code_length())
                .flat_map(|row| row[offset..offset + length].iter().copied())
                .collect();
            assert_eq!(
                b.read_indices(&ids).unwrap(),
                expected,
                "{} level {level}",
                case.name
            );
            offset += length;
            let values = b
                .reshape(&b.embedding(book, &ids).unwrap(), &[batch, length, width])
                .unwrap();
            let up = if level == 0 {
                codec.cascade(&values, true).unwrap()
            } else if level + 1 == levels.len() {
                values
            } else {
                b.linear1d(&values, codec.config().raw().context_length)
                    .unwrap()
            };
            summary.check(
                b,
                &up,
                f.array(case, &format!("{prefix}.upsampled")),
                "upsampled",
                1e-4,
                2e-4,
            );
            let contribution = codec.refine(&up, level).unwrap();
            summary.check(
                b,
                &contribution,
                f.array(case, &format!("{prefix}.contribution")),
                "Phi contribution",
                1e-4,
                2e-4,
            );
            cumulative = Some(match cumulative {
                Some(ref sum) => b.add(sum, &contribution).unwrap(),
                None => contribution,
            });
            summary.check(
                b,
                cumulative.as_ref().unwrap(),
                f.array(case, &format!("{prefix}.cumulative")),
                "cumulative",
                5e-4,
                2e-4,
            );
        }
    }
    summary
}

pub fn full_codec<B: CodecBackend>(backend: B) -> Summary {
    let f = Fixture::load();
    let codec = f.model(backend);
    let b = codec.backend();
    let mut summary = Summary::default();
    for case in &f.cases {
        let input = f.array(case, "input.ids");
        let supplied = f.array(case, "supplied.codes");
        let batch = input.shape[0];
        let mut capture = capture::Captured::default();
        capture
            .indices(
                "input.ids",
                &b.upload_indices(&input.indices(), &input.shape).unwrap(),
            )
            .unwrap();
        capture
            .indices(
                "supplied.codes",
                &b.upload_indices(&supplied.indices(), &supplied.shape)
                    .unwrap(),
            )
            .unwrap();
        let (encoded, decoded) = codec
            .reconstruct_traced(&input.indices(), batch, &mut capture)
            .unwrap();
        let forced = codec
            .decode_traced(&supplied.indices(), batch, "forced", &mut capture)
            .unwrap();
        assert_eq!(
            capture.tensors.len() + capture.indices.len(),
            case.traces.len()
        );
        for (name, value) in &capture.tensors {
            let (atol, rtol) = capture::tolerance(name);
            summary.check(b, value, f.array(case, name), name, atol, rtol);
        }
        for (name, value) in &capture.indices {
            let expected = f.array(case, name);
            assert_eq!(b.indices_shape(value), expected.shape, "{name}");
            assert_eq!(b.read_indices(value).unwrap(), expected.indices(), "{name}");
        }
        let (plain_codes, plain_decode) = codec.reconstruct(&input.indices(), batch).unwrap();
        let plain_forced = codec.decode(&supplied.indices(), batch).unwrap();
        assert_eq!(
            b.read_indices(&plain_codes.codes).unwrap(),
            f.array(case, "encode.codes").indices()
        );
        for (label, plain) in [("decode", plain_decode), ("forced", plain_forced)] {
            summary.check(
                b,
                &plain.logits,
                f.array(case, &format!("{label}.logits")),
                "uninstrumented logits",
                5e-4,
                2e-4,
            );
            assert_eq!(
                b.read_indices(&plain.ids).unwrap(),
                f.array(case, &format!("{label}.ids")).indices()
            );
        }
        // Retain and re-read every earlier API snapshot after subsequent decode.
        for (prefix, snapshots) in [
            ("encode", encoded.snapshots),
            ("decode", decoded.snapshots),
            ("forced", forced.snapshots),
        ] {
            assert_eq!(snapshots.len(), codec.config().levels().len());
            for (level, snapshot) in snapshots.iter().enumerate() {
                let name = format!("{prefix}.level_{level:02}.cumulative");
                summary.check(b, snapshot, f.array(case, &name), &name, 5e-4, 2e-4);
            }
        }
        assert!(codec.encode(&input.indices()[1..], batch).is_err());
        assert!(
            codec
                .decode(&vec![u32::MAX; supplied.indices().len()], batch)
                .is_err()
        );
    }
    summary
}
