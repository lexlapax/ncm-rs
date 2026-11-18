//! Compare the input and logits of a WMT14 base head at one frozen corpus position.
use ncm_io::sha256;
use std::{error::Error, fs};

const LEVELS: [usize; 11] = [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64];
const BATCH: usize = 4;
const ROWS: usize = 3003;

#[derive(Clone, Copy)]
struct Site {
    row: usize,
    level: usize,
    position: usize,
}

impl Site {
    fn batch_start(self) -> usize {
        self.row / BATCH * BATCH
    }
}

#[cfg(any(
    feature = "cuda-cudnn",
    all(feature = "mlx", target_os = "macos", target_arch = "aarch64")
))]
mod probe {
    use super::{LEVELS, Site};
    use ncm_core::{GeneratorBackend, Result};
    use ncm_io::Artifact;
    use ncm_model::{TraceSink, Wmt14Teacher};
    use serde_json::{Value, json};

    struct HeadTrace<B: GeneratorBackend> {
        level: usize,
        logits_name: String,
        input: Option<B::Tensor>,
        logits: Option<B::Tensor>,
        codes: Option<B::Indices>,
        ids: Option<B::Indices>,
    }

    impl<B: GeneratorBackend> TraceSink<B> for HeadTrace<B> {
        fn wmt14_head_input(&mut self, level: usize, input: &B::Tensor) -> Result<()> {
            if level == self.level {
                self.input = Some(input.clone());
            }
            Ok(())
        }

        fn tensor(&mut self, name: &str, value: &B::Tensor) -> Result<()> {
            if name == self.logits_name {
                self.logits = Some(value.clone());
            }
            Ok(())
        }

        fn indices(&mut self, name: &str, value: &B::Indices) -> Result<()> {
            match name {
                "generator.codes" => self.codes = Some(value.clone()),
                "generator.greedy_ids" => self.ids = Some(value.clone()),
                _ => {}
            }
            Ok(())
        }
    }

    pub fn run<B: GeneratorBackend + Clone>(
        backend: B,
        artifact: &str,
        prefixes: &[u32],
        site: Site,
    ) -> std::result::Result<Value, Box<dyn std::error::Error>> {
        let model = Wmt14Teacher::from_owned_artifact(
            backend.clone(),
            Artifact::load_inference(artifact)?,
        )?;
        let batch = prefixes.len() / 96;
        let plain = model.generate_top1(prefixes, batch)?;
        let codes = backend.read_indices(&plain.codes)?;
        let ids = backend.read_indices(&plain.ids)?;
        drop(plain);
        let mut trace = HeadTrace::<B> {
            level: site.level,
            logits_name: format!("generator.level_{:02}.logits", site.level),
            input: None,
            logits: None,
            codes: None,
            ids: None,
        };
        model.generate_top1_traced(prefixes, batch, &mut trace)?;
        if codes != backend.read_indices(&trace.codes.ok_or("missing traced codes")?)?
            || ids != backend.read_indices(&trace.ids.ok_or("missing traced IDs")?)?
            || codes.len() != batch * 313
            || ids.len() != batch * 64
        {
            return Err("traced and untraced WMT14 outputs differ".into());
        }

        let input = trace.input.ok_or("missing head input")?;
        let logits = trace.logits.ok_or("missing head logits")?;
        let positions = LEVELS[site.level];
        if backend.shape(&input) != [batch, positions, 768]
            || backend.shape(&logits) != [batch, positions, 4096]
        {
            return Err("unexpected WMT14 head shape".into());
        }
        let row = site.row - site.batch_start();
        let scores = backend.narrow(&backend.narrow(&logits, 0, row, 1)?, 1, site.position, 1)?;
        let inputs = backend.read_f32(&input)?;
        let scores = backend.read_f32(&scores)?;
        if inputs.len() != batch * positions * 768
            || scores.len() != 4096
            || inputs.iter().chain(&scores).any(|x| !x.is_finite())
        {
            return Err("invalid WMT14 head observation".into());
        }

        Ok(json!({
            "cfg": 0,
            "batch_start": site.batch_start(),
            "row": site.row,
            "level": site.level,
            "position": site.position,
            "head_input_shape": [batch, positions, 768],
            "head_input_bits": inputs.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            "row_site_logits_bits": scores.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            "codes": codes.as_chunks::<313>().0.iter().map(|row| row.as_slice()).collect::<Vec<_>>(),
            "raw_ids": ids.as_chunks::<64>().0.iter().map(|row| row.as_slice()).collect::<Vec<_>>(),
        }))
    }
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let [_, artifact, device, prefixes, row, level, position] = args.as_slice() else {
        return Err(
            "usage: wmt14_base_head_probe ARTIFACT cuda|mlx PREFIXES ROW LEVEL POSITION".into(),
        );
    };
    let site = Site {
        row: row.parse()?,
        level: level.parse()?,
        position: position.parse()?,
    };
    if site.row >= ROWS
        || LEVELS
            .get(site.level)
            .is_none_or(|&len| site.position >= len)
    {
        return Err("WMT14 corpus position is out of range".into());
    }
    if sha256(&fs::read(format!("{artifact}/manifest.json"))?)
        != "01772927d6c334edd37650b042bb37a6e915652de701226e0a6661a847c8cc81"
    {
        return Err("unexpected WMT14 artifact manifest".into());
    }
    let bytes = fs::read(prefixes)?;
    if sha256(&bytes) != "22c76d597ae51dffadf61661d6b7c4d4290fd14b8809bfa6f1959b5c668df93f" {
        return Err("unexpected WMT14 prefix reference".into());
    }
    let input: serde_json::Value = serde_json::from_slice(&bytes)?;
    let rows: Vec<Vec<u32>> = serde_json::from_value(input["prefixes"].clone())?;
    if rows.len() != ROWS || rows.iter().any(|row| row.len() != 96) {
        return Err("unexpected WMT14 prefix shape".into());
    }
    let start = site.batch_start();
    let prefixes: Vec<_> = rows[start..(start + BATCH).min(ROWS)]
        .iter()
        .flatten()
        .copied()
        .collect();

    let _ = &prefixes;
    let observed: Result<serde_json::Value, Box<dyn Error>> = match device.as_str() {
        #[cfg(feature = "cuda-cudnn")]
        "cuda" => probe::run(
            ncm_backend_cuda::CandleBackend::cuda_wmt14_fast_argmax_experimental(0)?,
            artifact,
            &prefixes,
            site,
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        "mlx" => probe::run(
            ncm_backend_mlx::MlxBackend::gpu()?,
            artifact,
            &prefixes,
            site,
        ),
        _ => Err("requested device is unavailable".into()),
    };
    println!("{}", observed?);
    Ok(())
}
