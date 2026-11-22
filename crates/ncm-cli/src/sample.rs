//! Translate the unified command into the existing, explicitly selected inference paths.
use super::Result;
use std::{fs, io::Read, path::Path};

const USAGE: &str = "usage: ncm sample ARTIFACT DEVICE INPUT_JSON [--seed N] [--cfg S] [--decode] [--text] [--reference-rope FILE] [--experimental-stochastic-xsum] [--experimental-positive-cfg-xsum] [--experimental-cfg-wmt14] [--top-p-fp32|--host-fp64]";

/// Read only enough untrusted metadata to choose a path. The chosen path must
/// subsequently load and validate the complete artifact before inference.
fn profile(root: &str) -> Result<String> {
    let root = Path::new(root);
    if !fs::symlink_metadata(root)?.file_type().is_dir() {
        return Err("artifact root must be a real directory".into());
    }
    const MAX_MANIFEST: u64 = 4 * 1024 * 1024;
    let path = root.join("manifest.json");
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_MANIFEST {
        return Err("artifact manifest must be a bounded regular file".into());
    }
    // As with Artifact::load, the artifact must not be concurrently mutated.
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(MAX_MANIFEST + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_MANIFEST {
        return Err("artifact manifest exceeds 4 MiB".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    value["profile"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "artifact manifest lacks a profile".into())
}

/// Preserve qualified positional alias syntax while requiring opt-in for experimental modes.
pub(super) fn validate_xsum_alias(args: &mut Vec<String>) -> Result<()> {
    if args.len() < 5 {
        return Err("invalid sample-xsum arguments".into());
    }
    let mut positional = Vec::new();
    let (mut stochastic, mut positive_cfg) = (false, false);
    for arg in &args[5..] {
        match arg.as_str() {
            "--experimental-stochastic-xsum" if !stochastic => stochastic = true,
            "--experimental-positive-cfg-xsum" if !positive_cfg => positive_cfg = true,
            value if !value.starts_with("--") => positional.push(value.to_owned()),
            _ => return Err("invalid sample-xsum arguments".into()),
        }
    }
    if positional.len() > 2 {
        return Err("invalid sample-xsum arguments".into());
    }
    let cfg = positional
        .first()
        .map(|s| s.parse::<f64>())
        .transpose()?
        .unwrap_or(0.0);
    if !cfg.is_finite() || cfg < 0.0 {
        return Err("CFG scale must be finite and nonnegative".into());
    }
    if (cfg > 0.0) != positive_cfg {
        return Err("positive XSum CFG requires --experimental-positive-cfg-xsum".into());
    }
    if (positional.len() == 2) != stochastic {
        return Err(
            "XSum stochastic sampling requires a seed and --experimental-stochastic-xsum".into(),
        );
    }
    if let Some(seed) = positional.get(1) {
        seed.parse::<u64>()?;
    }
    args.truncate(5);
    args.extend(positional);
    Ok(())
}

fn validate_wmt14_cfg(cfg: f64, experimental: bool) -> Result<()> {
    if !cfg.is_finite() || cfg < 0.0 {
        return Err("CFG scale must be finite and nonnegative".into());
    }
    if (cfg > 0.0 && cfg != 1.0) != experimental {
        return Err("WMT14 CFG other than 0 or 1 requires --experimental-cfg-wmt14; the flag requires such a scale".into());
    }
    Ok(())
}

pub(super) fn validate_wmt14_alias(args: &mut Vec<String>) -> Result<()> {
    if args.len() < 5 {
        return Err("invalid sample-wmt14 arguments".into());
    }
    let mut scale: Option<String> = None;
    let mut experimental = false;
    for arg in &args[5..] {
        match arg.as_str() {
            "--experimental-cfg-wmt14" if !experimental => experimental = true,
            value if !value.starts_with("--") && scale.is_none() => scale = Some(value.to_owned()),
            _ => return Err("invalid sample-wmt14 arguments".into()),
        }
    }
    validate_wmt14_cfg(
        scale.as_deref().map(str::parse).transpose()?.unwrap_or(0.0),
        experimental,
    )?;
    args.truncate(5);
    args.extend(scale);
    Ok(())
}

pub(super) fn translate(args: &[String]) -> Result<Vec<String>> {
    if args.len() < 5 {
        return Err(USAGE.into());
    }
    translate_profile(args, &profile(&args[2])?)
}

fn translate_profile(args: &[String], profile: &str) -> Result<Vec<String>> {
    if args.len() < 5 {
        return Err(USAGE.into());
    }
    let (mut seed, mut cfg, mut rope) = (None, None, None);
    let (mut decode, mut text, mut stochastic, mut positive_cfg, mut top_p, mut host_fp64) =
        (false, false, false, false, false, false);
    let mut wmt14_experimental = false;
    let mut options = args[5..].iter();
    while let Some(option) = options.next() {
        match option.as_str() {
            "--seed" if seed.is_none() => {
                seed = Some(options.next().ok_or(USAGE)?.parse::<u64>()?);
            }
            "--cfg" if cfg.is_none() => {
                let value = options.next().ok_or(USAGE)?.parse::<f64>()?;
                if !value.is_finite() || value < 0.0 {
                    return Err("CFG scale must be finite and nonnegative".into());
                }
                cfg = Some(value);
            }
            "--reference-rope" if rope.is_none() => rope = Some(options.next().ok_or(USAGE)?),
            "--decode" if !decode => decode = true,
            "--text" if !text => text = true,
            "--experimental-stochastic-xsum" if !stochastic => stochastic = true,
            "--experimental-positive-cfg-xsum" if !positive_cfg => positive_cfg = true,
            "--experimental-cfg-wmt14" if !wmt14_experimental => wmt14_experimental = true,
            "--top-p-fp32" if !top_p && !host_fp64 => top_p = true,
            "--host-fp64" if !top_p && !host_fp64 => host_fp64 = true,
            _ => return Err(USAGE.into()),
        }
    }
    if wmt14_experimental && profile != "wmt14-inference-v1" {
        return Err("--experimental-cfg-wmt14 is only supported for WMT14".into());
    }
    let mut result = vec![
        args[0].clone(),
        String::new(),
        args[2].clone(),
        args[3].clone(),
        args[4].clone(),
    ];
    match profile {
        "sudoku-easy-inspection-v1"
        | "sudoku-medium-inspection-v1"
        | "sudoku-hard-inspection-v1" => {
            if seed.is_some()
                || cfg.is_some()
                || rope.is_some()
                || decode
                || text
                || stochastic
                || positive_cfg
                || top_p
                || host_fp64
            {
                return Err("unsupported option for Sudoku".into());
            }
            result[1] = "generate".into();
        }
        "tinystories-inference-v1" => {
            if cfg.is_some() || rope.is_some() || stochastic || positive_cfg || top_p || host_fp64 {
                return Err("unsupported option for TinyStories".into());
            }
            result[1] = "sample-tinystories".into();
            result.push(seed.unwrap_or(0).to_string());
        }
        "xsum-inference-v1" | "xsum-inference-v2" => {
            if rope.is_some() || top_p || host_fp64 || (seed.is_some() != stochastic) {
                return Err("unsupported XSum option; stochastic sampling requires both --seed and --experimental-stochastic-xsum".into());
            }
            if (cfg.unwrap_or(0.0) > 0.0) != positive_cfg {
                return Err("positive XSum CFG requires --experimental-positive-cfg-xsum".into());
            }
            result[1] = "sample-xsum".into();
            result.push(cfg.unwrap_or(0.0).to_string());
            if let Some(seed) = seed {
                result.push(seed.to_string());
            }
        }
        "wmt14-inference-v1" => {
            if seed.is_some() || rope.is_some() || stochastic || positive_cfg || top_p || host_fp64
            {
                return Err("unsupported option for WMT14".into());
            }
            validate_wmt14_cfg(cfg.unwrap_or(0.0), wmt14_experimental)?;
            result[1] = "sample-wmt14".into();
            result.push(cfg.unwrap_or(0.0).to_string());
        }
        "lm1b-small-inference-v1"
        | "lm1b-medium-inference-v1"
        | "lm1b-large-inference-v1"
        | "lm1b-small-inference-v2"
        | "lm1b-medium-inference-v2"
        | "lm1b-large-inference-v2" => {
            if cfg.is_some() || stochastic || positive_cfg {
                return Err("unsupported option for LM1B".into());
            }
            result[1] = "sample-lm1b".into();
            result.push(seed.unwrap_or(0).to_string());
            let embedded = profile.ends_with("-v2");
            let default_rope = Path::new(&args[2])
                .join("..")
                .join("lm1b-reference-rope-v1.safetensors");
            result.push(match rope {
                Some(path) => path.clone(),
                None if embedded => "--embedded-rope".into(),
                None => default_rope
                    .to_str()
                    .ok_or("LM1B default rotary path is not valid UTF-8")?
                    .to_owned(),
            });
            if decode {
                result.push("--decode".into());
            }
            if top_p {
                result.push("--top-p-fp32".into());
            }
            if host_fp64 {
                result.push("--host-fp64".into());
            }
        }
        "local-ncm-v1" => {
            if cfg.is_some() || rope.is_some() || stochastic || positive_cfg || top_p || host_fp64 {
                return Err("unsupported option for local-ncm-v1".into());
            }
            result[1] = "sample-local".into();
            if let Some(seed) = seed {
                result.push(seed.to_string());
            }
        }
        _ => return Err(format!("unsupported artifact profile: {profile}").into()),
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{translate_profile, validate_wmt14_alias, validate_xsum_alias};

    fn args(options: &[&str]) -> Vec<String> {
        ["ncm", "sample", "artifact", "cuda-cudnn", "input"]
            .into_iter()
            .chain(options.iter().copied())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn xsum_alias_requires_explicit_experimental_opt_in() {
        let alias = |tail: &[&str]| {
            ["ncm", "sample-xsum", "artifact", "cuda-cudnn", "input"]
                .into_iter()
                .chain(tail.iter().copied())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        for tail in [
            &["1"][..],
            &["0", "7"],
            &["1", "--experimental-stochastic-xsum"],
            &["0", "--experimental-positive-cfg-xsum"],
            &["NaN"],
            &[
                "0",
                "7",
                "--experimental-stochastic-xsum",
                "--experimental-stochastic-xsum",
            ],
        ] {
            assert!(validate_xsum_alias(&mut alias(tail)).is_err(), "{tail:?}");
        }
        for tail in [
            &[][..],
            &["0"],
            &["1", "--experimental-positive-cfg-xsum"],
            &["0", "7", "--experimental-stochastic-xsum"],
            &[
                "1",
                "7",
                "--experimental-stochastic-xsum",
                "--experimental-positive-cfg-xsum",
            ],
        ] {
            let mut args = alias(tail);
            validate_xsum_alias(&mut args).unwrap();
            assert!(args[5..].iter().all(|arg| !arg.starts_with("--")));
        }
    }

    #[test]
    fn wmt14_cfg_boundary_applies_to_both_routes() {
        for (scale, flag, accepted) in [
            ("0", false, true),
            ("1", false, true),
            ("1.0", false, true),
            ("0.5", false, false),
            ("2", false, false),
            ("2", true, true),
            ("0.5", true, true),
            ("0", true, false),
            ("1", true, false),
            ("NaN", false, false),
            ("-1", true, false),
        ] {
            let mut options = vec!["--cfg", scale];
            if flag {
                options.push("--experimental-cfg-wmt14");
            }
            assert_eq!(
                translate_profile(&args(&options), "wmt14-inference-v1").is_ok(),
                accepted,
                "unified {scale} {flag}"
            );
            let mut alias: Vec<String> =
                ["ncm", "sample-wmt14", "artifact", "cuda", "input", scale]
                    .into_iter()
                    .map(str::to_owned)
                    .collect();
            if flag {
                alias.push("--experimental-cfg-wmt14".into());
            }
            assert_eq!(
                validate_wmt14_alias(&mut alias).is_ok(),
                accepted,
                "alias {scale} {flag}"
            );
            if accepted {
                assert_eq!(alias.len(), 6);
                assert_eq!(alias[5], scale);
            }
        }
        assert!(
            translate_profile(&args(&["--experimental-cfg-wmt14"]), "xsum-inference-v2").is_err()
        );
        let mut duplicate: Vec<String> = [
            "ncm",
            "sample-wmt14",
            "a",
            "cuda",
            "i",
            "2",
            "--experimental-cfg-wmt14",
            "--experimental-cfg-wmt14",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        assert!(validate_wmt14_alias(&mut duplicate).is_err());
    }

    #[test]
    fn routes_profiles_and_preserves_legacy_arguments() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            ("sudoku-easy-inspection-v1", &[], &["generate"]),
            (
                "tinystories-inference-v1",
                &["--seed", "42"],
                &["sample-tinystories", "42"],
            ),
            ("xsum-inference-v2", &[], &["sample-xsum", "0"]),
            (
                "wmt14-inference-v1",
                &["--cfg", "1"],
                &["sample-wmt14", "1"],
            ),
            (
                "lm1b-small-inference-v1",
                &["--reference-rope", "rope", "--decode"],
                &["sample-lm1b", "0", "rope", "--decode"],
            ),
            (
                "lm1b-medium-inference-v1",
                &["--reference-rope", "rope"],
                &["sample-lm1b", "0", "rope"],
            ),
            (
                "lm1b-large-inference-v1",
                &["--reference-rope", "rope"],
                &["sample-lm1b", "0", "rope"],
            ),
        ];
        for (profile, options, expected) in cases {
            let translated = translate_profile(&args(options), profile).unwrap();
            assert_eq!(&translated[1], expected[0]);
            assert_eq!(&translated[5..], &expected[1..]);
        }
    }

    #[test]
    fn rejects_ambiguous_or_unqualified_options() {
        for options in [
            &["--seed", "1"][..],
            &["--cfg", "1"],
            &["--cfg", "NaN"],
            &["--seed", "1", "--seed", "2"],
        ] {
            assert!(translate_profile(&args(options), "xsum-inference-v2").is_err());
        }
        for profile in [
            "tinystories-inference-v1",
            "xsum-inference-v2",
            "wmt14-inference-v1",
        ] {
            assert!(translate_profile(&args(&["--decode"]), profile).is_ok());
        }
        assert!(translate_profile(&args(&["--decode"]), "sudoku-easy-inspection-v1").is_err());
        for profile in [
            "tinystories-inference-v1",
            "xsum-inference-v2",
            "wmt14-inference-v1",
        ] {
            assert!(translate_profile(&args(&["--text", "--decode"]), profile).is_ok());
            assert!(translate_profile(&args(&["--text", "--text"]), profile).is_err());
        }
        assert!(translate_profile(&args(&["--text"]), "sudoku-easy-inspection-v1").is_err());
        assert!(
            translate_profile(
                &args(&["--text", "--reference-rope", "rope"]),
                "lm1b-small-inference-v1"
            )
            .is_ok()
        );
        assert!(translate_profile(&args(&[]), "owt-inference-v1").is_err());
        let default_rope = translate_profile(&args(&[]), "lm1b-small-inference-v1").unwrap();
        assert_eq!(
            default_rope[6],
            "artifact/../lm1b-reference-rope-v1.safetensors"
        );
        for v2 in [
            "lm1b-small-inference-v2",
            "lm1b-medium-inference-v2",
            "lm1b-large-inference-v2",
        ] {
            let embedded = translate_profile(&args(&["--seed", "7"]), v2).unwrap();
            assert_eq!(embedded[1], "sample-lm1b");
            assert_eq!(embedded[6], "--embedded-rope");
            let explicit =
                translate_profile(&args(&["--seed", "7", "--reference-rope", "custom"]), v2)
                    .unwrap();
            assert_eq!(explicit[6], "custom");
        }
        assert!(
            translate_profile(
                &args(&["--seed", "1", "--experimental-stochastic-xsum"]),
                "xsum-inference-v2"
            )
            .is_ok()
        );
        assert!(
            translate_profile(
                &args(&["--cfg", "1", "--experimental-positive-cfg-xsum"]),
                "xsum-inference-v2"
            )
            .is_ok()
        );
        assert_eq!(
            translate_profile(&args(&[]), "local-ncm-v1").unwrap(),
            ["ncm", "sample-local", "artifact", "cuda-cudnn", "input"]
        );
        assert_eq!(
            translate_profile(
                &args(&["--seed", "42", "--text", "--decode"]),
                "local-ncm-v1"
            )
            .unwrap(),
            [
                "ncm",
                "sample-local",
                "artifact",
                "cuda-cudnn",
                "input",
                "42"
            ]
        );
        for bad in [
            &["--cfg", "1"][..],
            &["--reference-rope", "r"][..],
            &["--top-p-fp32"][..],
            &["--host-fp64"][..],
        ] {
            assert!(translate_profile(&args(bad), "local-ncm-v1").is_err());
        }
    }
}
