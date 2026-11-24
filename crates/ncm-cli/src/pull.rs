//! Pure-Rust, zero-Python `ncm pull` implementation: verifies pinned upstream files,
//! fetches missing checkpoints/tokenizers over HTTPS via `curl`, parses PyTorch `.pt`
//! archives & YAML configs natively, builds bit-exact canonical tokenizer assets,
//! and exports verified `safetensors` + `manifest.json` artifacts atomically.

use crate::Result;
use ncm_io::{
    Artifact, CHECKPOINT_REVISION, CHECKPOINT_REVISION_2026_11_23, FileRecord, Manifest,
    TensorRecord, UPSTREAM, UPSTREAM_2026_11_23,
    pytorch::{
        OwnedRawTensor, PickleValue, PtCheckpoint, PtDtype, PtTensorDesc, parse_data_pickle,
        read_safetensors_raw, sha256_file, write_checked_safetensors_from_pt,
    },
    yaml::parse_yaml_to_json,
};
use ncm_tokenizers::{
    TextTokenizer, build_hf_tokenizer_asset, build_tinystories_tokenizer_asset, format_python_json,
};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const REPOSITORY: &str = "hazyresearch/ncm-checkpoints";
const LM1B_ROPE_BYTES: u64 = 32_920;
const LM1B_ROPE_SHA256: &str = "9dd9537934ccf599764488829b74a0cf1498bae6efc05b8f229005c94f0711bc";
const XSUM_ROPE_BYTES: u64 = 131_216;
const XSUM_ROPE_SHA256: &str = "5d205c1c42b467c00e10d03739e18b6433185bcc4c4f9b066d7c6be69ba8346a";
const XSUM_ROPE_COS_SHA256: &str =
    "f1376ecd82c6f574ade39abbdc95db403338b466306c02062a260079116c7ed6";
const XSUM_ROPE_SIN_SHA256: &str =
    "81716ee36cb0d78e78c093171a1751124c57467459608f5d810b3a163866e523";

/// Pinned MIT license text from `HazyResearch/ncm@c7027427` (SHA-256 `2b379bec...`).
const UPSTREAM_NCM_MIT_LICENSE_URL: &str = "https://raw.githubusercontent.com/HazyResearch/ncm/c70274273f489abc73832b2b98584a8483b6b203/LICENSE";
const UPSTREAM_NCM_MIT_LICENSE_BYTES: u64 = 1072;
const UPSTREAM_NCM_MIT_LICENSE_SHA256: &str =
    "2b379bec6e8f7f362eef851d10a2afc4171b0e96a3525d50350e2cf783f2c441";

#[derive(Debug, Clone, Copy)]
pub struct PinnedFile {
    pub local_rel: &'static str,
    pub remote_rel: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Sudoku(&'static str),
    TinyStories { is_15k_ema: bool },
    XSum,
    Wmt14,
    Lm1b(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub struct PullSpec {
    pub name: &'static str,
    pub kind: ModelKind,
    pub fetch_dir: &'static str,
    pub output_name: &'static str,
    pub tokenizer_profile: Option<&'static str>,
    pub upstream_commit: &'static str,
    pub revision: &'static str,
    pub files: &'static [PinnedFile],
    pub description: &'static str,
}

const SUDOKU_EASY_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "config.yaml",
        remote_rel: "sudoku/easy/config.yaml",
        bytes: 1980,
        sha256: "68d74fe310e909ca1892338258a2d7143293e685ec1c4747265d9134013d1605",
    },
    PinnedFile {
        local_rel: "generator.pt",
        remote_rel: "sudoku/easy/generator.pt",
        bytes: 102_560_203,
        sha256: "1cefebcd1eae0e210b7e991fdce0e86964d0e897e94a3b432d1008ab92d9926e",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "sudoku/easy/tokenizer.pt",
        bytes: 59_816_209,
        sha256: "3f9c87e99fd49900929cd650181f55bf00530280416992b11ddc4ea869b39d68",
    },
];

const SUDOKU_MEDIUM_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "config.yaml",
        remote_rel: "sudoku/medium/config.yaml",
        bytes: 1982,
        sha256: "a399820c7adbfd0c6ad6e443a63feb7fe7d35d066fb2981b79440af0b237beec",
    },
    PinnedFile {
        local_rel: "generator.pt",
        remote_rel: "sudoku/medium/generator.pt",
        bytes: 102_560_203,
        sha256: "8336b53fa3dcbddd9375374b8b7409feba2c2ae3b5a1fc5167ecd7852b26ae17",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "sudoku/medium/tokenizer.pt",
        bytes: 59_816_209,
        sha256: "aca5523a0c477a6f142bfd6c2ba157353c01b1e3f072bb0f75e34b3b75ebc211",
    },
];

const SUDOKU_HARD_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "config.yaml",
        remote_rel: "sudoku/hard/config.yaml",
        bytes: 1980,
        sha256: "8d58c49e6fe52e60191e06df4061937d5bb84f97c5ffbcd93ed9830c32f8df2f",
    },
    PinnedFile {
        local_rel: "generator.pt",
        remote_rel: "sudoku/hard/generator.pt",
        bytes: 102_560_203,
        sha256: "9e0bfe44abc1ca3930b7a63cd56cd0b9564b95e29edfe12dd0903207b89df8b8",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "sudoku/hard/tokenizer.pt",
        bytes: 59_816_209,
        sha256: "3bc6aecad8e15820f9d9bb09750e3c59bdf91ca953f1f11dbd109aaed1d80c0b",
    },
];

const TINYSTORIES_15K_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "bpe_merges_8192.pickle",
        remote_rel: "tinystories/bpe_merges_8192.pickle",
        bytes: 108_277,
        sha256: "0d4cab13f2e7b684b75eaa290a52e596ba87744ef13d943cf3a50a5a8e080ec1",
    },
    PinnedFile {
        local_rel: "bpe_vocab_8192.pickle",
        remote_rel: "tinystories/bpe_vocab_8192.pickle",
        bytes: 95_122,
        sha256: "fb176b4c68108e4b5ef61b9ff2af3731600124e3e160970a6bd78cd0a5d3ff6c",
    },
    PinnedFile {
        local_rel: "config.yaml",
        remote_rel: "tinystories/config.yaml",
        bytes: 2242,
        sha256: "efdd98b675a24f956e9b832d03a2a2e88db05931d434090ef65f0f34bcb2cd9c",
    },
    PinnedFile {
        local_rel: "generator.pt",
        remote_rel: "tinystories/generator.pt",
        bytes: 356_150_923,
        sha256: "9d66ab61de9223a5229175fea730747d03acdb1794a37038e7a0d771a45ca875",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "tinystories/tokenizer.pt",
        bytes: 221_863_115,
        sha256: "85ce688e0161f6e1a785d549e57d8da65ea67a40543a8fc7a67934b68550eac5",
    },
];

const TINYSTORIES_LEGACY_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "bpe_merges_8192.pickle",
        remote_rel: "tinystories/bpe_merges_8192.pickle",
        bytes: 108_277,
        sha256: "0d4cab13f2e7b684b75eaa290a52e596ba87744ef13d943cf3a50a5a8e080ec1",
    },
    PinnedFile {
        local_rel: "bpe_vocab_8192.pickle",
        remote_rel: "tinystories/bpe_vocab_8192.pickle",
        bytes: 95_122,
        sha256: "fb176b4c68108e4b5ef61b9ff2af3731600124e3e160970a6bd78cd0a5d3ff6c",
    },
    PinnedFile {
        local_rel: "config.yaml",
        remote_rel: "tinystories/config.yaml",
        bytes: 2483,
        sha256: "4009baccad5a159bfc12cd48f3da17c22c4a281b86f71f2fc70b7c1bbb62004b",
    },
    PinnedFile {
        local_rel: "generator.pt",
        remote_rel: "tinystories/generator.pt",
        bytes: 356_160_651,
        sha256: "0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "tinystories/tokenizer.pt",
        bytes: 221_861_835,
        sha256: "76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81",
    },
];

const XSUM_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "config.yaml",
        remote_rel: "xsum/config.yaml",
        bytes: 2345,
        sha256: "5dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493",
    },
    PinnedFile {
        local_rel: "generator.pt",
        remote_rel: "xsum/generator.pt",
        bytes: 354_355_211,
        sha256: "f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "xsum/tokenizer.pt",
        bytes: 299_626_079,
        sha256: "1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7",
    },
];

const WMT14_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "config.yaml",
        remote_rel: "wmt14/config.yaml",
        bytes: 2177,
        sha256: "6401af2a14d41c6582c3181a0b683e8970c1eeefe8cf7afaf74f1597da3fcd72",
    },
    PinnedFile {
        local_rel: "generator.pt",
        remote_rel: "wmt14/generator.pt",
        bytes: 480_582_539,
        sha256: "ae580cbbb400a27b30c616634745d4c9cfa8b66806f936c6c26041e5b9640d4f",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "wmt14/tokenizer.pt",
        bytes: 645_790_351,
        sha256: "74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702",
    },
];

const LM1B_SMALL_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "small/config.yaml",
        remote_rel: "lm1b/small/config.yaml",
        bytes: 2749,
        sha256: "4db04299a87f7eb1627de0abb2f0b257e50e5ea0138edda771ad287cdabd7edf",
    },
    PinnedFile {
        local_rel: "small/generator.pt",
        remote_rel: "lm1b/small/generator.pt",
        bytes: 390_138_443,
        sha256: "b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "lm1b/tokenizer.pt",
        bytes: 597_840_331,
        sha256: "d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c",
    },
];

const LM1B_MEDIUM_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "medium/config.yaml",
        remote_rel: "lm1b/medium/config.yaml",
        bytes: 2754,
        sha256: "93a42e0923b38f06f186072fdc37c1a2b4db5298c7577755390dd36c651d8142",
    },
    PinnedFile {
        local_rel: "medium/generator.pt",
        remote_rel: "lm1b/medium/generator.pt",
        bytes: 1_268_014_283,
        sha256: "5aa4621472c2e658ab343e50d98abc808e4c2c6dd26eaf8535897b6a540c1604",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "lm1b/tokenizer.pt",
        bytes: 597_840_331,
        sha256: "d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c",
    },
];

const LM1B_LARGE_FILES: &[PinnedFile] = &[
    PinnedFile {
        local_rel: "large/config.yaml",
        remote_rel: "lm1b/large/config.yaml",
        bytes: 2753,
        sha256: "4410e3710360cae162f8be4186a44a0c7c473be674a1ac3d1d940bfe45d5e70c",
    },
    PinnedFile {
        local_rel: "large/generator.pt",
        remote_rel: "lm1b/large/generator.pt",
        bytes: 2_901_184_267,
        sha256: "ec34d6c56a81f428303b2fa3a8e1615ceee81b7dec7f59380427f46808ec224c",
    },
    PinnedFile {
        local_rel: "tokenizer.pt",
        remote_rel: "lm1b/tokenizer.pt",
        bytes: 597_840_331,
        sha256: "d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c",
    },
];

pub const PULL_SPECS: &[PullSpec] = &[
    PullSpec {
        name: "tinystories",
        kind: ModelKind::TinyStories { is_15k_ema: true },
        fetch_dir: "tinystories-15k-ema",
        output_name: "tinystories-quickstart",
        tokenizer_profile: Some("tinystories"),
        upstream_commit: UPSTREAM_2026_11_23,
        revision: CHECKPOINT_REVISION_2026_11_23,
        files: TINYSTORIES_15K_FILES,
        description: "TinyStories 15k EMA text generator (~552 MiB exported; use --precision bf16, q8, q4, or fp32)",
    },
    PullSpec {
        name: "tinystories-legacy",
        kind: ModelKind::TinyStories { is_15k_ema: false },
        fetch_dir: "tinystories",
        output_name: "tinystories-legacy-quickstart",
        tokenizer_profile: Some("tinystories"),
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: TINYSTORIES_LEGACY_FILES,
        description: "TinyStories legacy d342ce8a text generator (~552 MiB exported; use --precision bf16, q8, q4, or fp32)",
    },
    PullSpec {
        name: "xsum",
        kind: ModelKind::XSum,
        fetch_dir: "xsum",
        output_name: "xsum-quickstart",
        tokenizer_profile: Some("gpt2"),
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: XSUM_FILES,
        description: "XSum BBC news summarizer (~627 MiB exported; great with --precision fp16)",
    },
    PullSpec {
        name: "wmt14",
        kind: ModelKind::Wmt14,
        fetch_dir: "wmt14",
        output_name: "wmt14-quickstart",
        tokenizer_profile: Some("qwen3"),
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: WMT14_FILES,
        description: "WMT14 German-to-English translator (~1.06 GiB exported; great with --precision fp16)",
    },
    PullSpec {
        name: "lm1b-small",
        kind: ModelKind::Lm1b("small"),
        fetch_dir: "lm1b-small",
        output_name: "lm1b-small-quickstart",
        tokenizer_profile: Some("bert"),
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: LM1B_SMALL_FILES,
        description: "LM1B Small unconditional language model (~943 MiB exported)",
    },
    PullSpec {
        name: "lm1b-medium",
        kind: ModelKind::Lm1b("medium"),
        fetch_dir: "lm1b-medium",
        output_name: "lm1b-medium-quickstart",
        tokenizer_profile: Some("bert"),
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: LM1B_MEDIUM_FILES,
        description: "LM1B Medium unconditional language model (~1.74 GiB exported)",
    },
    PullSpec {
        name: "lm1b-large",
        kind: ModelKind::Lm1b("large"),
        fetch_dir: "lm1b-large",
        output_name: "lm1b-large-quickstart",
        tokenizer_profile: Some("bert"),
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: LM1B_LARGE_FILES,
        description: "LM1B Large unconditional language model (~3.26 GiB exported; fits b=16 in ~1 GiB VRAM with --precision q4)",
    },
    PullSpec {
        name: "sudoku-easy",
        kind: ModelKind::Sudoku("easy"),
        fetch_dir: "sudoku-easy",
        output_name: "sudoku-easy-quickstart",
        tokenizer_profile: None,
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: SUDOKU_EASY_FILES,
        description: "Sudoku-easy solver (~155 MiB exported)",
    },
    PullSpec {
        name: "sudoku-medium",
        kind: ModelKind::Sudoku("medium"),
        fetch_dir: "sudoku-medium",
        output_name: "sudoku-medium-quickstart",
        tokenizer_profile: None,
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: SUDOKU_MEDIUM_FILES,
        description: "Sudoku-medium solver (~155 MiB exported)",
    },
    PullSpec {
        name: "sudoku-hard",
        kind: ModelKind::Sudoku("hard"),
        fetch_dir: "sudoku-hard",
        output_name: "sudoku-hard-quickstart",
        tokenizer_profile: None,
        upstream_commit: UPSTREAM,
        revision: CHECKPOINT_REVISION,
        files: SUDOKU_HARD_FILES,
        description: "Sudoku-hard solver (~155 MiB exported)",
    },
];

pub fn find_spec(model: &str) -> Result<&'static PullSpec> {
    PULL_SPECS.iter().find(|s| s.name == model).ok_or_else(|| {
        let names: Vec<&str> = PULL_SPECS.iter().map(|s| s.name).collect();
        format!(
            "unsupported pull model `{model}`; choose from: {}",
            names.join(", ")
        )
        .into()
    })
}

pub fn print_model_list() {
    println!("Available models (pure-Rust verified ncm pull):");
    for spec in PULL_SPECS {
        let total_mb = spec.files.iter().map(|f| f.bytes).sum::<u64>() as f64 / (1024.0 * 1024.0);
        println!(
            "  {:<20} ({:>6.1} MiB download)  {}",
            spec.name, total_mb, spec.description
        );
    }
}

/// Options for pure-Rust `ncm pull`.
#[derive(Debug, Clone)]
pub struct NativePullOptions {
    pub model: String,
    pub models_dir: PathBuf,
    pub upstream_dir: PathBuf,
    pub output_override: Option<PathBuf>,
    pub reference_rope: Option<PathBuf>,
    pub dry_run: bool,
}

fn verify_exact_file(path: &Path, expected_bytes: u64, expected_sha: &str) -> Result<()> {
    let (actual_bytes, actual_sha) = sha256_file(path)?;
    if actual_bytes != expected_bytes || actual_sha != expected_sha {
        return Err(format!(
            "integrity mismatch on {}: got ({actual_bytes} bytes, {actual_sha}), expected ({expected_bytes} bytes, {expected_sha})",
            path.display()
        )
        .into());
    }
    Ok(())
}

/// Download a single verified HTTPS file via `curl`, enforcing exact byte size and SHA-256.
fn fetch_verified_https(
    url: &str,
    dest: &Path,
    expected_bytes: u64,
    expected_sha: &str,
) -> Result<()> {
    if !url.starts_with("https://") {
        return Err(format!("refusing non-HTTPS URL: {url}").into());
    }
    if dest.exists() || dest.is_symlink() {
        return verify_exact_file(dest, expected_bytes, expected_sha);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("part");
    let _ = fs::remove_file(&tmp);
    eprintln!(
        "  downloading {} ({:.1} MiB)...",
        url,
        expected_bytes as f64 / (1024.0 * 1024.0)
    );
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--max-filesize",
            &expected_bytes.to_string(),
            "-o",
        ])
        .arg(&tmp)
        .arg(url)
        .status()
        .map_err(|e| format!("failed to launch `curl` for {url}: {e}"))?;
    if !status.success() {
        let _ = fs::remove_file(&tmp);
        return Err(format!("curl failed ({status}) fetching {url}").into());
    }
    if let Err(e) = verify_exact_file(&tmp, expected_bytes, expected_sha) {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    fs::rename(&tmp, dest)?;
    Ok(())
}

fn ensure_checkpoint_files(spec: &PullSpec, ckpt_dir: &Path, dry_run: bool) -> Result<()> {
    for f in spec.files {
        let dest = ckpt_dir.join(f.local_rel);
        let url = format!(
            "https://huggingface.co/{REPOSITORY}/resolve/{}/{}",
            spec.revision, f.remote_rel
        );
        if dest.exists() {
            if dry_run {
                println!("[cached] {} ({} bytes)", dest.display(), f.bytes);
            } else {
                verify_exact_file(&dest, f.bytes, f.sha256)?;
            }
        } else if dry_run {
            println!("[fetch]  {url} -> {} ({} bytes)", dest.display(), f.bytes);
        } else {
            fetch_verified_https(&url, &dest, f.bytes, f.sha256)?;
        }
    }
    Ok(())
}

fn ensure_canonical_tokenizer(
    profile: &str,
    ckpt_dir: &Path,
    upstream_dir: &Path,
    models_dir: &Path,
    dry_run: bool,
) -> Result<PathBuf> {
    let tok_dir = models_dir.join("tokenizers-qualified").join(profile);
    if tok_dir.is_dir() && TextTokenizer::load(&tok_dir).is_ok() {
        if dry_run {
            println!("[cached] canonical tokenizer {}", tok_dir.display());
        }
        return Ok(tok_dir);
    }
    if dry_run {
        println!(
            "[build]  pure-Rust canonical tokenizer `{profile}` -> {}",
            tok_dir.display()
        );
        return Ok(tok_dir);
    }

    let raw_tok_root = upstream_dir.join("tokenizers-v2").join(profile);
    let asset = if profile == "tinystories" {
        let vocab_path = ckpt_dir.join("bpe_vocab_8192.pickle");
        let merges_path = ckpt_dir.join("bpe_merges_8192.pickle");
        let lic_path = raw_tok_root.join("LICENSE");
        fetch_verified_https(
            UPSTREAM_NCM_MIT_LICENSE_URL,
            &lic_path,
            UPSTREAM_NCM_MIT_LICENSE_BYTES,
            UPSTREAM_NCM_MIT_LICENSE_SHA256,
        )?;
        let vocab = decode_tinystories_vocab(&fs::read(&vocab_path)?)?;
        let merges = decode_tinystories_merges(&fs::read(&merges_path)?)?;
        let license = fs::read(&lic_path)?;
        build_tinystories_tokenizer_asset(&vocab, &merges, &license)?
    } else {
        let lock: Value = serde_json::from_str(include_str!(
            "../../../tools/reference/tokenizer-sources.lock.json"
        ))?;
        let src = lock["sources"]
            .get(profile)
            .ok_or_else(|| format!("unknown tokenizer source `{profile}`"))?;
        let repo = src["repository"].as_str().ok_or("missing repo")?;
        let rev = src["revision"].as_str().ok_or("missing revision")?;
        let files = src["files"].as_object().ok_or("missing files")?;
        let lic_key = if profile == "gpt2" {
            "OPENAI_LICENSE.txt"
        } else {
            "LICENSE"
        };
        for file_key in ["tokenizer.json", lic_key] {
            let rec = files
                .get(file_key)
                .ok_or_else(|| format!("missing `{file_key}` in tokenizer lock"))?;
            let bytes = rec["bytes"].as_u64().ok_or("missing bytes")?;
            let sha = rec["sha256"].as_str().ok_or("missing sha256")?;
            let url = rec
                .get("url")
                .and_then(|u| u.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    format!("https://huggingface.co/{repo}/resolve/{rev}/{file_key}")
                });
            let dest = raw_tok_root.join(file_key);
            fetch_verified_https(&url, &dest, bytes, sha)?;
        }
        let raw_tok = fs::read(raw_tok_root.join("tokenizer.json"))?;
        let raw_lic = fs::read(raw_tok_root.join(lic_key))?;
        build_hf_tokenizer_asset(profile, &raw_tok, &raw_lic)?
    };

    // Write canonical tokenizer directory atomically.
    if let Some(parent) = tok_dir.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_dir = tok_dir.with_extension(format!("tmp-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir)?;
    let write_res: Result<()> = (|| {
        fs::write(tmp_dir.join("manifest.json"), &asset.manifest_json)?;
        fs::write(tmp_dir.join("tokenizer.json"), &asset.tokenizer_json)?;
        fs::write(tmp_dir.join("LICENSE"), &asset.license)?;
        TextTokenizer::load(&tmp_dir)?;
        if tok_dir.exists() {
            return Err(format!(
                "canonical tokenizer destination already exists: {}",
                tok_dir.display()
            )
            .into());
        }
        fs::rename(&tmp_dir, &tok_dir)?;
        Ok(())
    })();
    if write_res.is_err() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    write_res?;
    Ok(tok_dir)
}

fn decode_tinystories_vocab(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    let val = parse_data_pickle(bytes)?;
    let PickleValue::Dict(pairs) = val else {
        return Err("TinyStories vocab pickle must be a dict".into());
    };
    let mut map: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    for (k, v) in pairs {
        let PickleValue::Int(id) = k else {
            return Err("TinyStories vocab key must be int".into());
        };
        let PickleValue::Bytes(b) = v else {
            return Err("TinyStories vocab value must be bytes".into());
        };
        let idx = usize::try_from(id).map_err(|_| "negative vocab ID")?;
        if map.insert(idx, b).is_some() {
            return Err(format!("duplicate TinyStories vocab ID {idx}").into());
        }
    }
    if map.len() != 8192 {
        return Err(format!("TinyStories vocab size {} != 8192", map.len()).into());
    }
    let mut out = Vec::with_capacity(8192);
    for i in 0..8192 {
        out.push(
            map.remove(&i)
                .ok_or_else(|| format!("missing TinyStories vocab ID {i}"))?,
        );
    }
    Ok(out)
}

fn decode_tinystories_merges(bytes: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let val = parse_data_pickle(bytes)?;
    let PickleValue::List(items) = val else {
        return Err("TinyStories merges pickle must be a list".into());
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let PickleValue::Tuple(mut pair) = item else {
            return Err("TinyStories merge entry must be tuple".into());
        };
        if pair.len() != 2 {
            return Err("TinyStories merge tuple length != 2".into());
        }
        let PickleValue::Bytes(b) = pair.pop().unwrap() else {
            return Err("TinyStories merge rhs must be bytes".into());
        };
        let PickleValue::Bytes(a) = pair.pop().unwrap() else {
            return Err("TinyStories merge lhs must be bytes".into());
        };
        out.push((a, b));
    }
    Ok(out)
}

fn is_training_tensor(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("vq.") else {
        return false;
    };
    for prefix in ["ema_cluster_size_", "ema_w_", "codebook_hit_"] {
        if let Some(digits) = rest.strip_prefix(prefix)
            && !digits.is_empty()
            && digits.bytes().all(|b| b.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

fn validate_prob_dist(
    name: &str,
    desc: &PtTensorDesc,
    raw: &[u8],
    expected_len: usize,
) -> Result<()> {
    if desc.dtype != PtDtype::F32 || desc.shape != [expected_len] || raw.len() != expected_len * 4 {
        return Err(format!("invalid distribution tensor `{name}` shape/dtype").into());
    }
    let mut sum = 0.0f64;
    for chunk in raw.as_chunks::<4>().0 {
        let v = f32::from_le_bytes(*chunk);
        if !v.is_finite() || v < 0.0 {
            return Err(format!("invalid non-finite or negative probability in `{name}`").into());
        }
        sum += v as f64;
    }
    if (sum - 1.0).abs() > 1e-6 {
        return Err(format!("distribution `{name}` sums to {sum} (expected 1.0)").into());
    }
    Ok(())
}

fn validate_f32_or_i64(map: &BTreeMap<String, PtTensorDesc>) -> Result<()> {
    for (name, desc) in map {
        if !matches!(desc.dtype, PtDtype::F32 | PtDtype::I64) {
            return Err(format!("unsupported tensor dtype {:?} for `{name}`", desc.dtype).into());
        }
    }
    Ok(())
}

/// Export one verified upstream checkpoint directory into a verified canonical NCM artifact
/// directory entirely in pure Rust (zero Python).
pub fn export_native_from_cache(
    spec: &PullSpec,
    ckpt_dir: &Path,
    tokenizer_dir: Option<&Path>,
    out_dir: &Path,
    reference_rope: Option<&Path>,
) -> Result<Value> {
    if out_dir.exists() || out_dir.is_symlink() {
        return Err(format!("output already exists: {}", out_dir.display()).into());
    }
    // Re-verify all pinned source inputs before reading.
    let mut source_hashes = Map::new();
    for f in spec.files {
        verify_exact_file(&ckpt_dir.join(f.local_rel), f.bytes, f.sha256)?;
        source_hashes.insert(f.local_rel.to_owned(), Value::String(f.sha256.to_owned()));
    }

    let (cfg_rel, gen_rel, tok_rel) = match spec.kind {
        ModelKind::Lm1b(size) => (
            format!("{size}/config.yaml"),
            format!("{size}/generator.pt"),
            "tokenizer.pt".to_owned(),
        ),
        _ => (
            "config.yaml".to_owned(),
            "generator.pt".to_owned(),
            "tokenizer.pt".to_owned(),
        ),
    };

    let yaml_text = fs::read_to_string(ckpt_dir.join(&cfg_rel))?;
    let config_val = parse_yaml_to_json(&yaml_text)?;
    let expected_task = match spec.kind {
        ModelKind::Sudoku(diff) => format!("sudoku-{diff}"),
        ModelKind::TinyStories { .. } => "tinystories".to_owned(),
        ModelKind::XSum => "xsum".to_owned(),
        ModelKind::Wmt14 => "wmt14".to_owned(),
        ModelKind::Lm1b(size) => format!("lm1b-{size}"),
    };
    if config_val.get("task").and_then(|v| v.as_str()) != Some(expected_task.as_str()) {
        return Err(format!("config.yaml task != expected `{expected_task}`").into());
    }

    let mut gen_ckpt = PtCheckpoint::open(&ckpt_dir.join(&gen_rel))?;
    let mut tok_ckpt = PtCheckpoint::open(&ckpt_dir.join(&tok_rel))?;

    let parent = out_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let tmp_dir = parent.join(format!(".ncm-pull-{}-{}", spec.name, std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir)?;

    let export_result: Result<Value> = (|| {
        let mut inventory: BTreeMap<String, Vec<TensorRecord>> = BTreeMap::new();
        let profile_name: String;
        let mut inference_meta: Vec<&str> = Vec::new();

        match spec.kind {
            ModelKind::Sudoku(diff) => {
                profile_name = format!("sudoku-{diff}-inspection-v1");
                if gen_ckpt.top_keys() != BTreeSet::from(["model".to_owned()])
                    || tok_ckpt.top_keys() != BTreeSet::from(["model".to_owned()])
                {
                    return Err("unexpected Sudoku checkpoint metadata".into());
                }
                let gen_t = gen_ckpt.tensor_dict("model")?;
                let tok_t = tok_ckpt.tensor_dict("model")?;
                validate_f32_or_i64(&gen_t)?;
                validate_f32_or_i64(&tok_t)?;
                let (train_t, codec_t): (BTreeMap<_, _>, BTreeMap<_, _>) =
                    tok_t.into_iter().partition(|(k, _)| is_training_tensor(k));
                inventory.insert(
                    "codec.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &codec_t,
                        None,
                        &tmp_dir.join("codec.safetensors"),
                    )?,
                );
                inventory.insert(
                    "training.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &train_t,
                        None,
                        &tmp_dir.join("training.safetensors"),
                    )?,
                );
                inventory.insert(
                    "generator.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut gen_ckpt,
                        &gen_t,
                        None,
                        &tmp_dir.join("generator.safetensors"),
                    )?,
                );
                let sudoku_tok = json!({
                    "blank_id": 0,
                    "bos_id": 11,
                    "eos_id": null,
                    "kind": "sudoku-v1",
                    "pad_id": null,
                    "prefix_length": 91,
                    "separator_id": 10,
                    "sequence_length": 89,
                    "vocabulary": ["_", "1", "2", "3", "4", "5", "6", "7", "8", "9", "|", "<BOS>"],
                });
                fs::write(
                    tmp_dir.join("tokenizer.json"),
                    format_python_json(&sudoku_tok),
                )?;
            }
            ModelKind::TinyStories { .. } => {
                profile_name = "tinystories-inference-v1".into();
                if gen_ckpt.top_keys()
                    != BTreeSet::from(["model".to_owned(), "inference".to_owned()])
                {
                    return Err("unexpected TinyStories generator keys".into());
                }
                let tok_keys = tok_ckpt.top_keys();
                if tok_keys != BTreeSet::from(["model".to_owned()])
                    && tok_keys != BTreeSet::from(["model".to_owned(), "inference".to_owned()])
                {
                    return Err("unexpected TinyStories tokenizer keys".into());
                }
                let gen_t = gen_ckpt.tensor_dict("model")?;
                let tok_t = tok_ckpt.tensor_dict("model")?;
                if gen_t.len() != 197 || tok_t.len() != 106 {
                    return Err("unexpected TinyStories tensor inventory count".into());
                }
                validate_f32_or_i64(&gen_t)?;
                validate_f32_or_i64(&tok_t)?;
                let inf_keys = gen_ckpt.sub_dict_keys("inference")?;
                if inf_keys != BTreeSet::from(["level0_unigram".to_owned()])
                    && inf_keys
                        != BTreeSet::from(["level0_unigram".to_owned(), "config".to_owned()])
                {
                    return Err("unexpected TinyStories inference metadata keys".into());
                }
                let unigram_desc = gen_ckpt.single_tensor_in_dict("inference", "level0_unigram")?;
                let unigram_bytes = gen_ckpt.read_tensor_bytes(&unigram_desc)?;
                validate_prob_dist("level0_unigram", &unigram_desc, &unigram_bytes, 4096)?;
                let mut inf_map = BTreeMap::new();
                inf_map.insert("level0_unigram".to_owned(), unigram_desc);

                let (train_t, codec_t): (BTreeMap<_, _>, BTreeMap<_, _>) =
                    tok_t.into_iter().partition(|(k, _)| is_training_tensor(k));
                if codec_t.len() != 70 || train_t.len() != 36 {
                    return Err("unexpected TinyStories codec/training split".into());
                }
                inventory.insert(
                    "codec.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &codec_t,
                        None,
                        &tmp_dir.join("codec.safetensors"),
                    )?,
                );
                inventory.insert(
                    "training.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &train_t,
                        None,
                        &tmp_dir.join("training.safetensors"),
                    )?,
                );
                inventory.insert(
                    "generator.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut gen_ckpt,
                        &gen_t,
                        None,
                        &tmp_dir.join("generator.safetensors"),
                    )?,
                );
                inventory.insert(
                    "inference.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut gen_ckpt,
                        &inf_map,
                        None,
                        &tmp_dir.join("inference.safetensors"),
                    )?,
                );
                inference_meta.push("level0_unigram");
            }
            ModelKind::XSum => {
                if gen_ckpt.top_keys()
                    != BTreeSet::from(["model".to_owned(), "inference".to_owned()])
                    || tok_ckpt.top_keys() != BTreeSet::from(["model".to_owned()])
                {
                    return Err("unexpected XSum checkpoint metadata".into());
                }
                let gen_t = gen_ckpt.tensor_dict("model")?;
                let tok_t = tok_ckpt.tensor_dict("model")?;
                let inf_t = gen_ckpt.tensor_dict("inference")?;
                if gen_t.len() != 194 || tok_t.len() != 80 || inf_t.len() != 1 {
                    return Err("unexpected XSum tensor inventory count".into());
                }
                validate_f32_or_i64(&gen_t)?;
                validate_f32_or_i64(&tok_t)?;
                let unigram_desc = inf_t
                    .get("token_unigram")
                    .ok_or("missing XSum token_unigram")?;
                let unigram_bytes = gen_ckpt.read_tensor_bytes(unigram_desc)?;
                validate_prob_dist("token_unigram", unigram_desc, &unigram_bytes, 50_304)?;

                let (train_t, codec_t): (BTreeMap<_, _>, BTreeMap<_, _>) =
                    tok_t.into_iter().partition(|(k, _)| is_training_tensor(k));
                if codec_t.len() != 53 || train_t.len() != 27 {
                    return Err("unexpected XSum codec/training split".into());
                }

                let mut extra_inf: Option<BTreeMap<String, OwnedRawTensor>> = None;
                inference_meta.push("token_unigram");
                if let Some(rope_path) = reference_rope {
                    let rope_file = if rope_path.is_dir() {
                        rope_path.join("rope.safetensors")
                    } else {
                        rope_path.to_path_buf()
                    };
                    verify_exact_file(&rope_file, XSUM_ROPE_BYTES, XSUM_ROPE_SHA256)?;
                    let rope = read_safetensors_raw(&rope_file)?;
                    let mut mapped = BTreeMap::new();
                    for (k, expected_sha) in
                        [("cos", XSUM_ROPE_COS_SHA256), ("sin", XSUM_ROPE_SIN_SHA256)]
                    {
                        let t = rope.get(k).ok_or("missing XSum rope tensor")?;
                        if t.dtype != PtDtype::F32
                            || t.shape != [512, 32]
                            || ncm_io::sha256(&t.bytes) != expected_sha
                        {
                            return Err(format!("invalid pinned XSum rope tensor `{k}`").into());
                        }
                        mapped.insert(format!("rope_{k}"), t.clone());
                    }
                    extra_inf = Some(mapped);
                    inference_meta.extend(["rope_cos", "rope_sin"]);
                    profile_name = "xsum-inference-v2".into();
                } else {
                    profile_name = "xsum-inference-v1".into();
                }

                inventory.insert(
                    "codec.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &codec_t,
                        None,
                        &tmp_dir.join("codec.safetensors"),
                    )?,
                );
                inventory.insert(
                    "training.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &train_t,
                        None,
                        &tmp_dir.join("training.safetensors"),
                    )?,
                );
                inventory.insert(
                    "generator.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut gen_ckpt,
                        &gen_t,
                        None,
                        &tmp_dir.join("generator.safetensors"),
                    )?,
                );
                inventory.insert(
                    "inference.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut gen_ckpt,
                        &inf_t,
                        extra_inf.as_ref(),
                        &tmp_dir.join("inference.safetensors"),
                    )?,
                );
            }
            ModelKind::Wmt14 => {
                profile_name = "wmt14-inference-v1".into();
                if gen_ckpt.top_keys() != BTreeSet::from(["model".to_owned()])
                    || tok_ckpt.top_keys() != BTreeSet::from(["model".to_owned()])
                {
                    return Err("unexpected WMT14 checkpoint metadata".into());
                }
                let gen_t = gen_ckpt.tensor_dict("model")?;
                let tok_t = tok_ckpt.tensor_dict("model")?;
                if gen_t.len() != 186 || tok_t.len() != 96 {
                    return Err("unexpected WMT14 tensor inventory count".into());
                }
                validate_f32_or_i64(&gen_t)?;
                validate_f32_or_i64(&tok_t)?;
                let (train_t, codec_t): (BTreeMap<_, _>, BTreeMap<_, _>) =
                    tok_t.into_iter().partition(|(k, _)| is_training_tensor(k));
                if train_t.len() != 33 || codec_t.len() != 63 {
                    return Err("unexpected WMT14 codec/training split".into());
                }
                inventory.insert(
                    "codec.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &codec_t,
                        None,
                        &tmp_dir.join("codec.safetensors"),
                    )?,
                );
                inventory.insert(
                    "training.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &train_t,
                        None,
                        &tmp_dir.join("training.safetensors"),
                    )?,
                );
                inventory.insert(
                    "generator.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut gen_ckpt,
                        &gen_t,
                        None,
                        &tmp_dir.join("generator.safetensors"),
                    )?,
                );
            }
            ModelKind::Lm1b(size) => {
                let expected_gen_count = match size {
                    "small" => 213,
                    "medium" => 381,
                    "large" => 549,
                    _ => return Err("unknown LM1B size".into()),
                };
                if gen_ckpt.top_keys()
                    != BTreeSet::from(["model".to_owned(), "inference".to_owned()])
                    || tok_ckpt.top_keys() != BTreeSet::from(["model".to_owned()])
                {
                    return Err("unexpected LM1B checkpoint metadata".into());
                }
                let gen_t = gen_ckpt.tensor_dict("model")?;
                let tok_t = tok_ckpt.tensor_dict("model")?;
                let inf_t = gen_ckpt.tensor_dict("inference")?;
                if gen_t.len() != expected_gen_count || tok_t.len() != 136 || inf_t.len() != 2 {
                    return Err("unexpected LM1B tensor inventory count".into());
                }
                validate_f32_or_i64(&gen_t)?;
                validate_f32_or_i64(&tok_t)?;
                for (name, width) in [("level0_unigram", 8192), ("token_unigram", 30_528)] {
                    let desc = inf_t
                        .get(name)
                        .ok_or_else(|| format!("missing LM1B inference tensor `{name}`"))?;
                    let bytes = gen_ckpt.read_tensor_bytes(desc)?;
                    validate_prob_dist(name, desc, &bytes, width)?;
                }
                let (train_t, codec_t): (BTreeMap<_, _>, BTreeMap<_, _>) =
                    tok_t.into_iter().partition(|(k, _)| is_training_tensor(k));
                if train_t.len() != 36 || codec_t.len() != 100 {
                    return Err("unexpected LM1B codec/training split".into());
                }
                let mut extra_inf: Option<BTreeMap<String, OwnedRawTensor>> = None;
                inference_meta.extend(["level0_unigram", "token_unigram"]);
                if let Some(rope_file) = reference_rope {
                    verify_exact_file(rope_file, LM1B_ROPE_BYTES, LM1B_ROPE_SHA256)?;
                    let rope = read_safetensors_raw(rope_file)?;
                    let cos = rope.get("cosine").ok_or("missing LM1B cosine")?;
                    let sin = rope.get("sine").ok_or("missing LM1B sine")?;
                    let mut mapped = BTreeMap::new();
                    mapped.insert("rope_cos".to_owned(), cos.clone());
                    mapped.insert("rope_sin".to_owned(), sin.clone());
                    extra_inf = Some(mapped);
                    inference_meta.extend(["rope_cos", "rope_sin"]);
                    profile_name = format!("lm1b-{size}-inference-v2");
                } else {
                    profile_name = format!("lm1b-{size}-inference-v1");
                }

                inventory.insert(
                    "codec.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &codec_t,
                        None,
                        &tmp_dir.join("codec.safetensors"),
                    )?,
                );
                inventory.insert(
                    "training.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut tok_ckpt,
                        &train_t,
                        None,
                        &tmp_dir.join("training.safetensors"),
                    )?,
                );
                inventory.insert(
                    "generator.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut gen_ckpt,
                        &gen_t,
                        None,
                        &tmp_dir.join("generator.safetensors"),
                    )?,
                );
                inventory.insert(
                    "inference.safetensors".into(),
                    write_checked_safetensors_from_pt(
                        &mut gen_ckpt,
                        &inf_t,
                        extra_inf.as_ref(),
                        &tmp_dir.join("inference.safetensors"),
                    )?,
                );
            }
        }

        // Copy authenticated canonical tokenizer files for text models.
        if let Some(tok_dir) = tokenizer_dir {
            TextTokenizer::load(tok_dir)?;
            fs::copy(
                tok_dir.join("tokenizer.json"),
                tmp_dir.join("tokenizer.json"),
            )?;
            fs::copy(
                tok_dir.join("manifest.json"),
                tmp_dir.join("tokenizer-manifest.json"),
            )?;
            fs::copy(tok_dir.join("LICENSE"), tmp_dir.join("tokenizer.LICENSE"))?;
        }

        fs::write(tmp_dir.join("config.json"), format_python_json(&config_val))?;

        let provenance = json!({
            "aliases": [],
            "code_license": "MIT",
            "exporter": format!("ncm-rs {} native pure-Rust pull", env!("CARGO_PKG_VERSION")),
            "inference_metadata": inference_meta,
            "serialization": "safetensors; unchanged upstream layouts/dtypes",
            "source_loading": "ncm-io pure-Rust ZIP_STORED + bounded safe pickle VM",
            "source_sha256": Value::Object(source_hashes),
            "training_state": "EMA/count/hit buffers separated, not dropped",
            "weights_license": "MIT",
        });
        fs::write(
            tmp_dir.join("provenance.json"),
            format_python_json(&provenance),
        )?;

        let mut files_map: BTreeMap<String, FileRecord> = BTreeMap::new();
        let mut names: Vec<String> = fs::read_dir(&tmp_dir)?
            .map(|e| Ok(e?.file_name().to_string_lossy().into_owned()))
            .collect::<std::io::Result<Vec<_>>>()?;
        names.sort();
        for name in names {
            let (sz, digest) = sha256_file(&tmp_dir.join(&name))?;
            files_map.insert(
                name,
                FileRecord {
                    bytes: sz,
                    sha256: digest,
                },
            );
        }

        let manifest = Manifest {
            schema_version: 1,
            profile: profile_name.clone(),
            upstream_commit: spec.upstream_commit.to_owned(),
            checkpoint_repository: REPOSITORY.to_owned(),
            checkpoint_revision: spec.revision.to_owned(),
            files: files_map,
            tensors: inventory,
        };
        let manifest_val = serde_json::to_value(&manifest)?;
        fs::write(
            tmp_dir.join("manifest.json"),
            format_python_json(&manifest_val),
        )?;

        // Full fail-closed streaming verification via ncm-io BEFORE atomic publish!
        Artifact::load_inference_streaming(&tmp_dir)?;

        if out_dir.exists() || out_dir.is_symlink() {
            return Err(format!("output already exists: {}", out_dir.display()).into());
        }
        fs::rename(&tmp_dir, out_dir)?;
        let (_, manifest_sha) = sha256_file(&out_dir.join("manifest.json"))?;
        Ok(json!({
            "bit_exact_roundtrip": true,
            "exporter": "native-rust",
            "manifest_sha256": manifest_sha,
            "model": spec.name,
            "output": out_dir.display().to_string(),
            "profile": profile_name,
        }))
    })();

    if export_result.is_err() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    export_result
}

/// Run the full pure-Rust `ncm pull` pipeline (download missing files -> build canonical
/// tokenizer -> export verified artifact).
pub fn run_native_pull(opts: &NativePullOptions) -> Result<()> {
    let spec = find_spec(&opts.model)?;
    let out_dir = opts
        .output_override
        .clone()
        .unwrap_or_else(|| opts.models_dir.join(spec.output_name));

    if out_dir.is_dir() && out_dir.join("manifest.json").is_file() {
        if opts.dry_run {
            println!("[ready]  artifact already exists at {}", out_dir.display());
            return Ok(());
        }
        Artifact::load_inference_streaming(&out_dir)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "status": "already_present_and_verified",
                "model": spec.name,
                "output": out_dir.display().to_string(),
            }))?
        );
        return Ok(());
    }

    let ckpt_dir = opts.upstream_dir.join(spec.fetch_dir);
    ensure_checkpoint_files(spec, &ckpt_dir, opts.dry_run)?;

    let tok_dir = if let Some(tok_prof) = spec.tokenizer_profile {
        Some(ensure_canonical_tokenizer(
            tok_prof,
            &ckpt_dir,
            &opts.upstream_dir,
            &opts.models_dir,
            opts.dry_run,
        )?)
    } else {
        None
    };

    if opts.dry_run {
        println!(
            "[export] {} -> {} (pure-Rust native exporter)",
            ckpt_dir.display(),
            out_dir.display()
        );
        return Ok(());
    }

    let report = export_native_from_cache(
        spec,
        &ckpt_dir,
        tok_dir.as_deref(),
        &out_dir,
        opts.reference_rope.as_deref(),
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
