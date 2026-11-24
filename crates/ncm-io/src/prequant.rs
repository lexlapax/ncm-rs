//! Verified pre-quantized (`Q8`/`Q4`) sidecar cache format, writer, and streaming validator.
//!
//! Sidecars live beside an unmodified parent artifact (`prequant-{format}-{precision}.safetensors`
//! and `prequant-{format}-{precision}.json`) and are cryptographically bound to the parent
//! `manifest.json` SHA-256. Any missing pair, symlink, manifest mismatch, or SHA-256 mismatch
//! fails closed.

use crate::{
    Error, MAX_MANIFEST, Manifest, Result, bounded_read, check_hash_format, invalid, sha256,
    wmt14_scratch, worker_count,
};
use safetensors::{Dtype, tensor::TensorView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

/// Encoding kind for one logical tensor stored inside a pre-quantized sidecar.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum PrequantKind {
    Bf16,
    F32,
    I64,
    GgmlQ8_0,
    GgmlQ4_0,
    MlxQ8G64,
    MlxQ4G64,
}

impl PrequantKind {
    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::GgmlQ8_0 | Self::GgmlQ4_0 | Self::MlxQ8G64 | Self::MlxQ4G64
        )
    }

    fn expected_key_count(self) -> usize {
        match self {
            Self::Bf16 | Self::F32 | Self::I64 | Self::GgmlQ8_0 | Self::GgmlQ4_0 => 1,
            Self::MlxQ8G64 | Self::MlxQ4G64 => 3,
        }
    }

    fn matches_format_precision(self, format: &str, precision: &str) -> bool {
        match self {
            Self::Bf16 | Self::F32 | Self::I64 => true,
            Self::GgmlQ8_0 => format == "ggml" && precision == "q8",
            Self::GgmlQ4_0 => format == "ggml" && precision == "q4",
            Self::MlxQ8G64 => format == "mlx" && precision == "q8",
            Self::MlxQ4G64 => format == "mlx" && precision == "q4",
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct PrequantTensorRecord {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: PrequantKind,
    pub keys: Vec<String>,
    pub sha256s: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrequantManifest {
    pub schema_version: u32,
    pub format: String,
    pub precision: String,
    pub manifest_sha256: String,
    pub file: String,
    pub file_bytes: u64,
    pub file_sha256: String,
    pub bundles: BTreeMap<String, Vec<PrequantTensorRecord>>,
}

/// Borrowed tensor payload yielded while streaming a verified pre-quantized sidecar.
pub enum PrequantTensorView<'a> {
    Bf16(&'a [u8]),
    F32(&'a [u8]),
    I64(&'a [u8]),
    GgmlQ8_0(&'a [u8]),
    GgmlQ4_0(&'a [u8]),
    MlxAffine {
        bits: u8,
        group_size: usize,
        weights_u32: &'a [u8],
        weights_shape: &'a [usize],
        scales_bf16: &'a [u8],
        scales_shape: &'a [usize],
        biases_bf16: &'a [u8],
        biases_shape: &'a [usize],
    },
}

/// Single logical tensor prepared by `ncm quantize` for writing into a sidecar.
pub struct PrequantWriteEntry {
    pub bundle: String,
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: PrequantKind,
    /// `(safetensors_key, dtype, stored_shape, raw_le_bytes)`
    pub parts: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
}

/// Summary returned after atomically writing a verified pre-quantized sidecar.
#[derive(Debug, Clone, Serialize)]
pub struct PrequantSummary {
    pub manifest_path: PathBuf,
    pub weights_path: PathBuf,
    pub format: String,
    pub precision: String,
    pub manifest_sha256: String,
    pub file_bytes: u64,
    pub file_sha256: String,
    pub quantized_tensors: usize,
    pub total_tensors: usize,
}

#[derive(Clone)]
pub(crate) struct KeyLocation {
    pub(crate) offset: u64,
    pub(crate) len: usize,
    pub(crate) shape: Vec<usize>,
}

pub(crate) struct VerifiedPrequant {
    #[cfg(unix)]
    file: fs::File,
    bundles: BTreeMap<String, Vec<PrequantTensorRecord>>,
    keys: BTreeMap<String, KeyLocation>,
}

impl VerifiedPrequant {
    pub(crate) fn covers_bundle(&self, bundle: &str) -> bool {
        self.bundles.contains_key(bundle)
    }

    pub(crate) fn remove_bundle(&mut self, bundle: &str) -> bool {
        self.bundles.remove(bundle).is_some()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.bundles.is_empty()
    }

    pub(crate) fn read_single_tensor(&self, bundle: &str, name: &str) -> Result<Vec<u8>> {
        let records = self
            .bundles
            .get(bundle)
            .ok_or_else(|| invalid(format!("prequant missing bundle {bundle}")))?;
        let rec = records
            .iter()
            .find(|r| r.name == name)
            .ok_or_else(|| invalid(format!("prequant missing tensor {bundle}/{name}")))?;
        if !matches!(rec.kind, PrequantKind::F32 | PrequantKind::I64) || rec.keys.len() != 1 {
            return Err(invalid(format!(
                "prequant tensor {bundle}/{name} is not stored verbatim as F32/I64"
            )));
        }
        let loc = self
            .keys
            .get(&rec.keys[0])
            .ok_or_else(|| invalid("prequant key index missing"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            let mut out = vec![0u8; loc.len];
            self.file.read_exact_at(&mut out, loc.offset)?;
            Ok(out)
        }
        #[cfg(not(unix))]
        {
            let _ = loc;
            Err(invalid("prequant streaming requires unix"))
        }
    }

    pub(crate) fn visit_bundle<E>(
        &self,
        bundle: &str,
        mut map_io: impl FnMut(Error) -> E,
        mut f: impl FnMut(&str, &[usize], PrequantTensorView<'_>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let records = self
            .bundles
            .get(bundle)
            .ok_or_else(|| map_io(invalid(format!("prequant missing bundle {bundle}"))))?;
        #[cfg(unix)]
        {
            let mut max_part = 1usize;
            let mut max_aux = 1usize;
            for rec in records {
                for (idx, key) in rec.keys.iter().enumerate() {
                    let loc = self
                        .keys
                        .get(key)
                        .ok_or_else(|| map_io(invalid("prequant key missing")))?;
                    if idx == 0 {
                        max_part = max_part.max(loc.len);
                    } else {
                        max_aux = max_aux.max(loc.len);
                    }
                }
            }
            let mut main_buf = wmt14_scratch::Scratch::new(max_part).map_err(&mut map_io)?;
            let mut aux1_buf = wmt14_scratch::Scratch::new(max_aux).map_err(&mut map_io)?;
            let mut aux2_buf = wmt14_scratch::Scratch::new(max_aux).map_err(&mut map_io)?;
            for rec in records {
                match rec.kind {
                    PrequantKind::Bf16
                    | PrequantKind::F32
                    | PrequantKind::I64
                    | PrequantKind::GgmlQ8_0
                    | PrequantKind::GgmlQ4_0 => {
                        let loc = &self.keys[&rec.keys[0]];
                        main_buf
                            .read_at(&self.file, loc.offset, loc.len)
                            .map_err(&mut map_io)?;
                        let view = match rec.kind {
                            PrequantKind::Bf16 => PrequantTensorView::Bf16(main_buf.bytes()),
                            PrequantKind::F32 => PrequantTensorView::F32(main_buf.bytes()),
                            PrequantKind::I64 => PrequantTensorView::I64(main_buf.bytes()),
                            PrequantKind::GgmlQ8_0 => {
                                PrequantTensorView::GgmlQ8_0(main_buf.bytes())
                            }
                            PrequantKind::GgmlQ4_0 => {
                                PrequantTensorView::GgmlQ4_0(main_buf.bytes())
                            }
                            _ => unreachable!(),
                        };
                        f(&rec.name, &rec.shape, view)?;
                    }
                    PrequantKind::MlxQ8G64 | PrequantKind::MlxQ4G64 => {
                        let lw = &self.keys[&rec.keys[0]];
                        let ls = &self.keys[&rec.keys[1]];
                        let lb = &self.keys[&rec.keys[2]];
                        main_buf
                            .read_at(&self.file, lw.offset, lw.len)
                            .map_err(&mut map_io)?;
                        aux1_buf
                            .read_at(&self.file, ls.offset, ls.len)
                            .map_err(&mut map_io)?;
                        aux2_buf
                            .read_at(&self.file, lb.offset, lb.len)
                            .map_err(&mut map_io)?;
                        let bits = if rec.kind == PrequantKind::MlxQ8G64 {
                            8
                        } else {
                            4
                        };
                        f(
                            &rec.name,
                            &rec.shape,
                            PrequantTensorView::MlxAffine {
                                bits,
                                group_size: 64,
                                weights_u32: main_buf.bytes(),
                                weights_shape: &lw.shape,
                                scales_bf16: aux1_buf.bytes(),
                                scales_shape: &ls.shape,
                                biases_bf16: aux2_buf.bytes(),
                                biases_shape: &lb.shape,
                            },
                        )?;
                    }
                }
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = (records, &mut f);
            Err(map_io(invalid("prequant streaming requires unix")))
        }
    }
}

/// Exact IEEE 754 round-to-nearest-even conversion from `f32` to `bf16` bit pattern.
#[inline]
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

/// Convert little-endian IEEE 754 `F32` bytes into little-endian `BF16` bytes.
pub fn f32_bytes_to_bf16_bytes(f32_le: &[u8]) -> Result<Vec<u8>> {
    if !f32_le.len().is_multiple_of(4) {
        return Err(invalid("invalid F32 byte length for BF16 conversion"));
    }
    let mut out = Vec::with_capacity(f32_le.len() / 2);
    for &chunk in f32_le.as_chunks::<4>().0 {
        let v = f32::from_le_bytes(chunk);
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    Ok(out)
}

/// Expected sidecar filenames `(manifest_json, weights_safetensors)`.
pub fn sidecar_filenames(format: &str, precision: &str) -> Result<(String, String)> {
    if !matches!(format, "ggml" | "mlx") || !matches!(precision, "q8" | "q4") {
        return Err(invalid(format!(
            "unsupported prequant sidecar format/precision: {format}/{precision}"
        )));
    }
    Ok((
        format!("prequant-{format}-{precision}.json"),
        format!("prequant-{format}-{precision}.safetensors"),
    ))
}

/// Check whether a sidecar is absent (`Ok(false)`), present as a pair (`Ok(true)`),
/// or incomplete/symlinked (`Err`).
pub(crate) fn probe_sidecar(root: &Path, format: &str, precision: &str) -> Result<bool> {
    let (json_name, weights_name) = sidecar_filenames(format, precision)?;
    let json_path = root.join(&json_name);
    let weights_path = root.join(&weights_name);
    let json_meta = match fs::symlink_metadata(&json_path) {
        Ok(m) => Some(m),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(Error::Io(e)),
    };
    let weights_meta = match fs::symlink_metadata(&weights_path) {
        Ok(m) => Some(m),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(Error::Io(e)),
    };
    match (json_meta, weights_meta) {
        (None, None) => Ok(false),
        (Some(jm), Some(wm)) => {
            if !jm.file_type().is_file() || !wm.file_type().is_file() {
                return Err(invalid(format!(
                    "prequant sidecar {json_name}/{weights_name} must be regular files"
                )));
            }
            Ok(true)
        }
        _ => Err(invalid(format!(
            "incomplete prequant sidecar: both {json_name} and {weights_name} must exist"
        ))),
    }
}

/// Validate a pre-quantized sidecar strictly against `manifest` and `manifest_sha256`.
pub(crate) fn validate_sidecar(
    root: &Path,
    manifest: &Manifest,
    manifest_sha256: &str,
    format: &str,
    precision: &str,
    max_bytes: u64,
) -> Result<VerifiedPrequant> {
    let (json_name, weights_name) = sidecar_filenames(format, precision)?;
    let pm: PrequantManifest =
        serde_json::from_slice(&bounded_read(&root.join(&json_name), MAX_MANIFEST)?)?;
    if pm.schema_version != 1
        || pm.format != format
        || pm.precision != precision
        || pm.file != weights_name
    {
        return Err(invalid("prequant sidecar schema/format/precision mismatch"));
    }
    check_hash_format(&pm.manifest_sha256)?;
    check_hash_format(&pm.file_sha256)?;
    if pm.manifest_sha256 != manifest_sha256 {
        return Err(invalid(format!(
            "prequant sidecar manifest_sha256 mismatch (stale or tampered {json_name})"
        )));
    }
    if pm.file_bytes == 0 || pm.file_bytes > max_bytes {
        return Err(invalid("prequant sidecar exceeds size budget"));
    }

    // Verify bundle coverage & exact logical tensor inventory against parent manifest.
    let expected_bundles: BTreeSet<&str> = ["codec.safetensors", "generator.safetensors"]
        .into_iter()
        .filter(|b| manifest.tensors.contains_key(*b))
        .collect();
    let actual_bundles: BTreeSet<&str> = pm.bundles.keys().map(String::as_str).collect();
    if actual_bundles != expected_bundles {
        return Err(invalid("prequant sidecar bundle inventory mismatch"));
    }

    let mut expected_keys: BTreeMap<String, (PrequantKind, usize, String)> = BTreeMap::new();
    for (bundle, records) in &pm.bundles {
        let parent_entries = &manifest.tensors[bundle];
        if records.len() != parent_entries.len() {
            return Err(invalid(format!(
                "prequant tensor count mismatch for {bundle}"
            )));
        }
        let parent_by_name: BTreeMap<&str, &crate::TensorRecord> = parent_entries
            .iter()
            .map(|e| (e.name.as_str(), e))
            .collect();
        let mut seen_names = BTreeSet::new();
        for rec in records {
            if !seen_names.insert(rec.name.as_str()) {
                return Err(invalid(format!(
                    "duplicate prequant tensor {bundle}/{}",
                    rec.name
                )));
            }
            let parent = parent_by_name.get(rec.name.as_str()).ok_or_else(|| {
                invalid(format!("unexpected prequant tensor {bundle}/{}", rec.name))
            })?;
            if rec.shape != parent.shape {
                return Err(invalid(format!(
                    "prequant shape mismatch for {bundle}/{}",
                    rec.name
                )));
            }
            if !rec.kind.matches_format_precision(format, precision) {
                return Err(invalid(format!(
                    "prequant kind {:?} invalid for {format}/{precision}",
                    rec.kind
                )));
            }
            if rec.keys.len() != rec.kind.expected_key_count()
                || rec.sha256s.len() != rec.keys.len()
            {
                return Err(invalid(format!(
                    "prequant key/hash count mismatch for {bundle}/{}",
                    rec.name
                )));
            }
            for (part_idx, (key, hash)) in rec.keys.iter().zip(rec.sha256s.iter()).enumerate() {
                check_hash_format(hash)?;
                if expected_keys
                    .insert(key.clone(), (rec.kind, part_idx, hash.clone()))
                    .is_some()
                {
                    return Err(invalid(format!("duplicate prequant safetensors key {key}")));
                }
            }
        }
    }

    let weights_path = root.join(&weights_name);
    let meta = fs::symlink_metadata(&weights_path)?;
    if !meta.file_type().is_file() || meta.len() != pm.file_bytes {
        return Err(invalid(format!(
            "prequant weights size/type mismatch: {weights_name}"
        )));
    }
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt;
        fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&weights_path)?
    };
    #[cfg(not(unix))]
    let file = fs::File::open(&weights_path)?;
    let opened_meta = file.metadata()?;
    if !opened_meta.is_file() || opened_meta.len() != pm.file_bytes {
        return Err(invalid(format!(
            "prequant weights size/type mismatch: {weights_name}"
        )));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let mut prefix = [0u8; 8];
        file.read_exact_at(&mut prefix, 0)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > 100_000_000 || header_len.checked_add(8).is_none_or(|n| n > pm.file_bytes) {
            return Err(invalid("invalid prequant header length"));
        }
        let header_size = usize::try_from(header_len).map_err(|_| invalid("header overflow"))?;
        let mut header = vec![0u8; header_size];
        file.read_exact_at(&mut header, 8)?;
        let data_start = 8 + header_len;
        let metadata: safetensors::tensor::Metadata = serde_json::from_slice(&header)?;
        if (metadata.data_len() as u64).checked_add(data_start) != Some(pm.file_bytes) {
            return Err(invalid("prequant extent mismatch"));
        }
        if metadata.tensors().len() != expected_keys.len() {
            return Err(invalid("prequant safetensors tensor inventory mismatch"));
        }

        let mut keys_index: BTreeMap<String, KeyLocation> = BTreeMap::new();
        let mut ordered_checks: Vec<(String, u64, usize, String)> =
            Vec::with_capacity(expected_keys.len());
        for key in metadata.offset_keys() {
            let info = metadata
                .info(&key)
                .ok_or_else(|| invalid("missing prequant tensor metadata"))?;
            let &(kind, part_idx, ref expected_hash) = expected_keys
                .get(&key)
                .ok_or_else(|| invalid(format!("unexpected prequant tensor key {key}")))?;
            let expected_dtype = match (kind, part_idx) {
                (PrequantKind::Bf16, 0) => Dtype::BF16,
                (PrequantKind::F32, 0) => Dtype::F32,
                (PrequantKind::I64, 0) => Dtype::I64,
                (PrequantKind::GgmlQ8_0 | PrequantKind::GgmlQ4_0, 0) => Dtype::U8,
                (PrequantKind::MlxQ8G64 | PrequantKind::MlxQ4G64, 0) => Dtype::U32,
                (PrequantKind::MlxQ8G64 | PrequantKind::MlxQ4G64, 1 | 2) => Dtype::BF16,
                _ => return Err(invalid("invalid prequant part index")),
            };
            if info.dtype != expected_dtype {
                return Err(invalid(format!(
                    "prequant dtype mismatch for {key}: got {:?}, expected {:?}",
                    info.dtype, expected_dtype
                )));
            }
            let byte_len = info
                .data_offsets
                .1
                .checked_sub(info.data_offsets.0)
                .ok_or_else(|| invalid("prequant tensor offset order invalid"))?;
            let offset = data_start
                .checked_add(info.data_offsets.0 as u64)
                .ok_or_else(|| invalid("prequant offset overflow"))?;
            keys_index.insert(
                key.clone(),
                KeyLocation {
                    offset,
                    len: byte_len,
                    shape: info.shape.clone(),
                },
            );
            ordered_checks.push((key, offset, byte_len, expected_hash.clone()));
        }

        // Parallel whole-file SHA-256 + per-key SHA-256 verification.
        const STREAM_CHUNK: usize = 1024 * 1024;
        if pm.file_bytes < 4 * 1024 * 1024 {
            let mut file_hash = Sha256::new();
            file_hash.update(prefix);
            file_hash.update(&header);
            let mut scratch = wmt14_scratch::Scratch::new(STREAM_CHUNK)?;
            for (key, offset, len, expected_hash) in &ordered_checks {
                let mut key_hash = Sha256::new();
                let mut rem = *len;
                let mut cur = *offset;
                while rem > 0 {
                    let n = rem.min(STREAM_CHUNK);
                    scratch.read_at(&file, cur, n)?;
                    let chunk = scratch.bytes();
                    key_hash.update(chunk);
                    file_hash.update(chunk);
                    rem -= n;
                    cur += n as u64;
                }
                if format!("{:x}", key_hash.finalize()) != *expected_hash {
                    return Err(invalid(format!("prequant tensor SHA-256 mismatch: {key}")));
                }
            }
            let mut extra = [0u8; 1];
            if file.read_at(&mut extra, pm.file_bytes)? != 0
                || format!("{:x}", file_hash.finalize()) != pm.file_sha256
            {
                return Err(invalid(format!(
                    "prequant file SHA-256 mismatch: {weights_name}"
                )));
            }
        } else {
            let file_ref = &file;
            let checks_ref = &ordered_checks;
            let total_bytes = pm.file_bytes;
            let expected_file_sha = pm.file_sha256.as_str();
            let header_ref = &header;
            let tensor_workers = worker_count()
                .saturating_sub(1)
                .clamp(1, 6)
                .min(checks_ref.len().max(1));
            let next = std::sync::atomic::AtomicUsize::new(0);
            let next_ref = &next;
            std::thread::scope(|scope| -> Result<()> {
                const WHOLE_CHUNK: usize = 2 * 1024 * 1024;
                let whole_handle = scope.spawn(move || -> Result<()> {
                    let mut scratch = wmt14_scratch::Scratch::new(WHOLE_CHUNK)?;
                    let mut file_hash = Sha256::new();
                    file_hash.update(prefix);
                    file_hash.update(header_ref);
                    let mut pos = data_start;
                    while pos < total_bytes {
                        let n = usize::try_from((total_bytes - pos).min(WHOLE_CHUNK as u64))
                            .map_err(|_| invalid("chunk size overflow"))?;
                        scratch.read_at(file_ref, pos, n)?;
                        file_hash.update(scratch.bytes());
                        pos += n as u64;
                    }
                    let mut extra = [0u8; 1];
                    if file_ref.read_at(&mut extra, total_bytes)? != 0
                        || format!("{:x}", file_hash.finalize()) != expected_file_sha
                    {
                        return Err(invalid(format!(
                            "prequant file SHA-256 mismatch: {weights_name}"
                        )));
                    }
                    Ok(())
                });
                let worker_handles: Vec<_> = (0..tensor_workers)
                    .map(|_| {
                        scope.spawn(move || -> Result<()> {
                            let mut scratch = wmt14_scratch::Scratch::new(STREAM_CHUNK)?;
                            loop {
                                let i = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if i >= checks_ref.len() {
                                    return Ok(());
                                }
                                let (key, offset, len, expected_hash) = &checks_ref[i];
                                let mut key_hash = Sha256::new();
                                let mut rem = *len;
                                let mut cur = *offset;
                                while rem > 0 {
                                    let n = rem.min(STREAM_CHUNK);
                                    scratch.read_at(file_ref, cur, n)?;
                                    key_hash.update(scratch.bytes());
                                    rem -= n;
                                    cur += n as u64;
                                }
                                if format!("{:x}", key_hash.finalize()) != *expected_hash {
                                    return Err(invalid(format!(
                                        "prequant tensor SHA-256 mismatch: {key}"
                                    )));
                                }
                            }
                        })
                    })
                    .collect();
                whole_handle
                    .join()
                    .map_err(|_| invalid("prequant file hash thread panicked"))??;
                for h in worker_handles {
                    h.join()
                        .map_err(|_| invalid("prequant tensor hash thread panicked"))??;
                }
                Ok(())
            })?;
        }

        Ok(VerifiedPrequant {
            file,
            bundles: pm.bundles,
            keys: keys_index,
        })
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        Err(invalid("prequant streaming requires unix"))
    }
}

/// Atomically write a verified pre-quantized sidecar bound to `root/manifest.json`.
pub fn write_prequant_sidecar(
    root: &Path,
    format: &str,
    precision: &str,
    entries: Vec<PrequantWriteEntry>,
    overwrite: bool,
) -> Result<PrequantSummary> {
    let (json_name, weights_name) = sidecar_filenames(format, precision)?;
    let manifest_bytes = bounded_read(&root.join("manifest.json"), MAX_MANIFEST)?;
    let manifest_sha256 = sha256(&manifest_bytes);

    let json_path = root.join(&json_name);
    let weights_path = root.join(&weights_name);
    for p in [&json_path, &weights_path] {
        match fs::symlink_metadata(p) {
            Ok(meta) => {
                if !meta.file_type().is_file() {
                    return Err(invalid(format!(
                        "refusing to touch non-regular target: {}",
                        p.display()
                    )));
                }
                if !overwrite {
                    return Err(invalid(format!(
                        "prequant sidecar already exists (pass --force to overwrite): {}",
                        p.display()
                    )));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::Io(e)),
        }
    }

    let mut bundles: BTreeMap<String, Vec<PrequantTensorRecord>> = BTreeMap::new();
    let mut quantized_tensors = 0usize;
    let total_tensors = entries.len();
    for entry in &entries {
        if entry.kind.is_quantized() {
            quantized_tensors += 1;
        }
        let mut keys = Vec::with_capacity(entry.parts.len());
        let mut sha256s = Vec::with_capacity(entry.parts.len());
        for (key, _, _, bytes) in &entry.parts {
            keys.push(key.clone());
            sha256s.push(sha256(bytes));
        }
        bundles
            .entry(entry.bundle.clone())
            .or_default()
            .push(PrequantTensorRecord {
                name: entry.name.clone(),
                shape: entry.shape.clone(),
                kind: entry.kind,
                keys,
                sha256s,
            });
    }

    let views: Vec<(&str, TensorView<'_>)> = entries
        .iter()
        .flat_map(|e| e.parts.iter())
        .map(|(key, dtype, shape, bytes)| {
            let tv = TensorView::new(*dtype, shape.clone(), bytes)?;
            Ok((key.as_str(), tv))
        })
        .collect::<std::result::Result<Vec<_>, safetensors::SafeTensorError>>()?;

    let tmp_weights = root.join(format!("{weights_name}.tmp"));
    let tmp_json = root.join(format!("{json_name}.tmp"));
    safetensors::tensor::serialize_to_file(views, None, &tmp_weights)?;

    let mut file = fs::File::open(&tmp_weights)?;
    let file_bytes = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 2 * 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    drop(file);
    let file_sha256 = format!("{:x}", hasher.finalize());

    let pm = PrequantManifest {
        schema_version: 1,
        format: format.to_owned(),
        precision: precision.to_owned(),
        manifest_sha256: manifest_sha256.clone(),
        file: weights_name,
        file_bytes,
        file_sha256: file_sha256.clone(),
        bundles,
    };
    let json_bytes = serde_json::to_vec_pretty(&pm)?;
    fs::write(&tmp_json, json_bytes)?;

    fs::rename(&tmp_weights, &weights_path)?;
    fs::rename(&tmp_json, &json_path)?;

    Ok(PrequantSummary {
        manifest_path: json_path,
        weights_path,
        format: format.to_owned(),
        precision: precision.to_owned(),
        manifest_sha256,
        file_bytes,
        file_sha256,
        quantized_tensors,
        total_tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_bf16_round_to_nearest_even_matches_ieee() {
        assert_eq!(f32_to_bf16_bits(0.0), 0x0000);
        assert_eq!(f32_to_bf16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(f32_to_bf16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16_bits(f32::NEG_INFINITY), 0xff80);
        // Exact halfway cases: tie to even.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
        // Round up above halfway.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81);
    }
}
