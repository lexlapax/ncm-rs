//! Bounded copied loading for the Phase 1 Sudoku inspection bundle.
//!
//! Verifies integrity relative to a manifest, not authenticity of that manifest.
//! No pickle execution, artifact file mapping, model construction, or implicit downloads.

mod wmt14_scratch;

use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

pub const UPSTREAM: &str = "c70274273f489abc73832b2b98584a8483b6b203";
pub const CHECKPOINT_REVISION: &str = "d342ce8aaea58830eef3b8ad473664673e41f692";
pub const UPSTREAM_2026_11_23: &str = "094f25ca5fd8fe2bc7f0ff77bf3f860c3a34e7e8";
pub const CHECKPOINT_REVISION_2026_11_23: &str = "e34c3fab451ddbe7cf474cc97c44ab23321f2219";
const MAX_MANIFEST: u64 = 4 * 1024 * 1024;
const MAX_FILE: u64 = 256 * 1024 * 1024;
const MAX_TOTAL: u64 = 512 * 1024 * 1024;
// TinyStories' pinned generator contains 356,071,472 tensor bytes. These
// limits are profile-specific and do not relax the existing Sudoku bounds.
const TINYSTORIES_MAX_FILE: u64 = 384 * 1024 * 1024;
const TINYSTORIES_MAX_TOTAL: u64 = 640 * 1024 * 1024;
// Pinned XSum files total about 628 MiB; keep its limits independent of other profiles.
const XSUM_MAX_FILE: u64 = 384 * 1024 * 1024;
const XSUM_MAX_TOTAL: u64 = 640 * 1024 * 1024;
// WMT14's larger vocabulary and generator have separate bounded budgets.
const WMT14_MAX_FILE: u64 = 640 * 1024 * 1024;
const WMT14_MAX_TOTAL: u64 = 1280 * 1024 * 1024;
// Each LM1B size has its own bounds, set 5-8% above the measured exports:
// small's largest file is the 425 MiB codec; medium's and large's are the
// 1,209 MiB and 2,766 MiB generators, with totals of 1,781 and 3,338 MiB.
// Sizes never inherit another size's limits.
const LM1B_SMALL_MAX_FILE: u64 = 448 * 1024 * 1024;
const LM1B_SMALL_MAX_TOTAL: u64 = 1024 * 1024 * 1024;
const LM1B_MEDIUM_MAX_FILE: u64 = 1280 * 1024 * 1024;
const LM1B_MEDIUM_MAX_TOTAL: u64 = 1920 * 1024 * 1024;
const LM1B_LARGE_MAX_FILE: u64 = 2944 * 1024 * 1024;
const LM1B_LARGE_MAX_TOTAL: u64 = 3584 * 1024 * 1024;
// Generic local user checkpoints (`local-ncm-v1`) use a separate bounded budget
// that never weakens any pinned upstream profile's limits.
const LOCAL_MAX_FILE: u64 = 4 * 1024 * 1024 * 1024;
const LOCAL_MAX_TOTAL: u64 = 8 * 1024 * 1024 * 1024;
const LOCAL_REQUIRED_FILES: [&str; 4] = [
    "codec.safetensors",
    "generator.safetensors",
    "config.json",
    "provenance.json",
];
const LOCAL_ALLOWED_FILES: [&str; 9] = [
    "codec.safetensors",
    "generator.safetensors",
    "training.safetensors",
    "inference.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer-manifest.json",
    "tokenizer.LICENSE",
    "provenance.json",
];
const FILES: [&str; 6] = [
    "codec.safetensors",
    "generator.safetensors",
    "training.safetensors",
    "config.json",
    "tokenizer.json",
    "provenance.json",
];
const BUNDLES: [&str; 3] = [
    "codec.safetensors",
    "generator.safetensors",
    "training.safetensors",
];
const TINYSTORIES_FILES: [&str; 9] = [
    "codec.safetensors",
    "generator.safetensors",
    "training.safetensors",
    "inference.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer-manifest.json",
    "tokenizer.LICENSE",
    "provenance.json",
];
const XSUM_FILES: [&str; 9] = [
    "codec.safetensors",
    "generator.safetensors",
    "training.safetensors",
    "inference.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer-manifest.json",
    "tokenizer.LICENSE",
    "provenance.json",
];
const WMT14_FILES: [&str; 8] = [
    "codec.safetensors",
    "generator.safetensors",
    "training.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer-manifest.json",
    "tokenizer.LICENSE",
    "provenance.json",
];
const TINYSTORIES_BUNDLES: [&str; 4] = [
    "codec.safetensors",
    "generator.safetensors",
    "training.safetensors",
    "inference.safetensors",
];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Tensor(#[from] safetensors::SafeTensorError),
    #[error("invalid artifact: {0}")]
    Invalid(String),
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub profile: String,
    pub upstream_commit: String,
    pub checkpoint_repository: String,
    pub checkpoint_revision: String,
    #[serde(deserialize_with = "unique_map")]
    pub files: BTreeMap<String, FileRecord>,
    #[serde(deserialize_with = "unique_map")]
    pub tensors: BTreeMap<String, Vec<TensorRecord>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRecord {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
    pub bytes: usize,
    pub sha256: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    I64,
}
impl TensorDtype {
    fn native(self) -> Dtype {
        match self {
            Self::F32 => Dtype::F32,
            Self::I64 => Dtype::I64,
        }
    }
    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

fn unique_map<'de, D, V>(deserializer: D) -> std::result::Result<BTreeMap<String, V>, D::Error>
where
    D: serde::Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct Visitor<V>(std::marker::PhantomData<V>);
    impl<'de, V: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<V> {
        type Value = BTreeMap<String, V>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a map without duplicate keys")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut result = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, V>()? {
                if result.insert(key.clone(), value).is_some() {
                    return Err(serde::de::Error::custom(format!("duplicate key: {key}")));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Visitor(std::marker::PhantomData))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
fn check_hash_format(hash: &str) -> Result<()> {
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(invalid("invalid SHA-256 encoding"));
    }
    Ok(())
}
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    bounded_read_into(path, limit, &mut data)?;
    Ok(data)
}

fn bounded_read_into(path: &Path, limit: u64, data: &mut Vec<u8>) -> Result<()> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.file_type().is_file() || meta.len() > limit {
        return Err(invalid(format!(
            "not a bounded regular file: {}",
            path.display()
        )));
    }
    // Read once; validation and tensor use share these owned bytes. The artifact directory
    // must not be concurrently mutated by an untrusted process (no O_NOFOLLOW portability claim).
    // One spare byte permits bounded EOF/growth detection without geometric
    // reallocations. Freed intermediate buffers can remain resident on Darwin.
    let capacity =
        usize::try_from(meta.len() + 1).map_err(|_| invalid("file capacity overflow"))?;
    data.clear();
    data.try_reserve_exact(capacity)
        .map_err(|_| invalid("file allocation failed"))?;
    fs::File::open(path)?.take(limit + 1).read_to_end(data)?;
    if data.len() as u64 > limit {
        return Err(invalid("file grew beyond limit"));
    }
    Ok(())
}

// Training data is validated in one bounded pass and never enters the
// inference view. Header parsing uses the safetensors crate's own Metadata
// validation; the whole-file and individual tensor digests are both checked.
fn validate_training_stream(
    path: &Path,
    record: &FileRecord,
    entries: &[TensorRecord],
) -> Result<()> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.file_type().is_file() || meta.len() != record.bytes {
        return Err(invalid("invalid training bundle file"));
    }
    let mut file = fs::File::open(path)?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > 100_000_000 || header_len.checked_add(8).is_none_or(|n| n > record.bytes) {
        return Err(invalid("invalid training bundle header length"));
    }
    let mut header =
        vec![0u8; usize::try_from(header_len).map_err(|_| invalid("header overflow"))?];
    file.read_exact(&mut header)?;
    let metadata: safetensors::tensor::Metadata = serde_json::from_slice(&header)?;
    if (metadata.data_len() as u64).checked_add(header_len + 8) != Some(record.bytes) {
        return Err(invalid("training bundle extent mismatch"));
    }
    let by_name: BTreeMap<_, _> = entries
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect();
    if by_name.len() != entries.len() || metadata.tensors().len() != entries.len() {
        return Err(invalid("training bundle tensor inventory differs"));
    }
    let mut file_hash = Sha256::new();
    file_hash.update(prefix);
    file_hash.update(&header);
    let mut buffer = vec![0u8; 1024 * 1024];
    for name in metadata.offset_keys() {
        let info = metadata
            .info(&name)
            .ok_or_else(|| invalid("missing training metadata"))?;
        let entry = by_name
            .get(name.as_str())
            .ok_or_else(|| invalid("unexpected training tensor"))?;
        check_hash_format(&entry.sha256)?;
        let size = entry
            .shape
            .iter()
            .try_fold(entry.dtype.size(), |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| invalid("training tensor shape overflow"))?;
        if entry.shape.len() > 8
            || info.shape != entry.shape
            || info.dtype != entry.dtype.native()
            || size != entry.bytes
            || info.data_offsets.1 - info.data_offsets.0 != entry.bytes
        {
            return Err(invalid("training tensor metadata differs"));
        }
        let mut remaining = entry.bytes;
        let mut tensor_hash = Sha256::new();
        while remaining > 0 {
            let n = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..n])?;
            tensor_hash.update(&buffer[..n]);
            file_hash.update(&buffer[..n]);
            remaining -= n;
        }
        if format!("{:x}", tensor_hash.finalize()) != entry.sha256 {
            return Err(invalid("training tensor hash differs"));
        }
    }
    let mut extra = [0u8; 1];
    if file.read(&mut extra)? != 0 || format!("{:x}", file_hash.finalize()) != record.sha256 {
        return Err(invalid("training bundle size/hash differs"));
    }
    Ok(())
}

pub struct Artifact {
    manifest: Manifest,
    payloads: BTreeMap<String, Vec<u8>>,
    reusable_wmt14: Option<ReusableWmt14>,
    streaming: BTreeMap<String, StreamedBundle>,
}

struct ReusableWmt14 {
    root: PathBuf,
    name: &'static str,
    data: wmt14_scratch::Scratch,
}

struct StreamedBundle {
    file: fs::File,
    /// Ordered by file offset for sequential page-cache reads: (manifest_idx, file_offset, bytes).
    ordered: Vec<(usize, u64, usize)>,
    /// True when every F32 tensor except `attn_mask_for_masking` was checked finite during load.
    finite_checked_f32: bool,
}

fn has_nonfinite_f32_le(data: &[u8]) -> bool {
    debug_assert_eq!(data.len() % 4, 0);
    let (groups, tail) = data.as_chunks::<64>();
    for group in groups {
        let mut bad = false;
        for bytes in group.as_chunks::<4>().0 {
            bad |= u32::from_le_bytes(*bytes) & 0x7f80_0000 == 0x7f80_0000;
        }
        if bad {
            return true;
        }
    }
    tail.as_chunks::<4>()
        .0
        .iter()
        .any(|bytes| u32::from_le_bytes(*bytes) & 0x7f80_0000 == 0x7f80_0000)
}

fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .clamp(1, 8)
}

fn verify_payload(name: &str, bytes: &[u8], record: &FileRecord) -> Result<()> {
    if bytes.len() as u64 != record.bytes || sha256(bytes) != record.sha256 {
        return Err(invalid(format!("file size/hash mismatch: {name}")));
    }
    if name.ends_with(".json") {
        let _: serde_json::Value = serde_json::from_slice(bytes)?;
    }
    Ok(())
}

fn verify_bundle(bundle: &str, bytes: &[u8], entries: &[TensorRecord]) -> Result<()> {
    let tensors = SafeTensors::deserialize(bytes)?;
    let names = entries
        .iter()
        .map(|entry| entry.name.as_str())
        .collect::<BTreeSet<_>>();
    if names.len() != entries.len() || names != tensors.names().into_iter().collect() {
        return Err(invalid(format!("missing/extra/duplicate tensor: {bundle}")));
    }
    let mut slices = Vec::with_capacity(entries.len());
    for entry in entries {
        check_hash_format(&entry.sha256)?;
        if entry.shape.len() > 8 {
            return Err(invalid("tensor rank exceeds limit"));
        }
        let expected_bytes = entry
            .shape
            .iter()
            .try_fold(entry.dtype.size(), |size, dim| size.checked_mul(*dim))
            .ok_or_else(|| invalid("tensor shape overflow"))?;
        let tensor = tensors.tensor(&entry.name)?;
        if expected_bytes != entry.bytes
            || tensor.shape() != entry.shape
            || tensor.dtype() != entry.dtype.native()
            || tensor.data().len() != entry.bytes
        {
            return Err(invalid(format!(
                "tensor metadata/hash mismatch: {bundle}/{}",
                entry.name
            )));
        }
        slices.push((entry.name.as_str(), entry.sha256.as_str(), tensor.data()));
    }
    let threads = worker_count().min(slices.len().max(1));
    if threads <= 1 || bytes.len() < 4 * 1024 * 1024 {
        for (name, expected, data) in slices {
            if sha256(data) != expected {
                return Err(invalid(format!(
                    "tensor metadata/hash mismatch: {bundle}/{name}"
                )));
            }
        }
        return Ok(());
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slices = &slices;
    let next = &next;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(move || -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= slices.len() {
                            return Ok(());
                        }
                        let (name, expected, data) = slices[i];
                        if sha256(data) != expected {
                            return Err(invalid(format!(
                                "tensor metadata/hash mismatch: {bundle}/{name}"
                            )));
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join()
                .map_err(|_| invalid("hash worker thread panicked"))??;
        }
        Ok(())
    })
}

fn validate_streaming_bundle(
    path: &Path,
    bundle: &str,
    record: &FileRecord,
    entries: &[TensorRecord],
    check_finite_f32: bool,
) -> Result<StreamedBundle> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.file_type().is_file() || meta.len() != record.bytes {
        return Err(invalid(format!("invalid bundle file: {bundle}")));
    }
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt;
        fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?
    };
    #[cfg(not(unix))]
    let file = fs::File::open(path)?;
    let opened_meta = file.metadata()?;
    if !opened_meta.is_file() || opened_meta.len() != record.bytes {
        return Err(invalid(format!("invalid bundle file: {bundle}")));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let mut prefix = [0u8; 8];
        file.read_exact_at(&mut prefix, 0)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > 100_000_000 || header_len.checked_add(8).is_none_or(|n| n > record.bytes) {
            return Err(invalid(format!("invalid bundle header length: {bundle}")));
        }
        let header_size = usize::try_from(header_len).map_err(|_| invalid("header overflow"))?;
        let mut header = vec![0u8; header_size];
        file.read_exact_at(&mut header, 8)?;
        let data_start = 8 + header_len;
        let metadata: safetensors::tensor::Metadata = serde_json::from_slice(&header)?;
        if (metadata.data_len() as u64).checked_add(data_start) != Some(record.bytes) {
            return Err(invalid(format!("bundle extent mismatch: {bundle}")));
        }
        let by_name: BTreeMap<_, _> = entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| (entry.name.as_str(), (idx, entry)))
            .collect();
        if by_name.len() != entries.len() || metadata.tensors().len() != entries.len() {
            return Err(invalid(format!("missing/extra/duplicate tensor: {bundle}")));
        }
        let mut ordered = Vec::with_capacity(entries.len());
        for name in metadata.offset_keys() {
            let info = metadata
                .info(&name)
                .ok_or_else(|| invalid("missing tensor metadata"))?;
            let &(idx, entry) = by_name
                .get(name.as_str())
                .ok_or_else(|| invalid(format!("unexpected tensor: {bundle}/{name}")))?;
            check_hash_format(&entry.sha256)?;
            if entry.shape.len() > 8 {
                return Err(invalid("tensor rank exceeds limit"));
            }
            let size = entry
                .shape
                .iter()
                .try_fold(entry.dtype.size(), |n, &dim| n.checked_mul(dim))
                .ok_or_else(|| invalid("tensor shape overflow"))?;
            if info.shape != entry.shape
                || info.dtype != entry.dtype.native()
                || size != entry.bytes
                || info.data_offsets.1 - info.data_offsets.0 != entry.bytes
            {
                return Err(invalid(format!(
                    "tensor metadata/hash mismatch: {bundle}/{name}"
                )));
            }
            let offset = data_start
                .checked_add(info.data_offsets.0 as u64)
                .ok_or_else(|| invalid("tensor offset overflow"))?;
            ordered.push((idx, offset, entry.bytes));
        }

        // For small bundles (< 4 MiB), verify whole-file and per-tensor hashes in one
        // sequential pass without spawning threads. For larger bundles, hash the whole
        // file on one thread while verifying per-tensor hashes and finite FP32 values
        // in parallel using bounded 1 MiB scratch buffers (unmapped on Darwin drop).
        const STREAM_CHUNK: usize = 1024 * 1024;
        if record.bytes < 4 * 1024 * 1024 {
            let mut file_hash = Sha256::new();
            file_hash.update(prefix);
            file_hash.update(&header);
            let mut scratch = wmt14_scratch::Scratch::new(STREAM_CHUNK)?;
            for &(idx, offset, len) in &ordered {
                let entry = &entries[idx];
                let mut tensor_hash = Sha256::new();
                let mut rem = len;
                let mut cur = offset;
                let check_finite = check_finite_f32
                    && matches!(entry.dtype, TensorDtype::F32)
                    && entry.name != "attn_mask_for_masking";
                while rem > 0 {
                    let n = rem.min(STREAM_CHUNK);
                    scratch.read_at(&file, cur, n)?;
                    let chunk = scratch.bytes();
                    if check_finite && has_nonfinite_f32_le(chunk) {
                        return Err(invalid(format!(
                            "nonfinite FP32 value in {bundle}/{}",
                            entry.name
                        )));
                    }
                    tensor_hash.update(chunk);
                    file_hash.update(chunk);
                    rem -= n;
                    cur += n as u64;
                }
                if format!("{:x}", tensor_hash.finalize()) != entry.sha256 {
                    return Err(invalid(format!(
                        "tensor metadata/hash mismatch: {bundle}/{}",
                        entry.name
                    )));
                }
            }
            let mut extra = [0u8; 1];
            if file.read_at(&mut extra, record.bytes)? != 0
                || format!("{:x}", file_hash.finalize()) != record.sha256
            {
                return Err(invalid(format!("file size/hash mismatch: {bundle}")));
            }
            return Ok(StreamedBundle {
                file,
                ordered,
                finite_checked_f32: check_finite_f32,
            });
        }

        let file_ref = &file;
        let ordered_ref = &ordered;
        let total_bytes = record.bytes;
        let expected_file_sha = record.sha256.as_str();
        let header_ref = &header;
        let tensor_workers = worker_count()
            .saturating_sub(1)
            .clamp(1, 6)
            .min(ordered.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let next_ref = &next;
        std::thread::scope(|scope| -> Result<()> {
            const WHOLE_CHUNK: usize = 2 * 1024 * 1024;
            let whole_handle = scope.spawn(move || -> Result<()> {
                let (full_tx, full_rx) = std::sync::mpsc::sync_channel::<wmt14_scratch::Scratch>(2);
                let (empty_tx, empty_rx) =
                    std::sync::mpsc::sync_channel::<wmt14_scratch::Scratch>(2);
                for _ in 0..2 {
                    let _ = empty_tx.send(wmt14_scratch::Scratch::new(WHOLE_CHUNK)?);
                }
                std::thread::scope(|inner| -> Result<()> {
                    let reader = inner.spawn(move || -> Result<()> {
                        let mut pos = data_start;
                        while pos < total_bytes {
                            let n = usize::try_from((total_bytes - pos).min(WHOLE_CHUNK as u64))
                                .map_err(|_| invalid("chunk size overflow"))?;
                            let Ok(mut scratch) = empty_rx.recv() else {
                                return Ok(());
                            };
                            scratch.read_at(file_ref, pos, n)?;
                            if full_tx.send(scratch).is_err() {
                                return Ok(());
                            }
                            pos += n as u64;
                        }
                        let mut extra = [0u8; 1];
                        if file_ref.read_at(&mut extra, total_bytes)? != 0 {
                            return Err(invalid(format!("file size/hash mismatch: {bundle}")));
                        }
                        Ok(())
                    });
                    let mut file_hash = Sha256::new();
                    file_hash.update(prefix);
                    file_hash.update(header_ref);
                    while let Ok(scratch) = full_rx.recv() {
                        file_hash.update(scratch.bytes());
                        let _ = empty_tx.send(scratch);
                    }
                    drop(empty_tx);
                    reader
                        .join()
                        .map_err(|_| invalid("file reader thread panicked"))??;
                    if format!("{:x}", file_hash.finalize()) != expected_file_sha {
                        return Err(invalid(format!("file size/hash mismatch: {bundle}")));
                    }
                    Ok(())
                })
            });
            let worker_handles: Vec<_> = (0..tensor_workers)
                .map(|_| {
                    scope.spawn(move || -> Result<()> {
                        let mut scratch = wmt14_scratch::Scratch::new(STREAM_CHUNK)?;
                        loop {
                            let i = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= ordered_ref.len() {
                                return Ok(());
                            }
                            let (idx, offset, len) = ordered_ref[i];
                            let entry = &entries[idx];
                            let check_finite = check_finite_f32
                                && matches!(entry.dtype, TensorDtype::F32)
                                && entry.name != "attn_mask_for_masking";
                            let mut tensor_hash = Sha256::new();
                            let mut rem = len;
                            let mut cur = offset;
                            while rem > 0 {
                                let n = rem.min(STREAM_CHUNK);
                                scratch.read_at(file_ref, cur, n)?;
                                let chunk = scratch.bytes();
                                if check_finite && has_nonfinite_f32_le(chunk) {
                                    return Err(invalid(format!(
                                        "nonfinite FP32 value in {bundle}/{}",
                                        entry.name
                                    )));
                                }
                                tensor_hash.update(chunk);
                                rem -= n;
                                cur += n as u64;
                            }
                            if format!("{:x}", tensor_hash.finalize()) != entry.sha256 {
                                return Err(invalid(format!(
                                    "tensor metadata/hash mismatch: {bundle}/{}",
                                    entry.name
                                )));
                            }
                        }
                    })
                })
                .collect();
            whole_handle
                .join()
                .map_err(|_| invalid("file hash thread panicked"))??;
            for h in worker_handles {
                h.join()
                    .map_err(|_| invalid("tensor hash thread panicked"))??;
            }
            Ok(())
        })?;
        Ok(StreamedBundle {
            file,
            ordered,
            finite_checked_f32: check_finite_f32,
        })
    }
    #[cfg(not(unix))]
    {
        let _ = (file, check_finite_f32);
        Err(invalid("streaming bundle loader requires unix"))
    }
}

impl Artifact {
    /// Validate the complete artifact, streaming the training-only bundle.
    /// Inference payloads remain owned and immutable; `file("training.safetensors")`
    /// explicitly fails on this reduced view. `load` retains the full artifact.
    pub fn load_inference(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_impl(root.as_ref(), true, false, false)
    }

    /// Validate the complete artifact using bounded streaming buffers (~4 MiB per worker)
    /// and parallel SHA-256 verification. Weight bundles (`codec.safetensors`,
    /// `generator.safetensors`) stay on open validated file descriptors and are streamed
    /// one tensor at a time via `visit_bundle_tensors`, cutting both load time and peak RSS.
    pub fn load_inference_streaming(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_impl(root.as_ref(), true, false, true)
    }

    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_impl(root.as_ref(), false, false, false)
    }

    /// Validate both WMT14 bundles with one host buffer. Only the generator
    /// remains loaded. Swap bundles with `with_wmt14_bundle`, which checks the
    /// reread against the already validated manifest before exposing it.
    pub fn load_wmt14_inference_reusable(root: impl AsRef<Path>) -> Result<Self> {
        let root = std::path::absolute(root)?;
        Self::load_impl(&root, true, true, false)
    }

    fn load_impl(
        root: &Path,
        inference: bool,
        reusable: bool,
        streaming_mode: bool,
    ) -> Result<Self> {
        // Strip trailing separators/dot components before the no-symlink leaf check.
        let root = root.components().collect::<std::path::PathBuf>();
        if !fs::symlink_metadata(&root)?.file_type().is_dir() {
            return Err(invalid("artifact root must be a real directory"));
        }
        let manifest: Manifest =
            serde_json::from_slice(&bounded_read(&root.join("manifest.json"), MAX_MANIFEST)?)?;
        let is_local = manifest.profile == "local-ncm-v1";
        let valid_metadata_str = |s: &str| {
            !s.is_empty() && s.len() <= 256 && s.bytes().all(|b| (0x20..=0x7e).contains(&b))
        };
        if manifest.schema_version != 1 {
            return Err(invalid("unsupported schema, profile or source pins"));
        }
        if is_local {
            if !valid_metadata_str(&manifest.upstream_commit)
                || !valid_metadata_str(&manifest.checkpoint_repository)
                || !valid_metadata_str(&manifest.checkpoint_revision)
            {
                return Err(invalid("invalid local-ncm-v1 manifest provenance fields"));
            }
        } else if manifest.checkpoint_repository != "hazyresearch/ncm-checkpoints"
            || !matches!(
                (
                    manifest.upstream_commit.as_str(),
                    manifest.checkpoint_revision.as_str(),
                ),
                (UPSTREAM, CHECKPOINT_REVISION)
                    | (UPSTREAM_2026_11_23, CHECKPOINT_REVISION_2026_11_23)
            )
        {
            return Err(invalid("unsupported schema, profile or source pins"));
        }
        if reusable && manifest.profile != "wmt14-inference-v1" {
            return Err(invalid("reusable bundle loader requires WMT14"));
        }
        let (max_file, max_total): (u64, u64) = if is_local {
            let declared_files: BTreeSet<&str> =
                manifest.files.keys().map(String::as_str).collect();
            let required: BTreeSet<&str> = LOCAL_REQUIRED_FILES.iter().copied().collect();
            let allowed: BTreeSet<&str> = LOCAL_ALLOWED_FILES.iter().copied().collect();
            if !required.is_subset(&declared_files) || !declared_files.is_subset(&allowed) {
                return Err(invalid("unexpected file/bundle inventory"));
            }
            let expected_bundles: BTreeSet<&str> = declared_files
                .iter()
                .copied()
                .filter(|name| name.ends_with(".safetensors"))
                .collect();
            let declared_bundles: BTreeSet<&str> =
                manifest.tensors.keys().map(String::as_str).collect();
            if declared_bundles != expected_bundles {
                return Err(invalid("unexpected file/bundle inventory"));
            }
            (LOCAL_MAX_FILE, LOCAL_MAX_TOTAL)
        } else {
            let (files, bundles, max_file, max_total): (&[&str], &[&str], u64, u64) = match manifest
                .profile
                .as_str()
            {
                "sudoku-easy-inspection-v1"
                | "sudoku-medium-inspection-v1"
                | "sudoku-hard-inspection-v1" => (&FILES, &BUNDLES, MAX_FILE, MAX_TOTAL),
                "tinystories-inference-v1" => (
                    &TINYSTORIES_FILES,
                    &TINYSTORIES_BUNDLES,
                    TINYSTORIES_MAX_FILE,
                    TINYSTORIES_MAX_TOTAL,
                ),
                "xsum-inference-v1" | "xsum-inference-v2" => (
                    &XSUM_FILES,
                    &TINYSTORIES_BUNDLES,
                    XSUM_MAX_FILE,
                    XSUM_MAX_TOTAL,
                ),
                "wmt14-inference-v1" => (&WMT14_FILES, &BUNDLES, WMT14_MAX_FILE, WMT14_MAX_TOTAL),
                "lm1b-small-inference-v1" | "lm1b-small-inference-v2" => (
                    &TINYSTORIES_FILES,
                    &TINYSTORIES_BUNDLES,
                    LM1B_SMALL_MAX_FILE,
                    LM1B_SMALL_MAX_TOTAL,
                ),
                "lm1b-medium-inference-v1" | "lm1b-medium-inference-v2" => (
                    &TINYSTORIES_FILES,
                    &TINYSTORIES_BUNDLES,
                    LM1B_MEDIUM_MAX_FILE,
                    LM1B_MEDIUM_MAX_TOTAL,
                ),
                "lm1b-large-inference-v1" | "lm1b-large-inference-v2" => (
                    &TINYSTORIES_FILES,
                    &TINYSTORIES_BUNDLES,
                    LM1B_LARGE_MAX_FILE,
                    LM1B_LARGE_MAX_TOTAL,
                ),
                _ => return Err(invalid("unsupported schema, profile or source pins")),
            };
            // Exact flat filename sets reject unknown files, absolute paths, traversal and aliases.
            if manifest
                .files
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                != files.iter().copied().collect()
                || manifest
                    .tensors
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    != bundles.iter().copied().collect()
            {
                return Err(invalid("unexpected file/bundle inventory"));
            }
            (max_file, max_total)
        };
        let mut total = 0u64;
        for record in manifest.files.values() {
            check_hash_format(&record.sha256)?;
            total = total
                .checked_add(record.bytes)
                .ok_or_else(|| invalid("file-size overflow"))?;
            if record.bytes > max_file || total > max_total {
                return Err(invalid("artifact exceeds memory budget"));
            }
        }
        let mut payloads = BTreeMap::new();
        let mut streaming = BTreeMap::new();
        let reusable_wmt14 = if reusable {
            let codec = &manifest.files["codec.safetensors"];
            let generator = &manifest.files["generator.safetensors"];
            let capacity = usize::try_from(codec.bytes.max(generator.bytes) + 1)
                .map_err(|_| invalid("bundle capacity overflow"))?;
            let mut data = wmt14_scratch::Scratch::new(capacity)?;
            for (name, file) in [
                ("codec.safetensors", codec),
                ("generator.safetensors", generator),
            ] {
                data.read(&root.join(name), file.bytes)?;
                std::thread::scope(|scope| -> Result<()> {
                    let bytes = data.bytes();
                    let entries = &manifest.tensors[name];
                    let whole = scope.spawn(move || verify_payload(name, bytes, file));
                    let bundle_res = verify_bundle(name, bytes, entries);
                    whole
                        .join()
                        .map_err(|_| invalid("hash thread panicked"))??;
                    bundle_res
                })?;
            }
            Some(ReusableWmt14 {
                root: root.clone(),
                name: "generator.safetensors",
                data,
            })
        } else {
            None
        };
        if streaming_mode {
            for (name, record) in &manifest.files {
                if matches!(
                    name.as_str(),
                    "codec.safetensors" | "generator.safetensors" | "training.safetensors"
                ) {
                    continue;
                }
                let bytes = bounded_read(&root.join(name), record.bytes)?;
                verify_payload(name, &bytes, record)?;
                payloads.insert(name.clone(), bytes);
            }
            if let Some(entries) = manifest.tensors.get("inference.safetensors") {
                verify_bundle(
                    "inference.safetensors",
                    &payloads["inference.safetensors"],
                    entries,
                )?;
            }
            let stream_results: Vec<(&'static str, StreamedBundle)> =
                std::thread::scope(|scope| -> Result<Vec<(&'static str, StreamedBundle)>> {
                    let mut handles = Vec::new();
                    for name in [
                        "codec.safetensors",
                        "generator.safetensors",
                        "training.safetensors",
                    ] {
                        if let (Some(record), Some(entries)) =
                            (manifest.files.get(name), manifest.tensors.get(name))
                        {
                            let path = root.join(name);
                            let check_finite = name != "training.safetensors";
                            handles.push(scope.spawn(move || {
                                validate_streaming_bundle(
                                    &path,
                                    name,
                                    record,
                                    entries,
                                    check_finite,
                                )
                                .map(|b| (name, b))
                            }));
                        }
                    }
                    let mut out = Vec::new();
                    for h in handles {
                        out.push(
                            h.join()
                                .map_err(|_| invalid("streaming validation thread panicked"))??,
                        );
                    }
                    Ok(out)
                })?;
            for (name, bundle) in stream_results {
                if name != "training.safetensors" {
                    streaming.insert(name.to_owned(), bundle);
                }
            }
        } else {
            for (name, record) in &manifest.files {
                if inference && name == "training.safetensors" {
                    validate_training_stream(&root.join(name), record, &manifest.tensors[name])?;
                    continue;
                }
                if reusable
                    && matches!(name.as_str(), "codec.safetensors" | "generator.safetensors")
                {
                    continue;
                }
                let bytes = bounded_read(&root.join(name), record.bytes)?;
                if let Some(entries) = manifest.tensors.get(name) {
                    std::thread::scope(|scope| -> Result<()> {
                        let bytes_ref = bytes.as_slice();
                        let whole = scope.spawn(move || verify_payload(name, bytes_ref, record));
                        let bundle_res = verify_bundle(name, bytes_ref, entries);
                        whole
                            .join()
                            .map_err(|_| invalid("hash thread panicked"))??;
                        bundle_res
                    })?;
                } else {
                    verify_payload(name, &bytes, record)?;
                }
                payloads.insert(name.clone(), bytes);
            }
        }
        Ok(Self {
            manifest,
            payloads,
            reusable_wmt14,
            streaming,
        })
    }

    /// Replace the current checked WMT14 weight bundle with the other one.
    /// A file change or failed read consumes the incomplete view and returns an error.
    pub fn with_wmt14_bundle(mut self, name: &str) -> Result<Self> {
        let scratch = self
            .reusable_wmt14
            .as_mut()
            .ok_or_else(|| invalid("not a reusable WMT14 artifact"))?;
        let next = match name {
            "codec.safetensors" if scratch.name == "generator.safetensors" => "codec.safetensors",
            "generator.safetensors" if scratch.name == "codec.safetensors" => {
                "generator.safetensors"
            }
            _ => return Err(invalid("unexpected WMT14 bundle")),
        };
        let record = &self.manifest.files[next];
        scratch.data.read(&scratch.root.join(next), record.bytes)?;
        verify_payload(next, scratch.data.bytes(), record)?;
        if next == "generator.safetensors" {
            scratch.data.trim()?;
        }
        scratch.name = next;
        Ok(self)
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Return the validated byte length of a declared file.
    pub fn file_bytes(&self, name: &str) -> Result<u64> {
        self.manifest
            .files
            .get(name)
            .map(|r| r.bytes)
            .ok_or_else(|| invalid("undeclared artifact file"))
    }

    /// Return the validated tensor inventory for a declared bundle.
    pub fn bundle_tensors(&self, bundle: &str) -> Result<&[TensorRecord]> {
        self.manifest
            .tensors
            .get(bundle)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("undeclared artifact bundle"))
    }

    /// Whether the bundle is backed by an open validated file descriptor rather than an in-memory buffer.
    pub fn is_streaming_bundle(&self, bundle: &str) -> bool {
        self.streaming.contains_key(bundle)
    }

    /// Whether every F32 tensor in `bundle` (except `attn_mask_for_masking`) has already
    /// been verified finite during streaming artifact validation.
    pub fn bundle_f32_weights_verified_finite(&self, bundle: &str) -> bool {
        self.streaming
            .get(bundle)
            .is_some_and(|s| s.finite_checked_f32)
    }

    /// Read a single tensor's raw little-endian bytes from a validated bundle.
    pub fn read_bundle_tensor(&self, bundle: &str, name: &str) -> Result<Vec<u8>> {
        let entries = self.bundle_tensors(bundle)?;
        if let Some(streamed) = self.streaming.get(bundle) {
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                let &(idx, offset, len) = streamed
                    .ordered
                    .iter()
                    .find(|&&(idx, _, _)| entries[idx].name == name)
                    .ok_or_else(|| invalid(format!("missing tensor: {bundle}/{name}")))?;
                let _ = idx;
                let mut out = vec![0u8; len];
                streamed.file.read_exact_at(&mut out, offset)?;
                return Ok(out);
            }
            #[cfg(not(unix))]
            {
                let _ = streamed;
                return Err(invalid("streaming bundle requires unix"));
            }
        }
        let bytes = self.file(bundle)?;
        let tensors = SafeTensors::deserialize(bytes)?;
        let tensor = tensors.tensor(name)?;
        Ok(tensor.data().to_vec())
    }

    /// Visit each tensor in a validated bundle. For streaming artifacts, tensors are read
    /// sequentially one at a time into a single reusable buffer (immediately unmapped on macOS).
    pub fn visit_bundle_tensors<E>(
        &self,
        bundle: &str,
        mut map_io: impl FnMut(Error) -> E,
        mut f: impl FnMut(&str, &[usize], TensorDtype, &[u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let entries = self.bundle_tensors(bundle).map_err(&mut map_io)?;
        if let Some(streamed) = self.streaming.get(bundle) {
            #[cfg(unix)]
            {
                let max_bytes = entries.iter().map(|e| e.bytes).max().unwrap_or(0);
                let mut scratch =
                    wmt14_scratch::Scratch::new(max_bytes.max(1)).map_err(&mut map_io)?;
                for &(idx, offset, len) in &streamed.ordered {
                    let entry = &entries[idx];
                    scratch
                        .read_at(&streamed.file, offset, len)
                        .map_err(&mut map_io)?;
                    f(&entry.name, &entry.shape, entry.dtype, scratch.bytes())?;
                }
                return Ok(());
            }
            #[cfg(not(unix))]
            {
                let _ = streamed;
                return Err(map_io(invalid("streaming bundle requires unix")));
            }
        }
        let bytes = self.file(bundle).map_err(&mut map_io)?;
        let tensors = SafeTensors::deserialize(bytes).map_err(|e| map_io(Error::Tensor(e)))?;
        for entry in entries {
            let tensor = tensors
                .tensor(&entry.name)
                .map_err(|e| map_io(Error::Tensor(e)))?;
            f(&entry.name, &entry.shape, entry.dtype, tensor.data())?;
        }
        Ok(())
    }

    /// Release a validated payload when no longer needed; subsequent access fails.
    pub fn release_file(&mut self, name: &str) -> Result<()> {
        if self.streaming.remove(name).is_some() {
            return Ok(());
        }
        if self
            .reusable_wmt14
            .as_ref()
            .is_some_and(|scratch| scratch.name == name)
        {
            self.reusable_wmt14 = None;
            return Ok(());
        }
        self.payloads
            .remove(name)
            .map(drop)
            .ok_or_else(|| invalid("undeclared or released artifact file"))
    }
    /// Return validated owned backing storage, suitable for a copied backend loader.
    pub fn file(&self, name: &str) -> Result<&[u8]> {
        if let Some(scratch) = self
            .reusable_wmt14
            .as_ref()
            .filter(|scratch| scratch.name == name)
        {
            return Ok(scratch.data.bytes());
        }
        self.payloads
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("undeclared artifact file"))
    }
}

#[cfg(test)]
mod read_tests {
    use super::*;
    #[test]
    fn sha256_known_answers_cover_multiblock_hardware_path() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
    #[test]
    fn bounded_reads_reserve_once_and_keep_limits() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bytes");
        for length in [0, 1, 31, 32768, 1024 * 1024 + 17] {
            let bytes = vec![42u8; length];
            fs::write(&path, &bytes).unwrap();
            let loaded = bounded_read(&path, length as u64).unwrap();
            assert_eq!(loaded, bytes);
            assert_eq!(loaded.capacity(), length + 1);
            if length > 0 {
                assert!(bounded_read(&path, length as u64 - 1).is_err());
            }
        }
    }
}
