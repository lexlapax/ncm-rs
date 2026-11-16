//! Bounded copied loading for the Phase 1 Sudoku inspection bundle.
//!
//! Verifies integrity relative to a manifest, not authenticity of that manifest.
//! No pickle execution, mmap, model construction, or implicit downloads.

use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::Path,
};

pub const UPSTREAM: &str = "c70274273f489abc73832b2b98584a8483b6b203";
pub const CHECKPOINT_REVISION: &str = "d342ce8aaea58830eef3b8ad473664673e41f692";
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

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
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
    let mut data = Vec::new();
    data.try_reserve_exact(capacity)
        .map_err(|_| invalid("file allocation failed"))?;
    fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut data)?;
    if data.len() as u64 > limit {
        return Err(invalid("file grew beyond limit"));
    }
    Ok(data)
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
}

impl Artifact {
    /// Validate the complete artifact, streaming the training-only bundle.
    /// Inference payloads remain owned and immutable; `file("training.safetensors")`
    /// explicitly fails on this reduced view. `load` retains the full artifact.
    pub fn load_inference(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_impl(root.as_ref(), true)
    }

    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_impl(root.as_ref(), false)
    }

    fn load_impl(root: &Path, inference: bool) -> Result<Self> {
        // Strip trailing separators/dot components before the no-symlink leaf check.
        let root = root.components().collect::<std::path::PathBuf>();
        if !fs::symlink_metadata(&root)?.file_type().is_dir() {
            return Err(invalid("artifact root must be a real directory"));
        }
        let manifest: Manifest =
            serde_json::from_slice(&bounded_read(&root.join("manifest.json"), MAX_MANIFEST)?)?;
        if manifest.schema_version != 1
            || manifest.upstream_commit != UPSTREAM
            || manifest.checkpoint_revision != CHECKPOINT_REVISION
            || manifest.checkpoint_repository != "hazyresearch/ncm-checkpoints"
        {
            return Err(invalid("unsupported schema, profile or source pins"));
        }
        let (files, bundles, max_file, max_total): (&[&str], &[&str], u64, u64) =
            match manifest.profile.as_str() {
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
        for (name, record) in &manifest.files {
            if inference && name == "training.safetensors" {
                validate_training_stream(&root.join(name), record, &manifest.tensors[name])?;
                continue;
            }
            let bytes = bounded_read(&root.join(name), record.bytes)?;
            if bytes.len() as u64 != record.bytes || sha256(&bytes) != record.sha256 {
                return Err(invalid(format!("file size/hash mismatch: {name}")));
            }
            if name.ends_with(".json") {
                let _: serde_json::Value = serde_json::from_slice(&bytes)?;
            }
            payloads.insert(name.clone(), bytes);
        }
        for (bundle, entries) in &manifest.tensors {
            if inference && bundle == "training.safetensors" {
                continue;
            }
            let tensors = SafeTensors::deserialize(&payloads[bundle])?;
            let names = entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<BTreeSet<_>>();
            if names.len() != entries.len() || names != tensors.names().into_iter().collect() {
                return Err(invalid(format!("missing/extra/duplicate tensor: {bundle}")));
            }
            for entry in entries {
                check_hash_format(&entry.sha256)?;
                if entry.shape.len() > 8 {
                    return Err(invalid("tensor rank exceeds limit"));
                }
                let bytes = entry
                    .shape
                    .iter()
                    .try_fold(entry.dtype.size(), |size, dim| size.checked_mul(*dim))
                    .ok_or_else(|| invalid("tensor shape overflow"))?;
                let tensor = tensors.tensor(&entry.name)?;
                if bytes != entry.bytes
                    || tensor.shape() != entry.shape
                    || tensor.dtype() != entry.dtype.native()
                    || tensor.data().len() != entry.bytes
                    || sha256(tensor.data()) != entry.sha256
                {
                    return Err(invalid(format!(
                        "tensor metadata/hash mismatch: {bundle}/{}",
                        entry.name
                    )));
                }
            }
        }
        Ok(Self { manifest, payloads })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    /// Release a validated payload when no longer needed; subsequent access fails.
    pub fn release_file(&mut self, name: &str) -> Result<()> {
        self.payloads
            .remove(name)
            .map(drop)
            .ok_or_else(|| invalid("undeclared or released artifact file"))
    }
    /// Return validated owned backing storage, suitable for a copied backend loader.
    pub fn file(&self, name: &str) -> Result<&[u8]> {
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
