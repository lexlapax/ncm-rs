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
    let mut data = Vec::new();
    fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut data)?;
    if data.len() as u64 > limit {
        return Err(invalid("file grew beyond limit"));
    }
    Ok(data)
}

pub struct Artifact {
    manifest: Manifest,
    payloads: BTreeMap<String, Vec<u8>>,
}

impl Artifact {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        // Strip trailing separators/dot components before the no-symlink leaf check.
        let root = root.as_ref().components().collect::<std::path::PathBuf>();
        if !fs::symlink_metadata(&root)?.file_type().is_dir() {
            return Err(invalid("artifact root must be a real directory"));
        }
        let manifest: Manifest =
            serde_json::from_slice(&bounded_read(&root.join("manifest.json"), MAX_MANIFEST)?)?;
        if manifest.schema_version != 1
            || manifest.profile != "sudoku-easy-inspection-v1"
            || manifest.upstream_commit != UPSTREAM
            || manifest.checkpoint_revision != CHECKPOINT_REVISION
            || manifest.checkpoint_repository != "hazyresearch/ncm-checkpoints"
        {
            return Err(invalid("unsupported schema, profile or source pins"));
        }
        // Exact flat filename sets reject unknown files, absolute paths, traversal and aliases.
        if manifest
            .files
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != FILES.into_iter().collect()
            || manifest
                .tensors
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                != BUNDLES.into_iter().collect()
        {
            return Err(invalid("unexpected file/bundle inventory"));
        }
        let mut total = 0u64;
        for record in manifest.files.values() {
            check_hash_format(&record.sha256)?;
            total = total
                .checked_add(record.bytes)
                .ok_or_else(|| invalid("file-size overflow"))?;
            if record.bytes > MAX_FILE || total > MAX_TOTAL {
                return Err(invalid("artifact exceeds memory budget"));
            }
        }
        let mut payloads = BTreeMap::new();
        for (name, record) in &manifest.files {
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
    /// Return validated owned backing storage, suitable for a copied backend loader.
    pub fn file(&self, name: &str) -> Result<&[u8]> {
        self.payloads
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("undeclared artifact file"))
    }
}
