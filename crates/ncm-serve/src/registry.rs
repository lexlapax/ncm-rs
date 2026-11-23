//! Local NCM artifact discovery and friendly alias resolution.
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

/// Metadata for one discovered or explicitly registered local NCM artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Canonical model name exposed to OpenAI (`/v1/models`) and Ollama (`/api/tags`) clients.
    pub name: String,
    /// Additional lookup aliases (such as directory basename or `:latest` suffix).
    pub aliases: Vec<String>,
    /// Filesystem directory containing `manifest.json` and safetensors bundles.
    pub path: PathBuf,
    /// Artifact profile from `manifest.json` (e.g., `"xsum-inference-v1"`, `"local-ncm-v1"`).
    pub profile: String,
    /// Total byte size of files listed in `manifest.json` (or directory files if unavailable).
    pub size_bytes: u64,
    /// Last modification timestamp of `manifest.json` as Unix seconds.
    pub modified_unix: u64,
    /// Short stable digest prefix derived from manifest metadata.
    pub digest: String,
    /// Precision mode configured on the serving engine (`fp32`, `bf16`, `fp16`).
    pub precision: String,
    /// Whether this profile accepts UTF-8 text prompts directly.
    pub supports_text: bool,
    /// Whether this profile supports deterministic `--seed` sampling.
    pub supports_seed: bool,
    /// Whether this profile supports classifier-free guidance (`cfg_scale`).
    pub supports_cfg: bool,
}

#[derive(Debug, Deserialize)]
struct LightManifest {
    profile: String,
    #[serde(default)]
    files: BTreeMap<String, LightFile>,
}

#[derive(Debug, Deserialize)]
struct LightFile {
    #[serde(default)]
    bytes: u64,
    #[serde(default)]
    sha256: String,
}

/// Supported inference artifact profiles that can be served by `ncm serve`.
pub fn is_servable_profile(profile: &str) -> bool {
    matches!(
        profile,
        "sudoku-easy-inspection-v1"
            | "sudoku-medium-inspection-v1"
            | "sudoku-hard-inspection-v1"
            | "tinystories-inference-v1"
            | "xsum-inference-v1"
            | "xsum-inference-v2"
            | "wmt14-inference-v1"
            | "lm1b-small-inference-v1"
            | "lm1b-small-inference-v2"
            | "lm1b-medium-inference-v1"
            | "lm1b-medium-inference-v2"
            | "lm1b-large-inference-v1"
            | "lm1b-large-inference-v2"
            | "local-ncm-v1"
    )
}

fn profile_capabilities(profile: &str) -> (bool, bool, bool) {
    match profile {
        "xsum-inference-v1" | "xsum-inference-v2" => (true, true, true),
        "wmt14-inference-v1" => (true, false, true),
        "tinystories-inference-v1" => (true, true, false),
        "lm1b-small-inference-v1"
        | "lm1b-small-inference-v2"
        | "lm1b-medium-inference-v1"
        | "lm1b-medium-inference-v2"
        | "lm1b-large-inference-v1"
        | "lm1b-large-inference-v2" => (true, true, false),
        "local-ncm-v1" => (true, true, false),
        "sudoku-easy-inspection-v1"
        | "sudoku-medium-inspection-v1"
        | "sudoku-hard-inspection-v1" => (false, false, false),
        _ => (false, false, false),
    }
}

fn default_short_name(profile: &str, dir_name: &str) -> String {
    match profile {
        "sudoku-easy-inspection-v1" => "sudoku-easy".to_owned(),
        "sudoku-medium-inspection-v1" => "sudoku-medium".to_owned(),
        "sudoku-hard-inspection-v1" => "sudoku-hard".to_owned(),
        "xsum-inference-v1" | "xsum-inference-v2" => "xsum".to_owned(),
        "wmt14-inference-v1" => "wmt14".to_owned(),
        "tinystories-inference-v1" => "tinystories".to_owned(),
        "lm1b-small-inference-v1" | "lm1b-small-inference-v2" => "lm1b-small".to_owned(),
        "lm1b-medium-inference-v1" | "lm1b-medium-inference-v2" => "lm1b-medium".to_owned(),
        "lm1b-large-inference-v1" | "lm1b-large-inference-v2" => "lm1b-large".to_owned(),
        _ => dir_name.to_owned(),
    }
}

fn profile_priority(profile: &str) -> u8 {
    // Prefer self-contained v2 LM1B artifacts over v1 when both exist in the same directory.
    if profile.ends_with("-inference-v2") {
        2
    } else {
        1
    }
}

/// Inspect a single artifact directory and construct a `ModelEntry` if its `manifest.json`
/// declares a servable NCM inference profile.
pub fn inspect_model_dir(
    dir: &Path,
    explicit_name: Option<&str>,
    precision: &str,
) -> std::result::Result<Option<ModelEntry>, String> {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let meta = fs::metadata(&manifest_path)
        .map_err(|e| format!("failed to stat {}: {e}", manifest_path.display()))?;
    if meta.len() > 4 * 1024 * 1024 {
        return Ok(None);
    }
    let raw = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("failed to read {}: {e}", manifest_path.display()))?;
    let parsed: LightManifest = match serde_json::from_str(&raw) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    if !is_servable_profile(&parsed.profile) {
        return Ok(None);
    }
    let dir_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_owned();
    let name = explicit_name
        .map(str::to_owned)
        .unwrap_or_else(|| default_short_name(&parsed.profile, &dir_name));
    let mut aliases = Vec::new();
    if dir_name != name {
        aliases.push(dir_name);
    }
    let size_bytes: u64 = parsed.files.values().map(|f| f.bytes).sum();
    let modified_unix = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let digest = parsed
        .files
        .values()
        .next()
        .map(|f| f.sha256.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{:016x}", size_bytes));
    let (supports_text, supports_seed, supports_cfg) = profile_capabilities(&parsed.profile);
    Ok(Some(ModelEntry {
        name,
        aliases,
        path: dir.to_path_buf(),
        profile: parsed.profile,
        size_bytes,
        modified_unix,
        digest,
        precision: precision.to_owned(),
        supports_text,
        supports_seed,
        supports_cfg,
    }))
}

/// Collection of locally discovered and explicitly registered NCM models.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    entries: BTreeMap<String, ModelEntry>,
    aliases: BTreeMap<String, String>,
}

impl ModelRegistry {
    /// Discover models inside an optional root directory plus explicit `name=path` overrides.
    pub fn discover(
        models_dir: Option<&Path>,
        overrides: &[(String, PathBuf)],
        precision: &str,
    ) -> std::result::Result<Self, String> {
        let mut reg = Self::default();
        if let Some(root) = models_dir
            && root.is_dir()
        {
            // Support pointing `--models-dir` directly at a single artifact directory as well.
            if let Some(entry) = inspect_model_dir(root, None, precision)? {
                reg.insert_discovered(entry);
            } else {
                let mut subdirs: Vec<PathBuf> = fs::read_dir(root)
                    .map_err(|e| format!("failed to read models dir {}: {e}", root.display()))?
                    .filter_map(|res| res.ok().map(|e| e.path()))
                    .filter(|p| p.is_dir())
                    .collect();
                subdirs.sort();
                for sub in subdirs {
                    if let Ok(Some(entry)) = inspect_model_dir(&sub, None, precision) {
                        reg.insert_discovered(entry);
                    }
                }
            }
        }
        for (name, path) in overrides {
            if name.trim().is_empty() {
                return Err("model override name must be non-empty".into());
            }
            let entry =
                inspect_model_dir(path, Some(name.trim()), precision)?.ok_or_else(|| {
                    format!(
                        "model override `{name}` at {} is missing a supported manifest.json",
                        path.display()
                    )
                })?;
            reg.insert_override(entry);
        }
        Ok(reg)
    }

    fn insert_discovered(&mut self, entry: ModelEntry) {
        let canonical = entry.name.clone();
        if let Some(existing) = self.entries.get(&canonical) {
            let new_prio = profile_priority(&entry.profile);
            let old_prio = profile_priority(&existing.profile);
            if new_prio > old_prio {
                // Keep previous entry reachable by its directory basename while promoting v2.
                let demoted_name = existing
                    .path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("legacy")
                    .to_owned();
                let mut demoted = existing.clone();
                demoted.name = demoted_name.clone();
                demoted.aliases.clear();
                self.entries.insert(demoted_name, demoted);
                self.register_entry(entry);
            } else {
                // Keep existing canonical name and expose the additional artifact under its basename.
                let alt_name = entry
                    .path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&entry.name)
                    .to_owned();
                if !self.entries.contains_key(&alt_name) {
                    let mut alt = entry;
                    alt.name = alt_name;
                    alt.aliases.clear();
                    self.register_entry(alt);
                }
            }
        } else {
            self.register_entry(entry);
        }
    }

    fn insert_override(&mut self, entry: ModelEntry) {
        self.register_entry(entry);
    }

    fn register_entry(&mut self, entry: ModelEntry) {
        let name = entry.name.clone();
        for alias in &entry.aliases {
            self.aliases.insert(alias.clone(), name.clone());
        }
        self.entries.insert(name, entry);
    }

    /// Return all canonical model entries sorted by name.
    pub fn list(&self) -> Vec<ModelEntry> {
        self.entries.values().cloned().collect()
    }

    /// Resolve a requested model name, accepting exact names, directory aliases, and Ollama's
    /// `:latest` tag suffix (e.g., `"xsum:latest"` -> `"xsum"`).
    pub fn resolve(&self, requested: &str) -> Option<&ModelEntry> {
        let key = requested.trim();
        if let Some(e) = self.entries.get(key) {
            return Some(e);
        }
        if let Some(target) = self.aliases.get(key)
            && let Some(e) = self.entries.get(target)
        {
            return Some(e);
        }
        if let Some(stripped) = key.strip_suffix(":latest") {
            if let Some(e) = self.entries.get(stripped) {
                return Some(e);
            }
            if let Some(target) = self.aliases.get(stripped) {
                return self.entries.get(target);
            }
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
