//! Standard model and upstream cache path resolution for repo checkouts and global installs.
use std::{
    env,
    path::{Path, PathBuf},
};

/// Resolved models directory together with the rule that selected it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelsDirResolution {
    pub path: PathBuf,
    pub source: &'static str,
}

/// Resolve the default models directory using the standard precedence order:
/// 1. Explicit CLI flag (`--models-dir`)
/// 2. `NCM_MODELS_DIR` environment variable (if non-empty)
/// 3. `./artifacts/canonical` when present in the current working directory
/// 4. Platform user data directory (`$XDG_DATA_HOME/ncm/models` or `~/.local/share/ncm/models`
///    on Linux/Unix; `~/Library/Application Support/ncm/models` on macOS)
pub(crate) fn resolve_models_dir(explicit: Option<&Path>) -> ModelsDirResolution {
    let env_models = env::var("NCM_MODELS_DIR").ok();
    let repo_exists = Path::new("artifacts/canonical").is_dir();
    let xdg_data = env::var("XDG_DATA_HOME").ok();
    let home = env::var("HOME").ok();
    resolve_models_dir_with(
        explicit,
        env_models.as_deref(),
        repo_exists,
        xdg_data.as_deref(),
        home.as_deref(),
        cfg!(target_os = "macos"),
    )
}

/// Resolve the default upstream checkpoint cache directory for `ncm pull`:
/// 1. Explicit CLI flag (`--upstream-dir`)
/// 2. `NCM_UPSTREAM_DIR` environment variable (if non-empty)
/// 3. `./artifacts/upstream` when running inside a repo checkout (`./artifacts/canonical` or
///    `./artifacts/upstream` exists)
/// 4. Platform user cache directory (`$XDG_CACHE_HOME/ncm/upstream` or `~/.cache/ncm/upstream`
///    on Linux/Unix; `~/Library/Caches/ncm/upstream` on macOS)
pub(crate) fn resolve_upstream_dir(explicit: Option<&Path>) -> PathBuf {
    let env_upstream = env::var("NCM_UPSTREAM_DIR").ok();
    let repo_exists =
        Path::new("artifacts/canonical").is_dir() || Path::new("artifacts/upstream").is_dir();
    let xdg_cache = env::var("XDG_CACHE_HOME").ok();
    let home = env::var("HOME").ok();
    resolve_upstream_dir_with(
        explicit,
        env_upstream.as_deref(),
        repo_exists,
        xdg_cache.as_deref(),
        home.as_deref(),
        cfg!(target_os = "macos"),
    )
}

fn non_empty_trimmed(val: Option<&str>) -> Option<&str> {
    val.map(str::trim).filter(|s| !s.is_empty())
}

pub(crate) fn resolve_models_dir_with(
    explicit: Option<&Path>,
    env_models: Option<&str>,
    repo_exists: bool,
    xdg_data: Option<&str>,
    home: Option<&str>,
    is_macos: bool,
) -> ModelsDirResolution {
    if let Some(dir) = explicit {
        return ModelsDirResolution {
            path: dir.to_path_buf(),
            source: "cli",
        };
    }
    if let Some(dir) = non_empty_trimmed(env_models) {
        return ModelsDirResolution {
            path: PathBuf::from(dir),
            source: "env:NCM_MODELS_DIR",
        };
    }
    if repo_exists {
        return ModelsDirResolution {
            path: PathBuf::from("artifacts/canonical"),
            source: "repo:artifacts/canonical",
        };
    }
    if is_macos {
        if let Some(home) = non_empty_trimmed(home) {
            return ModelsDirResolution {
                path: PathBuf::from(home).join("Library/Application Support/ncm/models"),
                source: "user-data-dir",
            };
        }
    } else if let Some(xdg) = non_empty_trimmed(xdg_data) {
        return ModelsDirResolution {
            path: PathBuf::from(xdg).join("ncm/models"),
            source: "user-data-dir",
        };
    } else if let Some(home) = non_empty_trimmed(home) {
        return ModelsDirResolution {
            path: PathBuf::from(home).join(".local/share/ncm/models"),
            source: "user-data-dir",
        };
    }
    ModelsDirResolution {
        path: PathBuf::from("artifacts/canonical"),
        source: "fallback:artifacts/canonical",
    }
}

pub(crate) fn resolve_upstream_dir_with(
    explicit: Option<&Path>,
    env_upstream: Option<&str>,
    repo_exists: bool,
    xdg_cache: Option<&str>,
    home: Option<&str>,
    is_macos: bool,
) -> PathBuf {
    if let Some(dir) = explicit {
        return dir.to_path_buf();
    }
    if let Some(dir) = non_empty_trimmed(env_upstream) {
        return PathBuf::from(dir);
    }
    if repo_exists {
        return PathBuf::from("artifacts/upstream");
    }
    if is_macos {
        if let Some(home) = non_empty_trimmed(home) {
            return PathBuf::from(home).join("Library/Caches/ncm/upstream");
        }
    } else if let Some(xdg) = non_empty_trimmed(xdg_cache) {
        return PathBuf::from(xdg).join("ncm/upstream");
    } else if let Some(home) = non_empty_trimmed(home) {
        return PathBuf::from(home).join(".cache/ncm/upstream");
    }
    PathBuf::from("artifacts/upstream")
}

#[cfg(test)]
mod tests {
    use super::{resolve_models_dir_with, resolve_upstream_dir_with};
    use std::path::{Path, PathBuf};

    #[test]
    fn models_dir_precedence_respects_cli_env_repo_and_platform_defaults() {
        let cli = resolve_models_dir_with(
            Some(Path::new("/custom/models")),
            Some("/env/models"),
            true,
            Some("/xdg"),
            Some("/home/u"),
            false,
        );
        assert_eq!(cli.path, PathBuf::from("/custom/models"));
        assert_eq!(cli.source, "cli");

        let env = resolve_models_dir_with(
            None,
            Some("/env/models"),
            true,
            Some("/xdg"),
            Some("/home/u"),
            false,
        );
        assert_eq!(env.path, PathBuf::from("/env/models"));
        assert_eq!(env.source, "env:NCM_MODELS_DIR");

        let repo = resolve_models_dir_with(
            None,
            Some("   "),
            true,
            Some("/xdg"),
            Some("/home/u"),
            false,
        );
        assert_eq!(repo.path, PathBuf::from("artifacts/canonical"));
        assert_eq!(repo.source, "repo:artifacts/canonical");

        let linux_xdg =
            resolve_models_dir_with(None, None, false, Some("/xdg/data"), Some("/home/u"), false);
        assert_eq!(linux_xdg.path, PathBuf::from("/xdg/data/ncm/models"));
        assert_eq!(linux_xdg.source, "user-data-dir");

        let linux_home = resolve_models_dir_with(None, None, false, None, Some("/home/u"), false);
        assert_eq!(
            linux_home.path,
            PathBuf::from("/home/u/.local/share/ncm/models")
        );

        let mac_home = resolve_models_dir_with(None, None, false, None, Some("/Users/u"), true);
        assert_eq!(
            mac_home.path,
            PathBuf::from("/Users/u/Library/Application Support/ncm/models")
        );
    }

    #[test]
    fn upstream_dir_follows_platform_cache_when_outside_repo() {
        assert_eq!(
            resolve_upstream_dir_with(None, None, true, None, Some("/home/u"), false),
            PathBuf::from("artifacts/upstream")
        );
        assert_eq!(
            resolve_upstream_dir_with(
                None,
                None,
                false,
                Some("/xdg/cache"),
                Some("/home/u"),
                false
            ),
            PathBuf::from("/xdg/cache/ncm/upstream")
        );
        assert_eq!(
            resolve_upstream_dir_with(None, None, false, None, Some("/Users/u"), true),
            PathBuf::from("/Users/u/Library/Caches/ncm/upstream")
        );
    }
}
