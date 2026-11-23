//! Zero-dependency OpenAI- and Ollama-compatible HTTP/1.1 serving layer for local NCM models.
//!
//! Exposes both OpenAI drop-in endpoints (`/v1/models`, `/v1/completions`,
//! `/v1/chat/completions`) and Ollama drop-in endpoints (`/`, `/health`, `/api/tags`,
//! `/api/version`, `/api/show`, `/api/generate`, `/api/chat`), supporting streaming
//! (`text/event-stream` SSE for OpenAI, `application/x-ndjson` for Ollama) and
//! non-streaming JSON responses over a caller-provided [`InferenceEngine`].

mod api;
mod http;
mod registry;
mod server;

pub use registry::{ModelEntry, ModelRegistry, inspect_model_dir, is_servable_profile};
pub use server::{Server, ServerHandle};

/// Normalized generation request passed from HTTP endpoints to the backend engine.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub seed: Option<u64>,
    pub cfg_scale: Option<f64>,
    pub max_tokens: Option<usize>,
    pub stop: Vec<String>,
}

/// Normalized generation result returned by the backend engine.
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub model: String,
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub load_duration_ns: u64,
    pub eval_duration_ns: u64,
    pub finish_reason: String,
}

/// Structured engine errors mapped to HTTP 404, 400, and 500 JSON responses.
#[derive(Debug, Clone)]
pub enum EngineError {
    NotFound(String),
    BadRequest(String),
    Internal(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(m) | Self::BadRequest(m) | Self::Internal(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for EngineError {}

/// Runtime metadata for one model currently resident in device memory (`GET /api/ps`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LoadedModelInfo {
    pub name: String,
    pub profile: String,
    pub precision: String,
    pub size_bytes: u64,
    pub vram_estimate_bytes: u64,
    pub digest: String,
    pub loaded_at_unix: u64,
    pub last_used_unix: u64,
    pub requests_served: u64,
}

/// Backend-agnostic inference contract served by [`Server`].
pub trait InferenceEngine: Send + Sync + 'static {
    fn list_models(&self) -> Vec<ModelEntry>;
    fn get_model(&self, name: &str) -> Option<ModelEntry>;
    fn loaded_models(&self) -> Vec<LoadedModelInfo> {
        Vec::new()
    }
    fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse, EngineError>;
}
