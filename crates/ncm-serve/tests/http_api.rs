//! End-to-end loopback TCP tests for OpenAI (`/v1/*`) and Ollama (`/api/*`) HTTP endpoints.
use ncm_serve::{
    EngineError, GenerateRequest, GenerateResponse, InferenceEngine, ModelEntry, ModelRegistry,
    Server,
};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};
use tempfile::tempdir;

struct MockEngine {
    models: Vec<ModelEntry>,
    calls: Arc<AtomicUsize>,
}

impl MockEngine {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let entry = ModelEntry {
            name: "xsum".to_owned(),
            aliases: vec!["xsum-inference-rope-v2-final".to_owned()],
            path: PathBuf::from("/models/xsum"),
            profile: "xsum-inference-v1".to_owned(),
            size_bytes: 512_000_000,
            modified_unix: 1_700_000_000,
            digest: "abcdef0123456789".to_owned(),
            precision: "fp16".to_owned(),
            supports_text: true,
            supports_seed: true,
            supports_cfg: true,
        };
        (
            Self {
                models: vec![entry],
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

impl InferenceEngine for MockEngine {
    fn list_models(&self) -> Vec<ModelEntry> {
        self.models.clone()
    }
    fn get_model(&self, name: &str) -> Option<ModelEntry> {
        let key = name.strip_suffix(":latest").unwrap_or(name);
        self.models
            .iter()
            .find(|m| m.name == key || m.aliases.iter().any(|a| a == key))
            .cloned()
    }
    fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse, EngineError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let Some(model) = self.get_model(&req.model) else {
            return Err(EngineError::NotFound(format!(
                "model `{}` not found",
                req.model
            )));
        };
        let seed_tag = req.seed.map(|s| format!(" seed={s}")).unwrap_or_default();
        let cfg_tag = req
            .cfg_scale
            .map(|c| format!(" cfg={c}"))
            .unwrap_or_default();
        let text = format!("Summary of: {}{seed_tag}{cfg_tag}. DONE.", req.prompt);
        Ok(GenerateResponse {
            model: model.name,
            text,
            prompt_tokens: req.prompt.split_whitespace().count().max(1),
            completion_tokens: 7,
            load_duration_ns: 1_000_000,
            eval_duration_ns: 5_000_000,
            finish_reason: "stop".to_owned(),
        })
    }
}

fn http_raw(addr: SocketAddr, raw_req: &[u8]) -> (u16, String, String) {
    let mut stream = TcpStream::connect(addr).expect("connect to test server");
    stream.write_all(raw_req).unwrap();
    stream.flush().unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).unwrap();
    let text = String::from_utf8(buf).expect("UTF-8 HTTP response");
    let (head, body) = text.split_once("\r\n\r\n").expect("HTTP header separator");
    let status: u16 = head
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    (status, head.to_owned(), body.to_owned())
}

fn http_req(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> (u16, String, String) {
    let body_bytes = body.unwrap_or("").as_bytes();
    let mut raw = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n");
    if body.is_some() {
        raw.push_str("Content-Type: application/json\r\n");
        raw.push_str(&format!("Content-Length: {}\r\n", body_bytes.len()));
    }
    raw.push_str("Connection: close\r\n\r\n");
    let mut bytes = raw.into_bytes();
    bytes.extend_from_slice(body_bytes);
    http_raw(addr, &bytes)
}

#[test]
fn registry_discovers_profiles_and_prefers_v2_aliases() {
    let dir = tempdir().unwrap();
    for (folder, profile) in [
        ("xsum-inference-rope-v2-final", "xsum-inference-v1"),
        ("lm1b-small-inference-v1", "lm1b-small-inference-v1"),
        ("lm1b-small-inference-v2", "lm1b-small-inference-v2"),
        ("my-custom-model", "local-ncm-v1"),
        ("ignored-folder", "unsupported-profile"),
    ] {
        let p = dir.path().join(folder);
        fs::create_dir_all(&p).unwrap();
        let manifest = json!({
            "schema_version": 1,
            "profile": profile,
            "files": {
                "codec.safetensors": {"bytes": 1234, "sha256": "deadbeef"}
            }
        });
        fs::write(p.join("manifest.json"), manifest.to_string()).unwrap();
    }

    let reg = ModelRegistry::discover(Some(dir.path()), &[], "fp16").unwrap();
    assert_eq!(reg.resolve("xsum").unwrap().profile, "xsum-inference-v1");
    assert_eq!(
        reg.resolve("xsum:latest").unwrap().profile,
        "xsum-inference-v1"
    );
    assert_eq!(
        reg.resolve("xsum-inference-rope-v2-final").unwrap().name,
        "xsum"
    );
    // Self-contained v2 wins the short `lm1b-small` name while v1 stays reachable by folder name.
    assert_eq!(
        reg.resolve("lm1b-small").unwrap().profile,
        "lm1b-small-inference-v2"
    );
    assert_eq!(
        reg.resolve("lm1b-small-inference-v1").unwrap().profile,
        "lm1b-small-inference-v1"
    );
    assert_eq!(
        reg.resolve("my-custom-model").unwrap().profile,
        "local-ncm-v1"
    );
    assert!(reg.resolve("ignored-folder").is_none());
}

#[test]
fn openai_and_ollama_endpoints_work_end_to_end() {
    let (engine, calls) = MockEngine::new();
    let server = Server::bind("127.0.0.1:0", engine).unwrap();
    let handle = server.handle().unwrap();
    let addr = handle.addr();
    let worker = thread::spawn(move || server.serve().unwrap());

    // Health, CORS preflight, and version
    let (st, head, body) = http_req(addr, "GET", "/", None);
    assert_eq!(st, 200);
    assert_eq!(body, "NCM server is running");
    assert!(head.contains("Access-Control-Allow-Origin: *"));

    let (st, _, _) = http_req(addr, "OPTIONS", "/v1/chat/completions", None);
    assert_eq!(st, 204);

    let (st, _, body) = http_req(addr, "GET", "/health", None);
    assert_eq!(st, 200);
    let health: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["models"], 1);

    let (st, _, body) = http_req(addr, "GET", "/api/version", None);
    assert_eq!(st, 200);
    assert!(serde_json::from_str::<Value>(&body).unwrap()["version"].is_string());

    // OpenAI /v1/models and /v1/models/<id>
    let (st, _, body) = http_req(addr, "GET", "/v1/models", None);
    assert_eq!(st, 200);
    let models: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(models["object"], "list");
    assert_eq!(models["data"][0]["id"], "xsum");

    let (st, _, body) = http_req(addr, "GET", "/v1/models/xsum", None);
    assert_eq!(st, 200);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["id"], "xsum");

    // OpenAI /v1/completions (non-streaming + stop sequence)
    let payload = json!({
        "model": "xsum",
        "prompt": "Hello world",
        "seed": 42,
        "stop": ["DONE."],
        "stream": false
    })
    .to_string();
    let (st, _, body) = http_req(addr, "POST", "/v1/completions", Some(&payload));
    assert_eq!(st, 200);
    let cmpl: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(cmpl["object"], "text_completion");
    assert_eq!(
        cmpl["choices"][0]["text"],
        "Summary of: Hello world seed=42. "
    );
    assert_eq!(cmpl["usage"]["completion_tokens"], 7);

    // OpenAI /v1/chat/completions (streaming SSE + content parts)
    let payload = json!({
        "model": "xsum",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Brief news"}]}],
        "stream": true
    })
    .to_string();
    let (st, head, body) = http_req(addr, "POST", "/v1/chat/completions", Some(&payload));
    assert_eq!(st, 200);
    assert!(head.contains("text/event-stream"));
    let mut reassembled = String::new();
    let mut saw_done = false;
    for line in body.lines().filter(|l| l.starts_with("data: ")) {
        let data = &line["data: ".len()..];
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        let chunk: Value = serde_json::from_str(data).unwrap();
        assert_eq!(chunk["object"], "chat.completion.chunk");
        if let Some(c) = chunk["choices"][0]["delta"]["content"].as_str() {
            reassembled.push_str(c);
        }
    }
    assert!(saw_done);
    assert_eq!(reassembled, "Summary of: Brief news. DONE.");

    // Ollama /api/tags and /api/show
    let (st, _, body) = http_req(addr, "GET", "/api/tags", None);
    assert_eq!(st, 200);
    let tags: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(tags["models"][0]["name"], "xsum");
    assert!(
        tags["models"][0]["modified_at"]
            .as_str()
            .unwrap()
            .ends_with('Z')
    );

    let (st, _, body) = http_req(
        addr,
        "POST",
        "/api/show",
        Some(&json!({"model": "xsum:latest"}).to_string()),
    );
    assert_eq!(st, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["model_info"]["ncm.profile"],
        "xsum-inference-v1"
    );

    // Ollama /api/generate (streaming default NDJSON)
    let payload = json!({"model": "xsum:latest", "prompt": "Article"}).to_string();
    let (st, head, body) = http_req(addr, "POST", "/api/generate", Some(&payload));
    assert_eq!(st, 200);
    assert!(head.contains("application/x-ndjson"));
    let mut ollama_text = String::new();
    let mut final_done = false;
    for line in body.lines().filter(|l| !l.is_empty()) {
        let obj: Value = serde_json::from_str(line).unwrap();
        ollama_text.push_str(obj["response"].as_str().unwrap());
        if obj["done"] == true {
            final_done = true;
            assert!(obj["eval_duration"].as_u64().unwrap() > 0);
        }
    }
    assert!(final_done);
    assert_eq!(ollama_text, "Summary of: Article. DONE.");

    // Ollama /api/chat (non-streaming)
    let payload = json!({
        "model": "xsum",
        "messages": [{"role": "user", "content": "Chat prompt"}],
        "stream": false,
        "options": {"seed": 7, "cfg_scale": 1.0}
    })
    .to_string();
    let (st, _, body) = http_req(addr, "POST", "/api/chat", Some(&payload));
    assert_eq!(st, 200);
    let chat: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(chat["done"], true);
    assert_eq!(
        chat["message"]["content"],
        "Summary of: Chat prompt seed=7 cfg=1. DONE."
    );

    // Error handling: 404 unknown model, 400 malformed JSON, 413 oversized Content-Length
    let (st, _, body) = http_req(
        addr,
        "POST",
        "/v1/completions",
        Some(&json!({"model": "nonexistent", "prompt": "hi"}).to_string()),
    );
    assert_eq!(st, 404);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
        "model_not_found"
    );

    let (st, _, _) = http_req(addr, "POST", "/v1/completions", Some("{not valid json"));
    assert_eq!(st, 400);

    let big_req =
        b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 99999999\r\n\r\n";
    let (st, _, _) = http_raw(addr, big_req);
    assert_eq!(st, 413);

    // Chunked transfer encoding
    let chunk_body = r#"{"model":"xsum","prompt":"Chunked","stream":false}"#;
    let chunk_req = format!(
        "POST /api/generate HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{chunk_body}\r\n0\r\n\r\n",
        chunk_body.len()
    );
    let (st, _, body) = http_raw(addr, chunk_req.as_bytes());
    assert_eq!(st, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["response"],
        "Summary of: Chunked. DONE."
    );

    // Concurrent clients
    let threads: Vec<_> = (0..6)
        .map(|i| {
            thread::spawn(move || {
                let p = json!({
                    "model": "xsum",
                    "messages": [{"role": "user", "content": format!("req-{i}")}],
                    "stream": false
                })
                .to_string();
                let (st, _, body) = http_req(addr, "POST", "/v1/chat/completions", Some(&p));
                assert_eq!(st, 200);
                let v: Value = serde_json::from_str(&body).unwrap();
                assert!(
                    v["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap()
                        .contains(&format!("req-{i}"))
                );
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert!(calls.load(Ordering::SeqCst) >= 11);

    handle.shutdown();
    worker.join().unwrap();
}
