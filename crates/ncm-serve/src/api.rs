//! OpenAI (`/v1/*`) and Ollama (`/api/*`) HTTP endpoint routing and JSON/SSE/NDJSON framing.
use crate::{
    EngineError, GenerateRequest, GenerateResponse, InferenceEngine, ModelEntry,
    http::{HttpRequest, write_empty, write_json, write_ndjson, write_sse, write_text},
};
use serde_json::{Value, json};
use std::{
    io,
    net::TcpStream,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id(prefix: &str) -> String {
    let seq = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{seq}", unix_secs())
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Format Unix timestamp seconds as RFC 3339 UTC (`YYYY-MM-DDTHH:MM:SSZ`) without extra crates.
pub(crate) fn format_rfc3339_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3_600;
    let min = (rem % 3_600) / 60;
    let sec = rem % 60;
    // Howard Hinnant's civil_from_days algorithm (valid for all proleptic Gregorian dates).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + i64::from(m <= 2);
    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Split generated text into natural whitespace-preserving chunks for SSE/NDJSON streaming.
fn stream_chunks(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            chunks.push(text[start..i].to_owned());
            start = i;
        } else {
            i += 1;
        }
    }
    if start < text.len() {
        chunks.push(text[start..].to_owned());
    }
    chunks
}

fn openai_error(status: u16, msg: &str, err_type: &str, code: &str) -> (u16, Value) {
    (
        status,
        json!({
            "error": {
                "message": msg,
                "type": err_type,
                "param": Value::Null,
                "code": code,
            }
        }),
    )
}

fn ollama_error(status: u16, msg: &str) -> (u16, Value) {
    (status, json!({ "error": msg }))
}

fn map_engine_error(err: EngineError, is_openai: bool) -> (u16, Value) {
    match err {
        EngineError::NotFound(m) => {
            if is_openai {
                openai_error(404, &m, "invalid_request_error", "model_not_found")
            } else {
                ollama_error(404, &m)
            }
        }
        EngineError::BadRequest(m) => {
            if is_openai {
                openai_error(400, &m, "invalid_request_error", "invalid_request")
            } else {
                ollama_error(400, &m)
            }
        }
        EngineError::Internal(m) => {
            if is_openai {
                openai_error(500, &m, "server_error", "internal_error")
            } else {
                ollama_error(500, &m)
            }
        }
    }
}

fn openai_model_object(m: &ModelEntry) -> Value {
    json!({
        "id": m.name,
        "object": "model",
        "created": m.modified_unix,
        "owned_by": "ncm-rs",
        "ncm": {
            "profile": m.profile,
            "precision": m.precision,
            "size_bytes": m.size_bytes,
            "supports_text": m.supports_text,
            "supports_seed": m.supports_seed,
            "supports_cfg": m.supports_cfg,
        }
    })
}

fn ollama_model_object(m: &ModelEntry) -> Value {
    json!({
        "name": m.name,
        "model": m.name,
        "modified_at": format_rfc3339_utc(m.modified_unix),
        "size": m.size_bytes,
        "digest": m.digest,
        "details": {
            "parent_model": "",
            "format": "safetensors",
            "family": "ncm",
            "families": ["ncm", m.profile],
            "parameter_size": "",
            "quantization_level": m.precision.to_ascii_uppercase(),
        }
    })
}

fn parse_json_body(body: &[u8]) -> Result<Value, String> {
    if body.is_empty() {
        return Err("empty JSON request body".into());
    }
    serde_json::from_slice(body).map_err(|e| format!("invalid JSON body: {e}"))
}

fn parse_stop(value: Option<&Value>) -> Result<Vec<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                let s = item
                    .as_str()
                    .ok_or_else(|| "stop array entries must be strings".to_owned())?;
                out.push(s.to_owned());
            }
            Ok(out)
        }
        _ => Err("`stop` must be a string or array of strings".into()),
    }
}

fn extract_message_content(val: &Value) -> Result<String, String> {
    match val {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut joined = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !joined.is_empty() {
                        joined.push('\n');
                    }
                    joined.push_str(text);
                }
            }
            Ok(joined)
        }
        Value::Null => Ok(String::new()),
        _ => Err("message `content` must be a string or content-part array".into()),
    }
}

/// Convert OpenAI / Ollama `messages` array into a single prompt string.
/// For single-message requests (the common summarization / translation / completion case),
/// passes the message content verbatim so non-chat NCM models receive clean text.
fn messages_to_prompt(messages: &Value) -> Result<String, String> {
    let arr = messages
        .as_array()
        .ok_or_else(|| "`messages` must be an array".to_owned())?;
    if arr.is_empty() {
        return Err("`messages` must contain at least one message".into());
    }
    let mut pairs = Vec::with_capacity(arr.len());
    for msg in arr {
        let role = msg
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
            .to_owned();
        let content = extract_message_content(msg.get("content").unwrap_or(&Value::Null))?;
        pairs.push((role, content));
    }
    if pairs.len() == 1 {
        return Ok(pairs.remove(0).1);
    }
    // If all messages are system/user with only one user message and empty system, pass verbatim.
    let non_empty: Vec<&(String, String)> =
        pairs.iter().filter(|(_, c)| !c.trim().is_empty()).collect();
    if non_empty.len() == 1 {
        return Ok(non_empty[0].1.clone());
    }
    let mut out = String::new();
    for (role, content) in pairs {
        if content.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        if role == "system" {
            out.push_str(&content);
        } else {
            out.push_str(&format!("{role}: {content}"));
        }
    }
    Ok(out)
}

/// Apply optional stop sequences to a generated response.
pub(crate) fn apply_stop_sequences(
    mut resp: GenerateResponse,
    stop: &[String],
) -> GenerateResponse {
    if stop.is_empty() {
        return resp;
    }
    let mut earliest: Option<usize> = None;
    for seq in stop {
        if seq.is_empty() {
            continue;
        }
        if let Some(pos) = resp.text.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |cur| cur.min(pos)));
        }
    }
    if let Some(pos) = earliest {
        resp.text.truncate(pos);
        resp.finish_reason = "stop".to_owned();
    }
    resp
}

pub fn dispatch<E: InferenceEngine>(
    engine: &E,
    req: &HttpRequest,
    stream: &mut TcpStream,
) -> io::Result<()> {
    if req.method == "OPTIONS" {
        return write_empty(stream, 204);
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("HEAD", "/") => write_text(stream, 200, "NCM server is running"),
        ("GET", "/health") => {
            let count = engine.list_models().len();
            write_json(stream, 200, &json!({ "status": "ok", "models": count }))
        }
        ("GET", "/api/version") => write_json(
            stream,
            200,
            &json!({ "version": env!("CARGO_PKG_VERSION") }),
        ),
        ("GET", "/v1/models") => {
            let data: Vec<Value> = engine
                .list_models()
                .iter()
                .map(openai_model_object)
                .collect();
            write_json(stream, 200, &json!({ "object": "list", "data": data }))
        }
        ("GET", path) if path.starts_with("/v1/models/") => {
            let id = &path["/v1/models/".len()..];
            match engine.get_model(id) {
                Some(m) => write_json(stream, 200, &openai_model_object(&m)),
                None => {
                    let (st, val) = openai_error(
                        404,
                        &format!("model `{id}` not found"),
                        "invalid_request_error",
                        "model_not_found",
                    );
                    write_json(stream, st, &val)
                }
            }
        }
        ("GET", "/api/tags") => {
            let models: Vec<Value> = engine
                .list_models()
                .iter()
                .map(ollama_model_object)
                .collect();
            write_json(stream, 200, &json!({ "models": models }))
        }
        ("GET", "/api/ps") => {
            let models: Vec<Value> = engine
                .loaded_models()
                .iter()
                .map(|m| {
                    json!({
                        "name": m.name,
                        "model": m.name,
                        "size": m.size_bytes,
                        "size_vram": m.vram_estimate_bytes,
                        "digest": m.digest,
                        "details": {
                            "parent_model": "",
                            "format": "safetensors",
                            "family": "ncm",
                            "families": ["ncm", m.profile],
                            "parameter_size": "",
                            "quantization_level": m.precision.to_ascii_uppercase(),
                        },
                        "loaded_at": format_rfc3339_utc(m.loaded_at_unix),
                        "last_used_at": format_rfc3339_utc(m.last_used_unix),
                        "requests_served": m.requests_served,
                    })
                })
                .collect();
            write_json(stream, 200, &json!({ "models": models }))
        }
        ("POST", "/api/show") => handle_ollama_show(engine, &req.body, stream),
        ("POST", "/v1/completions") => handle_openai_completions(engine, &req.body, stream),
        ("POST", "/v1/chat/completions") => handle_openai_chat(engine, &req.body, stream),
        ("POST", "/api/generate") => handle_ollama_generate(engine, &req.body, stream),
        ("POST", "/api/chat") => handle_ollama_chat(engine, &req.body, stream),
        (_, path) if path.starts_with("/v1/") => {
            let (st, val) = openai_error(
                404,
                &format!("unknown endpoint `{} {path}`", req.method),
                "invalid_request_error",
                "not_found",
            );
            write_json(stream, st, &val)
        }
        _ => {
            let (st, val) = ollama_error(
                404,
                &format!("unknown endpoint `{} {}`", req.method, req.path),
            );
            write_json(stream, st, &val)
        }
    }
}

fn handle_ollama_show<E: InferenceEngine>(
    engine: &E,
    body: &[u8],
    stream: &mut TcpStream,
) -> io::Result<()> {
    let payload = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => {
            let (st, val) = ollama_error(400, &msg);
            return write_json(stream, st, &val);
        }
    };
    let Some(name) = payload
        .get("model")
        .or_else(|| payload.get("name"))
        .and_then(Value::as_str)
    else {
        let (st, val) = ollama_error(400, "missing `model` or `name` field");
        return write_json(stream, st, &val);
    };
    match engine.get_model(name) {
        Some(m) => write_json(
            stream,
            200,
            &json!({
                "modelfile": format!("FROM {}", m.path.display()),
                "parameters": format!("precision {}", m.precision),
                "template": "{{ .Prompt }}",
                "details": {
                    "parent_model": "",
                    "format": "safetensors",
                    "family": "ncm",
                    "families": ["ncm", m.profile],
                    "parameter_size": "",
                    "quantization_level": m.precision.to_ascii_uppercase(),
                },
                "model_info": {
                    "ncm.profile": m.profile,
                    "ncm.precision": m.precision,
                    "ncm.size_bytes": m.size_bytes,
                    "ncm.supports_text": m.supports_text,
                    "ncm.supports_seed": m.supports_seed,
                    "ncm.supports_cfg": m.supports_cfg,
                },
                "modified_at": format_rfc3339_utc(m.modified_unix),
            }),
        ),
        None => {
            let (st, val) = ollama_error(404, &format!("model `{name}` not found"));
            write_json(stream, st, &val)
        }
    }
}

fn handle_openai_completions<E: InferenceEngine>(
    engine: &E,
    body: &[u8],
    stream: &mut TcpStream,
) -> io::Result<()> {
    let payload = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => {
            let (st, val) = openai_error(400, &msg, "invalid_request_error", "invalid_json");
            return write_json(stream, st, &val);
        }
    };
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        let (st, val) = openai_error(
            400,
            "missing `model` field",
            "invalid_request_error",
            "missing_model",
        );
        return write_json(stream, st, &val);
    };
    let prompt = match payload.get("prompt") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) if arr.len() == 1 => match arr[0].as_str() {
            Some(s) => s.to_owned(),
            None => {
                let (st, val) = openai_error(
                    400,
                    "`prompt` array entry must be a string",
                    "invalid_request_error",
                    "invalid_prompt",
                );
                return write_json(stream, st, &val);
            }
        },
        _ => {
            let (st, val) = openai_error(
                400,
                "`prompt` must be a string or single-element string array",
                "invalid_request_error",
                "invalid_prompt",
            );
            return write_json(stream, st, &val);
        }
    };
    let stop = match parse_stop(payload.get("stop")) {
        Ok(s) => s,
        Err(msg) => {
            let (st, val) = openai_error(400, &msg, "invalid_request_error", "invalid_stop");
            return write_json(stream, st, &val);
        }
    };
    let is_stream = payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let seed = payload.get("seed").and_then(Value::as_u64);
    let max_tokens = payload
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as usize);
    let cfg_scale = payload
        .get("cfg_scale")
        .or_else(|| payload.get("ncm_cfg"))
        .and_then(Value::as_f64);

    let gen_req = GenerateRequest {
        model: model.to_owned(),
        prompt,
        seed,
        cfg_scale,
        max_tokens,
        stop: stop.clone(),
    };
    let resp = match engine.generate(gen_req) {
        Ok(r) => apply_stop_sequences(r, &stop),
        Err(err) => {
            let (st, val) = map_engine_error(err, true);
            return write_json(stream, st, &val);
        }
    };

    let id = next_id("cmpl");
    let created = unix_secs();
    if is_stream {
        let mut events = Vec::new();
        for chunk in stream_chunks(&resp.text) {
            events.push(
                json!({
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": resp.model,
                    "choices": [{
                        "text": chunk,
                        "index": 0,
                        "logprobs": Value::Null,
                        "finish_reason": Value::Null,
                    }]
                })
                .to_string(),
            );
        }
        events.push(
            json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": resp.model,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": Value::Null,
                    "finish_reason": resp.finish_reason,
                }]
            })
            .to_string(),
        );
        events.push("[DONE]".to_owned());
        write_sse(stream, &events)
    } else {
        write_json(
            stream,
            200,
            &json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": resp.model,
                "choices": [{
                    "text": resp.text,
                    "index": 0,
                    "logprobs": Value::Null,
                    "finish_reason": resp.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "total_tokens": resp.prompt_tokens + resp.completion_tokens,
                }
            }),
        )
    }
}

fn handle_openai_chat<E: InferenceEngine>(
    engine: &E,
    body: &[u8],
    stream: &mut TcpStream,
) -> io::Result<()> {
    let payload = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => {
            let (st, val) = openai_error(400, &msg, "invalid_request_error", "invalid_json");
            return write_json(stream, st, &val);
        }
    };
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        let (st, val) = openai_error(
            400,
            "missing `model` field",
            "invalid_request_error",
            "missing_model",
        );
        return write_json(stream, st, &val);
    };
    let prompt = match payload.get("messages").map(messages_to_prompt) {
        Some(Ok(p)) => p,
        Some(Err(msg)) => {
            let (st, val) = openai_error(400, &msg, "invalid_request_error", "invalid_messages");
            return write_json(stream, st, &val);
        }
        None => {
            let (st, val) = openai_error(
                400,
                "missing `messages` field",
                "invalid_request_error",
                "missing_messages",
            );
            return write_json(stream, st, &val);
        }
    };
    let stop = match parse_stop(payload.get("stop")) {
        Ok(s) => s,
        Err(msg) => {
            let (st, val) = openai_error(400, &msg, "invalid_request_error", "invalid_stop");
            return write_json(stream, st, &val);
        }
    };
    let is_stream = payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let seed = payload.get("seed").and_then(Value::as_u64);
    let max_tokens = payload
        .get("max_tokens")
        .or_else(|| payload.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .map(|n| n as usize);
    let cfg_scale = payload
        .get("cfg_scale")
        .or_else(|| payload.get("ncm_cfg"))
        .and_then(Value::as_f64);

    let gen_req = GenerateRequest {
        model: model.to_owned(),
        prompt,
        seed,
        cfg_scale,
        max_tokens,
        stop: stop.clone(),
    };
    let resp = match engine.generate(gen_req) {
        Ok(r) => apply_stop_sequences(r, &stop),
        Err(err) => {
            let (st, val) = map_engine_error(err, true);
            return write_json(stream, st, &val);
        }
    };

    let id = next_id("chatcmpl");
    let created = unix_secs();
    if is_stream {
        let mut events = Vec::new();
        events.push(
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": resp.model,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant", "content": "" },
                    "finish_reason": Value::Null,
                }]
            })
            .to_string(),
        );
        for chunk in stream_chunks(&resp.text) {
            events.push(
                json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resp.model,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": chunk },
                        "finish_reason": Value::Null,
                    }]
                })
                .to_string(),
            );
        }
        events.push(
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": resp.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": resp.finish_reason,
                }]
            })
            .to_string(),
        );
        events.push("[DONE]".to_owned());
        write_sse(stream, &events)
    } else {
        write_json(
            stream,
            200,
            &json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": resp.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": resp.text,
                    },
                    "finish_reason": resp.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "total_tokens": resp.prompt_tokens + resp.completion_tokens,
                }
            }),
        )
    }
}

struct OllamaOptions {
    seed: Option<u64>,
    max_tokens: Option<usize>,
    cfg_scale: Option<f64>,
    stop: Vec<String>,
}

fn parse_ollama_options(payload: &Value) -> Result<OllamaOptions, String> {
    let opts = payload.get("options");
    let seed = opts
        .and_then(|o| o.get("seed"))
        .or_else(|| payload.get("seed"))
        .and_then(Value::as_u64);
    let max_tokens = opts
        .and_then(|o| o.get("num_predict"))
        .and_then(Value::as_i64)
        .filter(|&n| n > 0)
        .map(|n| n as usize);
    let cfg_scale = opts
        .and_then(|o| o.get("cfg_scale"))
        .or_else(|| payload.get("cfg_scale"))
        .and_then(Value::as_f64);
    let stop = parse_stop(
        opts.and_then(|o| o.get("stop"))
            .or_else(|| payload.get("stop")),
    )?;
    Ok(OllamaOptions {
        seed,
        max_tokens,
        cfg_scale,
        stop,
    })
}

fn handle_ollama_generate<E: InferenceEngine>(
    engine: &E,
    body: &[u8],
    stream: &mut TcpStream,
) -> io::Result<()> {
    let payload = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => {
            let (st, val) = ollama_error(400, &msg);
            return write_json(stream, st, &val);
        }
    };
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        let (st, val) = ollama_error(400, "missing `model` field");
        return write_json(stream, st, &val);
    };
    let prompt = payload
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    // Ollama's /api/generate streams NDJSON by default unless `"stream": false` is passed.
    let is_stream = payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let opts = match parse_ollama_options(&payload) {
        Ok(v) => v,
        Err(msg) => {
            let (st, val) = ollama_error(400, &msg);
            return write_json(stream, st, &val);
        }
    };

    let gen_req = GenerateRequest {
        model: model.to_owned(),
        prompt,
        seed: opts.seed,
        cfg_scale: opts.cfg_scale,
        max_tokens: opts.max_tokens,
        stop: opts.stop.clone(),
    };
    let resp = match engine.generate(gen_req) {
        Ok(r) => apply_stop_sequences(r, &opts.stop),
        Err(err) => {
            let (st, val) = map_engine_error(err, false);
            return write_json(stream, st, &val);
        }
    };

    let created_at = format_rfc3339_utc(unix_secs());
    let total_duration = resp.load_duration_ns.saturating_add(resp.eval_duration_ns);
    if is_stream {
        let mut lines = Vec::new();
        for chunk in stream_chunks(&resp.text) {
            lines.push(json!({
                "model": resp.model,
                "created_at": created_at,
                "response": chunk,
                "done": false,
            }));
        }
        lines.push(json!({
            "model": resp.model,
            "created_at": created_at,
            "response": "",
            "done": true,
            "done_reason": resp.finish_reason,
            "total_duration": total_duration,
            "load_duration": resp.load_duration_ns,
            "prompt_eval_count": resp.prompt_tokens,
            "eval_count": resp.completion_tokens,
            "eval_duration": resp.eval_duration_ns,
        }));
        write_ndjson(stream, &lines)
    } else {
        write_json(
            stream,
            200,
            &json!({
                "model": resp.model,
                "created_at": created_at,
                "response": resp.text,
                "done": true,
                "done_reason": resp.finish_reason,
                "total_duration": total_duration,
                "load_duration": resp.load_duration_ns,
                "prompt_eval_count": resp.prompt_tokens,
                "eval_count": resp.completion_tokens,
                "eval_duration": resp.eval_duration_ns,
            }),
        )
    }
}

fn handle_ollama_chat<E: InferenceEngine>(
    engine: &E,
    body: &[u8],
    stream: &mut TcpStream,
) -> io::Result<()> {
    let payload = match parse_json_body(body) {
        Ok(v) => v,
        Err(msg) => {
            let (st, val) = ollama_error(400, &msg);
            return write_json(stream, st, &val);
        }
    };
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
        let (st, val) = ollama_error(400, "missing `model` field");
        return write_json(stream, st, &val);
    };
    let prompt = match payload.get("messages").map(messages_to_prompt) {
        Some(Ok(p)) => p,
        Some(Err(msg)) => {
            let (st, val) = ollama_error(400, &msg);
            return write_json(stream, st, &val);
        }
        None => {
            let (st, val) = ollama_error(400, "missing `messages` field");
            return write_json(stream, st, &val);
        }
    };
    let is_stream = payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let opts = match parse_ollama_options(&payload) {
        Ok(v) => v,
        Err(msg) => {
            let (st, val) = ollama_error(400, &msg);
            return write_json(stream, st, &val);
        }
    };

    let gen_req = GenerateRequest {
        model: model.to_owned(),
        prompt,
        seed: opts.seed,
        cfg_scale: opts.cfg_scale,
        max_tokens: opts.max_tokens,
        stop: opts.stop.clone(),
    };
    let resp = match engine.generate(gen_req) {
        Ok(r) => apply_stop_sequences(r, &opts.stop),
        Err(err) => {
            let (st, val) = map_engine_error(err, false);
            return write_json(stream, st, &val);
        }
    };

    let created_at = format_rfc3339_utc(unix_secs());
    let total_duration = resp.load_duration_ns.saturating_add(resp.eval_duration_ns);
    if is_stream {
        let mut lines = Vec::new();
        for chunk in stream_chunks(&resp.text) {
            lines.push(json!({
                "model": resp.model,
                "created_at": created_at,
                "message": { "role": "assistant", "content": chunk },
                "done": false,
            }));
        }
        lines.push(json!({
            "model": resp.model,
            "created_at": created_at,
            "message": { "role": "assistant", "content": "" },
            "done": true,
            "done_reason": resp.finish_reason,
            "total_duration": total_duration,
            "load_duration": resp.load_duration_ns,
            "prompt_eval_count": resp.prompt_tokens,
            "eval_count": resp.completion_tokens,
            "eval_duration": resp.eval_duration_ns,
        }));
        write_ndjson(stream, &lines)
    } else {
        write_json(
            stream,
            200,
            &json!({
                "model": resp.model,
                "created_at": created_at,
                "message": {
                    "role": "assistant",
                    "content": resp.text,
                },
                "done": true,
                "done_reason": resp.finish_reason,
                "total_duration": total_duration,
                "load_duration": resp.load_duration_ns,
                "prompt_eval_count": resp.prompt_tokens,
                "eval_count": resp.completion_tokens,
                "eval_duration": resp.eval_duration_ns,
            }),
        )
    }
}
