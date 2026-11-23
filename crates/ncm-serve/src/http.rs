//! Minimal, bounded, pure-std HTTP/1.1 parser and response writer (JSON, text, SSE, NDJSON).
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpStream,
    time::Duration,
};

/// Maximum accepted HTTP request header section (64 KiB).
pub const MAX_HEADER_BYTES: usize = 64 * 1024;
/// Maximum accepted HTTP request body (4 MiB).
pub const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

#[derive(Debug)]
pub enum ParseError {
    Eof,
    BadRequest(String),
    PayloadTooLarge(String),
    Io,
}

impl From<io::Error> for ParseError {
    fn from(_: io::Error) -> Self {
        Self::Io
    }
}

/// Read one HTTP/1.1 request from a TCP stream with bounded headers and body.
pub fn read_request(stream: &TcpStream) -> Result<HttpRequest, ParseError> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let mut reader = BufReader::new(stream);
    let mut req_line = String::new();
    let n = read_bounded_line(&mut reader, &mut req_line, 8192)?;
    if n == 0 {
        return Err(ParseError::Eof);
    }
    let trimmed = req_line.trim_end_matches(['\r', '\n']);
    let mut parts = trimmed.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| ParseError::BadRequest("missing HTTP method".into()))?
        .to_ascii_uppercase();
    let raw_target = parts
        .next()
        .ok_or_else(|| ParseError::BadRequest("missing HTTP target".into()))?;
    let version = parts
        .next()
        .ok_or_else(|| ParseError::BadRequest("missing HTTP version".into()))?;
    if !version.starts_with("HTTP/1.") {
        return Err(ParseError::BadRequest(format!(
            "unsupported HTTP version `{version}`"
        )));
    }
    let path = raw_target
        .split_once('?')
        .map_or(raw_target, |(p, _)| p)
        .to_owned();

    let mut headers = BTreeMap::new();
    let mut header_bytes = n;
    loop {
        let mut line = String::new();
        let read = read_bounded_line(&mut reader, &mut line, 16 * 1024)?;
        if read == 0 {
            return Err(ParseError::BadRequest("unexpected EOF in headers".into()));
        }
        header_bytes += read;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(ParseError::PayloadTooLarge(
                "HTTP headers exceed 64 KiB limit".into(),
            ));
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        let (k, v) = line
            .split_once(':')
            .ok_or_else(|| ParseError::BadRequest("malformed HTTP header".into()))?;
        headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_owned());
    }

    let is_chunked = headers
        .get("transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));

    let body = if is_chunked {
        read_chunked_body(&mut reader)?
    } else if let Some(len_str) = headers.get("content-length") {
        let len: usize = len_str
            .parse()
            .map_err(|_| ParseError::BadRequest("invalid Content-Length".into()))?;
        if len > MAX_BODY_BYTES {
            return Err(ParseError::PayloadTooLarge(format!(
                "request body {len} bytes exceeds {MAX_BODY_BYTES} byte limit"
            )));
        }
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        buf
    } else {
        Vec::new()
    };

    Ok(HttpRequest { method, path, body })
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    buf: &mut String,
    max_len: usize,
) -> Result<usize, ParseError> {
    let mut total = 0;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(total);
        }
        if let Some(pos) = available.iter().position(|&b| b == b'\n') {
            let take = pos + 1;
            if total + take > max_len {
                return Err(ParseError::PayloadTooLarge("HTTP line too long".into()));
            }
            let s = std::str::from_utf8(&available[..take])
                .map_err(|_| ParseError::BadRequest("non-UTF-8 HTTP header".into()))?;
            buf.push_str(s);
            reader.consume(take);
            return Ok(total + take);
        }
        let take = available.len();
        if total + take > max_len {
            return Err(ParseError::PayloadTooLarge("HTTP line too long".into()));
        }
        let s = std::str::from_utf8(available)
            .map_err(|_| ParseError::BadRequest("non-UTF-8 HTTP header".into()))?;
        buf.push_str(s);
        reader.consume(take);
        total += take;
    }
}

fn read_chunked_body<R: BufRead>(reader: &mut R) -> Result<Vec<u8>, ParseError> {
    let mut out = Vec::new();
    loop {
        let mut size_line = String::new();
        let n = read_bounded_line(reader, &mut size_line, 1024)?;
        if n == 0 {
            return Err(ParseError::BadRequest(
                "unexpected EOF in chunked body".into(),
            ));
        }
        let hex = size_line
            .trim()
            .split_once(';')
            .map_or(size_line.trim(), |(h, _)| h);
        let chunk_len = usize::from_str_radix(hex, 16)
            .map_err(|_| ParseError::BadRequest("invalid chunk size".into()))?;
        if chunk_len == 0 {
            let mut trailer = String::new();
            let _ = read_bounded_line(reader, &mut trailer, 1024);
            break;
        }
        if out.len().saturating_add(chunk_len) > MAX_BODY_BYTES {
            return Err(ParseError::PayloadTooLarge(format!(
                "chunked body exceeds {MAX_BODY_BYTES} byte limit"
            )));
        }
        let start = out.len();
        out.resize(start + chunk_len, 0);
        reader.read_exact(&mut out[start..])?;
        let mut crlf = [0u8; 2];
        reader.read_exact(&mut crlf)?;
    }
    Ok(out)
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "Error",
    }
}

const CORS_HEADERS: &str = "Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

pub fn write_empty(stream: &mut TcpStream, status: u16) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {}\r\n{CORS_HEADERS}Content-Length: 0\r\nConnection: close\r\n\r\n",
        reason_phrase(status)
    );
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

pub fn write_text(stream: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {
    let bytes = body.as_bytes();
    let head = format!(
        "HTTP/1.1 {status} {}\r\n{CORS_HEADERS}Content-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason_phrase(status),
        bytes.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}

pub fn write_json(stream: &mut TcpStream, status: u16, value: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    let head = format!(
        "HTTP/1.1 {status} {}\r\n{CORS_HEADERS}Content-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason_phrase(status),
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()
}

pub fn write_sse(stream: &mut TcpStream, events: &[String]) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\n{CORS_HEADERS}Content-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes())?;
    for event in events {
        stream.write_all(b"data: ")?;
        stream.write_all(event.as_bytes())?;
        stream.write_all(b"\n\n")?;
        stream.flush()?;
    }
    Ok(())
}

pub fn write_ndjson(stream: &mut TcpStream, lines: &[Value]) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\n{CORS_HEADERS}Content-Type: application/x-ndjson; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes())?;
    for line in lines {
        let raw = serde_json::to_vec(line).unwrap_or_else(|_| b"{}".to_vec());
        stream.write_all(&raw)?;
        stream.write_all(b"\n")?;
        stream.flush()?;
    }
    Ok(())
}
