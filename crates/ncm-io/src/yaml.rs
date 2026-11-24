//! Bounded, zero-dependency YAML-to-JSON parser for pinned upstream `config.yaml` files.
//!
//! Supports the canonical PyYAML block subset used by upstream NCM checkpoints:
//! indented mappings, block sequences (`- item`), inline empty collections (`{}`, `[]`),
//! inline flow lists (`[1, 2, 3]`), comments (`# ...`), single/double quoted strings,
//! booleans (`true`/`false`), `null`/`~`, integers, and finite floats.
//! Rejects anchors, aliases, tags, multi-document streams, and tabs for indentation.

use crate::{Error, Result};
use serde_json::{Map, Number, Value};

const MAX_YAML_BYTES: usize = 256 * 1024;
const MAX_DEPTH: usize = 32;

struct Line<'a> {
    indent: usize,
    text: &'a str,
    line_no: usize,
}

fn err(line_no: usize, msg: &str) -> Error {
    Error::Invalid(format!("YAML line {line_no}: {msg}"))
}

/// Parse a bounded YAML document into a `serde_json::Value`.
pub fn parse_yaml_to_json(input: &str) -> Result<Value> {
    if input.len() > MAX_YAML_BYTES {
        return Err(Error::Invalid("YAML input exceeds 256 KiB bound".into()));
    }
    let mut lines = Vec::new();
    for (idx, raw) in input.lines().enumerate() {
        let line_no = idx + 1;
        let stripped = strip_comment(raw);
        let trimmed = stripped.trim_end();
        if trimmed.trim().is_empty() {
            continue;
        }
        let indent = trimmed.bytes().take_while(|&b| b == b' ').count();
        if trimmed[indent..].starts_with('\t') {
            return Err(err(line_no, "tab indentation is not supported"));
        }
        let text = &trimmed[indent..];
        if text == "---" || text == "..." {
            continue;
        }
        lines.push(Line {
            indent,
            text,
            line_no,
        });
    }
    if lines.is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    let mut cursor = 0usize;
    let val = parse_block(&lines, &mut cursor, lines[0].indent, 0)?;
    if cursor != lines.len() {
        return Err(err(lines[cursor].line_no, "unexpected trailing YAML block"));
    }
    Ok(val)
}

/// Strip `# ...` comments outside single or double quotes.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_double {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_double = false;
            }
        } else if in_single {
            if b == b'\'' {
                in_single = false;
            }
        } else if b == b'"' {
            in_double = true;
        } else if b == b'\'' {
            in_single = true;
        } else if b == b'#' && (i == 0 || bytes[i - 1].is_ascii_whitespace()) {
            return &line[..i];
        }
    }
    line
}

fn parse_block(
    lines: &[Line<'_>],
    cursor: &mut usize,
    indent: usize,
    depth: usize,
) -> Result<Value> {
    if depth > MAX_DEPTH {
        return Err(err(lines[*cursor].line_no, "YAML nesting depth exceeded"));
    }
    if lines[*cursor].text == "-" || lines[*cursor].text.starts_with("- ") {
        parse_sequence(lines, cursor, indent, depth)
    } else {
        parse_mapping(lines, cursor, indent, depth)
    }
}

fn parse_sequence(
    lines: &[Line<'_>],
    cursor: &mut usize,
    indent: usize,
    depth: usize,
) -> Result<Value> {
    let mut items = Vec::new();
    while *cursor < lines.len() {
        let line = &lines[*cursor];
        if line.indent < indent {
            break;
        }
        if line.indent != indent {
            return Err(err(line.line_no, "unexpected sequence indentation"));
        }
        let rest = if line.text == "-" {
            ""
        } else if let Some(r) = line.text.strip_prefix("- ") {
            r.trim()
        } else {
            break;
        };
        let line_no = line.line_no;
        *cursor += 1;
        if rest.is_empty() {
            if *cursor < lines.len() && lines[*cursor].indent > indent {
                let child_indent = lines[*cursor].indent;
                items.push(parse_block(lines, cursor, child_indent, depth + 1)?);
            } else {
                items.push(Value::Null);
            }
        } else if let Some((k, v_str)) = split_key_value(rest) {
            // Inline mapping starting on the `- key: val` line.
            let mut map = Map::new();
            let key = parse_key(k, line_no)?;
            let val = if v_str.is_empty() {
                if *cursor < lines.len() && lines[*cursor].indent > indent + 2 {
                    let ci = lines[*cursor].indent;
                    parse_block(lines, cursor, ci, depth + 1)?
                } else {
                    Value::Null
                }
            } else {
                parse_inline(v_str, line_no)?
            };
            map.insert(key, val);
            while *cursor < lines.len() && lines[*cursor].indent > indent {
                let ci = lines[*cursor].indent;
                if let Value::Object(extra) = parse_mapping(lines, cursor, ci, depth + 1)? {
                    for (ek, ev) in extra {
                        map.insert(ek, ev);
                    }
                } else {
                    return Err(err(line_no, "expected mapping continuation in sequence"));
                }
            }
            items.push(Value::Object(map));
        } else {
            items.push(parse_inline(rest, line_no)?);
        }
    }
    Ok(Value::Array(items))
}

fn parse_mapping(
    lines: &[Line<'_>],
    cursor: &mut usize,
    indent: usize,
    depth: usize,
) -> Result<Value> {
    let mut map = Map::new();
    while *cursor < lines.len() {
        let line = &lines[*cursor];
        if line.indent < indent {
            break;
        }
        if line.indent != indent {
            return Err(err(line.line_no, "unexpected mapping indentation"));
        }
        if line.text == "-" || line.text.starts_with("- ") {
            break;
        }
        let line_no = line.line_no;
        let (k_raw, v_raw) = split_key_value(line.text)
            .ok_or_else(|| err(line_no, "expected `key: value` mapping entry"))?;
        let key = parse_key(k_raw, line_no)?;
        *cursor += 1;
        let val = if v_raw.is_empty() {
            if *cursor < lines.len() {
                let next = &lines[*cursor];
                // PyYAML often writes block sequences at the SAME indent as the parent key:
                //   vocab_size:
                //   - 4096
                //   - 4096
                if next.indent > indent {
                    let ci = next.indent;
                    parse_block(lines, cursor, ci, depth + 1)?
                } else if next.indent == indent && (next.text == "-" || next.text.starts_with("- "))
                {
                    parse_sequence(lines, cursor, indent, depth + 1)?
                } else {
                    Value::Null
                }
            } else {
                Value::Null
            }
        } else {
            parse_inline(v_raw, line_no)?
        };
        map.insert(key, val);
    }
    Ok(Value::Object(map))
}

/// Split `key: value` at the first unquoted `:` followed by end-of-line or whitespace.
fn split_key_value(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_double {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_double = false;
            }
        } else if in_single {
            if b == b'\'' {
                in_single = false;
            }
        } else if b == b'"' {
            in_double = true;
        } else if b == b'\'' {
            in_single = true;
        } else if b == b':' && (i + 1 == bytes.len() || bytes[i + 1].is_ascii_whitespace()) {
            let k = text[..i].trim();
            let v = text[i + 1..].trim();
            return Some((k, v));
        }
    }
    None
}

fn parse_key(raw: &str, line_no: usize) -> Result<String> {
    if raw.is_empty() {
        return Err(err(line_no, "empty YAML key"));
    }
    if (raw.starts_with('\'') && raw.ends_with('\''))
        || (raw.starts_with('"') && raw.ends_with('"'))
    {
        match parse_inline(raw, line_no)? {
            Value::String(s) => Ok(s),
            _ => Err(err(line_no, "non-string quoted key")),
        }
    } else {
        Ok(raw.to_owned())
    }
}

fn parse_inline(raw: &str, line_no: usize) -> Result<Value> {
    let s = raw.trim();
    if s.is_empty() || s == "null" || s == "Null" || s == "NULL" || s == "~" {
        return Ok(Value::Null);
    }
    if s == "{}" {
        return Ok(Value::Object(Map::new()));
    }
    if s == "[]" {
        return Ok(Value::Array(Vec::new()));
    }
    if s.starts_with('[') && s.ends_with(']') {
        return parse_flow_list(&s[1..s.len() - 1], line_no);
    }
    if s.starts_with('{') && s.ends_with('}') {
        return parse_flow_map(&s[1..s.len() - 1], line_no);
    }
    if s.starts_with('&') || s.starts_with('*') || s.starts_with('!') {
        return Err(err(
            line_no,
            "YAML anchors, aliases, and tags are unsupported",
        ));
    }
    if s.starts_with('"') {
        return serde_json::from_str::<String>(s)
            .map(Value::String)
            .map_err(|e| err(line_no, &format!("invalid double-quoted string: {e}")));
    }
    if s.starts_with('\'') {
        if s.len() < 2 || !s.ends_with('\'') {
            return Err(err(line_no, "unterminated single-quoted string"));
        }
        // In YAML single-quoted strings, '' escapes a literal single quote.
        let inner = &s[1..s.len() - 1];
        return Ok(Value::String(inner.replace("''", "'")));
    }
    match s {
        "true" | "True" | "TRUE" => return Ok(Value::Bool(true)),
        "false" | "False" | "FALSE" => return Ok(Value::Bool(false)),
        _ => {}
    }
    if let Ok(i) = s.parse::<i64>() {
        return Ok(Value::Number(Number::from(i)));
    }
    if let Ok(u) = s.parse::<u64>() {
        return Ok(Value::Number(Number::from(u)));
    }
    if (s.contains('.') || s.contains('e') || s.contains('E'))
        && let Ok(f) = s.parse::<f64>()
        && f.is_finite()
        && let Some(n) = Number::from_f64(f)
    {
        return Ok(Value::Number(n));
    }
    Ok(Value::String(s.to_owned()))
}

fn split_flow_items(inner: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = inner.as_bytes();
    let mut start = 0usize;
    let mut depth = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_double {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_double = false;
            }
        } else if in_single {
            if b == b'\'' {
                in_single = false;
            }
        } else {
            match b {
                b'"' => in_double = true,
                b'\'' => in_single = true,
                b'[' | b'{' => depth += 1,
                b']' | b'}' => depth = depth.saturating_sub(1),
                b',' if depth == 0 => {
                    out.push(inner[start..i].trim());
                    start = i + 1;
                }
                _ => {}
            }
        }
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn parse_flow_list(inner: &str, line_no: usize) -> Result<Value> {
    let mut out = Vec::new();
    for item in split_flow_items(inner) {
        out.push(parse_inline(item, line_no)?);
    }
    Ok(Value::Array(out))
}

fn parse_flow_map(inner: &str, line_no: usize) -> Result<Value> {
    let mut map = Map::new();
    for item in split_flow_items(inner) {
        let (k, v) =
            split_key_value(item).ok_or_else(|| err(line_no, "invalid inline flow mapping"))?;
        map.insert(parse_key(k, line_no)?, parse_inline(v, line_no)?);
    }
    Ok(Value::Object(map))
}

#[cfg(test)]
mod tests {
    use super::parse_yaml_to_json;
    use serde_json::json;

    #[test]
    fn parses_pyyaml_config_structures() {
        let sample = r#"
task: lm1b-small
data: {}
model:
  n_layer: 12
  dropout: 0.1
  eps: 1.0e-05
  bias: true
  use_flex_attention: false
  pad_id: null
  vocab_size:
  - 8192
  - 8192
  inline_list: [1, 2, 3]
tokenizer:
  special_tokens:
    eos_token: '[CLS]'
    pad_token: '<|pad|>'
"#;
        let parsed = parse_yaml_to_json(sample).unwrap();
        assert_eq!(
            parsed,
            json!({
                "task": "lm1b-small",
                "data": {},
                "model": {
                    "n_layer": 12,
                    "dropout": 0.1,
                    "eps": 1e-5,
                    "bias": true,
                    "use_flex_attention": false,
                    "pad_id": null,
                    "vocab_size": [8192, 8192],
                    "inline_list": [1, 2, 3]
                },
                "tokenizer": {
                    "special_tokens": {
                        "eos_token": "[CLS]",
                        "pad_token": "<|pad|>"
                    }
                }
            })
        );
    }

    #[test]
    fn rejects_anchors_and_tabs() {
        assert!(parse_yaml_to_json("a: &anchor 1\n").is_err());
        assert!(parse_yaml_to_json("a:\n\tb: 1\n").is_err());
    }
}
