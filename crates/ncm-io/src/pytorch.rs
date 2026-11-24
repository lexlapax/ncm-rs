//! Bounded, zero-dependency pure-Rust reader for uncompressed PyTorch `.pt` archives
//! and protocol 2/3/4 data-only Python pickle payloads.
//!
//! Never executes arbitrary Python callables or unwhitelisted globals.
//! Supports only `STORED` (uncompressed) ZIP entries (standard for PyTorch `torch.save`)
//! and streams tensor payloads directly from file offsets to keep peak RSS low.

use crate::{Error, Result, TensorDtype, TensorRecord, sha256};
use safetensors::{Dtype as SafetensorsDtype, SafeTensors, tensor::TensorView};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
};

const MAX_PICKLE_BYTES: usize = 16 * 1024 * 1024;
const MAX_PICKLE_OPS: usize = 500_000;
const MAX_STACK: usize = 65_536;
const MAX_MEMO: usize = 65_536;

// ---------------------------------------------------------------------------
// Bounded ZIP_STORED archive reader (supports ZIP32 and ZIP64)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct ZipEntry {
    data_offset: u64,
    size: u64,
}

/// Read-only index over a PyTorch ZIP_STORED archive.
pub struct ZipStoredArchive {
    file: File,
    entries: BTreeMap<String, ZipEntry>,
}

fn read_u16_le(buf: &[u8], off: usize) -> Result<u16> {
    let slice = buf
        .get(off..off + 2)
        .ok_or_else(|| Error::Invalid("truncated ZIP u16".into()))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(buf: &[u8], off: usize) -> Result<u32> {
    let slice = buf
        .get(off..off + 4)
        .ok_or_else(|| Error::Invalid("truncated ZIP u32".into()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_le(buf: &[u8], off: usize) -> Result<u64> {
    let slice = buf
        .get(off..off + 8)
        .ok_or_else(|| Error::Invalid("truncated ZIP u64".into()))?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

impl ZipStoredArchive {
    /// Open and index a ZIP_STORED archive without loading entry payloads into memory.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < 22 {
            return Err(Error::Invalid("file too small to be a ZIP archive".into()));
        }

        // Scan tail (up to 65 KiB + 22 bytes) for End of Central Directory (PK\x05\x06).
        let tail_len = file_len.min(65536 + 22) as usize;
        let tail_start = file_len - tail_len as u64;
        file.seek(SeekFrom::Start(tail_start))?;
        let mut tail = vec![0u8; tail_len];
        file.read_exact(&mut tail)?;

        let eocd_rel = (0..=tail_len - 22)
            .rev()
            .find(|&i| tail[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
            .ok_or_else(|| Error::Invalid("missing ZIP EOCD signature".into()))?;

        let mut cd_entries = read_u16_le(&tail, eocd_rel + 10)? as u64;
        let mut cd_size = read_u32_le(&tail, eocd_rel + 12)? as u64;
        let mut cd_offset = read_u32_le(&tail, eocd_rel + 16)? as u64;

        // Check for ZIP64 EOCD locator (PK\x06\x07) immediately preceding EOCD.
        if (cd_entries == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF)
            && eocd_rel >= 20
            && tail[eocd_rel - 20..eocd_rel - 16] == [0x50, 0x4b, 0x06, 0x07]
        {
            let z64_off = read_u64_le(&tail, eocd_rel - 12)?;
            if z64_off + 56 > file_len {
                return Err(Error::Invalid("invalid ZIP64 EOCD offset".into()));
            }
            file.seek(SeekFrom::Start(z64_off))?;
            let mut z64 = [0u8; 56];
            file.read_exact(&mut z64)?;
            if z64[0..4] != [0x50, 0x4b, 0x06, 0x06] {
                return Err(Error::Invalid("invalid ZIP64 EOCD signature".into()));
            }
            cd_entries = read_u64_le(&z64, 32)?;
            cd_size = read_u64_le(&z64, 40)?;
            cd_offset = read_u64_le(&z64, 48)?;
        }

        if cd_entries > 65_536 || cd_size > 32 * 1024 * 1024 || cd_offset + cd_size > file_len {
            return Err(Error::Invalid(
                "ZIP central directory exceeds safety bounds".into(),
            ));
        }

        file.seek(SeekFrom::Start(cd_offset))?;
        let mut cd = vec![0u8; cd_size as usize];
        file.read_exact(&mut cd)?;

        let mut entries = BTreeMap::new();
        let mut pos = 0usize;
        for _ in 0..cd_entries {
            if pos + 46 > cd.len() || cd[pos..pos + 4] != [0x50, 0x4b, 0x01, 0x02] {
                return Err(Error::Invalid(
                    "corrupt ZIP central directory header".into(),
                ));
            }
            let comp_method = read_u16_le(&cd, pos + 10)?;
            if comp_method != 0 {
                return Err(Error::Invalid(format!(
                    "unsupported ZIP compression method {comp_method} (only STORED=0 is supported)"
                )));
            }
            let mut comp_size = read_u32_le(&cd, pos + 20)? as u64;
            let mut uncomp_size = read_u32_le(&cd, pos + 24)? as u64;
            let name_len = read_u16_le(&cd, pos + 28)? as usize;
            let extra_len = read_u16_le(&cd, pos + 30)? as usize;
            let comment_len = read_u16_le(&cd, pos + 32)? as usize;
            let mut local_hdr_off = read_u32_le(&cd, pos + 42)? as u64;

            let name_start = pos + 46;
            let extra_start = name_start + name_len;
            let next_pos = extra_start + extra_len + comment_len;
            if next_pos > cd.len() {
                return Err(Error::Invalid(
                    "truncated ZIP central directory entry".into(),
                ));
            }
            let name = std::str::from_utf8(&cd[name_start..extra_start])
                .map_err(|_| Error::Invalid("non-UTF-8 ZIP entry name".into()))?
                .to_owned();

            // Parse ZIP64 extra field (header ID 0x0001) if any 32-bit field is sentinel.
            if uncomp_size == 0xFFFF_FFFF
                || comp_size == 0xFFFF_FFFF
                || local_hdr_off == 0xFFFF_FFFF
            {
                let extra = &cd[extra_start..extra_start + extra_len];
                let mut ep = 0usize;
                while ep + 4 <= extra.len() {
                    let hdr_id = read_u16_le(extra, ep)?;
                    let sz = read_u16_le(extra, ep + 2)? as usize;
                    ep += 4;
                    if ep + sz > extra.len() {
                        return Err(Error::Invalid("truncated ZIP64 extra field".into()));
                    }
                    if hdr_id == 0x0001 {
                        let mut zpos = ep;
                        if uncomp_size == 0xFFFF_FFFF {
                            uncomp_size = read_u64_le(extra, zpos)?;
                            zpos += 8;
                        }
                        if comp_size == 0xFFFF_FFFF {
                            comp_size = read_u64_le(extra, zpos)?;
                            zpos += 8;
                        }
                        if local_hdr_off == 0xFFFF_FFFF {
                            local_hdr_off = read_u64_le(extra, zpos)?;
                        }
                        break;
                    }
                    ep += sz;
                }
            }

            if comp_size != uncomp_size {
                return Err(Error::Invalid(format!(
                    "STORED ZIP entry `{name}` has mismatched compressed/uncompressed sizes"
                )));
            }

            // Read 30-byte local file header to find exact payload offset (handles PyTorch alignment padding!).
            if local_hdr_off + 30 > file_len {
                return Err(Error::Invalid("ZIP local header out of bounds".into()));
            }
            file.seek(SeekFrom::Start(local_hdr_off))?;
            let mut lfh = [0u8; 30];
            file.read_exact(&mut lfh)?;
            if lfh[0..4] != [0x50, 0x4b, 0x03, 0x04] {
                return Err(Error::Invalid(
                    "invalid ZIP local file header signature".into(),
                ));
            }
            let lfh_name_len = read_u16_le(&lfh, 26)? as u64;
            let lfh_extra_len = read_u16_le(&lfh, 28)? as u64;
            let data_offset = local_hdr_off
                .checked_add(30)
                .and_then(|v| v.checked_add(lfh_name_len))
                .and_then(|v| v.checked_add(lfh_extra_len))
                .ok_or_else(|| Error::Invalid("ZIP data offset overflow".into()))?;
            if data_offset
                .checked_add(uncomp_size)
                .is_none_or(|end| end > file_len)
            {
                return Err(Error::Invalid(format!(
                    "ZIP entry `{name}` extends past end of file"
                )));
            }
            entries.insert(
                name,
                ZipEntry {
                    data_offset,
                    size: uncomp_size,
                },
            );
            pos = next_pos;
        }
        Ok(Self { file, entries })
    }

    pub fn entry_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn entry_size(&self, name: &str) -> Option<u64> {
        self.entries.get(name).map(|e| e.size)
    }

    /// Read a whole entry into memory (bounded by `max_bytes`).
    pub fn read_entry(&mut self, name: &str, max_bytes: usize) -> Result<Vec<u8>> {
        let entry = *self
            .entries
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing ZIP entry `{name}`")))?;
        if entry.size > max_bytes as u64 {
            return Err(Error::Invalid(format!(
                "ZIP entry `{name}` ({} bytes) exceeds limit {max_bytes}",
                entry.size
            )));
        }
        let mut buf = vec![0u8; entry.size as usize];
        self.file.seek(SeekFrom::Start(entry.data_offset))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read a byte subrange `[offset .. offset + len]` of an entry directly from disk.
    pub fn read_entry_range(&mut self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let entry = *self
            .entries
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing ZIP entry `{name}`")))?;
        let end = offset
            .checked_add(len as u64)
            .ok_or_else(|| Error::Invalid("ZIP range overflow".into()))?;
        if end > entry.size {
            return Err(Error::Invalid(format!(
                "ZIP entry `{name}` range {offset}..{end} exceeds entry size {}",
                entry.size
            )));
        }
        let mut buf = vec![0u8; len];
        self.file
            .seek(SeekFrom::Start(entry.data_offset + offset))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// Safe, bounded Python Pickle VM (protocols 2, 3, 4)
// ---------------------------------------------------------------------------

/// Supported PyTorch tensor dtypes in pinned checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtDtype {
    F32,
    I64,
    F16,
    BF16,
    U8,
}

impl PtDtype {
    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::U8 => 1,
        }
    }

    pub const fn as_safetensors(self) -> SafetensorsDtype {
        match self {
            Self::F32 => SafetensorsDtype::F32,
            Self::I64 => SafetensorsDtype::I64,
            Self::F16 => SafetensorsDtype::F16,
            Self::BF16 => SafetensorsDtype::BF16,
            Self::U8 => SafetensorsDtype::U8,
        }
    }
}

/// Descriptor for one contiguous tensor stored inside a PyTorch `.pt` ZIP archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtTensorDesc {
    pub storage_key: String,
    pub dtype: PtDtype,
    pub storage_offset_elems: usize,
    pub shape: Vec<usize>,
    pub byte_len: usize,
}

/// Safe value representation produced by the bounded pickle VM.
#[derive(Debug, Clone)]
pub enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
    String(String),
    List(Vec<PickleValue>),
    Tuple(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    Global(String, String),
    Storage {
        dtype: PtDtype,
        key: String,
        numel: usize,
    },
    Tensor(PtTensorDesc),
    Mark,
}

fn parse_pickle(data: &[u8], allow_torch_globals: bool) -> Result<PickleValue> {
    if data.len() > MAX_PICKLE_BYTES {
        return Err(Error::Invalid(format!(
            "pickle payload ({} bytes) exceeds {} MiB limit",
            data.len(),
            MAX_PICKLE_BYTES / (1024 * 1024)
        )));
    }
    let mut pos = 0usize;
    let mut stack: Vec<PickleValue> = Vec::new();
    let mut memo: BTreeMap<u32, PickleValue> = BTreeMap::new();
    let mut next_memo_id: u32 = 0;
    let mut ops = 0usize;

    let read_u8 = |p: &mut usize| -> Result<u8> {
        let b = *data
            .get(*p)
            .ok_or_else(|| Error::Invalid("unexpected EOF in pickle".into()))?;
        *p += 1;
        Ok(b)
    };
    let read_bytes = |p: &mut usize, n: usize| -> Result<&[u8]> {
        let end = p
            .checked_add(n)
            .ok_or_else(|| Error::Invalid("pickle length overflow".into()))?;
        let slice = data
            .get(*p..end)
            .ok_or_else(|| Error::Invalid("unexpected EOF reading pickle bytes".into()))?;
        *p = end;
        Ok(slice)
    };
    let read_line = |p: &mut usize| -> Result<String> {
        let start = *p;
        while *p < data.len() && data[*p] != b'\n' {
            *p += 1;
        }
        if *p >= data.len() {
            return Err(Error::Invalid("unterminated pickle line".into()));
        }
        let s = std::str::from_utf8(&data[start..*p])
            .map_err(|_| Error::Invalid("non-UTF-8 pickle line".into()))?
            .to_owned();
        *p += 1; // skip '\n'
        Ok(s)
    };

    while pos < data.len() {
        ops += 1;
        if ops > MAX_PICKLE_OPS || stack.len() > MAX_STACK || memo.len() > MAX_MEMO {
            return Err(Error::Invalid("pickle complexity bound exceeded".into()));
        }
        let op = read_u8(&mut pos)?;
        match op {
            // PROTO
            0x80 => {
                let proto = read_u8(&mut pos)?;
                if !(2..=4).contains(&proto) {
                    return Err(Error::Invalid(format!(
                        "unsupported pickle protocol {proto} (expected 2..=4)"
                    )));
                }
            }
            // FRAME (protocol 4 framing hint)
            0x95 => {
                let b = read_bytes(&mut pos, 8)?;
                let frame_len = u64::from_le_bytes(b.try_into().unwrap());
                if pos as u64 + frame_len > data.len() as u64 {
                    return Err(Error::Invalid("invalid pickle FRAME length".into()));
                }
            }
            // STOP '.'
            b'.' => {
                if stack.len() != 1 {
                    return Err(Error::Invalid(format!(
                        "pickle STOP with stack depth {} (expected 1)",
                        stack.len()
                    )));
                }
                return Ok(stack.pop().unwrap());
            }
            // MARK '('
            b'(' => stack.push(PickleValue::Mark),
            // NONE 'N'
            b'N' => stack.push(PickleValue::None),
            // NEWTRUE / NEWFALSE
            0x88 => stack.push(PickleValue::Bool(true)),
            0x89 => stack.push(PickleValue::Bool(false)),
            // BININT1 'K'
            b'K' => stack.push(PickleValue::Int(read_u8(&mut pos)? as i64)),
            // BININT2 'M'
            b'M' => {
                let b = read_bytes(&mut pos, 2)?;
                stack.push(PickleValue::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
            }
            // BININT 'J'
            b'J' => {
                let b = read_bytes(&mut pos, 4)?;
                stack.push(PickleValue::Int(
                    i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
                ));
            }
            // LONG1
            0x8a => {
                let n = read_u8(&mut pos)? as usize;
                let b = read_bytes(&mut pos, n)?;
                if n > 8 {
                    return Err(Error::Invalid("pickle LONG1 exceeds i64".into()));
                }
                let mut val: i64 = if n > 0 && (b[n - 1] & 0x80) != 0 {
                    -1
                } else {
                    0
                };
                for (i, &byte) in b.iter().enumerate() {
                    let mask = !(0xFFi64 << (i * 8));
                    val = (val & mask) | ((byte as i64) << (i * 8));
                }
                stack.push(PickleValue::Int(val));
            }
            // BINFLOAT 'G' (IEEE 754 big-endian f64)
            b'G' => {
                let b = read_bytes(&mut pos, 8)?;
                let f = f64::from_be_bytes(b.try_into().unwrap());
                stack.push(PickleValue::Float(f));
            }
            // SHORT_BINBYTES 'C'
            b'C' => {
                let n = read_u8(&mut pos)? as usize;
                let b = read_bytes(&mut pos, n)?;
                stack.push(PickleValue::Bytes(b.to_vec()));
            }
            // BINBYTES 'B'
            b'B' => {
                let lb = read_bytes(&mut pos, 4)?;
                let n = u32::from_le_bytes(lb.try_into().unwrap()) as usize;
                let b = read_bytes(&mut pos, n)?;
                stack.push(PickleValue::Bytes(b.to_vec()));
            }
            // SHORT_BINUNICODE
            0x8c => {
                let n = read_u8(&mut pos)? as usize;
                let b = read_bytes(&mut pos, n)?;
                let s = std::str::from_utf8(b)
                    .map_err(|_| Error::Invalid("invalid UTF-8 in SHORT_BINUNICODE".into()))?;
                stack.push(PickleValue::String(s.to_owned()));
            }
            // BINUNICODE 'X'
            b'X' => {
                let lb = read_bytes(&mut pos, 4)?;
                let n = u32::from_le_bytes(lb.try_into().unwrap()) as usize;
                let b = read_bytes(&mut pos, n)?;
                let s = std::str::from_utf8(b)
                    .map_err(|_| Error::Invalid("invalid UTF-8 in BINUNICODE".into()))?;
                stack.push(PickleValue::String(s.to_owned()));
            }
            // EMPTY_LIST ']'
            b']' => stack.push(PickleValue::List(Vec::new())),
            // EMPTY_DICT '}'
            b'}' => stack.push(PickleValue::Dict(Vec::new())),
            // EMPTY_TUPLE ')'
            b')' => stack.push(PickleValue::Tuple(Vec::new())),
            // TUPLE 't'
            b't' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(PickleValue::Tuple(items));
            }
            // TUPLE1 / TUPLE2 / TUPLE3
            0x85 => {
                let a = pop1(&mut stack)?;
                stack.push(PickleValue::Tuple(vec![a]));
            }
            0x86 => {
                let b = pop1(&mut stack)?;
                let a = pop1(&mut stack)?;
                stack.push(PickleValue::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop1(&mut stack)?;
                let b = pop1(&mut stack)?;
                let a = pop1(&mut stack)?;
                stack.push(PickleValue::Tuple(vec![a, b, c]));
            }
            // APPEND 'a'
            b'a' => {
                let v = pop1(&mut stack)?;
                match stack.last_mut() {
                    Some(PickleValue::List(l)) => l.push(v),
                    _ => return Err(Error::Invalid("pickle APPEND target is not a list".into())),
                }
            }
            // APPENDS 'e'
            b'e' => {
                let items = pop_to_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(PickleValue::List(l)) => l.extend(items),
                    _ => return Err(Error::Invalid("pickle APPENDS target is not a list".into())),
                }
            }
            // SETITEM 's'
            b's' => {
                let v = pop1(&mut stack)?;
                let k = pop1(&mut stack)?;
                match stack.last_mut() {
                    Some(PickleValue::Dict(d)) => d.push((k, v)),
                    _ => return Err(Error::Invalid("pickle SETITEM target is not a dict".into())),
                }
            }
            // SETITEMS 'u'
            b'u' => {
                let items = pop_to_mark(&mut stack)?;
                if !items.len().is_multiple_of(2) {
                    return Err(Error::Invalid("pickle SETITEMS odd pair count".into()));
                }
                let Some(PickleValue::Dict(d)) = stack.last_mut() else {
                    return Err(Error::Invalid(
                        "pickle SETITEMS target is not a dict".into(),
                    ));
                };
                let mut iter = items.into_iter();
                while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
                    d.push((k, v));
                }
            }
            // BINPUT 'q'
            b'q' => {
                let idx = read_u8(&mut pos)? as u32;
                let top = stack
                    .last()
                    .ok_or_else(|| Error::Invalid("pickle BINPUT on empty stack".into()))?
                    .clone();
                memo.insert(idx, top);
                if idx >= next_memo_id {
                    next_memo_id = idx + 1;
                }
            }
            // LONG_BINPUT 'r'
            b'r' => {
                let b = read_bytes(&mut pos, 4)?;
                let idx = u32::from_le_bytes(b.try_into().unwrap());
                let top = stack
                    .last()
                    .ok_or_else(|| Error::Invalid("pickle LONG_BINPUT on empty stack".into()))?
                    .clone();
                memo.insert(idx, top);
                if idx >= next_memo_id {
                    next_memo_id = idx + 1;
                }
            }
            // MEMOIZE
            0x94 => {
                let top = stack
                    .last()
                    .ok_or_else(|| Error::Invalid("pickle MEMOIZE on empty stack".into()))?
                    .clone();
                memo.insert(next_memo_id, top);
                next_memo_id += 1;
            }
            // BINGET 'h'
            b'h' => {
                let idx = read_u8(&mut pos)? as u32;
                let v = memo
                    .get(&idx)
                    .ok_or_else(|| Error::Invalid(format!("invalid pickle BINGET {idx}")))?
                    .clone();
                stack.push(v);
            }
            // LONG_BINGET 'j'
            b'j' => {
                let b = read_bytes(&mut pos, 4)?;
                let idx = u32::from_le_bytes(b.try_into().unwrap());
                let v = memo
                    .get(&idx)
                    .ok_or_else(|| Error::Invalid(format!("invalid pickle LONG_BINGET {idx}")))?
                    .clone();
                stack.push(v);
            }
            // GLOBAL 'c'
            b'c' if allow_torch_globals => {
                let module = read_line(&mut pos)?;
                let name = read_line(&mut pos)?;
                validate_torch_global(&module, &name)?;
                stack.push(PickleValue::Global(module, name));
            }
            // STACK_GLOBAL
            0x93 if allow_torch_globals => {
                let PickleValue::String(name) = pop1(&mut stack)? else {
                    return Err(Error::Invalid("STACK_GLOBAL name must be string".into()));
                };
                let PickleValue::String(module) = pop1(&mut stack)? else {
                    return Err(Error::Invalid("STACK_GLOBAL module must be string".into()));
                };
                validate_torch_global(&module, &name)?;
                stack.push(PickleValue::Global(module, name));
            }
            // BINPERSID 'Q' (PyTorch storage persistent reference)
            b'Q' if allow_torch_globals => {
                let pid = pop1(&mut stack)?;
                let storage = resolve_storage_pid(pid)?;
                stack.push(storage);
            }
            // REDUCE 'R'
            b'R' if allow_torch_globals => {
                let args = pop1(&mut stack)?;
                let func = pop1(&mut stack)?;
                let result = apply_torch_reduce(func, args)?;
                stack.push(result);
            }
            // BUILD 'b' (PyTorch OrderedDict state: empty {} or {'_metadata': ...})
            b'b' if allow_torch_globals => {
                let state = pop1(&mut stack)?;
                match state {
                    PickleValue::None => {}
                    PickleValue::Dict(ref d)
                        if d.iter().all(
                            |(k, _)| matches!(k, PickleValue::String(s) if s == "_metadata"),
                        ) => {}
                    _ => {
                        return Err(Error::Invalid(
                            "unexpected non-empty pickle BUILD state".into(),
                        ));
                    }
                }
            }
            other => {
                return Err(Error::Invalid(format!(
                    "unsupported or forbidden pickle opcode 0x{other:02x}"
                )));
            }
        }
    }
    Err(Error::Invalid("pickle missing STOP opcode".into()))
}

fn pop1(stack: &mut Vec<PickleValue>) -> Result<PickleValue> {
    stack
        .pop()
        .ok_or_else(|| Error::Invalid("pickle stack underflow".into()))
}

fn pop_to_mark(stack: &mut Vec<PickleValue>) -> Result<Vec<PickleValue>> {
    let idx = stack
        .iter()
        .rposition(|v| matches!(v, PickleValue::Mark))
        .ok_or_else(|| Error::Invalid("missing pickle MARK".into()))?;
    let items = stack.split_off(idx + 1);
    stack.pop(); // remove Mark
    Ok(items)
}

fn validate_torch_global(module: &str, name: &str) -> Result<()> {
    match (module, name) {
        ("collections", "OrderedDict")
        | ("torch._utils", "_rebuild_tensor_v2")
        | ("torch", "FloatStorage")
        | ("torch", "LongStorage")
        | ("torch", "HalfStorage")
        | ("torch", "BFloat16Storage")
        | ("torch", "ByteStorage") => Ok(()),
        _ => Err(Error::Invalid(format!(
            "forbidden pickle global `{module}.{name}`"
        ))),
    }
}

fn resolve_storage_pid(pid: PickleValue) -> Result<PickleValue> {
    // PyTorch persistent ID tuple: ("storage", StorageClass, key_str, location, numel)
    let PickleValue::Tuple(mut elems) = pid else {
        return Err(Error::Invalid("PyTorch BINPERSID must be a tuple".into()));
    };
    if elems.len() < 5 {
        return Err(Error::Invalid(
            "PyTorch storage persistent ID has too few fields".into(),
        ));
    }
    let PickleValue::String(ref tag) = elems[0] else {
        return Err(Error::Invalid("storage pid tag must be string".into()));
    };
    if tag != "storage" {
        return Err(Error::Invalid(format!(
            "unsupported persistent ID kind `{tag}`"
        )));
    }
    let dtype = match &elems[1] {
        PickleValue::Global(m, n) if m == "torch" => match n.as_str() {
            "FloatStorage" => PtDtype::F32,
            "LongStorage" => PtDtype::I64,
            "HalfStorage" => PtDtype::F16,
            "BFloat16Storage" => PtDtype::BF16,
            "ByteStorage" => PtDtype::U8,
            other => {
                return Err(Error::Invalid(format!(
                    "unsupported torch storage type `{other}`"
                )));
            }
        },
        _ => return Err(Error::Invalid("invalid storage class in pid".into())),
    };
    let PickleValue::String(key) = std::mem::replace(&mut elems[2], PickleValue::None) else {
        return Err(Error::Invalid("storage key must be string".into()));
    };
    let PickleValue::Int(numel_i) = elems[4] else {
        return Err(Error::Invalid("storage numel must be integer".into()));
    };
    let numel =
        usize::try_from(numel_i).map_err(|_| Error::Invalid("negative storage numel".into()))?;
    Ok(PickleValue::Storage { dtype, key, numel })
}

fn parse_usize_tuple(val: &PickleValue, label: &str) -> Result<Vec<usize>> {
    let PickleValue::Tuple(elems) = val else {
        return Err(Error::Invalid(format!("{label} must be a tuple")));
    };
    let mut out = Vec::with_capacity(elems.len());
    for e in elems {
        let PickleValue::Int(i) = *e else {
            return Err(Error::Invalid(format!("{label} element must be integer")));
        };
        out.push(usize::try_from(i).map_err(|_| Error::Invalid(format!("negative {label} dim")))?);
    }
    Ok(out)
}

fn check_contiguous_stride(shape: &[usize], stride: &[usize]) -> Result<()> {
    if shape.len() != stride.len() {
        return Err(Error::Invalid("tensor shape/stride rank mismatch".into()));
    }
    let mut expected = 1usize;
    for (&dim, &st) in shape.iter().rev().zip(stride.iter().rev()) {
        if dim > 1 && st != expected {
            return Err(Error::Invalid(format!(
                "non-contiguous tensor stride {stride:?} for shape {shape:?}"
            )));
        }
        expected = expected
            .checked_mul(dim)
            .ok_or_else(|| Error::Invalid("tensor shape product overflow".into()))?;
    }
    Ok(())
}

fn apply_torch_reduce(func: PickleValue, args: PickleValue) -> Result<PickleValue> {
    let PickleValue::Global(module, name) = func else {
        return Err(Error::Invalid(
            "pickle REDUCE target must be a global".into(),
        ));
    };
    match (module.as_str(), name.as_str()) {
        ("collections", "OrderedDict") => match args {
            PickleValue::Tuple(t) if t.is_empty() => Ok(PickleValue::Dict(Vec::new())),
            PickleValue::Tuple(mut t) if t.len() == 1 => {
                if let PickleValue::List(pairs) = t.pop().unwrap() {
                    let mut out = Vec::with_capacity(pairs.len());
                    for p in pairs {
                        let PickleValue::Tuple(mut kv) = p else {
                            return Err(Error::Invalid("OrderedDict pair must be tuple".into()));
                        };
                        if kv.len() != 2 {
                            return Err(Error::Invalid("OrderedDict pair length != 2".into()));
                        }
                        let v = kv.pop().unwrap();
                        let k = kv.pop().unwrap();
                        out.push((k, v));
                    }
                    Ok(PickleValue::Dict(out))
                } else {
                    Err(Error::Invalid("unexpected OrderedDict argument".into()))
                }
            }
            _ => Err(Error::Invalid("unsupported OrderedDict reduce args".into())),
        },
        ("torch._utils", "_rebuild_tensor_v2") => {
            let PickleValue::Tuple(elems) = args else {
                return Err(Error::Invalid(
                    "_rebuild_tensor_v2 args must be a tuple".into(),
                ));
            };
            if elems.len() < 4 {
                return Err(Error::Invalid(
                    "_rebuild_tensor_v2 missing arguments".into(),
                ));
            }
            let PickleValue::Storage { dtype, key, numel } = &elems[0] else {
                return Err(Error::Invalid(
                    "_rebuild_tensor_v2 arg 0 must be storage".into(),
                ));
            };
            let PickleValue::Int(offset_i) = elems[1] else {
                return Err(Error::Invalid("storage_offset must be integer".into()));
            };
            let storage_offset_elems = usize::try_from(offset_i)
                .map_err(|_| Error::Invalid("negative storage_offset".into()))?;
            let shape = parse_usize_tuple(&elems[2], "shape")?;
            let stride = parse_usize_tuple(&elems[3], "stride")?;
            check_contiguous_stride(&shape, &stride)?;
            let tensor_elems: usize = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| Error::Invalid("tensor element count overflow".into()))?;
            let end_elems = storage_offset_elems
                .checked_add(tensor_elems)
                .ok_or_else(|| Error::Invalid("tensor storage end overflow".into()))?;
            if end_elems > *numel {
                return Err(Error::Invalid(format!(
                    "tensor `{key}` view {storage_offset_elems}..{end_elems} exceeds storage numel {numel}"
                )));
            }
            let byte_len = tensor_elems
                .checked_mul(dtype.elem_bytes())
                .ok_or_else(|| Error::Invalid("tensor byte length overflow".into()))?;
            Ok(PickleValue::Tensor(PtTensorDesc {
                storage_key: key.clone(),
                dtype: *dtype,
                storage_offset_elems,
                shape,
                byte_len,
            }))
        }
        _ => Err(Error::Invalid(format!(
            "unsupported pickle REDUCE `{module}.{name}`"
        ))),
    }
}

/// Parse a data-only Python pickle (protocol 2..=4) forbidding ALL globals, callables,
/// and persistent references. Used for TinyStories `bpe_vocab_8192.pickle` & `bpe_merges_8192.pickle`.
pub fn parse_data_pickle(data: &[u8]) -> Result<PickleValue> {
    parse_pickle(data, false)
}

// ---------------------------------------------------------------------------
// High-level PyTorch `.pt` checkpoint reader
// ---------------------------------------------------------------------------

/// Opened PyTorch `.pt` checkpoint archive with parsed metadata and lazy streaming tensor reads.
pub struct PtCheckpoint {
    archive: ZipStoredArchive,
    prefix: String,
    root: BTreeMap<String, PickleValue>,
}

impl PtCheckpoint {
    /// Open a PyTorch `.pt` ZIP_STORED checkpoint and parse its `data.pkl` safely.
    pub fn open(path: &Path) -> Result<Self> {
        let mut archive = ZipStoredArchive::open(path)?;
        let pkl_names: Vec<String> = archive
            .entry_names()
            .filter(|n| n.ends_with("/data.pkl") || *n == "data.pkl")
            .map(str::to_owned)
            .collect();
        if pkl_names.len() != 1 {
            return Err(Error::Invalid(format!(
                "expected exactly one data.pkl in {}, found {}",
                path.display(),
                pkl_names.len()
            )));
        }
        let pkl_name = &pkl_names[0];
        let prefix = pkl_name.strip_suffix("data.pkl").unwrap_or("").to_owned();
        let pkl_bytes = archive.read_entry(pkl_name, MAX_PICKLE_BYTES)?;
        let parsed = parse_pickle(&pkl_bytes, true)?;
        let PickleValue::Dict(pairs) = parsed else {
            return Err(Error::Invalid(
                "PyTorch checkpoint root must be a dictionary".into(),
            ));
        };
        let mut root = BTreeMap::new();
        for (k, v) in pairs {
            let PickleValue::String(ks) = k else {
                return Err(Error::Invalid(
                    "PyTorch checkpoint top-level key must be string".into(),
                ));
            };
            root.insert(ks, v);
        }
        Ok(Self {
            archive,
            prefix,
            root,
        })
    }

    /// Top-level keys present in the checkpoint dict (e.g. `{"model"}` or `{"model", "inference"}`).
    pub fn top_keys(&self) -> BTreeSet<String> {
        self.root.keys().cloned().collect()
    }

    /// Extract a named tensor map (e.g. `"model"` or `"inference"`), validating unique storage
    /// keys (refusing unexpected storage aliases) and allowed dtypes (`F32` / `I64`).
    pub fn tensor_dict(&self, key: &str) -> Result<BTreeMap<String, PtTensorDesc>> {
        let val = self
            .root
            .get(key)
            .ok_or_else(|| Error::Invalid(format!("checkpoint missing top-level key `{key}`")))?;
        let PickleValue::Dict(pairs) = val else {
            return Err(Error::Invalid(format!(
                "checkpoint key `{key}` is not a dictionary"
            )));
        };
        let mut out = BTreeMap::new();
        let mut seen_storages = BTreeSet::new();
        for (k, v) in pairs {
            let PickleValue::String(name) = k else {
                return Err(Error::Invalid(format!(
                    "non-string key inside checkpoint `{key}`"
                )));
            };
            let PickleValue::Tensor(desc) = v else {
                return Err(Error::Invalid(format!(
                    "non-tensor value `{name}` inside checkpoint `{key}`"
                )));
            };
            if !seen_storages.insert(desc.storage_key.clone()) {
                return Err(Error::Invalid(format!(
                    "unexpected shared tensor storage `{}` for `{name}`",
                    desc.storage_key
                )));
            }
            out.insert(name.clone(), desc.clone());
        }
        Ok(out)
    }

    /// Extract a single named tensor from a sub-dictionary (e.g. `inference["level0_unigram"]`),
    /// allowing non-tensor sibling metadata keys like `inference["config"]`.
    pub fn single_tensor_in_dict(&self, dict_key: &str, tensor_key: &str) -> Result<PtTensorDesc> {
        let val = self
            .root
            .get(dict_key)
            .ok_or_else(|| Error::Invalid(format!("checkpoint missing `{dict_key}`")))?;
        let PickleValue::Dict(pairs) = val else {
            return Err(Error::Invalid(format!(
                "checkpoint `{dict_key}` is not a dictionary"
            )));
        };
        for (k, v) in pairs {
            if let PickleValue::String(name) = k
                && name == tensor_key
            {
                if let PickleValue::Tensor(desc) = v {
                    return Ok(desc.clone());
                }
                return Err(Error::Invalid(format!(
                    "`{dict_key}.{tensor_key}` is not a tensor"
                )));
            }
        }
        Err(Error::Invalid(format!(
            "missing `{dict_key}.{tensor_key}` in checkpoint"
        )))
    }

    /// Return the sub-keys present inside a top-level dictionary (e.g. `"inference"`).
    pub fn sub_dict_keys(&self, dict_key: &str) -> Result<BTreeSet<String>> {
        let val = self
            .root
            .get(dict_key)
            .ok_or_else(|| Error::Invalid(format!("checkpoint missing `{dict_key}`")))?;
        let PickleValue::Dict(pairs) = val else {
            return Err(Error::Invalid(format!(
                "checkpoint `{dict_key}` is not a dictionary"
            )));
        };
        let mut keys = BTreeSet::new();
        for (k, _) in pairs {
            let PickleValue::String(name) = k else {
                return Err(Error::Invalid(format!(
                    "non-string key inside `{dict_key}`"
                )));
            };
            keys.insert(name.clone());
        }
        Ok(keys)
    }

    /// Stream raw little-endian bytes for one tensor directly from the ZIP_STORED archive.
    pub fn read_tensor_bytes(&mut self, desc: &PtTensorDesc) -> Result<Vec<u8>> {
        let entry_name = format!("{}data/{}", self.prefix, desc.storage_key);
        let byte_offset = (desc.storage_offset_elems as u64)
            .checked_mul(desc.dtype.elem_bytes() as u64)
            .ok_or_else(|| Error::Invalid("tensor byte offset overflow".into()))?;
        self.archive
            .read_entry_range(&entry_name, byte_offset, desc.byte_len)
    }
}

/// Compute streaming SHA-256 and exact byte size of a regular non-symlink file.
pub fn sha256_file(path: &Path) -> Result<(u64, String)> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(Error::Invalid(format!(
            "expected regular non-symlink file: {}",
            path.display()
        )));
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        hasher.update(&buf[..n]);
    }
    Ok((total, format!("{:x}", hasher.finalize())))
}

/// Convert `PtDtype` to `ncm_io::TensorDtype` (rejecting non-F32/I64 dtypes for upstream bundles).
fn pt_to_manifest_dtype(dt: PtDtype) -> Result<TensorDtype> {
    match dt {
        PtDtype::F32 => Ok(TensorDtype::F32),
        PtDtype::I64 => Ok(TensorDtype::I64),
        other => Err(Error::Invalid(format!(
            "unsupported manifest tensor dtype {other:?} (expected F32 or I64)"
        ))),
    }
}

/// Owned raw tensor ready to write into a checked safetensors bundle.
#[derive(Debug, Clone)]
pub struct OwnedRawTensor {
    pub dtype: PtDtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Read all tensors from an existing safetensors file (e.g. a pinned `--reference-rope` sidecar).
pub fn read_safetensors_raw(path: &Path) -> Result<BTreeMap<String, OwnedRawTensor>> {
    let bytes = fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)
        .map_err(|e| Error::Invalid(format!("safetensors deserialize {}: {e}", path.display())))?;
    let mut out = BTreeMap::new();
    for (name, view) in st.tensors() {
        let dtype = match view.dtype() {
            SafetensorsDtype::F32 => PtDtype::F32,
            SafetensorsDtype::I64 => PtDtype::I64,
            SafetensorsDtype::F16 => PtDtype::F16,
            SafetensorsDtype::BF16 => PtDtype::BF16,
            SafetensorsDtype::U8 => PtDtype::U8,
            other => {
                return Err(Error::Invalid(format!(
                    "unsupported safetensors dtype {other:?} in `{name}`"
                )));
            }
        };
        out.insert(
            name,
            OwnedRawTensor {
                dtype,
                shape: view.shape().to_vec(),
                bytes: view.data().to_vec(),
            },
        );
    }
    Ok(out)
}

/// Write a `BTreeMap<String, OwnedRawTensor>` to `dest_path` as safetensors (streaming directly
/// to disk via `serialize_to_file`), verify the written file's tensor headers and data slices,
/// and return sorted per-tensor SHA-256 `TensorRecord`s.
pub fn write_checked_safetensors(
    dest_path: &Path,
    tensors: &BTreeMap<String, OwnedRawTensor>,
) -> Result<Vec<TensorRecord>> {
    if dest_path.exists() || dest_path.is_symlink() {
        return Err(Error::Invalid(format!(
            "refusing to overwrite existing safetensors file {}",
            dest_path.display()
        )));
    }
    let mut views = Vec::with_capacity(tensors.len());
    let mut records = Vec::with_capacity(tensors.len());
    for (name, t) in tensors {
        let tv = TensorView::new(t.dtype.as_safetensors(), t.shape.clone(), &t.bytes)
            .map_err(|e| Error::Invalid(format!("safetensors view `{name}`: {e}")))?;
        views.push((name.as_str(), tv));
        records.push(TensorRecord {
            name: name.clone(),
            shape: t.shape.clone(),
            dtype: pt_to_manifest_dtype(t.dtype)?,
            bytes: t.bytes.len(),
            sha256: sha256(&t.bytes),
        });
    }
    safetensors::tensor::serialize_to_file(views, None, dest_path)
        .map_err(|e| Error::Invalid(format!("safetensors serialize_to_file: {e}")))?;

    // Stream-verify every written tensor offset + SHA-256 directly from disk without doubling RAM.
    let mut file = File::open(dest_path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Err(Error::Invalid(
            "invalid written safetensors header length".into(),
        ));
    }
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header_bytes)?;
    let data_base = 8u64 + header_len as u64;
    let mut count = 0usize;
    for (name, rec) in tensors.keys().zip(&records) {
        let entry = header
            .get(name)
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Invalid(format!("written safetensors missing `{name}`")))?;
        let offsets = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Invalid(format!("missing data_offsets for `{name}`")))?;
        let start = offsets
            .first()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Invalid("invalid start offset".into()))?;
        let end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Invalid("invalid end offset".into()))?;
        if (end - start) as usize != rec.bytes {
            return Err(Error::Invalid(format!(
                "written safetensors byte length mismatch for `{name}`"
            )));
        }
        file.seek(SeekFrom::Start(data_base + start))?;
        let mut hasher = Sha256::new();
        let mut remaining = rec.bytes;
        let mut buf = vec![0u8; remaining.min(1024 * 1024)];
        while remaining > 0 {
            let chunk = remaining.min(buf.len());
            file.read_exact(&mut buf[..chunk])?;
            hasher.update(&buf[..chunk]);
            remaining -= chunk;
        }
        let written_sha = format!("{:x}", hasher.finalize());
        if written_sha != rec.sha256 {
            return Err(Error::Invalid(format!(
                "non-bit-exact safetensors roundtrip on `{name}`"
            )));
        }
        count += 1;
    }
    if count != tensors.len() {
        return Err(Error::Invalid(
            "safetensors roundtrip tensor count mismatch".into(),
        ));
    }
    Ok(records)
}

/// Stream a selected subset of tensors from an open `PtCheckpoint` (plus optional extra tensors
/// such as `rope_cos`/`rope_sin`), write them to `dest_path`, verify bit-exact roundtrip, and
/// return sorted `TensorRecord`s.
pub fn write_checked_safetensors_from_pt(
    ckpt: &mut PtCheckpoint,
    descs: &BTreeMap<String, PtTensorDesc>,
    extras: Option<&BTreeMap<String, OwnedRawTensor>>,
    dest_path: &Path,
) -> Result<Vec<TensorRecord>> {
    let mut owned = BTreeMap::new();
    for (name, desc) in descs {
        let bytes = ckpt.read_tensor_bytes(desc)?;
        owned.insert(
            name.clone(),
            OwnedRawTensor {
                dtype: desc.dtype,
                shape: desc.shape.clone(),
                bytes,
            },
        );
    }
    if let Some(ex) = extras {
        for (name, t) in ex {
            owned.insert(name.clone(), t.clone());
        }
    }
    write_checked_safetensors(dest_path, &owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal valid uncompressed ZIP_STORED archive in memory.
    fn build_zip_stored(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cd = Vec::new();
        for &(name, data) in files {
            let local_off = out.len() as u32;
            let name_bytes = name.as_bytes();
            let sz = data.len() as u32;
            // Local file header
            out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
            out.extend_from_slice(&20u16.to_le_bytes()); // version
            out.extend_from_slice(&0u16.to_le_bytes()); // flags
            out.extend_from_slice(&0u16.to_le_bytes()); // method=STORED
            out.extend_from_slice(&0u16.to_le_bytes()); // mtime
            out.extend_from_slice(&0u16.to_le_bytes()); // mdate
            out.extend_from_slice(&0u32.to_le_bytes()); // crc (unused for our SHA-verified inputs)
            out.extend_from_slice(&sz.to_le_bytes());
            out.extend_from_slice(&sz.to_le_bytes());
            out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra len
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(data);

            // Central directory header
            cd.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
            cd.extend_from_slice(&20u16.to_le_bytes());
            cd.extend_from_slice(&20u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes()); // method=STORED
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&sz.to_le_bytes());
            cd.extend_from_slice(&sz.to_le_bytes());
            cd.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&local_off.to_le_bytes());
            cd.extend_from_slice(name_bytes);
        }
        let cd_off = out.len() as u32;
        let cd_sz = cd.len() as u32;
        out.extend_from_slice(&cd);
        // EOCD
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_sz.to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn parses_synthetic_pytorch_pt_and_rejects_arbitrary_globals() {
        // Construct protocol-2 pickle equivalent to:
        // {"model": OrderedDict([("w", _rebuild_tensor_v2(storage(FloatStorage, "0", "cpu", 4), 0, (2, 2), (2, 1), False, OrderedDict()))])}
        let mut pkl = Vec::new();
        pkl.extend_from_slice(&[0x80, 0x02]); // PROTO 2
        pkl.push(b'}'); // EMPTY_DICT
        pkl.extend_from_slice(&[0x8c, 5]);
        pkl.extend_from_slice(b"model");
        pkl.extend_from_slice(b"ccollections\nOrderedDict\n)R"); // OrderedDict()
        pkl.extend_from_slice(&[0x8c, 1, b'w']);
        pkl.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        pkl.push(b'('); // args mark
        // storage pid tuple: ("storage", torch.FloatStorage, "0", "cpu", 4)
        pkl.push(b'(');
        pkl.extend_from_slice(&[0x8c, 7]);
        pkl.extend_from_slice(b"storage");
        pkl.extend_from_slice(b"ctorch\nFloatStorage\n");
        pkl.extend_from_slice(&[0x8c, 1, b'0']);
        pkl.extend_from_slice(&[0x8c, 3, b'c', b'p', b'u']);
        pkl.extend_from_slice(&[b'K', 4]);
        pkl.push(b't'); // TUPLE
        pkl.push(b'Q'); // BINPERSID
        pkl.extend_from_slice(&[b'K', 0]); // offset 0
        pkl.extend_from_slice(&[b'K', 2, b'K', 2, 0x86]); // shape (2, 2)
        pkl.extend_from_slice(&[b'K', 2, b'K', 1, 0x86]); // stride (2, 1)
        pkl.push(0x89); // NEWFALSE
        pkl.extend_from_slice(b"ccollections\nOrderedDict\n)R");
        pkl.push(b't'); // args tuple
        pkl.push(b'R'); // REDUCE
        pkl.push(b's'); // SETITEM on model OrderedDict
        pkl.push(b's'); // SETITEM on root dict
        pkl.push(b'.'); // STOP

        let tensor_vals: [f32; 4] = [1.0, -2.5, 3.25, 0.0];
        let mut raw_bytes = Vec::new();
        for v in tensor_vals {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let zip_bytes =
            build_zip_stored(&[("archive/data.pkl", &pkl), ("archive/data/0", &raw_bytes)]);
        let dir = tempfile::tempdir().unwrap();
        let pt_path = dir.path().join("test.pt");
        File::create(&pt_path)
            .unwrap()
            .write_all(&zip_bytes)
            .unwrap();

        let mut ckpt = PtCheckpoint::open(&pt_path).unwrap();
        assert!(ckpt.top_keys().contains("model"));
        let model = ckpt.tensor_dict("model").unwrap();
        assert_eq!(model.len(), 1);
        let desc = &model["w"];
        assert_eq!(desc.shape, vec![2, 2]);
        assert_eq!(desc.dtype, PtDtype::F32);
        assert_eq!(ckpt.read_tensor_bytes(desc).unwrap(), raw_bytes);

        // Arbitrary Python global (e.g. os.system) MUST be rejected!
        let mut evil = Vec::new();
        evil.extend_from_slice(&[0x80, 0x02]);
        evil.extend_from_slice(b"cos\nsystem\n");
        evil.extend_from_slice(&[0x8c, 2, b'i', b'd', 0x85, b'R', b'.']);
        assert!(parse_pickle(&evil, true).is_err());
        assert!(parse_data_pickle(&evil).is_err());
    }

    #[test]
    fn data_pickle_parses_bpe_structures_and_forbids_globals() {
        // {0: b"a", 1: b"b"}
        let mut pkl = Vec::new();
        pkl.extend_from_slice(&[0x80, 0x04]); // PROTO 4
        let body = [
            b'}', b'(', b'K', 0, b'C', 1, b'a', b'K', 1, b'C', 1, b'b', b'u', b'.',
        ];
        pkl.push(0x95); // FRAME
        pkl.extend_from_slice(&(body.len() as u64).to_le_bytes());
        pkl.extend_from_slice(&body);
        let val = parse_data_pickle(&pkl).unwrap();
        let PickleValue::Dict(pairs) = val else {
            panic!("expected dict");
        };
        assert_eq!(pairs.len(), 2);
    }
}
