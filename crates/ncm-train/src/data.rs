//! Dataset loading and deterministic batch sampling for `local-ncm-v1` training.
use crate::{Result, TrainError, schedule::splitmix64};
use ncm_core::LocalNcmConfig;
use ncm_tokenizers::LocalTokenizer;
use serde::Deserialize;
use std::{fs, path::Path};

/// One validated training pair `(prefix_tokens, target_tokens)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingSample {
    pub prefix: Vec<u32>,
    pub target: Vec<u32>,
}

/// Validated in-memory token dataset for codec and generator training.
#[derive(Debug, Clone)]
pub struct TokenDataset {
    samples: Vec<TrainingSample>,
    prefix_len: usize,
    context_len: usize,
    vocab_size: usize,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawDatasetJson {
    Pairs(Vec<RawPair>),
    TokenRows(Vec<Vec<u32>>),
    TextRows(Vec<String>),
}

#[derive(Deserialize)]
struct RawPair {
    prefix: Vec<u32>,
    target: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawJsonlRecord {
    Pair { prefix: Vec<u32>, target: Vec<u32> },
    PromptCompletion { prompt: String, completion: String },
    Text { text: String },
}

impl TokenDataset {
    /// Construct a validated dataset from explicit `(prefix, target)` token sequences.
    pub fn from_samples(config: &LocalNcmConfig, samples: Vec<TrainingSample>) -> Result<Self> {
        if samples.is_empty() {
            return Err(TrainError::Data("training dataset is empty".into()));
        }
        let prefix_len = config.model().prefix_len;
        let context_len = config.codec().raw().context_length;
        let vocab_size = config.codec().raw().in_vocab_size;
        let pad_id = config.model().pad_token_id;
        for (idx, s) in samples.iter().enumerate() {
            if s.prefix.len() != prefix_len {
                return Err(TrainError::Data(format!(
                    "sample {idx}: prefix length {} != expected {prefix_len}",
                    s.prefix.len()
                )));
            }
            if s.target.len() != context_len {
                return Err(TrainError::Data(format!(
                    "sample {idx}: target length {} != expected {context_len}",
                    s.target.len()
                )));
            }
            if let Some(&bad) = s
                .prefix
                .iter()
                .chain(&s.target)
                .find(|&&t| t as usize >= vocab_size)
            {
                return Err(TrainError::Data(format!(
                    "sample {idx}: token id {bad} >= vocab size {vocab_size}"
                )));
            }
            if let Some(pad) = pad_id
                && s.prefix.iter().all(|&t| t == pad)
            {
                return Err(TrainError::Data(format!(
                    "sample {idx}: prefix row is all pad token {pad}"
                )));
            }
        }
        Ok(Self {
            samples,
            prefix_len,
            context_len,
            vocab_size,
        })
    }

    /// Load a training or validation dataset from:
    /// - `.jsonl` / `.ndjson` (`{"text": "..."}`, `{"prompt": "...", "completion": "..."}`,
    ///   or `{"prefix": [...], "target": [...]}` with sliding-window chunking for long texts),
    /// - `.txt` (newline-delimited text, requires `tokenizer`), or
    /// - `.json` (`[[u32]]`, `[{"prefix":[..],"target":[..]}]`, or `["text"]`).
    pub fn load(
        path: &Path,
        config: &LocalNcmConfig,
        tokenizer: Option<&LocalTokenizer>,
    ) -> Result<Self> {
        let bytes = fs::read(path)
            .map_err(|e| TrainError::Io(format!("read dataset {}: {e}", path.display())))?;
        let prefix_len = config.model().prefix_len;
        let context_len = config.codec().raw().context_length;
        let pad_id = config.model().pad_token_id.unwrap_or(0);

        if path
            .extension()
            .is_some_and(|e| e == "jsonl" || e == "ndjson")
        {
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| TrainError::Data(format!("invalid UTF-8 JSONL dataset: {e}")))?;
            return Self::from_jsonl_str(config, tokenizer, text);
        }

        if path.extension().is_some_and(|e| e == "txt") {
            let tok = tokenizer.ok_or_else(|| {
                TrainError::Data(
                    "plain text dataset requires --tokenizer PATH (or tokenizer.json in --init-from)"
                        .into(),
                )
            })?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| TrainError::Data(format!("invalid UTF-8 dataset: {e}")))?;
            let lines: Vec<String> = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(ToOwned::to_owned)
                .collect();
            return Self::from_texts(config, tok, &lines);
        }

        let parsed: RawDatasetJson = serde_json::from_slice(&bytes)
            .map_err(|e| TrainError::Data(format!("invalid dataset JSON: {e}")))?;
        match parsed {
            RawDatasetJson::Pairs(pairs) => {
                let samples = pairs
                    .into_iter()
                    .map(|p| TrainingSample {
                        prefix: p.prefix,
                        target: p.target,
                    })
                    .collect();
                Self::from_samples(config, samples)
            }
            RawDatasetJson::TokenRows(rows) => {
                let mut samples = Vec::with_capacity(rows.len());
                for (idx, row) in rows.into_iter().enumerate() {
                    if row.len() == prefix_len + context_len {
                        let prefix = row[..prefix_len].to_vec();
                        let target = row[prefix_len..].to_vec();
                        samples.push(TrainingSample { prefix, target });
                    } else if row.len() == context_len {
                        let mut prefix = Vec::with_capacity(prefix_len);
                        for i in 0..prefix_len {
                            prefix.push(*row.get(i).unwrap_or(&pad_id));
                        }
                        samples.push(TrainingSample {
                            prefix,
                            target: row,
                        });
                    } else {
                        return Err(TrainError::Data(format!(
                            "row {idx}: expected length {context_len} or {}, got {}",
                            prefix_len + context_len,
                            row.len()
                        )));
                    }
                }
                Self::from_samples(config, samples)
            }
            RawDatasetJson::TextRows(texts) => {
                let tok = tokenizer.ok_or_else(|| {
                    TrainError::Data(
                        "string JSON dataset requires --tokenizer PATH (or tokenizer.json in --init-from)"
                            .into(),
                    )
                })?;
                Self::from_texts(config, tok, &texts)
            }
        }
    }

    fn push_chunked_ids(
        samples: &mut Vec<TrainingSample>,
        ids: &[u32],
        prefix_len: usize,
        context_len: usize,
        pad_id: u32,
    ) {
        if ids.is_empty() {
            return;
        }
        if ids.len() <= context_len {
            let mut prefix: Vec<u32> = ids.iter().copied().take(prefix_len).collect();
            while prefix.len() < prefix_len {
                prefix.push(pad_id);
            }
            let mut target = ids.to_vec();
            while target.len() < context_len {
                target.push(pad_id);
            }
            samples.push(TrainingSample { prefix, target });
            return;
        }
        // Sliding-window chunking with 50% overlap (stride = max(context_len / 2, 1)).
        let stride = (context_len / 2).max(1);
        let mut start = 0usize;
        while start < ids.len() {
            let end = (start + context_len).min(ids.len());
            let slice = &ids[start..end];
            let mut prefix: Vec<u32> = slice.iter().copied().take(prefix_len).collect();
            while prefix.len() < prefix_len {
                prefix.push(pad_id);
            }
            let mut target = slice.to_vec();
            while target.len() < context_len {
                target.push(pad_id);
            }
            samples.push(TrainingSample { prefix, target });
            if end == ids.len() {
                break;
            }
            start += stride;
        }
    }

    fn from_jsonl_str(
        config: &LocalNcmConfig,
        tokenizer: Option<&LocalTokenizer>,
        content: &str,
    ) -> Result<Self> {
        let prefix_len = config.model().prefix_len;
        let context_len = config.codec().raw().context_length;
        let pad_id = config.model().pad_token_id.unwrap_or(0);
        let mut samples = Vec::new();
        for (line_idx, raw_line) in content.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let rec: RawJsonlRecord = serde_json::from_str(line).map_err(|e| {
                TrainError::Data(format!("invalid JSONL on line {}: {e}", line_idx + 1))
            })?;
            match rec {
                RawJsonlRecord::Pair { prefix, target } => {
                    samples.push(TrainingSample { prefix, target });
                }
                RawJsonlRecord::PromptCompletion { prompt, completion } => {
                    let tok = tokenizer.ok_or_else(|| {
                        TrainError::Data(
                            "text JSONL dataset requires --tokenizer PATH (or tokenizer.json in --init-from)"
                                .into(),
                        )
                    })?;
                    let p_ids = tok
                        .encode(&prompt)
                        .map_err(|e| TrainError::Data(format!("tokenize prompt: {e}")))?;
                    let c_ids = tok
                        .encode(&completion)
                        .map_err(|e| TrainError::Data(format!("tokenize completion: {e}")))?;
                    if p_ids.is_empty() || c_ids.is_empty() {
                        continue;
                    }
                    let mut prefix: Vec<u32> = p_ids.into_iter().take(prefix_len).collect();
                    while prefix.len() < prefix_len {
                        prefix.push(pad_id);
                    }
                    let mut target: Vec<u32> = c_ids.into_iter().take(context_len).collect();
                    while target.len() < context_len {
                        target.push(pad_id);
                    }
                    samples.push(TrainingSample { prefix, target });
                }
                RawJsonlRecord::Text { text } => {
                    let tok = tokenizer.ok_or_else(|| {
                        TrainError::Data(
                            "text JSONL dataset requires --tokenizer PATH (or tokenizer.json in --init-from)"
                                .into(),
                        )
                    })?;
                    let ids = tok
                        .encode(&text)
                        .map_err(|e| TrainError::Data(format!("tokenize error: {e}")))?;
                    Self::push_chunked_ids(&mut samples, &ids, prefix_len, context_len, pad_id);
                }
            }
        }
        Self::from_samples(config, samples)
    }

    fn from_texts(
        config: &LocalNcmConfig,
        tokenizer: &LocalTokenizer,
        texts: &[String],
    ) -> Result<Self> {
        let prefix_len = config.model().prefix_len;
        let context_len = config.codec().raw().context_length;
        let pad_id = config.model().pad_token_id.unwrap_or(0);
        let mut samples = Vec::with_capacity(texts.len());
        for text in texts {
            let ids = tokenizer
                .encode(text)
                .map_err(|e| TrainError::Data(format!("tokenize error: {e}")))?;
            Self::push_chunked_ids(&mut samples, &ids, prefix_len, context_len, pad_id);
        }
        Self::from_samples(config, samples)
    }

    /// Return sequential validation batches covering up to `max_samples` rows.
    pub fn eval_batches(
        &self,
        batch_size: usize,
        max_samples: usize,
    ) -> Vec<(Vec<u32>, Vec<u32>, usize)> {
        let limit = self.samples.len().min(max_samples.max(1));
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < limit {
            let end = (start + batch_size).min(limit);
            let count = end - start;
            let mut prefixes = Vec::with_capacity(count * self.prefix_len);
            let mut targets = Vec::with_capacity(count * self.context_len);
            for s in &self.samples[start..end] {
                prefixes.extend_from_slice(&s.prefix);
                targets.extend_from_slice(&s.target);
            }
            out.push((prefixes, targets, count));
            start = end;
        }
        out
    }

    pub fn samples(&self) -> &[TrainingSample] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    pub fn context_len(&self) -> usize {
        self.context_len
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Deterministically sample a minibatch of `(flat_prefixes, flat_targets)` for a given step.
    pub fn sample_batch(&self, batch_size: usize, step: usize, seed: u64) -> (Vec<u32>, Vec<u32>) {
        let mut prefixes = Vec::with_capacity(batch_size * self.prefix_len);
        let mut targets = Vec::with_capacity(batch_size * self.context_len);
        let mut state = splitmix64(seed ^ ((step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        for _ in 0..batch_size {
            state = splitmix64(state);
            let idx = (state as usize) % self.samples.len();
            let s = &self.samples[idx];
            prefixes.extend_from_slice(&s.prefix);
            targets.extend_from_slice(&s.target);
        }
        (prefixes, targets)
    }
}
