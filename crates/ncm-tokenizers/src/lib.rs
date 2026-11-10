//! Python-free Sudoku, byte-BPE and immutable HF tokenizer adapters.
mod assets;
mod bpe;
mod hf_bpe;
pub use assets::{AssetError, TextTokenizer, TokenizerMetadata};
pub use bpe::ByteBpe;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Character(char),
    Token(u32),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Character(c) => write!(f, "unknown Sudoku character: {c:?}"),
            Self::Token(id) => write!(f, "invalid Sudoku token ID: {id}"),
        }
    }
}
impl std::error::Error for Error {}

/// Matches the pinned Python Sudoku tokenizer, including 0 -> blank and BOS handling.
pub fn encode_sudoku(text: &str) -> Result<Vec<u32>, Error> {
    text.replace("<BOS>", "B")
        .chars()
        .filter(|c| {
            // Python str.isspace additionally recognizes the ASCII information separators.
            !c.is_whitespace() && !matches!(c, '\u{1c}'..='\u{1f}')
        })
        .map(|c| match c {
            '_' | '0' => Ok(0),
            '1'..='9' => Ok(c as u32 - '0' as u32),
            '|' => Ok(10),
            'B' => Ok(11),
            _ => Err(Error::Character(c)),
        })
        .collect()
}

pub fn decode_sudoku(ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
    let mut text = String::new();
    for &id in ids {
        match id {
            0 => text.push('_'),
            1..=9 => text.push(char::from(b'0' + id as u8)),
            10 => text.push('|'),
            11 if skip_special_tokens => {}
            11 => text.push_str("<BOS>"),
            _ => return Err(Error::Token(id)),
        }
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_upstream_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/sudoku-tokenizer-v1.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(encode_sudoku(case["text"].as_str().unwrap()).unwrap(), ids);
            assert_eq!(
                decode_sudoku(&ids, false).unwrap(),
                case["decoded"].as_str().unwrap()
            );
            assert_eq!(
                decode_sudoku(&ids, true).unwrap(),
                case["decoded_skip_special"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn punctuation_zero_and_special_tokens() {
        let ids = encode_sudoku("<BOS> 0_12|9 B").unwrap();
        assert_eq!(ids, [11, 0, 0, 1, 2, 10, 9, 11]);
        assert_eq!(decode_sudoku(&ids, false).unwrap(), "<BOS>__12|9<BOS>");
        assert_eq!(decode_sudoku(&ids, true).unwrap(), "__12|9");
    }
    #[test]
    fn whitespace_and_invalid_inputs() {
        assert_eq!(encode_sudoku("1\u{1c}\u{a0}\u{2003}\n2").unwrap(), [1, 2]);
        assert!(encode_sudoku("x").is_err());
        assert!(decode_sudoku(&[12], false).is_err());
        assert!(encode_sudoku("").unwrap().is_empty());
    }
}
