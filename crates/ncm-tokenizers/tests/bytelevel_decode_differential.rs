use std::collections::HashMap;
use tokenizers::{Decoder, pre_tokenizers::byte_level::ByteLevel};

fn reference(tokens: Vec<String>) -> String {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars: Vec<u32> = bytes.iter().map(|b| u32::from(*b)).collect();
    let mut n = 0;
    for b in 0..=255_u8 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    let map: HashMap<char, u8> = chars
        .into_iter()
        .map(|v| char::from_u32(v).unwrap())
        .zip(bytes)
        .collect();
    let result: Vec<u8> = tokens
        .into_iter()
        .flat_map(|token| {
            token
                .chars()
                .try_fold(Vec::new(), |mut out, c| {
                    map.get(&c).map(|b| {
                        out.push(*b);
                        out
                    })
                })
                .unwrap_or_else(|| token.as_bytes().to_vec())
        })
        .collect();
    String::from_utf8_lossy(&result).into_owned()
}

#[test]
fn decoder_preserves_token_fallback_and_lossy_utf8() {
    let mut tokens: Vec<String> = (0..=511)
        .filter_map(char::from_u32)
        .map(|c| c.to_string())
        .collect();
    tokens.extend([
        "猫".into(),
        "a猫b".into(),
        "😀".into(),
        "".into(),
        "<|endoftext|>".into(),
    ]);
    let decoder = ByteLevel::default();
    for (index, token) in tokens.iter().enumerate() {
        for other in [
            "",
            "a",
            "猫",
            "ÿ",
            "Ġ",
            tokens[(index * 37) % tokens.len()].as_str(),
        ] {
            let input = vec![token.clone(), other.to_owned()];
            assert_eq!(
                decoder.decode_chain(input.clone()).unwrap(),
                vec![reference(input)]
            );
        }
    }
}
