use std::collections::HashMap;
use tokenizers::{
    pre_tokenizers::byte_level::ByteLevel,
    tokenizer::{
        OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Result,
        SplitDelimiterBehavior,
    },
    utils::SysRegex,
};
fn reference(input: &mut PreTokenizedString, prefix: bool, regex: bool) -> Result<()> {
    let pattern = SysRegex::new(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    )?;
    input.split(|_, mut normalized| {
        if prefix && !normalized.get().starts_with(' ') {
            normalized.prepend(" ");
        }
        if regex {
            normalized.split(&pattern, SplitDelimiterBehavior::Isolated)
        } else {
            Ok(vec![normalized])
        }
    })?;
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
    let map: HashMap<u8, char> = bytes
        .into_iter()
        .zip(chars.into_iter().map(|v| char::from_u32(v).unwrap()))
        .collect();
    input.normalize(|normalized| {
        let s = normalized.get();
        let mut transforms = Vec::with_capacity(s.len());
        for (i, c) in s.char_indices() {
            transforms.extend(
                s.as_bytes()[i..i + c.len_utf8()]
                    .iter()
                    .enumerate()
                    .map(|(j, b)| (map[b], isize::from(j > 0))),
            );
        }
        normalized.transform(transforms, 0);
        Ok(())
    })
}
#[test]
fn bytelevel_matches_unoptimized_splits_and_offsets() -> Result<()> {
    let mut samples = vec![
        String::new(),
        "Hello world!".into(),
        "猫 café\n 😀".into(),
        "a   b\t!".into(),
    ];
    for byte in 0_u8..=127 {
        samples.push(format!("a{}Z", char::from(byte)));
    }
    for text in samples {
        for prefix in [false, true] {
            for regex in [false, true] {
                for pre in 0..4 {
                    let mut actual = PreTokenizedString::from(text.as_str());
                    if pre != 0 {
                        actual.normalize(|n| {
                            match pre {
                                1 => {
                                    n.prepend("X");
                                }
                                2 => {
                                    n.append("Y");
                                }
                                _ => {
                                    n.filter(|c| c != 'a');
                                }
                            }
                            Ok(())
                        })?;
                    }
                    let mut expected = actual.clone();
                    ByteLevel::new(prefix, true, regex).pre_tokenize(&mut actual)?;
                    reference(&mut expected, prefix, regex)?;
                    for referential in [OffsetReferential::Original, OffsetReferential::Normalized]
                    {
                        let a: Vec<_> = actual
                            .get_splits(referential, OffsetType::Byte)
                            .into_iter()
                            .map(|(s, o, _)| (s.to_owned(), o))
                            .collect();
                        let b: Vec<_> = expected
                            .get_splits(referential, OffsetType::Byte)
                            .into_iter()
                            .map(|(s, o, _)| (s.to_owned(), o))
                            .collect();
                        assert_eq!(a, b, "{text:?} {prefix} {regex} {pre}");
                    }
                }
            }
        }
    }
    Ok(())
}
