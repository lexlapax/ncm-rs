use tokenizers::{NormalizedString, tokenizer::normalizer::Range};

#[test]
fn printable_identity_preserves_preexisting_alignments() {
    let mut samples = vec![String::new(), "Hello!".into(), "abc123".into()];
    for byte in b'!'..=b'~' {
        samples.push(char::from(byte).to_string());
        samples.push(format!("a{}Z", char::from(byte)));
    }
    for text in samples {
        for variant in 0..4 {
            let mut value = NormalizedString::from(text.as_str());
            match variant {
                1 => {
                    value.prepend("X");
                }
                2 => {
                    value.append("Y");
                }
                3 => {
                    value.filter(|c| c != 'a');
                }
                _ => {}
            }
            let mut expected = value.clone();
            let replacement = expected.get().chars().map(|c| (c, 0)).collect::<Vec<_>>();
            expected.transform(replacement, 0);
            if value.convert_offsets(Range::Original(..)) != Some(0..value.get().len()) {
                value.transform(value.get().chars().map(|c| (c, 0)).collect::<Vec<_>>(), 0);
            }
            assert_eq!(value, expected, "{text:?}, variant {variant}");
        }
    }
}
