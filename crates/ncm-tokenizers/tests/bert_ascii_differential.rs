//! Compare the pinned BERT normalizer's ASCII path with its original steps.
use tokenizers::{NormalizedString, Normalizer, normalizers::bert::BertNormalizer};

// This reference is restricted to ASCII normalized inputs, where no character
// is CJK or a nonspacing mark. It retains the original sequence of operations.
fn reference(n: &BertNormalizer, s: &mut NormalizedString) {
    if n.clean_text {
        s.filter(|c| !(c == '\0' || (c.is_control() && !matches!(c, '\t' | '\n' | '\r'))))
            .map(|c| if c.is_whitespace() { ' ' } else { c });
    }
    if n.handle_chinese_chars {
        let mut chars = Vec::new();
        s.for_each(|c| chars.push((c, 0)));
        s.transform(chars, 0);
    }
    if n.strip_accents.unwrap_or(n.lowercase) {
        s.nfd().filter(|_| true);
    }
    if n.lowercase {
        s.lowercase();
    }
}

#[test]
fn ascii_fast_path_preserves_values_and_alignments() {
    let mut inputs: Vec<String> = (0u8..=127).map(|b| (b as char).to_string()).collect();
    inputs.extend(
        ["", "Hello, world!", "already lower 123", "A\tB", "éABC"]
            .into_iter()
            .map(String::from),
    );
    for seed in 0..128_usize {
        inputs.push(
            (0..31)
                .map(|i| ((seed * 37 + i * 17) % 128) as u8 as char)
                .collect(),
        );
    }
    for clean in [false, true] {
        for chinese in [false, true] {
            for strip in [None, Some(false), Some(true)] {
                for lower in [false, true] {
                    let n = BertNormalizer::new(clean, chinese, strip, lower);
                    for text in &inputs {
                        for pretransform in [false, true] {
                            let mut actual = NormalizedString::from(text.as_str());
                            if pretransform {
                                actual.nfd().filter(|c| c.is_ascii());
                            }
                            if !actual.get().is_ascii() {
                                continue;
                            }
                            let mut expected = actual.clone();
                            reference(&n, &mut expected);
                            n.normalize(&mut actual).unwrap();
                            assert_eq!(
                                actual, expected,
                                "{n:?} input={text:?} pretransform={pretransform}"
                            );
                        }
                    }
                }
            }
        }
    }
}
