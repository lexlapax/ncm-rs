//! Encode/decode microprobe over the fixed upstream golden cases.
use ncm_tokenizers::{decode_sudoku, encode_sudoku};
use std::{fs, hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || cfg!(debug_assertions) {
        return Err("usage (release): probe <fixture> <new-report>".into());
    }
    let fixture: serde_json::Value = serde_json::from_slice(&fs::read(&args[0])?)?;
    let cases = fixture["cases"].as_array().ok_or("missing cases")?;
    let inputs: Vec<_> = cases
        .iter()
        .map(|c| {
            (
                c["text"].as_str().unwrap(),
                serde_json::from_value::<Vec<u32>>(c["ids"].clone()).unwrap(),
                c["decoded"].as_str().unwrap(),
            )
        })
        .collect();
    for (text, ids, decoded) in &inputs {
        if encode_sudoku(text)? != *ids || decode_sudoku(ids, false)? != *decoded {
            return Err("oracle mismatch".into());
        }
    }
    let run = || -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..1024 {
            for (text, ids, _) in &inputs {
                black_box(encode_sudoku(black_box(text))?);
                black_box(decode_sudoku(black_box(ids), false)?);
            }
        }
        Ok(())
    };
    for _ in 0..5 {
        run()?;
    }
    let mut trials = Vec::new();
    for _ in 0..30 {
        let start = Instant::now();
        run()?;
        trials.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let report = serde_json::json!({"runtime":"rust-sudoku", "warmups":5, "trials_ms":trials,
        "repetitions":1024, "cases":inputs.len(), "parity_passed":true,
        "scope":"1024 repetitions of all five golden encode+decode pairs; not model tokenization"});
    serde_json::to_writer_pretty(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[1])?,
        &report,
    )?;
    Ok(())
}
