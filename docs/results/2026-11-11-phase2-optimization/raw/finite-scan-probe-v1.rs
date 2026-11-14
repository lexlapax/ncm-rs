use std::{fs, hint::black_box, time::Instant};
fn original(data: &[u8]) -> bool { data.as_chunks::<4>().0.iter().any(|b| !f32::from_le_bytes(*b).is_finite()) }
fn unrolled(data: &[u8]) -> bool {
    let (chunks, tail) = data.as_chunks::<64>();
    for chunk in chunks {
        let mut bad = false;
        for word in chunk.as_chunks::<4>().0 { bad |= u32::from_le_bytes(*word) & 0x7f80_0000 == 0x7f80_0000; }
        if bad { return true; }
    }
    tail.as_chunks::<4>().0.iter().any(|word| u32::from_le_bytes(*word) & 0x7f80_0000 == 0x7f80_0000)
}
fn main() {
 let bytes=fs::read("artifacts/canonical/sudoku-easy-qualified/codec.safetensors").unwrap();
 let start=8+u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize; let bytes=&bytes[start..]; assert_eq!(bytes.len()%4,0); assert_eq!(original(bytes),unrolled(bytes));
 let mut old=Vec::new(); let mut new=Vec::new();
 for i in 0..32 { for kind in if i%2==0 {[0,1]} else {[1,0]} { let t=Instant::now(); let bad=if kind==0 {original(black_box(bytes))} else {unrolled(black_box(bytes))}; black_box(bad); let ms=t.elapsed().as_secs_f64()*1e3; if i>=2 {if kind==0 {old.push(ms)} else {new.push(ms)}} } }
 old.sort_by(|a,b|a.total_cmp(b)); new.sort_by(|a,b|a.total_cmp(b)); println!("bytes={} old_ms={:.6} candidate_ms={:.6} original_nonfinite={}",bytes.len(),old[15],new[15],original(bytes));
}
