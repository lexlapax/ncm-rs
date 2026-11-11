//! Diagnostic-only Rust allocation request counters, never wall-time evidence.
//! No allocator behavior is changed: every operation delegates to System.
//! Native-library allocations, allocator overhead and physical RSS are excluded.
#![allow(unsafe_code)]

use serde::Serialize;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
    },
};

pub struct Counting;
static ENABLED: AtomicBool = AtomicBool::new(false);
static SCOPE: Mutex<()> = Mutex::new(());
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static DEALLOCS: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED: AtomicU64 = AtomicU64::new(0);

// SAFETY: layouts/pointers are forwarded unchanged to System. Counters use only
// atomics, never allocate or unwind, and no pointer is read or written here.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller satisfies GlobalAlloc's layout contract.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() && ENABLED.load(Relaxed) {
            ALLOCS.fetch_add(1, Relaxed);
            ALLOCATED.fetch_add(layout.size() as u64, Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller satisfies GlobalAlloc's layout contract.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() && ENABLED.load(Relaxed) {
            ALLOCS.fetch_add(1, Relaxed);
            ALLOCATED.fetch_add(layout.size() as u64, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ENABLED.load(Relaxed) {
            DEALLOCS.fetch_add(1, Relaxed);
            DEALLOCATED.fetch_add(layout.size() as u64, Relaxed);
        }
        // SAFETY: caller supplies the live allocation and its original layout.
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller satisfies GlobalAlloc's pointer/layout/new-size contract.
        let p = unsafe { System.realloc(ptr, layout, size) };
        if !p.is_null() && ENABLED.load(Relaxed) {
            REALLOCS.fetch_add(1, Relaxed);
            ALLOCATED.fetch_add(size as u64, Relaxed);
            DEALLOCATED.fetch_add(layout.size() as u64, Relaxed);
        }
        p
    }
}

#[derive(Debug, Serialize)]
pub struct Counts {
    allocations: u64,
    deallocations: u64,
    reallocations: u64,
    requested_allocated_bytes: u64,
    requested_deallocated_bytes: u64,
}
struct Recording;
impl Drop for Recording {
    fn drop(&mut self) {
        ENABLED.store(false, Relaxed);
    }
}
/// Serialize measured scopes; run a single-threaded diagnostic with parallel
/// tokenization disabled. All Rust threads' calls during the region count.
/// Return values remain owned by the caller: their destruction is not included.
/// Realloc counts its requested new size and released old size, not internal
/// malloc copying or physical peak. No live/peak-memory claim follows from this.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Counts) {
    let _scope = SCOPE.lock().unwrap_or_else(|e| e.into_inner());
    for counter in [&ALLOCS, &DEALLOCS, &REALLOCS, &ALLOCATED, &DEALLOCATED] {
        counter.store(0, Relaxed);
    }
    ENABLED.store(true, Relaxed);
    let recording = Recording;
    let value = f();
    drop(recording);
    let counts = Counts {
        allocations: ALLOCS.load(Relaxed),
        deallocations: DEALLOCS.load(Relaxed),
        reallocations: REALLOCS.load(Relaxed),
        requested_allocated_bytes: ALLOCATED.load(Relaxed),
        requested_deallocated_bytes: DEALLOCATED.load(Relaxed),
    };
    (value, counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_reallocation_and_contents_are_preserved() {
        let (mut bytes, before) = measure(|| std::hint::black_box(vec![42u8; 32]));
        assert!(before.allocations >= 1);
        assert!(before.requested_allocated_bytes >= 32);
        let (_, grown) = measure(|| bytes.reserve_exact(4096));
        assert!(grown.reallocations >= 1);
        assert!(grown.requested_allocated_bytes >= 4128);
        assert!(grown.requested_deallocated_bytes >= 32);
        assert_eq!(bytes, [42; 32]);
        let (_, freed) = measure(|| drop(bytes));
        assert!(freed.deallocations >= 1);
        assert!(freed.requested_deallocated_bytes >= 4128);
    }
    #[test]
    fn unwinding_disables_recording() {
        let _ = std::panic::catch_unwind(|| measure(|| panic!("diagnostic test")));
        // Acquire the same scope lock to avoid another test starting a measurement.
        let _scope = SCOPE.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!ENABLED.load(Relaxed));
    }
}
