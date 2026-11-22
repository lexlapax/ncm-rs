//! Private scratch storage for copied WMT14 payloads.
use std::path::Path;

use crate::Result;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(super) struct Scratch {
    buf: Vec<u8>,
    len: usize,
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl Scratch {
    pub(super) fn new(capacity: usize) -> Result<Self> {
        let mut buf = Vec::new();
        buf.try_reserve_exact(capacity)
            .map_err(|_| crate::invalid("WMT14 scratch allocation failed"))?;
        Ok(Self { buf, len: 0 })
    }

    pub(super) fn read(&mut self, path: &Path, limit: u64) -> Result<()> {
        crate::bounded_read_into(path, limit, &mut self.buf)?;
        self.len = self.buf.len();
        Ok(())
    }

    #[cfg(unix)]
    pub(super) fn read_at(&mut self, file: &std::fs::File, offset: u64, len: usize) -> Result<()> {
        use std::os::unix::fs::FileExt;
        if len > self.buf.len() {
            if len > self.buf.capacity() {
                self.buf = Vec::new();
                self.buf
                    .try_reserve_exact(len)
                    .map_err(|_| crate::invalid("scratch allocation failed"))?;
            }
            self.buf.resize(len, 0);
        }
        file.read_exact_at(&mut self.buf[..len], offset)?;
        self.len = len;
        Ok(())
    }

    pub(super) fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    pub(super) fn trim(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) use darwin::Scratch;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(unsafe_code)]
mod darwin {
    use super::{Path, Result};
    use std::{
        fs,
        io::{self, Read},
        os::unix::fs::OpenOptionsExt,
        ptr::NonNull,
        slice,
    };

    /// Owns one private anonymous mapping. Artifact bytes are copied from a
    /// checked file. No mapping remains backed by mutable weight files.
    pub(crate) struct Scratch {
        start: NonNull<u8>,
        mapped: usize,
        len: usize,
        page: usize,
    }

    // SAFETY: the mapping is not thread-affine and shares no external state.
    // Mutable access, trimming and unmapping require exclusive ownership.
    unsafe impl Send for Scratch {}
    // SAFETY: shared access exposes only immutable slices tied to the borrow.
    unsafe impl Sync for Scratch {}

    impl Scratch {
        pub(crate) fn new(capacity: usize) -> Result<Self> {
            // SAFETY: sysconf takes the standardized page-size selector.
            let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
                .map_err(|_| crate::invalid("invalid system page size"))?;
            if page == 0 {
                return Err(crate::invalid("invalid system page size"));
            }
            let mapped = capacity
                .max(1)
                .checked_next_multiple_of(page)
                .ok_or_else(|| crate::invalid("WMT14 scratch allocation overflow"))?;
            if mapped > isize::MAX as usize {
                return Err(crate::invalid("WMT14 scratch mapping exceeds slice limit"));
            }
            // SAFETY: private, zero-initialized anonymous pages with no file
            // descriptor or fixed address. The mapping is owned until Drop.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    mapped,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error().into());
            }
            let Some(start) = NonNull::new(ptr.cast::<u8>()) else {
                // SAFETY: mmap returned a valid mapping at address zero.
                unsafe { libc::munmap(ptr, mapped) };
                return Err(crate::invalid("invalid WMT14 scratch address"));
            };
            Ok(Self {
                start,
                mapped,
                len: 0,
                page,
            })
        }

        pub(crate) fn read(&mut self, path: &Path, limit: u64) -> Result<()> {
            let mut file = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(path)?;
            let meta = file.metadata()?;
            if !meta.is_file() || meta.len() != limit {
                return Err(crate::invalid("invalid WMT14 scratch input file"));
            }
            let len = usize::try_from(limit)
                .map_err(|_| crate::invalid("WMT14 scratch input size overflow"))?;
            if len > self.mapped {
                *self = Self::new(len)?;
            }
            self.len = 0;
            // SAFETY: the first len bytes lie in the initialized anonymous
            // mapping, with exclusive access for the duration of the read.
            let target = unsafe { slice::from_raw_parts_mut(self.start.as_ptr(), len) };
            file.read_exact(target)?;
            if file.read(&mut [0u8; 1])? != 0 {
                return Err(crate::invalid("WMT14 scratch input file grew"));
            }
            self.len = len;
            Ok(())
        }

        pub(crate) fn read_at(&mut self, file: &fs::File, offset: u64, len: usize) -> Result<()> {
            use std::os::unix::fs::FileExt;
            if len > self.mapped {
                *self = Self::new(len)?;
            }
            self.len = 0;
            // SAFETY: the first len bytes lie in the initialized anonymous
            // mapping, with exclusive access for the duration of the read.
            let target = unsafe { slice::from_raw_parts_mut(self.start.as_ptr(), len) };
            file.read_exact_at(target, offset)?;
            self.len = len;
            Ok(())
        }

        pub(crate) fn bytes(&self) -> &[u8] {
            // SAFETY: the published prefix is initialized and lives for self.
            unsafe { slice::from_raw_parts(self.start.as_ptr(), self.len) }
        }

        pub(crate) fn trim(&mut self) -> io::Result<()> {
            let keep = self.len.max(1).next_multiple_of(self.page);
            if keep < self.mapped {
                // SAFETY: both boundaries are page-aligned. The released tail
                // is wholly within the mapping and disjoint from live bytes.
                let result = unsafe {
                    libc::munmap(self.start.as_ptr().add(keep).cast(), self.mapped - keep)
                };
                if result != 0 {
                    return Err(io::Error::last_os_error());
                }
                self.mapped = keep;
            }
            Ok(())
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // SAFETY: this is the remaining owned mapping, possibly trimmed.
            unsafe { libc::munmap(self.start.as_ptr().cast(), self.mapped) };
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Scratch;
        use std::fs;

        #[test]
        fn trimming_keeps_live_bytes_and_allows_a_later_larger_read() {
            let mut storage = Scratch::new(1).unwrap();
            let page = storage.page;
            let large = vec![0x2a; page * 3 + 1];
            let small = vec![0x42; page - 1];
            let dir = tempfile::tempdir().unwrap();
            let large_path = dir.path().join("large");
            let small_path = dir.path().join("small");
            fs::write(&large_path, &large).unwrap();
            fs::write(&small_path, &small).unwrap();
            storage.read(&large_path, large.len() as u64).unwrap();
            assert_eq!(storage.bytes(), large);
            storage.read(&small_path, small.len() as u64).unwrap();
            let start = storage.bytes().as_ptr();
            storage.trim().unwrap();
            assert_eq!(storage.mapped, page);
            assert_eq!(storage.bytes().as_ptr(), start);
            assert_eq!(storage.bytes(), small);
            storage.read(&large_path, large.len() as u64).unwrap();
            assert_eq!(storage.bytes(), large);
        }
    }
}
