//! Cross-product Levenshtein edit distances over a batch of prepared queries.
//!
//! File: rust/stringzilla/levenshtein.rs
//! Author: Ash Vardanian

use core::ffi::c_void;
use core::mem::MaybeUninit;

use super::*;

/// Which symbols a batch counts, since the alphabet picks the transpose and the mask layout alike.
///
/// Corresponds to `sz_levenshtein_symbol_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevenshteinSymbol {
    /// Every byte is its own symbol, and a distance counts bytes.
    Bytes = 0,
    /// Every UTF-8 rune is one symbol, an ill-formed byte decoding to `U+FFFD`.
    Runes = 1,
}

/// A batch of prepared queries, scored against as many batches of candidates as a caller has.
///
/// Myers' bit-parallel masks are built once per batch and reused by every round, so the preparation
/// a one-shot distance repeats per pair is paid here exactly once. Construction also fixes the
/// capability and, under `new_on`, the device and its launch geometry.
///
/// The fields mirror `sz_levenshtein_engine_t` one for one and only `count` and `symbol` are read
/// from Rust, so the layout is load-bearing and the engine travels to C by pointer. Owning raw
/// pointers makes it neither `Send` nor `Sync`, which is what the C contract wants: a compute
/// verb grows the engine's round scratch, so it mutates, and every verb below takes `&mut self`
/// for that reason.
///
/// # Examples
///
/// ```rust
/// use stringzilla::sz::{LevenshteinEngine, LevenshteinSymbol};
///
/// let mut engine = LevenshteinEngine::new(&["kitten", "saturday"], LevenshteinSymbol::Bytes).unwrap();
///
/// // A `[queries, candidates]` block, query `q` starting at `q * stride`.
/// let mut distances = [0usize; 4];
/// engine.distances(&["sitting", "sunday"], &mut distances, 2).unwrap();
/// assert_eq!(distances[0], 3); // kitten vs sitting
/// assert_eq!(distances[3], 3); // saturday vs sunday
/// ```
#[repr(C)]
#[allow(dead_code)] // Every member is the C side's to write; the layout is what Rust keeps.
pub struct LevenshteinEngine {
    masks: *const u64,
    masks_offsets: *const usize,
    symbol_to_class: *const c_void,
    lengths: *const u32,
    count: usize,
    symbol: LevenshteinSymbol,
    capability: u64,
    ordinal: usize,
    allocator: _SzMemoryAllocator,
    memory: *mut c_void,
    memory_bytes: usize,
    scratch: *mut c_void,
    scratch_bytes: usize,
}

impl LevenshteinEngine {
    /// Prepares `queries` on the host, on the CPU's [`Device::capabilities_enabled`], fixing the
    /// CPU capability once for every round that follows.
    ///
    /// Ice Lake answers a [`LevenshteinSymbol::Runes`] batch through its Skylake kernel, because
    /// Ice Lake's byte lanes have no rune arm.
    pub fn new<Query>(queries: &[Query], symbol: LevenshteinSymbol) -> Result<Self, Status>
    where
        Query: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        with_sequence(queries, |sequence| unsafe {
            sz_levenshtein_engine_init(
                engine.as_mut_ptr(),
                sequence,
                symbol,
                enabled_cpu_capabilities_mask(),
                0,
                core::ptr::null(),
                core::ptr::null_mut(),
            )
        })
        .check()?;
        Ok(unsafe { engine.assume_init() })
    }

    /// Prepares `queries` on `device`, with its [`Device::capabilities_enabled`], fixing the
    /// capability and launch geometry once; the engine keeps the device.
    ///
    /// The queries themselves are read on the host, so they need no device residency; everything
    /// the engine builds from them does.
    ///
    /// `stream` serves this call's own work alone: a `cudaStream_t` or `hipStream_t` of `device`,
    /// or null for its default one, on Metal a `sz_metal_device_t *` opened on `device`, and null
    /// on the CPU. The verbs below pass a null stream and host candidates, so a GPU engine answers
    /// them with an error status such as `Status::DeviceMemoryMismatch` until this crate can build
    /// device-resident candidates.
    ///
    /// # Safety
    ///
    /// `stream` must be a live stream or Metal device of `device`, or null; this call may join it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stringzilla::sz::{Device, LevenshteinEngine, LevenshteinSymbol};
    ///
    /// // SAFETY: the CPU takes a null stream.
    /// let mut engine = unsafe {
    ///     LevenshteinEngine::new_on(&["kitten"], LevenshteinSymbol::Bytes, Device::cpu(), core::ptr::null_mut())
    /// }?;
    /// let mut distances = [0usize; 1];
    /// engine.distances(&["sitting"], &mut distances, 1)?;
    /// assert_eq!(distances[0], 3);
    /// # Ok::<(), stringzilla::sz::Status>(())
    /// ```
    pub unsafe fn new_on<Query>(
        queries: &[Query],
        symbol: LevenshteinSymbol,
        device: Device,
        stream: *mut c_void,
    ) -> Result<Self, Status>
    where
        Query: AsRef<[u8]>,
    {
        let capabilities = device.capabilities_enabled()?;
        let mut engine = MaybeUninit::<Self>::uninit();
        with_sequence(queries, |sequence| unsafe {
            sz_levenshtein_engine_init(
                engine.as_mut_ptr(),
                sequence,
                symbol,
                capabilities.bits(),
                device.ordinal(),
                core::ptr::null(),
                stream,
            )
        })
        .check()?;
        Ok(unsafe { engine.assume_init() })
    }

    /// Queries the batch holds, which is the first axis of every output.
    pub fn queries_count(&self) -> usize {
        self.count
    }

    /// The alphabet the batch was prepared over.
    pub fn symbol(&self) -> LevenshteinSymbol {
        self.symbol
    }

    /// Edit distances from every prepared query to every candidate, on the capability
    /// the constructor fixed.
    ///
    /// `distances` receives a `[queries, candidates]` block: query `q` against candidate `c` lands
    /// at `distances[q * distances_stride + c]`, the stride counting entries rather than bytes and
    /// being at least the candidate count. The round's scratch grows here when a batch needs more
    /// than the last one did, and is never shrunk.
    pub fn distances<Candidate>(
        &mut self,
        candidates: &[Candidate],
        distances: &mut [usize],
        distances_stride: usize,
    ) -> Result<(), Status>
    where
        Candidate: AsRef<[u8]>,
    {
        if distances_stride < candidates.len() {
            return Err(Status::UnexpectedDimensions);
        }
        let span = match self.count {
            0 => 0,
            count => (count - 1)
                .checked_mul(distances_stride)
                .and_then(|rows| rows.checked_add(candidates.len()))
                .ok_or(Status::OverflowRisk)?,
        };
        if distances.len() < span {
            return Err(Status::UnexpectedDimensions);
        }

        let engine = self as *mut Self;
        let output = distances.as_mut_ptr();
        with_sequence(candidates, |sequence| unsafe {
            sz_levenshtein_distances(engine, sequence, output, distances_stride, core::ptr::null_mut())
        })
        .check()
    }
}

impl Drop for LevenshteinEngine {
    fn drop(&mut self) {
        unsafe { sz_levenshtein_engine_free(self as *mut Self) };
    }
}
