//! Cross-product Levenshtein edit distances over a batch of prepared queries.
//!
//! File: rust/stringzilla/levenshtein.rs
//! Author: Ash Vardanian

use super::*;
use core::ffi::c_void;
use core::mem::MaybeUninit;

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
/// a one-shot distance repeats per pair is paid here exactly once. Construction also fixes the ISA
/// tier and, under `new_on_gpu`, the launch geometry.
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
    capability: i32,
    alloc: _SzMemoryAllocator,
    memory: *mut c_void,
    memory_bytes: usize,
    scratch: *mut c_void,
    scratch_bytes: usize,
}

impl LevenshteinEngine {
    /// Prepares `queries` on the host, resolving the CPU tier once for every round that follows.
    ///
    /// Under [`LevenshteinSymbol::Runes`] the batch resolves to Skylake however capable the machine
    /// is, because Ice Lake's byte lanes have no rune arm.
    pub fn new<Query>(queries: &[Query], symbol: LevenshteinSymbol) -> Result<Self, Status>
    where
        Query: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        let status = with_sequence(queries, |sequence| unsafe {
            sz_levenshtein_engine_init_cpu(sequence, symbol, core::ptr::null(), engine.as_mut_ptr())
        });
        match status {
            Status::Success => Ok(unsafe { engine.assume_init() }),
            error => Err(error),
        }
    }

    /// Prepares `queries` on `stream`'s device, resolving the launch geometry once.
    ///
    /// `stream` is a `cudaStream_t`, or null for the current device's default one. The queries
    /// themselves are read on the host, so they need no device residency; everything the engine
    /// builds from them does.
    ///
    /// A compute verb of a device engine also needs a candidate sequence whose accessors run on the
    /// device, which this crate cannot build yet, so such an engine is constructible here before it
    /// is drivable and every verb below answers `Status::DeviceMemoryMismatch` for it.
    ///
    /// # Safety
    ///
    /// `stream` must be a live stream of the current context, and it makes every later verb of this
    /// engine asynchronous: [`LevenshteinEngine::distances`] enqueues and returns, so its output
    /// slice has to be device-reachable, has to outlive the launch, and must not be read before the
    /// caller joins `stream` itself.
    #[cfg(feature = "cuda")]
    pub unsafe fn new_on_gpu<Query>(
        queries: &[Query],
        symbol: LevenshteinSymbol,
        stream: *mut c_void,
    ) -> Result<Self, Status>
    where
        Query: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        let status = with_sequence(queries, |sequence| unsafe {
            sz_levenshtein_engine_init_gpu(sequence, symbol, core::ptr::null(), stream, engine.as_mut_ptr())
        });
        match status {
            Status::Success => Ok(unsafe { engine.assume_init() }),
            error => Err(error),
        }
    }

    /// Queries the batch holds, which is the first axis of every output.
    pub fn queries_count(&self) -> usize {
        self.count
    }

    /// The alphabet the batch was prepared over.
    pub fn symbol(&self) -> LevenshteinSymbol {
        self.symbol
    }

    /// Edit distances from every prepared query to every candidate, on the tier
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
        let status = with_sequence(candidates, |sequence| unsafe {
            sz_levenshtein_distances(engine, sequence, output, distances_stride)
        });
        match status {
            Status::Success => Ok(()),
            error => Err(error),
        }
    }
}

impl Drop for LevenshteinEngine {
    fn drop(&mut self) {
        unsafe { sz_levenshtein_engine_free(self as *mut Self) };
    }
}
