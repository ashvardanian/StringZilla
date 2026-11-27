//! Cross-product Levenshtein edit distances over a batch of prepared queries.
//!
//! File: rust/stringzilla/levenshtein.rs
//! Author: Ash Vardanian
//! Date: September 24, 2026

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
    /// Every UTF-8 rune is one symbol, each maximal ill-formed subpart decoding to one `U+FFFD`.
    Runes = 1,
}

/// A batch of prepared queries, scored against as many batches of candidates as a caller has.
///
/// Myers' bit-parallel masks are built once per batch and reused by every round, so the preparation
/// a one-shot distance repeats per pair is paid here exactly once. Construction also fixes the
/// capability and, on a GPU, the launch geometry.
///
/// The fields mirror `sz_levenshtein_engine_t` one for one and only `count`, `symbol` and
/// `capability` are read from Rust, so the layout is load-bearing and the engine travels to C by
/// pointer. Owning raw pointers makes it neither `Send` nor `Sync`, which is what the C contract
/// wants: a compute verb grows the engine's round scratch, so it mutates, and every verb below
/// takes `&mut self` for that reason.
///
/// # Examples
///
/// ```rust
/// use stringzilla::{Capabilities, LevenshteinEngine, LevenshteinSymbol, MatrixSpan, Stream, StringsView};
///
/// let cpu = Stream::default(Capabilities::enabled_cpu());
/// let mut engine = LevenshteinEngine::new(&["kitten", "saturday"], LevenshteinSymbol::Bytes, &cpu).unwrap();
///
/// // A `[queries, candidates]` block.
/// let mut storage = [0usize; 4];
/// let mut distances = MatrixSpan::from_slice(&mut storage, [2, 2]).unwrap();
/// let candidates = StringsView::from_slices(&["sitting", "sunday"]);
/// cpu.scope(|scope| engine.distances(scope, candidates, None, &mut distances)).unwrap();
/// assert_eq!(storage, [3, 6, 6, 3]); // kitten vs sitting, saturday vs sunday
///
/// // A candidate past a bound of 1 reports exactly 2.
/// let mut distances = MatrixSpan::from_slice(&mut storage, [2, 2]).unwrap();
/// cpu.scope(|scope| engine.distances(scope, candidates, Some(1), &mut distances)).unwrap();
/// assert_eq!(storage, [2, 2, 2, 2]);
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
    capability: Capabilities,
    allocator: sz_allocator_t,
    memory: *mut c_void,
    memory_bytes: usize,
    scratch: *mut c_void,
    scratch_bytes: usize,
}

impl LevenshteinEngine {
    /// Prepares `queries` with the capabilities of `stream`, like [`Capabilities::enabled_cpu`] or
    /// a GPU's [`Capabilities::enabled_cuda`], fixing the capability and, on a GPU, the launch
    /// geometry once for every round that follows, on the device `stream` names, which it may join.
    ///
    /// The queries themselves are read on the host, so they need no device residency; everything
    /// the engine builds from them does. Ice Lake answers a [`LevenshteinSymbol::Runes`] batch
    /// through its Skylake kernel, because Ice Lake's byte lanes have no rune arm.
    pub fn new<Query>(queries: &[Query], symbol: LevenshteinSymbol, stream: &Stream) -> Result<Self, Status>
    where
        Query: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        unsafe {
            sz_levenshtein_engine_init(
                engine.as_mut_ptr(),
                &StringsView::from_slices(queries).view,
                symbol,
                stream.capabilities().bits(),
                core::ptr::null(),
                stream.as_raw(),
            )
        }
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

    /// Queues on the stream of `scope` the edit distances from every prepared query to every
    /// candidate, on the capability the constructor fixed.
    ///
    /// `distances` is a `[queries, candidates]` block, query `q` against candidate `c` landing at
    /// its entry `[q, c]`, and any other shape is refused with [`Status::UnexpectedDimensions`].
    /// The round's scratch grows here when a batch needs more than the last one did, and is never
    /// shrunk.
    ///
    /// A `bound` of `Some(limit)` keeps a distance exact up to `limit` and reports `limit + 1` for
    /// every candidate past it, on every backend, which lets a round stop early. `None` keeps every
    /// distance exact.
    ///
    /// A stream of another group than the engine's is refused with
    /// [`Status::DeviceMemoryMismatch`].
    pub fn distances<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        candidates: StringsView<'scope>,
        bound: Option<usize>,
        distances: &'scope mut MatrixSpan<'_, usize>,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        if distances.shape() != [self.count, candidates.len()] {
            return Err(Status::UnexpectedDimensions);
        }
        unsafe {
            sz_levenshtein_distances(
                self,
                &candidates.view,
                core::ptr::null(),
                core::ptr::null(),
                bound.unwrap_or(usize::MAX),
                distances.as_mut_ptr(),
                distances.stride_bytes(0) as usize,
                stream.as_raw(),
            )
        }
        .check()
    }
}

impl Drop for LevenshteinEngine {
    fn drop(&mut self) {
        unsafe { sz_levenshtein_engine_free(self as *mut Self, core::ptr::null_mut()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A candidate within the bound or at it is exact, and one past it reports the bound plus one.
    #[test]
    fn bound_clips_distances_past_it() {
        let cpu = Stream::default(Capabilities::enabled_cpu());
        let mut engine = LevenshteinEngine::new(&["kitten"], LevenshteinSymbol::Bytes, &cpu).unwrap();
        let candidates = StringsView::from_slices(&["kitten", "mitten", "mittens", "sitting"]);
        let distances_for = |engine: &mut LevenshteinEngine, bound: Option<usize>| {
            let mut storage = [0usize; 4];
            let mut distances = MatrixSpan::from_slice(&mut storage, [1, 4]).unwrap();
            cpu.scope(|scope| engine.distances(scope, candidates, bound, &mut distances)).unwrap();
            storage
        };
        let exact = [0, 1, 2, 3];
        assert_eq!(distances_for(&mut engine, None), exact);
        assert_eq!(distances_for(&mut engine, Some(usize::MAX)), exact);
        assert_eq!(distances_for(&mut engine, Some(3)), exact);
        assert_eq!(distances_for(&mut engine, Some(2)), exact);
        assert_eq!(distances_for(&mut engine, Some(1)), [0, 1, 2, 2]);
        assert_eq!(distances_for(&mut engine, Some(0)), [0, 1, 1, 1]);
    }

    /// A window into a wider block fills its own entries and leaves the rest untouched.
    #[test]
    fn distances_fill_a_window_of_a_wider_block() {
        let cpu = Stream::default(Capabilities::enabled_cpu());
        let mut engine = LevenshteinEngine::new(&["kitten", "saturday"], LevenshteinSymbol::Bytes, &cpu).unwrap();
        let candidates = StringsView::from_slices(&["sitting", "sunday"]);
        let entry_bytes = size_of::<usize>() as isize;
        let mut storage = [usize::MAX; 6];
        let mut distances =
            MatrixSpan::from_slice_strided(&mut storage, [2, 2], [3 * entry_bytes, entry_bytes]).unwrap();
        cpu.scope(|scope| engine.distances(scope, candidates, None, &mut distances)).unwrap();
        assert_eq!(storage, [3, 6, usize::MAX, 6, 3, usize::MAX]);
    }
}
