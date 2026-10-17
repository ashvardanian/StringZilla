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
/// use stringzilla::sz::{Capabilities, LevenshteinEngine, LevenshteinSymbol, Stream, StringsView};
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// let mut engine = LevenshteinEngine::new(&["kitten", "saturday"], LevenshteinSymbol::Bytes, &cpu).unwrap();
///
/// // A `[queries, candidates]` block, query `q` starting at `q * stride`.
/// let mut distances = [0usize; 4];
/// let candidates = StringsView::from_slices(&["sitting", "sunday"]);
/// cpu.scope(|scope| engine.distances(scope, candidates, None, &mut distances, 2))
///     .unwrap();
/// assert_eq!(distances[0], 3); // kitten vs sitting
/// assert_eq!(distances[3], 3); // saturday vs sunday
///
/// // A candidate past a bound of 1 reports exactly 2.
/// let candidates = StringsView::from_slices(&["sitting", "sunday"]);
/// cpu.scope(|scope| engine.distances(scope, candidates, Some(1), &mut distances, 2))
///     .unwrap();
/// assert_eq!(distances[0], 2);
/// assert_eq!(distances[3], 2);
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
    /// Prepares `queries` with the capabilities of `stream`, like [`Capabilities::cpu_enabled`] or
    /// a GPU's [`Capabilities::cuda_enabled`], fixing the capability and, on a GPU, the launch
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
    /// `distances` receives a `[queries, candidates]` block: query `q` against candidate `c` lands
    /// at `distances[q * distances_stride + c]`, the stride counting entries rather than bytes and
    /// being at least the candidate count. The round's scratch grows here when a batch needs more
    /// than the last one did, and is never shrunk.
    ///
    /// A `bound` of `Some(limit)` keeps a distance exact up to `limit` and reports `limit + 1` for
    /// every candidate past it, on every backend, which lets a round stop early. `None` keeps every
    /// distance exact.
    ///
    /// A GPU engine takes candidates and `distances` in memory its device reaches, such as a
    /// [`Strings`] and a `Vec` in a [`UnifiedAllocator`]. A stream of another group than the
    /// engine's is refused with [`Status::DeviceMemoryMismatch`].
    pub fn distances<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        candidates: StringsView<'scope>,
        bound: Option<usize>,
        distances: &'scope mut [usize],
        distances_stride: usize,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
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
        unsafe {
            sz_levenshtein_distances(
                self,
                &candidates.view,
                core::ptr::null(),
                core::ptr::null(),
                bound.unwrap_or(usize::MAX),
                distances.as_mut_ptr(),
                distances_stride,
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
        let cpu = Stream::default(Capabilities::cpu_enabled());
        let mut engine = LevenshteinEngine::new(&["kitten"], LevenshteinSymbol::Bytes, &cpu).unwrap();
        let candidates = StringsView::from_slices(&["kitten", "mitten", "mittens", "sitting"]);
        let distances_for = |engine: &mut LevenshteinEngine, bound: Option<usize>| {
            let mut distances = [0usize; 4];
            cpu.scope(|scope| engine.distances(scope, candidates, bound, &mut distances, 4))
                .unwrap();
            distances
        };
        let exact = [0, 1, 2, 3];
        assert_eq!(distances_for(&mut engine, None), exact);
        assert_eq!(distances_for(&mut engine, Some(usize::MAX)), exact);
        assert_eq!(distances_for(&mut engine, Some(3)), exact);
        assert_eq!(distances_for(&mut engine, Some(2)), exact);
        assert_eq!(distances_for(&mut engine, Some(1)), [0, 1, 2, 2]);
        assert_eq!(distances_for(&mut engine, Some(0)), [0, 1, 1, 1]);
    }
}
