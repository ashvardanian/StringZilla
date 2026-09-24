//! Window overlap between a batch of prepared queries and any number of candidate batches.

use super::*;
use core::ffi::c_void;
use core::mem::MaybeUninit;

/// A forest of prepared query trees, probed by as many batches of candidates as a caller has.
///
/// A window is a fixed-width byte n-gram, and the overlap of two texts at that width is the count of
/// one's windows that occur in the other, over whichever of the two holds more. Every query is hashed
/// and sorted into one B-tree over every width at construction, so the sort is paid once per batch
/// rather than once per round, and nothing is stored per candidate.
///
/// The score is asymmetric: a candidate's window occurrences count against a query's distinct windows,
/// so swapping the two sides changes the answer whenever either repeats a window.
///
/// The fields mirror `sz_overlap_engine_t` one for one and only `count` and `widths_count` are read
/// from Rust, so the layout is load-bearing and the engine travels to C by pointer. Owning raw
/// pointers makes it neither `Send` nor `Sync`, which is what the C contract wants: a compute verb
/// grows the engine's round scratch, so it mutates, and [`OverlapEngine::scores`] takes `&mut self`
/// for that reason.
///
/// # Examples
///
/// ```rust
/// use stringzilla::sz::OverlapEngine;
///
/// let mut engine = OverlapEngine::new(&["the quick brown fox"], &[4, 8]).unwrap();
///
/// // A `[queries, candidates, widths]` block with the width axis unit-strided.
/// let mut scores = [0.0f32; 2];
/// engine.scores(&["the quick brown cat"], &mut scores, 2, 2).unwrap();
/// assert!(scores.iter().all(|share| (0.0..=1.0).contains(share)));
/// ```
#[repr(C)]
#[allow(dead_code)] // Every member is the C side's to write; the layout is what Rust keeps.
pub struct OverlapEngine {
    nodes: *const u32,
    nodes_offsets: *const usize,
    keys_counts: *const u32,
    widths: *const u32,
    powers: *const u32,
    lengths: *const u32,
    count: usize,
    widths_count: usize,
    capability: i32,
    alloc: _SzMemoryAllocator,
    memory: *mut c_void,
    memory_bytes: usize,
    scratch: *mut c_void,
    scratch_bytes: usize,
}

impl OverlapEngine {
    /// Sorts `queries` into one forest on the host, resolving the CPU tier once.
    ///
    /// `window_widths` are n-gram widths in bytes and need not form a doubling chain; a width past a
    /// text scores zero for every pair it spans. Zero widths are refused.
    pub fn new<Query>(queries: &[Query], window_widths: &[usize]) -> Result<Self, Status>
    where
        Query: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        let status = with_sequence(queries, |sequence| unsafe {
            sz_overlap_engine_init_cpu(
                sequence,
                window_widths.as_ptr(),
                window_widths.len(),
                core::ptr::null(),
                engine.as_mut_ptr(),
            )
        });
        match status {
            Status::Success => Ok(unsafe { engine.assume_init() }),
            error => Err(error),
        }
    }

    /// Sorts `queries` into one forest on `stream`'s device, resolving the launch geometry once.
    ///
    /// `stream` is a `cudaStream_t`, or null for the current device's default one. A width the device
    /// backend's per-thread ring cannot hold is refused here rather than at the first round.
    ///
    /// A compute verb of a device engine also needs a candidate sequence whose accessors run on the
    /// device, which this crate cannot build yet, so such an engine is constructible here before it
    /// is drivable and every verb below answers `Status::DeviceMemoryMismatch` for it.
    ///
    /// # Safety
    ///
    /// `stream` must be a live stream of the current context, and it makes every later verb of this
    /// engine asynchronous: [`OverlapEngine::scores`] enqueues and returns, so its output slice has to
    /// be device-reachable, has to outlive the launch, and must not be read before the caller joins
    /// `stream` itself.
    #[cfg(feature = "cuda")]
    pub unsafe fn new_on_gpu<Query>(
        queries: &[Query],
        window_widths: &[usize],
        stream: *mut c_void,
    ) -> Result<Self, Status>
    where
        Query: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        let status = with_sequence(queries, |sequence| unsafe {
            sz_overlap_engine_init_gpu(
                sequence,
                window_widths.as_ptr(),
                window_widths.len(),
                core::ptr::null(),
                stream,
                engine.as_mut_ptr(),
            )
        });
        match status {
            Status::Success => Ok(unsafe { engine.assume_init() }),
            error => Err(error),
        }
    }

    /// Queries the forest holds, which is the first axis of every output.
    pub fn queries_count(&self) -> usize {
        self.count
    }

    /// Window widths the batch scores at, which is the last axis of every output.
    pub fn widths_count(&self) -> usize {
        self.widths_count
    }

    /// Window overlap of every prepared query with every candidate, at every width of the batch.
    ///
    /// `scores` receives a `[queries, candidates, widths]` block: share `[q, c, w]` lands at
    /// `scores[q * scores_query_stride + c * scores_candidate_stride + w]`, each in `[0, 1]`, with the
    /// width axis unit-strided. Both strides count entries rather than bytes, the candidate stride
    /// being at least the width count and the query stride at least the candidates times that.
    pub fn scores<Candidate>(
        &mut self,
        candidates: &[Candidate],
        scores: &mut [f32],
        scores_query_stride: usize,
        scores_candidate_stride: usize,
    ) -> Result<(), Status>
    where
        Candidate: AsRef<[u8]>,
    {
        if scores_candidate_stride < self.widths_count {
            return Err(Status::UnexpectedDimensions);
        }
        let candidates_span = candidates
            .len()
            .checked_mul(scores_candidate_stride)
            .ok_or(Status::OverflowRisk)?;
        if scores_query_stride < candidates_span {
            return Err(Status::UnexpectedDimensions);
        }
        let span = match (self.count, candidates.len()) {
            (0, _) | (_, 0) => 0,
            (queries, candidates_count) => {
                let planes = (queries - 1)
                    .checked_mul(scores_query_stride)
                    .ok_or(Status::OverflowRisk)?;
                let rows = (candidates_count - 1)
                    .checked_mul(scores_candidate_stride)
                    .ok_or(Status::OverflowRisk)?;
                planes
                    .checked_add(rows)
                    .and_then(|offset| offset.checked_add(self.widths_count))
                    .ok_or(Status::OverflowRisk)?
            }
        };
        if scores.len() < span {
            return Err(Status::UnexpectedDimensions);
        }

        let engine = self as *mut Self;
        let output = scores.as_mut_ptr();
        let status = with_sequence(candidates, |sequence| unsafe {
            sz_overlap_scores(
                engine,
                sequence,
                output,
                scores_query_stride,
                scores_candidate_stride,
            )
        });
        match status {
            Status::Success => Ok(()),
            error => Err(error),
        }
    }
}

impl Drop for OverlapEngine {
    fn drop(&mut self) {
        unsafe { sz_overlap_engine_free(self as *mut Self) };
    }
}
