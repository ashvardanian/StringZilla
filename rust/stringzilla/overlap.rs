//! Window overlap between a batch of prepared queries and any number of candidate batches.
//!
//! File: rust/stringzilla/overlap.rs
//! Author: Ash Vardanian
//! Date: September 24, 2026

use core::ffi::c_void;
use core::mem::MaybeUninit;

use super::*;

/// A forest of prepared query trees, probed by as many batches of candidates as a caller has.
///
/// A window is a fixed-width byte n-gram, and the overlap of two texts at that width is the count
/// of one's windows that occur in the other, over whichever of the two holds more. Every query is
/// hashed and sorted into one B-tree over every width at construction, so the sort is paid once per
/// batch rather than once per round, and nothing is stored per candidate.
///
/// The score is asymmetric: a candidate's window occurrences count against a query's distinct
/// windows, so swapping the two sides changes the answer whenever either repeats a window.
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
/// use stringzilla::sz::{Capabilities, OverlapEngine, Stream, StringsView};
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// let mut engine = OverlapEngine::new(&["the quick brown fox"], &[4, 8], 0, &cpu).unwrap();
///
/// // A `[queries, candidates, widths]` block with the width axis unit-strided.
/// let mut scores = [0.0f32; 2];
/// let candidates = StringsView::from_slices(&["the quick brown cat"]);
/// cpu.scope(|scope| engine.scores(scope, candidates, None, &mut scores, 2, 2))
///     .unwrap();
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
    capability: Capabilities,
    allocator: sz_allocator_t,
    memory: *mut c_void,
    memory_bytes: usize,
    scratch: *mut c_void,
    scratch_bytes: usize,
}

impl OverlapEngine {
    /// Sorts `queries` into one forest with the capabilities of `stream`, like
    /// [`Capabilities::cpu_enabled`] or a GPU's [`Capabilities::cuda_enabled`], fixing the
    /// capability once, on the device `stream` names, which it may join.
    ///
    /// `window_widths` are n-gram widths in bytes and need not form a doubling chain; a width past
    /// a text scores zero for every pair it spans. An empty list of widths is refused here rather
    /// than at the first round.
    /// `candidates_budget` bounds one round for a backend that keeps state per candidate, zero
    /// asking for its default, and is ignored by one that keeps none.
    pub fn new<Query>(
        queries: &[Query],
        window_widths: &[usize],
        candidates_budget: usize,
        stream: &Stream,
    ) -> Result<Self, Status>
    where
        Query: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        unsafe {
            sz_overlap_engine_init(
                engine.as_mut_ptr(),
                &StringsView::from_slices(queries).view,
                window_widths.as_ptr(),
                window_widths.len(),
                candidates_budget,
                stream.capabilities().bits(),
                core::ptr::null(),
                stream.as_raw(),
            )
        }
        .check()?;
        Ok(unsafe { engine.assume_init() })
    }

    /// Queries the forest holds, which is the first axis of every output.
    pub fn queries_count(&self) -> usize {
        self.count
    }

    /// Window widths the batch scores at, which is the last axis of every output.
    pub fn widths_count(&self) -> usize {
        self.widths_count
    }

    /// Queues on the stream of `scope` the window overlap of every prepared query with every
    /// candidate, at every width of the batch.
    ///
    /// `scores` receives a `[queries, candidates, widths]` block: share `[q, c, w]` lands at
    /// `scores[q * scores_query_stride + c * scores_candidate_stride + w]`, each in `[0, 1]`,
    /// with the width axis unit-strided. Both strides count entries rather than bytes, the
    /// candidate stride being at least the width count and the query stride at least the
    /// candidates times that.
    ///
    /// A pair's distance is the windows it misses over its longer side's, summed over every width.
    /// A `radius` of `Some(share)` keeps every pair within that distance exact: a pair past it
    /// scores `0.0` at every width, which lets a round skip it. `None`, or a share of `1.0` or
    /// more, keeps every score exact.
    ///
    /// A GPU engine takes candidates and `scores` in memory its device reaches, such as a
    /// [`Strings`] and a `Vec` in a [`UnifiedAllocator`], and joins the stream before turning its
    /// counts into shares. A stream of another group than the engine's is refused with
    /// [`Status::DeviceMemoryMismatch`].
    pub fn scores<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        candidates: StringsView<'scope>,
        radius: Option<f32>,
        scores: &'scope mut [f32],
        scores_query_stride: usize,
        scores_candidate_stride: usize,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        let widths_count = self.widths_count;
        if scores_candidate_stride < widths_count {
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
                    .and_then(|offset| offset.checked_add(widths_count))
                    .ok_or(Status::OverflowRisk)?
            }
        };
        if scores.len() < span {
            return Err(Status::UnexpectedDimensions);
        }
        unsafe {
            sz_overlap_scores(
                self,
                &candidates.view,
                radius.unwrap_or(1.0),
                scores.as_mut_ptr().cast::<u32>(),
                scores_query_stride,
                scores_candidate_stride,
                stream.as_raw(),
            )
        }
        .check()?;
        if span == 0 {
            return Ok(());
        }
        stream.synchronize()?;

        // Each slot holds its count, which becomes that count over the longer side's windows.
        let widths = unsafe { core::slice::from_raw_parts(self.widths, widths_count) };
        let query_lengths = unsafe { core::slice::from_raw_parts(self.lengths, self.count) };
        let windows = |length: usize, width: u32| match width as usize {
            0 => 0,
            width if width > length => 0,
            width => length - width + 1,
        };
        for (candidate, text) in candidates.iter().enumerate() {
            for (query, &query_length) in query_lengths.iter().enumerate() {
                let first = query * scores_query_stride + candidate * scores_candidate_stride;
                for (share, &width) in scores[first..first + widths_count].iter_mut().zip(widths) {
                    let longer = windows(text.len(), width).max(windows(query_length as usize, width));
                    let matches = share.to_bits();
                    *share = if longer == 0 {
                        0.0
                    } else {
                        (matches as f64 / longer as f64) as f32
                    };
                }
            }
        }
        Ok(())
    }
}

impl Drop for OverlapEngine {
    fn drop(&mut self) {
        unsafe { sz_overlap_engine_free(self as *mut Self, core::ptr::null_mut()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pair is exact while its misses over its longer side's windows stay within the radius, and
    /// scores zero at every width once they pass it.
    #[test]
    fn radius_zeroes_pairs_past_it() {
        let cpu = Stream::default(Capabilities::cpu_enabled());
        let mut engine = OverlapEngine::new(&["abcdefgh"], &[3, 4], 0, &cpu).unwrap();
        // Misses over the longer side's eleven windows at both widths: none, two, and eight.
        let candidates = StringsView::from_slices(&["abcdefgh", "abcdefgx", "abcdxxxx"]);
        let scores_for = |engine: &mut OverlapEngine, radius: Option<f32>| {
            let mut scores = [f32::NAN; 6];
            cpu.scope(|scope| engine.scores(scope, candidates, radius, &mut scores, 6, 2))
                .unwrap();
            scores
        };
        let exact = scores_for(&mut engine, None);
        assert_eq!(exact[..2], [1.0, 1.0]);
        assert!(exact[2] > 0.0 && exact[2] < 1.0 && exact[3] > 0.0 && exact[3] < 1.0);
        assert!(exact[4] > 0.0 && exact[5] > 0.0);
        assert_eq!(scores_for(&mut engine, Some(1.0)), exact);
        assert_eq!(scores_for(&mut engine, Some(8.0 / 11.0)), exact);
        let past_seven = scores_for(&mut engine, Some(7.0 / 11.0));
        assert_eq!(past_seven[..4], exact[..4]);
        assert_eq!(past_seven[4..], [0.0, 0.0]);
        assert_eq!(
            scores_for(&mut engine, Some(2.0 / 11.0)),
            [1.0, 1.0, exact[2], exact[3], 0.0, 0.0]
        );
        assert_eq!(
            scores_for(&mut engine, Some(1.0 / 11.0)),
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(scores_for(&mut engine, Some(0.0)), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
