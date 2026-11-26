//! N-gram overlaps between a batch of prepared queries and any number of candidate batches, and
//! MinHash sketches of any texts over the same N-grams.
//!
//! File: rust/stringzilla/ngrams.rs
//! Author: Ash Vardanian
//! Date: September 24, 2026

use core::ffi::c_void;
use core::mem::MaybeUninit;

use super::*;

/// Scores how much of each prepared query every candidate holds, and summarizes any texts into
/// MinHash sketches, both over the N-grams of the engine's widths.
///
/// An N-gram is a run of a fixed number of bytes, and the engine works at one or more widths at
/// once. Every query's distinct N-grams are hashed into one table at construction, so that is paid
/// once per batch rather than once per round, and nothing is stored per candidate. An engine of no
/// queries only sketches.
///
/// A GPU engine's verbs take every input and output in memory its device reaches, such as a
/// [`Strings`] and a `Vec` in a [`UnifiedAllocator`].
///
/// # Examples
///
/// ```rust
/// use stringzilla::{Capabilities, NgramsEngine, Stream, StringsView};
///
/// let cpu = Stream::default(Capabilities::enabled_cpu());
/// let mut engine = NgramsEngine::new(&["the quick brown fox"], &[4, 8], 0, &cpu).unwrap();
///
/// // A `[queries, candidates, widths]` block with the width axis unit-strided.
/// let mut overlaps = [0.0f32; 2];
/// let candidates = StringsView::from_slices(&["the quick brown cat"]);
/// cpu.scope(|scope| engine.overlaps(scope, candidates, None, &mut overlaps, 2, 2))
///     .unwrap();
/// assert!(overlaps.iter().all(|share| (0.0..=1.0).contains(share)));
///
/// // Two equal texts sketch alike.
/// let mut sketches = [0u8; 2 * 256];
/// let texts = StringsView::from_slices(&["hello world", "hello world"]);
/// cpu.scope(|scope| engine.sketches(scope, texts, &mut sketches, 256, 256)).unwrap();
/// assert_eq!(sketches[..256], sketches[256..]);
/// ```
// The fields mirror `sz_ngrams_engine_t` one for one and only the counts and `distinct_counts` are
// read from Rust, so the layout is load-bearing and the engine travels to C by pointer. Owning raw
// pointers makes it neither `Send` nor `Sync`, which is what the C contract wants: a compute verb
// grows the engine's scratch, so it mutates, and both verbs take `&mut self` for that reason.
#[repr(C)]
#[allow(dead_code)] // Every member is the C side's to write; the layout is what Rust keeps.
pub struct NgramsEngine {
    slots: *const u64,
    slots_offsets: *const usize,
    filters: *const u32,
    distinct_counts: *const u32,
    widths: *const usize,
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

impl NgramsEngine {
    /// Files `queries` into their tables with the capabilities of `stream`, like
    /// [`Capabilities::enabled_cpu`] or a GPU's [`Capabilities::enabled_cuda`], fixing the
    /// capability once, on the device `stream` names, which it may join.
    ///
    /// `window_widths` are N-gram widths in bytes and need not form a doubling chain; a width past
    /// a text scores zero for every pair it spans. An empty list of widths is refused here rather
    /// than at the first round, while an empty batch of `queries` makes an engine that only
    /// sketches. `candidates_budget` bounds one round for a backend that keeps state per
    /// candidate, zero asking for its default, and is ignored by one that keeps none.
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
            sz_ngrams_engine_init(
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

    /// Queries the engine holds, which is the first axis of every overlap output.
    pub fn queries_count(&self) -> usize {
        self.count
    }

    /// Window widths the batch scores and sketches at, which is the last axis of every overlap
    /// output.
    pub fn widths_count(&self) -> usize {
        self.widths_count
    }

    /// Queues on the stream of `scope` the overlap of every prepared query with every candidate:
    /// the share of the query's distinct N-grams that the candidate holds, per width, in `[0, 1]`.
    ///
    /// The score is asymmetric, so a candidate holding the whole query scores `1.0` however much
    /// more it holds. Share `[q, c, w]` lands at
    /// `overlaps[q * overlaps_query_stride + c * overlaps_candidate_stride + w]`; the candidate
    /// stride is at least the width count, and the query stride at least the candidates times that.
    ///
    /// A pair's distance is its misses over the query's distinct N-grams, summed over every width.
    /// A `radius` of `Some(share)` keeps every pair within that distance exact: a pair past it
    /// scores `0.0` at every width, which lets a round skip it. `None`, or a share of `1.0` or
    /// more, keeps every score exact.
    ///
    /// A GPU engine joins the stream before turning its counts into shares. A stream of another
    /// group than the engine's is refused with [`Status::DeviceMemoryMismatch`].
    pub fn overlaps<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        candidates: StringsView<'scope>,
        radius: Option<f32>,
        overlaps: &'scope mut [f32],
        overlaps_query_stride: usize,
        overlaps_candidate_stride: usize,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        let widths_count = self.widths_count;
        if overlaps_candidate_stride < widths_count {
            return Err(Status::UnexpectedDimensions);
        }
        let candidates_span = candidates.len().checked_mul(overlaps_candidate_stride).ok_or(Status::OverflowRisk)?;
        if overlaps_query_stride < candidates_span {
            return Err(Status::UnexpectedDimensions);
        }
        let span = match (self.count, candidates.len()) {
            (0, _) | (_, 0) => 0,
            (queries, candidates_count) => {
                let planes = (queries - 1).checked_mul(overlaps_query_stride).ok_or(Status::OverflowRisk)?;
                let rows = (candidates_count - 1).checked_mul(overlaps_candidate_stride).ok_or(Status::OverflowRisk)?;
                planes
                    .checked_add(rows)
                    .and_then(|offset| offset.checked_add(widths_count))
                    .ok_or(Status::OverflowRisk)?
            }
        };
        if overlaps.len() < span {
            return Err(Status::UnexpectedDimensions);
        }
        unsafe {
            sz_ngrams_overlaps(
                self,
                &candidates.view,
                radius.unwrap_or(1.0),
                overlaps.as_mut_ptr().cast::<u32>(),
                overlaps_query_stride,
                overlaps_candidate_stride,
                stream.as_raw(),
            )
        }
        .check()?;
        if span == 0 {
            return Ok(());
        }
        stream.synchronize()?;

        // Each slot holds its count, which becomes that count over the query's distinct N-grams.
        let distinct_counts = unsafe { core::slice::from_raw_parts(self.distinct_counts, self.count * widths_count) };
        for candidate in 0..candidates.len() {
            for (query, distinct) in distinct_counts.chunks_exact(widths_count).enumerate() {
                let first = query * overlaps_query_stride + candidate * overlaps_candidate_stride;
                for (share, &distinct) in overlaps[first..first + widths_count].iter_mut().zip(distinct) {
                    let matches = share.to_bits();
                    *share = if distinct == 0 { 0.0 } else { (matches as f64 / distinct as f64) as f32 };
                }
            }
        }
        Ok(())
    }

    /// Queues on the stream of `scope` one MinHash sketch of `slots` bytes per text of `texts`,
    /// summarizing the set of its N-grams at the engine's widths.
    ///
    /// The share of positions where two sketches differ estimates one minus the Jaccard similarity
    /// of their N-gram sets, and more slots give a finer estimate. Only sketches of one slot count
    /// compare. Sketch `i` starts at `sketches[i * sketches_stride]`; the stride is at least
    /// `slots`, itself at least one.
    pub fn sketches<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        texts: StringsView<'scope>,
        sketches: &'scope mut [u8],
        slots: usize,
        sketches_stride: usize,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        if slots == 0 || sketches_stride < slots {
            return Err(Status::UnexpectedDimensions);
        }
        let span = match texts.len() {
            0 => 0,
            count => (count - 1)
                .checked_mul(sketches_stride)
                .and_then(|offset| offset.checked_add(slots))
                .ok_or(Status::OverflowRisk)?,
        };
        if sketches.len() < span {
            return Err(Status::UnexpectedDimensions);
        }
        unsafe { sz_ngrams_sketches(self, &texts.view, slots, sketches.as_mut_ptr(), sketches_stride, stream.as_raw()) }
            .check()
    }
}

impl Drop for NgramsEngine {
    fn drop(&mut self) {
        unsafe { sz_ngrams_engine_free(self as *mut Self, core::ptr::null_mut()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLOTS: usize = 512;

    fn agreements(first: &[u8], second: &[u8]) -> usize {
        first.iter().zip(second).filter(|(left, right)| left == right).count()
    }

    /// A pair is exact while its misses over the query's distinct N-grams stay within the radius,
    /// and scores zero at every width once they pass it.
    #[test]
    fn radius_zeroes_pairs_past_it() {
        let cpu = Stream::default(Capabilities::enabled_cpu());
        let mut engine = NgramsEngine::new(&["abcdefgh"], &[3, 4], 0, &cpu).unwrap();
        // Misses over the query's eleven distinct N-grams at both widths: none, two, and eight.
        let candidates = StringsView::from_slices(&["abcdefgh", "abcdefgx", "abcdxxxx"]);
        let overlaps_for = |engine: &mut NgramsEngine, radius: Option<f32>| {
            let mut overlaps = [f32::NAN; 6];
            cpu.scope(|scope| engine.overlaps(scope, candidates, radius, &mut overlaps, 6, 2)).unwrap();
            overlaps
        };
        let exact = overlaps_for(&mut engine, None);
        assert_eq!(exact[..2], [1.0, 1.0]);
        assert!(exact[2] > 0.0 && exact[2] < 1.0 && exact[3] > 0.0 && exact[3] < 1.0);
        assert!(exact[4] > 0.0 && exact[5] > 0.0);
        assert_eq!(overlaps_for(&mut engine, Some(1.0)), exact);
        assert_eq!(overlaps_for(&mut engine, Some(8.0 / 11.0)), exact);
        let past_seven = overlaps_for(&mut engine, Some(7.0 / 11.0));
        assert_eq!(past_seven[..4], exact[..4]);
        assert_eq!(past_seven[4..], [0.0, 0.0]);
        assert_eq!(overlaps_for(&mut engine, Some(2.0 / 11.0)), [1.0, 1.0, exact[2], exact[3], 0.0, 0.0]);
        assert_eq!(overlaps_for(&mut engine, Some(1.0 / 11.0)), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(overlaps_for(&mut engine, Some(0.0)), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// An engine of no queries sketches: equal texts alike, a near text at more slots than a far
    /// one, a text under every width as `0x4A`, leaving the stride's tail untouched.
    #[test]
    fn sketches_agree_on_shared_ngrams() {
        const STRIDE: usize = SLOTS + 3;
        let cpu = Stream::default(Capabilities::enabled_cpu());
        let texts = ["Hello, world", "Hello, world", "Hello, word", "Привет, мир", "Hi"];
        for widths in [&[5, 8][..], &[4], &[3, 5, 8]] {
            let mut engine = NgramsEngine::new::<&str>(&[], widths, 0, &cpu).unwrap();
            let mut sketches = [0x55u8; 5 * STRIDE];
            let texts = StringsView::from_slices(&texts);
            cpu.scope(|scope| engine.sketches(scope, texts, &mut sketches, SLOTS, STRIDE)).unwrap();
            let row = |index: usize| &sketches[index * STRIDE..index * STRIDE + SLOTS];
            assert_eq!(row(0), row(1));
            assert!(agreements(row(0), row(2)) > agreements(row(0), row(3)));
            assert!(row(4).iter().all(|&slot| slot == 0x4A));
            assert!(sketches.chunks_exact(STRIDE).all(|row| row[SLOTS..] == [0x55; 3]));
        }
    }

    /// No slots, a stride under the slots, and an output too short for its texts are refused.
    #[test]
    fn bad_sketch_arguments_are_refused() {
        let cpu = Stream::default(Capabilities::enabled_cpu());
        let mut engine = NgramsEngine::new(&["kitten"], &[3], 0, &cpu).unwrap();
        let texts = StringsView::from_slices(&["kitten", "sitting"]);
        let mut sketches = [0u8; 2 * SLOTS];
        let mut sketch = |sketches: &mut [u8], slots: usize, stride: usize| {
            cpu.scope(|scope| engine.sketches(scope, texts, sketches, slots, stride))
        };
        assert_eq!(sketch(&mut sketches, 0, SLOTS), Err(Status::UnexpectedDimensions));
        assert_eq!(sketch(&mut sketches, SLOTS, SLOTS - 1), Err(Status::UnexpectedDimensions));
        assert_eq!(sketch(&mut sketches[1..], SLOTS, SLOTS), Err(Status::UnexpectedDimensions));
        assert_eq!(sketch(&mut sketches, SLOTS, SLOTS), Ok(()));
    }
}
