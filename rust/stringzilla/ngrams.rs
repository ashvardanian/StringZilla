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
/// # Examples
///
/// ```rust
/// use stringzilla::{Capabilities, MatrixSpan, NgramsEngine, Stream, StringsView, TensorSpan};
///
/// let cpu = Stream::default(Capabilities::enabled_cpu());
/// let mut engine = NgramsEngine::new(&["the quick brown fox"], &[4, 8], 0, &cpu).unwrap();
///
/// // A `[queries, candidates, widths]` block: 13 of the 16 distinct 4-grams, 9 of the 12 8-grams.
/// let mut storage = [0.0f32; 2];
/// let mut shares = TensorSpan::from_slice(&mut storage, [1, 1, 2]).unwrap();
/// let candidates = StringsView::from_slices(&["the quick brown cat"]);
/// cpu.scope(|scope| engine.overlaps(scope, candidates, None, &mut shares)).unwrap();
/// assert_eq!(storage, [13.0 / 16.0, 9.0 / 12.0]);
///
/// // Two equal texts sketch alike.
/// let mut storage = [0u8; 2 * 256];
/// let mut sketches = MatrixSpan::from_slice(&mut storage, [2, 256]).unwrap();
/// let texts = StringsView::from_slices(&["hello world", "hello world"]);
/// cpu.scope(|scope| engine.sketches(scope, texts, &mut sketches)).unwrap();
/// assert_eq!(storage[..256], storage[256..]);
/// ```
// The fields mirror `sz_ngrams_engine_t` one for one and only the counts are read from Rust, so the
// layout is load-bearing and the engine travels to C by pointer. Owning raw pointers makes it
// neither `Send` nor `Sync`, which is what the C contract wants: a compute verb grows the engine's
// scratch, so it mutates, and both verbs take `&mut self` for that reason.
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
    /// more it holds. `shares` is a `[queries, candidates, widths]` block, and any other shape is
    /// refused with [`Status::UnexpectedDimensions`].
    ///
    /// A pair's distance is its misses over the query's distinct N-grams, summed over every width.
    /// A `radius` of `Some(share)` keeps every pair within that distance exact: a pair past it
    /// scores `0.0` at every width, which lets a round skip it. `None`, or a share of `1.0` or
    /// more, keeps every score exact.
    ///
    /// A stream of another group than the engine's is refused with
    /// [`Status::DeviceMemoryMismatch`].
    pub fn overlaps<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        candidates: StringsView<'scope>,
        radius: Option<f32>,
        shares: &'scope mut TensorSpan<'_, f32, 3>,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        if shares.shape() != [self.count, candidates.len(), self.widths_count] {
            return Err(Status::UnexpectedDimensions);
        }
        unsafe {
            sz_ngrams_overlaps(
                self,
                &candidates.view,
                radius.unwrap_or(1.0),
                shares.as_mut_ptr(),
                shares.stride_bytes(0) as usize,
                shares.stride_bytes(1) as usize,
                stream.as_raw(),
            )
        }
        .check()
    }

    /// Queues on the stream of `scope` one MinHash sketch per text of `texts`, summarizing the set
    /// of its N-grams at the engine's widths.
    ///
    /// The share of positions where two sketches differ estimates one minus the Jaccard similarity
    /// of their N-gram sets, and more slots give a finer estimate. Only sketches of one slot count
    /// compare. `sketches` is a `[texts, slots]` block of at least one slot, and any other shape is
    /// refused with [`Status::UnexpectedDimensions`].
    pub fn sketches<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        texts: StringsView<'scope>,
        sketches: &'scope mut MatrixSpan<'_, u8>,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        let [texts_count, slots] = sketches.shape();
        if texts_count != texts.len() || slots == 0 {
            return Err(Status::UnexpectedDimensions);
        }
        let stride = sketches.stride_bytes(0) as usize;
        unsafe { sz_ngrams_sketches(self, &texts.view, slots, sketches.as_mut_ptr(), stride, stream.as_raw()) }.check()
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
            let mut storage = [f32::NAN; 6];
            let mut shares = TensorSpan::from_slice(&mut storage, [1, 3, 2]).unwrap();
            cpu.scope(|scope| engine.overlaps(scope, candidates, radius, &mut shares)).unwrap();
            storage
        };
        let exact = overlaps_for(&mut engine, None);
        assert_eq!(exact, [1.0, 1.0, 5.0 / 6.0, 4.0 / 5.0, 2.0 / 6.0, 1.0 / 5.0]);
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
            let mut span = MatrixSpan::from_slice_strided(&mut sketches, [5, SLOTS], [STRIDE as isize, 1]).unwrap();
            let texts = StringsView::from_slices(&texts);
            cpu.scope(|scope| engine.sketches(scope, texts, &mut span)).unwrap();
            let row = |index: usize| &sketches[index * STRIDE..index * STRIDE + SLOTS];
            assert_eq!(row(0), row(1));
            assert!(agreements(row(0), row(2)) > agreements(row(0), row(3)));
            assert!(row(4).iter().all(|&slot| slot == 0x4A));
            assert!(sketches.chunks_exact(STRIDE).all(|row| row[SLOTS..] == [0x55; 3]));
        }
    }

    /// No slots and a row per text other than one are refused.
    #[test]
    fn bad_sketch_shapes_are_refused() {
        let cpu = Stream::default(Capabilities::enabled_cpu());
        let mut engine = NgramsEngine::new(&["kitten"], &[3], 0, &cpu).unwrap();
        let texts = StringsView::from_slices(&["kitten", "sitting"]);
        let mut sketches = [0u8; 2 * SLOTS];
        let mut sketch = |shape: [usize; 2]| {
            let mut sketches = MatrixSpan::from_slice(&mut sketches[..shape[0] * shape[1]], shape).unwrap();
            cpu.scope(|scope| engine.sketches(scope, texts, &mut sketches))
        };
        assert_eq!(sketch([2, 0]), Err(Status::UnexpectedDimensions));
        assert_eq!(sketch([1, SLOTS]), Err(Status::UnexpectedDimensions));
        assert_eq!(sketch([2, SLOTS]), Ok(()));
    }

    /// A share block whose shape disagrees with the queries, candidates or widths is refused.
    #[test]
    fn overlaps_refuse_mismatched_shapes() {
        let cpu = Stream::default(Capabilities::enabled_cpu());
        let mut engine = NgramsEngine::new(&["kitten"], &[3, 4], 0, &cpu).unwrap();
        let candidates = StringsView::from_slices(&["sitting", "kitten"]);
        let mut storage = [0.0f32; 4];
        for shape in [[2, 1, 2], [1, 2, 1]] {
            let mut shares = TensorSpan::from_slice(&mut storage[..shape.iter().product::<usize>()], shape).unwrap();
            let refused = cpu.scope(|scope| engine.overlaps(scope, candidates, None, &mut shares));
            assert_eq!(refused, Err(Status::UnexpectedDimensions));
        }
    }
}
