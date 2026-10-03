//! Newline, whitespace, and delimiter segmentation of UTF-8 text.
//!
//! Also home to the generic iterators the segmenters share.
//!
//! File: rust/stringzilla/utf8_tokens.rs
//! Author: Ash Vardanian

use core::ffi::c_void;
use core::marker::PhantomData;

use super::*;

/// A zero-sized selector of a UTF-8 kernel reporting separator runs. Each implementor binds one
/// FFI tokenizer, so the shared [`Utf8Split`] iterator monomorphizes to a direct, branch-free call
/// rather than a function pointer.
pub trait TokenizerKernel {
    /// Reports up to `capacity` separator runs of `text` into `offsets` / `lengths`, returning the
    /// count and writing the number of consumed bytes to `consumed`: the end of the last run when
    /// the count reaches `capacity`, otherwise `length`.
    ///
    /// # Safety
    /// `offsets` and `lengths` must each point to at least `capacity` writable `usize` slots, and
    /// `text` to `length` readable bytes.
    unsafe fn tokenize(
        text: *const c_void,
        length: usize,
        offsets: *mut usize,
        lengths: *mut usize,
        capacity: usize,
        consumed: *mut usize,
    ) -> usize;
}

/// A zero-sized selector of a UTF-8 kernel tiling text into segments. Each implementor binds one
/// FFI segmenter, so the shared [`Utf8Segments`] iterator monomorphizes to a direct call.
pub trait SegmenterKernel {
    /// Reports the lengths of up to `capacity` segments tiling the front of `text`, returning the
    /// count; each segment starts where the previous one ends.
    ///
    /// # Safety
    /// `lengths` must point to at least `capacity` writable `usize` slots, and `text` to `length`
    /// readable bytes.
    unsafe fn segment(text: *const c_void, length: usize, lengths: *mut usize, capacity: usize) -> usize;
}

/// Kernel behind [`Utf8SplitNewlines`] (`sz_utf8_newlines_best`).
pub struct Newlines;
impl TokenizerKernel for Newlines {
    unsafe fn tokenize(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        let mut count = 0;
        sz_utf8_newlines_best(
            t,
            n,
            o,
            l,
            c,
            &mut count,
            u,
            Capabilities::CPUS.bits(),
            core::ptr::null_mut(),
        )
        .infallible();
        count
    }
}

/// Kernel behind [`Utf8SplitWhitespaces`] (`sz_utf8_whitespaces_best`).
pub struct Whitespaces;
impl TokenizerKernel for Whitespaces {
    unsafe fn tokenize(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        let mut count = 0;
        sz_utf8_whitespaces_best(
            t,
            n,
            o,
            l,
            c,
            &mut count,
            u,
            Capabilities::CPUS.bits(),
            core::ptr::null_mut(),
        )
        .infallible();
        count
    }
}

/// Kernel behind [`Utf8SplitDelimiters`] (`sz_utf8_delimiters_best`).
pub struct Delimiters;
impl TokenizerKernel for Delimiters {
    unsafe fn tokenize(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        let mut count = 0;
        sz_utf8_delimiters_best(
            t,
            n,
            o,
            l,
            c,
            &mut count,
            u,
            Capabilities::CPUS.bits(),
            core::ptr::null_mut(),
        )
        .infallible();
        count
    }
}

/// Which parts a [`Utf8Split`] yields, as a compile-time `(FIRST, STRIDE)` over the
/// span boundaries: the segments between separators, the separators themselves, or
/// both interleaved losslessly.
pub trait SplitParts {
    /// First boundary index to visit (0 for between/both, 1 for separators).
    const FIRST: usize;
    /// Step between visited boundaries (2 for between/separators, 1 for both).
    const STRIDE: usize;
}
/// Yields the segments between separators, the default.
pub struct Between;
impl SplitParts for Between {
    const FIRST: usize = 0;
    const STRIDE: usize = 2;
}
/// Yields the separator runs themselves.
pub struct Separators;
impl SplitParts for Separators {
    const FIRST: usize = 1;
    const STRIDE: usize = 2;
}
/// Yields segments and separators interleaved, so concatenating them reproduces the input.
pub struct Both;
impl SplitParts for Both {
    const FIRST: usize = 0;
    const STRIDE: usize = 1;
}

/// A range over UTF-8 text split on the separators a kernel reports, selecting which
/// parts to yield.
///
/// A batch's boundaries are `0`, each separator's start and end, and the end of the text once the
/// batch reaches it; part `k` spans boundaries `k` and `k + 1`, and `Parts` walks them as `(FIRST,
/// STRIDE)`, so all three modes share one formula. A batch holding `STEPS` separators stops at the
/// end of its last one, where the next batch begins.
pub struct Utf8Split<
    'a,
    Kernel: TokenizerKernel,
    Parts: SplitParts = Between,
    Empty: EmptySegments = KeepEmpty,
    const STEPS: usize = ITERATORS_DEFAULT_STEPS,
> {
    rest: Option<&'a [u8]>,  // The current batch through the end, `None` once exhausted
    starts: [usize; STEPS],  // Separator offsets in the batch
    lengths: [usize; STEPS], // Separator lengths
    count: usize,            // Separators in the batch
    index: usize,            // Boundary starting the next part to yield
    _markers: PhantomData<(Kernel, Parts, Empty)>,
}

impl<'a, Kernel: TokenizerKernel, Parts: SplitParts, Empty: EmptySegments>
    Utf8Split<'a, Kernel, Parts, Empty, ITERATORS_DEFAULT_STEPS>
{
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, Kernel: TokenizerKernel, Parts: SplitParts, Empty: EmptySegments, const STEPS: usize>
    Utf8Split<'a, Kernel, Parts, Empty, STEPS>
{
    /// Constructs an iterator buffering up to `STEPS` separators per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        const { assert!(STEPS > 0, "STEPS must be positive") };
        // A drained full batch ending at 0, so the first `next` fetches from the start.
        Self {
            rest: Some(text),
            starts: [0; STEPS],
            lengths: [0; STEPS],
            count: STEPS,
            index: 2 * STEPS,
            _markers: PhantomData,
        }
    }

    /// The `boundary`-th boundary of the batch starting `rest`.
    #[inline]
    fn bound(&self, rest: &[u8], boundary: usize) -> usize {
        if boundary == 0 {
            0
        } else if boundary > 2 * self.count {
            rest.len()
        } else if boundary % 2 == 1 {
            self.starts[boundary / 2]
        } else {
            self.starts[boundary / 2 - 1] + self.lengths[boundary / 2 - 1]
        }
    }

    /// Fetches the batch of separators starting `rest`.
    fn refill(&mut self, rest: &'a [u8]) {
        let mut consumed = 0usize;
        self.count = unsafe {
            Kernel::tokenize(
                rest.as_ptr() as *const c_void,
                rest.len(),
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        self.rest = Some(rest);
        self.index = Parts::FIRST;
        debug_assert!(
            self.count <= STEPS,
            "tokenizer reported more runs than the capacity STEPS"
        );
        debug_assert!(
            (0..self.count).all(|s| self.lengths[s] > 0
                && self.starts[s] + self.lengths[s] <= rest.len()
                && (s == 0 || self.starts[s] >= self.starts[s - 1] + self.lengths[s - 1])),
            "separator runs are empty, run past the text, overlap, or are out of order"
        );
        debug_assert!(
            consumed == self.bound(rest, 2 * self.count + usize::from(self.count < STEPS)),
            "tokenizer resumed anywhere but the end of its last run or of the text"
        );
    }
}

impl<'a, Kernel: TokenizerKernel, Parts: SplitParts, const STEPS: usize>
    Utf8Split<'a, Kernel, Parts, KeepEmpty, STEPS>
{
    /// Skips zero-length spans, returning the `SkipEmpty` variant. A compile-time policy rather
    /// than a runtime flag, so the keep-empties default stays branchless.
    pub fn skip_empty(self) -> Utf8Split<'a, Kernel, Parts, SkipEmpty, STEPS> {
        Utf8Split {
            rest: self.rest,
            starts: self.starts,
            lengths: self.lengths,
            count: self.count,
            index: self.index,
            _markers: PhantomData,
        }
    }
}

impl<'a, Kernel: TokenizerKernel, Empty: EmptySegments, const STEPS: usize>
    Utf8Split<'a, Kernel, Between, Empty, STEPS>
{
    /// The same split yielding segments __and__ separators interleaved. Lossless (concatenation
    /// reproduces the input) only when empties are kept; the `Empty` policy carries through the
    /// type, so `.skip_empty()` and `.with_separators()` compose in either order.
    pub fn with_separators(self) -> Utf8Split<'a, Kernel, Both, Empty, STEPS> {
        Utf8Split {
            rest: self.rest,
            starts: self.starts,
            lengths: self.lengths,
            count: self.count,
            index: self.index,
            _markers: PhantomData,
        }
    }
}

impl<'a, Kernel: TokenizerKernel, Parts: SplitParts, Empty: EmptySegments, const STEPS: usize> Iterator
    for Utf8Split<'a, Kernel, Parts, Empty, STEPS>
{
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let rest = self.rest?;
            let full = self.count == STEPS;
            if self.index >= 2 * self.count + usize::from(!full) {
                if !full {
                    self.rest = None;
                    return None;
                }
                self.refill(&rest[self.bound(rest, 2 * self.count)..]);
                continue;
            }
            let part = &rest[self.bound(rest, self.index)..self.bound(rest, self.index + 1)];
            self.index += Parts::STRIDE;
            if !(Empty::SKIP && part.is_empty()) {
                return Some(part);
            }
        }
    }
}

/// An iterator over substrings of UTF-8 text split by newline characters.
///
/// This iterator yields slices between newline characters. The newline characters themselves
/// are not included in the yielded slices. Handles all 8 Unicode newline characters including
/// CRLF as a single delimiter.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{Utf8SplitNewlines};
///
/// let text = b"Hello\nWorld\r\nRust";
/// let lines: Vec<&[u8]> = Utf8SplitNewlines::new(text).collect();
/// assert_eq!(lines, vec![&b"Hello"[..], &b"World"[..], &b"Rust"[..]]);
/// ```
pub type Utf8SplitNewlines<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Newlines, Between, KeepEmpty, STEPS>;

/// An iterator over the newline runs themselves, the separators, in order.
pub type Utf8Newlines<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Newlines, Separators, KeepEmpty, STEPS>;

/// An iterator over segments of UTF-8 text split by whitespace characters.
///
/// Splits on all 25 Unicode "White_Space" characters; N whitespace delimiters yield N+1 segments.
/// By default empty segments are __kept__, matching `str::split`, so runs of whitespace and
/// leading/trailing whitespace produce empty slices. Call [`Self::skip_empty`] for the
/// `str::split_whitespace`-style behavior that drops empties and yields only non-empty tokens.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{Utf8SplitWhitespaces};
///
/// // Default KEEP policy: empties around the words are preserved.
/// let text = b"  hi  ";
/// let segments: Vec<&[u8]> = Utf8SplitWhitespaces::new(text).collect();
/// assert_eq!(segments, vec![&b""[..], &b""[..], &b"hi"[..], &b""[..], &b""[..]]);
///
/// // Opt in to dropping empties for token-style splitting.
/// let tokens: Vec<&[u8]> = Utf8SplitWhitespaces::new(text).skip_empty().collect();
/// assert_eq!(tokens, vec![&b"hi"[..]]);
/// ```
pub type Utf8SplitWhitespaces<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Whitespaces, Between, KeepEmpty, STEPS>;

/// An iterator over the whitespace runs themselves, the separators, in order.
pub type Utf8Whitespaces<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Whitespaces, Separators, KeepEmpty, STEPS>;

/// An iterator over segments of UTF-8 text split by any Unicode delimiter codepoint.
///
/// Splits on every codepoint whose Unicode general category is punctuation (`P*`), symbol (`S*`),
/// or separator/whitespace (`Z*`) — the superset of [`Utf8SplitWhitespaces`]. N delimiters yield
/// N+1 segments; empty segments are __kept__ by default (call [`Self::skip_empty`] to drop them
/// for token-style splitting).
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{Utf8SplitDelimiters};
///
/// // "Hi, world—foo" splits on ',', ' ', and U+2014 EM DASH.
/// let tokens: Vec<&[u8]> = Utf8SplitDelimiters::new("Hi, world\u{2014}foo".as_bytes()).skip_empty().collect();
/// assert_eq!(tokens, vec![&b"Hi"[..], &b"world"[..], &b"foo"[..]]);
/// ```
pub type Utf8SplitDelimiters<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Delimiters, Between, KeepEmpty, STEPS>;

/// An iterator over the delimiter runs themselves, the separators, in order.
pub type Utf8Delimiters<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Delimiters, Separators, KeepEmpty, STEPS>;

/// An iterator over the segments a kernel tiles UTF-8 text into, in order.
pub struct Utf8Segments<'a, Kernel: SegmenterKernel, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    rest: &'a [u8],          // Text from the next segment to yield through the end
    lengths: [usize; STEPS], // Lengths of the buffered segments
    count: usize,            // Number of buffered segments
    index: usize,            // Index of the next segment to yield from the buffer
    _kernel: PhantomData<Kernel>,
}

impl<'a, Kernel: SegmenterKernel> Utf8Segments<'a, Kernel, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish, e.g.
    /// `Utf8Wordbreaks::<1>::with_steps(text)`.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, Kernel: SegmenterKernel, const STEPS: usize> Utf8Segments<'a, Kernel, STEPS> {
    /// Constructs an iterator buffering up to `STEPS` segments per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        const { assert!(STEPS > 0, "STEPS must be positive") };
        Self {
            rest: text,
            lengths: [0; STEPS],
            count: 0,
            index: 0,
            _kernel: PhantomData,
        }
    }
}

impl<'a, Kernel: SegmenterKernel, const STEPS: usize> Iterator for Utf8Segments<'a, Kernel, STEPS> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            if self.rest.is_empty() {
                return None;
            }
            self.count = unsafe {
                Kernel::segment(
                    self.rest.as_ptr() as *const c_void,
                    self.rest.len(),
                    self.lengths.as_mut_ptr(),
                    STEPS,
                )
            };
            self.index = 0;
            debug_assert!(
                self.count > 0 && self.lengths[..self.count].iter().all(|&length| length > 0),
                "segmenter made no progress (the iterator would loop forever)"
            );
        }
        let (segment, rest) = self.rest.split_at(self.lengths[self.index]);
        self.rest = rest;
        self.index += 1;
        Some(segment)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    extern crate alloc;
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    /// Segments tile the input, so the yielded segments must match regardless of the batch
    /// size `STEPS`; a tiny batch (STEPS == 1) exercises the refill seam on every boundary
    /// the kernel reports.
    pub(crate) fn assert_steps_invariant<Kernel: SegmenterKernel>(text: &[u8]) {
        let forward: Vec<&[u8]> = Utf8Segments::<Kernel, ITERATORS_DEFAULT_STEPS>::new(text).collect();
        assert_eq!(Utf8Segments::<Kernel, 1>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Segments::<Kernel, 3>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(
            Utf8Segments::<Kernel, 65>::with_steps(text).collect::<Vec<_>>(),
            forward
        );
    }

    #[test]
    fn utf8_delimiters() {
        // `split_delimiters` yields the content between ',', ' ', U+2014; skip_empty
        // drops the empties.
        let toks: Vec<&[u8]> = "Hi, world\u{2014}foo"
            .as_bytes()
            .sz_utf8_split_delimiters()
            .skip_empty()
            .collect();
        assert_eq!(toks, vec![&b"Hi"[..], &b"world"[..], &b"foo"[..]]);
        // Default policy keeps the empty segment between adjacent delimiters.
        let kept: Vec<&[u8]> = "a,,b".as_bytes().sz_utf8_split_delimiters().collect();
        assert_eq!(kept, vec![&b"a"[..], &b""[..], &b"b"[..]]);
    }

    #[test]
    fn utf8_split_delimiters_sparse_batches() {
        // Sparse delimiters with long undelimited runs: each batch fills on the last delimiter its
        // vector window holds, and the letters between that delimiter and the window edge must
        // survive into the next segment. Dense inputs cannot reach this path, since a filled batch
        // there always leaves hits behind in the window.
        for run in [16usize, 31, 62, 63, 64, 100] {
            let text = format!("a {} c", "b".repeat(run));
            let expected: Vec<&[u8]> = vec![b"a", text.as_bytes()[2..2 + run].as_ref(), b"c"];
            for tiny in [
                Utf8SplitDelimiters::<1>::with_steps(text.as_bytes()).collect::<Vec<_>>(),
                Utf8SplitDelimiters::<2>::with_steps(text.as_bytes()).collect::<Vec<_>>(),
                text.as_bytes().sz_utf8_split_delimiters().collect::<Vec<_>>(),
            ] {
                assert_eq!(tiny, expected, "run of {} undelimited bytes", run);
            }
            // Separators included, the split is lossless however small the batch.
            let both: Vec<&[u8]> = Utf8SplitDelimiters::<1>::with_steps(text.as_bytes())
                .with_separators()
                .collect();
            assert_eq!(both.concat(), text.as_bytes());
        }
    }

    #[test]
    fn utf8_split_modes() {
        // Scheme C: the bare name yields the separators; `split_` yields the content between.
        let text = "a b  c".as_bytes();
        let between: Vec<&[u8]> = text.sz_utf8_split_whitespaces().collect();
        assert_eq!(between, vec![&b"a"[..], &b"b"[..], &b""[..], &b"c"[..]]);
        let seps: Vec<&[u8]> = text.sz_utf8_whitespaces().collect();
        assert_eq!(seps, vec![&b" "[..], &b" "[..], &b" "[..]]);
        // `with_separators` interleaves them losslessly: concatenation reproduces the input.
        let both: Vec<&[u8]> = text.sz_utf8_split_whitespaces().with_separators().collect();
        assert_eq!(both.concat(), text);
        // Lossless round-trip also holds across leading/trailing separators and empty input.
        for t in ["  x  ", "", "abc", "a\r\nb"] {
            let rt: Vec<&[u8]> = t.as_bytes().sz_utf8_split_newlines().with_separators().collect();
            assert_eq!(rt.concat(), t.as_bytes());
        }
        // Empty input still yields one empty segment (matches C++ `[""]`).
        let empty: Vec<&[u8]> = "".as_bytes().sz_utf8_split_whitespaces().collect();
        assert_eq!(empty, vec![&b""[..]]);
        // Small batch size must agree with the default across all modes (exercises refill
        // boundaries for separators and both, not just between - the paths where a trailing gap
        // straddles a batch).
        let many = "w ".repeat(50) + "end";
        let between_small: Vec<&[u8]> = Utf8SplitWhitespaces::<2>::with_steps(many.as_bytes()).collect();
        assert_eq!(
            between_small,
            many.as_bytes().sz_utf8_split_whitespaces().collect::<Vec<_>>()
        );
        let seps_small: Vec<&[u8]> = Utf8Whitespaces::<2>::with_steps(many.as_bytes()).collect();
        assert_eq!(seps_small, many.as_bytes().sz_utf8_whitespaces().collect::<Vec<_>>());
        let both_small: Vec<&[u8]> = Utf8SplitWhitespaces::<2>::with_steps(many.as_bytes())
            .with_separators()
            .collect();
        assert_eq!(both_small.concat(), many.as_bytes()); // lossless even across many refills
        assert_eq!(
            both_small,
            many.as_bytes()
                .sz_utf8_split_whitespaces()
                .with_separators()
                .collect::<Vec<_>>()
        );
        // `with_separators` preserves `skip_empty` regardless of chaining order.
        let dropped: Vec<&[u8]> = "a  b"
            .as_bytes()
            .sz_utf8_split_whitespaces()
            .skip_empty()
            .with_separators()
            .collect();
        assert!(dropped.iter().all(|s| !s.is_empty()));
        // `utf8_wordbreaks` tiles into all UAX-29 segments: words and the separators between them.
        let segs: Vec<&[u8]> = "Hello, world!".as_bytes().sz_utf8_wordbreaks().collect();
        assert_eq!(segs.concat(), &b"Hello, world!"[..]);
        assert_eq!(segs.len(), 5);
    }

    #[test]
    fn iter_newline_utf8_splits() {
        let text = b"a\nb\r\nc\n\nd";
        let lines: Vec<_> = Utf8SplitNewlines::new(text).collect();
        assert_eq!(lines, vec![b"a", b"b", b"c", &b""[..], b"d"]);
    }

    #[test]
    fn iter_newline_utf8_splits_unicode() {
        let text = "Hello\u{2028}World".as_bytes(); // LINE SEPARATOR
        let lines: Vec<_> = Utf8SplitNewlines::new(text).collect();
        assert_eq!(lines, vec!["Hello".as_bytes(), "World".as_bytes()]);
    }

    #[test]
    fn iter_whitespace_utf8_splits() {
        // KEEP (default): every one of the 8 whitespace delimiters yields a segment, so leading,
        // trailing, and inner runs all surface empties (str::split semantics, matching C++/Python).
        let text = b"  a \t b\n\nc  ";
        let segments: Vec<_> = Utf8SplitWhitespaces::new(text).collect();
        assert_eq!(
            segments,
            vec![
                &b""[..],
                &b""[..],
                b"a",
                &b""[..],
                &b""[..],
                b"b",
                &b""[..],
                b"c",
                &b""[..],
                &b""[..],
            ]
        );
        // skip_empty: recovers the str::split_whitespace token behavior.
        let tokens: Vec<_> = Utf8SplitWhitespaces::new(text).skip_empty().collect();
        assert_eq!(tokens, vec![b"a", b"b", b"c"]);
    }

    #[test]
    fn iter_whitespace_utf8_splits_keep_default() {
        // The simple example from the doc comment: KEEP yields the surrounding empties,
        // skip_empty drops them.
        let text = b"  hi  ";
        let kept: Vec<_> = Utf8SplitWhitespaces::new(text).collect();
        assert_eq!(kept, vec![&b""[..], &b""[..], b"hi", &b""[..], &b""[..]]);
        let tokens: Vec<_> = Utf8SplitWhitespaces::new(text).skip_empty().collect();
        assert_eq!(tokens, vec![b"hi"]);
    }

    #[test]
    fn iter_whitespace_utf8_splits_unicode() {
        let text = "a\u{3000}b\u{2000}c".as_bytes(); // IDEOGRAPHIC SPACE, EN QUAD
        let segments: Vec<_> = Utf8SplitWhitespaces::new(text).collect();
        assert_eq!(segments, vec![b"a", b"b", b"c"]); // single delimiters between words: no empties
        let tokens: Vec<_> = Utf8SplitWhitespaces::new(text).skip_empty().collect();
        assert_eq!(tokens, vec![b"a", b"b", b"c"]);
    }

    #[test]
    fn iter_whitespace_utf8_splits_skip_empty_all_whitespace() {
        let text = b"   \t  ";
        let kept: Vec<_> = Utf8SplitWhitespaces::new(text).collect();
        assert_eq!(kept.len(), 7); // 6 delimiters → 7 (all empty) segments
        assert!(kept.iter().all(|segment| segment.is_empty()));
        let tokens: Vec<&[u8]> = Utf8SplitWhitespaces::new(text).skip_empty().collect();
        assert!(tokens.is_empty());
    }

    #[test]
    fn iter_newline_utf8_splits_skip_empty() {
        let text = b"a\nb\r\nc\n\nd";
        // Default KEEP: the back-to-back "\n\n" yields an empty line.
        let kept: Vec<_> = Utf8SplitNewlines::new(text).collect();
        assert_eq!(kept, vec![b"a", b"b", b"c", &b""[..], b"d"]);
        // skip_empty: the empty line between "c" and "d" disappears.
        let nonempty: Vec<_> = Utf8SplitNewlines::new(text).skip_empty().collect();
        assert_eq!(nonempty, vec![b"a", b"b", b"c", b"d"]);
    }

    #[test]
    fn iter_newline_utf8_splits_steps_invariance() {
        // The yielded segments must be identical regardless of the batch size `STEPS`; a tiny batch
        // (STEPS == 1) exercises the refill/trailing-segment seam on every delimiter, while large
        // batches fit the whole input in one call.
        let text = b"\r\na\r\n\r\nb\r\nc\nd\n";
        let expected: Vec<&[u8]> = vec![b"", b"a", b"", b"b", b"c", b"d", b""];
        let from_1: Vec<_> = Utf8SplitNewlines::<1>::with_steps(text).collect();
        let from_3: Vec<_> = Utf8SplitNewlines::<3>::with_steps(text).collect();
        let from_65: Vec<_> = Utf8SplitNewlines::<65>::with_steps(text).collect();
        assert_eq!(from_1, expected);
        assert_eq!(from_3, expected);
        assert_eq!(from_65, expected);

        // skip_empty across the same batch sizes.
        let nonempty: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        assert_eq!(
            Utf8SplitNewlines::<1>::with_steps(text)
                .skip_empty()
                .collect::<Vec<_>>(),
            nonempty
        );
        assert_eq!(
            Utf8SplitNewlines::<3>::with_steps(text)
                .skip_empty()
                .collect::<Vec<_>>(),
            nonempty
        );
        assert_eq!(
            Utf8SplitNewlines::<65>::with_steps(text)
                .skip_empty()
                .collect::<Vec<_>>(),
            nonempty
        );
    }

    #[test]
    fn iter_whitespace_utf8_splits_steps_invariance() {
        let text = b"  a \t b\n\nc  ";
        let expected: Vec<&[u8]> = vec![b"", b"", b"a", b"", b"", b"b", b"", b"c", b"", b""];
        assert_eq!(
            Utf8SplitWhitespaces::<1>::with_steps(text).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            Utf8SplitWhitespaces::<3>::with_steps(text).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            Utf8SplitWhitespaces::<65>::with_steps(text).collect::<Vec<_>>(),
            expected
        );
        let tokens: Vec<&[u8]> = vec![b"a", b"b", b"c"];
        assert_eq!(
            Utf8SplitWhitespaces::<1>::with_steps(text)
                .skip_empty()
                .collect::<Vec<_>>(),
            tokens
        );
    }

    #[test]
    fn iter_newline_utf8_splits_trailing_newline() {
        // "\r\na\r\n\r\nb\r\n" should produce ["", "a", "", "b", ""]
        let text = b"\r\na\r\n\r\nb\r\n";
        let lines: Vec<&[u8]> = Utf8SplitNewlines::new(text).collect();
        assert_eq!(lines.len(), 5, "Expected 5 lines");
        let expected: Vec<&[u8]> = vec![b"", b"a", b"", b"b", b""];
        assert_eq!(lines, expected);
    }

    #[test]
    fn iter_newline_utf8_splits_no_trailing() {
        let text = b"a\nb\nc";
        let lines: Vec<&[u8]> = Utf8SplitNewlines::new(text).collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines, vec![b"a", b"b", b"c"]);
    }

    #[test]
    fn iter_newline_utf8_splits_empty_string() {
        let text = b"";
        let lines: Vec<&[u8]> = Utf8SplitNewlines::new(text).collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines, vec![b""]);
    }

    #[test]
    fn iter_newline_utf8_splits_single_newline() {
        let text = b"\n";
        let lines: Vec<&[u8]> = Utf8SplitNewlines::new(text).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines, vec![b"", b""]);
    }
}
