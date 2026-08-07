//! Newline, whitespace, and delimiter segmentation of UTF-8 text.
//!
//! Also home to the generic iterators the segmenters share.

use super::*;
use core::ffi::c_void;
use core::marker::PhantomData;

/// A zero-sized UTF-8 segmentation kernel selector. Each implementor binds one FFI segmenter, so the shared
/// [`Utf8Split`] / [`Utf8Segments`] iterators monomorphize to a direct, branch-free call (no function pointer).
pub trait SegmenterKernel {
    /// Reports up to `capacity` segments of `text` into `offsets` / `lengths`, returning the count and writing
    /// the number of consumed bytes to `consumed`.
    ///
    /// # Safety
    /// `offsets` and `lengths` must each point to at least `capacity` writable `usize` slots, and `text` to
    /// `length` readable bytes.
    unsafe fn segment(
        text: *const c_void,
        length: usize,
        offsets: *mut usize,
        lengths: *mut usize,
        capacity: usize,
        consumed: *mut usize,
    ) -> usize;
}

/// Kernel behind [`Utf8SplitNewlines`] (`sz_utf8_newlines`).
pub struct Newlines;
impl SegmenterKernel for Newlines {
    unsafe fn segment(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        sz_utf8_newlines(t, n, o, l, c, u)
    }
}

/// Kernel behind [`Utf8SplitWhitespaces`] (`sz_utf8_whitespaces`).
pub struct Whitespaces;
impl SegmenterKernel for Whitespaces {
    unsafe fn segment(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        sz_utf8_whitespaces(t, n, o, l, c, u)
    }
}

/// Kernel behind [`Utf8SplitDelimiters`] (`sz_utf8_delimiters`).
pub struct Delimiters;
impl SegmenterKernel for Delimiters {
    unsafe fn segment(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        sz_utf8_delimiters(t, n, o, l, c, u)
    }
}

/// Which parts a [`Utf8Split`] yields, as a compile-time `(FIRST, STRIDE)` over the span boundaries:
/// the segments BETWEEN separators, the SEPARATORS themselves, or BOTH interleaved (lossless).
pub trait SplitParts {
    /// First boundary index to visit (0 for between/both, 1 for separators).
    const FIRST: usize;
    /// Step between visited boundaries (2 for between/separators, 1 for both).
    const STRIDE: usize;
}
/// Yields the segments between separators (the default).
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
/// Yields segments and separators interleaved (concatenating them reproduces the input).
pub struct Both;
impl SplitParts for Both {
    const FIRST: usize = 0;
    const STRIDE: usize = 1;
}

/// A range over UTF-8 text split on the separators a kernel reports, selecting which parts to yield.
///
/// The kernel's separator endpoints are the span boundaries `{0, s0.start, s0.end, ..., [len]}`; span `k` is
/// `bound(k)..bound(k+1)`, and `P` reduces the mode to a `(FIRST, STRIDE)` walk over them - so the hot path is one
/// formula for all three modes. Rust stable cannot size `[usize; 2*STEPS+2]`, so the raw separator spans are kept and
/// each boundary is computed on the fly rather than materialized into an array.
pub struct Utf8Split<
    'a,
    Kernel: SegmenterKernel,
    Parts: SplitParts = Between,
    Empty: EmptySegments = KeepEmpty,
    const STEPS: usize = ITERATORS_DEFAULT_STEPS,
> {
    text: &'a [u8],
    suffix: usize,           // Base of the current batch (absolute offset into `text`)
    starts: [usize; STEPS],  // Raw separator offsets from the kernel, relative to `suffix`
    lengths: [usize; STEPS], // Raw separator lengths
    separators: usize,       // Separators in the current batch (the kernel's return value)
    region: usize,           // Bytes of the current batch (`text.len() - suffix` at the last refill)
    spans: usize,            // Number of yieldable boundary spans; `spans == 0` is the end sentinel
    index: usize,            // Current boundary cursor (span is `bound(index)..bound(index + 1)`)
    advance: usize,          // Bytes to advance `suffix` by when the batch drains
    _markers: PhantomData<(Kernel, Parts, Empty)>,
}

impl<'a, Kernel: SegmenterKernel, Parts: SplitParts, Empty: EmptySegments>
    Utf8Split<'a, Kernel, Parts, Empty, ITERATORS_DEFAULT_STEPS>
{
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, Kernel: SegmenterKernel, Parts: SplitParts, Empty: EmptySegments, const STEPS: usize>
    Utf8Split<'a, Kernel, Parts, Empty, STEPS>
{
    /// Constructs an iterator buffering up to `STEPS` separators per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        let mut splits = Self {
            text,
            suffix: 0,
            starts: [0; STEPS],
            lengths: [0; STEPS],
            separators: 0,
            region: 0,
            spans: 0,
            index: 0,
            advance: 0,
            _markers: PhantomData,
        };
        splits.refill();
        splits.settle();
        splits
    }

    /// The `boundary`-th span boundary relative to `suffix`: `{0, s0.start, s0.end, s1.start, ..., [region]}`.
    #[inline]
    fn bound(&self, boundary: usize) -> usize {
        if boundary == 0 {
            0
        } else if boundary > 2 * self.separators {
            self.region // the end-of-text closing boundary
        } else if boundary & 1 == 1 {
            self.starts[(boundary - 1) / 2]
        } else {
            let separator = boundary / 2 - 1;
            self.starts[separator] + self.lengths[separator]
        }
    }

    /// Refill from `suffix`: fetch a separator batch; boundaries are derived lazily by [`Self::bound`].
    fn refill(&mut self) {
        self.region = self.text.len() - self.suffix;
        let mut consumed = 0usize;
        self.separators = unsafe {
            Kernel::segment(
                self.text[self.suffix..].as_ptr() as *const c_void,
                self.region,
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        debug_assert!(
            self.separators <= STEPS,
            "segmenter reported more spans than the capacity STEPS"
        );
        debug_assert!(consumed <= self.region, "segmenter consumed past the region end");
        debug_assert!(
            consumed > 0 || self.region == 0,
            "segmenter made no progress (the iterator would loop forever)"
        );
        debug_assert!(
            (0..self.separators).all(|s| self.starts[s] + self.lengths[s] <= self.region
                && (s == 0 || self.starts[s] >= self.starts[s - 1] + self.lengths[s - 1])),
            "separator spans run past the region, overlap, or are out of order"
        );
        // A batch cut short by `STEPS` must resume exactly at the end of its last separator: the bytes between
        // that separator and wherever the kernel's own scan stopped belong to the next segment, and a resume
        // offset past them drops them from the output.
        debug_assert!(
            self.separators < STEPS || consumed == self.starts[self.separators - 1] + self.lengths[self.separators - 1],
            "segmenter resumed past the end of its last emitted separator"
        );
        let eof = consumed == self.region;
        // Boundaries: `0`, then 2 per separator, plus the closing `region` at end-of-text.
        self.spans = 2 * self.separators + if eof { 1 } else { 0 };
        self.advance = if eof { self.region + 1 } else { consumed };
        self.index = Parts::FIRST;
    }

    /// Position `index` on the next yieldable span, refilling and (when `Empty::SKIP`) skipping empty spans.
    /// `Empty::SKIP` is a const, so the skip loop folds away entirely for the default keep-empties (`KeepEmpty`) case.
    fn settle(&mut self) {
        loop {
            if Empty::SKIP {
                while self.index < self.spans && self.bound(self.index + 1) == self.bound(self.index) {
                    self.index += Parts::STRIDE;
                }
            }
            if self.index < self.spans || self.spans == 0 {
                return;
            }
            self.suffix += self.advance;
            if self.suffix > self.text.len() {
                self.spans = 0;
                return;
            }
            self.refill();
        }
    }
}

impl<'a, Kernel: SegmenterKernel, Parts: SplitParts, const STEPS: usize>
    Utf8Split<'a, Kernel, Parts, KeepEmpty, STEPS>
{
    /// Skips zero-length spans, returning the `SkipEmpty` variant. A compile-time policy rather than a
    /// runtime flag, so the keep-empties default stays branchless.
    pub fn skip_empty(self) -> Utf8Split<'a, Kernel, Parts, SkipEmpty, STEPS> {
        Utf8Split::with_steps(self.text)
    }
}

impl<'a, Kernel: SegmenterKernel, Empty: EmptySegments, const STEPS: usize>
    Utf8Split<'a, Kernel, Between, Empty, STEPS>
{
    /// The same split yielding segments **and** separators interleaved. Lossless (concatenation reproduces the
    /// input) only when empties are kept; the `Empty` policy carries through the type, so `.skip_empty()` and
    /// `.with_separators()` compose in either order.
    pub fn with_separators(self) -> Utf8Split<'a, Kernel, Both, Empty, STEPS> {
        Utf8Split::with_steps(self.text)
    }
}

impl<'a, Kernel: SegmenterKernel, Parts: SplitParts, Empty: EmptySegments, const STEPS: usize> Iterator
    for Utf8Split<'a, Kernel, Parts, Empty, STEPS>
{
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.spans == 0 {
            return None;
        }
        let begin = self.suffix + self.bound(self.index);
        let end = self.suffix + self.bound(self.index + 1);
        self.index += Parts::STRIDE;
        self.settle();
        Some(&self.text[begin..end])
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

/// An iterator over the newline runs themselves (the separators), in order.
pub type Utf8Newlines<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Newlines, Separators, KeepEmpty, STEPS>;

/// An iterator over segments of UTF-8 text split by whitespace characters.
///
/// Splits on all 25 Unicode "White_Space" characters; N whitespace delimiters yield N+1 segments. By
/// default empty segments are **kept** (matching `str::split`), so runs of
/// whitespace and leading/trailing whitespace produce empty slices. Call [`Self::skip_empty`] for the
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

/// An iterator over the whitespace runs themselves (the separators), in order.
pub type Utf8Whitespaces<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Whitespaces, Separators, KeepEmpty, STEPS>;

/// An iterator over segments of UTF-8 text split by any Unicode delimiter codepoint.
///
/// Splits on every codepoint whose Unicode general category is punctuation (`P*`), symbol (`S*`), or
/// separator/whitespace (`Z*`) — the superset of [`Utf8SplitWhitespaces`]. N delimiters yield N+1 segments; empty
/// segments are **kept** by default (call [`Self::skip_empty`] to drop them for token-style splitting).
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

/// An iterator over the delimiter runs themselves (the separators), in order.
pub type Utf8Delimiters<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> =
    Utf8Split<'a, Delimiters, Separators, KeepEmpty, STEPS>;

pub struct Utf8Segments<'a, Kernel: SegmenterKernel, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    text: &'a [u8],
    suffix: usize, // Start of the not-yet-segmented suffix (a UAX-29 boundary; `text.len()` once exhausted)
    starts: [usize; STEPS], // Buffered word offsets, relative to `suffix`
    lengths: [usize; STEPS], // Buffered word lengths
    count: usize,  // Number of buffered words (0 once exhausted)
    index: usize,  // Index of the next word to yield from the buffer
    _kernel: PhantomData<Kernel>, // Zero-sized; selects the FFI segmenter at monomorphization.
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
    /// Constructs an iterator buffering up to `STEPS` words per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        let mut splits = Self {
            text,
            suffix: 0,
            starts: [0; STEPS],
            lengths: [0; STEPS],
            count: 0,
            index: 0,
            _kernel: PhantomData,
        };
        splits.fill();
        splits
    }

    /// Refills the buffer from the current suffix; `count` becomes 0 once the suffix is empty.
    fn fill(&mut self) {
        let mut consumed = 0usize;
        self.count = unsafe {
            Kernel::segment(
                self.text[self.suffix..].as_ptr() as *const c_void,
                self.text.len() - self.suffix,
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        self.index = 0;
    }
}

impl<'a, Kernel: SegmenterKernel, const STEPS: usize> Iterator for Utf8Segments<'a, Kernel, STEPS> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            if self.count == 0 {
                return None; // Empty input or fully drained.
            }
            // Batch drained: advance past the last word (a UAX-29 boundary) and refill from the remaining suffix.
            self.suffix += self.starts[self.count - 1] + self.lengths[self.count - 1];
            self.fill();
            if self.count == 0 {
                return None;
            }
        }
        let begin = self.suffix + self.starts[self.index];
        let end = begin + self.lengths[self.index];
        self.index += 1;
        Some(&self.text[begin..end])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    extern crate alloc;
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::sz::*;

    /// Segments tile the input, so the yielded segments must match regardless of the batch size `STEPS`;
    /// a tiny batch (STEPS == 1) exercises the refill seam on every boundary the kernel reports.
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
        // `split_delimiters` yields the content BETWEEN ',', ' ', U+2014; skip_empty drops the empties.
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
        // Sparse delimiters with long undelimited runs: each batch fills on the last delimiter its vector window
        // holds, and the letters between that delimiter and the window edge must survive into the next segment.
        // Dense inputs cannot reach this path, since a filled batch there always leaves hits behind in the window.
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
        // Small batch size must agree with the default across ALL modes (exercises refill boundaries for
        // separators and both, not just between - the paths where a trailing gap straddles a batch).
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
        // `utf8_wordbreaks` tiles into all UAX-29 segments (words and the separators between them).
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
        // The simple example from the doc comment: KEEP yields the surrounding empties, skip_empty drops them.
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
