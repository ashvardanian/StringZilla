//! Case-insensitive UTF-8 search, ordering, and match iteration.
//!
//! File: rust/stringzilla/utf8_uncased.rs
//! Author: Ash Vardanian

use super::*;
use core::cmp::Ordering;
use core::ffi::c_void;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

/// Performs uncased search for `needle` in UTF-8 `haystack`.
///
/// Unlike ASCII uncased search, this handles Unicode case folding,
/// like German ß matching "ss" and Turkish İ matching "i".
///
/// # Arguments
///
/// - `haystack`: The UTF-8 text to search in.
/// - `needle`: The UTF-8 pattern to search for.
///
/// # Returns
///
/// If found, returns `Some((offset, match_length))` where:
/// - `offset` is the byte position in haystack where the match starts
/// - `match_length` is the number of bytes matched in haystack, which may differ from
///   the needle length
///
/// Returns `None` if no match is found.
///
/// # Examples
///
/// Basic usage with string slices:
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let haystack = "Hello WORLD";
/// if let Some((offset, len)) = sz::utf8_uncased_search(haystack, "world") {
///     assert_eq!(offset, 6);
///     assert_eq!(len, 5);
/// }
/// ```
///
/// With a prepared needle for repeated searches:
///
/// ```
/// use stringzilla::stringzilla::{utf8_uncased_search, Utf8UncasedNeedle};
///
/// let needle = Utf8UncasedNeedle::new(b"hello");
///
/// // Analysed once in `new`, then only read by each search
/// let result1 = utf8_uncased_search(b"Hello World", &needle);
/// let result2 = utf8_uncased_search(b"HELLO there", &needle);
///
/// assert_eq!(result1, Some((0, 5)));
/// assert_eq!(result2, Some((0, 5)));
/// ```
///
pub fn utf8_uncased_search<Haystack, Needle>(haystack: Haystack, needle: Needle) -> Option<(usize, usize)>
where
    Haystack: AsRef<[u8]>,
    Needle: Utf8UncasedNeedleArg,
{
    needle.find_uncased_in(haystack.as_ref())
}

/// A needle prepared once for repeated uncased searches of UTF-8 text.
///
/// Mirrors the C `sz_utf8_uncased_needle_t`: the needle's bytes, borrowed for `'a`, and the folding
/// analysis [`Self::new`] runs once. Searches only read it, so one needle serves any number of
/// threads at once.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{utf8_uncased_search, Utf8UncasedNeedle};
///
/// let needle = Utf8UncasedNeedle::new(b"hello");
/// assert_eq!(utf8_uncased_search(b"Hello World", &needle), Some((0, 5)));
/// assert_eq!(utf8_uncased_search(b"say HELLO", &needle), Some((4, 5)));
/// ```
#[repr(C)]
pub struct Utf8UncasedNeedle<'a> {
    start: *const u8,
    length: usize,
    offset_in_unfolded: usize,
    length_in_unfolded: usize,
    folded_slice: [u8; 16],
    folded_slice_length: u8,
    probe_second: u8,
    probe_third: u8,
    script: u8,
    _needle: PhantomData<&'a [u8]>,
}

// The pointer borrows `&'a [u8]` and searches never write through it or the analysis.
unsafe impl Send for Utf8UncasedNeedle<'_> {}
unsafe impl Sync for Utf8UncasedNeedle<'_> {}

impl<'a> Utf8UncasedNeedle<'a> {
    /// Prepares `needle` for uncased search, analysing how it folds.
    pub fn new(needle: &'a [u8]) -> Self {
        let mut prepared = MaybeUninit::<Self>::zeroed();
        unsafe {
            sz_utf8_uncased_needle_init_best(
                needle.as_ptr() as *const c_void,
                needle.len(),
                prepared.as_mut_ptr(),
                enabled_cpu_capabilities_mask(),
                core::ptr::null_mut(),
            )
            .infallible();
            prepared.assume_init()
        }
    }

    /// Returns the needle bytes.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.start, self.length) }
    }

    /// Returns the length of the needle in bytes.
    #[inline]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the needle is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// Trait for types that can be used as a uncased search needle.
///
/// This trait is implemented for:
/// - Any type implementing `AsRef<[u8]>`, such as strings and byte slices
/// - [`Utf8UncasedNeedle`] references for efficient repeated searches
pub trait Utf8UncasedNeedleArg {
    /// Performs the uncased search in the given haystack.
    fn find_uncased_in(self, haystack: &[u8]) -> Option<(usize, usize)>;
}

impl<Source: AsRef<[u8]>> Utf8UncasedNeedleArg for Source {
    fn find_uncased_in(self, haystack: &[u8]) -> Option<(usize, usize)> {
        utf8_uncased_search(haystack, &Utf8UncasedNeedle::new(self.as_ref()))
    }
}

impl Utf8UncasedNeedleArg for &Utf8UncasedNeedle<'_> {
    fn find_uncased_in(self, haystack: &[u8]) -> Option<(usize, usize)> {
        let mut match_length: usize = 0;
        let mut result = core::ptr::null();
        unsafe {
            sz_utf8_uncased_search_best(
                haystack.as_ptr() as *const c_void,
                haystack.len(),
                self,
                &mut result,
                &mut match_length,
                enabled_cpu_capabilities_mask(),
                core::ptr::null_mut(),
            )
        }
        .infallible();

        if result.is_null() {
            None
        } else {
            Some((match_offset(result, haystack.as_ptr() as *const c_void), match_length))
        }
    }
}

/// Compares two UTF-8 strings in uncased manner.
///
/// Uses Unicode case folding for comparison, handling characters like
/// German ß, Turkish İ/ı, and other case variants.
///
/// # Arguments
///
/// - `first`: First UTF-8 string.
/// - `second`: Second UTF-8 string.
///
/// # Returns
///
/// - `Ordering::Less` if `first < second`
/// - `Ordering::Equal` if `first == second` uncasedly
/// - `Ordering::Greater` if `first > second`
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// use std::cmp::Ordering;
/// assert_eq!(sz::utf8_uncased_order("Hello", "HELLO"), Ordering::Equal);
/// assert_eq!(sz::utf8_uncased_order("abc", "ABD"), Ordering::Less);
/// ```
///
pub fn utf8_uncased_order<First, Second>(first: First, second: Second) -> Ordering
where
    First: AsRef<[u8]>,
    Second: AsRef<[u8]>,
{
    let first_ref = first.as_ref();
    let second_ref = second.as_ref();

    let mut result = 0;
    unsafe {
        sz_utf8_uncased_order_best(
            first_ref.as_ptr() as *const c_void,
            first_ref.len(),
            second_ref.as_ptr() as *const c_void,
            second_ref.len(),
            &mut result,
            enabled_cpu_capabilities_mask(),
            core::ptr::null_mut(),
        )
    }
    .infallible();

    match result {
        x if x < 0 => Ordering::Less,
        0 => Ordering::Equal,
        _ => Ordering::Greater,
    }
}

/// An iterator over uncased matches of a UTF-8 pattern in a string.
///
/// This iterator yields `IndexSpan` values representing the byte offset and length
/// of each match. The match length may differ from the needle length due to Unicode
/// case folding: "ß" matches "SS", as German eszett expands to two characters.
///
/// The iterator prepares the needle once, at construction. Matches start on codepoint boundaries:
/// overlapping iteration and any zero-length match advance one codepoint, so an empty needle
/// matches at every codepoint boundary, the end included.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{Utf8UncasedMatches, IndexSpan};
///
/// let haystack = b"Hello WORLD, hello world";
/// let matches: Vec<IndexSpan> = Utf8UncasedMatches::new(haystack, b"hello").collect();
/// assert_eq!(matches.len(), 2);
/// assert_eq!(matches[0], IndexSpan::new(0, 5));
/// assert_eq!(matches[1], IndexSpan::new(13, 5));
/// ```
///
/// With overlapping matches:
///
/// ```
/// use stringzilla::stringzilla::{Utf8UncasedMatches, IndexSpan};
///
/// let haystack = b"aAaAa";
/// let matches: Vec<IndexSpan> = Utf8UncasedMatches::new(haystack, b"aA").overlapping().collect();
/// assert_eq!(matches.len(), 4); // Overlapping matches
/// ```
pub struct Utf8UncasedMatches<'a, O: Overlaps = NonOverlapping> {
    needle: Utf8UncasedNeedle<'a>,
    rest: Option<&'a [u8]>, // The text after the last match's stride, `None` once exhausted
    offset: usize,          // Bytes of the haystack before `rest`
    _overlaps: PhantomData<O>,
}

impl<'a> Utf8UncasedMatches<'a, NonOverlapping> {
    /// Creates a new iterator for non-overlapping uncased matches.
    pub fn new(haystack: &'a [u8], needle: &'a [u8]) -> Self {
        Self {
            needle: Utf8UncasedNeedle::new(needle),
            rest: Some(haystack),
            offset: 0,
            _overlaps: PhantomData,
        }
    }

    /// Report overlapping matches too, a compile-time policy returning the `Overlapping` variant.
    pub fn overlapping(self) -> Utf8UncasedMatches<'a, Overlapping> {
        Utf8UncasedMatches {
            needle: self.needle,
            rest: self.rest,
            offset: self.offset,
            _overlaps: PhantomData,
        }
    }
}

impl<'a, O: Overlaps> Iterator for Utf8UncasedMatches<'a, O> {
    type Item = IndexSpan;

    fn next(&mut self) -> Option<Self::Item> {
        let rest = self.rest?;
        let Some((start, length)) = utf8_uncased_search(rest, &self.needle) else {
            self.rest = None;
            return None;
        };
        // An empty match at the end steps past it, parking the iterator.
        let stride = if length == 0 {
            rest.get(start)
                .map_or(1, |&lead| rune_length(lead).min(rest.len() - start))
        } else if O::OVERLAP {
            rune_length(rest[start]).min(length)
        } else {
            length
        };
        self.rest = rest.get(start + stride..);
        let span = IndexSpan::new(self.offset + start, length);
        self.offset += start + stride;
        Some(span)
    }
}

/// Bytes in the UTF-8 sequence `lead` starts, or one for a byte that starts none.
fn rune_length(lead: u8) -> usize {
    match lead.leading_ones() {
        length @ 2..=4 => length as usize,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;
    use alloc::vec::Vec;

    use super::*;
    use crate::sz;

    /// Folds a single codepoint into a fixed-size buffer, returning the buffer and its
    /// used length. A single codepoint case-folds to at most a handful of bytes (the
    /// longest known expansion is the Greek "ΐ" growing to 6 bytes), so a 16-byte buffer
    /// is comfortably oversized.
    fn fold_codepoint(codepoint: char) -> ([u8; 16], usize) {
        let mut source_buffer = [0u8; 4];
        let source = codepoint.encode_utf8(&mut source_buffer);
        let mut folded = [0u8; 16];
        let folded_length = sz::utf8_uncased_fold(source.as_bytes(), &mut folded[..]).unwrap();
        (folded, folded_length)
    }

    /// Independent oracle for uncased UTF-8 search. A match exists iff the fold of `needle` is a
    /// contiguous run of the fold of `haystack`; the earliest such run wins. The reported
    /// `(offset, length)` is in original haystack bytes, snapped to codepoint boundaries.
    /// Implemented by folding each haystack codepoint and remembering, for every folded byte, the
    /// original byte span of the codepoint that produced it.
    fn reference_uncased_find(haystack: &str, needle: &str) -> Option<(usize, usize)> {
        // Fixed-size accumulators sized for the short test inputs.
        const CAPACITY: usize = 512;
        let mut haystack_folded = [0u8; CAPACITY];
        // For each folded byte, the [start, end) byte range in the original haystack of the
        // codepoint that produced it.
        let mut source_starts = [0usize; CAPACITY];
        let mut source_ends = [0usize; CAPACITY];
        let mut haystack_folded_length = 0usize;

        let mut original_offset = 0usize;
        for codepoint in haystack.chars() {
            let codepoint_length = codepoint.len_utf8();
            let codepoint_start = original_offset;
            let codepoint_end = original_offset + codepoint_length;
            let (folded, folded_length) = fold_codepoint(codepoint);
            for &byte in &folded[..folded_length] {
                debug_assert!(haystack_folded_length < CAPACITY, "haystack fold overflow");
                haystack_folded[haystack_folded_length] = byte;
                source_starts[haystack_folded_length] = codepoint_start;
                source_ends[haystack_folded_length] = codepoint_end;
                haystack_folded_length += 1;
            }
            original_offset = codepoint_end;
        }

        // Fold the needle independently.
        let mut needle_folded = [0u8; CAPACITY];
        let mut needle_folded_length = 0usize;
        let mut needle_buffer = [0u8; 4];
        for codepoint in needle.chars() {
            let source = codepoint.encode_utf8(&mut needle_buffer);
            let mut folded = [0u8; 16];
            let folded_length = sz::utf8_uncased_fold(source.as_bytes(), &mut folded[..]).unwrap();
            for &byte in &folded[..folded_length] {
                debug_assert!(needle_folded_length < CAPACITY, "needle fold overflow");
                needle_folded[needle_folded_length] = byte;
                needle_folded_length += 1;
            }
        }

        let haystack_fold = &haystack_folded[..haystack_folded_length];
        let needle_fold = &needle_folded[..needle_folded_length];

        // An empty needle-fold matches at the very start with zero length.
        if needle_fold.is_empty() {
            return Some((0, 0));
        }
        if needle_fold.len() > haystack_fold.len() {
            return None;
        }

        // Slide the needle-fold over the haystack-fold; earliest run wins.
        for run_start in 0..=(haystack_fold.len() - needle_fold.len()) {
            let run_end = run_start + needle_fold.len();
            if &haystack_fold[run_start..run_end] == needle_fold {
                let offset = source_starts[run_start];
                let length = source_ends[run_end - 1] - offset;
                return Some((offset, length));
            }
        }
        None
    }

    #[test]
    fn utf8_uncased_search_crossing_expansions() {
        // Curated cross-expansion cases where folding changes byte counts and matches can
        // straddle multiple expanding codepoints. Swept across prefix paddings so the match
        // lands at varied alignments relative to the SIMD window boundaries.
        let cases: &[(&str, &str)] = &[
            ("\u{00DF}\u{00DF}", "sss"),              // ßß → "ssss", needle "sss"
            ("\u{00DF}\u{00DF}", "\u{017F}\u{00DF}"), // ßß vs ſß → "sss" inside "ssss"
            ("\u{1E9E}\u{00DF}", "ssss"),             // ẞß → "ssss"
            ("\u{1E9E}\u{00DF}", "sss"),              // ẞß → "ssss", needle "sss"
            ("\u{FB03}", "fi"),                       // ﬃ → "ffi", needle "fi"
            ("\u{FB03}", "ffi"),                      // ﬃ → "ffi"
            ("\u{FB00}\u{FB01}", "ffi"),              // ﬀﬁ → "ff" + "fi" = "fffi"
        ];
        let paddings: &[usize] = &[0, 30, 62, 63, 64, 65];

        for (haystack_core, needle) in cases {
            for &padding in paddings {
                let mut haystack = String::with_capacity(padding + haystack_core.len());
                for _ in 0..padding {
                    haystack.push('z'); // non-folding filler
                }
                haystack.push_str(haystack_core);

                let actual = sz::utf8_uncased_search(haystack.as_bytes(), needle.as_bytes());
                let expected = reference_uncased_find(&haystack, needle);
                assert_eq!(
                    actual, expected,
                    "mismatch for haystack_core={:?} needle={:?} padding={}",
                    haystack_core, needle, padding
                );
            }
        }
    }

    #[test]
    fn utf8_uncased_matches_empty_needle() {
        let matches: Vec<_> = Utf8UncasedMatches::new(b"abc", b"").collect();
        assert_eq!(matches.len(), 4);
        assert!(matches.iter().all(|span| span.length == 0));
        assert_eq!(sz::utf8_uncased_search(b"abc", b""), Some((0, 0)));
        // One match per codepoint boundary, never inside the two-byte "é".
        let boundaries = [0, 1, 3].map(|offset| IndexSpan::new(offset, 0));
        let haystack = "a\u{e9}".as_bytes();
        assert_eq!(Utf8UncasedMatches::new(haystack, b"").collect::<Vec<_>>(), boundaries);
        assert_eq!(
            Utf8UncasedMatches::new(haystack, b"").overlapping().collect::<Vec<_>>(),
            boundaries
        );
        // A truncated three-byte rune runs to the end as one step, and the end still matches.
        let truncated = [0, 1, 3].map(|offset| IndexSpan::new(offset, 0));
        assert_eq!(
            Utf8UncasedMatches::new(b"a\xE4\xB8", b"").collect::<Vec<_>>(),
            truncated
        );
    }

    #[test]
    fn utf8_uncased_needle_is_shared_across_threads() {
        fn assert_shareable<Shared: Send + Sync>(_: &Shared) {}
        let needle = Utf8UncasedNeedle::new("stra\u{df}e".as_bytes());
        assert_shareable(&needle);
        assert_eq!(needle.as_bytes(), "stra\u{df}e".as_bytes());
        assert_eq!(sz::utf8_uncased_search(b"Die STRASSE", &needle), Some((4, 7)));
    }
}
