//! Case-insensitive UTF-8 search, ordering, and match iteration.

use super::*;
use core::cell::UnsafeCell;
use core::cmp::Ordering;
use core::ffi::c_void;
use core::marker::PhantomData;

/// Performs uncased search for `needle` in UTF-8 `haystack`.
///
/// Unlike ASCII uncased search, this handles Unicode case folding
/// (e.g., German ß matches "ss", Turkish İ matches "i").
///
/// # Arguments
///
/// * `haystack`: The UTF-8 text to search in.
/// * `needle`: The UTF-8 pattern to search for.
///
/// # Returns
///
/// If found, returns `Some((offset, matched_length))` where:
/// - `offset` is the byte position in haystack where the match starts
/// - `matched_length` is the number of bytes matched in haystack (may differ from needle length)
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
/// With a pre-compiled needle for repeated searches:
///
/// ```
/// use stringzilla::stringzilla::{utf8_uncased_search, Utf8UncasedNeedle};
///
/// let needle = Utf8UncasedNeedle::new(b"hello");
///
/// // Metadata is computed once, reused for subsequent searches
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

/// Internal metadata for uncased UTF-8 search operations.
///
/// This structure caches pre-computed information about the needle for reuse
/// across multiple searches. Zero-initialization (default) triggers automatic
/// analysis on first use.
///
/// Matches C's `sz_utf8_uncased_needle_metadata_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Utf8UncasedNeedleMetadata {
    // sz_size_t offset_in_unfolded
    offset_in_unfolded: usize,
    // sz_size_t length_in_unfolded
    length_in_unfolded: usize,
    // sz_u8_t folded_slice[16]
    folded_slice: [u8; 16],
    // sz_u8_t folded_slice_length
    folded_slice_length: u8,
    // sz_u8_t probe_second
    probe_second: u8,
    // sz_u8_t probe_third
    probe_third: u8,
    // sz_u8_t kernel_id
    kernel_id: u8,
}

impl Utf8UncasedNeedleMetadata {
    /// The state that requests analysis on first use, usable in a `const` unlike the `Default` it backs.
    pub(crate) const UNANALYZED: Self = Self {
        offset_in_unfolded: 0,
        length_in_unfolded: 0,
        folded_slice: [0; 16],
        folded_slice_length: 0,
        probe_second: 0,
        probe_third: 0,
        kernel_id: 0, // sz_utf8_uncased_rune_unknown_k = 0, triggers analysis
    };
}

impl Default for Utf8UncasedNeedleMetadata {
    fn default() -> Self {
        Self::UNANALYZED
    }
}

/// Pre-compiled uncased search pattern for UTF-8 strings.
///
/// Caches metadata for efficient repeated searches with the same needle.
/// Useful when searching multiple haystacks for the same pattern.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{utf8_uncased_search, Utf8UncasedNeedle};
///
/// let needle = Utf8UncasedNeedle::new(b"hello");
/// let haystack1 = b"Hello World";
/// let haystack2 = b"HELLO there";
///
/// // Metadata is computed once on first search, reused for subsequent searches
/// let result1 = utf8_uncased_search(haystack1, &needle);
/// let result2 = utf8_uncased_search(haystack2, &needle);
///
/// assert!(result1.is_some());
/// assert!(result2.is_some());
/// ```
pub struct Utf8UncasedNeedle<'a> {
    needle: &'a [u8],
    metadata: UnsafeCell<Utf8UncasedNeedleMetadata>,
}

impl<'a> Utf8UncasedNeedle<'a> {
    /// Creates a new pre-compiled uncased needle.
    ///
    /// The metadata will be computed lazily on first use.
    #[inline]
    pub const fn new(needle: &'a [u8]) -> Self {
        Self {
            needle,
            metadata: UnsafeCell::new(Utf8UncasedNeedleMetadata::UNANALYZED),
        }
    }

    /// Returns the needle bytes.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8] {
        self.needle
    }

    /// Returns the length of the needle in bytes.
    #[inline]
    pub const fn len(&self) -> usize {
        self.needle.len()
    }

    /// Returns true if the needle is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.needle.is_empty()
    }

    /// Internal: returns a mutable pointer to the metadata for FFI calls.
    #[inline]
    pub(crate) fn metadata_ptr(&self) -> *mut Utf8UncasedNeedleMetadata {
        self.metadata.get()
    }
}

// Safety: The metadata is only mutated through FFI during search operations,
// which internally synchronize access. The needle reference is immutable.
unsafe impl<'a> Send for Utf8UncasedNeedle<'a> {}
unsafe impl<'a> Sync for Utf8UncasedNeedle<'a> {}

/// Trait for types that can be used as a uncased search needle.
///
/// This trait is implemented for:
/// - Any type implementing `AsRef<[u8]>` (strings, byte slices, etc.)
/// - [`Utf8UncasedNeedle`] references for efficient repeated searches
pub trait Utf8UncasedNeedleArg {
    /// Performs the uncased search in the given haystack.
    fn find_uncased_in(self, haystack: &[u8]) -> Option<(usize, usize)>;
}

impl<Source: AsRef<[u8]>> Utf8UncasedNeedleArg for Source {
    fn find_uncased_in(self, haystack: &[u8]) -> Option<(usize, usize)> {
        let needle_ref = self.as_ref();
        let mut matched_length: usize = 0;
        let mut needle_metadata = Utf8UncasedNeedleMetadata::default();

        let result = unsafe {
            sz_utf8_uncased_search(
                haystack.as_ptr() as *const c_void,
                haystack.len(),
                needle_ref.as_ptr() as *const c_void,
                needle_ref.len(),
                &mut needle_metadata,
                &mut matched_length,
            )
        };

        if result.is_null() {
            None
        } else {
            let offset = unsafe { result.offset_from(haystack.as_ptr() as *const c_void) };
            Some((offset as usize, matched_length))
        }
    }
}

impl<'a, 'b> Utf8UncasedNeedleArg for &'b Utf8UncasedNeedle<'a> {
    fn find_uncased_in(self, haystack: &[u8]) -> Option<(usize, usize)> {
        let needle_bytes = self.as_bytes();
        let mut matched_length: usize = 0;

        let result = unsafe {
            sz_utf8_uncased_search(
                haystack.as_ptr() as *const c_void,
                haystack.len(),
                needle_bytes.as_ptr() as *const c_void,
                needle_bytes.len(),
                &mut *self.metadata_ptr(),
                &mut matched_length,
            )
        };

        if result.is_null() {
            None
        } else {
            let offset = unsafe { result.offset_from(haystack.as_ptr() as *const c_void) };
            Some((offset as usize, matched_length))
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
/// * `first`: First UTF-8 string.
/// * `second`: Second UTF-8 string.
///
/// # Returns
///
/// * `Ordering::Less` if `first < second`
/// * `Ordering::Equal` if `first == second` (uncasedly)
/// * `Ordering::Greater` if `first > second`
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

    let result = unsafe {
        sz_utf8_uncased_order(
            first_ref.as_ptr() as *const c_void,
            first_ref.len(),
            second_ref.as_ptr() as *const c_void,
            second_ref.len(),
        )
    };

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
/// case folding (e.g., "ß" matches "SS", German eszett expands to two characters).
///
/// The iterator caches needle metadata internally for efficient repeated searches.
///
/// Unlike [`find`]/[`rfind`], the underlying UTF-8 search reports an empty needle as a real
/// zero-length match rather than "not found". Each zero-length match still advances the
/// search position by at least one byte, so the iterator always terminates instead of
/// looping forever on the same spot.
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
    haystack: &'a [u8],
    needle: &'a [u8],
    metadata: Utf8UncasedNeedleMetadata,
    position: usize,
    _overlaps: PhantomData<O>,
}

impl<'a> Utf8UncasedMatches<'a, NonOverlapping> {
    /// Creates a new iterator for non-overlapping uncased matches.
    pub fn new(haystack: &'a [u8], needle: &'a [u8]) -> Self {
        Self {
            haystack,
            needle,
            metadata: Utf8UncasedNeedleMetadata::default(),
            position: 0,
            _overlaps: PhantomData,
        }
    }

    /// Report overlapping matches too (compile-time policy; returns the `Overlapping` variant).
    pub fn overlapping(self) -> Utf8UncasedMatches<'a, Overlapping> {
        Utf8UncasedMatches {
            haystack: self.haystack,
            needle: self.needle,
            metadata: self.metadata,
            position: self.position,
            _overlaps: PhantomData,
        }
    }
}

impl<'a, O: Overlaps> Iterator for Utf8UncasedMatches<'a, O> {
    type Item = IndexSpan;

    fn next(&mut self) -> Option<Self::Item> {
        // Empty needle also matches at `haystack.len()`; park one past it once exhausted.
        if self.position > self.haystack.len() {
            return None;
        }

        let remaining = &self.haystack[self.position..];
        let mut matched_length: usize = 0;

        let result = unsafe {
            sz_utf8_uncased_search(
                remaining.as_ptr() as *const c_void,
                remaining.len(),
                self.needle.as_ptr() as *const c_void,
                self.needle.len(),
                &mut self.metadata,
                &mut matched_length,
            )
        };

        if result.is_null() {
            self.position = self.haystack.len() + 1;
            None
        } else {
            let offset_in_remaining = unsafe { result.offset_from(remaining.as_ptr() as *const c_void) } as usize;
            let absolute_offset = self.position + offset_in_remaining;

            // Advance position for next search. A zero-length match (empty needle) must still
            // advance by at least one byte in the non-overlapping case, or this would loop
            // forever re-matching the same position; the overlapping case already always
            // advances by 1 regardless of `matched_length`.
            if O::OVERLAP {
                self.position = absolute_offset + 1;
            } else {
                self.position = absolute_offset + matched_length.max(1);
            }

            Some(IndexSpan::new(absolute_offset, matched_length))
        }
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
        let folded_length = sz::utf8_uncased_fold(source.as_bytes(), &mut folded[..]);
        debug_assert!(folded_length <= folded.len(), "fold expansion exceeded buffer");
        (folded, folded_length)
    }

    /// Independent oracle for uncased UTF-8 search. A match exists iff the fold of
    /// `needle` is a contiguous run of the fold of `haystack`; the earliest such run wins.
    /// The reported `(offset, length)` is in ORIGINAL haystack bytes, snapped to codepoint
    /// boundaries. Implemented by folding each haystack codepoint and remembering, for every
    /// folded byte, the original byte span of the codepoint that produced it.
    fn reference_uncased_find(haystack: &str, needle: &str) -> Option<(usize, usize)> {
        // Fixed-size accumulators sized for the short test inputs.
        const CAPACITY: usize = 512;
        let mut haystack_folded = [0u8; CAPACITY];
        // For each folded byte, the [start, end) byte range in the ORIGINAL haystack of the
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
            for byte_index in 0..folded_length {
                debug_assert!(haystack_folded_length < CAPACITY, "haystack fold overflow");
                haystack_folded[haystack_folded_length] = folded[byte_index];
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
            let folded_length = sz::utf8_uncased_fold(source.as_bytes(), &mut folded[..]);
            for byte_index in 0..folded_length {
                debug_assert!(needle_folded_length < CAPACITY, "needle fold overflow");
                needle_folded[needle_folded_length] = folded[byte_index];
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
    }
}
