//! UAX-29 word segmentation of UTF-8 text.

use super::*;

/// Kernel behind [`Utf8Wordbreaks`] (`sz_utf8_wordbreaks`).
pub struct Wordbreaks;
impl SegmenterKernel for Wordbreaks {
    unsafe fn segment(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        sz_utf8_wordbreaks(t, n, o, l, c, u)
    }
}

/// An iterator over UAX-29 words in UTF-8 text, in order.
///
/// Unlike whitespace splitting, the words tile the input: every byte belongs to exactly one word, so
/// consecutive words are contiguous and no empty slices are produced. Follows the Unicode UAX-29 rules.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::Utf8Wordbreaks;
///
/// let words: Vec<&[u8]> = Utf8Wordbreaks::new(b"Hi, world").collect();
/// assert_eq!(words, vec![&b"Hi"[..], &b","[..], &b" "[..], &b"world"[..]]);
/// ```
pub type Utf8Wordbreaks<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> = Utf8Segments<'a, Wordbreaks, STEPS>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stringzilla::fixtures::*;
    use crate::stringzilla::utf8_tokens::tests::assert_steps_invariant;
    use crate::sz::*;

    // A U+2019 contraction tiles as a single word, like the ASCII apostrophe.
    const PROSE_MICRO_APOSTROPHE: &str = "it\u{2019}s worth it";

    #[test]
    fn utf8_prose_wordbreak_counts() {
        assert_eq!(PROSE_HOTEL_REVIEW.as_bytes().sz_utf8_wordbreaks().count(), 100);
        assert_eq!(PROSE_NEWS_LEDE.as_bytes().sz_utf8_wordbreaks().count(), 83);
        assert_eq!(PROSE_CONCERT_POST.as_bytes().sz_utf8_wordbreaks().count(), 69);
        assert_eq!(PROSE_RTL_SCRIPTS.as_bytes().sz_utf8_wordbreaks().count(), 98);
        assert_eq!(PROSE_MICRO_APOSTROPHE.as_bytes().sz_utf8_wordbreaks().count(), 5);
    }

    #[test]
    fn iter_word_utf8_splits_steps_invariance() {
        assert_steps_invariant::<Wordbreaks>(b"Hi, world! A second sentence.");
    }
}
