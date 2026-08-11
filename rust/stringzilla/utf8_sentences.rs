//! UAX-29 sentence segmentation of UTF-8 text.

use super::*;

/// Kernel behind [`Utf8Sentences`] (`sz_utf8_sentences`).
pub struct Sentences;
impl SegmenterKernel for Sentences {
    unsafe fn segment(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        sz_utf8_sentences(t, n, o, l, c, u)
    }
}

/// An iterator over UAX-29 sentences in UTF-8 text, in order.
///
/// Unlike whitespace splitting, the sentences tile the input: every byte belongs to exactly one
/// sentence, so consecutive sentences are contiguous and no empty slices are produced. Follows the
/// Unicode UAX-29 rules.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::Utf8Sentences;
///
/// let sentences: Vec<&[u8]> = Utf8Sentences::new(b"Hi. Bye.").collect();
/// assert_eq!(sentences, vec![&b"Hi. "[..], &b"Bye."[..]]);
/// ```
pub type Utf8Sentences<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> = Utf8Segments<'a, Sentences, STEPS>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stringzilla::fixtures::*;
    use crate::stringzilla::utf8_tokens::tests::assert_steps_invariant;
    use crate::sz::*;

    // A CR-LF pair and a U+2028 line separator: both Sep (force sentence and line breaks); CR-LF is one grapheme.
    const PROSE_MICRO_HARDBREAKS: &str = "A.\u{d}\u{a}B.\u{2028}C.";

    #[test]
    fn utf8_prose_sentence_counts() {
        assert_eq!(PROSE_HOTEL_REVIEW.as_bytes().sz_utf8_sentences().count(), 5);
        assert_eq!(PROSE_CONCERT_POST.as_bytes().sz_utf8_sentences().count(), 4);
        assert_eq!(PROSE_NEWS_LEDE.as_bytes().sz_utf8_sentences().count(), 6);
        assert_eq!(PROSE_MICRO_HARDBREAKS.as_bytes().sz_utf8_sentences().count(), 3);
    }

    #[test]
    fn iter_sentence_utf8_splits_steps_invariance() {
        assert_steps_invariant::<Sentences>(b"Hi, world! A second sentence.");
    }
}
