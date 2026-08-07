//! UAX-14 line break opportunities in UTF-8 text.

use super::*;

/// Kernel behind [`Utf8Linebreaks`] (`sz_utf8_linebreaks`).
pub struct Linebreaks;
impl SegmenterKernel for Linebreaks {
    unsafe fn segment(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        sz_utf8_linebreaks(t, n, o, l, c, u)
    }
}

/// An iterator over UAX-14 line break opportunities in UTF-8 text, in order.
///
/// Unlike whitespace splitting, the lines tile the input: every byte belongs to exactly one line, so
/// consecutive lines are contiguous and no empty slices are produced. Follows the Unicode TR14 rules.
///
/// Each yielded segment ends at a TR14 break opportunity, including soft breaks where a renderer *may*
/// wrap but is not required to. To split only on hard line breaks (the "splitlines" behaviour), use the
/// newline API ([`StringZillableUnary::sz_utf8_split_newlines`]) instead.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::Utf8Linebreaks;
///
/// let lines: Vec<&[u8]> = Utf8Linebreaks::new(b"Hi\nBye").collect();
/// assert_eq!(lines, vec![&b"Hi\n"[..], &b"Bye"[..]]);
/// ```
pub type Utf8Linebreaks<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> = Utf8Segments<'a, Linebreaks, STEPS>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stringzilla::fixtures::*;
    use crate::stringzilla::utf8_tokens::tests::assert_steps_invariant;
    use crate::sz::*;

    // Science abstract: NFKC ligatures/superscripts/Roman/full-width, Kelvin and Angstrom singletons, NBSP, WJ + ZWSP.
    const PROSE_SCIENCE_ABSTRACT: &str = concat!(
        "The \u{fb01}lm grew at 300\u{a0}\u{212a} on a 5\u{a0}\u{212b} buffer (\u{2248} 2\u{b2} monolayer",
        "s). Section \u{216b} covers the \u{ff21}-phase; see Fig. 2 for the \u{3a3}-band dispersion. Resi",
        "stivity scaled as T\u{b2}, vanishing at the 4.2\u{a0}\u{212a} transition. Full dataset: doi:10.1",
        "000\u{2060}/\u{200b}xyz (mirror in Box \u{2461})."
    );

    #[test]
    fn utf8_prose_linebreak_counts() {
        assert_eq!(PROSE_HOTEL_REVIEW.as_bytes().sz_utf8_linebreaks().count(), 45);
        assert_eq!(PROSE_SCIENCE_ABSTRACT.as_bytes().sz_utf8_linebreaks().count(), 43);
        assert_eq!(PROSE_NEWS_LEDE.as_bytes().sz_utf8_linebreaks().count(), 32);
    }

    #[test]
    fn iter_linebreak_utf8_splits_steps_invariance() {
        assert_steps_invariant::<Linebreaks>(b"Hi, world! A second sentence.");
    }
}
