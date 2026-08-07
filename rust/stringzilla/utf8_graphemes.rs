//! UAX-29 grapheme cluster segmentation of UTF-8 text.

use super::*;

/// Kernel behind [`Utf8Graphemes`] (`sz_utf8_graphemes`).
pub struct Graphemes;
impl SegmenterKernel for Graphemes {
    unsafe fn segment(t: *const c_void, n: usize, o: *mut usize, l: *mut usize, c: usize, u: *mut usize) -> usize {
        sz_utf8_graphemes(t, n, o, l, c, u)
    }
}

/// An iterator over UAX-29 grapheme clusters in UTF-8 text, in order.
///
/// Unlike whitespace splitting, the grapheme clusters tile the input: every byte belongs to exactly one
/// grapheme cluster, so consecutive clusters are contiguous and no empty slices are produced. Follows the
/// Unicode UAX-29 rules.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::Utf8Graphemes;
///
/// let graphemes: Vec<&[u8]> = Utf8Graphemes::new(b"Hi!").collect();
/// assert_eq!(graphemes, vec![&b"H"[..], &b"i"[..], &b"!"[..]]);
/// ```
pub type Utf8Graphemes<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> = Utf8Segments<'a, Graphemes, STEPS>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stringzilla::fixtures::*;
    use crate::stringzilla::utf8_tokens::tests::assert_steps_invariant;
    use crate::sz::*;

    // Pride caption: a ZWJ family and VS16 rainbow flag, a skin-tone modifier, a keycap, an odd regional-indicator run.
    const PROSE_PRIDE_CAPTION: &str = concat!(
        "Best Pride yet \u{1f3f3}\u{fe0f}\u{200d}\u{1f308} \u{2014} the whole crew showed up. Even my par",
        "ents \u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}\u{200d}\u{1f466} and grandma \u{1f44d}\u{1f3fd}",
        " came through! We met at booth 5\u{fe0f}\u{20e3}, then waved every flag we packed \u{1f1fa}",
        "\u{1f1f8}\u{1f1ef}\u{1f1f5}\u{1f1eb}. Texting \u{260e}\u{fe0e} over calling \u{2708}\u{fe0f} all",
        " day; 10/10, would march again."
    );
    // Devanagari note: a virama conjunct, ZWJ/ZWNJ half-forms, a spacing vowel sign, and an NFKC vulgar fraction.
    const PROSE_DEVANAGARI_TIP: &str = concat!(
        "Quick Devanagari tip: \u{915}\u{94d}\u{937} is one cluster (\u{915} + \u{94d} + \u{937}), not th",
        "ree. Force the half-form with ZWJ \u{2014} \u{915}\u{94d}\u{200d}\u{937} \u{2014} or split it wi",
        "th ZWNJ \u{2014} \u{915}\u{94d}\u{200c}\u{937}. The same logic hits \u{915}\u{94d}\u{937}\u{924}",
        "\u{94d}\u{930}\u{93f}\u{92f} and spacing vowel signs like \u{915}\u{940}. Renderers disagree, so",
        " test (\u{bd} the bugs are font bugs) before you ship!"
    );
    // Two Prepend characters (Arabic number sign, Malayalam dot-reph): clusters fewer than codepoints.
    const PROSE_MICRO_PREPEND: &str = "\u{600}\u{664} \u{d4e}\u{d15}";

    #[test]
    fn utf8_prose_grapheme_counts() {
        assert_eq!(PROSE_PRIDE_CAPTION.as_bytes().sz_utf8_graphemes().count(), 206);
        assert_eq!(PROSE_DEVANAGARI_TIP.as_bytes().sz_utf8_graphemes().count(), 252);
        assert_eq!(PROSE_CONCERT_POST.as_bytes().sz_utf8_graphemes().count(), 134);
        assert_eq!(PROSE_RTL_SCRIPTS.as_bytes().sz_utf8_graphemes().count(), 256);
        assert_eq!(PROSE_MICRO_PREPEND.as_bytes().sz_utf8_graphemes().count(), 3);
        // Codepoints are not clusters: the emoji paragraph has more runes than grapheme clusters.
        assert_eq!(PROSE_PRIDE_CAPTION.as_bytes().sz_utf8_runes().iter().count(), 222);
        assert!(
            PROSE_PRIDE_CAPTION.as_bytes().sz_utf8_runes().iter().count()
                > PROSE_PRIDE_CAPTION.as_bytes().sz_utf8_graphemes().count()
        );
    }

    #[test]
    fn iter_grapheme_utf8_splits_steps_invariance() {
        assert_steps_invariant::<Graphemes>(b"Hi, world! A second sentence.");
    }
}
