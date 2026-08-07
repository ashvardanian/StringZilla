//! Unicode case folding of UTF-8 text.

use super::*;
use core::ffi::c_void;

/// Applies Unicode case folding to a UTF-8 string, writing the result to a destination buffer.
///
/// Case folding normalizes text for uncased comparisons by mapping uppercase letters
/// to their lowercase equivalents and handling special cases like German U+00DF -> ss expansion.
///
/// # Arguments
///
/// * `source`: The UTF-8 string to case-fold.
/// * `destination`: The destination buffer to write the case-folded string.
///
/// # Returns
///
/// Returns the number of bytes written to the destination buffer.
///
/// # Safety
///
/// The caller must ensure the destination buffer is large enough.
/// Use `source.len() * 3` bytes for worst-case 3:1 expansion ratio.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let source = "HELLO WORLD";
/// let mut dest = [0u8; 32];
/// let len = sz::utf8_uncased_fold(source, &mut dest);
/// assert_eq!(&dest[..len], b"hello world");
/// ```
///
pub fn utf8_uncased_fold<Source, Destination>(source: Source, destination: &mut Destination) -> usize
where
    Source: AsRef<[u8]>,
    Destination: AsMut<[u8]> + ?Sized,
{
    let source_ref = source.as_ref();
    let dest_slice = destination.as_mut();

    unsafe {
        sz_utf8_uncased_fold(
            source_ref.as_ptr() as *const c_void,
            source_ref.len(),
            dest_slice.as_mut_ptr() as *mut c_void,
        )
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use super::*;
    use crate::sz;

    #[test]
    fn utf8_uncased_fold_golden_vectors() {
        // One probe per kernel family: ASCII, Latin-1 (C3), Latin Extended (C4/C6),
        // Greek (incl. final sigma), Cyrillic, Vietnamese (E1 BA), letterlike symbols,
        // ligature expansions, and the post-Unicode-15 Garay block (4-byte sequences).
        let golden: &[(&str, &[u8])] = &[
            ("HeLLo", b"hello"),                                           // ASCII fast path
            ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", b"abcdefghijklmnopqrstuvwxyz"), // >16B ASCII: SIMD fold loop
            ("Hello, WASM World! 12345.", b"hello, wasm world! 12345."),   // >16B mixed: only A-Z fold
            // Long ASCII run, then a multi-byte codepoint, then more ASCII: SIMD → serial → scalar tail.
            (
                "LONG ASCII PREFIX \u{00C4} SUFFIX",
                "long ascii prefix \u{00E4} suffix".as_bytes(),
            ),
            ("\u{00DF}", b"ss"),                   // ß → ss expansion
            ("\u{1E9E}", b"ss"),                   // ẞ → ss (E1 BA lead bytes)
            ("\u{03A3}", "\u{03C3}".as_bytes()),   // Σ → σ
            ("\u{03C2}", "\u{03C3}".as_bytes()),   // final sigma ς → σ
            ("\u{FB03}", b"ffi"),                  // ﬃ ligature → ffi
            ("\u{041A}", "\u{043A}".as_bytes()),   // Cyrillic К → к
            ("\u{00C4}", "\u{00E4}".as_bytes()),   // Ä → ä (C3 lead byte)
            ("\u{0110}", "\u{0111}".as_bytes()),   // Đ → đ (C4 lead byte)
            ("\u{0111}", "\u{0111}".as_bytes()),   // đ → đ (already folded)
            ("\u{01A0}", "\u{01A1}".as_bytes()),   // Ơ → ơ (C6 lead byte)
            ("\u{01A1}", "\u{01A1}".as_bytes()),   // ơ → ơ (already folded)
            ("\u{1EA0}", "\u{1EA1}".as_bytes()),   // Ạ → ạ (E1 BA lead bytes)
            ("\u{1EA1}", "\u{1EA1}".as_bytes()),   // ạ → ạ (already folded)
            ("\u{212A}", b"k"),                    // Kelvin sign K → k
            ("\u{10D50}", "\u{10D70}".as_bytes()), // Garay capital Ca → small Ca
        ];
        for (source, expected) in golden {
            let mut destination = vec![0u8; source.len() * 3];
            let folded_length = sz::utf8_uncased_fold(source, &mut destination[..]);
            assert_eq!(&destination[..folded_length], *expected, "folding {:?}", source);
        }

        // Returned length tracks expansion: ẞ shrinks 3 → 2 bytes, ΐ grows 2 → 6 bytes
        let mut destination = [0u8; 16];
        assert_eq!(sz::utf8_uncased_fold("\u{1E9E}", &mut destination), 2);
        let folded_length = sz::utf8_uncased_fold("\u{0390}", &mut destination);
        assert_eq!(folded_length, 6);
        assert_eq!(&destination[..folded_length], "\u{03B9}\u{0308}\u{0301}".as_bytes());
    }
}
