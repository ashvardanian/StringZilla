//! Unicode case folding of UTF-8 text.
//!
//! File: rust/stringzilla/utf8_uncased_fold.rs
//! Author: Ash Vardanian

use core::ffi::c_void;

use super::*;

/// Applies Unicode case folding to a UTF-8 string, writing the result to a target buffer.
///
/// Case folding normalizes text for uncased comparisons by mapping uppercase letters to their
/// lowercase equivalents and handling special cases like German U+00DF → ss expansion.
///
/// # Arguments
///
/// - `source`: The UTF-8 string to case-fold.
/// - `target`: The target buffer to write the case-folded string.
///
/// # Returns
///
/// Returns the number of bytes written to the target buffer.
///
/// # Errors
///
/// The C kernel writes without a capacity, so `target` must hold the worst-case 3:1 expansion of
/// `source.len() * 3` bytes whatever the input, or [`Status::UnexpectedDimensions`] is returned.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let source = "HELLO WORLD";
/// let mut dest = [0u8; 33];
/// let len = sz::utf8_uncased_fold(source, &mut dest).unwrap();
/// assert_eq!(&dest[..len], b"hello world");
/// ```
///
pub fn utf8_uncased_fold<Source, Target>(source: Source, target: &mut Target) -> Result<usize, Status>
where
    Source: AsRef<[u8]>,
    Target: AsMut<[u8]> + ?Sized,
{
    let source_ref = source.as_ref();
    let target_slice = target.as_mut();
    let worst_case = source_ref.len().checked_mul(3).ok_or(Status::OverflowRisk)?;
    if target_slice.len() < worst_case {
        return Err(Status::UnexpectedDimensions);
    }

    let mut written = 0;
    unsafe {
        sz_utf8_uncased_fold_best(
            source_ref.as_ptr() as *const c_void,
            source_ref.len(),
            target_slice.as_mut_ptr() as *mut c_void,
            &mut written,
            Capabilities::CPUS.bits(),
            core::ptr::null_mut(),
        )
    }
    .check()?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use crate::sz;

    #[test]
    fn utf8_uncased_fold_golden_vectors() {
        // One probe per kernel family: ASCII, Latin-1 at C3, Latin Extended at C4/C6, Greek with
        // final sigma, Cyrillic, Vietnamese at E1 BA, letterlike symbols, ligature expansions, and
        // the post-Unicode-15 Garay block of 4-byte sequences.
        let golden: &[(&str, &[u8])] = &[
            ("HeLLo", b"hello"),                                           // ASCII fast path
            ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", b"abcdefghijklmnopqrstuvwxyz"), // >16B ASCII: SIMD fold loop
            ("Hello, WASM World! 12345.", b"hello, wasm world! 12345."),   // >16B mixed: only A-Z fold
            // Long ASCII run, then a multi-byte codepoint, then more ASCII: SIMD → serial
            // → scalar tail.
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
            let mut target = vec![0u8; source.len() * 3];
            let folded_length = sz::utf8_uncased_fold(source, &mut target[..]).unwrap();
            assert_eq!(&target[..folded_length], *expected, "folding {:?}", source);
        }

        // Returned length tracks expansion: ẞ shrinks 3 → 2 bytes, ΐ grows 2 → 6 bytes
        let mut target = [0u8; 16];
        assert_eq!(sz::utf8_uncased_fold("\u{1E9E}", &mut target), Ok(2));
        let folded_length = sz::utf8_uncased_fold("\u{0390}", &mut target).unwrap();
        assert_eq!(folded_length, 6);
        assert_eq!(&target[..folded_length], "\u{03B9}\u{0308}\u{0301}".as_bytes());

        // Refused below the worst case although ASCII fits, as the kernel takes no capacity.
        assert_eq!(
            sz::utf8_uncased_fold("HELLO", &mut target[..14]),
            Err(sz::Status::UnexpectedDimensions)
        );
    }
}
