//! Unicode normalization of UTF-8 text — NFC, NFD, NFKC, and NFKD.

use super::*;
use core::ffi::c_void;

/// Normalizes a UTF-8 string to the requested Unicode Normal Form, writing the result to a
/// destination buffer.
///
/// Covers all four standard forms: NFD, NFC, NFKD, and NFKC. NFC is the most common form on
/// the web; NFD is useful for collation. Compatibility forms (NFKD/NFKC) additionally decompose
/// ligatures and compatibility characters (e.g., U+FB03 ﬃ → "ffi").
///
/// # Arguments
///
/// * `source`: The UTF-8 string to normalize.
/// * `form`: The target Unicode normalization form.
/// * `destination`: The destination buffer to write the normalized string.
///
/// # Returns
///
/// Returns the number of bytes written to the destination buffer.
///
/// # Safety
///
/// The caller must ensure the destination buffer is large enough.
/// Use `source.len() * 18` bytes for worst-case expansion (canonical decomposition).
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// use sz::Utf8NormalForm;
/// let source = "caf\u{00E9}"; // "café" NFC (precomposed é)
/// let mut dest = vec![0u8; source.len() * 18];
/// let len = sz::utf8_norm(source, Utf8NormalForm::Nfc, &mut dest);
/// assert_eq!(&dest[..len], "caf\u{00E9}".as_bytes()); // unchanged — already NFC
/// ```
///
pub fn utf8_norm<Source, Destination>(source: Source, form: Utf8NormalForm, destination: &mut Destination) -> usize
where
    Source: AsRef<[u8]>,
    Destination: AsMut<[u8]> + ?Sized,
{
    let source_ref = source.as_ref();
    let dest_slice = destination.as_mut();

    unsafe {
        sz_utf8_norm(
            source_ref.as_ptr() as *const c_void,
            source_ref.len(),
            form as i32,
            dest_slice.as_mut_ptr() as *mut c_void,
        )
    }
}

/// Returns the byte offset of the first byte in `source` that violates the given Unicode Normal
/// Form, or `None` if `source` is already in the requested form.
///
/// This is a fast check — it does not produce the normalized output. Use it to avoid an
/// unnecessary [`utf8_norm`] call when the input is likely already normalized.
///
/// # Arguments
///
/// * `source`: The UTF-8 string to inspect.
/// * `form`: The normalization form to check against.
///
/// # Returns
///
/// * `None` if `source` already conforms to `form`.
/// * `Some(offset)` with the byte offset of the first offending byte otherwise.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// use sz::Utf8NormalForm;
/// // NFD string (decomposed): base 'e' + combining acute U+0301
/// let nfd = "cafe\u{0301}";
/// assert!(sz::utf8_find_denormalized(nfd, Utf8NormalForm::Nfc).is_some());
/// assert!(sz::utf8_find_denormalized("café", Utf8NormalForm::Nfc).is_none());
/// ```
///
pub fn utf8_find_denormalized<Source>(source: Source, form: Utf8NormalForm) -> Option<usize>
where
    Source: AsRef<[u8]>,
{
    let source_ref = source.as_ref();
    let ptr = unsafe { sz_utf8_find_denormalized(source_ref.as_ptr() as *const c_void, source_ref.len(), form as i32) };
    if ptr.is_null() {
        None
    } else {
        let offset = unsafe { (ptr as *const u8).offset_from(source_ref.as_ptr()) } as usize;
        Some(offset)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::sz::{self, *};

    // "cafe": precomposed e-acute (U+00E9, NFC) vs. base 'e' + combining acute U+0301 (NFD) - the
    // shared NFC/NFD fixture pair for the UTF-8 normalization tests.
    const CAFE_NFC: &str = "caf\u{00E9}";
    const CAFE_NFD: &str = "cafe\u{0301}";

    #[test]
    fn utf8_norm_golden_vectors() {
        // ASCII is invariant under all normalization forms.
        for form in [
            Utf8NormalForm::Nfd,
            Utf8NormalForm::Nfc,
            Utf8NormalForm::Nfkd,
            Utf8NormalForm::Nfkc,
        ] {
            let source = "Hello, world! 123";
            let mut dest = vec![0u8; source.len() * 18];
            let len = sz::utf8_norm(source, form, &mut dest);
            assert_eq!(&dest[..len], source.as_bytes(), "ASCII unchanged under {:?}", form);
        }

        // CAFE_NFC has precomposed é (U+00E9); NFC → NFC is a no-op (same bytes out).
        {
            let mut dest = vec![0u8; CAFE_NFC.len() * 18];
            let len = sz::utf8_norm(CAFE_NFC, Utf8NormalForm::Nfc, &mut dest);
            assert_eq!(&dest[..len], CAFE_NFC.as_bytes(), "café NFC→NFC unchanged");
        }

        // CAFE_NFD has decomposed é (base 'e' + combining acute U+0301); NFD → NFC must produce
        // the precomposed form.
        {
            let mut dest = vec![0u8; CAFE_NFD.len() * 18];
            let len = sz::utf8_norm(CAFE_NFD, Utf8NormalForm::Nfc, &mut dest);
            assert_eq!(&dest[..len], CAFE_NFC.as_bytes(), "café NFD→NFC gives precomposed form");
        }

        // NFD of the precomposed form must give the decomposed form.
        {
            let mut dest = vec![0u8; CAFE_NFC.len() * 18];
            let len = sz::utf8_norm(CAFE_NFC, Utf8NormalForm::Nfd, &mut dest);
            assert_eq!(&dest[..len], CAFE_NFD.as_bytes(), "café NFC→NFD gives decomposed form");
        }

        // Ligature U+FB03 ﬃ: NFKD and NFKC both decompose to "ffi".
        let ligature = "\u{FB03}"; // 3 bytes: 0xEF 0xAC 0x83
        {
            let mut dest = vec![0u8; ligature.len() * 18];
            let len = sz::utf8_norm(ligature, Utf8NormalForm::Nfkd, &mut dest);
            assert_eq!(&dest[..len], b"ffi", "ligature NFKD → ffi");
        }
        {
            let mut dest = vec![0u8; ligature.len() * 18];
            let len = sz::utf8_norm(ligature, Utf8NormalForm::Nfkc, &mut dest);
            assert_eq!(&dest[..len], b"ffi", "ligature NFKC → ffi");
        }

        // Idempotence: norm(norm(x, NFC), NFC) == norm(x, NFC).
        {
            let source = CAFE_NFD;
            let mut first = vec![0u8; source.len() * 18];
            let first_len = sz::utf8_norm(source, Utf8NormalForm::Nfc, &mut first);
            let first_result = first[..first_len].to_vec();

            let mut second = vec![0u8; first_len * 18];
            let second_len = sz::utf8_norm(&first_result[..], Utf8NormalForm::Nfc, &mut second);
            assert_eq!(&second[..second_len], &first_result[..], "NFC is idempotent");
        }
    }

    #[test]
    fn utf8_find_denormalized() {
        // NFC string: precomposed é — no violation.
        assert_eq!(
            sz::utf8_find_denormalized(CAFE_NFC, Utf8NormalForm::Nfc),
            None,
            "NFC string has no NFC violation"
        );

        // NFD string: decomposed e + combining acute U+0301.
        // The combining mark violates NFC (it should be composed with the preceding base).
        let violation = sz::utf8_find_denormalized(CAFE_NFD, Utf8NormalForm::Nfc);
        assert!(violation.is_some(), "NFD string must report an NFC violation");
        // The violation may point to the base 'e' (byte 3) or to the combining mark (byte 4);
        // either is within the suffix that must change during composition.
        assert!(
            violation.unwrap() >= 3,
            "violation offset must be ≥ 3 (at 'e' or the combining mark)"
        );

        // NFC string has no NFD violation only if it contains no precomposed characters.
        // ASCII is valid NFD.
        assert_eq!(
            sz::utf8_find_denormalized("hello", Utf8NormalForm::Nfd),
            None,
            "pure ASCII has no NFD violation"
        );
    }
}
