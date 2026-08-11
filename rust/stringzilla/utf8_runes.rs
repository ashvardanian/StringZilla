//! Codepoint-level UTF-8 machinery — counting, seeking, decoding, and rune iteration.

use super::*;
use core::ffi::c_void;

/// Unpacks a UTF-8 byte sequence into UTF-32 codepoints.
///
/// This function decodes UTF-8 encoded text into individual Unicode codepoints, storing them in a u32 array.
/// It fills the output buffer (or drains the input) in a single call, looping internally regardless of how many
/// byte-widths the text mixes. Ill-formed bytes decode to the replacement character U+FFFD (one per maximal
/// ill-formed subpart), so every written value is a valid Unicode scalar value; a well-formed but truncated
/// trailing sequence is left unconsumed so a streaming caller can resume once more bytes arrive.
///
/// # Arguments
///
/// * `text`: The UTF-8 encoded byte slice to decode.
/// * `runes`: Output buffer to store decoded codepoints.
///
/// # Returns
///
/// A tuple `(bytes_consumed, runes_unpacked)` where:
/// - `bytes_consumed` is the number of bytes processed from `text`
/// - `runes_unpacked` is the number of codepoints written to `runes`
///
/// # Examples
///
/// Processing pure ASCII text (most common case, single chunk):
/// ```
/// use stringzilla::stringzilla as sz;
/// let text = "Hello World!";
/// let mut runes = [0u32; 16];
/// let (bytes, count) = sz::utf8_decode(text.as_bytes(), &mut runes);
/// assert_eq!(count, 12);  // All 12 ASCII characters
/// assert_eq!(bytes, 12);  // 12 bytes consumed
/// assert_eq!(runes[0], 'H' as u32);
/// assert_eq!(runes[11], '!' as u32);
/// ```
///
/// Each call fills the output buffer or drains the input; call repeatedly (resuming at `bytes_consumed`)
/// to process a string longer than the buffer:
/// ```
/// use stringzilla::stringzilla as sz;
/// let text = "Hi世界";  // 2 ASCII + 2 CJK
/// let bytes = text.as_bytes();
/// let mut runes = [0u32; 16];
/// let mut all_runes = Vec::new();
/// let mut offset = 0;
/// while offset < bytes.len() {
///     let (consumed, count) = sz::utf8_decode(&bytes[offset..], &mut runes);
///     all_runes.extend_from_slice(&runes[..count]);
///     offset += consumed;
/// }
/// assert_eq!(all_runes.len(), 4);  // 2 ASCII + 2 CJK = 4 codepoints
/// ```
///
pub fn utf8_decode(text: &[u8], runes: &mut [u32]) -> (usize, usize) {
    let mut runes_unpacked: usize = 0;

    let result = unsafe {
        sz_utf8_decode(
            text.as_ptr() as *const c_void,
            text.len(),
            runes.as_mut_ptr(),
            runes.len(),
            &mut runes_unpacked,
        )
    };

    let bytes_consumed = if result.is_null() {
        0
    } else {
        unsafe { result.offset_from(text.as_ptr() as *const c_void) as usize }
    };

    (bytes_consumed, runes_unpacked)
}

/// Counts the number of UTF-8 characters in the text.
///
/// This function efficiently counts UTF-8 characters by identifying character start bytes
/// (non-continuation bytes). Uses SIMD acceleration when available.
///
/// # Arguments
///
/// * `text`: The UTF-8 encoded byte slice to count characters in.
///
/// # Returns
///
/// The number of UTF-8 characters (codepoints) in the text.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// let text = "Hello";
/// assert_eq!(sz::count_utf8(text), 5);
///
/// let text_unicode = "Hello🌍";
/// assert_eq!(sz::count_utf8(text_unicode), 6);
///
/// let text_cjk = "你好世界";
/// assert_eq!(sz::count_utf8(text_cjk), 4);
/// ```
pub fn count_utf8<Text>(text: Text) -> usize
where
    Text: AsRef<[u8]>,
{
    let text_ref = text.as_ref();
    let text_pointer = text_ref.as_ptr() as *const c_void;
    let text_length = text_ref.len();

    unsafe { sz_utf8_count(text_pointer, text_length) }
}

/// Finds the byte offset of the Nth UTF-8 character (0-indexed).
///
/// This function efficiently locates the Nth UTF-8 character without decoding
/// the entire string. Uses SIMD acceleration when available.
///
/// # Arguments
///
/// * `text`: The UTF-8 encoded byte slice to search.
/// * `n`: The 0-based index of the character to find.
///
/// # Returns
///
/// An `Option<usize>` containing the byte offset of the Nth character.
/// Returns `None` if the string has fewer than N+1 characters.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// let text = "Hello";
/// assert_eq!(sz::find_nth_utf8(text, 0), Some(0)); // 'H'
/// assert_eq!(sz::find_nth_utf8(text, 4), Some(4)); // 'o'
/// assert_eq!(sz::find_nth_utf8(text, 5), None);
///
/// let text_unicode = "Hello🌍";
/// assert_eq!(sz::find_nth_utf8(text_unicode, 5), Some(5)); // 🌍 starts at byte 5
/// assert_eq!(sz::find_nth_utf8(text_unicode, 6), None);
/// ```
pub fn find_nth_utf8<Text>(text: Text, n: usize) -> Option<usize>
where
    Text: AsRef<[u8]>,
{
    let text_ref = text.as_ref();
    let text_pointer = text_ref.as_ptr() as *const c_void;
    let text_length = text_ref.len();

    let result = unsafe { sz_utf8_seek(text_pointer, text_length, n) };

    if result.is_null() {
        None
    } else {
        let offset = unsafe { (result as *const u8).offset_from(text_pointer as *const u8) }
            .try_into()
            .unwrap();
        Some(offset)
    }
}

/// Lazy UTF-8 character view with SIMD-accelerated operations.
///
/// Provides O(1) construction with lazy character counting and efficient random access.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// let text = "Hello🌍";
/// let view = sz::Utf8View::new(text.as_bytes());
///
/// // Lazy character count (computed once, then cached)
/// assert_eq!(view.len(), 6);
///
/// // Random access to byte offset of Nth character
/// assert_eq!(view.offset_of(5), Some(5)); // 🌍 at byte 5
///
/// // Iterate over characters
/// let chars: Vec<char> = view.iter().collect();
/// assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o', '🌍']);
/// ```
pub struct Utf8View<'a> {
    octets: &'a [u8],
    cached_len: core::cell::Cell<Option<usize>>,
}

impl<'a> Utf8View<'a> {
    /// Creates a new UTF-8 view (O(1) - no scanning).
    pub const fn new(octets: &'a [u8]) -> Self {
        Self {
            octets,
            cached_len: core::cell::Cell::new(None),
        }
    }

    /// Returns the number of UTF-8 characters (lazy evaluation, cached after first call).
    pub fn len(&self) -> usize {
        if let Some(len) = self.cached_len.get() {
            return len;
        }
        let len = count_utf8(self.octets);
        self.cached_len.set(Some(len));
        len
    }

    /// Checks if the view is empty.
    pub const fn is_empty(&self) -> bool {
        self.octets.is_empty()
    }

    /// Gets the byte offset of the Nth character (0-indexed, SIMD-accelerated).
    pub fn offset_of(&self, n: usize) -> Option<usize> {
        find_nth_utf8(self.octets, n)
    }

    /// Returns an iterator over UTF-8 characters.
    pub fn iter(&self) -> Utf8Runes<'a> {
        Utf8Runes::new(self.octets)
    }
}

/// Iterator over UTF-8 characters using batched decoding.
///
/// Each refill decodes up to `STEPS` codepoints in a single `sz_utf8_decode` FFI call (the decoder fills
/// the whole buffer regardless of script width), then yields them one at a time - far cheaper than decoding
/// character-by-character. Ill-formed bytes decode to the replacement character U+FFFD, so iteration is total
/// and never silently truncates.
///
/// Typically created through [`Utf8View::iter()`].
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// let text = "Hello🌍";
/// let view = sz::Utf8View::new(text.as_bytes());
/// let chars: Vec<char> = view.iter().collect();
/// assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o', '🌍']);
/// ```
pub struct Utf8Runes<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    octets: &'a [u8],
    octets_offset: usize,
    runes: [u32; STEPS], // Buffered codepoints decoded from the current chunk
    runes_count: usize,  // Number of buffered codepoints (0 once exhausted)
    runes_offset: usize, // Index of the next codepoint to yield from the buffer
}

impl<'a> Utf8Runes<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish, e.g.
    /// `Utf8Runes::<256>::with_steps(octets)`.
    fn new(octets: &'a [u8]) -> Self {
        Self::with_steps(octets)
    }
}

impl<'a, const STEPS: usize> Utf8Runes<'a, STEPS> {
    /// Constructs an iterator buffering up to `STEPS` codepoints per FFI call.
    pub fn with_steps(octets: &'a [u8]) -> Self {
        let mut iter = Self {
            octets,
            octets_offset: 0,
            runes: [0; STEPS],
            runes_count: 0,
            runes_offset: 0,
        };
        iter.decode_batch();
        iter
    }

    /// Decodes the next chunk of UTF-8 bytes into the runes buffer; `runes_count` becomes 0 once drained.
    fn decode_batch(&mut self) {
        if self.octets_offset >= self.octets.len() {
            self.runes_count = 0;
            return;
        }

        let octets_ptr = unsafe { self.octets.as_ptr().add(self.octets_offset) as *const c_void };
        let mut unpacked_count: usize = 0;
        let next_ptr = unsafe {
            sz_utf8_decode(
                octets_ptr,
                self.octets.len() - self.octets_offset,
                self.runes.as_mut_ptr(),
                STEPS,
                &mut unpacked_count as *mut usize,
            )
        };

        let bytes_consumed: usize = unsafe {
            let offset = (next_ptr as *const u8).offset_from(octets_ptr as *const u8);
            debug_assert!(offset >= 0, "sz_utf8_decode returned a pointer before the input");
            offset.try_into().expect("offset should be non-negative")
        };
        self.octets_offset += bytes_consumed;
        self.runes_offset = 0;

        // The decoder stops (yielding nothing) on a well-formed but truncated trailing sequence so a streaming
        // caller can resume. We own the whole slice, so there is nothing more to resume with: finalize that tail
        // as a single U+FFFD (its maximal subpart) instead of silently dropping it, matching `from_utf8_lossy`.
        if unpacked_count == 0 && self.octets_offset < self.octets.len() {
            self.runes[0] = 0xFFFD;
            self.runes_count = 1;
            self.octets_offset = self.octets.len();
        } else {
            self.runes_count = unpacked_count;
        }
    }
}

impl<'a, const STEPS: usize> Iterator for Utf8Runes<'a, STEPS> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        // If the buffer is drained, decode the next chunk.
        if self.runes_offset >= self.runes_count {
            self.decode_batch();
            if self.runes_count == 0 {
                return None;
            }
        }

        let codepoint = self.runes[self.runes_offset];
        self.runes_offset += 1;
        // Safety: `sz_utf8_decode` only emits valid Unicode scalar values (ill-formed input becomes U+FFFD),
        // so the conversion never sees a surrogate or an out-of-range value - no per-codepoint re-validation needed.
        Some(unsafe { char::from_u32_unchecked(codepoint) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Lower bound: remaining runes in current buffer; upper bound unknown without counting the whole string.
        let lower = self.runes_count.saturating_sub(self.runes_offset);
        (lower, None)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::sz::{self, *};

    #[test]
    fn utf8_runes_match_std_chars() {
        // Multilingual valid UTF-8, including a long mixed run that spans several decode batches.
        let long_mixed = "Hello, \u{43C}\u{438}\u{440}! \u{4E16}\u{754C} \u{1F30D}\u{1F680} \u{627}\u{644}".repeat(50);
        let samples = [
            "",
            "A",
            "Hello\u{1F30D}",
            "\u{3A9}\u{3BC}\u{3AD}\u{3B3}\u{3B1}",
            long_mixed.as_str(),
        ];
        for text in samples {
            let expected: Vec<char> = text.chars().collect();
            let via_view: Vec<char> = sz::Utf8View::new(text.as_bytes()).iter().collect();
            assert_eq!(
                via_view, expected,
                "rune iteration diverged from std::chars for {:?}",
                text
            );
            let via_trait: Vec<char> = text.as_bytes().sz_utf8_runes().iter().collect();
            assert_eq!(via_trait, expected);
        }
    }

    #[test]
    fn utf8_runes_with_steps_match_default() {
        // The batch width is a performance knob only - every `STEPS` must yield the same codepoints.
        let text = "Hello, \u{43C}\u{438}\u{440}! \u{4E16}\u{754C} \u{1F30D} \u{627}\u{644}".repeat(10);
        let expected: Vec<char> = text.chars().collect();
        let tiny: Vec<char> = sz::Utf8Runes::<1>::with_steps(text.as_bytes()).collect();
        let wide: Vec<char> = sz::Utf8Runes::<256>::with_steps(text.as_bytes()).collect();
        assert_eq!(tiny, expected);
        assert_eq!(wide, expected);
    }

    #[test]
    fn utf8_decode_replaces_ill_formed() {
        // The decoder is total: ill-formed bytes become U+FFFD and it never emits a non-scalar value.
        let ill_formed: [&[u8]; 4] = [b"\x80", b"\xC0\x80", b"\xED\xA0\x80", b"a\xFFb"];
        for bytes in ill_formed {
            let mut runes = [0u32; 16];
            let mut offset = 0;
            while offset < bytes.len() {
                let (consumed, count) = sz::utf8_decode(&bytes[offset..], &mut runes);
                for &rune in &runes[..count] {
                    assert!(
                        rune <= 0x10FFFF && !(0xD800..=0xDFFF).contains(&rune),
                        "non-scalar value 0x{:X}",
                        rune
                    );
                }
                if consumed == 0 {
                    break;
                }
                offset += consumed;
            }
        }
        // A lone 0xFF between two ASCII bytes yields exactly 'a', U+FFFD, 'b' - lossy, never truncated.
        let mut runes = [0u32; 8];
        let (_, count) = sz::utf8_decode(b"a\xFFb", &mut runes);
        assert_eq!(&runes[..count], &['a' as u32, 0xFFFD, 'b' as u32]);
    }

    #[test]
    fn utf8_runes_finalize_truncated_tail() {
        // A string ending mid-codepoint yields the leading runes then a single U+FFFD for the truncated tail,
        // never silently dropping it (matching `String::from_utf8_lossy`).
        let truncated = b"hi\xF0\x9F\x98"; // "hi" + the first 3 bytes of a 4-byte emoji
        let runes: Vec<char> = sz::Utf8View::new(truncated).iter().collect();
        assert_eq!(runes, vec!['h', 'i', '\u{FFFD}']);
    }
}
