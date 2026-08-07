//! Core single-string operations with SIMD acceleration.
//!
//! Provides fast string search, comparison, hashing, and manipulation
//! functions optimized with SWAR and SIMD instructions.

mod cipher;
mod compare;
mod find;
mod hash;
mod intersect;
mod memory;
mod sort;
mod types;
mod utf8_graphemes;
mod utf8_linebreaks;
mod utf8_norm;
mod utf8_runes;
mod utf8_sentences;
mod utf8_tokens;
mod utf8_uncased;
mod utf8_uncased_fold;
mod utf8_wordbreaks;

pub use cipher::*;
pub use compare::*;
pub use find::*;
pub use hash::*;
pub use intersect::*;
pub use memory::*;
pub use sort::*;
pub use types::*;
pub use utf8_graphemes::*;
pub use utf8_linebreaks::*;
pub use utf8_norm::*;
pub use utf8_runes::*;
pub use utf8_sentences::*;
pub use utf8_tokens::*;
pub use utf8_uncased::*;
pub use utf8_uncased_fold::*;
pub use utf8_wordbreaks::*;

use core::ffi::c_void;

// Import the functions from the StringZillable C library.
extern "C" {

    pub(crate) fn sz_dynamic_dispatch() -> i32;
    pub(crate) fn sz_version_major() -> i32;
    pub(crate) fn sz_version_minor() -> i32;
    pub(crate) fn sz_version_patch() -> i32;
    pub(crate) fn sz_capabilities() -> u32;
    pub(crate) fn sz_capabilities_to_string(caps: u32) -> *const c_void;

    pub(crate) fn sz_copy(target: *const c_void, source: *const c_void, length: usize);
    pub(crate) fn sz_fill(target: *const c_void, length: usize, value: u8);
    pub(crate) fn sz_move(target: *const c_void, source: *const c_void, length: usize);
    pub(crate) fn sz_fill_random(text: *mut c_void, length: usize, seed: u64);
    pub(crate) fn sz_lookup(target: *const c_void, length: usize, source: *const c_void, lut: *const u8);

    pub(crate) fn sz_find(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    pub(crate) fn sz_rfind(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    pub(crate) fn sz_find_byteset(
        haystack: *const c_void,
        haystack_length: usize,
        byteset: *const c_void,
    ) -> *const c_void;
    pub(crate) fn sz_rfind_byteset(
        haystack: *const c_void,
        haystack_length: usize,
        byteset: *const c_void,
    ) -> *const c_void;

    pub(crate) fn sz_utf8_count(text: *const c_void, length: usize) -> usize;
    pub(crate) fn sz_utf8_seek(text: *const c_void, length: usize, n: usize) -> *const c_void;
    pub(crate) fn sz_utf8_decode(
        text: *const c_void,
        length: usize,
        runes: *mut u32,
        runes_capacity: usize,
        runes_unpacked: *mut usize,
    ) -> *const c_void;
    pub(crate) fn sz_utf8_newlines(
        text: *const c_void,
        length: usize,
        match_offsets: *mut usize,
        match_lengths: *mut usize,
        matches_capacity: usize,
        bytes_consumed: *mut usize,
    ) -> usize;
    pub(crate) fn sz_utf8_whitespaces(
        text: *const c_void,
        length: usize,
        match_offsets: *mut usize,
        match_lengths: *mut usize,
        matches_capacity: usize,
        bytes_consumed: *mut usize,
    ) -> usize;
    pub(crate) fn sz_utf8_delimiters(
        text: *const c_void,
        length: usize,
        match_offsets: *mut usize,
        match_lengths: *mut usize,
        matches_capacity: usize,
        bytes_consumed: *mut usize,
    ) -> usize;
    pub(crate) fn sz_utf8_uncased_fold(source: *const c_void, source_length: usize, destination: *mut c_void) -> usize;
    pub(crate) fn sz_utf8_norm(
        source: *const c_void,
        source_length: usize,
        form: i32,
        destination: *mut c_void,
    ) -> usize;
    pub(crate) fn sz_utf8_find_denormalized(source: *const c_void, source_length: usize, form: i32) -> *const c_void;
    pub(crate) fn sz_utf8_uncased_search(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
        needle_metadata: *mut Utf8UncasedNeedleMetadata,
        matched_length: *mut usize,
    ) -> *const c_void;
    pub(crate) fn sz_utf8_uncased_order(a: *const c_void, a_length: usize, b: *const c_void, b_length: usize) -> i32;

    pub(crate) fn sz_utf8_wordbreaks(
        text: *const c_void,
        length: usize,
        word_starts: *mut usize,
        word_lengths: *mut usize,
        words_capacity: usize,
        bytes_consumed: *mut usize,
    ) -> usize;

    pub(crate) fn sz_utf8_graphemes(
        text: *const c_void,
        length: usize,
        starts: *mut usize,
        lengths: *mut usize,
        cap: usize,
        consumed: *mut usize,
    ) -> usize;
    pub(crate) fn sz_utf8_sentences(
        text: *const c_void,
        length: usize,
        starts: *mut usize,
        lengths: *mut usize,
        cap: usize,
        consumed: *mut usize,
    ) -> usize;
    pub(crate) fn sz_utf8_linebreaks(
        text: *const c_void,
        length: usize,
        starts: *mut usize,
        lengths: *mut usize,
        cap: usize,
        consumed: *mut usize,
    ) -> usize;

    pub(crate) fn sz_equal(a: *const c_void, b: *const c_void, length: usize) -> i32;
    pub(crate) fn sz_order(a: *const c_void, a_length: usize, b: *const c_void, b_length: usize) -> i32;

    pub(crate) fn sz_bytesum(text: *const c_void, length: usize) -> u64;
    pub(crate) fn sz_hash(text: *const c_void, length: usize, seed: u64) -> u64;
    pub(crate) fn sz_hash_multiseed(
        text: *const c_void,
        length: usize,
        seeds: *const u64,
        seeds_count: usize,
        hashes: *mut u64,
    );
    pub(crate) fn sz_hash_state_init(state: *const c_void, seed: u64);
    pub(crate) fn sz_hash_state_update(state: *const c_void, text: *const c_void, length: usize);
    pub(crate) fn sz_hash_state_digest(state: *const c_void) -> u64;
    pub(crate) fn sz_sha256_state_init(state: *const c_void);
    pub(crate) fn sz_sha256_state_update(state: *const c_void, data: *const c_void, length: usize);
    pub(crate) fn sz_sha256_state_digest(state: *const c_void, digest: *mut u8);
    pub(crate) fn sz_sha256_multistate_update(states: *mut c_void, texts: *const _SzSequence);
    pub(crate) fn sz_sha256_multistate_digest(states: *const c_void, states_count: usize, digests: *mut u8);

    pub(crate) fn sz_aes256_key_init(key: *mut c_void, secret: *const u8);
    pub(crate) fn sz_aes256_ctr_xor(
        key: *const c_void,
        nonce: *const u8,
        byte_offset: u64,
        text: *const c_void,
        length: usize,
        output: *mut c_void,
    );
    pub(crate) fn sz_aes256_gcm_key_init(key: *mut c_void, secret: *const u8);
    pub(crate) fn sz_aes256_gcm_encrypt(
        key: *const c_void,
        nonce: *const u8,
        associated: *const c_void,
        associated_length: usize,
        text: *const c_void,
        length: usize,
        output: *mut c_void,
        tag: *mut u8,
    );
    /// Returns `sz_status_t` as a raw code, because a value outside `Status` would be undefined
    /// behavior to receive as that enum, and a decryption must never mistake one for success.
    pub(crate) fn sz_aes256_gcm_decrypt(
        key: *const c_void,
        nonce: *const u8,
        associated: *const c_void,
        associated_length: usize,
        text: *const c_void,
        length: usize,
        output: *mut c_void,
        tag: *const u8,
    ) -> i32;
    pub(crate) fn sz_aes256_gcm_encryptor_init(encryptor: *mut c_void, key: *const c_void, nonce: *const u8);
    pub(crate) fn sz_aes256_gcm_encryptor_associate(encryptor: *mut c_void, text: *const c_void, length: usize);
    pub(crate) fn sz_aes256_gcm_encryptor_update(
        encryptor: *mut c_void,
        text: *const c_void,
        length: usize,
        output: *mut c_void,
    );
    pub(crate) fn sz_aes256_gcm_encryptor_digest(encryptor: *const c_void, tag: *mut u8);
    pub(crate) fn sz_aes256_gcm_decryptor_init(decryptor: *mut c_void, key: *const c_void, nonce: *const u8);
    pub(crate) fn sz_aes256_gcm_decryptor_associate(decryptor: *mut c_void, text: *const c_void, length: usize);
    pub(crate) fn sz_aes256_gcm_decryptor_update_unverified(
        decryptor: *mut c_void,
        text: *const c_void,
        length: usize,
        output: *mut c_void,
    );
    /// Returns `sz_status_t` as a raw code, for the same reason as `sz_aes256_gcm_decrypt`.
    pub(crate) fn sz_aes256_gcm_decryptor_verify(decryptor: *const c_void, tag: *const u8) -> i32;

    pub(crate) fn sz_sequence_argsort(
        //
        sequence: *const _SzSequence,
        alloc: *const c_void,
        order: *mut SortedIdx,
        top_count: usize,
        reverse: i32,
    ) -> Status;

    pub(crate) fn sz_sequence_argsort_uncased(
        //
        sequence: *const _SzSequence,
        alloc: *const c_void,
        order: *mut SortedIdx,
        top_count: usize,
        reverse: i32,
    ) -> Status;

    pub(crate) fn sz_sequence_intersect(
        first_sequence: *const _SzSequence,
        second_sequence: *const _SzSequence,
        alloc: *const c_void,
        seed: u64,
        intersection_size: *mut usize,
        first_positions: *mut SortedIdx,
        second_positions: *mut SortedIdx,
    ) -> Status;

}

/// Trait for unary string operations that only operate on `self` without needle parameters.
/// These operations include hash computation and byte sum calculation.
///
/// # Examples
///
/// Basic usage on a byte slice:
///
/// ```
/// use stringzilla::sz::StringZillableUnary;
///
/// let text = b"Hello";
/// assert_eq!(text.sz_bytesum(), 500);
/// ```
pub trait StringZillableUnary {
    /// Computes the bytesum value of unsigned bytes in a given string.
    /// This function is useful for verifying data integrity and detecting changes in
    /// binary data, such as files or network packets.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = b"Hello";
    /// assert_eq!(text.sz_bytesum(), 500);
    /// ```
    fn sz_bytesum(&self) -> u64;

    /// Computes a 64-bit AES-based hash value for a given string.
    /// This function is designed to provide a high-quality hash value for use in
    /// hash tables, data structures, and cryptographic applications.
    /// Unlike the bytesum function, the hash function is order-sensitive.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let s1 = b"Hello";
    /// let s2 = b"World";
    /// assert_ne!(s1.sz_hash(), s2.sz_hash());
    /// ```
    fn sz_hash(&self) -> u64;

    /// Returns a lazy UTF-8 character view with SIMD-accelerated operations.
    ///
    /// The view provides:
    /// - `.len()` for character count (lazy: computed on first call, cached)
    /// - `.offset_of(n)` for random access to Nth character offset
    /// - `.iter()` for efficient batched iteration over characters
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = "Hello🌍";
    /// let view = text.sz_utf8_runes();
    ///
    /// // Lazy character count
    /// assert_eq!(view.len(), 6);
    ///
    /// // Random access (byte offset of Nth character)
    /// assert_eq!(view.offset_of(5), Some(5)); // 🌍 at byte 5
    ///
    /// // Efficient batched iteration
    /// let chars: Vec<char> = view.iter().collect();
    /// assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o', '🌍']);
    /// ```
    fn sz_utf8_runes(&self) -> Utf8View<'_>;

    /// Returns an iterator over lines split by UTF-8 newline characters.
    ///
    /// The iterator yields slices between newlines. Handles all Unicode newline characters
    /// including CRLF as a single delimiter.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = "Hello\nWorld\r\nRust";
    /// let lines: Vec<&str> = text.sz_utf8_split_newlines()
    ///     .map(|line| std::str::from_utf8(line).unwrap())
    ///     .collect();
    /// assert_eq!(lines, vec!["Hello", "World", "Rust"]);
    /// ```
    fn sz_utf8_split_newlines(&self) -> Utf8SplitNewlines<'_>;

    /// Returns an iterator over the newline runs themselves (the separators).
    fn sz_utf8_newlines(&self) -> Utf8Newlines<'_>;

    /// Returns an iterator over segments split by UTF-8 whitespace characters.
    ///
    /// Handles all 25 Unicode "White_Space" characters; N delimiters yield N+1 segments. By default
    /// **empty segments are kept** (matching `str::split`), so runs of
    /// whitespace surface empty slices. Chain `.skip_empty()` on the returned iterator to recover the
    /// `str::split_whitespace` token behavior that drops empties.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// // KEEP (default): the double space between "Hello" and "World" yields an empty segment.
    /// let text = "Hello  World\tRust";
    /// let segments: Vec<&str> = text.sz_utf8_split_whitespaces()
    ///     .map(|segment| std::str::from_utf8(segment).unwrap())
    ///     .collect();
    /// assert_eq!(segments, vec!["Hello", "", "World", "Rust"]);
    ///
    /// // skip_empty: drops the empties to yield only the non-empty tokens.
    /// let tokens: Vec<&str> = text.sz_utf8_split_whitespaces()
    ///     .skip_empty()
    ///     .map(|token| std::str::from_utf8(token).unwrap())
    ///     .collect();
    /// assert_eq!(tokens, vec!["Hello", "World", "Rust"]);
    /// ```
    fn sz_utf8_split_whitespaces(&self) -> Utf8SplitWhitespaces<'_>;

    /// Returns an iterator over the whitespace runs themselves (the separators).
    fn sz_utf8_whitespaces(&self) -> Utf8Whitespaces<'_>;

    /// Returns an iterator splitting on any Unicode delimiter (punctuation/symbol/separator/whitespace).
    fn sz_utf8_split_delimiters(&self) -> Utf8SplitDelimiters<'_>;

    /// Returns an iterator over the delimiter runs themselves (the separators).
    fn sz_utf8_delimiters(&self) -> Utf8Delimiters<'_>;

    /// Returns an iterator over UAX-29 words, in order. Words tile the input contiguously.
    fn sz_utf8_wordbreaks(&self) -> Utf8Wordbreaks<'_>;

    /// Returns an iterator over UAX-29 grapheme clusters, in order. Clusters tile the input contiguously.
    fn sz_utf8_graphemes(&self) -> Utf8Graphemes<'_>;

    /// Returns an iterator over UAX-29 sentences, in order. Sentences tile the input contiguously.
    fn sz_utf8_sentences(&self) -> Utf8Sentences<'_>;

    /// Returns an iterator over UAX-14 line-break opportunities (Unicode TR14), in order. Linewrap segments tile the
    /// input contiguously, including soft break opportunities. For hard line splits only, use
    /// [`Self::sz_utf8_split_newlines`].
    fn sz_utf8_linebreaks(&self) -> Utf8Linebreaks<'_>;
}

impl<Source> StringZillableUnary for Source
where
    Source: AsRef<[u8]> + ?Sized,
{
    fn sz_bytesum(&self) -> u64 {
        bytesum(self)
    }

    fn sz_hash(&self) -> u64 {
        hash(self)
    }

    fn sz_utf8_runes(&self) -> Utf8View<'_> {
        Utf8View::new(self.as_ref())
    }

    fn sz_utf8_split_newlines(&self) -> Utf8SplitNewlines<'_> {
        Utf8SplitNewlines::new(self.as_ref())
    }

    fn sz_utf8_newlines(&self) -> Utf8Newlines<'_> {
        Utf8Newlines::new(self.as_ref())
    }

    fn sz_utf8_split_whitespaces(&self) -> Utf8SplitWhitespaces<'_> {
        Utf8SplitWhitespaces::new(self.as_ref())
    }

    fn sz_utf8_whitespaces(&self) -> Utf8Whitespaces<'_> {
        Utf8Whitespaces::new(self.as_ref())
    }

    fn sz_utf8_split_delimiters(&self) -> Utf8SplitDelimiters<'_> {
        Utf8SplitDelimiters::new(self.as_ref())
    }

    fn sz_utf8_delimiters(&self) -> Utf8Delimiters<'_> {
        Utf8Delimiters::new(self.as_ref())
    }

    fn sz_utf8_wordbreaks(&self) -> Utf8Wordbreaks<'_> {
        Utf8Wordbreaks::new(self.as_ref())
    }

    fn sz_utf8_graphemes(&self) -> Utf8Graphemes<'_> {
        Utf8Graphemes::new(self.as_ref())
    }

    fn sz_utf8_sentences(&self) -> Utf8Sentences<'_> {
        Utf8Sentences::new(self.as_ref())
    }

    fn sz_utf8_linebreaks(&self) -> Utf8Linebreaks<'_> {
        Utf8Linebreaks::new(self.as_ref())
    }
}

/// Prose fixtures shared by the tests of more than one UTF-8 domain module.
///
/// Realistic multi-script prose (ASCII-source `\u{}` escapes; rendered prose in comments).
/// Per-family segment counts are oracle-locked (ICU root / uniseg). Fixtures read by a single
/// module live beside the test that reads them instead.
#[cfg(test)]
pub(crate) mod fixtures {
    // Hotel review (German + Japanese): NFD cafe, NBSP-glued units, a sentence-ending abbreviation, a CJK run.
    pub(crate) const PROSE_HOTEL_REVIEW: &str = concat!(
        "Last spring we strolled down M\u{fc}nchner Stra\u{df}e; the cafe\u{301} cortado cost 3,50\u{a0}",
        "\u{20ac} and was unreal. Dr. Vogel, our guide, swore it's the city's finest. Worth the detour?! ",
        "Absolutely \u{2014} and \u{6771}\u{4eac}\u{30bf}\u{30ef}\u{30fc} the next week, all 333\u{a0}m o",
        "f it, was breathtaking at dusk\u{2026}"
    );
    // Concert post (Korean + Japanese): conjoining L+V+T jamo, a Katakana run, an ideographic stop, a 'p.m.' no-break.
    pub(crate) const PROSE_CONCERT_POST: &str = concat!(
        "\u{c624}\u{b298} \u{cf58}\u{c11c}\u{d2b8}, \u{c9c4}\u{c9dc} \u{bbf8}\u{cce4}\u{b2e4}!! \u{1112}",
        "\u{1161}\u{11ab}\u{ad6d} \u{d32c}\u{b4e4}\u{c774} \u{b2e4} \u{baa8}\u{c600}\u{ace0}, the staff b",
        "owed and said \u{c548}\u{b155}\u{d788} \u{ac00}\u{c138}\u{c694}. Setlist was pure \u{30cf}",
        "\u{30fc}\u{30c9}\u{30b3}\u{30a2}; \u{4eca}\u{65e5}\u{306f}\u{6700}\u{9ad8}\u{3060}\u{3063}",
        "\u{305f}\u{3002} We screamed \u{c0ac}\u{b791}\u{d574} till 11 p.m. sharp."
    );
    // News lede: 'U.S.A.' before a lowercase word (no break), curly quotes, thousands, currency, a date range.
    pub(crate) const PROSE_NEWS_LEDE: &str = concat!(
        "The U.S.A. wasn't ready, analysts said. \u{201c}We lost 1,000 jobs,\u{201d} the mayor warned. ",
        "\u{201c}Recovery starts now.\u{201d} Filings spiked 2024/06\u{2013}2024/09, topping $1,000 per c",
        "laim. Will it hold?! No one knows for sure."
    );
    // Language lesson: a Greek final sigma, Cyrillic case pairs, a Croatian titlecase digraph, and a fold-only match.
    #[allow(dead_code)] // used by the Python uncased prose test, not Rust
    pub(crate) const PROSE_LANGUAGE_LESSON: &str = concat!(
        "Greek lesson: \u{39f}\u{394}\u{39f}\u{3a3} becomes \u{3bf}\u{3b4}\u{3cc}\u{3c2} when lowercased,",
        " ending in a final \u{3c2}. Russian's easy too \u{2014} \u{41c}\u{41e}\u{421}\u{41a}\u{412}",
        "\u{410} \u{2194} \u{43c}\u{43e}\u{441}\u{43a}\u{432}\u{430}, no drama. Croatian has the digraph ",
        "\u{1c4}: titlecase \u{1c5}, lowercase \u{1c6}. Quiz \u{2014} does \u{201c}stra\u{df}e\u{201d} ma",
        "tch STRASSE? Yes, once you fold."
    );
    // RTL scripts: Hebrew gershayim, Arabic, a number-sign Prepend, an NFC niqqud reorder, a Malayalam dot-reph.
    pub(crate) const PROSE_RTL_SCRIPTS: &str = concat!(
        "Hebrew acronyms take gershayim: \u{5e6}\u{5d4}\u{5f4}\u{5dc} and \u{5d0}\u{5e8}\u{5d4}\u{5f4}",
        "\u{5d1} aren't typos. Arabic flows right-to-left too \u{2014} \u{645}\u{631}\u{62d}\u{628}",
        "\u{627} \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645} \u{2014} and finance text can carry th",
        "e number sign \u{600}\u{664}. Niqqud stacks marks: \u{5e9}\u{5c1}\u{5b8}\u{5dc}\u{5d5}\u{5b9}",
        "\u{5dd} must reorder under NFC. Malayalam even has a true prepend, the dot-reph \u{d4e}\u{d15}."
    );
}
