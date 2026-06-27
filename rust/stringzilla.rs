/// A simple semantic version structure.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SemVer {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
}

#[repr(C)]
#[derive(Debug, PartialEq)]
pub enum Status {
    /// For algorithms that return a status, this status indicates that the operation was successful.
    /// Corresponds to `sz_success_k = 0` in C.
    Success = 0,
    /// For algorithms that require memory allocation, this status indicates that the allocation failed.
    /// Corresponds to `sz_bad_alloc_k = -10` in C.
    BadAlloc = -10,
    /// For algorithms that require UTF8 input, this status indicates that the input is invalid.
    /// Corresponds to `sz_invalid_utf8_k = -12` in C.
    InvalidUtf8 = -12,
    /// For algorithms that take collections of unique elements, this status indicates presence of duplicates.
    /// Corresponds to `sz_contains_duplicates_k = -13` in C.
    ContainsDuplicates = -13,
    /// For algorithms dealing with large inputs, this error reports the need to upcast the logic to larger types.
    /// Corresponds to `sz_overflow_risk_k = -14` in C.
    OverflowRisk = -14,
    /// For algorithms with multi-stage pipelines indicates input/output size mismatch.
    /// Corresponds to `sz_unexpected_dimensions_k = -15` in C.
    UnexpectedDimensions = -15,
    /// GPU support is missing in the library.
    /// Corresponds to `sz_missing_gpu_k = -16` in C.
    MissingGpu = -16,
    /// Backend-device mismatch (e.g., GPU kernel with CPU/default executor).
    /// Corresponds to `sz_device_code_mismatch_k = -17` in C.
    DeviceCodeMismatch = -17,
    /// Device memory mismatch (e.g., pageable host memory where Unified/Device memory is required).
    /// Corresponds to `sz_device_memory_mismatch_k = -18` in C.
    DeviceMemoryMismatch = -18,
    /// A sink-hole status for unknown errors.
    /// Corresponds to `sz_status_unknown_k = -1` in C.
    StatusUnknown = -1,
}

/// Unicode normalization forms for UTF-8 normalization operations.
///
/// Corresponds to `sz_normal_form_t` in the C API.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Utf8NormalForm {
    /// Canonical Decomposition. Decomposes precomposed characters into base + combining marks.
    Nfd = 0,
    /// Canonical Decomposition followed by Canonical Composition. The most common Unicode form.
    Nfc = 1,
    /// Compatibility Decomposition. Decomposes ligatures and compatibility characters.
    Nfkd = 2,
    /// Compatibility Decomposition followed by Canonical Composition.
    Nfkc = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Byteset {
    bits: [u64; 4],
}

/// Represents a byte span with offset and length.
///
/// Used for matches of UTF-8 characters, substrings, or any byte-level operations.
/// Stores the byte offset from the start of the text and the length in bytes.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::IndexSpan;
///
/// let text = "Hello\nWorld";
/// let span = IndexSpan::new(5, 1);
/// assert_eq!(span.offset, 5);
/// assert_eq!(span.length, 1);
/// let matched = span.extract(text.as_bytes());
/// assert_eq!(matched, b"\n");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexSpan {
    /// Byte offset from the start of the text
    pub offset: usize,
    /// Length in bytes of the matched span
    pub length: usize,
}

impl IndexSpan {
    /// Creates a new IndexSpan with the given offset and length.
    #[inline]
    pub fn new(offset: usize, length: usize) -> Self {
        Self { offset, length }
    }

    /// Returns the range of bytes covered by this span.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::stringzilla::IndexSpan;
    ///
    /// let span = IndexSpan::new(5, 3);
    /// assert_eq!(span.range(), 5..8);
    /// ```
    #[inline]
    pub fn range(&self) -> core::ops::Range<usize> {
        self.offset..self.offset + self.length
    }

    /// Extracts the matched bytes from the source text.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::stringzilla::IndexSpan;
    ///
    /// let text = b"Hello World";
    /// let span = IndexSpan::new(6, 5);
    /// assert_eq!(span.extract(text), b"World");
    /// ```
    #[inline]
    pub fn extract<'a>(&self, text: &'a [u8]) -> &'a [u8] {
        &text[self.range()]
    }

    /// Returns the end offset (offset + length).
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::stringzilla::IndexSpan;
    ///
    /// let span = IndexSpan::new(5, 3);
    /// assert_eq!(span.end(), 8);
    /// ```
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.length
    }
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

impl Default for Utf8UncasedNeedleMetadata {
    fn default() -> Self {
        Self {
            offset_in_unfolded: 0,
            length_in_unfolded: 0,
            folded_slice: [0; 16],
            folded_slice_length: 0,
            probe_second: 0,
            probe_third: 0,
            kernel_id: 0, // sz_utf8_uncased_rune_unknown_k = 0, triggers analysis
        }
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
    pub fn new(needle: &'a [u8]) -> Self {
        Self {
            needle,
            metadata: UnsafeCell::new(Utf8UncasedNeedleMetadata::default()),
        }
    }

    /// Returns the needle bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.needle
    }

    /// Returns the length of the needle in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.needle.len()
    }

    /// Returns true if the needle is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
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

/// Incremental hasher state for StringZilla's 64-bit hash.
///
/// Use `Hasher::new(seed)` to construct, then call `update(&mut self, data)`
/// zero or more times, and finally call `digest(&self)` to read the current
/// hash value without consuming the state.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[repr(align(64))] // For optimal performance we align to 64 bytes.
pub struct Hasher {
    aes: [u64; 8],
    sum: [u64; 8],
    ins: [u64; 8], // Ignored in comparisons
    key: [u64; 2],
    ins_length: usize, // Ignored in comparisons
}

/// Incremental SHA256 hasher state for cryptographic hashing.
///
/// # Examples
///
/// One-shot hashing:
///
/// ```
/// use stringzilla::stringzilla::Sha256;
/// let digest = Sha256::hash(b"Hello, world!");
/// assert_eq!(digest.len(), 32); // 256 bits = 32 bytes
/// ```
///
/// Incremental hashing:
///
/// ```
/// use stringzilla::stringzilla::Sha256;
/// let mut hasher = Sha256::new();
/// hasher.update(b"Hello, ");
/// hasher.update(b"world!");
/// let digest = hasher.digest();
/// assert_eq!(digest, Sha256::hash(b"Hello, world!"));
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[repr(align(64))] // For optimal performance we align to 64 bytes.
pub struct Sha256 {
    hash: [u32; 8],      // Current hash state (h0-h7)
    block: [u8; 64],     // 64-byte message block buffer
    block_length: usize, // Current bytes in block (0-63)
    total_length: u64,   // Total message length in bytes
}

pub type SortedIdx = usize;

/// A trait for types that support indexed lookup.
pub trait SequenceData {
    type Item;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn index(&self, idx: usize) -> &Self::Item;
}

// Implement SequenceData for slices.
impl<T> SequenceData for [T] {
    type Item = T;
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn index(&self, idx: usize) -> &T {
        &self[idx]
    }
}

#[repr(C)]
pub struct _SzSequence {
    pub handle: *const c_void,
    pub count: usize,
    pub get_start: Option<unsafe extern "C" fn(handle: *const c_void, idx: usize) -> *const c_void>,
    pub get_length: Option<unsafe extern "C" fn(handle: *const c_void, idx: usize) -> usize>,
}

impl Byteset {
    /// Initializes a bit-set to an empty collection (all characters banned).
    #[inline]
    pub fn new() -> Self {
        Self { bits: [0; 4] }
    }

    /// Initializes a bit-set to contain all ASCII characters.
    #[inline]
    pub fn new_ascii() -> Self {
        Self {
            bits: [u64::MAX, u64::MAX, 0, 0],
        }
    }

    /// Adds a byte to the set.
    #[inline]
    pub fn add_u8(&mut self, c: u8) {
        let idx = (c >> 6) as usize; // Divide by 64.
        let bit = c & 63; // Remainder modulo 64.
        self.bits[idx] |= 1 << bit;
    }

    /// Adds a character to the set.
    ///
    /// This function assumes the character is in the ASCII range.
    #[inline]
    pub fn add(&mut self, c: char) {
        self.add_u8(c as u8);
    }

    /// Inverts the bit-set so that all set bits become unset and vice versa.
    #[inline]
    pub fn invert(&mut self) {
        for b in self.bits.iter_mut() {
            *b = !*b;
        }
    }

    /// Returns a new Byteset with all bits inverted, leaving self unchanged.
    #[inline]
    pub fn inverted(&self) -> Self {
        Self {
            bits: [!self.bits[0], !self.bits[1], !self.bits[2], !self.bits[3]],
        }
    }

    /// Constructs a Byteset from a slice of bytes.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut set = Self::new();
        for &b in bytes {
            set.add_u8(b);
        }
        set
    }
}

impl Default for Byteset {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: AsRef<[u8]>> From<T> for Byteset {
    #[inline]
    fn from(bytes: T) -> Self {
        Self::from_bytes(bytes.as_ref())
    }
}

use core::cell::UnsafeCell;
use core::cmp::Ordering;
use core::ffi::{c_char, c_void, CStr};
use core::fmt::{self, Write};

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

    pub(crate) fn sz_utf8_words(
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

    pub(crate) fn sz_sequence_argsort(
        //
        sequence: *const _SzSequence,
        alloc: *const c_void,
        order: *mut SortedIdx,
        top_count: usize,
        reverse: i32,
    ) -> Status;

    pub(crate) fn sz_sequence_argsort_utf8_uncased(
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

impl SemVer {
    pub const fn new(major: i32, minor: i32, patch: i32) -> Self {
        Self { major, minor, patch }
    }
}

impl Hasher {
    /// Creates a new hasher initialized with `seed`.
    pub fn new(seed: u64) -> Self {
        let mut state = Hasher {
            aes: [0; 8],
            sum: [0; 8],
            ins: [0; 8],
            key: [0; 2],
            ins_length: 0,
        };
        unsafe {
            sz_hash_state_init(&mut state as *mut _ as *mut c_void, seed);
        }
        state
    }

    /// Updates the hasher with more data.
    pub fn update(&mut self, data: &[u8]) -> &mut Self {
        unsafe {
            sz_hash_state_update(
                self as *mut _ as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len(),
            );
        }
        self
    }

    /// Returns the current hash value without consuming the state.
    pub fn digest(&self) -> u64 {
        unsafe { sz_hash_state_digest(self as *const _ as *const c_void) }
    }
}

impl PartialEq for Hasher {
    fn eq(&self, other: &Self) -> bool {
        self.aes == other.aes && self.sum == other.sum && self.key == other.key
    }
}

impl Default for Hasher {
    #[inline]
    fn default() -> Self {
        Hasher::new(0)
    }
}

impl Sha256 {
    /// Creates a new SHA256 hasher with the initial state.
    pub fn new() -> Self {
        let mut state = Sha256 {
            hash: [0; 8],
            block: [0; 64],
            block_length: 0,
            total_length: 0,
        };
        unsafe {
            sz_sha256_state_init(&mut state as *mut _ as *mut c_void);
        }
        state
    }

    /// Updates the hasher with more data.
    pub fn update(&mut self, data: &[u8]) -> &mut Self {
        unsafe {
            sz_sha256_state_update(
                self as *mut _ as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len(),
            );
        }
        self
    }

    /// Returns the current SHA256 hash digest as a 32-byte array.
    pub fn digest(&self) -> [u8; 32] {
        let mut digest = [0u8; 32];
        unsafe {
            sz_sha256_state_digest(self as *const _ as *const c_void, digest.as_mut_ptr());
        }
        digest
    }

    /// Convenience method to hash data in one call.
    pub fn hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.digest()
    }
}

impl Default for Sha256 {
    #[inline]
    fn default() -> Self {
        Sha256::new()
    }
}

/// Computes HMAC-SHA256 (Hash-based Message Authentication Code) for the given key and message.
///
/// # Arguments
///
/// * `key` - The secret key (can be any length, will be hashed if > 64 bytes)
/// * `message` - The message to authenticate
///
/// # Returns
///
/// A 32-byte HMAC-SHA256 digest
///
/// # Example
///
/// ```
/// use stringzilla::stringzilla::hmac_sha256;
/// let key = b"secret_key";
/// let message = b"important message";
/// let mac = hmac_sha256(key, message);
/// assert_eq!(mac.len(), 32);
/// ```
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    // Prepare key: hash if > 64 bytes, zero-pad to 64 bytes
    let mut key_pad = [0u8; 64];
    if key.len() > 64 {
        let key_hash = Sha256::hash(key);
        key_pad[..32].copy_from_slice(&key_hash);
    } else {
        key_pad[..key.len()].copy_from_slice(key);
    }

    // Compute inner hash: SHA256((key ^ 0x36) || message)
    let mut inner_hasher = Sha256::new();
    let mut inner_pad = [0u8; 64];
    for i in 0..64 {
        inner_pad[i] = key_pad[i] ^ 0x36;
    }
    inner_hasher.update(&inner_pad);
    inner_hasher.update(message);
    let inner_hash = inner_hasher.digest();

    // Compute outer hash: SHA256((key ^ 0x5c) || inner_hash)
    let mut outer_hasher = Sha256::new();
    let mut outer_pad = [0u8; 64];
    for i in 0..64 {
        outer_pad[i] = key_pad[i] ^ 0x5c;
    }
    outer_hasher.update(&outer_pad);
    outer_hasher.update(&inner_hash);
    outer_hasher.digest()
}

/// Standard Hasher trait to interoperate with `std::collections`.
impl core::hash::Hasher for Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.digest()
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let _ = self.update(bytes);
    }

    // Feed integers as little-endian bytes for cross-platform stability
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.write(&[i]);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.write(&i.to_le_bytes());
    }
    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.write(&i.to_le_bytes());
    }
}

/// BuildHasher for constructing `Hasher` instances, enabling use with HashMap/HashSet.
///
/// By default uses seed 0 for deterministic hashing across runs and platforms.
/// If you need DOS-resistant randomized seeding, consider wrapping this in your
/// application with a per-process random seed.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildSzHasher {
    pub seed: u64,
}

#[cfg(feature = "std")]
impl BuildSzHasher {
    #[inline]
    pub const fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

#[cfg(feature = "std")]
impl std::hash::BuildHasher for BuildSzHasher {
    type Hasher = Hasher;
    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        Hasher::new(self.seed)
    }
}

/// Checks if the library was compiled with dynamic dispatch enabled.
pub fn dynamic_dispatch() -> bool {
    unsafe { sz_dynamic_dispatch() != 0 }
}

/// Returns the semantic version information.
pub fn version() -> SemVer {
    SemVer {
        major: unsafe { sz_version_major() },
        minor: unsafe { sz_version_minor() },
        patch: unsafe { sz_version_patch() },
    }
}

/// A fixed-size, compile-time known C-string buffer type.
/// It keeps track of the number of written bytes (excluding the null terminator).
pub struct FixedCString<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> FixedCString<N> {
    /// Create a new, empty buffer.
    /// The buffer always has a terminating NUL (0) byte at position `len`.
    pub const fn new() -> Self {
        Self { buf: [0u8; N], len: 0 }
    }

    /// Returns the raw pointer to the C string.
    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    /// Returns a reference as a CStr.
    /// # Safety
    /// The buffer must be correctly NUL terminated.
    pub fn as_c_str(&self) -> &CStr {
        // We know buf[..=len] is NUL-terminated because write_str() always sets it.
        unsafe { CStr::from_bytes_with_nul_unchecked(&self.buf[..=self.len]) }
    }

    /// Returns the current content as a &str.
    /// Returns an empty string if the content isn’t valid UTF-8.
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

impl<const N: usize> Default for FixedCString<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Write for FixedCString<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        // Ensure we have room for the new bytes and a NUL terminator.
        if self.len + bytes.len() >= N {
            return Err(fmt::Error);
        }
        self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        // Always set a null terminator.
        self.buf[self.len] = 0;
        Ok(())
    }
}

pub type SmallCString = FixedCString<256>;

/// Copies the capabilities C-string into a fixed buffer and returns it.
/// The returned SmallCString is guaranteed to be null-terminated.
pub(crate) fn capabilities_from_enum(caps: u32) -> SmallCString {
    let caps_ptr = unsafe { sz_capabilities_to_string(caps) };
    // Assume that the external function returns a valid null-terminated C string.
    let cstr = unsafe { CStr::from_ptr(caps_ptr as *const c_char) };
    let bytes = cstr.to_bytes();

    let mut buf = SmallCString::new();
    // Use core::fmt::Write to copy the bytes.
    // If the string is too long, it will fail. You might want to truncate in a real-world use.
    // Here, we assume it fits.
    let s = core::str::from_utf8(bytes).unwrap_or("");
    let _ = buf.write_str(s);
    buf
}

/// Copies the capabilities C-string into a fixed buffer and returns it.
/// The returned SmallCString is guaranteed to be null-terminated.
pub fn capabilities() -> SmallCString {
    let caps = unsafe { sz_capabilities() };
    capabilities_from_enum(caps)
}

/// Computes the checksum value of unsigned bytes in a given byte slice `text`.
/// This function is useful for verifying data integrity and detecting changes in
/// binary data, such as files or network packets.
///
/// # Arguments
///
/// * `text`: The byte slice to compute the checksum for.
///
/// # Returns
///
/// A `u64` representing the checksum value of the input byte slice.
#[inline(always)]
pub fn bytesum<T>(text: T) -> u64
where
    T: AsRef<[u8]>,
{
    let text_ref = text.as_ref();
    let text_pointer = text_ref.as_ptr() as _;
    let text_length = text_ref.len();
    unsafe { sz_bytesum(text_pointer, text_length) }
}

/// Moves the contents of `source` into `target`, overwriting the existing contents of `target`.
/// This function is useful for scenarios where you need to replace the contents of a byte slice
/// with the contents of another byte slice.
#[inline(always)]
pub fn move_<T, S>(target: &mut T, source: &S)
where
    T: AsMut<[u8]> + ?Sized,
    S: AsRef<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    let source_slice = source.as_ref();
    assert!(target_slice.len() >= source_slice.len());
    unsafe {
        sz_move(
            target_slice.as_mut_ptr() as *const c_void,
            source_slice.as_ptr() as *const c_void,
            source_slice.len(),
        );
    }
}

/// Fills the contents of `target` with the specified `value`. This function is useful for
/// scenarios where you need to set all bytes in a byte slice to a specific value, such as
/// zeroing out a buffer or initializing a buffer with a specific byte pattern.
#[inline(always)]
pub fn fill<T>(target: &mut T, value: u8)
where
    T: AsMut<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    unsafe {
        sz_fill(target_slice.as_ptr() as *const c_void, target_slice.len(), value);
    }
}

/// Copies the contents of `source` into `target`, overwriting the existing contents of `target`.
/// This function is useful for scenarios where you need to replace the contents of a byte slice
/// with the contents of another byte slice.
#[inline(always)]
pub fn copy<T, S>(target: &mut T, source: &S)
where
    T: AsMut<[u8]> + ?Sized,
    S: AsRef<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    let source_slice = source.as_ref();
    assert!(target_slice.len() >= source_slice.len());
    unsafe {
        sz_copy(
            target_slice.as_mut_ptr() as *mut c_void,
            source_slice.as_ptr() as *const c_void,
            source_slice.len(),
        );
    }
}

/// Performs a lookup transformation (LUT), mapping contents of a buffer into the same or other
/// memory region, taking a byte substitution value from the provided table.
///
/// # Arguments
///
/// * `target`: A mutable buffer to populate.
/// * `source`: An immutable buffer to map from.
/// * `table`: Lookup table of 256 substitution values.
///
/// # Examples
///
/// To convert uppercase ASCII characters to lowercase:
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let mut to_lower: [u8; 256] = core::array::from_fn(|i| i as u8);
/// for (upper, lower) in ('A'..='Z').zip('a'..='z') {
///     to_lower[upper as usize] = lower as u8;
/// }
/// let source = "HELLO WORLD!";
/// let mut target = vec![0u8; source.len()];
/// sz::lookup(&mut target, &source, to_lower);
/// let result = String::from_utf8(target).expect("Invalid UTF-8 sequence");
/// assert_eq!(result, "hello world!");
/// ```
///
pub fn lookup<T, S>(target: &mut T, source: &S, table: [u8; 256])
where
    T: AsMut<[u8]> + ?Sized,
    S: AsRef<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    let source_slice = source.as_ref();
    assert!(target_slice.len() >= source_slice.len());
    unsafe {
        sz_lookup(
            target_slice.as_mut_ptr() as *mut c_void,
            source_slice.len(),
            source_slice.as_ptr() as *const c_void,
            table.as_ptr() as _,
        );
    }
}

/// Performs a lookup transformation (LUT), mapping contents of a buffer into the same or other
/// memory region, taking a byte substitution value from the provided table.
///
/// # Arguments
///
/// * `buffer`: A mutable buffer to update inplace.
/// * `table`: Lookup table of 256 substitution values.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let mut to_lower: [u8; 256] = core::array::from_fn(|i| i as u8);
/// for (upper, lower) in ('A'..='Z').zip('a'..='z') {
///     to_lower[upper as usize] = lower as u8;
/// }
/// let mut text = *b"HELLO WORLD!";
/// sz::lookup_inplace(&mut text, to_lower);
/// assert_eq!(text, *b"hello world!");
/// ```
///
pub fn lookup_inplace<T>(buffer: &mut T, table: [u8; 256])
where
    T: AsMut<[u8]> + ?Sized,
{
    let buffer_slice = buffer.as_mut();
    unsafe {
        sz_lookup(
            buffer_slice.as_mut_ptr() as *mut c_void,
            buffer_slice.len(),
            buffer_slice.as_ptr() as *const c_void,
            table.as_ptr() as _,
        );
    }
}

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
pub fn utf8_uncased_fold<T, D>(source: T, destination: &mut D) -> usize
where
    T: AsRef<[u8]>,
    D: AsMut<[u8]> + ?Sized,
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
pub fn utf8_norm<T, D>(source: T, form: Utf8NormalForm, destination: &mut D) -> usize
where
    T: AsRef<[u8]>,
    D: AsMut<[u8]> + ?Sized,
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
pub fn utf8_find_denormalized<T>(source: T, form: Utf8NormalForm) -> Option<usize>
where
    T: AsRef<[u8]>,
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
pub fn utf8_uncased_search<H, N>(haystack: H, needle: N) -> Option<(usize, usize)>
where
    H: AsRef<[u8]>,
    N: Utf8UncasedNeedleArg,
{
    needle.find_uncased_in(haystack.as_ref())
}

/// Trait for types that can be used as a uncased search needle.
///
/// This trait is implemented for:
/// - Any type implementing `AsRef<[u8]>` (strings, byte slices, etc.)
/// - [`Utf8UncasedNeedle`] references for efficient repeated searches
pub trait Utf8UncasedNeedleArg {
    /// Performs the uncased search in the given haystack.
    fn find_uncased_in(self, haystack: &[u8]) -> Option<(usize, usize)>;
}

impl<T: AsRef<[u8]>> Utf8UncasedNeedleArg for T {
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
/// * `a`: First UTF-8 string.
/// * `b`: Second UTF-8 string.
///
/// # Returns
///
/// * `Ordering::Less` if `a < b`
/// * `Ordering::Equal` if `a == b` (uncasedly)
/// * `Ordering::Greater` if `a > b`
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
pub fn utf8_uncased_order<A, B>(a: A, b: B) -> Ordering
where
    A: AsRef<[u8]>,
    B: AsRef<[u8]>,
{
    let a_ref = a.as_ref();
    let b_ref = b.as_ref();

    let result = unsafe {
        sz_utf8_uncased_order(
            a_ref.as_ptr() as *const c_void,
            a_ref.len(),
            b_ref.as_ptr() as *const c_void,
            b_ref.len(),
        )
    };

    match result {
        x if x < 0 => Ordering::Less,
        0 => Ordering::Equal,
        _ => Ordering::Greater,
    }
}

/// Lexicographic (byte-order) comparison of two strings, SIMD-accelerated.
///
/// Mirrors `Ord` on `&[u8]` but uses StringZilla's vectorized `sz_order`.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
/// use stringzilla::stringzilla as sz;
///
/// assert_eq!(sz::order("apple", "banana"), Ordering::Less);
/// assert_eq!(sz::order("abc", "abc"), Ordering::Equal);
/// ```
pub fn order<A, B>(a: A, b: B) -> Ordering
where
    A: AsRef<[u8]>,
    B: AsRef<[u8]>,
{
    let a_ref = a.as_ref();
    let b_ref = b.as_ref();
    let result = unsafe {
        sz_order(
            a_ref.as_ptr() as *const c_void,
            a_ref.len(),
            b_ref.as_ptr() as *const c_void,
            b_ref.len(),
        )
    };
    match result {
        x if x < 0 => Ordering::Less,
        0 => Ordering::Equal,
        _ => Ordering::Greater,
    }
}

/// Byte-level equality of two strings, SIMD-accelerated via `sz_equal`.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// assert!(sz::equal("abc", "abc"));
/// assert!(!sz::equal("abc", "abd"));
/// ```
pub fn equal<A, B>(a: A, b: B) -> bool
where
    A: AsRef<[u8]>,
    B: AsRef<[u8]>,
{
    let a_ref = a.as_ref();
    let b_ref = b.as_ref();
    // `sz_equal` assumes equal lengths; differing lengths can never be byte-equal.
    a_ref.len() == b_ref.len()
        && unsafe {
            sz_equal(
                a_ref.as_ptr() as *const c_void,
                b_ref.as_ptr() as *const c_void,
                a_ref.len(),
            ) != 0
        }
}

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

/// Computes a 64-bit AES-based hash value for a given byte slice `text`.
/// This function is designed to provide a high-quality hash value for use in
/// hash tables, data structures, and cryptographic applications.
/// Unlike the bytesum function, the hash function is order-sensitive.
///
/// # Arguments
///
/// * `text`: The byte slice to compute the checksum for.
/// * `seed`: A 64-bit value that acts as the seed for the hash function.
///
/// # Returns
///
/// A `u64` representing the hash value of the input byte slice.
#[inline(always)]
pub fn hash_with_seed<T>(text: T, seed: u64) -> u64
where
    T: AsRef<[u8]>,
{
    let text_ref = text.as_ref();
    let text_pointer = text_ref.as_ptr() as _;
    let text_length = text_ref.len();
    unsafe { sz_hash(text_pointer, text_length, seed) }
}

/// Computes a 64-bit AES-based hash value for a given byte slice `text`.
/// This function is designed to provide a high-quality hash value for use in
/// hash tables, data structures, and cryptographic applications.
/// Unlike the bytesum function, the hash function is order-sensitive.
///
/// # Arguments
///
/// * `text`: The byte slice to compute the checksum for.
///
/// # Returns
///
/// A `u64` representing the hash value of the input byte slice.
#[inline(always)]
pub fn hash<T>(text: T) -> u64
where
    T: AsRef<[u8]>,
{
    hash_with_seed(text, 0)
}

/// Hashes one byte slice under many seeds at once, writing the results into `out`.
/// Equivalent to `out[i] = hash_with_seed(text, seeds[i])`, but normalizes the input into AES
/// blocks once and replays the cheap per-seed rounds - markedly faster for short strings under
/// many seeds (feature hashing, Count-Min sketches, Bloom/cuckoo filters, MinHash/LSH).
///
/// # Arguments
///
/// * `text`: The byte slice to hash.
/// * `seeds`: The 64-bit seeds to hash under.
/// * `out`: The output buffer, filled with one hash per seed. Must be the same length as `seeds`.
///
/// # Panics
///
/// Panics if `out.len() != seeds.len()`.
#[inline(always)]
pub fn hash_multiseed_into<T>(text: T, seeds: &[u64], out: &mut [u64])
where
    T: AsRef<[u8]>,
{
    assert_eq!(seeds.len(), out.len(), "`out` must have one slot per seed");
    let text_ref = text.as_ref();
    unsafe {
        sz_hash_multiseed(
            text_ref.as_ptr() as _,
            text_ref.len(),
            seeds.as_ptr(),
            seeds.len(),
            out.as_mut_ptr(),
        )
    }
}

/// Locates the first matching substring within `haystack` that equals `needle`.
/// This function is similar to the `memmem()` function in LibC, but, unlike `strstr()`,
/// it requires the length of both haystack and needle to be known beforehand.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needle`: The byte slice to find within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the starting index of the first occurrence of `needle`
/// within `haystack` if found, otherwise `None`.
pub fn find<H, N>(haystack: H, needle: N) -> Option<usize>
where
    H: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let needle_ref = needle.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();
    let needle_pointer = needle_ref.as_ptr() as _;
    let needle_length = needle_ref.len();
    let result = unsafe { sz_find(haystack_pointer, haystack_length, needle_pointer, needle_length) };

    if result.is_null() {
        None
    } else {
        Some(unsafe { result.offset_from(haystack_pointer) }.try_into().unwrap())
    }
}

/// Locates the last matching substring within `haystack` that equals `needle`.
/// This function is useful for finding the most recent or last occurrence of a pattern
/// within a byte slice.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needle`: The byte slice to find within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the starting index of the last occurrence of `needle`
/// within `haystack` if found, otherwise `None`.
#[inline(always)]
pub fn rfind<H, N>(haystack: H, needle: N) -> Option<usize>
where
    H: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let needle_ref = needle.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();
    let needle_pointer = needle_ref.as_ptr() as _;
    let needle_length = needle_ref.len();
    let result = unsafe { sz_rfind(haystack_pointer, haystack_length, needle_pointer, needle_length) };

    if result.is_null() {
        None
    } else {
        Some(unsafe { result.offset_from(haystack_pointer) }.try_into().unwrap())
    }
}

/// Finds the index of the first character in `haystack` that is also present in `needles`.
/// This function is particularly useful for parsing and tokenization tasks where a set of
/// delimiter characters is used.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes to search for within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the first occurrence of any byte from
/// `needles` within `haystack`, if found, otherwise `None`.
#[inline(always)]
pub fn find_byteset<H>(haystack: H, needles: Byteset) -> Option<usize>
where
    H: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();

    let result = unsafe { sz_find_byteset(haystack_pointer, haystack_length, &needles as *const _ as *const c_void) };
    if result.is_null() {
        None
    } else {
        Some(unsafe { result.offset_from(haystack_pointer) }.try_into().unwrap())
    }
}

/// Finds the index of the last character in `haystack` that is also present in `needles`.
/// This can be used to find the last occurrence of any character from a specified set,
/// useful in parsing scenarios such as finding the last delimiter in a string.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes to search for within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the last occurrence of any byte from
/// `needles` within `haystack`, if found, otherwise `None`.
pub fn rfind_byteset<H>(haystack: H, needles: Byteset) -> Option<usize>
where
    H: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();

    let result = unsafe { sz_rfind_byteset(haystack_pointer, haystack_length, &needles as *const _ as *const c_void) };
    if result.is_null() {
        None
    } else {
        Some(unsafe { result.offset_from(haystack_pointer) }.try_into().unwrap())
    }
}

/// Finds the index of the first character in `haystack` that is also present in `needles`.
/// This function is particularly useful for parsing and tokenization tasks where a set of
/// delimiter characters is used.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes to search for within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the first occurrence of any byte from
/// `needles` within `haystack`, if found, otherwise `None`.
#[inline(always)]
pub fn find_byte_from<H, N>(haystack: H, needles: N) -> Option<usize>
where
    H: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    find_byteset(haystack, Byteset::from(needles))
}

/// Finds the index of the last character in `haystack` that is also present in `needles`.
/// This can be used to find the last occurrence of any character from a specified set,
/// useful in parsing scenarios such as finding the last delimiter in a string.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes to search for within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the last occurrence of any byte from
/// `needles` within `haystack`, if found, otherwise `None`.
pub fn rfind_byte_from<H, N>(haystack: H, needles: N) -> Option<usize>
where
    H: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    rfind_byteset(haystack, Byteset::from(needles))
}

/// Finds the index of the first character in `haystack` that is not present in `needles`.
/// This function is useful for skipping over a known set of characters and finding the
/// first character that does not belong to that set.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes that should not be matched within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the first occurrence of any byte not in
/// `needles` within `haystack`, if found, otherwise `None`.
pub fn find_byte_not_from<H, N>(haystack: H, needles: N) -> Option<usize>
where
    H: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    find_byteset(haystack, Byteset::from(needles).inverted())
}

/// Finds the index of the last character in `haystack` that is not present in `needles`.
/// Useful for text processing tasks such as trimming trailing characters that belong to
/// a specified set.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes that should not be matched within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the last occurrence of any byte not in
/// `needles` within `haystack`, if found, otherwise `None`.
pub fn rfind_byte_not_from<H, N>(haystack: H, needles: N) -> Option<usize>
where
    H: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    rfind_byteset(haystack, Byteset::from(needles).inverted())
}

fn replace_all_with_finder<F, R>(
    buffer: &mut Vec<u8>,
    needle_length: usize,
    replacement: &[u8],
    mut find_next: F,
    mut find_prev: R,
) -> Result<usize, Status>
where
    F: FnMut(&[u8], usize) -> Option<usize>,
    R: FnMut(&[u8], usize) -> Option<usize>,
{
    if needle_length == 0 || buffer.is_empty() {
        return Ok(0);
    }

    // Case 1: needle and replacement are the same length – overwrite each match in place.
    if needle_length == replacement.len() {
        let mut replaced = 0;
        let mut search_from = 0;
        while let Some(pos) = find_next(buffer.as_slice(), search_from) {
            copy(&mut buffer[pos..pos + needle_length], &replacement);
            search_from = pos + needle_length;
            replaced += 1;
        }
        return Ok(replaced);
    }

    // Case 2: replacement is shorter – compact forward to minimize memmoves and avoid allocations.
    if needle_length > replacement.len() {
        let mut replaced = 0;
        let mut read = 0;
        let mut write = 0;
        let len = buffer.len();

        while let Some(pos) = find_next(buffer.as_slice(), read) {
            if pos > read {
                let chunk = pos - read;
                unsafe {
                    sz_move(
                        buffer.as_mut_ptr().add(write) as *const c_void,
                        buffer.as_ptr().add(read) as *const c_void,
                        chunk,
                    );
                }
                write += chunk;
            }
            copy(&mut buffer[write..write + replacement.len()], replacement);
            write += replacement.len();
            read = pos + needle_length;
            replaced += 1;
        }

        if read < len {
            let chunk = len - read;
            unsafe {
                sz_move(
                    buffer.as_mut_ptr().add(write) as *const c_void,
                    buffer.as_ptr().add(read) as *const c_void,
                    chunk,
                );
            }
            write += len - read;
        }
        buffer.truncate(write);
        return Ok(replaced);
    }

    // Case 3: replacement is longer – collect match positions once, resize once, then rewrite from the back.
    let mut match_count = 0usize;
    let mut search_from = 0;
    while let Some(pos) = find_next(buffer.as_slice(), search_from) {
        match_count += 1;
        search_from = pos + needle_length;
    }

    if match_count == 0 {
        return Ok(0);
    }

    let original_len = buffer.len();
    let delta = replacement.len() - needle_length;
    let added = match match_count.checked_mul(delta) {
        Some(v) => v,
        None => return Err(Status::OverflowRisk),
    };
    let new_len = match original_len.checked_add(added) {
        Some(v) => v,
        None => return Err(Status::OverflowRisk),
    };
    if let Err(_) = buffer.try_reserve_exact(added) {
        return Err(Status::BadAlloc);
    }
    buffer.resize(new_len, 0);

    let mut read_end = original_len;
    let mut write_end = new_len;

    while let Some(pos) = find_prev(buffer.as_slice(), read_end) {
        let match_end = pos + needle_length;
        let tail_len = read_end - match_end;
        if tail_len > 0 {
            unsafe {
                sz_move(
                    buffer.as_mut_ptr().add(write_end - tail_len) as *const c_void,
                    buffer.as_ptr().add(match_end) as *const c_void,
                    tail_len,
                );
            }
        }
        write_end -= tail_len;
        write_end -= replacement.len();
        copy(&mut buffer[write_end..write_end + replacement.len()], replacement);
        read_end = pos;
    }

    debug_assert_eq!(write_end, read_end, "replace_all backfill mismatch");
    Ok(match_count)
}

/// Tries to replace all non-overlapping occurrences of `needle` inside `buffer` in place.
///
/// The algorithm mirrors the C++ `replace_all` logic:
/// - equal-length replacements simply overwrite matches,
/// - shorter replacements compact forward without allocating,
/// - longer replacements count matches once, resize once, and rewrite from the back.
///
/// Returns the number of replacements performed.
pub fn try_replace_all(buffer: &mut Vec<u8>, needle: &[u8], replacement: &[u8]) -> Result<usize, Status> {
    replace_all_with_finder(
        buffer,
        needle.len(),
        replacement,
        |haystack, start| {
            if start >= haystack.len() {
                None
            } else {
                find(&haystack[start..], needle).map(|offset| start + offset)
            }
        },
        |haystack, end| {
            if end == 0 {
                None
            } else {
                rfind(&haystack[..end], needle)
            }
        },
    )
}

/// Tries to replace all non-overlapping bytes in `buffer` that belong to `byteset` with `replacement`.
///
/// Uses the same three-way strategy as [`try_replace_all`]. If the byteset is empty, the buffer is
/// left untouched. Returns the number of replacements performed.
pub fn try_replace_all_byteset(buffer: &mut Vec<u8>, byteset: Byteset, replacement: &[u8]) -> Result<usize, Status> {
    if byteset.bits.iter().all(|&b| b == 0) {
        return Ok(0);
    }

    replace_all_with_finder(
        buffer,
        1,
        replacement,
        |haystack, start| {
            if start >= haystack.len() {
                None
            } else {
                find_byteset(&haystack[start..], byteset).map(|offset| start + offset)
            }
        },
        |haystack, end| {
            if end == 0 {
                None
            } else {
                rfind_byteset(&haystack[..end], byteset)
            }
        },
    )
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
pub fn count_utf8<T>(text: T) -> usize
where
    T: AsRef<[u8]>,
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
pub fn find_nth_utf8<T>(text: T, n: usize) -> Option<usize>
where
    T: AsRef<[u8]>,
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
    pub fn new(octets: &'a [u8]) -> Self {
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
    pub fn is_empty(&self) -> bool {
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

/// Randomizes the contents of a given byte slice `text` using characters from
/// a specified `alphabet`. This function mutates `text` in place, replacing each
/// byte with a random one from `alphabet`. It is designed for situations where
/// you need to generate random strings or data sequences based on a specific set
/// of characters, such as generating random DNA sequences or testing inputs.
///
/// # Arguments
///
/// * `buffer`: A mutable reference to the data to randomize. This data will be mutated in place.
/// * `nonce`: A 64-bit "number used once" (nonce) value to seed the random number generator.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let mut buffer = vec![0; 10];
/// sz::fill_random(&mut buffer, 42);
/// ```
///
/// After than,  `buffer` is filled with random byte values from 0 to 255.
pub fn fill_random<T>(buffer: &mut T, nonce: u64)
where
    T: AsMut<[u8]> + ?Sized, // Allows for mutable references to dynamically sized types.
{
    let buffer_slice = buffer.as_mut();
    unsafe {
        sz_fill_random(buffer_slice.as_ptr() as _, buffer_slice.len(), nonce);
    }
}

/// A helper type that holds a mapper closure which, given an index,
/// returns the corresponding byte-slice representation.
///
/// The closure is expected to have type `Fn(usize) -> &[u8]` so that callers
/// can write closures like `|i| data[i].as_ref()` or `|i| people[i].name.as_bytes()`.
struct _SliceLookupView<F: Fn(usize) -> &'static [u8]> {
    mapper: F,
}

/// Type-punned wrapper for the slice lookup view
struct _PunnedSliceLookupView {
    get_slice: unsafe fn(*const c_void, usize) -> &'static [u8],
    data: *const c_void,
}

unsafe extern "C" fn _slice_get_start_punned(handle: *const c_void, idx: SortedIdx) -> *const c_void {
    let view = &*(handle as *const _PunnedSliceLookupView);
    let slice = (view.get_slice)(view.data, idx);
    slice.as_ptr() as *const c_void
}

unsafe extern "C" fn _slice_get_length_punned(handle: *const c_void, idx: SortedIdx) -> usize {
    let view = &*(handle as *const _PunnedSliceLookupView);
    let slice = (view.get_slice)(view.data, idx);
    slice.len()
}

/// Type-specific function generator for each concrete type
unsafe fn _get_slice_fn<F>() -> unsafe fn(*const c_void, usize) -> &'static [u8]
where
    F: Fn(usize) -> &'static [u8],
{
    unsafe fn get_slice_impl<F>(data: *const c_void, idx: usize) -> &'static [u8]
    where
        F: Fn(usize) -> &'static [u8],
    {
        let mapper = &*(data as *const F);
        mapper(idx)
    }
    get_slice_impl::<F>
}

/// Knobs for [`argsort`] and [`argsort_by`].
///
/// The default is a full, ascending, byte-lexicographic, **stable** sort (equal elements keep their
/// input order). Tweak the public fields directly or chain the builder methods:
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// let descending = sz::ArgsortOptions::default().reversed();
/// let top_10_folded = sz::ArgsortOptions { uncased: true, top: Some(10), ..Default::default() };
/// # let _ = (descending, top_10_folded);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArgsortOptions {
    /// Sort in descending order; equal elements still keep their input order (stable).
    pub reverse: bool,
    /// Order under Unicode case-folding instead of raw bytes.
    pub uncased: bool,
    /// Only fully order the leading `Some(k)` elements (top-K / partial sort); `None` sorts everything.
    /// The remaining entries of `order` stay a valid - but arbitrary - permutation of the leftover indices.
    pub top: Option<usize>,
}

impl ArgsortOptions {
    /// Sort in descending order.
    pub fn reversed(mut self) -> Self {
        self.reverse = true;
        self
    }
    /// Order under Unicode case-folding instead of raw bytes.
    pub fn uncased(mut self) -> Self {
        self.uncased = true;
        self
    }
    /// Only fully order the leading `count` elements (top-K / partial sort).
    pub fn top(mut self, count: usize) -> Self {
        self.top = Some(count);
        self
    }
}

/// Computes the permutation that sorts `data` by its byte-slice representations.
///
/// The caller supplies an output buffer `order` of length at least `data.len()`; on success the sorted
/// permutation indices are written into its first `data.len()` slots. See [`ArgsortOptions`] for
/// descending, uncased, and top-K variants.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// let fruits = ["banana", "apple", "cherry"];
/// let mut order = [0; 3];
/// sz::argsort(&fruits, &mut order, Default::default()).expect("sort failed");
/// assert_eq!(&order, &[1, 0, 2]); // "apple", "banana", "cherry"
///
/// // Descending, uncased:
/// let labels = ["beta", "Alpha", "BETA"];
/// let mut order = [0; 3];
/// sz::argsort(&labels, &mut order, sz::ArgsortOptions::default().reversed().uncased()).unwrap();
/// assert_eq!(labels[order[0]], "beta"); // "beta"/"BETA" (fold-equal) before "Alpha", stable on ties
/// ```
pub fn argsort<T: AsRef<[u8]>>(data: &[T], order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status> {
    if data.len() > order.len() {
        return Err(Status::BadAlloc);
    }
    argsort_by(|i| data[i].as_ref(), &mut order[..data.len()], options)
}

/// Computes the permutation that sorts items by a caller-provided byte-slice key.
/// The number of items is inferred from the length of the `order` slice.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// struct Person { name: &'static str, age: u32 }
/// let people = [
///     Person { name: "Charlie", age: 20 },
///     Person { name: "Alice", age: 25 },
///     Person { name: "Bob", age: 30 },
/// ];
/// let mut order = [0; 3];
/// sz::argsort_by(|i| people[i].name.as_bytes(), &mut order, Default::default()).expect("sort failed");
/// assert_eq!(&order, &[1, 2, 0]); // "Alice", "Bob", "Charlie"
/// ```
pub fn argsort_by<F, A>(mapper: F, order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status>
where
    F: Fn(usize) -> A,
    A: AsRef<[u8]>,
{
    // Adapter closure: given an index, call the provided mapper and then transmute the
    // resulting slice to have a `'static` lifetime. This transmute is safe as long as
    // the FFI call is synchronous and the returned slices are only used during the call.
    let adapter = move |i: usize| -> &'static [u8] {
        let binding = mapper(i);
        let slice = binding.as_ref();
        unsafe { core::mem::transmute(slice) }
    };

    _argsort_impl(adapter, order, options)
}

/// Helper that takes an adapter (with a concrete type) and performs the FFI call.
fn _argsort_impl<FAdapter>(adapter: FAdapter, order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status>
where
    FAdapter: Fn(usize) -> &'static [u8],
{
    let wrapper = _PunnedSliceLookupView {
        get_slice: unsafe { _get_slice_fn::<FAdapter>() },
        data: &adapter as *const FAdapter as *const c_void,
    };
    let seq = _SzSequence {
        handle: &wrapper as *const _ as *const c_void,
        count: order.len(),
        get_start: Some(_slice_get_start_punned),
        get_length: Some(_slice_get_length_punned),
    };
    let top_count = options.top.unwrap_or(0);
    let reverse = options.reverse as i32;
    let status = unsafe {
        if options.uncased {
            sz_sequence_argsort_utf8_uncased(&seq, core::ptr::null(), order.as_mut_ptr(), top_count, reverse)
        } else {
            sz_sequence_argsort(&seq, core::ptr::null(), order.as_mut_ptr(), top_count, reverse)
        }
    };
    if status == Status::Success {
        Ok(())
    } else {
        Err(status)
    }
}

// ----------------------------------------------------------------------
// Intersection functions
// ----------------------------------------------------------------------

/// Intersects two sequences (inner join) using their default byte-slice views.
///
/// Both sequences must have an output buffer provided (for first and second positions)
/// whose length is at least the minimum of the two input lengths.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// let set1 = ["banana", "apple", "cherry"];
/// let set2 = ["cherry", "orange", "pineapple", "banana"];
/// let mut positions1 = [0; 3]; // at least min(3, 4) == 3 elements.
/// let mut positions2 = [0; 3];
/// let n = sz::intersection(&set1, &set2, 0, &mut positions1, &mut positions2).expect("intersect failed");
/// assert!(n == 2); // "banana" and "cherry" are common.
/// ```
pub fn intersection<T: AsRef<[u8]>>(
    data1: &[T],
    data2: &[T],
    seed: u64,
    positions1: &mut [SortedIdx],
    positions2: &mut [SortedIdx],
) -> Result<usize, Status> {
    let min_count = data1.len().min(data2.len());
    if positions1.len() < min_count || positions2.len() < min_count {
        return Err(Status::BadAlloc);
    }

    // Call the lower-level implementation with accurate counts for both sequences.
    let adapter1 = move |i: usize| -> &'static [u8] {
        // SAFETY: used only during the FFI call
        unsafe { core::mem::transmute::<&[u8], &'static [u8]>(data1[i].as_ref()) }
    };
    let adapter2 = move |j: usize| -> &'static [u8] {
        // SAFETY: used only during the FFI call
        unsafe { core::mem::transmute::<&[u8], &'static [u8]>(data2[j].as_ref()) }
    };
    _intersection_by_impl(
        adapter1,
        adapter2,
        seed,
        positions1,
        positions2,
        data1.len(),
        data2.len(),
    )
}

/// Intersects two sequences (inner join) using their elements corresponding byte-slice views.
/// The caller must provide a closure that maps an index to the byte slice representation of
/// the corresponding element in the first and second sequences.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// #[derive(Debug)]
/// struct Person { name: &'static str, age: u32 }
///
/// let people1 = [
///     Person { name: "Charlie", age: 20 },
///     Person { name: "Alice", age: 25 },
///     Person { name: "Bob", age: 30 },
/// ];
/// let people2 = [
///     Person { name: "Alice", age: 25 },
///     Person { name: "Bob", age: 30 },
///     Person { name: "Charlie", age: 20 },
/// ];
/// let mut positions1 = [0; 3]; // min(people1.len(), people2.len())
/// let mut positions2 = [0; 3]; // min(people1.len(), people2.len())
/// let n = sz::intersection_by(
///     |i| people1[i].name.as_bytes(),
///     |j| people2[j].name.as_bytes(),
///     0,
///     &mut positions1,
///     &mut positions2,
/// ).expect("intersect failed");
/// assert!(n == 3); // "Alice", "Bob", and "Charlie" are common.
/// ```
pub fn intersection_by<F, G, A, B>(
    mapper1: F,
    mapper2: G,
    seed: u64,
    positions1: &mut [SortedIdx],
    positions2: &mut [SortedIdx],
) -> Result<usize, Status>
where
    F: Fn(usize) -> A,
    A: AsRef<[u8]>,
    G: Fn(usize) -> B,
    B: AsRef<[u8]>,
{
    if positions1.len() != positions2.len() {
        return Err(Status::BadAlloc);
    }

    // Adapter closure: given an index, call the provided mapper and then transmute the
    // resulting slice to have a `'static` lifetime. This transmute is safe as long as
    // the FFI call is synchronous and the returned slices are only used during the call.
    let adapter1 = move |i: usize| -> &'static [u8] {
        let binding = mapper1(i);
        let slice = binding.as_ref();
        unsafe { core::mem::transmute(slice) }
    };
    let adapter2 = move |i: usize| -> &'static [u8] {
        let binding = mapper2(i);
        let slice = binding.as_ref();
        unsafe { core::mem::transmute(slice) }
    };

    _intersection_by_impl(
        adapter1,
        adapter2,
        seed,
        positions1,
        positions2,
        positions1.len(),
        positions2.len(),
    )
}

fn _intersection_by_impl<FAdapter, GAdapter>(
    adapter1: FAdapter,
    adapter2: GAdapter,
    seed: u64,
    positions1: &mut [SortedIdx],
    positions2: &mut [SortedIdx],
    count1: usize,
    count2: usize,
) -> Result<usize, Status>
where
    FAdapter: Fn(usize) -> &'static [u8],
    GAdapter: Fn(usize) -> &'static [u8],
{
    let wrapper1 = _PunnedSliceLookupView {
        get_slice: unsafe { _get_slice_fn::<FAdapter>() },
        data: &adapter1 as *const FAdapter as *const c_void,
    };
    let wrapper2 = _PunnedSliceLookupView {
        get_slice: unsafe { _get_slice_fn::<GAdapter>() },
        data: &adapter2 as *const GAdapter as *const c_void,
    };
    let seq1 = _SzSequence {
        handle: &wrapper1 as *const _ as *const c_void,
        count: count1,
        get_start: Some(_slice_get_start_punned),
        get_length: Some(_slice_get_length_punned),
    };
    let seq2 = _SzSequence {
        handle: &wrapper2 as *const _ as *const c_void,
        count: count2,
        get_start: Some(_slice_get_start_punned),
        get_length: Some(_slice_get_length_punned),
    };
    let mut inter_size: usize = 0;
    let status = unsafe {
        sz_sequence_intersect(
            &seq1,
            &seq2,
            core::ptr::null(),
            seed,
            &mut inter_size as *mut usize,
            positions1.as_mut_ptr(),
            positions2.as_mut_ptr(),
        )
    };
    if status == Status::Success {
        Ok(inter_size)
    } else {
        Err(status)
    }
}

pub trait Matcher<'a> {
    fn find(&self, haystack: &'a [u8]) -> Option<usize>;
    fn needle_length(&self) -> usize;
    fn skip_length(&self, include_overlaps: bool, is_reverse: bool) -> usize;
}

pub enum MatcherType<'a> {
    Find(&'a [u8]),
    RFind(&'a [u8]),
    FindFirstOf(&'a [u8]),
    FindLastOf(&'a [u8]),
    FindFirstNotOf(&'a [u8]),
    FindLastNotOf(&'a [u8]),
}

impl<'a> Matcher<'a> for MatcherType<'a> {
    fn find(&self, haystack: &'a [u8]) -> Option<usize> {
        match self {
            MatcherType::Find(needle) => find(haystack, needle),
            MatcherType::RFind(needle) => rfind(haystack, needle),
            MatcherType::FindFirstOf(needles) => find_byte_from(haystack, needles),
            MatcherType::FindLastOf(needles) => rfind_byte_from(haystack, needles),
            MatcherType::FindFirstNotOf(needles) => find_byte_not_from(haystack, needles),
            MatcherType::FindLastNotOf(needles) => rfind_byte_not_from(haystack, needles),
        }
    }

    fn needle_length(&self) -> usize {
        match self {
            MatcherType::Find(needle) | MatcherType::RFind(needle) => needle.len(),
            _ => 1,
        }
    }

    fn skip_length(&self, include_overlaps: bool, is_reverse: bool) -> usize {
        match (include_overlaps, is_reverse) {
            (true, true) => self.needle_length().saturating_sub(1),
            (true, false) => 1,
            (false, true) => 0,
            (false, false) => self.needle_length(),
        }
    }
}

/// An iterator over non-overlapping matches of a pattern in a string slice.
/// This iterator yields the matched substrings in the order they are found.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, FindMatches}};
///
/// let haystack = b"abababa";
/// let matcher = MatcherType::Find(b"aba");
/// let matches: Vec<&[u8]> = FindMatches::new(haystack, matcher, false).collect();
/// assert_eq!(matches, vec![b"aba", b"aba"]);
/// ```
pub struct FindMatches<'a> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
    include_overlaps: bool,
}

impl<'a> FindMatches<'a> {
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>, include_overlaps: bool) -> Self {
        Self {
            haystack,
            matcher,
            position: 0,
            include_overlaps,
        }
    }
}

impl<'a> Iterator for FindMatches<'a> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.haystack.len() {
            return None;
        }

        if let Some(index) = self.matcher.find(&self.haystack[self.position..]) {
            let start = self.position + index;
            let end = start + self.matcher.needle_length();
            self.position = start + self.matcher.skip_length(self.include_overlaps, false);
            Some(&self.haystack[start..end])
        } else {
            self.position = self.haystack.len();
            None
        }
    }
}

/// An iterator over non-overlapping splits of a string slice by a pattern.
/// This iterator yields the substrings between the matches of the pattern.
///
/// By default empty segments are **kept** (adjacent delimiters and leading/trailing matches yield empty
/// slices, mirroring `str::split`). Call [`Self::skip_empty`] to drop zero-length segments. The `STEPS`
/// const-generic mirrors the UTF-8 split iterators for API uniformity; substring/byteset splits search
/// match-by-match, so it does not affect the yielded segments.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, FindSplits}};
///
/// let haystack = b"a,b,c,d";
/// let matcher = MatcherType::Find(b",");
/// let splits: Vec<&[u8]> = FindSplits::new(haystack, matcher).collect();
/// assert_eq!(splits, vec![b"a", b"b", b"c", b"d"]);
/// ```
pub struct FindSplits<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
    last_match: Option<usize>,
    skip_empty: bool,
}

impl<'a> FindSplits<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self::with_steps(haystack, matcher)
    }
}

impl<'a, const STEPS: usize> FindSplits<'a, STEPS> {
    /// Constructs an iterator with an explicit batch size (kept for API uniformity with the UTF-8 splits).
    pub fn with_steps(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self {
            haystack,
            matcher,
            position: 0,
            last_match: None,
            skip_empty: false,
        }
    }

    /// When set, zero-length segments are skipped instead of yielded (opt-in; the default keeps them).
    pub fn skip_empty(mut self) -> Self {
        self.skip_empty = true;
        self
    }

    /// Yields the next raw segment without the `skip_empty` filter.
    #[inline(always)]
    fn next_raw(&mut self) -> Option<&'a [u8]> {
        if self.position > self.haystack.len() {
            return None;
        }

        if let Some(index) = self.matcher.find(&self.haystack[self.position..]) {
            let start = self.position;
            let end = self.position + index;
            self.position = end + self.matcher.needle_length();
            self.last_match = Some(end);
            Some(&self.haystack[start..end])
        } else if self.position < self.haystack.len() || self.last_match.is_some() {
            let start = self.position;
            self.position = self.haystack.len() + 1;
            Some(&self.haystack[start..])
        } else {
            None
        }
    }
}

impl<'a, const STEPS: usize> Iterator for FindSplits<'a, STEPS> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let segment = self.next_raw()?;
            if self.skip_empty && segment.is_empty() {
                continue;
            }
            return Some(segment);
        }
    }
}

/// An iterator over non-overlapping matches of a pattern in a string slice, searching from the end.
/// This iterator yields the matched substrings in reverse order.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, RFindMatches}};
///
/// let haystack = b"abababa";
/// let matcher = MatcherType::RFind(b"aba");
/// let matches: Vec<&[u8]> = RFindMatches::new(haystack, matcher, false).collect();
/// assert_eq!(matches, vec![b"aba", b"aba"]);
/// ```
pub struct RFindMatches<'a> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
    include_overlaps: bool,
}

impl<'a> RFindMatches<'a> {
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>, include_overlaps: bool) -> Self {
        Self {
            haystack,
            matcher,
            position: haystack.len(),
            include_overlaps,
        }
    }
}

impl<'a> Iterator for RFindMatches<'a> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position == 0 {
            return None;
        }

        let search_area = &self.haystack[..self.position];
        if let Some(index) = self.matcher.find(search_area) {
            let start = index;
            let end = start + self.matcher.needle_length();
            let result = Some(&self.haystack[start..end]);

            let skip = self.matcher.skip_length(self.include_overlaps, true);
            self.position = start + skip;

            result
        } else {
            None
        }
    }
}

/// An iterator over non-overlapping splits of a string slice by a pattern, searching from the end.
/// This iterator yields the substrings between the matches of the pattern in reverse order.
///
/// By default empty segments are **kept** (mirroring `str::rsplit`). Call [`Self::skip_empty`] to drop
/// zero-length segments. The `STEPS` const-generic mirrors the UTF-8 split iterators for API uniformity;
/// substring/byteset splits search match-by-match, so it does not affect the yielded segments.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, RFindSplits}};
///
/// let haystack = b"a,b,c,d";
/// let matcher = MatcherType::RFind(b",");
/// let splits: Vec<&[u8]> = RFindSplits::new(haystack, matcher).collect();
/// assert_eq!(splits, vec![b"d", b"c", b"b", b"a"]);
/// ```
pub struct RFindSplits<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: Option<usize>, // End of the not-yet-segmented prefix; `None` once the final segment is yielded
    skip_empty: bool,
}

impl<'a> RFindSplits<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self::with_steps(haystack, matcher)
    }
}

impl<'a, const STEPS: usize> RFindSplits<'a, STEPS> {
    /// Constructs an iterator with an explicit batch size (kept for API uniformity with the UTF-8 splits).
    pub fn with_steps(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self {
            haystack,
            matcher,
            position: Some(haystack.len()),
            skip_empty: false,
        }
    }

    /// When set, zero-length segments are skipped instead of yielded (opt-in; the default keeps them).
    pub fn skip_empty(mut self) -> Self {
        self.skip_empty = true;
        self
    }

    /// Yields the next raw segment (reverse order) without the `skip_empty` filter.
    #[inline(always)]
    fn next_raw(&mut self) -> Option<&'a [u8]> {
        let position = self.position?;
        let search_area = &self.haystack[..position];
        if let Some(index) = self.matcher.find(search_area) {
            let start = index + self.matcher.needle_length();
            self.position = Some(index);
            Some(&self.haystack[start..position])
        } else {
            self.position = None;
            Some(&self.haystack[..position])
        }
    }
}

impl<'a, const STEPS: usize> Iterator for RFindSplits<'a, STEPS> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let segment = self.next_raw()?;
            if self.skip_empty && segment.is_empty() {
                continue;
            }
            return Some(segment);
        }
    }
}

/// An iterator over substrings of UTF-8 text split by newline characters.
///
/// This iterator yields slices between newline characters. The newline characters themselves
/// are not included in the yielded slices. Handles all 8 Unicode newline characters including
/// CRLF as a single delimiter.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{Utf8Lines};
///
/// let text = b"Hello\nWorld\r\nRust";
/// let lines: Vec<&[u8]> = Utf8Lines::new(text).collect();
/// assert_eq!(lines, vec![&b"Hello"[..], &b"World"[..], &b"Rust"[..]]);
/// ```
pub struct Utf8Lines<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    text: &'a [u8],
    suffix: usize,           // Start of the not-yet-segmented suffix (moves past `text.len()` once done)
    starts: [usize; STEPS],  // Buffered segment offsets, relative to `suffix`
    lengths: [usize; STEPS], // Buffered segment lengths
    trailing: Option<(usize, usize)>, // (start, length) of the end-of-text segment when the batch reached EOF
    count: usize,            // Number of yieldable segments in the current batch; `count == 0` is the end sentinel
    index: usize,            // Index of the next segment to yield from the buffer
    advance: usize,          // Bytes to advance `suffix` by once the batch drains
    skip_empty: bool,        // When set, `next()` skips zero-length segments
}

impl<'a> Utf8Lines<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish, e.g.
    /// `Utf8Lines::<1>::with_steps(text)`.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, const STEPS: usize> Utf8Lines<'a, STEPS> {
    /// Constructs an iterator buffering up to `STEPS` delimiters per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        let mut splits = Self {
            text,
            suffix: 0,
            starts: [0; STEPS],
            lengths: [0; STEPS],
            trailing: None,
            count: 0,
            index: 0,
            advance: 0,
            skip_empty: false,
        };
        splits.refill();
        splits.settle();
        splits
    }

    /// When set, zero-length segments are skipped instead of yielded (opt-in; the default keeps them).
    pub fn skip_empty(mut self) -> Self {
        self.skip_empty = true;
        self.settle();
        self
    }

    /// Refill from `suffix`: fetch a delimiter batch and transform it to segments in place.
    fn refill(&mut self) {
        let region = self.text.len() - self.suffix;
        let mut consumed = 0usize;
        let delimiters = unsafe {
            sz_utf8_newlines(
                self.text[self.suffix..].as_ptr() as *const c_void,
                region,
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        // In place: delimiter `d` spans `[starts[d], starts[d] + lengths[d])`; the segment before it runs
        // from the previous delimiter's end to this delimiter's start.
        let mut previous_end = 0usize;
        for d in 0..delimiters {
            let delimiter_start = self.starts[d];
            let delimiter_length = self.lengths[d];
            self.starts[d] = previous_end;
            self.lengths[d] = delimiter_start - previous_end;
            previous_end = delimiter_start + delimiter_length;
        }
        if consumed == region {
            // Batch reached end-of-text: append the trailing segment.
            self.trailing = Some((previous_end, region - previous_end));
            self.count = delimiters + 1;
            self.advance = region + 1; // `region + 1` pushes `suffix` past `text.len()` when drained
        } else {
            self.trailing = None;
            self.count = delimiters;
            self.advance = consumed;
        }
        self.index = 0;
    }

    /// Offset of the segment at `index` relative to `suffix` (the trailing slot lives just past `STEPS`).
    #[inline]
    fn segment(&self, index: usize) -> (usize, usize) {
        match self.trailing {
            Some(trailing) if index == self.count - 1 => trailing,
            _ => (self.starts[index], self.lengths[index]),
        }
    }

    /// Position `index` on the next yieldable segment, refilling and (when `skip_empty`) skipping empties.
    fn settle(&mut self) {
        loop {
            if self.skip_empty {
                while self.index < self.count && self.segment(self.index).1 == 0 {
                    self.index += 1;
                }
            }
            if self.index < self.count || self.count == 0 {
                return;
            }
            self.suffix += self.advance;
            if self.suffix > self.text.len() {
                self.count = 0;
                return;
            }
            self.refill();
        }
    }
}

impl<'a, const STEPS: usize> Iterator for Utf8Lines<'a, STEPS> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        let (offset, length) = self.segment(self.index);
        let begin = self.suffix + offset;
        let end = begin + length;
        self.index += 1;
        self.settle();
        Some(&self.text[begin..end])
    }
}

/// An iterator over segments of UTF-8 text split by whitespace characters.
///
/// Splits on all 25 Unicode "White_Space" characters; N whitespace delimiters yield N+1 segments. By
/// default empty segments are **kept** (matching the C++/Python bindings and `str::split`), so runs of
/// whitespace and leading/trailing whitespace produce empty slices. Call [`Self::skip_empty`] for the
/// `str::split_whitespace`-style behavior that drops empties and yields only non-empty tokens.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{Utf8Tokens};
///
/// // Default KEEP policy: empties around the words are preserved.
/// let text = b"  hi  ";
/// let segments: Vec<&[u8]> = Utf8Tokens::new(text).collect();
/// assert_eq!(segments, vec![&b""[..], &b""[..], &b"hi"[..], &b""[..], &b""[..]]);
///
/// // Opt in to dropping empties for token-style splitting.
/// let tokens: Vec<&[u8]> = Utf8Tokens::new(text).skip_empty().collect();
/// assert_eq!(tokens, vec![&b"hi"[..]]);
/// ```
pub struct Utf8Tokens<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    text: &'a [u8],
    suffix: usize,           // Start of the not-yet-segmented suffix (moves past `text.len()` once done)
    starts: [usize; STEPS],  // Buffered segment offsets, relative to `suffix`
    lengths: [usize; STEPS], // Buffered segment lengths
    trailing: Option<(usize, usize)>, // (start, length) of the end-of-text segment when the batch reached EOF
    count: usize,            // Number of yieldable segments in the current batch; `count == 0` is the end sentinel
    index: usize,            // Index of the next segment to yield from the buffer
    advance: usize,          // Bytes to advance `suffix` by once the batch drains
    skip_empty: bool,        // When set, `next()` skips zero-length segments
}

impl<'a> Utf8Tokens<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish, e.g.
    /// `Utf8Tokens::<1>::with_steps(text)`.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, const STEPS: usize> Utf8Tokens<'a, STEPS> {
    /// Constructs an iterator buffering up to `STEPS` delimiters per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        let mut splits = Self {
            text,
            suffix: 0,
            starts: [0; STEPS],
            lengths: [0; STEPS],
            trailing: None,
            count: 0,
            index: 0,
            advance: 0,
            skip_empty: false,
        };
        splits.refill();
        splits.settle();
        splits
    }

    /// When set, zero-length segments are skipped, recovering the `str::split_whitespace` token behavior
    /// (no leading/trailing/inner empties). The default keeps empties to mirror C++/Python.
    pub fn skip_empty(mut self) -> Self {
        self.skip_empty = true;
        self.settle();
        self
    }

    /// Refill from `suffix`: fetch a delimiter batch and transform it to segments in place.
    fn refill(&mut self) {
        let region = self.text.len() - self.suffix;
        let mut consumed = 0usize;
        let delimiters = unsafe {
            sz_utf8_whitespaces(
                self.text[self.suffix..].as_ptr() as *const c_void,
                region,
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        let mut previous_end = 0usize;
        for d in 0..delimiters {
            let delimiter_start = self.starts[d];
            let delimiter_length = self.lengths[d];
            self.starts[d] = previous_end;
            self.lengths[d] = delimiter_start - previous_end;
            previous_end = delimiter_start + delimiter_length;
        }
        if consumed == region {
            self.trailing = Some((previous_end, region - previous_end));
            self.count = delimiters + 1;
            self.advance = region + 1;
        } else {
            self.trailing = None;
            self.count = delimiters;
            self.advance = consumed;
        }
        self.index = 0;
    }

    /// Offset/length of the segment at `index` relative to `suffix` (the trailing slot lives past `STEPS`).
    #[inline]
    fn segment(&self, index: usize) -> (usize, usize) {
        match self.trailing {
            Some(trailing) if index == self.count - 1 => trailing,
            _ => (self.starts[index], self.lengths[index]),
        }
    }

    /// Position `index` on the next yieldable segment, refilling and (when `skip_empty`) skipping empties.
    fn settle(&mut self) {
        loop {
            if self.skip_empty {
                while self.index < self.count && self.segment(self.index).1 == 0 {
                    self.index += 1;
                }
            }
            if self.index < self.count || self.count == 0 {
                return;
            }
            self.suffix += self.advance;
            if self.suffix > self.text.len() {
                self.count = 0;
                return;
            }
            self.refill();
        }
    }
}

impl<'a, const STEPS: usize> Iterator for Utf8Tokens<'a, STEPS> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        let (offset, length) = self.segment(self.index);
        let begin = self.suffix + offset;
        let end = begin + length;
        self.index += 1;
        self.settle();
        Some(&self.text[begin..end])
    }
}

/// Default batch size for buffering the `sz_utf8_*` boundary kernels' output, mirroring the core
/// `sz_iterators_default_steps_k` enum in `include/stringzilla/utf8_words.h`. Buffering this many
/// boundaries per call amortizes the per-item dispatch/FFI overhead without an unbounded buffer; the
/// kernels report `bytes_consumed`, so a full buffer simply resumes on the next call.
pub const ITERATORS_DEFAULT_STEPS: usize = 64;

/// An iterator over UAX-29 words in UTF-8 text, in order.
///
/// Unlike whitespace splitting, the words tile the input: every byte belongs to exactly one word, so
/// consecutive words are contiguous and no empty slices are produced. Follows the Unicode TR29 rules.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::Utf8Words;
///
/// let words: Vec<&[u8]> = Utf8Words::new(b"Hi, world").collect();
/// assert_eq!(words, vec![&b"Hi"[..], &b","[..], &b" "[..], &b"world"[..]]);
/// ```
pub struct Utf8Words<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    text: &'a [u8],
    suffix: usize, // Start of the not-yet-segmented suffix (a TR29 boundary; `text.len()` once exhausted)
    starts: [usize; STEPS], // Buffered word offsets, relative to `suffix`
    lengths: [usize; STEPS], // Buffered word lengths
    count: usize,  // Number of buffered words (0 once exhausted)
    index: usize,  // Index of the next word to yield from the buffer
}

impl<'a> Utf8Words<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish, e.g.
    /// `Utf8Words::<1>::with_steps(text)`.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, const STEPS: usize> Utf8Words<'a, STEPS> {
    /// Constructs an iterator buffering up to `STEPS` words per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        let mut splits = Self {
            text,
            suffix: 0,
            starts: [0; STEPS],
            lengths: [0; STEPS],
            count: 0,
            index: 0,
        };
        splits.fill();
        splits
    }

    /// Refills the buffer from the current suffix; `count` becomes 0 once the suffix is empty.
    fn fill(&mut self) {
        let mut consumed = 0usize;
        self.count = unsafe {
            sz_utf8_words(
                self.text[self.suffix..].as_ptr() as *const c_void,
                self.text.len() - self.suffix,
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        self.index = 0;
    }
}

impl<'a, const STEPS: usize> Iterator for Utf8Words<'a, STEPS> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            if self.count == 0 {
                return None; // Empty input or fully drained.
            }
            // Batch drained: advance past the last word (a TR29 boundary) and refill from the remaining suffix.
            self.suffix += self.starts[self.count - 1] + self.lengths[self.count - 1];
            self.fill();
            if self.count == 0 {
                return None;
            }
        }
        let begin = self.suffix + self.starts[self.index];
        let end = begin + self.lengths[self.index];
        self.index += 1;
        Some(&self.text[begin..end])
    }
}

/// An iterator over UAX-29 grapheme clusters in UTF-8 text, in order.
///
/// Unlike whitespace splitting, the grapheme clusters tile the input: every byte belongs to exactly one
/// grapheme cluster, so consecutive clusters are contiguous and no empty slices are produced. Follows the
/// Unicode TR29 rules.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::Utf8Graphemes;
///
/// let graphemes: Vec<&[u8]> = Utf8Graphemes::new(b"Hi!").collect();
/// assert_eq!(graphemes, vec![&b"H"[..], &b"i"[..], &b"!"[..]]);
/// ```
pub struct Utf8Graphemes<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    text: &'a [u8],
    suffix: usize, // Start of the not-yet-segmented suffix (a TR29 boundary; `text.len()` once exhausted)
    starts: [usize; STEPS], // Buffered grapheme offsets, relative to `suffix`
    lengths: [usize; STEPS], // Buffered grapheme lengths
    count: usize,  // Number of buffered graphemes (0 once exhausted)
    index: usize,  // Index of the next grapheme to yield from the buffer
}

impl<'a> Utf8Graphemes<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish, e.g.
    /// `Utf8Graphemes::<1>::with_steps(text)`.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, const STEPS: usize> Utf8Graphemes<'a, STEPS> {
    /// Constructs an iterator buffering up to `STEPS` grapheme clusters per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        let mut splits = Self {
            text,
            suffix: 0,
            starts: [0; STEPS],
            lengths: [0; STEPS],
            count: 0,
            index: 0,
        };
        splits.fill();
        splits
    }

    /// Refills the buffer from the current suffix; `count` becomes 0 once the suffix is empty.
    fn fill(&mut self) {
        let mut consumed = 0usize;
        self.count = unsafe {
            sz_utf8_graphemes(
                self.text[self.suffix..].as_ptr() as *const c_void,
                self.text.len() - self.suffix,
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        self.index = 0;
    }
}

impl<'a, const STEPS: usize> Iterator for Utf8Graphemes<'a, STEPS> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            if self.count == 0 {
                return None; // Empty input or fully drained.
            }
            // Batch drained: advance past the last grapheme (a TR29 boundary) and refill from the remaining suffix.
            self.suffix += self.starts[self.count - 1] + self.lengths[self.count - 1];
            self.fill();
            if self.count == 0 {
                return None;
            }
        }
        let begin = self.suffix + self.starts[self.index];
        let end = begin + self.lengths[self.index];
        self.index += 1;
        Some(&self.text[begin..end])
    }
}

/// An iterator over UAX-29 sentences in UTF-8 text, in order.
///
/// Unlike whitespace splitting, the sentences tile the input: every byte belongs to exactly one
/// sentence, so consecutive sentences are contiguous and no empty slices are produced. Follows the
/// Unicode TR29 rules.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::Utf8Sentences;
///
/// let sentences: Vec<&[u8]> = Utf8Sentences::new(b"Hi. Bye.").collect();
/// assert_eq!(sentences, vec![&b"Hi. "[..], &b"Bye."[..]]);
/// ```
pub struct Utf8Sentences<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    text: &'a [u8],
    suffix: usize, // Start of the not-yet-segmented suffix (a TR29 boundary; `text.len()` once exhausted)
    starts: [usize; STEPS], // Buffered sentence offsets, relative to `suffix`
    lengths: [usize; STEPS], // Buffered sentence lengths
    count: usize,  // Number of buffered sentences (0 once exhausted)
    index: usize,  // Index of the next sentence to yield from the buffer
}

impl<'a> Utf8Sentences<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish, e.g.
    /// `Utf8Sentences::<1>::with_steps(text)`.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, const STEPS: usize> Utf8Sentences<'a, STEPS> {
    /// Constructs an iterator buffering up to `STEPS` sentences per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        let mut splits = Self {
            text,
            suffix: 0,
            starts: [0; STEPS],
            lengths: [0; STEPS],
            count: 0,
            index: 0,
        };
        splits.fill();
        splits
    }

    /// Refills the buffer from the current suffix; `count` becomes 0 once the suffix is empty.
    fn fill(&mut self) {
        let mut consumed = 0usize;
        self.count = unsafe {
            sz_utf8_sentences(
                self.text[self.suffix..].as_ptr() as *const c_void,
                self.text.len() - self.suffix,
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        self.index = 0;
    }
}

impl<'a, const STEPS: usize> Iterator for Utf8Sentences<'a, STEPS> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            if self.count == 0 {
                return None; // Empty input or fully drained.
            }
            // Batch drained: advance past the last sentence (a TR29 boundary) and refill from the remaining suffix.
            self.suffix += self.starts[self.count - 1] + self.lengths[self.count - 1];
            self.fill();
            if self.count == 0 {
                return None;
            }
        }
        let begin = self.suffix + self.starts[self.index];
        let end = begin + self.lengths[self.index];
        self.index += 1;
        Some(&self.text[begin..end])
    }
}

/// An iterator over UAX-14 line break opportunities in UTF-8 text, in order.
///
/// Unlike whitespace splitting, the lines tile the input: every byte belongs to exactly one line, so
/// consecutive lines are contiguous and no empty slices are produced. Follows the Unicode TR14 rules.
///
/// Each yielded segment ends at a TR14 break opportunity, including soft breaks where a renderer *may*
/// wrap but is not required to. To split only on hard line breaks (the "splitlines" behaviour), use the
/// newline API ([`StringZillable::sz_utf8_lines`]) instead.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::Utf8Linebreaks;
///
/// let lines: Vec<&[u8]> = Utf8Linebreaks::new(b"Hi\nBye").collect();
/// assert_eq!(lines, vec![&b"Hi\n"[..], &b"Bye"[..]]);
/// ```
pub struct Utf8Linebreaks<'a, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    text: &'a [u8],
    suffix: usize, // Start of the not-yet-segmented suffix (a TR14 boundary; `text.len()` once exhausted)
    starts: [usize; STEPS], // Buffered line offsets, relative to `suffix`
    lengths: [usize; STEPS], // Buffered line lengths
    count: usize,  // Number of buffered lines (0 once exhausted)
    index: usize,  // Index of the next line to yield from the buffer
}

impl<'a> Utf8Linebreaks<'a, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    /// For an explicit batch size use [`Self::with_steps`] with a turbofish, e.g.
    /// `Utf8Linebreaks::<1>::with_steps(text)`.
    pub fn new(text: &'a [u8]) -> Self {
        Self::with_steps(text)
    }
}

impl<'a, const STEPS: usize> Utf8Linebreaks<'a, STEPS> {
    /// Constructs an iterator buffering up to `STEPS` lines per FFI call.
    pub fn with_steps(text: &'a [u8]) -> Self {
        let mut splits = Self {
            text,
            suffix: 0,
            starts: [0; STEPS],
            lengths: [0; STEPS],
            count: 0,
            index: 0,
        };
        splits.fill();
        splits
    }

    /// Refills the buffer from the current suffix; `count` becomes 0 once the suffix is empty.
    fn fill(&mut self) {
        let mut consumed = 0usize;
        self.count = unsafe {
            sz_utf8_linebreaks(
                self.text[self.suffix..].as_ptr() as *const c_void,
                self.text.len() - self.suffix,
                self.starts.as_mut_ptr(),
                self.lengths.as_mut_ptr(),
                STEPS,
                &mut consumed,
            )
        };
        self.index = 0;
    }
}

impl<'a, const STEPS: usize> Iterator for Utf8Linebreaks<'a, STEPS> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            if self.count == 0 {
                return None; // Empty input or fully drained.
            }
            // Batch drained: advance past the last line (a TR14 boundary) and refill from the remaining suffix.
            self.suffix += self.starts[self.count - 1] + self.lengths[self.count - 1];
            self.fill();
            if self.count == 0 {
                return None;
            }
        }
        let begin = self.suffix + self.starts[self.index];
        let end = begin + self.lengths[self.index];
        self.index += 1;
        Some(&self.text[begin..end])
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
/// let matches: Vec<IndexSpan> = Utf8UncasedMatches::with_overlaps(haystack, b"aA", true).collect();
/// assert_eq!(matches.len(), 4); // Overlapping matches
/// ```
pub struct Utf8UncasedMatches<'a> {
    haystack: &'a [u8],
    needle: &'a [u8],
    metadata: Utf8UncasedNeedleMetadata,
    position: usize,
    include_overlaps: bool,
}

impl<'a> Utf8UncasedMatches<'a> {
    /// Creates a new iterator for non-overlapping uncased matches.
    pub fn new(haystack: &'a [u8], needle: &'a [u8]) -> Self {
        Self {
            haystack,
            needle,
            metadata: Utf8UncasedNeedleMetadata::default(),
            position: 0,
            include_overlaps: false,
        }
    }

    /// Creates a new iterator with configurable overlap behavior.
    pub fn with_overlaps(haystack: &'a [u8], needle: &'a [u8], include_overlaps: bool) -> Self {
        Self {
            haystack,
            needle,
            metadata: Utf8UncasedNeedleMetadata::default(),
            position: 0,
            include_overlaps,
        }
    }
}

impl<'a> Iterator for Utf8UncasedMatches<'a> {
    type Item = IndexSpan;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.haystack.len() {
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
            self.position = self.haystack.len();
            None
        } else {
            let offset_in_remaining = unsafe { result.offset_from(remaining.as_ptr() as *const c_void) } as usize;
            let absolute_offset = self.position + offset_in_remaining;

            // Advance position for next search
            if self.include_overlaps {
                self.position = absolute_offset + 1;
            } else {
                self.position = absolute_offset + matched_length;
            }

            Some(IndexSpan::new(absolute_offset, matched_length))
        }
    }
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
    /// let lines: Vec<&str> = text.sz_utf8_lines()
    ///     .map(|line| std::str::from_utf8(line).unwrap())
    ///     .collect();
    /// assert_eq!(lines, vec!["Hello", "World", "Rust"]);
    /// ```
    fn sz_utf8_lines(&self) -> Utf8Lines<'_>;

    /// Returns an iterator over segments split by UTF-8 whitespace characters.
    ///
    /// Handles all 25 Unicode "White_Space" characters; N delimiters yield N+1 segments. By default
    /// **empty segments are kept** (matching the C++/Python bindings and `str::split`), so runs of
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
    /// let segments: Vec<&str> = text.sz_utf8_tokens()
    ///     .map(|segment| std::str::from_utf8(segment).unwrap())
    ///     .collect();
    /// assert_eq!(segments, vec!["Hello", "", "World", "Rust"]);
    ///
    /// // skip_empty: drops the empties to yield only the non-empty tokens.
    /// let tokens: Vec<&str> = text.sz_utf8_tokens()
    ///     .skip_empty()
    ///     .map(|token| std::str::from_utf8(token).unwrap())
    ///     .collect();
    /// assert_eq!(tokens, vec!["Hello", "World", "Rust"]);
    /// ```
    fn sz_utf8_tokens(&self) -> Utf8Tokens<'_>;

    /// Returns an iterator over UAX-29 words (Unicode TR29), in order. Words tile the input contiguously.
    fn sz_utf8_words(&self) -> Utf8Words<'_>;

    /// Returns an iterator over UAX-29 grapheme clusters (Unicode TR29), in order. Clusters tile the input contiguously.
    fn sz_utf8_graphemes(&self) -> Utf8Graphemes<'_>;

    /// Returns an iterator over UAX-29 sentences (Unicode TR29), in order. Sentences tile the input contiguously.
    fn sz_utf8_sentences(&self) -> Utf8Sentences<'_>;

    /// Returns an iterator over UAX-14 line-break opportunities (Unicode TR14), in order. Linewrap segments tile the
    /// input contiguously, including soft break opportunities. For hard line splits only, use
    /// [`Self::sz_utf8_lines`].
    fn sz_utf8_linebreaks(&self) -> Utf8Linebreaks<'_>;
}

/// Trait for binary string operations that take a needle parameter.
/// These operations include searching, splitting, and pattern matching.
///
/// # Examples
///
/// Basic usage on a string slice:
///
/// ```
/// use stringzilla::sz::StringZillableBinary;
///
/// let haystack = "Hello, world!";
/// assert_eq!(haystack.sz_find("world".as_bytes()), Some(7));
/// ```
pub trait StringZillableBinary<'a, N>
where
    N: AsRef<[u8]> + 'a,
{
    /// Searches for the first occurrence of `needle` in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find("world".as_bytes()), Some(7));
    /// ```
    fn sz_find(&self, needle: N) -> Option<usize>;

    /// Searches for the last occurrence of `needle` in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world, world!";
    /// assert_eq!(haystack.sz_rfind("world".as_bytes()), Some(14));
    /// ```
    fn sz_rfind(&self, needle: N) -> Option<usize>;

    /// Finds the index of the first character in `self` that is also present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find_byte_from("aeiou".as_bytes()), Some(1));
    /// ```
    fn sz_find_byte_from(&self, needles: N) -> Option<usize>;

    /// Finds the index of the last character in `self` that is also present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_rfind_byte_from("aeiou".as_bytes()), Some(8));
    /// ```
    fn sz_rfind_byte_from(&self, needles: N) -> Option<usize>;

    /// Finds the index of the first character in `self` that is not present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find_byte_not_from("aeiou".as_bytes()), Some(0));
    /// ```
    fn sz_find_byte_not_from(&self, needles: N) -> Option<usize>;

    /// Finds the index of the last character in `self` that is not present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_rfind_byte_not_from("aeiou".as_bytes()), Some(12));
    /// ```
    fn sz_rfind_byte_not_from(&self, needles: N) -> Option<usize>;

    /// Returns an iterator over all non-overlapping matches of the given `needle` in `self`.
    ///
    /// # Arguments
    ///
    /// * `needle`: The byte slice to search for within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"abababa";
    /// let needle = b"aba";
    /// let matches: Vec<&[u8]> = haystack.sz_matches(needle).collect();
    /// assert_eq!(matches, vec![b"aba", b"aba", b"aba"]);
    /// ```
    fn sz_matches(&'a self, needle: &'a N) -> FindMatches<'a>;

    /// Returns an iterator over all non-overlapping matches of the given `needle` in `self`, searching from the end.
    ///
    /// # Arguments
    ///
    /// * `needle`: The byte slice to search for within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"abababa";
    /// let needle = b"aba";
    /// let matches: Vec<&[u8]> = haystack.sz_rmatches(needle).collect();
    /// assert_eq!(matches, vec![b"aba", b"aba", b"aba"]);
    /// ```
    fn sz_rmatches(&'a self, needle: &'a N) -> RFindMatches<'a>;

    /// Returns an iterator over the substrings of `self` that are separated by the given `needle`.
    ///
    /// # Arguments
    ///
    /// * `needle`: The byte slice to split `self` by.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"a,b,c,d";
    /// let needle = b",";
    /// let splits: Vec<&[u8]> = haystack.sz_splits(needle).collect();
    /// assert_eq!(splits, vec![b"a", b"b", b"c", b"d"]);
    /// ```
    fn sz_splits(&'a self, needle: &'a N) -> FindSplits<'a>;

    /// Returns an iterator over the substrings of `self` that are separated by the given `needle`, searching from the end.
    ///
    /// # Arguments
    ///
    /// * `needle`: The byte slice to split `self` by.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"a,b,c,d";
    /// let needle = b",";
    /// let splits: Vec<&[u8]> = haystack.sz_rsplits(needle).collect();
    /// assert_eq!(splits, vec![b"d", b"c", b"b", b"a"]);
    /// ```
    fn sz_rsplits(&'a self, needle: &'a N) -> RFindSplits<'a>;

    /// Returns an iterator over all non-overlapping matches of any of the bytes in `needles` within `self`.
    ///
    /// # Arguments
    ///
    /// * `needles`: The set of bytes to search for within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"Hello, world!";
    /// let needles = b"aeiou";
    /// let matches: Vec<&[u8]> = haystack.sz_find_first_of(needles).collect();
    /// assert_eq!(matches, vec![b"e", b"o", b"o"]);
    /// ```
    fn sz_find_first_of(&'a self, needles: &'a N) -> FindMatches<'a>;

    /// Returns an iterator over all non-overlapping matches of any of the bytes in `needles` within `self`, searching from the end.
    ///
    /// # Arguments
    ///
    /// * `needles`: The set of bytes to search for within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"Hello, world!";
    /// let needles = b"aeiou";
    /// let matches: Vec<&[u8]> = haystack.sz_find_last_of(needles).collect();
    /// assert_eq!(matches, vec![b"o", b"o", b"e"]);
    /// ```
    fn sz_find_last_of(&'a self, needles: &'a N) -> RFindMatches<'a>;

    /// Returns an iterator over all non-overlapping matches of any byte not in `needles` within `self`.
    ///
    /// # Arguments
    ///
    /// * `needles`: The set of bytes that should not be matched within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"Hello, world!";
    /// let needles = b"aeiou";
    /// let matches: Vec<&[u8]> = haystack.sz_find_first_not_of(needles).collect();
    /// assert_eq!(matches, vec![b"H", b"l", b"l", b",", b" ", b"w", b"r", b"l", b"d", b"!"]);
    /// ```
    fn sz_find_first_not_of(&'a self, needles: &'a N) -> FindMatches<'a>;

    /// Returns an iterator over all non-overlapping matches of any byte not in `needles` within `self`, searching from the end.
    ///
    /// # Arguments
    ///
    /// * `needles`: The set of bytes that should not be matched within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"Hello, world!";
    /// let needles = b"aeiou";
    /// let matches: Vec<&[u8]> = haystack.sz_find_last_not_of(needles).collect();
    /// assert_eq!(matches, vec![b"!", b"d", b"l", b"r", b"w", b" ", b",", b"l", b"l", b"H"]);
    /// ```
    fn sz_find_last_not_of(&'a self, needles: &'a N) -> RFindMatches<'a>;
}

impl<T> StringZillableUnary for T
where
    T: AsRef<[u8]> + ?Sized,
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

    fn sz_utf8_lines(&self) -> Utf8Lines<'_> {
        Utf8Lines::new(self.as_ref())
    }

    fn sz_utf8_tokens(&self) -> Utf8Tokens<'_> {
        Utf8Tokens::new(self.as_ref())
    }

    fn sz_utf8_words(&self) -> Utf8Words<'_> {
        Utf8Words::new(self.as_ref())
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

impl<'a, T, N> StringZillableBinary<'a, N> for T
where
    T: AsRef<[u8]> + ?Sized,
    N: AsRef<[u8]> + 'a,
{
    fn sz_find(&self, needle: N) -> Option<usize> {
        find(self, needle)
    }

    fn sz_rfind(&self, needle: N) -> Option<usize> {
        rfind(self, needle)
    }

    fn sz_find_byte_from(&self, needles: N) -> Option<usize> {
        find_byte_from(self, needles)
    }

    fn sz_rfind_byte_from(&self, needles: N) -> Option<usize> {
        rfind_byte_from(self, needles)
    }

    fn sz_find_byte_not_from(&self, needles: N) -> Option<usize> {
        find_byte_not_from(self, needles)
    }

    fn sz_rfind_byte_not_from(&self, needles: N) -> Option<usize> {
        rfind_byte_not_from(self, needles)
    }

    fn sz_matches(&'a self, needle: &'a N) -> FindMatches<'a> {
        FindMatches::new(self.as_ref(), MatcherType::Find(needle.as_ref()), true)
    }

    fn sz_rmatches(&'a self, needle: &'a N) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), MatcherType::RFind(needle.as_ref()), true)
    }

    fn sz_splits(&'a self, needle: &'a N) -> FindSplits<'a> {
        FindSplits::new(self.as_ref(), MatcherType::Find(needle.as_ref()))
    }

    fn sz_rsplits(&'a self, needle: &'a N) -> RFindSplits<'a> {
        RFindSplits::new(self.as_ref(), MatcherType::RFind(needle.as_ref()))
    }

    fn sz_find_first_of(&'a self, needles: &'a N) -> FindMatches<'a> {
        FindMatches::new(self.as_ref(), MatcherType::FindFirstOf(needles.as_ref()), true)
    }

    fn sz_find_last_of(&'a self, needles: &'a N) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), MatcherType::FindLastOf(needles.as_ref()), true)
    }

    fn sz_find_first_not_of(&'a self, needles: &'a N) -> FindMatches<'a> {
        FindMatches::new(self.as_ref(), MatcherType::FindFirstNotOf(needles.as_ref()), true)
    }

    fn sz_find_last_not_of(&'a self, needles: &'a N) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), MatcherType::FindLastNotOf(needles.as_ref()), true)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use std::borrow::Cow;
    use std::collections::{HashMap, HashSet};
    use std::hash::Hasher as _;

    use super::*;
    use crate::sz;

    #[test]
    fn metadata() {
        // Runtime dispatch is on with the `dynamic-dispatch` feature (default) and off for the
        // compile-time-dispatch build, where the best ISA tier is baked in instead of table-routed.
        assert_eq!(sz::dynamic_dispatch(), cfg!(feature = "dynamic-dispatch"));
        assert!(sz::capabilities().as_str().len() > 0);
    }

    #[test]
    fn bytesum() {
        assert_eq!(sz::bytesum("hi"), 209u64);
    }

    #[test]
    fn hash() {
        let hash_hello = sz::hash("Hello");
        let hash_world = sz::hash("World");
        assert_ne!(hash_hello, hash_world);

        // Hashing should work the same for any seed
        for seed in [0u64, 42, 123456789].iter() {
            // Single-pass hashing
            assert_eq!(
                sz::Hasher::new(*seed).update("Hello".as_bytes()).digest(),
                sz::hash_with_seed("Hello", *seed)
            );
            // Dual pass for short strings
            assert_eq!(
                sz::Hasher::new(*seed)
                    .update("Hello".as_bytes())
                    .update("World".as_bytes())
                    .digest(),
                sz::hash_with_seed("HelloWorld", *seed)
            );
        }
    }

    #[test]
    fn streaming_hash() {
        let mut hasher = sz::Hasher::new(123);
        hasher.write(b"Hello, ");
        hasher.write(b"world!");
        let streamed = hasher.finish();

        let mut hasher = sz::Hasher::new(123);
        hasher.write(b"Hello, world!");
        let expected = hasher.finish();
        assert_eq!(streamed, expected);
    }

    #[test]
    fn multiseed_hash() {
        // More than four seeds to exercise the 4-wide tail handling on the Ice Lake backend.
        let seeds: Vec<u64> = (0..9u64)
            .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(7))
            .collect();
        let texts: [&[u8]; 5] = [
            b"",
            b"token",
            b"sixteen_bytes!!!",
            b"sixty four chars exactly here to fill one whole block boundary..",
            b"a string definitely longer than sixty four bytes to hit the wide path here please",
        ];
        for text in texts {
            for k in 0..=seeds.len() {
                let mut out = vec![0u64; k];
                sz::hash_multiseed_into(text, &seeds[..k], &mut out);
                for i in 0..k {
                    assert_eq!(
                        out[i],
                        sz::hash_with_seed(text, seeds[i]),
                        "len={} k={} i={}",
                        text.len(),
                        k,
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn hashmap_with_sz() {
        let mut map: HashMap<&str, i32, sz::BuildSzHasher> = HashMap::with_hasher(sz::BuildSzHasher::with_seed(0));
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);
        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.get("b"), Some(&2));
        assert_eq!(map.get("c"), Some(&3));
        assert!(map.get("z").is_none());
    }

    #[test]
    fn hashset_with_sz() {
        let mut set: HashSet<&str, sz::BuildSzHasher> = HashSet::with_hasher(sz::BuildSzHasher::with_seed(42));
        assert!(set.insert("alpha"));
        assert!(set.insert("beta"));
        assert!(set.contains("alpha"));
        assert!(set.contains("beta"));
        assert!(!set.contains("gamma"));
        let len_before = set.len();
        assert!(!set.insert("alpha"));
        assert_eq!(set.len(), len_before);
    }

    #[test]
    fn search() {
        let my_string: String = String::from("Hello, world!");
        let my_str: &str = my_string.as_str();
        let my_cow_str: Cow<'_, str> = Cow::from(&my_string);

        // Identical to `memchr::memmem::find` and `memchr::memmem::rfind` functions
        assert_eq!(sz::find("Hello, world!", "world"), Some(7));
        assert_eq!(sz::rfind("Hello, world!", "world"), Some(7));

        // Use the generic function with a String
        let world_string = String::from("world");
        assert_eq!(my_string.sz_find(&world_string), Some(7));
        assert_eq!(my_string.sz_rfind(&world_string), Some(7));
        assert_eq!(my_string.sz_find_byte_from(&world_string), Some(2));
        assert_eq!(my_string.sz_rfind_byte_from(&world_string), Some(11));
        assert_eq!(my_string.sz_find_byte_not_from(&world_string), Some(0));
        assert_eq!(my_string.sz_rfind_byte_not_from(&world_string), Some(12));

        // Use the generic function with a &str
        assert_eq!(my_str.sz_find("world"), Some(7));
        assert_eq!(my_str.sz_rfind("world"), Some(7));
        assert_eq!(my_str.sz_find_byte_from("world"), Some(2));
        assert_eq!(my_str.sz_rfind_byte_from("world"), Some(11));
        assert_eq!(my_str.sz_find_byte_not_from("world"), Some(0));
        assert_eq!(my_str.sz_rfind_byte_not_from("world"), Some(12));

        // Use the generic function with a Cow<'_, str>
        assert_eq!(my_cow_str.as_ref().sz_find("world"), Some(7));
        assert_eq!(my_cow_str.as_ref().sz_rfind("world"), Some(7));
        assert_eq!(my_cow_str.as_ref().sz_find_byte_from("world"), Some(2));
        assert_eq!(my_cow_str.as_ref().sz_rfind_byte_from("world"), Some(11));
        assert_eq!(my_cow_str.as_ref().sz_find_byte_not_from("world"), Some(0));
        assert_eq!(my_cow_str.as_ref().sz_rfind_byte_not_from("world"), Some(12));
    }

    #[test]
    fn fill_random() {
        let mut first_buffer: Vec<u8> = vec![0; 10]; // Ten zeros
        let mut second_buffer: Vec<u8> = vec![1; 10]; // Ten ones
        sz::fill_random(&mut first_buffer, 42);
        sz::fill_random(&mut second_buffer, 42);

        // Same nonce will produce the same outputs
        assert_eq!(first_buffer, second_buffer);
    }

    #[test]
    fn iter_matches_forward() {
        let haystack = b"hello world hello universe";
        let needle = b"hello";
        let matches: Vec<_> = haystack.sz_matches(needle).collect();
        assert_eq!(matches, vec![b"hello", b"hello"]);
    }

    #[test]
    fn iter_matches_reverse() {
        let haystack = b"hello world hello universe";
        let needle = b"hello";
        let matches: Vec<_> = haystack.sz_rmatches(needle).collect();
        assert_eq!(matches, vec![b"hello", b"hello"]);
    }

    #[test]
    fn iter_splits_forward() {
        let haystack = b"alpha,beta;gamma";
        let needle = b",";
        let splits: Vec<_> = haystack.sz_splits(needle).collect();
        assert_eq!(splits, vec![&b"alpha"[..], &b"beta;gamma"[..]]);
    }

    #[test]
    fn iter_splits_reverse() {
        let haystack = b"alpha,beta;gamma";
        let needle = b";";
        let splits: Vec<_> = haystack.sz_rsplits(needle).collect();
        assert_eq!(splits, vec![&b"gamma"[..], &b"alpha,beta"[..]]);
    }

    #[test]
    fn iter_splits_with_empty_parts() {
        let haystack = b"a,,b,";
        let needle = b",";
        let splits: Vec<_> = haystack.sz_splits(needle).collect();
        assert_eq!(splits, vec![b"a", &b""[..], b"b", &b""[..]]);
    }

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

    #[test]
    fn iter_splits_forward_skip_empty() {
        // Default KEEP yields empties; skip_empty drops every zero-length segment.
        let haystack = b"a,,b,";
        let needle = b",";
        let kept: Vec<_> = haystack.sz_splits(needle).collect();
        assert_eq!(kept, vec![b"a", &b""[..], b"b", &b""[..]]);
        let nonempty: Vec<_> = haystack.sz_splits(needle).skip_empty().collect();
        assert_eq!(nonempty, vec![b"a", b"b"]);
    }

    #[test]
    fn iter_splits_reverse_skip_empty() {
        // KEEP rsplit of "a,,b," is the reverse of the forward split, empties included.
        let haystack = b"a,,b,";
        let needle = b",";
        let kept: Vec<_> = haystack.sz_rsplits(needle).collect();
        assert_eq!(kept, vec![&b""[..], b"b", &b""[..], b"a"]);
        let nonempty: Vec<_> = haystack.sz_rsplits(needle).skip_empty().collect();
        assert_eq!(nonempty, vec![b"b", b"a"]);
    }

    #[test]
    fn iter_splits_byteset_skip_empty() {
        // Byteset matcher (split on any of ",;"): adjacent delimiters yield empties under the KEEP default.
        let haystack = b",a;;b,";
        let kept: Vec<_> = FindSplits::new(haystack, MatcherType::FindFirstOf(b",;")).collect();
        assert_eq!(kept, vec![&b""[..], b"a", &b""[..], b"b", &b""[..]]);
        let nonempty: Vec<_> = FindSplits::new(haystack, MatcherType::FindFirstOf(b",;"))
            .skip_empty()
            .collect();
        assert_eq!(nonempty, vec![b"a", b"b"]);
    }

    #[test]
    fn iter_matches_with_overlaps() {
        let haystack = b"aaaa";
        let needle = b"aa";
        let matches: Vec<_> = haystack.sz_matches(needle).collect();
        assert_eq!(matches, vec![b"aa", b"aa", b"aa"]);
    }

    #[test]
    fn iter_splits_with_utf8_haystack() {
        let haystack = "こんにちは,世界".as_bytes();
        let needle = b",";
        let splits: Vec<_> = haystack.sz_splits(needle).collect();
        assert_eq!(splits, vec!["こんにちは".as_bytes(), "世界".as_bytes()]);
    }

    #[test]
    fn iter_find_first_of() {
        let haystack = b"hello world";
        let needles = b"or";
        let matches: Vec<_> = haystack.sz_find_first_of(needles).collect();
        assert_eq!(matches, vec![b"o", b"o", b"r"]);
    }

    #[test]
    fn iter_find_last_of() {
        let haystack = b"hello world";
        let needles = b"or";
        let matches: Vec<_> = haystack.sz_find_last_of(needles).collect();
        assert_eq!(matches, vec![b"r", b"o", b"o"]);
    }

    #[test]
    fn iter_find_first_not_of() {
        let haystack = b"aabbbcccd";
        let needles = b"ab";
        let matches: Vec<_> = haystack.sz_find_first_not_of(needles).collect();
        assert_eq!(matches, vec![b"c", b"c", b"c", b"d"]);
    }

    #[test]
    fn iter_find_last_not_of() {
        let haystack = b"aabbbcccd";
        let needles = b"cd";
        let matches: Vec<_> = haystack.sz_find_last_not_of(needles).collect();
        assert_eq!(matches, vec![b"b", b"b", b"b", b"a", b"a"]);
    }

    #[test]
    fn iter_find_first_of_empty_needles() {
        let haystack = b"hello world";
        let needles = b"";
        let matches: Vec<_> = haystack.sz_find_first_of(needles).collect();
        assert_eq!(matches, Vec::<&[u8]>::new());
    }

    #[test]
    fn iter_find_last_of_empty_haystack() {
        let haystack = b"";
        let needles = b"abc";
        let matches: Vec<_> = haystack.sz_find_last_of(needles).collect();
        assert_eq!(matches, Vec::<&[u8]>::new());
    }

    #[test]
    fn iter_find_first_not_of_all_matching() {
        let haystack = b"aaabbbccc";
        let needles = b"abc";
        let matches: Vec<_> = haystack.sz_find_first_not_of(needles).collect();
        assert_eq!(matches, Vec::<&[u8]>::new());
    }

    #[test]
    fn iter_find_last_not_of_all_not_matching() {
        let haystack = b"hello world";
        let needles = b"xyz";
        let matches: Vec<_> = haystack.sz_find_last_not_of(needles).collect();
        assert_eq!(
            matches,
            vec![b"d", b"l", b"r", b"o", b"w", b" ", b"o", b"l", b"l", b"e", b"h"]
        );
    }

    #[test]
    fn iter_find_matches_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::Find(b"aa");
        let matches: Vec<_> = FindMatches::new(haystack, matcher, true).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_find_matches_non_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::Find(b"aa");
        let matches: Vec<_> = FindMatches::new(haystack, matcher, false).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_rfind_matches_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::RFind(b"aa");
        let matches: Vec<_> = RFindMatches::new(haystack, matcher, true).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_rfind_matches_non_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::RFind(b"aa");
        let matches: Vec<_> = RFindMatches::new(haystack, matcher, false).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn argsort_default() {
        // Test with a slice of string literals.
        let fruits = ["banana", "apple", "cherry"];
        let mut order = [0; 3]; // output buffer must be at least fruits.len()
        sz::argsort(&fruits, &mut order, Default::default()).expect("argsort failed");

        // Reconstruct sorted order using the returned indices.
        let sorted_from_api: Vec<_> = order.iter().map(|&i| fruits[i]).collect();

        // Compute expected order using the standard sort.
        let mut expected = fruits.to_vec();
        expected.sort();

        assert_eq!(sorted_from_api, expected);
    }

    #[test]
    fn argsort_by_custom() {
        // Define a custom type.
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Person {
            name: &'static str,
            age: u32, //? We won't use this field for intersection
        }

        let people = [
            Person {
                name: "Charlie",
                age: 30,
            },
            Person { name: "Alice", age: 25 },
            Person { name: "Bob", age: 40 },
        ];
        let mut order = [0; 3];
        sz::argsort_by(|i: usize| people[i].name.as_bytes(), &mut order, Default::default())
            .expect("argsort_by failed");

        let sorted_from_api: Vec<_> = order.iter().map(|&i| people[i].name).collect();

        // Compute expected order using standard sorting on the names.
        let mut expected: Vec<_> = people.iter().map(|p| p.name).collect();
        expected.sort();

        assert_eq!(sorted_from_api, expected);
    }

    #[test]
    fn argsort_reverse_is_stable() {
        // Two equal "beta"s must keep their input order even when sorting descending.
        let labels = ["beta", "alpha", "beta", "gamma"];
        let mut order = [0; 4];
        sz::argsort(&labels, &mut order, sz::ArgsortOptions::default().reversed()).expect("argsort failed");
        let sorted: Vec<_> = order.iter().map(|&i| labels[i]).collect();
        assert_eq!(sorted, vec!["gamma", "beta", "beta", "alpha"]);
        // Stability: the first "beta" (index 0) precedes the second (index 2).
        let beta_positions: Vec<_> = order.iter().filter(|&&i| labels[i] == "beta").copied().collect();
        assert_eq!(beta_positions, vec![0, 2]);
    }

    #[test]
    fn argsort_top_k_prefix() {
        let words = ["delta", "alpha", "echo", "bravo", "charlie"];
        let mut order = [0; 5];
        sz::argsort(&words, &mut order, sz::ArgsortOptions::default().top(2)).expect("argsort failed");
        // Only the first two entries are guaranteed sorted (the two smallest).
        assert_eq!(words[order[0]], "alpha");
        assert_eq!(words[order[1]], "bravo");
        // `order` is still a full permutation.
        let mut seen = order.to_vec();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn argsort_uncased() {
        let labels = ["Banana", "apple", "BANANA", "Apple"];
        let mut order = [0; 4];
        sz::argsort(&labels, &mut order, sz::ArgsortOptions::default().uncased()).expect("argsort failed");
        let sorted: Vec<_> = order.iter().map(|&i| labels[i]).collect();
        // Fold-equal strings group together and stay in input order: "apple","Apple" then "Banana","BANANA".
        assert_eq!(sorted, vec!["apple", "Apple", "Banana", "BANANA"]);
    }

    #[test]
    fn intersection_default() {
        // Two slices of string literals.
        let set1 = ["banana", "apple", "cherry"];
        let set2 = ["cherry", "orange", "pineapple", "banana"];
        // Output buffers: size must be at least min(set1.len(), set2.len()).
        let mut out1 = [0; 3];
        let mut out2 = [0; 3];

        let n = sz::intersection(&set1, &set2, 0, &mut out1, &mut out2).expect("intersection failed");
        assert!(n <= set1.len().min(set2.len()));

        // For simplicity, we will compare the intersection from the first set.
        // Our API returns indices (for set1 in out1).
        let common_from_api: HashSet<_> = out1[..n].iter().map(|&i| set1[i]).collect();

        // Compute the expected intersection using a `HashSet`.
        let expected: HashSet<_> = set1
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .intersection(&set2.iter().cloned().collect())
            .cloned()
            .collect();

        assert_eq!(common_from_api, expected);
    }

    #[test]
    fn intersection_by_custom() {
        // Define a custom type.
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Person {
            name: &'static str,
            age: u32, //? We won't use this field for intersection
        }

        let group1 = [
            Person { name: "Alice", age: 25 },
            Person { name: "Bob", age: 30 },
            Person {
                name: "Charlie",
                age: 35,
            },
        ];
        let group2 = [
            Person { name: "David", age: 40 },
            Person {
                name: "Charlie",
                age: 50,
            },
            Person { name: "Alice", age: 60 },
        ];
        let mut out1 = [0; 3];
        let mut out2 = [0; 3];

        let n = sz::intersection_by(
            |i: sz::SortedIdx| group1[i].name.as_bytes(),
            |j: sz::SortedIdx| group2[j].name.as_bytes(),
            0,
            &mut out1,
            &mut out2,
        )
        .expect("intersection_by failed");
        assert!(n <= group1.len().min(group2.len()));

        // Use the indices for `group1` to get common names.
        let common_from_api: HashSet<_> = out1[..n].iter().map(|&i| group1[i].name).collect();

        // Compute expected common names using a `HashSet`.
        let expected: HashSet<_> = group1
            .iter()
            .map(|p| p.name)
            .collect::<HashSet<_>>()
            .intersection(&group2.iter().map(|p| p.name).collect())
            .cloned()
            .collect();

        assert_eq!(common_from_api, expected);
    }

    #[test]
    #[should_panic]
    fn intersection_size_checks() {
        let mut indices = [0usize; 10];
        let mut indices2 = [0usize; 5];
        let data = vec![0x41u8; 12];

        intersection_by(|_: usize| &data, |_: usize| &data, 1, &mut indices, &mut indices2).unwrap();
    }

    #[test]
    fn intersection_debug() {
        println!("Starting intersection debug test...");

        let set1 = ["banana", "apple", "cherry"];
        let set2 = ["cherry", "orange", "pineapple", "banana"];
        let mut positions1 = [0; 3];
        let mut positions2 = [0; 3];

        println!("About to call intersection function...");
        let n = intersection(&set1, &set2, 0, &mut positions1, &mut positions2).expect("intersect failed");

        println!("Intersection found {} common elements", n);
        assert!(n == 2);
        println!("Test passed!");
    }

    #[test]
    fn sha256_empty() {
        let hash = sz::Sha256::hash(b"");
        let expected = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24, 0x27, 0xae,
            0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(hash, expected);
    }

    #[test]
    fn sha256_abc() {
        let hash = sz::Sha256::hash(b"abc");
        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03,
            0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(hash, expected);
    }

    #[test]
    fn sha256_incremental() {
        let mut hasher = sz::Sha256::new();
        hasher.update(b"ab");
        hasher.update(b"c");
        let hash = hasher.digest();
        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03,
            0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(hash, expected);
    }

    #[test]
    fn sha256_long() {
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let hash = sz::Sha256::hash(msg);
        let expected = [
            0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8, 0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e, 0x60, 0x39, 0xa3, 0x3c,
            0xe4, 0x59, 0x64, 0xff, 0x21, 0x67, 0xf6, 0xec, 0xed, 0xd4, 0x19, 0xdb, 0x06, 0xc1,
        ];
        assert_eq!(hash, expected);
    }

    #[test]
    fn hmac_sha256_basic() {
        // Test vector from RFC 4231 (HMAC-SHA256 test case 1)
        let key = b"";
        let message = b"";
        let mac = sz::hmac_sha256(key, message);
        // HMAC-SHA256("", "") = b613...
        let expected = [
            0xb6, 0x13, 0x67, 0x9a, 0x08, 0x14, 0xd9, 0xec, 0x77, 0x2f, 0x95, 0xd7, 0x78, 0xc3, 0x5f, 0xc5, 0xff, 0x16,
            0x97, 0xc4, 0x93, 0x71, 0x56, 0x53, 0xc6, 0xc7, 0x12, 0x14, 0x42, 0x92, 0xc5, 0xad,
        ];
        assert_eq!(mac, expected);
    }

    #[test]
    fn hmac_sha256_short_key() {
        // Test with short key and message
        let key = b"key";
        let message = b"The quick brown fox jumps over the lazy dog";
        let mac = sz::hmac_sha256(key, message);
        // HMAC-SHA256("key", "The quick brown fox jumps over the lazy dog")
        let expected = [
            0xf7, 0xbc, 0x83, 0xf4, 0x30, 0x53, 0x84, 0x24, 0xb1, 0x32, 0x98, 0xe6, 0xaa, 0x6f, 0xb1, 0x43, 0xef, 0x4d,
            0x59, 0xa1, 0x49, 0x46, 0x17, 0x59, 0x97, 0x47, 0x9d, 0xbc, 0x2d, 0x1a, 0x3c, 0xd8,
        ];
        assert_eq!(mac, expected);
    }

    #[test]
    fn hmac_sha256_long_key() {
        // Test with key longer than block size (> 64 bytes)
        let key = b"this is a very long key that exceeds the SHA256 block size of 64 bytes for testing purposes";
        let message = b"message";
        let mac = sz::hmac_sha256(key, message);
        // Expected value computed with Python: hmac.new(key, message, hashlib.sha256).digest()
        let expected = [
            0xd1, 0x3f, 0xdb, 0x7b, 0xe0, 0x9a, 0x9e, 0x07, 0x04, 0xc6, 0x5b, 0xd7, 0x85, 0xa6, 0x33, 0xbb, 0xc0, 0xee,
            0x2b, 0x99, 0xef, 0xd6, 0x32, 0x2c, 0xa9, 0x4c, 0xd3, 0x2c, 0x1e, 0x45, 0x09, 0xfd,
        ];
        assert_eq!(mac, expected);
    }

    #[test]
    #[should_panic]
    fn copy_size_checks() {
        let long: Vec<u8> = vec![0; 20];
        let mut less_long: Vec<u8> = vec![0; 10];

        copy(&mut less_long, &long);
    }

    #[test]
    #[should_panic]
    fn move_size_checks() {
        let long: Vec<u8> = vec![0; 20];
        let mut less_long: Vec<u8> = vec![0; 10];

        move_(&mut less_long, &long);
    }

    #[test]
    #[should_panic]
    fn lookup_size_checks() {
        let long: Vec<u8> = vec![0; 20];
        let mut less_long: Vec<u8> = vec![0; 10];

        let lut: [u8; 256] = (0..=255u8).collect::<Vec<_>>().try_into().unwrap();
        lookup(&mut less_long, &long, lut);
    }

    #[test]
    fn replace_all_same_length() {
        let mut buffer = b"abcabc".to_vec();
        let replaced = sz::try_replace_all(&mut buffer, b"ab", b"XY").unwrap();
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"XYcXYc");
    }

    #[test]
    fn replace_all_shrinks() {
        let mut buffer = b"aaaa".to_vec();
        let replaced = sz::try_replace_all(&mut buffer, b"aa", b"b").unwrap();
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"bb");
    }

    #[test]
    fn replace_all_grows() {
        let mut buffer = b"aba".to_vec();
        let replaced = sz::try_replace_all(&mut buffer, b"a", b"XYZ").unwrap();
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"XYZbXYZ");
    }

    #[test]
    fn replace_all_byteset_basic() {
        let mut buffer = b"hello world".to_vec();
        let vowels = sz::Byteset::from("aeiou");
        let replaced = sz::try_replace_all_byteset(&mut buffer, vowels, b"_").unwrap();
        assert_eq!(replaced, 3);
        assert_eq!(buffer, b"h_ll_ w_rld");
    }

    #[test]
    fn replace_all_byteset_grows() {
        let mut buffer = b"yzz".to_vec();
        let vowels = sz::Byteset::from("y");
        let replaced = sz::try_replace_all_byteset(&mut buffer, vowels, b"(y)").unwrap();
        assert_eq!(replaced, 1);
        assert_eq!(buffer, b"(y)zz");
    }

    #[test]
    fn replace_all_noop_on_empty_pattern() {
        let mut buffer = b"unchanged".to_vec();
        let replaced = sz::try_replace_all(&mut buffer, b"", b"anything").unwrap();
        assert_eq!(replaced, 0);
        assert_eq!(buffer, b"unchanged");
    }

    #[test]
    fn iter_newline_utf8_splits() {
        let text = b"a\nb\r\nc\n\nd";
        let lines: Vec<_> = Utf8Lines::new(text).collect();
        assert_eq!(lines, vec![b"a", b"b", b"c", &b""[..], b"d"]);
    }

    #[test]
    fn iter_newline_utf8_splits_unicode() {
        let text = "Hello\u{2028}World".as_bytes(); // LINE SEPARATOR
        let lines: Vec<_> = Utf8Lines::new(text).collect();
        assert_eq!(lines, vec!["Hello".as_bytes(), "World".as_bytes()]);
    }

    #[test]
    fn iter_whitespace_utf8_splits() {
        // KEEP (default): every one of the 8 whitespace delimiters yields a segment, so leading,
        // trailing, and inner runs all surface empties (str::split semantics, matching C++/Python).
        let text = b"  a \t b\n\nc  ";
        let segments: Vec<_> = Utf8Tokens::new(text).collect();
        assert_eq!(
            segments,
            vec![
                &b""[..],
                &b""[..],
                b"a",
                &b""[..],
                &b""[..],
                b"b",
                &b""[..],
                b"c",
                &b""[..],
                &b""[..],
            ]
        );
        // skip_empty: recovers the str::split_whitespace token behavior.
        let tokens: Vec<_> = Utf8Tokens::new(text).skip_empty().collect();
        assert_eq!(tokens, vec![b"a", b"b", b"c"]);
    }

    #[test]
    fn iter_whitespace_utf8_splits_keep_default() {
        // The simple example from the doc comment: KEEP yields the surrounding empties, skip_empty drops them.
        let text = b"  hi  ";
        let kept: Vec<_> = Utf8Tokens::new(text).collect();
        assert_eq!(kept, vec![&b""[..], &b""[..], b"hi", &b""[..], &b""[..]]);
        let tokens: Vec<_> = Utf8Tokens::new(text).skip_empty().collect();
        assert_eq!(tokens, vec![b"hi"]);
    }

    #[test]
    fn iter_whitespace_utf8_splits_unicode() {
        let text = "a\u{3000}b\u{2000}c".as_bytes(); // IDEOGRAPHIC SPACE, EN QUAD
        let segments: Vec<_> = Utf8Tokens::new(text).collect();
        assert_eq!(segments, vec![b"a", b"b", b"c"]); // single delimiters between words: no empties
        let tokens: Vec<_> = Utf8Tokens::new(text).skip_empty().collect();
        assert_eq!(tokens, vec![b"a", b"b", b"c"]);
    }

    #[test]
    fn iter_whitespace_utf8_splits_skip_empty_all_whitespace() {
        let text = b"   \t  ";
        let kept: Vec<_> = Utf8Tokens::new(text).collect();
        assert_eq!(kept.len(), 7); // 6 delimiters → 7 (all empty) segments
        assert!(kept.iter().all(|segment| segment.is_empty()));
        let tokens: Vec<&[u8]> = Utf8Tokens::new(text).skip_empty().collect();
        assert!(tokens.is_empty());
    }

    #[test]
    fn iter_newline_utf8_splits_skip_empty() {
        let text = b"a\nb\r\nc\n\nd";
        // Default KEEP: the back-to-back "\n\n" yields an empty line.
        let kept: Vec<_> = Utf8Lines::new(text).collect();
        assert_eq!(kept, vec![b"a", b"b", b"c", &b""[..], b"d"]);
        // skip_empty: the empty line between "c" and "d" disappears.
        let nonempty: Vec<_> = Utf8Lines::new(text).skip_empty().collect();
        assert_eq!(nonempty, vec![b"a", b"b", b"c", b"d"]);
    }

    #[test]
    fn iter_newline_utf8_splits_steps_invariance() {
        // The yielded segments must be identical regardless of the batch size `STEPS`; a tiny batch
        // (STEPS == 1) exercises the refill/trailing-segment seam on every delimiter, while large
        // batches fit the whole input in one call.
        let text = b"\r\na\r\n\r\nb\r\nc\nd\n";
        let expected: Vec<&[u8]> = vec![b"", b"a", b"", b"b", b"c", b"d", b""];
        let from_1: Vec<_> = Utf8Lines::<1>::with_steps(text).collect();
        let from_3: Vec<_> = Utf8Lines::<3>::with_steps(text).collect();
        let from_65: Vec<_> = Utf8Lines::<65>::with_steps(text).collect();
        assert_eq!(from_1, expected);
        assert_eq!(from_3, expected);
        assert_eq!(from_65, expected);

        // skip_empty across the same batch sizes.
        let nonempty: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        assert_eq!(
            Utf8Lines::<1>::with_steps(text).skip_empty().collect::<Vec<_>>(),
            nonempty
        );
        assert_eq!(
            Utf8Lines::<3>::with_steps(text).skip_empty().collect::<Vec<_>>(),
            nonempty
        );
        assert_eq!(
            Utf8Lines::<65>::with_steps(text).skip_empty().collect::<Vec<_>>(),
            nonempty
        );
    }

    #[test]
    fn iter_whitespace_utf8_splits_steps_invariance() {
        let text = b"  a \t b\n\nc  ";
        let expected: Vec<&[u8]> = vec![b"", b"", b"a", b"", b"", b"b", b"", b"c", b"", b""];
        assert_eq!(Utf8Tokens::<1>::with_steps(text).collect::<Vec<_>>(), expected);
        assert_eq!(Utf8Tokens::<3>::with_steps(text).collect::<Vec<_>>(), expected);
        assert_eq!(Utf8Tokens::<65>::with_steps(text).collect::<Vec<_>>(), expected);
        let tokens: Vec<&[u8]> = vec![b"a", b"b", b"c"];
        assert_eq!(
            Utf8Tokens::<1>::with_steps(text).skip_empty().collect::<Vec<_>>(),
            tokens
        );
    }

    #[test]
    fn iter_newline_utf8_splits_trailing_newline() {
        // "\r\na\r\n\r\nb\r\n" should produce ["", "a", "", "b", ""]
        let text = b"\r\na\r\n\r\nb\r\n";
        let lines: Vec<&[u8]> = Utf8Lines::new(text).collect();
        assert_eq!(lines.len(), 5, "Expected 5 lines");
        let expected: Vec<&[u8]> = vec![b"", b"a", b"", b"b", b""];
        assert_eq!(lines, expected);
    }

    #[test]
    fn iter_newline_utf8_splits_no_trailing() {
        let text = b"a\nb\nc";
        let lines: Vec<&[u8]> = Utf8Lines::new(text).collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines, vec![b"a", b"b", b"c"]);
    }

    #[test]
    fn iter_newline_utf8_splits_empty_string() {
        let text = b"";
        let lines: Vec<&[u8]> = Utf8Lines::new(text).collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines, vec![b""]);
    }

    #[test]
    fn iter_newline_utf8_splits_single_newline() {
        let text = b"\n";
        let lines: Vec<&[u8]> = Utf8Lines::new(text).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines, vec![b"", b""]);
    }

    #[test]
    fn iter_word_utf8_splits_steps_invariance() {
        // Words tile the input, so the yielded segments must match regardless of the batch size `STEPS`;
        // a tiny batch (STEPS == 1) exercises the refill seam on every word boundary.
        let text = b"Hi, world! A second sentence.";
        let forward: Vec<&[u8]> = Utf8Words::new(text).collect();
        assert_eq!(Utf8Words::<1>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Words::<3>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Words::<65>::with_steps(text).collect::<Vec<_>>(), forward);
    }

    #[test]
    fn iter_grapheme_utf8_splits_steps_invariance() {
        // Grapheme clusters tile the input, so the yielded segments must match regardless of the batch size
        // `STEPS`; a tiny batch (STEPS == 1) exercises the refill seam on every cluster boundary.
        let text = b"Hi, world! A second sentence.";
        let forward: Vec<&[u8]> = Utf8Graphemes::new(text).collect();
        assert_eq!(Utf8Graphemes::<1>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Graphemes::<3>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Graphemes::<65>::with_steps(text).collect::<Vec<_>>(), forward);
    }

    #[test]
    fn iter_sentence_utf8_splits_steps_invariance() {
        // Sentences tile the input, so the yielded segments must match regardless of the batch size `STEPS`;
        // a tiny batch (STEPS == 1) exercises the refill seam on every sentence boundary.
        let text = b"Hi, world! A second sentence.";
        let forward: Vec<&[u8]> = Utf8Sentences::new(text).collect();
        assert_eq!(Utf8Sentences::<1>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Sentences::<3>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Sentences::<65>::with_steps(text).collect::<Vec<_>>(), forward);
    }

    #[test]
    fn iter_linewrap_utf8_splits_steps_invariance() {
        // Linewrap segments tile the input, so the yielded segments must match regardless of
        // the batch size `STEPS`; a tiny batch (STEPS == 1) exercises the refill seam on every line-break opportunity.
        let text = b"Hi, world! A second sentence.";
        let forward: Vec<&[u8]> = Utf8Linebreaks::new(text).collect();
        assert_eq!(Utf8Linebreaks::<1>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Linebreaks::<3>::with_steps(text).collect::<Vec<_>>(), forward);
        assert_eq!(Utf8Linebreaks::<65>::with_steps(text).collect::<Vec<_>>(), forward);
    }

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
    fn utf8_norm_golden_vectors() {
        use sz::Utf8NormalForm;

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

        // "café" with precomposed é (U+00E9) is already NFC.
        // NFC → NFC is a no-op (same bytes out).
        let cafe_nfc = "caf\u{00E9}"; // 5 bytes: c a f 0xC3 0xA9
        {
            let mut dest = vec![0u8; cafe_nfc.len() * 18];
            let len = sz::utf8_norm(cafe_nfc, Utf8NormalForm::Nfc, &mut dest);
            assert_eq!(&dest[..len], cafe_nfc.as_bytes(), "café NFC→NFC unchanged");
        }

        // "café" with decomposed é = base 'e' + combining acute U+0301 is NFD.
        // NFD → NFC must produce the precomposed form.
        let cafe_nfd = "cafe\u{0301}"; // 6 bytes: c a f e 0xCC 0x81
        {
            let mut dest = vec![0u8; cafe_nfd.len() * 18];
            let len = sz::utf8_norm(cafe_nfd, Utf8NormalForm::Nfc, &mut dest);
            assert_eq!(&dest[..len], cafe_nfc.as_bytes(), "café NFD→NFC gives precomposed form");
        }

        // NFD of the precomposed form must give the decomposed form.
        {
            let mut dest = vec![0u8; cafe_nfc.len() * 18];
            let len = sz::utf8_norm(cafe_nfc, Utf8NormalForm::Nfd, &mut dest);
            assert_eq!(&dest[..len], cafe_nfd.as_bytes(), "café NFC→NFD gives decomposed form");
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
            let source = cafe_nfd;
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
        use sz::Utf8NormalForm;

        // NFC string: precomposed é — no violation.
        let nfc_str = "caf\u{00E9}";
        assert_eq!(
            sz::utf8_find_denormalized(nfc_str, Utf8NormalForm::Nfc),
            None,
            "NFC string has no NFC violation"
        );

        // NFD string: decomposed e + combining acute U+0301.
        // The combining mark violates NFC (it should be composed with the preceding base).
        let nfd_str = "cafe\u{0301}";
        let violation = sz::utf8_find_denormalized(nfd_str, Utf8NormalForm::Nfc);
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
