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
/// use stringzilla::stringzilla::{IndexSpan, find_newline_utf8};
///
/// let text = "Hello\nWorld";
/// if let Some(span) = find_newline_utf8(text) {
///     assert_eq!(span.offset, 5);
///     assert_eq!(span.length, 1);
///     let matched = span.extract(text.as_bytes());
///     assert_eq!(matched, b"\n");
/// }
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
    pub(crate) fn sz_lookup(
        target: *const c_void,
        length: usize,
        source: *const c_void,
        lut: *const u8,
    ) -> *const c_void;

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
    pub(crate) fn sz_utf8_find_nth(text: *const c_void, length: usize, n: usize) -> *const c_void;
    pub(crate) fn sz_utf8_unpack_chunk(
        text: *const c_void,
        length: usize,
        runes: *mut u32,
        runes_capacity: usize,
        runes_unpacked: *mut usize,
    ) -> *const c_void;
    pub(crate) fn sz_utf8_find_newline(text: *const c_void, length: usize, matched_length: *mut usize)
        -> *const c_void;
    pub(crate) fn sz_utf8_find_whitespace(
        text: *const c_void,
        length: usize,
        matched_length: *mut usize,
    ) -> *const c_void;

    pub(crate) fn sz_bytesum(text: *const c_void, length: usize) -> u64;
    pub(crate) fn sz_hash(text: *const c_void, length: usize, seed: u64) -> u64;
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
pub fn try_replace_all_byteset(
    buffer: &mut Vec<u8>,
    byteset: Byteset,
    replacement: &[u8],
) -> Result<usize, Status> {
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

/// Finds the first newline character in UTF-8 encoded text.
///
/// Searches for any of the 8 Unicode newline characters:
/// - U+000A (LF - Line Feed `\n`)
/// - U+000B (VT - Vertical Tab `\v`)
/// - U+000C (FF - Form Feed `\f`)
/// - U+000D (CR - Carriage Return `\r`, handles `\r\n` as single newline)
/// - U+001C (FILE SEPARATOR)
/// - U+001D (GROUP SEPARATOR)
/// - U+001E (RECORD SEPARATOR)
/// - U+0085 (NEL - Next Line)
/// - U+2028 (LINE SEPARATOR)
/// - U+2029 (PARAGRAPH SEPARATOR)
///
/// # Arguments
///
/// * `text`: The UTF-8 encoded byte slice to search.
///
/// # Returns
///
/// An `Option<IndexSpan>` containing the byte offset and length of the matched newline.
/// The length can be 1-3 bytes for single characters, or 2 bytes for CRLF sequence.
///
/// Returns `None` if no newline is found.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// let text = "Hello\nWorld";
/// let span = sz::find_newline_utf8(text).unwrap();
/// assert_eq!(span.offset, 5);
/// assert_eq!(span.length, 1);
///
/// let text_crlf = "Hello\r\nWorld";
/// let span = sz::find_newline_utf8(text_crlf).unwrap();
/// assert_eq!(span.offset, 5);
/// assert_eq!(span.length, 2);
///
/// let text_unicode = "Hello\u{2028}World"; // LINE SEPARATOR
/// let span = sz::find_newline_utf8(text_unicode).unwrap();
/// assert_eq!(span.offset, 5);
/// assert_eq!(span.length, 3);
/// ```
pub fn find_newline_utf8<T>(text: T) -> Option<IndexSpan>
where
    T: AsRef<[u8]>,
{
    let text_ref = text.as_ref();
    let text_pointer = text_ref.as_ptr() as *const c_void;
    let text_length = text_ref.len();
    let mut matched_length: usize = 0;

    let result = unsafe { sz_utf8_find_newline(text_pointer, text_length, &mut matched_length as *mut usize) };

    if result.is_null() {
        None
    } else {
        let offset = unsafe { (result as *const u8).offset_from(text_pointer as *const u8) }
            .try_into()
            .unwrap();
        Some(IndexSpan::new(offset, matched_length))
    }
}

/// Finds the first whitespace character in UTF-8 encoded text.
///
/// Searches for any of the 29 Unicode whitespace characters (includes all newlines
/// per Unicode standard, plus spaces, tabs, and various Unicode space characters).
///
/// The complete set includes:
/// - All 8 newline characters (see [`find_newline_utf8`])
/// - U+0009 (CHARACTER TABULATION `\t`)
/// - U+001F (UNIT SEPARATOR)
/// - U+0020 (SPACE)
/// - U+00A0 (NO-BREAK SPACE)
/// - U+1680 (OGHAM SPACE MARK)
/// - U+2000-U+200A (11 various spaces)
/// - U+202F (NARROW NO-BREAK SPACE)
/// - U+205F (MEDIUM MATHEMATICAL SPACE)
/// - U+3000 (IDEOGRAPHIC SPACE)
///
/// # Arguments
///
/// * `text`: The UTF-8 encoded byte slice to search.
///
/// # Returns
///
/// An `Option<IndexSpan>` containing the byte offset and length of the matched whitespace.
/// The length can be 1-3 bytes depending on the UTF-8 character.
///
/// Returns `None` if no whitespace is found.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// let text = "Hello World";
/// let span = sz::find_whitespace_utf8(text).unwrap();
/// assert_eq!(span.offset, 5);
/// assert_eq!(span.length, 1);
///
/// let text_unicode = "Hello\u{3000}World"; // IDEOGRAPHIC SPACE
/// let span = sz::find_whitespace_utf8(text_unicode).unwrap();
/// assert_eq!(span.offset, 5);
/// assert_eq!(span.length, 3);
///
/// // Whitespace includes newlines
/// let text_newline = "Hello\nWorld";
/// let span = sz::find_whitespace_utf8(text_newline).unwrap();
/// assert_eq!(span.offset, 5);
/// assert_eq!(span.length, 1);
/// ```
pub fn find_whitespace_utf8<T>(text: T) -> Option<IndexSpan>
where
    T: AsRef<[u8]>,
{
    let text_ref = text.as_ref();
    let text_pointer = text_ref.as_ptr() as *const c_void;
    let text_length = text_ref.len();
    let mut matched_length: usize = 0;

    let result = unsafe { sz_utf8_find_whitespace(text_pointer, text_length, &mut matched_length as *mut usize) };

    if result.is_null() {
        None
    } else {
        let offset = unsafe { (result as *const u8).offset_from(text_pointer as *const u8) }
            .try_into()
            .unwrap();
        Some(IndexSpan::new(offset, matched_length))
    }
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

    let result = unsafe { sz_utf8_find_nth(text_pointer, text_length, n) };

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
    pub fn iter(&self) -> Utf8Chars<'a> {
        Utf8Chars::new(self.octets)
    }
}

/// Iterator over UTF-8 characters using batched decoding.
///
/// Decodes up to 64 bytes at a time into UTF-32 codepoints, then yields them one at a time.
/// This is much more efficient than decoding character-by-character.
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
pub struct Utf8Chars<'a> {
    octets: &'a [u8],
    octets_offset: usize,
    runes: [u32; 64],
    runes_count: usize,
    runes_offset: usize,
}

impl<'a> Utf8Chars<'a> {
    fn new(octets: &'a [u8]) -> Self {
        let mut iter = Self {
            octets,
            octets_offset: 0,
            runes: [0; 64],
            runes_count: 0,
            runes_offset: 0,
        };
        // Decode first batch
        iter.decode_batch();
        iter
    }

    /// Decodes the next batch of UTF-8 bytes into the runes buffer.
    fn decode_batch(&mut self) {
        if self.octets_offset >= self.octets.len() {
            self.runes_count = 0;
            return;
        }

        let remaining = self.octets.len() - self.octets_offset;
        let chunk_size = remaining.min(64);
        let octets_ptr = unsafe { self.octets.as_ptr().add(self.octets_offset) as *const c_void };

        let mut unpacked_count: usize = 0;

        let next_ptr = unsafe {
            sz_utf8_unpack_chunk(
                octets_ptr,
                chunk_size,
                self.runes.as_mut_ptr(),
                64, // Capacity of runes buffer
                &mut unpacked_count as *mut usize,
            )
        };

        // Update position
        let bytes_consumed: usize = unsafe {
            let offset = (next_ptr as *const u8).offset_from(octets_ptr as *const u8);
            debug_assert!(offset >= 0, "sz_utf8_unpack_chunk returned a pointer before the input");
            offset.try_into().expect("offset should be non-negative")
        };
        self.octets_offset += bytes_consumed;
        self.runes_count = unpacked_count;
        self.runes_offset = 0;
    }
}

impl<'a> Iterator for Utf8Chars<'a> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        // If runes buffer is exhausted, decode next batch
        if self.runes_offset >= self.runes_count {
            self.decode_batch();
            if self.runes_count == 0 {
                return None;
            }
        }

        let codepoint = self.runes[self.runes_offset];
        self.runes_offset += 1;
        char::from_u32(codepoint)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Lower bound: remaining runes in current buffer
        let lower = self.runes_count.saturating_sub(self.runes_offset);
        // Upper bound: unknown without counting entire string
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

/// Sorts a sequence of items by comparing their byte-slice representations.
///
/// The caller must supply an output buffer `order` whose length is at least
/// equal to the length of `data`. On success, the function writes the sorted
/// permutation indices into `order`.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// let fruits = ["banana", "apple", "cherry"];
/// let mut order = [0; 3];
/// sz::argsort_permutation(&fruits, &mut order).expect("sort failed");
/// assert_eq!(&order, &[1, 0, 2]); // "apple", "banana", "cherry"
/// ```
pub fn argsort_permutation<T: AsRef<[u8]>>(data: &[T], order: &mut [SortedIdx]) -> Result<(), Status> {
    if data.len() > order.len() {
        return Err(Status::BadAlloc);
    }
    argsort_permutation_by(|i| data[i].as_ref(), order[..data.len()].as_mut())
}

/// Sorts a sequence of items by comparing their corresponding byte-slice representations.
/// The size of the permutation is inferred from the length of the `order` slice.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// #[derive(Debug)]
/// struct Person { name: &'static str, age: u32 }
///
/// let people = [
///     Person { name: "Charlie", age: 20 },
///     Person { name: "Alice", age: 25 },
///     Person { name: "Bob", age: 30 },
/// ];
/// let mut order = [0; 3];
/// sz::argsort_permutation_by(|i| people[i].name.as_bytes(), &mut order).expect("sort failed");
/// assert_eq!(&order, &[1, 2, 0]); // "Alice", "Bob", "Charlie"
/// ```
pub fn argsort_permutation_by<F, A>(mapper: F, order: &mut [SortedIdx]) -> Result<(), Status>
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

    _argsort_permutation_impl(adapter, order)
}

/// Helper that takes an adapter (with a concrete type) and performs the FFI call.
fn _argsort_permutation_impl<FAdapter>(adapter: FAdapter, order: &mut [SortedIdx]) -> Result<(), Status>
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
    let status = unsafe { sz_sequence_argsort(&seq, core::ptr::null(), order.as_mut_ptr()) };
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
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, RangeMatches}};
///
/// let haystack = b"abababa";
/// let matcher = MatcherType::Find(b"aba");
/// let matches: Vec<&[u8]> = RangeMatches::new(haystack, matcher, false).collect();
/// assert_eq!(matches, vec![b"aba", b"aba"]);
/// ```
pub struct RangeMatches<'a> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
    include_overlaps: bool,
}

impl<'a> RangeMatches<'a> {
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>, include_overlaps: bool) -> Self {
        Self {
            haystack,
            matcher,
            position: 0,
            include_overlaps,
        }
    }
}

impl<'a> Iterator for RangeMatches<'a> {
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
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, RangeSplits}};
///
/// let haystack = b"a,b,c,d";
/// let matcher = MatcherType::Find(b",");
/// let splits: Vec<&[u8]> = RangeSplits::new(haystack, matcher).collect();
/// assert_eq!(splits, vec![b"a", b"b", b"c", b"d"]);
/// ```
pub struct RangeSplits<'a> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
    last_match: Option<usize>,
}

impl<'a> RangeSplits<'a> {
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self {
            haystack,
            matcher,
            position: 0,
            last_match: None,
        }
    }
}

impl<'a> Iterator for RangeSplits<'a> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
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

/// An iterator over non-overlapping matches of a pattern in a string slice, searching from the end.
/// This iterator yields the matched substrings in reverse order.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, RangeRMatches}};
///
/// let haystack = b"abababa";
/// let matcher = MatcherType::RFind(b"aba");
/// let matches: Vec<&[u8]> = RangeRMatches::new(haystack, matcher, false).collect();
/// assert_eq!(matches, vec![b"aba", b"aba"]);
/// ```
pub struct RangeRMatches<'a> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
    include_overlaps: bool,
}

impl<'a> RangeRMatches<'a> {
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>, include_overlaps: bool) -> Self {
        Self {
            haystack,
            matcher,
            position: haystack.len(),
            include_overlaps,
        }
    }
}

impl<'a> Iterator for RangeRMatches<'a> {
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
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, RangeRSplits}};
///
/// let haystack = b"a,b,c,d";
/// let matcher = MatcherType::RFind(b",");
/// let splits: Vec<&[u8]> = RangeRSplits::new(haystack, matcher).collect();
/// assert_eq!(splits, vec![b"d", b"c", b"b", b"a"]);
/// ```
pub struct RangeRSplits<'a> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
}

impl<'a> RangeRSplits<'a> {
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self {
            haystack,
            matcher,
            position: haystack.len(),
        }
    }
}

impl<'a> Iterator for RangeRSplits<'a> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position == 0 {
            return None;
        }

        let search_area = &self.haystack[..self.position];
        if let Some(index) = self.matcher.find(search_area) {
            let end = self.position;
            let start = index + self.matcher.needle_length();
            let result = Some(&self.haystack[start..end]);

            self.position = index;

            result
        } else {
            let result = Some(&self.haystack[..self.position]);
            self.position = 0;
            result
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
/// use stringzilla::stringzilla::{RangeNewlineUtf8Splits};
///
/// let text = b"Hello\nWorld\r\nRust";
/// let lines: Vec<&[u8]> = RangeNewlineUtf8Splits::new(text).collect();
/// assert_eq!(lines, vec![&b"Hello"[..], &b"World"[..], &b"Rust"[..]]);
/// ```
pub struct RangeNewlineUtf8Splits<'a> {
    text: &'a [u8],
    position: usize,
    finished: bool,
}

impl<'a> RangeNewlineUtf8Splits<'a> {
    pub fn new(text: &'a [u8]) -> Self {
        Self {
            text,
            position: 0,
            finished: false,
        }
    }
}

impl<'a> Iterator for RangeNewlineUtf8Splits<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if self.position >= self.text.len() {
            if self.position == self.text.len() && !self.finished {
                // Return empty slice for trailing newline case
                self.finished = true;
                return Some(&self.text[self.text.len()..]);
            }
            return None;
        }

        let start = self.position;

        // Search for next newline
        if let Some(span) = find_newline_utf8(&self.text[self.position..]) {
            let end = self.position + span.offset;
            self.position = end + span.length;
            Some(&self.text[start..end])
        } else {
            // No more newlines, return rest of text
            self.finished = true;
            self.position = self.text.len();
            Some(&self.text[start..])
        }
    }
}

/// An iterator over words in UTF-8 text split by whitespace characters.
///
/// This iterator yields non-empty slices between whitespace characters. The whitespace
/// characters themselves are not included. Handles all 29 Unicode whitespace characters.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{RangeWhitespaceUtf8Splits};
///
/// let text = b"Hello  World\tRust";
/// let words: Vec<&[u8]> = RangeWhitespaceUtf8Splits::new(text).collect();
/// assert_eq!(words, vec![&b"Hello"[..], &b"World"[..], &b"Rust"[..]]);
/// ```
pub struct RangeWhitespaceUtf8Splits<'a> {
    text: &'a [u8],
    position: usize,
}

impl<'a> RangeWhitespaceUtf8Splits<'a> {
    pub fn new(text: &'a [u8]) -> Self {
        Self { text, position: 0 }
    }
}

impl<'a> Iterator for RangeWhitespaceUtf8Splits<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.text.len() {
            return None;
        }

        // Skip leading whitespace
        while self.position < self.text.len() {
            if let Some(span) = find_whitespace_utf8(&self.text[self.position..]) {
                if span.offset == 0 {
                    // Whitespace at current position, skip it
                    self.position += span.length;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if self.position >= self.text.len() {
            return None;
        }

        let start = self.position;

        // Find next whitespace or end of text
        if let Some(span) = find_whitespace_utf8(&self.text[self.position..]) {
            let end = self.position + span.offset;
            self.position = end + span.length;
            Some(&self.text[start..end])
        } else {
            // No more whitespace, return rest of text
            self.position = self.text.len();
            Some(&self.text[start..])
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

    /// Finds the first newline character in UTF-8 encoded text.
    ///
    /// Returns an `IndexSpan` containing the byte offset and length of the matched newline.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = "Hello\nWorld";
    /// let span = text.sz_utf8_find_newline().unwrap();
    /// assert_eq!(span.offset, 5);
    /// assert_eq!(span.length, 1);
    ///
    /// let text_crlf = "Hello\r\nWorld";
    /// let span = text_crlf.sz_utf8_find_newline().unwrap();
    /// assert_eq!(span.offset, 5);
    /// assert_eq!(span.length, 2);
    /// ```
    fn sz_utf8_find_newline(&self) -> Option<IndexSpan>;

    /// Finds the first whitespace character in UTF-8 encoded text.
    ///
    /// Returns an `IndexSpan` containing the byte offset and length of the matched whitespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = "Hello World";
    /// let span = text.sz_utf8_find_whitespace().unwrap();
    /// assert_eq!(span.offset, 5);
    /// assert_eq!(span.length, 1);
    /// ```
    fn sz_utf8_find_whitespace(&self) -> Option<IndexSpan>;

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
    /// let view = text.sz_utf8_chars();
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
    fn sz_utf8_chars(&self) -> Utf8View<'_>;

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
    /// let lines: Vec<&str> = text.sz_utf8_newline_splits()
    ///     .map(|line| std::str::from_utf8(line).unwrap())
    ///     .collect();
    /// assert_eq!(lines, vec!["Hello", "World", "Rust"]);
    /// ```
    fn sz_utf8_newline_splits(&self) -> RangeNewlineUtf8Splits<'_>;

    /// Returns an iterator over words split by UTF-8 whitespace characters.
    ///
    /// The iterator yields non-empty slices between whitespace. Handles all 29 Unicode
    /// whitespace characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = "Hello  World\tRust";
    /// let words: Vec<&str> = text.sz_utf8_whitespace_splits()
    ///     .map(|word| std::str::from_utf8(word).unwrap())
    ///     .collect();
    /// assert_eq!(words, vec!["Hello", "World", "Rust"]);
    /// ```
    fn sz_utf8_whitespace_splits(&self) -> RangeWhitespaceUtf8Splits<'_>;
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
    fn sz_matches(&'a self, needle: &'a N) -> RangeMatches<'a>;

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
    fn sz_rmatches(&'a self, needle: &'a N) -> RangeRMatches<'a>;

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
    fn sz_splits(&'a self, needle: &'a N) -> RangeSplits<'a>;

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
    fn sz_rsplits(&'a self, needle: &'a N) -> RangeRSplits<'a>;

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
    fn sz_find_first_of(&'a self, needles: &'a N) -> RangeMatches<'a>;

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
    fn sz_find_last_of(&'a self, needles: &'a N) -> RangeRMatches<'a>;

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
    fn sz_find_first_not_of(&'a self, needles: &'a N) -> RangeMatches<'a>;

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
    fn sz_find_last_not_of(&'a self, needles: &'a N) -> RangeRMatches<'a>;
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

    fn sz_utf8_find_newline(&self) -> Option<IndexSpan> {
        find_newline_utf8(self)
    }

    fn sz_utf8_find_whitespace(&self) -> Option<IndexSpan> {
        find_whitespace_utf8(self)
    }

    fn sz_utf8_chars(&self) -> Utf8View<'_> {
        Utf8View::new(self.as_ref())
    }

    fn sz_utf8_newline_splits(&self) -> RangeNewlineUtf8Splits<'_> {
        RangeNewlineUtf8Splits::new(self.as_ref())
    }

    fn sz_utf8_whitespace_splits(&self) -> RangeWhitespaceUtf8Splits<'_> {
        RangeWhitespaceUtf8Splits::new(self.as_ref())
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

    fn sz_matches(&'a self, needle: &'a N) -> RangeMatches<'a> {
        RangeMatches::new(self.as_ref(), MatcherType::Find(needle.as_ref()), true)
    }

    fn sz_rmatches(&'a self, needle: &'a N) -> RangeRMatches<'a> {
        RangeRMatches::new(self.as_ref(), MatcherType::RFind(needle.as_ref()), true)
    }

    fn sz_splits(&'a self, needle: &'a N) -> RangeSplits<'a> {
        RangeSplits::new(self.as_ref(), MatcherType::Find(needle.as_ref()))
    }

    fn sz_rsplits(&'a self, needle: &'a N) -> RangeRSplits<'a> {
        RangeRSplits::new(self.as_ref(), MatcherType::RFind(needle.as_ref()))
    }

    fn sz_find_first_of(&'a self, needles: &'a N) -> RangeMatches<'a> {
        RangeMatches::new(self.as_ref(), MatcherType::FindFirstOf(needles.as_ref()), true)
    }

    fn sz_find_last_of(&'a self, needles: &'a N) -> RangeRMatches<'a> {
        RangeRMatches::new(self.as_ref(), MatcherType::FindLastOf(needles.as_ref()), true)
    }

    fn sz_find_first_not_of(&'a self, needles: &'a N) -> RangeMatches<'a> {
        RangeMatches::new(self.as_ref(), MatcherType::FindFirstNotOf(needles.as_ref()), true)
    }

    fn sz_find_last_not_of(&'a self, needles: &'a N) -> RangeRMatches<'a> {
        RangeRMatches::new(self.as_ref(), MatcherType::FindLastNotOf(needles.as_ref()), true)
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
        assert!(sz::dynamic_dispatch());
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
    fn iter_range_matches_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::Find(b"aa");
        let matches: Vec<_> = RangeMatches::new(haystack, matcher, true).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_range_matches_non_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::Find(b"aa");
        let matches: Vec<_> = RangeMatches::new(haystack, matcher, false).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_range_rmatches_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::RFind(b"aa");
        let matches: Vec<_> = RangeRMatches::new(haystack, matcher, true).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_range_rmatches_non_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::RFind(b"aa");
        let matches: Vec<_> = RangeRMatches::new(haystack, matcher, false).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn argsort_permutation_default() {
        // Test with a slice of string literals.
        let fruits = ["banana", "apple", "cherry"];
        let mut order = [0; 3]; // output buffer must be at least fruits.len()
        sz::argsort_permutation(&fruits, &mut order).expect("argsort_permutation failed");

        // Reconstruct sorted order using the returned indices.
        let sorted_from_api: Vec<_> = order.iter().map(|&i| fruits[i]).collect();

        // Compute expected order using the standard sort.
        let mut expected = fruits.to_vec();
        expected.sort();

        assert_eq!(sorted_from_api, expected);
    }

    #[test]
    fn argsort_permutation_by_custom() {
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
        sz::argsort_permutation_by(|i: usize| people[i].name.as_bytes(), &mut order)
            .expect("argsort_permutation_by failed");

        let sorted_from_api: Vec<_> = order.iter().map(|&i| people[i].name).collect();

        // Compute expected order using standard sorting on the names.
        let mut expected: Vec<_> = people.iter().map(|p| p.name).collect();
        expected.sort();

        assert_eq!(sorted_from_api, expected);
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
    fn find_newline_utf8_lf() {
        let text = "Hello\nWorld";
        let span = sz::find_newline_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_newline_utf8_crlf() {
        let text = "Hello\r\nWorld";
        let span = sz::find_newline_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 2);
    }

    #[test]
    fn find_newline_utf8_vt() {
        let text = "Hello\x0BWorld";
        let span = sz::find_newline_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_newline_utf8_ff() {
        let text = "Hello\x0CWorld";
        let span = sz::find_newline_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_newline_utf8_cr() {
        let text = "Hello\rWorld";
        let span = sz::find_newline_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_newline_utf8_file_separator_not_detected() {
        // U+001C (FILE SEPARATOR) is intentionally NOT detected as a newline
        // These are data structure delimiters used in formats like USV, not line breaks
        let text = "Hello\x1CWorld";
        let span = sz::find_newline_utf8(text);
        assert!(span.is_none(), "FILE SEPARATOR should not be detected as newline");
    }

    #[test]
    fn find_newline_utf8_group_separator_not_detected() {
        // U+001D (GROUP SEPARATOR) is intentionally NOT detected as a newline
        // These are data structure delimiters used in formats like USV, not line breaks
        let text = "Hello\x1DWorld";
        let span = sz::find_newline_utf8(text);
        assert!(span.is_none(), "GROUP SEPARATOR should not be detected as newline");
    }

    #[test]
    fn find_newline_utf8_record_separator_not_detected() {
        // U+001E (RECORD SEPARATOR) is intentionally NOT detected as a newline
        // These are data structure delimiters used in formats like USV, not line breaks
        let text = "Hello\x1EWorld";
        let span = sz::find_newline_utf8(text);
        assert!(span.is_none(), "RECORD SEPARATOR should not be detected as newline");
    }

    #[test]
    fn find_newline_utf8_nel() {
        // U+0085 (NEL - Next Line) is 2 bytes in UTF-8: C2 85
        let text = "Hello\u{0085}World";
        let result = sz::find_newline_utf8(text);
        assert!(result.is_some());
        let span = result.unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 2);
    }

    #[test]
    fn find_newline_utf8_line_separator() {
        // U+2028 (LINE SEPARATOR) is 3 bytes in UTF-8: E2 80 A8
        let text = "Hello\u{2028}World";
        let result = sz::find_newline_utf8(text);
        assert!(result.is_some());
        let span = result.unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 3);
    }

    #[test]
    fn find_newline_utf8_paragraph_separator() {
        // U+2029 (PARAGRAPH SEPARATOR) is 3 bytes in UTF-8: E2 80 A9
        let text = "Hello\u{2029}World";
        let result = sz::find_newline_utf8(text);
        assert!(result.is_some());
        let span = result.unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 3);
    }

    #[test]
    fn find_newline_utf8_not_found() {
        let text = "Hello World";
        assert_eq!(sz::find_newline_utf8(text), None);
    }

    #[test]
    fn find_newline_utf8_empty() {
        let text = "";
        assert_eq!(sz::find_newline_utf8(text), None);
    }

    #[test]
    fn find_newline_utf8_trait_method() {
        use crate::sz::StringZillableUnary;
        let text = "Hello\nWorld";
        let span = text.sz_utf8_find_newline().unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_newline_utf8_trait_method_string() {
        use crate::sz::StringZillableUnary;
        let text = String::from("Hello\nWorld");
        let span = text.sz_utf8_find_newline().unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_whitespace_utf8_space() {
        let text = "Hello World";
        let span = sz::find_whitespace_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_whitespace_utf8_tab() {
        let text = "Hello\tWorld";
        let span = sz::find_whitespace_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_whitespace_utf8_newline() {
        // Whitespace should include newlines
        let text = "Hello\nWorld";
        let span = sz::find_whitespace_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_whitespace_utf8_cr() {
        // Whitespace should include CR (finds CR as 1-byte, not CRLF as 2-byte)
        let text = "Hello\r\nWorld";
        let span = sz::find_whitespace_utf8(text).unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_whitespace_utf8_nbsp() {
        // U+00A0 (NO-BREAK SPACE) is 2 bytes in UTF-8: C2 A0
        let text = "Hello\u{00A0}World";
        let result = sz::find_whitespace_utf8(text);
        assert!(result.is_some());
        let span = result.unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 2);
    }

    #[test]
    fn find_whitespace_utf8_ideographic() {
        // U+3000 (IDEOGRAPHIC SPACE) is 3 bytes in UTF-8: E3 80 80
        let text = "Hello\u{3000}World";
        let result = sz::find_whitespace_utf8(text);
        assert!(result.is_some());
        let span = result.unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 3);
    }

    #[test]
    fn find_whitespace_utf8_en_quad() {
        // U+2000 (EN QUAD) is 3 bytes in UTF-8: E2 80 80
        let text = "Hello\u{2000}World";
        let result = sz::find_whitespace_utf8(text);
        assert!(result.is_some());
        let span = result.unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 3);
    }

    #[test]
    fn find_whitespace_utf8_ogham() {
        // U+1680 (OGHAM SPACE MARK) is 3 bytes in UTF-8: E1 9A 80
        let text = "Hello\u{1680}World";
        let result = sz::find_whitespace_utf8(text);
        assert!(result.is_some());
        let span = result.unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 3);
    }

    #[test]
    fn find_whitespace_utf8_not_found() {
        let text = "HelloWorld";
        assert_eq!(sz::find_whitespace_utf8(text), None);
    }

    #[test]
    fn find_whitespace_utf8_empty() {
        let text = "";
        assert_eq!(sz::find_whitespace_utf8(text), None);
    }

    #[test]
    fn find_whitespace_utf8_trait_method() {
        use crate::sz::StringZillableUnary;
        let text = "Hello World";
        let span = text.sz_utf8_find_whitespace().unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_whitespace_utf8_trait_method_string() {
        use crate::sz::StringZillableUnary;
        let text = String::from("Hello World");
        let span = text.sz_utf8_find_whitespace().unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn find_whitespace_utf8_trait_method_bytes() {
        use crate::sz::StringZillableUnary;
        let text = b"Hello World";
        let span = text.sz_utf8_find_whitespace().unwrap();
        assert_eq!(span.offset, 5);
        assert_eq!(span.length, 1);
    }

    #[test]
    fn iter_newline_utf8_splits() {
        let text = b"a\nb\r\nc\n\nd";
        let lines: Vec<_> = RangeNewlineUtf8Splits::new(text).collect();
        assert_eq!(lines, vec![b"a", b"b", b"c", &b""[..], b"d"]);
    }

    #[test]
    fn iter_newline_utf8_splits_unicode() {
        let text = "Hello\u{2028}World".as_bytes(); // LINE SEPARATOR
        let lines: Vec<_> = RangeNewlineUtf8Splits::new(text).collect();
        assert_eq!(lines, vec!["Hello".as_bytes(), "World".as_bytes()]);
    }

    #[test]
    fn iter_whitespace_utf8_splits() {
        let text = b"  a \t b\n\nc  ";
        let words: Vec<_> = RangeWhitespaceUtf8Splits::new(text).collect();
        assert_eq!(words, vec![b"a", b"b", b"c"]);
    }

    #[test]
    fn iter_whitespace_utf8_splits_unicode() {
        let text = "a\u{3000}b\u{2000}c".as_bytes(); // IDEOGRAPHIC SPACE, EN QUAD
        let words: Vec<_> = RangeWhitespaceUtf8Splits::new(text).collect();
        assert_eq!(words, vec![b"a", b"b", b"c"]);
    }
}
