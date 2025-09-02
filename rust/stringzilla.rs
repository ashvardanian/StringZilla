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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[repr(align(64))] // For optimal performance we align to 64 bytes.
pub struct HashState {
    aes: [u64; 8],
    sum: [u64; 8],
    ins: [u64; 8], // Ignored in comparisons
    key: [u64; 2],
    ins_length: usize, // Ignored in comparisons
}

pub type SortedIdx = usize;

/// A trait for types that support indexed lookup.
pub trait SequenceData {
    type Item;
    fn len(&self) -> usize;
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
    /// Initializes a bit‑set to an empty collection (all characters banned).
    #[inline]
    pub fn new() -> Self {
        Self { bits: [0; 4] }
    }

    /// Initializes a bit‑set to contain all ASCII characters.
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

impl<T: AsRef<[u8]>> From<T> for Byteset {
    #[inline]
    fn from(bytes: T) -> Self {
        Self::from_bytes(bytes.as_ref())
    }
}

use core::fmt::{self, Write};
use core::{ffi::c_void, ffi::CStr, usize};

// Import the functions from the StringZillable C library.
extern "C" {

    fn sz_dynamic_dispatch() -> i32;
    fn sz_version_major() -> i32;
    fn sz_version_minor() -> i32;
    fn sz_version_patch() -> i32;
    fn sz_capabilities() -> u32;
    fn sz_capabilities_to_string(caps: u32) -> *const c_void;

    fn sz_copy(target: *const c_void, source: *const c_void, length: usize);
    fn sz_fill(target: *const c_void, length: usize, value: u8);
    fn sz_move(target: *const c_void, source: *const c_void, length: usize);
    fn sz_fill_random(text: *mut c_void, length: usize, seed: u64);
    fn sz_lookup(target: *const c_void, length: usize, source: *const c_void, lut: *const u8) -> *const c_void;

    fn sz_find(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    fn sz_rfind(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    fn sz_find_byteset(haystack: *const c_void, haystack_length: usize, byteset: *const c_void) -> *const c_void;
    fn sz_rfind_byteset(haystack: *const c_void, haystack_length: usize, byteset: *const c_void) -> *const c_void;

    fn sz_bytesum(text: *const c_void, length: usize) -> u64;
    fn sz_hash(text: *const c_void, length: usize, seed: u64) -> u64;
    fn sz_hash_state_init(state: *const c_void, seed: u64);
    fn sz_hash_state_stream(state: *const c_void, text: *const c_void, length: usize);
    fn sz_hash_state_fold(state: *const c_void) -> u64;

    pub fn sz_sequence_argsort(
        //
        sequence: *const _SzSequence,
        alloc: *const c_void,
        order: *mut SortedIdx,
    ) -> Status;

    pub fn sz_sequence_intersect(
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

impl HashState {
    /// Creates a new `HashState` and initializes it with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut state = HashState {
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

    /// Streams data into the hash state.
    pub fn stream(&mut self, data: &[u8]) -> &mut Self {
        unsafe {
            sz_hash_state_stream(
                self as *mut _ as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len(),
            );
        }
        self
    }

    /// Finalizes the hash and returns the folded value.
    pub fn fold(&self) -> u64 {
        unsafe { sz_hash_state_fold(self as *const _ as *const c_void) }
    }
}

impl PartialEq for HashState {
    fn eq(&self, other: &Self) -> bool {
        self.aes == other.aes && self.sum == other.sum && self.key == other.key
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
    /// Returns an empty string if the content isn’t valid UTF‑8.
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
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
pub fn capabilities() -> SmallCString {
    let caps = unsafe { sz_capabilities() };
    let caps_ptr = unsafe { sz_capabilities_to_string(caps) };
    // Assume that the external function returns a valid null-terminated C string.
    let cstr = unsafe { CStr::from_ptr(caps_ptr as *const i8) };
    let bytes = cstr.to_bytes();

    let mut buf = SmallCString::new();
    // Use core::fmt::Write to copy the bytes.
    // If the string is too long, it will fail. You might want to truncate in a real-world use.
    // Here, we assume it fits.
    let s = core::str::from_utf8(bytes).unwrap_or("");
    let _ = buf.write_str(s);
    buf
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
    let result = unsafe { sz_bytesum(text_pointer, text_length) };
    return result;
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
    let result = unsafe { sz_hash(text_pointer, text_length, seed) };
    return result;
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
        Some(unsafe { result.offset_from(haystack_pointer) } as usize)
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
        Some(unsafe { result.offset_from(haystack_pointer) } as usize)
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
        Some(unsafe { result.offset_from(haystack_pointer) } as usize)
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
        Some(unsafe { result.offset_from(haystack_pointer) } as usize)
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

/// Generates a default substitution matrix for use with the Needleman-Wunsch
/// alignment algorithm. This matrix is initialized such that diagonal entries
/// (representing matching characters) are zero, and off-diagonal entries
/// (representing mismatches) are -1. This setup effectively produces distances
/// equal to the negative Levenshtein edit distance, suitable for basic sequence
/// alignment tasks where all mismatches are penalized equally and there are no
/// rewards for matches.
///
/// # Returns
///
/// A 256x256 array of `i8`, where each element represents the substitution cost
/// between two characters (byte values). Matching characters are assigned a cost
/// of 0, and non-matching characters are assigned a cost of -1.
pub fn error_costs_256x256_unary() -> [[i8; 256]; 256] {
    let mut result = [[0i8; 256]; 256];

    for i in 0..256 {
        for j in 0..256 {
            result[i][j] = if i == j { 0 } else { -1 };
        }
    }

    result
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
/// returns the corresponding byte‑slice representation.
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

/// Sorts a sequence of items by comparing their byte‑slice representations.
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
    argsort_permutation_by(|i| data[i].as_ref(), order)
}

/// Sorts a sequence of items by comparing their corresponding byte‑slice representations.
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

/// Intersects two sequences (inner join) using their default byte‑slice views.
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

/// Intersects two sequences (inner join) using their elements corresponding byte‑slice views.
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashSet;

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
                sz::HashState::new(*seed).stream("Hello".as_bytes()).fold(),
                sz::hash_with_seed("Hello", *seed)
            );
            // Dual pass for short strings
            assert_eq!(
                sz::HashState::new(*seed)
                    .stream("Hello".as_bytes())
                    .stream("World".as_bytes())
                    .fold(),
                sz::hash_with_seed("HelloWorld", *seed)
            );
        }
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

    mod search_split_iterators {
        use super::*;

        #[test]
        fn test_matches() {
            let haystack = b"hello world hello universe";
            let needle = b"hello";
            let matches: Vec<_> = haystack.sz_matches(needle).collect();
            assert_eq!(matches, vec![b"hello", b"hello"]);
        }

        #[test]
        fn test_rmatches() {
            let haystack = b"hello world hello universe";
            let needle = b"hello";
            let matches: Vec<_> = haystack.sz_rmatches(needle).collect();
            assert_eq!(matches, vec![b"hello", b"hello"]);
        }

        #[test]
        fn test_splits() {
            let haystack = b"alpha,beta;gamma";
            let needle = b",";
            let splits: Vec<_> = haystack.sz_splits(needle).collect();
            assert_eq!(splits, vec![&b"alpha"[..], &b"beta;gamma"[..]]);
        }

        #[test]
        fn test_rsplits() {
            let haystack = b"alpha,beta;gamma";
            let needle = b";";
            let splits: Vec<_> = haystack.sz_rsplits(needle).collect();
            assert_eq!(splits, vec![&b"gamma"[..], &b"alpha,beta"[..]]);
        }

        #[test]
        fn test_splits_with_empty_parts() {
            let haystack = b"a,,b,";
            let needle = b",";
            let splits: Vec<_> = haystack.sz_splits(needle).collect();
            assert_eq!(splits, vec![b"a", &b""[..], b"b", &b""[..]]);
        }

        #[test]
        fn test_matches_with_overlaps() {
            let haystack = b"aaaa";
            let needle = b"aa";
            let matches: Vec<_> = haystack.sz_matches(needle).collect();
            assert_eq!(matches, vec![b"aa", b"aa", b"aa"]);
        }

        #[test]
        fn test_splits_with_utf8() {
            let haystack = "こんにちは,世界".as_bytes();
            let needle = b",";
            let splits: Vec<_> = haystack.sz_splits(needle).collect();
            assert_eq!(splits, vec!["こんにちは".as_bytes(), "世界".as_bytes()]);
        }

        #[test]
        fn test_find_first_of() {
            let haystack = b"hello world";
            let needles = b"or";
            let matches: Vec<_> = haystack.sz_find_first_of(needles).collect();
            assert_eq!(matches, vec![b"o", b"o", b"r"]);
        }

        #[test]
        fn test_find_last_of() {
            let haystack = b"hello world";
            let needles = b"or";
            let matches: Vec<_> = haystack.sz_find_last_of(needles).collect();
            assert_eq!(matches, vec![b"r", b"o", b"o"]);
        }

        #[test]
        fn test_find_first_not_of() {
            let haystack = b"aabbbcccd";
            let needles = b"ab";
            let matches: Vec<_> = haystack.sz_find_first_not_of(needles).collect();
            assert_eq!(matches, vec![b"c", b"c", b"c", b"d"]);
        }

        #[test]
        fn test_find_last_not_of() {
            let haystack = b"aabbbcccd";
            let needles = b"cd";
            let matches: Vec<_> = haystack.sz_find_last_not_of(needles).collect();
            assert_eq!(matches, vec![b"b", b"b", b"b", b"a", b"a"]);
        }

        #[test]
        fn test_find_first_of_empty_needles() {
            let haystack = b"hello world";
            let needles = b"";
            let matches: Vec<_> = haystack.sz_find_first_of(needles).collect();
            assert_eq!(matches, Vec::<&[u8]>::new());
        }

        #[test]
        fn test_find_last_of_empty_haystack() {
            let haystack = b"";
            let needles = b"abc";
            let matches: Vec<_> = haystack.sz_find_last_of(needles).collect();
            assert_eq!(matches, Vec::<&[u8]>::new());
        }

        #[test]
        fn test_find_first_not_of_all_matching() {
            let haystack = b"aaabbbccc";
            let needles = b"abc";
            let matches: Vec<_> = haystack.sz_find_first_not_of(needles).collect();
            assert_eq!(matches, Vec::<&[u8]>::new());
        }

        #[test]
        fn test_find_last_not_of_all_not_matching() {
            let haystack = b"hello world";
            let needles = b"xyz";
            let matches: Vec<_> = haystack.sz_find_last_not_of(needles).collect();
            assert_eq!(
                matches,
                vec![b"d", b"l", b"r", b"o", b"w", b" ", b"o", b"l", b"l", b"e", b"h"]
            );
        }

        #[test]
        fn test_range_matches_overlapping() {
            let haystack = b"aaaa";
            let matcher = MatcherType::Find(b"aa");
            let matches: Vec<_> = RangeMatches::new(haystack, matcher, true).collect();
            assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
        }

        #[test]
        fn test_range_matches_non_overlapping() {
            let haystack = b"aaaa";
            let matcher = MatcherType::Find(b"aa");
            let matches: Vec<_> = RangeMatches::new(haystack, matcher, false).collect();
            assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
        }

        #[test]
        fn test_range_rmatches_overlapping() {
            let haystack = b"aaaa";
            let matcher = MatcherType::RFind(b"aa");
            let matches: Vec<_> = RangeRMatches::new(haystack, matcher, true).collect();
            assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
        }

        #[test]
        fn test_range_rmatches_non_overlapping() {
            let haystack = b"aaaa";
            let matcher = MatcherType::RFind(b"aa");
            let matches: Vec<_> = RangeRMatches::new(haystack, matcher, false).collect();
            assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
        }
    }

    #[test]
    fn test_argsort_permutation_default() {
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
    fn test_argsort_permutation_by_custom() {
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
    fn test_intersection_default() {
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
    fn test_intersection_by_custom() {
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
    fn test_intersection_debug() {
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
}
