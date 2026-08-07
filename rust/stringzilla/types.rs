//! Shared value types, status codes, and library introspection.

use super::*;
use core::ffi::{c_char, c_void, CStr};
use core::fmt::{self, Write};

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
    /// An authenticated decryption saw a tag that does not match the ciphertext it accompanies.
    /// Corresponds to `sz_authentication_failed_k = -19` in C.
    AuthenticationFailed = -19,
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
    pub(crate) bits: [u64; 4],
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
    pub const fn new(offset: usize, length: usize) -> Self {
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
    pub const fn range(&self) -> core::ops::Range<usize> {
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
    pub const fn end(&self) -> usize {
        self.offset + self.length
    }
}

pub type SortedIdx = usize;

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
    pub const fn new() -> Self {
        Self { bits: [0; 4] }
    }

    /// Initializes a bit-set to contain all ASCII characters.
    #[inline]
    pub const fn new_ascii() -> Self {
        Self {
            bits: [u64::MAX, u64::MAX, 0, 0],
        }
    }

    /// Adds a byte to the set.
    #[inline]
    pub const fn add_u8(&mut self, byte: u8) {
        let word = (byte >> 6) as usize; // Divide by 64.
        let bit = byte & 63; // Remainder modulo 64.
        self.bits[word] |= 1 << bit;
    }

    /// Adds a character to the set.
    ///
    /// This function assumes the character is in the ASCII range.
    #[inline]
    pub const fn add(&mut self, character: char) {
        self.add_u8(character as u8);
    }

    /// Inverts the bit-set so that all set bits become unset and vice versa.
    #[inline]
    pub const fn invert(&mut self) {
        let mut word = 0;
        while word < 4 {
            self.bits[word] = !self.bits[word];
            word += 1;
        }
    }

    /// Returns a new Byteset with all bits inverted, leaving self unchanged.
    #[inline]
    pub const fn inverted(&self) -> Self {
        Self {
            bits: [!self.bits[0], !self.bits[1], !self.bits[2], !self.bits[3]],
        }
    }

    /// Constructs a Byteset from a slice of bytes.
    ///
    /// Usable in a `const`, so a hot loop can name a set instead of rebuilding one per call:
    ///
    /// ```rust
    /// use stringzilla::sz::Byteset;
    /// const WHITESPACE: Byteset = Byteset::from_bytes(b" \t\r\n");
    /// assert_eq!(stringzilla::sz::find_byteset("ab cd", WHITESPACE), Some(2));
    /// ```
    #[inline]
    pub const fn from_bytes(bytes: &[u8]) -> Self {
        let mut set = Self::new();
        let mut index = 0;
        while index < bytes.len() {
            set.add_u8(bytes[index]);
            index += 1;
        }
        set
    }
}

impl Default for Byteset {
    fn default() -> Self {
        Self::new()
    }
}

impl<Source: AsRef<[u8]>> From<Source> for Byteset {
    #[inline]
    fn from(bytes: Source) -> Self {
        Self::from_bytes(bytes.as_ref())
    }
}

impl SemVer {
    pub const fn new(major: i32, minor: i32, patch: i32) -> Self {
        Self { major, minor, patch }
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
pub struct FixedCString<const CAPACITY: usize> {
    buf: [u8; CAPACITY],
    len: usize,
}

impl<const CAPACITY: usize> FixedCString<CAPACITY> {
    /// Create a new, empty buffer.
    /// The buffer always has a terminating NUL (0) byte at position `len`.
    pub const fn new() -> Self {
        Self {
            buf: [0u8; CAPACITY],
            len: 0,
        }
    }

    /// Returns the raw pointer to the C string.
    pub const fn as_ptr(&self) -> *const u8 {
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

impl<const CAPACITY: usize> Default for FixedCString<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> Write for FixedCString<CAPACITY> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        // Ensure we have room for the new bytes and a NUL terminator.
        if self.len + bytes.len() >= CAPACITY {
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

/// Type-punned wrapper for the slice lookup view.
///
/// Carries a mapper closure which, given an index, returns the corresponding byte-slice representation,
/// alongside the monomorphized accessor that recovers the closure's concrete type. Callers write closures
/// like `|i| data[i].as_ref()` or `|i| people[i].name.as_bytes()`.
pub(crate) struct _PunnedSliceLookupView {
    pub(crate) get_slice: unsafe fn(*const c_void, usize) -> &'static [u8],
    pub(crate) data: *const c_void,
}

pub(crate) unsafe extern "C" fn _slice_get_start_punned(handle: *const c_void, idx: SortedIdx) -> *const c_void {
    let view = &*(handle as *const _PunnedSliceLookupView);
    let slice = (view.get_slice)(view.data, idx);
    slice.as_ptr() as *const c_void
}

pub(crate) unsafe extern "C" fn _slice_get_length_punned(handle: *const c_void, idx: SortedIdx) -> usize {
    let view = &*(handle as *const _PunnedSliceLookupView);
    let slice = (view.get_slice)(view.data, idx);
    slice.len()
}

/// Type-specific function generator for each concrete type
pub(crate) unsafe fn _get_slice_fn<Mapper>() -> unsafe fn(*const c_void, usize) -> &'static [u8]
where
    Mapper: Fn(usize) -> &'static [u8],
{
    unsafe fn get_slice_impl<Mapper>(data: *const c_void, idx: usize) -> &'static [u8]
    where
        Mapper: Fn(usize) -> &'static [u8],
    {
        let mapper = &*(data as *const Mapper);
        mapper(idx)
    }
    get_slice_impl::<Mapper>
}

/// Compile-time policy for whether a split keeps or drops empty (zero-length) segments.
///
/// A named marker type rather than a raw `bool`, so the choice is branchless and readable at call sites.
pub trait EmptySegments {
    /// Whether zero-length segments are skipped.
    const SKIP: bool;
}
/// Keep empty segments (the default).
pub struct KeepEmpty;
impl EmptySegments for KeepEmpty {
    const SKIP: bool = false;
}
/// Drop empty segments (via `.skip_empty()`).
pub struct SkipEmpty;
impl EmptySegments for SkipEmpty {
    const SKIP: bool = true;
}

/// Compile-time policy for whether overlapping matches are reported - a named marker, not a raw `bool`.
pub trait Overlaps {
    /// Whether overlapping matches are included.
    const OVERLAP: bool;
}
/// Report only non-overlapping matches (the default; like `str::matches`).
pub struct NonOverlapping;
impl Overlaps for NonOverlapping {
    const OVERLAP: bool = false;
}
/// Report overlapping matches too (via `.overlapping()`).
pub struct Overlapping;
impl Overlaps for Overlapping {
    const OVERLAP: bool = true;
}

/// Default number of boundaries the UTF-8 segmentation iterators buffer per call.
///
/// Buffering this many amortizes the per-item dispatch overhead without an unbounded buffer; a full
/// buffer simply resumes where it left off on the next call.
pub const ITERATORS_DEFAULT_STEPS: usize = 64;

#[cfg(test)]
mod tests {
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
    fn const_apis() {
        // Each constant is built at compile time, so dropping a `const fn` breaks the build, not this run.
        const SPAN: sz::IndexSpan = sz::IndexSpan::new(6, 5);
        const WHITESPACE: sz::Byteset = sz::Byteset::from_bytes(b" \t\r\n");
        const NEEDLE: sz::Utf8UncasedNeedle = sz::Utf8UncasedNeedle::new(b"hello");
        const TOP_TWO_DESCENDING: sz::ArgsortOptions = sz::ArgsortOptions {
            reverse: false,
            uncased: false,
            top: None,
        }
        .reversed()
        .top(2);

        assert_eq!(SPAN.extract(b"Hello World"), b"World");
        assert_eq!(sz::find_byteset("ab cd", WHITESPACE), Some(2));
        assert_eq!(sz::utf8_uncased_search(b"say HELLO now", &NEEDLE), Some((4, 5)));

        let fruits = ["banana", "apple", "cherry"];
        let mut order = [0; 3];
        sz::argsort(&fruits, &mut order, TOP_TWO_DESCENDING).expect("argsort failed");
        assert_eq!(fruits[order[0]], "cherry");
    }
}
