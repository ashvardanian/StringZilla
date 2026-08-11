//! Shared value types, unified-memory containers, and library introspection.

use core::ffi::{c_char, c_void, CStr};
use core::ops::Index;
use core::ptr;

use allocator_api2::{alloc::AllocError, alloc::Allocator, alloc::Layout};
use stringtape::{BytesTape, BytesTapeView, CharsTape, CharsTapeView};

// Re-export common types from stringzilla
pub use crate::stringzilla::{SortedIdx, Status as SzStatus};

/// Capability flags
pub type Capability = u32;

// Import from stringzilla module
pub use crate::stringzilla::Status;

/// Custom error type that preserves detailed error messages from the C API.
#[derive(Debug)]
pub struct Error {
    pub status: Status,
    pub message: Option<String>,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.message {
            Some(msg) => write!(f, "{}", msg),
            None => write!(f, "{:?}", self.status),
        }
    }
}

impl std::error::Error for Error {}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Error { status, message: None }
    }
}

pub(crate) fn rust_error_from_c_message(status: Status, error_msg: *const c_char) -> Error {
    let message = if !error_msg.is_null() && status != Status::Success {
        unsafe { CStr::from_ptr(error_msg).to_str().ok().map(|s| s.to_string()) }
    } else {
        None
    };

    Error { status, message }
}

/// Tape variant that can hold either 32-bit or 64-bit string tapes with unsigned offsets
pub enum AnyCharsTape<'a> {
    Tape32(CharsTape<u32, UnifiedAlloc>),
    Tape64(CharsTape<u64, UnifiedAlloc>),
    // Zero-copy FFI views (UTF-8)
    View32(CharsTapeView<'a, u32>),
    View64(CharsTapeView<'a, u64>),
}

impl<'a> AnyCharsTape<'a> {
    /// Copy string sequences into an owned unified-memory tape, choosing 32- or 64-bit offsets
    /// automatically from the input size. Use this to feed the engines' `compute_into` methods
    /// without depending on the `stringtape` crate directly.
    pub fn from_sequences<Sequence: AsRef<str>>(sequences: &[Sequence]) -> Result<Self, Error> {
        copy_chars_into_tape(sequences, false)
    }
}

/// Tape variant that can hold either 32-bit or 64-bit byte tapes with unsigned offsets
pub enum AnyBytesTape<'a> {
    Tape32(BytesTape<u32, UnifiedAlloc>),
    Tape64(BytesTape<u64, UnifiedAlloc>),
    // Zero-copy FFI views (bytes)
    View32(BytesTapeView<'a, u32>),
    View64(BytesTapeView<'a, u64>),
}

impl<'a> AnyBytesTape<'a> {
    /// Copy byte sequences into an owned unified-memory tape, choosing 32- or 64-bit offsets
    /// automatically from the input size. Use this to feed the engines' `compute_into` methods
    /// without depending on the `stringtape` crate directly.
    pub fn from_sequences<Sequence: AsRef<[u8]>>(sequences: &[Sequence]) -> Result<Self, Error> {
        copy_bytes_into_tape(sequences, false)
    }
}

/// Internal representation of `sz_sequence_t` for passing to C
#[repr(C)]
pub(crate) struct SzSequence {
    handle: *mut c_void,
    count: usize,
    get_start: extern "C" fn(*mut c_void, usize) -> *const u8,
    get_length: extern "C" fn(*mut c_void, usize) -> usize,
    // Additional fields for our implementation
    starts: *const *const u8,
    lengths: *const usize,
}

/// Raw C API tape structure for 32-bit offsets (data < 4GB),
/// matching `sz_sequence_u32tape_t` in the C API
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct SzSequenceU32Tape {
    data: *const u8,
    offsets: *const u32,
    pub(crate) count: usize,
}

/// Raw C API tape structure for 64-bit offsets (data >= 4GB),
/// matching `sz_sequence_u64tape_t` in the C API
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct SzSequenceU64Tape {
    data: *const u8,
    offsets: *const u64,
    pub(crate) count: usize,
}

// Conversions from tape containers to FFI views
impl From<&BytesTape<u32, UnifiedAlloc>> for SzSequenceU32Tape {
    fn from(tape: &BytesTape<u32, UnifiedAlloc>) -> Self {
        let parts = tape.as_raw_parts();
        SzSequenceU32Tape {
            data: parts.data_ptr,
            offsets: parts.offsets_ptr,
            count: parts.items_count,
        }
    }
}

impl From<&CharsTape<u32, UnifiedAlloc>> for SzSequenceU32Tape {
    fn from(tape: &CharsTape<u32, UnifiedAlloc>) -> Self {
        let parts = tape.as_raw_parts();
        SzSequenceU32Tape {
            data: parts.data_ptr,
            offsets: parts.offsets_ptr,
            count: parts.items_count,
        }
    }
}

impl From<&BytesTape<u64, UnifiedAlloc>> for SzSequenceU64Tape {
    fn from(tape: &BytesTape<u64, UnifiedAlloc>) -> Self {
        let parts = tape.as_raw_parts();
        SzSequenceU64Tape {
            data: parts.data_ptr,
            offsets: parts.offsets_ptr,
            count: parts.items_count,
        }
    }
}

impl From<&CharsTape<u64, UnifiedAlloc>> for SzSequenceU64Tape {
    fn from(tape: &CharsTape<u64, UnifiedAlloc>) -> Self {
        let parts = tape.as_raw_parts();
        SzSequenceU64Tape {
            data: parts.data_ptr,
            offsets: parts.offsets_ptr,
            count: parts.items_count,
        }
    }
}

// Conversions from stringtape views to FFI views
impl<'a> From<BytesTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: BytesTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<BytesTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: BytesTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&BytesTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: &BytesTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&BytesTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: &BytesTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<CharsTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: CharsTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<CharsTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: CharsTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&CharsTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: &CharsTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&CharsTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: &CharsTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

/// Generic callback to get start of string at index for byte slices
extern "C" fn sz_sequence_get_start_generic<Sequence: AsRef<[u8]>>(handle: *mut c_void, index: usize) -> *const u8 {
    unsafe {
        let strings = core::slice::from_raw_parts(handle as *const Sequence, index + 1);
        strings[index].as_ref().as_ptr()
    }
}

/// Generic callback to get length of string at index for byte slices
extern "C" fn sz_sequence_get_length_generic<Sequence: AsRef<[u8]>>(handle: *mut c_void, index: usize) -> usize {
    unsafe {
        let strings = core::slice::from_raw_parts(handle as *const Sequence, index + 1);
        strings[index].as_ref().len()
    }
}

/// Generic callback to get start of string at index for string slices
extern "C" fn sz_sequence_get_start_str<Sequence: AsRef<str>>(handle: *mut c_void, index: usize) -> *const u8 {
    unsafe {
        let strings = core::slice::from_raw_parts(handle as *const Sequence, index + 1);
        strings[index].as_ref().as_bytes().as_ptr()
    }
}

/// Generic callback to get length of string at index for string slices
extern "C" fn sz_sequence_get_length_str<Sequence: AsRef<str>>(handle: *mut c_void, index: usize) -> usize {
    unsafe {
        let strings = core::slice::from_raw_parts(handle as *const Sequence, index + 1);
        strings[index].as_ref().as_bytes().len()
    }
}

/// Trait for types that can be converted to SzSequence for byte sequences
pub(crate) trait SzSequenceFromBytes {
    fn to_sz_sequence(&self) -> SzSequence;
}

impl<Sequence: AsRef<[u8]>> SzSequenceFromBytes for [Sequence] {
    fn to_sz_sequence(&self) -> SzSequence {
        SzSequence {
            handle: self.as_ptr() as *mut c_void,
            count: self.len(),
            get_start: sz_sequence_get_start_generic::<Sequence>,
            get_length: sz_sequence_get_length_generic::<Sequence>,
            starts: ptr::null(),
            lengths: ptr::null(),
        }
    }
}

/// Trait for types that can be converted to SzSequence for string sequences
pub(crate) trait SzSequenceFromChars {
    fn to_sz_sequence(&self) -> SzSequence;
}

impl<Sequence: AsRef<str>> SzSequenceFromChars for [Sequence] {
    fn to_sz_sequence(&self) -> SzSequence {
        SzSequence {
            handle: self.as_ptr() as *mut c_void,
            count: self.len(),
            get_start: sz_sequence_get_start_str::<Sequence>,
            get_length: sz_sequence_get_length_str::<Sequence>,
            starts: ptr::null(),
            lengths: ptr::null(),
        }
    }
}

// C API bindings
extern "C" {

    // Metadata functions
    fn szs_version_major() -> i32;
    fn szs_version_minor() -> i32;
    fn szs_version_patch() -> i32;
    fn szs_capabilities() -> u32;

    // Unified allocator functions
    fn szs_unified_alloc(size_bytes: usize) -> *mut c_void;
    fn szs_unified_free(ptr: *mut c_void, size_bytes: usize);

}

/// Unified memory allocator that uses CUDA unified memory when available,
/// falls back to malloc otherwise. Works with allocator-api2.
pub struct UnifiedAlloc;

unsafe impl Allocator for UnifiedAlloc {
    fn allocate(&self, layout: Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if size == 0 {
            // For zero-sized allocations, return a properly aligned non-null dangling pointer
            let ptr = core::ptr::NonNull::new(layout.align() as *mut u8).ok_or(AllocError)?;
            return Ok(core::ptr::NonNull::slice_from_raw_parts(ptr, 0));
        }

        let ptr = unsafe { szs_unified_alloc(size) };
        if ptr.is_null() {
            return Err(AllocError);
        }

        let ptr = core::ptr::NonNull::new(ptr as *mut u8).ok_or(AllocError)?;
        Ok(core::ptr::NonNull::slice_from_raw_parts(ptr, size))
    }

    unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            szs_unified_free(ptr.as_ptr() as *mut c_void, layout.size());
        }
    }
}

/// Type alias for Vec with unified allocator
pub type UnifiedVec<T> = allocator_api2::vec::Vec<T, UnifiedAlloc>;

/// Row-major cross-product result matrix produced by the similarity engines.
///
/// Every similarity engine computes a dense `queries_count × candidates_count`
/// matrix where `matrix[(query_index, candidate_index)]` holds the distance or
/// score between `queries[query_index]` and `candidates[candidate_index]`.
///
/// The backing storage is a [`UnifiedVec`], so the buffer lives in unified
/// memory and can be consumed by a GPU kernel without an extra copy. Rows are
/// laid out contiguously with a stride of `row_stride` elements; for a freshly
/// allocated matrix `row_stride == candidates_count`.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, LevenshteinDistances};
/// let device = DeviceScope::default().unwrap();
/// let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
///
/// let queries = vec!["kitten", "saturday"];
/// let candidates = vec!["sitting", "sunday"];
/// let matrix = engine.compute(&device, &queries, &candidates).unwrap();
/// assert_eq!(matrix.dimensions(), (2, 2));
/// assert_eq!(matrix[(0, 0)], 3); // kitten vs sitting
/// ```
pub struct Mat<Element, Alloc: Allocator = allocator_api2::alloc::Global> {
    pub(crate) data: allocator_api2::vec::Vec<Element, Alloc>,
    pub(crate) queries_count: usize,
    pub(crate) candidates_count: usize,
    pub(crate) row_stride: usize,
}

/// A [`Mat`] backed by unified (CPU+GPU) memory - the matrix counterpart of [`UnifiedVec`].
pub type UnifiedMat<Element> = Mat<Element, UnifiedAlloc>;

impl<Element> Mat<Element, UnifiedAlloc>
where
    Element: Copy + Default,
{
    /// Allocate a zero-initialized `queries_count × candidates_count` matrix in unified memory, returning
    /// an error on allocation failure rather than panicking.
    ///
    /// The row stride equals `candidates_count`, so rows are stored back-to-back with no padding. The buffer
    /// is suitable for direct GPU consumption.
    pub fn try_allocate(queries_count: usize, candidates_count: usize) -> Result<Self, Error> {
        let element_count = queries_count.saturating_mul(candidates_count);
        let mut data = allocator_api2::vec::Vec::new_in(UnifiedAlloc);
        data.try_reserve_exact(element_count)
            .map_err(|_| Error::from(SzStatus::BadAlloc))?;
        data.resize(element_count, Element::default()); // No reallocation: capacity was just reserved.
        Ok(Mat {
            data,
            queries_count,
            candidates_count,
            row_stride: candidates_count,
        })
    }
}

impl<Element, Alloc: Allocator> Mat<Element, Alloc> {
    /// Returns the `(queries_count, candidates_count)` shape of the matrix.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.queries_count, self.candidates_count)
    }

    /// Returns the number of query rows.
    pub fn queries_count(&self) -> usize {
        self.queries_count
    }

    /// Returns the number of candidate columns.
    pub fn candidates_count(&self) -> usize {
        self.candidates_count
    }

    /// Returns the element stride between consecutive rows in the backing buffer.
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns the row of distances for `query_index` as a contiguous slice of `candidates_count` elements.
    pub fn row(&self, query_index: usize) -> &[Element] {
        let row_start = query_index * self.row_stride;
        &self.data[row_start..row_start + self.candidates_count]
    }

    /// Returns the entire backing buffer (including any inter-row padding) as a flat slice.
    pub fn as_slice(&self) -> &[Element] {
        &self.data[..]
    }
}

impl<Element, Alloc: Allocator> Index<(usize, usize)> for Mat<Element, Alloc> {
    type Output = Element;

    fn index(&self, (query_index, candidate_index): (usize, usize)) -> &Element {
        &self.data[query_index * self.row_stride + candidate_index]
    }
}

/// Returns StringZillas similarity engine version information.
pub fn version() -> crate::stringzilla::SemVer {
    crate::stringzilla::SemVer {
        major: unsafe { szs_version_major() },
        minor: unsafe { szs_version_minor() },
        patch: unsafe { szs_version_patch() },
    }
}

/// Copies the capabilities C-string into a fixed buffer and returns it.
/// The returned SmallCString is guaranteed to be null-terminated.
pub fn capabilities() -> crate::stringzilla::SmallCString {
    let caps = unsafe { szs_capabilities() };
    crate::stringzilla::capabilities_from_enum(caps)
}

/// Check if either byte collection requires 64-bit tapes
pub(crate) fn should_use_64bit_for_bytes<Sequence: AsRef<[u8]>>(seq_a: &[Sequence], seq_b: &[Sequence]) -> bool {
    let total_size_a: usize = seq_a.iter().map(|s| s.as_ref().len()).sum();
    let total_size_b: usize = seq_b.iter().map(|s| s.as_ref().len()).sum();
    total_size_a > u32::MAX as usize
        || seq_a.len() > u32::MAX as usize
        || total_size_b > u32::MAX as usize
        || seq_b.len() > u32::MAX as usize
}

/// Check if either string collection requires 64-bit tapes
pub(crate) fn should_use_64bit_for_strings<Sequence: AsRef<str>>(seq_a: &[Sequence], seq_b: &[Sequence]) -> bool {
    let total_size_a: usize = seq_a.iter().map(|s| s.as_ref().len()).sum();
    let total_size_b: usize = seq_b.iter().map(|s| s.as_ref().len()).sum();
    total_size_a > u32::MAX as usize
        || seq_a.len() > u32::MAX as usize
        || total_size_b > u32::MAX as usize
        || seq_b.len() > u32::MAX as usize
}

/// Convert byte sequences to BytesTape
pub(crate) fn copy_bytes_into_tape<'a, Sequence>(
    sequences: &[Sequence],
    force_64bit: bool,
) -> Result<AnyBytesTape<'a>, Error>
where
    Sequence: AsRef<[u8]>,
{
    // Estimate total size to decide between 32-bit and 64-bit tapes
    let total_size: usize = sequences.iter().map(|s| s.as_ref().len()).sum();
    let use_64bit = force_64bit || total_size > u32::MAX as usize || sequences.len() > u32::MAX as usize;

    if use_64bit {
        let mut tape = BytesTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyBytesTape::Tape64(tape))
    } else {
        let mut tape = BytesTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyBytesTape::Tape32(tape))
    }
}

/// Convert string sequences to CharsTape
pub(crate) fn copy_chars_into_tape<'a, Sequence: AsRef<str>>(
    sequences: &[Sequence],
    force_64bit: bool,
) -> Result<AnyCharsTape<'a>, Error> {
    // Estimate total size to decide between 32-bit and 64-bit tapes
    let total_size: usize = sequences.iter().map(|s| s.as_ref().len()).sum();
    let use_64bit = force_64bit || total_size > u32::MAX as usize || sequences.len() > u32::MAX as usize;

    if use_64bit {
        let mut tape = CharsTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyCharsTape::Tape64(tape))
    } else {
        let mut tape = CharsTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyCharsTape::Tape32(tape))
    }
}

/// Get information about the compiled backend
///
/// # Examples
///
/// ```
/// # use stringzilla::szs::backend_info;
/// let info = backend_info();
/// println!("Using backend: {}", info);
/// ```
pub fn backend_info() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA GPU acceleration enabled"
    } else if cfg!(all(feature = "rocm", not(feature = "cuda"))) {
        "ROCm GPU acceleration enabled"
    } else if cfg!(all(feature = "cpus", not(any(feature = "cuda", feature = "rocm")))) {
        "Multi-threaded CPU backend enabled"
    } else if cfg!(not(any(feature = "cpus", feature = "cuda", feature = "rocm"))) {
        "StringZillas not available - enable cpus, cuda, or rocm feature"
    } else {
        "Unknown backend"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_info() {
        let info = super::backend_info();
        assert!(!info.is_empty());
        println!("Backend: {}", info);
    }

    #[test]
    fn unified_allocator() {
        // Test basic allocation
        let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
        let alloc = UnifiedAlloc;

        let result = alloc.allocate(layout);
        match result {
            Ok(memory) => {
                println!("Unified allocation successful: {} bytes", memory.len());
                unsafe { alloc.deallocate(memory.cast(), layout) };
            }
            Err(_) => println!("Unified allocation failed"),
        }

        // Test zero-size allocation
        let zero_layout = std::alloc::Layout::from_size_align(0, 1).unwrap();
        let zero_result = alloc.allocate(zero_layout);
        match zero_result {
            Ok(memory) => {
                assert_eq!(memory.len(), 0);
                unsafe { alloc.deallocate(memory.cast(), zero_layout) };
            }
            Err(_) => println!("Zero-size allocation failed"),
        }
    }
}
