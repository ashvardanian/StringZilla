//! Shared value types, status codes, capabilities and the memory GPUs read, and library
//! introspection.
//!
//! The producers report the [`Capabilities`] of the CPU or of one GPU, by its runtime's ordinal,
//! along two axes that have nothing to do with each other, plus the set dispatch uses:
//!
//! - [`Capabilities::cpu_detected`], [`Capabilities::cuda_detected`]: what the device can
//!   execute, from CPUID / `getauxval` / HWCAP on the CPU and from the runtime on a GPU
//! - [`Capabilities::cpu_compiled`], [`Capabilities::cuda_compiled`]: what this binary contains,
//!   from the build's probes
//! - [`Capabilities::cpu_enabled`], [`Capabilities::cuda_enabled`]: what dispatch uses — both axes
//!   at once, which the library clamps every CPU mask to
//!
//! ROCm and Metal have the same three. Reach for the enabled set unless you specifically mean one
//! of the raw axes. Every consumer, an engine, a [`Sequence`] copy or a [`UnifiedAllocator`], takes
//! a [`Stream`] of that mask, which names the device it queues work on: [`Stream::default`] for the
//! default one, or [`Stream::new`] on any device by its ordinal.
//!
//! File: rust/stringzilla/types.rs
//! Author: Ash Vardanian

use core::ffi::{c_void, CStr};
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::BitOr;

use super::*;

#[allow(non_camel_case_types)]
pub(crate) type sz_capability_t = u64;
#[allow(non_camel_case_types)]
pub(crate) type sz_size_t = usize;
#[allow(non_camel_case_types)]
pub(crate) type sz_status_t = i32;
#[allow(non_camel_case_types)]
pub(crate) type sz_bool_t = i32;
#[allow(non_camel_case_types)]
pub(crate) type sz_ordering_t = i32;

/// A simple semantic version structure.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SemVer {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
}

/// Why a StringZilla call failed: C's `sz_status_t`, each variant holding the header's value.
///
/// Success has no variant, being the `Ok` side of every `Result` this crate returns. A code the
/// header this crate was built from does not list reads as [`Status::Unrecognized`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// `sz_bad_alloc_k`: a memory allocation failed.
    BadAlloc = -10,
    /// `sz_invalid_utf8_k`: an input that must be UTF-8 is not.
    InvalidUtf8 = -12,
    /// `sz_contains_duplicates_k`: a collection of unique elements holds duplicates.
    ContainsDuplicates = -13,
    /// `sz_overflow_risk_k`: an input is too large for the integer types the algorithm counts in.
    OverflowRisk = -14,
    /// `sz_unexpected_dimensions_k`: input and output sizes contradict each other.
    UnexpectedDimensions = -15,
    /// `sz_missing_gpu_k`: no GPU device answers, so nothing can be prepared or scheduled on one.
    MissingGpu = -16,
    /// `sz_device_code_mismatch_k`: no kernel for the call, like a device engine handed to a host
    /// verb or the reverse, no GPU runtime compiled in, or a launch the device refused.
    DeviceCodeMismatch = -17,
    /// `sz_device_memory_mismatch_k`: an operand lies in memory the device cannot address.
    DeviceMemoryMismatch = -18,
    /// `sz_authentication_failed_k`: a tag does not match the ciphertext it accompanies.
    AuthenticationFailed = -19,
    /// `sz_missing_kernel_k`: no capability in the mask the call dispatched on has its kernel.
    MissingKernel = -20,
    /// `sz_missing_library_k`: a dispatch point reached from a header-only build, which links no
    /// library.
    MissingLibrary = -21,
    /// `sz_status_unknown_k`: a failure the C core has no specific code for.
    Unknown = -1,
    /// A code outside this list, which only a library built from another header returns.
    Unrecognized = i32::MIN,
}

impl Status {
    /// Every variant C defines, which a raw code is matched against.
    const LISTED: [Status; 12] = [
        Status::BadAlloc,
        Status::InvalidUtf8,
        Status::ContainsDuplicates,
        Status::OverflowRisk,
        Status::UnexpectedDimensions,
        Status::MissingGpu,
        Status::DeviceCodeMismatch,
        Status::DeviceMemoryMismatch,
        Status::AuthenticationFailed,
        Status::MissingKernel,
        Status::MissingLibrary,
        Status::Unknown,
    ];

    /// The variant holding a failing `code`, or [`Status::Unrecognized`] for
    /// a value C does not list.
    fn from_code(code: sz_status_t) -> Self {
        Self::LISTED
            .into_iter()
            .find(|&status| status as sz_status_t == code)
            .unwrap_or(Status::Unrecognized)
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = unsafe { CStr::from_ptr(sz_status_name(*self as sz_status_t)) };
        f.pad(description.to_str().map_err(|_| fmt::Error)?)
    }
}

impl core::error::Error for Status {}

/// Reads the `sz_status_t` a C call returns, matching it against [`Status`] rather than
/// transmuting, so a code the header does not list never lands in an enum it isn't a variant of.
///
/// Every status is either checked or dropped through [`StatusCode::infallible`], and a status is
/// dropped only where the call is proven to return `sz_success_k` on every path: a capability query
/// that only reads CPUID, HWCAP or the build's probes, or a CPU dispatch point called with
/// [`Capabilities::CPUS`], which the library clamps to a set always holding [`Capability::Serial`]
/// and so always picks a kernel, where every kernel of that dispatch point returns nothing but
/// `sz_success_k`.
pub(crate) trait StatusCode {
    /// `Ok(())` on success, so a method returning `Result` returns the status through `?`.
    fn check(self) -> Result<(), Status>;

    /// Drops a status the rule above proves to be `sz_success_k`.
    fn infallible(self);
}

impl StatusCode for sz_status_t {
    fn check(self) -> Result<(), Status> {
        match self {
            0 => Ok(()),
            code => Err(Status::from_code(code)),
        }
    }

    fn infallible(self) {}
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
/// assert_eq!(matched, Some(&b"\n"[..]));
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

    /// Returns the range of bytes covered by this span, ending at [`Self::end`].
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
        self.offset..self.end()
    }

    /// Extracts the matched bytes from the source text, or `None` if the span runs past its end.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::stringzilla::IndexSpan;
    ///
    /// let text = b"Hello World";
    /// assert_eq!(IndexSpan::new(6, 5).extract(text), Some(&b"World"[..]));
    /// assert_eq!(IndexSpan::new(6, 6).extract(text), None);
    /// ```
    #[inline]
    pub fn extract<'a>(&self, text: &'a [u8]) -> Option<&'a [u8]> {
        text.get(self.range())
    }

    /// Returns the end offset, `offset + length`, saturating at `usize::MAX`.
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
        self.offset.saturating_add(self.length)
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
    /// Initializes a bit-set to an empty collection, banning all characters.
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
    pub const fn add(&mut self, byte: u8) {
        let word = (byte >> 6) as usize; // Divide by 64.
        let bit = byte & 63; // Remainder modulo 64.
        self.bits[word] |= 1 << bit;
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
            set.add(bytes[index]);
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

/// Returns the semantic version information.
pub fn version() -> SemVer {
    SemVer {
        major: unsafe { sz_version_major() },
        minor: unsafe { sz_version_minor() },
        patch: unsafe { sz_version_patch() },
    }
}

/// One capability, numbered like the C `sz_cap_<capability>_k` bits: each architecture's in a run,
/// ascending by dispatch preference, with the GPU vendors' groups above bit 47.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    Serial = 1 << 0,       // Always: fallback
    Westmere = 1 << 1,     // Intel SSE4.2 + AES-NI
    Goldmont = 1 << 2,     // Intel SHA-NI
    Haswell = 1 << 3,      // Intel AVX2
    Skylake = 1 << 4,      // Intel AVX-512
    IceLake = 1 << 5,      // Intel AVX-512 VBMI2 + VAES
    Neon = 1 << 6,         // Arm NEON
    NeonAes = 1 << 7,      // Arm NEON + AES
    NeonSha = 1 << 8,      // Arm NEON + SHA-256
    Sve = 1 << 9,          // Arm SVE
    Sve2 = 1 << 10,        // Arm SVE2
    Sve2Aes = 1 << 11,     // Arm SVE2 + AES
    Rvv = 1 << 12,         // RISC-V Vector
    RvvCrypto = 1 << 13,   // RISC-V Vector Crypto
    V128 = 1 << 14,        // WASM SIMD128
    V128Relaxed = 1 << 15, // WASM Relaxed SIMD
    LoongsonAsx = 1 << 16, // LoongArch LASX 256-bit SIMD
    PowerVsx = 1 << 17,    // Power VSX 128-bit SIMD
    Cuda = 1 << 48,        // NVIDIA: every CUDA device
    Rocm = 1 << 56,        // AMD: every ROCm device
    Metal = 1 << 60,       // Apple: every Metal device
}

/// Every [`Capability`], in bit order.
const CAPABILITIES: [Capability; 21] = [
    Capability::Serial,
    Capability::Westmere,
    Capability::Goldmont,
    Capability::Haswell,
    Capability::Skylake,
    Capability::IceLake,
    Capability::Neon,
    Capability::NeonAes,
    Capability::NeonSha,
    Capability::Sve,
    Capability::Sve2,
    Capability::Sve2Aes,
    Capability::Rvv,
    Capability::RvvCrypto,
    Capability::V128,
    Capability::V128Relaxed,
    Capability::LoongsonAsx,
    Capability::PowerVsx,
    Capability::Cuda,
    Capability::Rocm,
    Capability::Metal,
];

/// A set of capabilities, printed as comma-separated names like `serial,haswell`.
///
/// # Examples
///
/// ```
/// use stringzilla::sz::{Capabilities, Capability};
///
/// let enabled = Capabilities::cpu_enabled();
/// enabled.configure_thread()?;
/// println!("dispatching to {enabled}");
/// assert!(enabled.contains(Capability::Serial));
/// let narrowed = enabled.without(Capability::Haswell);
/// assert!(!narrowed.contains(Capability::Haswell));
///
/// // Counting a vendor with no devices fails with `Status::MissingGpu`.
/// for ordinal in 0..Capabilities::cuda_count_devices().unwrap_or(0) {
///     println!("CUDA device {ordinal} runs {}", Capabilities::cuda_enabled(ordinal)?);
/// }
/// # Ok::<(), stringzilla::sz::Status>(())
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Capabilities(sz_capability_t);

impl Capabilities {
    /// Every capability, C's `sz_cap_any_k`.
    pub const ANY: Self = Capabilities(sz_capability_t::MAX);

    /// Every CPU capability, the bits below the first GPU vendor's, C's `sz_cap_cpus_k`.
    pub const CPUS: Self = Capabilities(Capability::Cuda as sz_capability_t - 1);

    /// Every GPU capability, which the CPU never detects or enables, C's `sz_cap_gpus_k`.
    pub const GPUS: Self = Capabilities(
        Capability::Cuda as sz_capability_t
            | Capability::Rocm as sz_capability_t
            | Capability::Metal as sz_capability_t,
    );

    /// The raw `sz_capability_t` mask.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Whether `capability` is in this set.
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & capability as sz_capability_t != 0
    }

    /// This set less `capability`, like `enabled.without(Capability::Skylake)`.
    pub const fn without(self, capability: Capability) -> Self {
        Capabilities(self.0 & !(capability as sz_capability_t))
    }

    /// The capabilities in this set, in bit order.
    pub fn iter(self) -> impl Iterator<Item = Capability> {
        CAPABILITIES
            .into_iter()
            .filter(move |&capability| self.contains(capability))
    }

    /// The GPU vendor whose bit tops this set, which C's dispatch reads as the set's group, or none
    /// for the CPU's.
    pub(crate) fn vendor(self) -> Option<Capability> {
        [Capability::Metal, Capability::Rocm, Capability::Cuda]
            .into_iter()
            .find(|&vendor| self.0 >= vendor as sz_capability_t)
    }
}

/// Offset of a non-null match `result` inside the haystack starting at `haystack_pointer`.
#[allow(clippy::cast_sign_loss)] // A match never precedes its haystack, so the offset is non-negative.
pub(crate) fn match_offset<T>(result: *const T, haystack_pointer: *const T) -> usize {
    (unsafe { result.offset_from(haystack_pointer) }) as usize
}

/// One GPU vendor's stream producer and the release of what it made.
type StreamProducers = (
    unsafe extern "C" fn(sz_size_t, *mut *mut c_void) -> sz_status_t,
    unsafe extern "C" fn(*mut c_void) -> sz_status_t,
);

/// The stream producers of the vendor `capabilities` belong to, none for the CPU's.
fn stream_producers(capabilities: Capabilities) -> Option<StreamProducers> {
    match capabilities.vendor() {
        Some(Capability::Cuda) => Some((sz_cuda_stream_init, sz_cuda_stream_free)),
        Some(Capability::Rocm) => Some((sz_rocm_stream_init, sz_rocm_stream_free)),
        Some(Capability::Metal) => Some((sz_metal_stream_init, sz_metal_stream_free)),
        _ => None,
    }
}

impl Capabilities {
    /// Capabilities this CPU supports, whether or not their kernels were compiled in.
    pub fn cpu_detected() -> Self {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_cpu_capabilities_detected(&mut capabilities) }.infallible();
        Capabilities(capabilities)
    }

    /// CPU capabilities whose kernels were compiled in, whether or not this CPU supports them.
    pub fn cpu_compiled() -> Self {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_cpu_capabilities_compiled(&mut capabilities) }.infallible();
        Capabilities(capabilities)
    }

    /// CPU capabilities dispatch runs: [`Capabilities::cpu_detected`] &
    /// [`Capabilities::cpu_compiled`], always with [`Capability::Serial`]. The library clamps every
    /// CPU mask to this set itself, so calls without a mask pass [`Capabilities::CPUS`].
    pub fn cpu_enabled() -> Self {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_cpu_capabilities_enabled(&mut capabilities) }.infallible();
        Capabilities(capabilities)
    }

    /// Sets up the calling thread for the CPU kernels in this set, usually
    /// [`Capabilities::cpu_enabled`]. Call it once per thread before using those kernels; it is
    /// idempotent.
    pub fn configure_thread(self) -> Result<(), Status> {
        unsafe { sz_cpu_configure_thread(self.0) }.check()
    }

    /// How many CUDA devices this process sees, failing with [`Status::MissingGpu`] where the
    /// runtime finds none or this build lacks its kernels.
    pub fn cuda_count_devices() -> Result<usize, Status> {
        let mut count: sz_size_t = 0;
        unsafe { sz_cuda_count_devices(&mut count) }.check()?;
        Ok(count)
    }

    /// Capabilities CUDA device `ordinal` supports, as its runtime numbers them, whether or not
    /// their kernels were compiled in; fails with [`Status::MissingGpu`] past the last device.
    pub fn cuda_detected(ordinal: usize) -> Result<Self, Status> {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_cuda_capabilities_detected(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// CUDA capabilities whose kernels were compiled in, whether or not any device supports them.
    pub fn cuda_compiled() -> Self {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_cuda_capabilities_compiled(&mut capabilities) }.infallible();
        Capabilities(capabilities)
    }

    /// Capabilities kernels run with on CUDA device `ordinal`: [`Capabilities::cuda_detected`] &
    /// [`Capabilities::cuda_compiled`]. The mask names the vendor, and a stream the device.
    pub fn cuda_enabled(ordinal: usize) -> Result<Self, Status> {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_cuda_capabilities_enabled(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// [`Capabilities::cuda_count_devices`], for ROCm.
    pub fn rocm_count_devices() -> Result<usize, Status> {
        let mut count: sz_size_t = 0;
        unsafe { sz_rocm_count_devices(&mut count) }.check()?;
        Ok(count)
    }

    /// [`Capabilities::cuda_detected`], for ROCm.
    pub fn rocm_detected(ordinal: usize) -> Result<Self, Status> {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_rocm_capabilities_detected(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// [`Capabilities::cuda_compiled`], for ROCm.
    pub fn rocm_compiled() -> Self {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_rocm_capabilities_compiled(&mut capabilities) }.infallible();
        Capabilities(capabilities)
    }

    /// [`Capabilities::cuda_enabled`], for ROCm.
    pub fn rocm_enabled(ordinal: usize) -> Result<Self, Status> {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_rocm_capabilities_enabled(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// [`Capabilities::cuda_count_devices`], for Metal.
    pub fn metal_count_devices() -> Result<usize, Status> {
        let mut count: sz_size_t = 0;
        unsafe { sz_metal_count_devices(&mut count) }.check()?;
        Ok(count)
    }

    /// [`Capabilities::cuda_detected`], for Metal.
    pub fn metal_detected(ordinal: usize) -> Result<Self, Status> {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_metal_capabilities_detected(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// [`Capabilities::cuda_compiled`], for Metal.
    pub fn metal_compiled() -> Self {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_metal_capabilities_compiled(&mut capabilities) }.infallible();
        Capabilities(capabilities)
    }

    /// [`Capabilities::cuda_enabled`], for Metal.
    pub fn metal_enabled(ordinal: usize) -> Result<Self, Status> {
        let mut capabilities: sz_capability_t = 0;
        unsafe { sz_metal_capabilities_enabled(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }
}

/// A stream of one capability group's device, where engines, [`Sequence`] copies and
/// [`UnifiedAllocator`] blocks queue work: a vendor's default when null, the CPU's always null.
///
/// # Examples
///
/// ```
/// use stringzilla::sz::{Capabilities, Stream};
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// assert!(cpu.as_raw().is_null());
///
/// // A vendor with no devices fails with `Status::MissingGpu`.
/// for ordinal in 0..Capabilities::cuda_count_devices().unwrap_or(0) {
///     let stream = Stream::new(Capabilities::cuda_enabled(ordinal)?, ordinal)?;
///     stream.synchronize()?;
/// }
/// # Ok::<(), stringzilla::sz::Status>(())
/// ```
#[derive(Debug)]
pub struct Stream {
    capabilities: Capabilities,
    handle: *mut c_void,
    owned: bool,
}

impl Stream {
    /// The default stream of the default device of the group `capabilities` name.
    pub fn default(capabilities: Capabilities) -> Self {
        Stream {
            capabilities,
            handle: core::ptr::null_mut(),
            owned: false,
        }
    }

    /// A new stream on device `ordinal` of the vendor `capabilities` name, a `cudaStream_t`, a
    /// `hipStream_t` or an `id<MTLCommandQueue>`, freed on drop. The CPU gets [`Stream::default`].
    ///
    /// Fails with [`Status::MissingGpu`] past the last device or where this build lacks the vendor.
    pub fn new(capabilities: Capabilities, ordinal: usize) -> Result<Self, Status> {
        let Some((init, _)) = stream_producers(capabilities) else {
            return Ok(Self::default(capabilities));
        };
        let mut handle = core::ptr::null_mut();
        unsafe { init(ordinal, &mut handle) }.check()?;
        Ok(Stream {
            capabilities,
            handle,
            owned: true,
        })
    }

    /// Borrows a stream the caller made and keeps owning, which is never freed here.
    ///
    /// # Safety
    ///
    /// `handle` must be null or a live stream of the vendor `capabilities` name, a `cudaStream_t`,
    /// a `hipStream_t` or an `id<MTLCommandQueue>`, null on the CPU, and outlive the result.
    pub unsafe fn from_raw(capabilities: Capabilities, handle: *mut c_void) -> Self {
        Stream {
            capabilities,
            handle,
            owned: false,
        }
    }

    /// The capabilities every consumer of this stream dispatches with.
    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    /// The vendor's handle, null for a default stream.
    pub fn as_raw(&self) -> *mut c_void {
        self.handle
    }

    /// Waits for everything queued on this stream, after which whatever its rounds wrote is
    /// readable from the host. The CPU has nothing to wait for.
    pub fn synchronize(&self) -> Result<(), Status> {
        unsafe { sz_stream_synchronize_best(self.capabilities.0, self.handle) }.check()
    }

    /// Runs `body`, whose engine verbs queue on this stream, then joins the stream, even when
    /// `body` fails or unwinds, and reports the first failure, the body's or the join's.
    ///
    /// A verb borrows its engine, inputs and outputs for the whole scope, so none of them can be
    /// touched before the join, and an engine runs one verb per scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::{Capabilities, LevenshteinEngine, LevenshteinSymbol, Stream};
    ///
    /// let stream = Stream::default(Capabilities::cpu_enabled());
    /// let mut engine = LevenshteinEngine::new(&["kitten"], LevenshteinSymbol::Bytes, &stream)?;
    /// let mut distances = [0usize; 1];
    /// stream.scope(|scope| engine.distances(scope, &["sitting"], &mut distances, 1))?;
    /// assert_eq!(distances[0], 3);
    /// # Ok::<(), stringzilla::sz::Status>(())
    /// ```
    pub fn scope<'env, Output>(
        &'env self,
        body: impl for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> Result<Output, Status>,
    ) -> Result<Output, Status> {
        let join_on_unwind = JoinOnUnwind(self);
        let output = body(&Scope {
            stream: self,
            scope: PhantomData,
        });
        core::mem::forget(join_on_unwind);
        let joined = self.synchronize();
        let output = output?;
        joined.map(|()| output)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Some((_, free)) = stream_producers(self.capabilities).filter(|_| self.owned) {
            // A drop has nowhere to report a failed release.
            let _ = unsafe { free(self.handle) }.check();
        }
    }
}

/// Joins a stream when dropped, which a [`Stream::scope`] body lets happen only by unwinding.
struct JoinOnUnwind<'stream>(&'stream Stream);

impl Drop for JoinOnUnwind<'_> {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}

/// The span of a [`Stream::scope`] body, which every engine verb takes to queue on that stream.
///
/// A verb borrows what it reads and writes for `'scope`, which ends only once the stream is joined.
pub struct Scope<'scope, 'env: 'scope> {
    stream: &'env Stream,
    scope: PhantomData<&'scope mut &'scope ()>,
}

impl Scope<'_, '_> {
    /// The stream an engine of `capability` queues on, refusing one of another group, whose handle
    /// that engine's runtime cannot read, with [`Status::DeviceMemoryMismatch`].
    pub(crate) fn stream_for(&self, capability: Capabilities) -> Result<&Stream, Status> {
        if capability.vendor() != self.stream.capabilities.vendor() {
            return Err(Status::DeviceMemoryMismatch);
        }
        Ok(self.stream)
    }
}

/// A batch of strings an engine's verb reads: a slice or an array of byte strings, read where it
/// lies through accessors over it, or a [`Sequence`] tape.
///
/// A GPU engine reads its batch in place, so it takes a [`Sequence`] in memory its device reaches;
/// given host strings, it refuses the round with [`Status::DeviceMemoryMismatch`].
///
/// # Safety
///
/// The sequence an implementation hands `call` reads `count` strings that live as long as `self`.
pub unsafe trait Strings {
    /// Hands `call` the `sz_sequence_t` an engine of the group of `stream` reads the batch through.
    fn with_sequence<Return>(
        &self,
        stream: &Stream,
        call: impl FnOnce(&_SzSequence) -> Return,
    ) -> Result<Return, Status>;
}

unsafe impl<Text: AsRef<[u8]>> Strings for [Text] {
    fn with_sequence<Return>(
        &self,
        _stream: &Stream,
        call: impl FnOnce(&_SzSequence) -> Return,
    ) -> Result<Return, Status> {
        Ok(with_sequence(self, call))
    }
}

unsafe impl<Text: AsRef<[u8]>, const COUNT: usize> Strings for [Text; COUNT] {
    fn with_sequence<Return>(
        &self,
        _stream: &Stream,
        call: impl FnOnce(&_SzSequence) -> Return,
    ) -> Result<Return, Status> {
        Ok(with_sequence(self.as_slice(), call))
    }
}

/// A batch of strings copied into one tape, `count + 1` offsets counted from the block's own start
/// and then the bytes, in a single block of a [`UnifiedAllocator`], which an engine reads in place.
///
/// It is C's `sz_sequence_realloc_best`, and the tape is freed when the sequence drops. A host
/// engine reads a tape copied for any capabilities, and a GPU engine one its device reaches,
/// refusing any other with [`Status::DeviceMemoryMismatch`].
///
/// # Examples
///
/// ```
/// use stringzilla::sz::{Capabilities, LevenshteinEngine, LevenshteinSymbol, Sequence, Stream, UnifiedAllocator};
///
/// let stream = Stream::default(Capabilities::cpu_enabled());
/// let unified = UnifiedAllocator::new(&stream);
/// let candidates = Sequence::copy(&["sitting", "kitten"], &unified, &stream)?;
/// let mut engine = LevenshteinEngine::new(&["kitten"], LevenshteinSymbol::Bytes, &stream)?;
/// let mut distances = [0usize; 2];
/// stream.scope(|scope| engine.distances(scope, &candidates, &mut distances, 2))?;
/// assert_eq!(distances, [3, 0]);
/// # Ok::<(), stringzilla::sz::Status>(())
/// ```
pub struct Sequence {
    sequence: _SzSequence,
    capabilities: Capabilities,
    allocator: _SzMemoryAllocator,
    allocated_bytes: usize,
}

impl Sequence {
    /// Copies `texts` on the host into one tape from `allocator`, pointed at with the accessors
    /// engines of the group of `stream` read it through, and queues only its migration there.
    ///
    /// An `allocator` of another group than `stream` is refused with
    /// [`Status::DeviceMemoryMismatch`].
    pub fn copy<Text: AsRef<[u8]>>(
        texts: &[Text],
        allocator: &UnifiedAllocator,
        stream: &Stream,
    ) -> Result<Self, Status> {
        if allocator.capabilities().vendor() != stream.capabilities.vendor() {
            return Err(Status::DeviceMemoryMismatch);
        }
        let mut allocator = allocator.c_allocator()?;
        let mut sequence = MaybeUninit::<_SzSequence>::uninit();
        let mut allocated_bytes = 0;
        with_sequence(texts, |source| unsafe {
            sz_sequence_realloc_best(
                sequence.as_mut_ptr(),
                source,
                &mut allocator,
                &mut allocated_bytes,
                stream.capabilities.0,
                stream.handle,
            )
        })
        .check()?;
        Ok(Sequence {
            sequence: unsafe { sequence.assume_init() },
            capabilities: stream.capabilities,
            allocator,
            allocated_bytes,
        })
    }

    /// Strings the tape holds.
    pub fn len(&self) -> usize {
        self.sequence.count
    }

    /// Whether the tape holds no strings.
    pub fn is_empty(&self) -> bool {
        self.sequence.count == 0
    }
}

unsafe impl Strings for Sequence {
    fn with_sequence<Return>(
        &self,
        stream: &Stream,
        call: impl FnOnce(&_SzSequence) -> Return,
    ) -> Result<Return, Status> {
        // The host reads a tape of any group in place, while a GPU's own accessors run only there.
        if stream.capabilities.vendor().is_none() {
            let host = _SzSequence {
                handle: self.sequence.handle,
                count: self.sequence.count,
                get_start: Some(tape_start),
                get_length: Some(tape_length),
            };
            return Ok(call(&host));
        }
        // A tape of another group may hold another device's accessors, which no host may call.
        if self.capabilities.vendor() != stream.capabilities.vendor() {
            return Err(Status::DeviceMemoryMismatch);
        }
        let mut allocator = self.allocator;
        let mut sequence = MaybeUninit::<_SzSequence>::uninit();
        let mut allocated_bytes = 0;
        unsafe {
            sz_sequence_realloc_best(
                sequence.as_mut_ptr(),
                &self.sequence,
                &mut allocator,
                &mut allocated_bytes,
                stream.capabilities.0,
                stream.handle,
            )
        }
        .check()?;
        let sequence = unsafe { sequence.assume_init() };
        // A tape the device reaches is only re-pointed, so one it would have to copy is refused.
        if allocated_bytes != 0 {
            if let Some(free) = allocator.free {
                let tape = sequence.handle as *mut c_void;
                unsafe { free(tape, allocated_bytes, allocator.handle, stream.handle) };
            }
            return Err(Status::DeviceMemoryMismatch);
        }
        Ok(call(&sequence))
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        if self.allocated_bytes == 0 {
            return;
        }
        if let Some(free) = self.allocator.free {
            let tape = self.sequence.handle as *mut c_void;
            unsafe { free(tape, self.allocated_bytes, self.allocator.handle, core::ptr::null_mut()) };
        }
    }
}

/// String `index` of a tape, whose `count + 1` offsets count from the block's own start.
unsafe extern "C" fn tape_start(handle: *const c_void, index: usize) -> *const c_void {
    let offsets = handle as *const u64;
    handle.cast::<u8>().add(*offsets.add(index) as usize).cast()
}

/// Bytes of string `index` of a tape.
unsafe extern "C" fn tape_length(handle: *const c_void, index: usize) -> usize {
    let offsets = handle as *const u64;
    (*offsets.add(index + 1) - *offsets.add(index)) as usize
}

/// Memory the host and the devices of one capability group both address, for the tapes and the
/// outputs of their engines.
///
/// It is the C library's `sz_allocator_init_unified_best` behind the standard
/// [`Allocator`](core::alloc::Allocator), so `Vec::new_in` and every other allocator-aware
/// collection places its contents there. Every block lands on the device of the stream it borrows,
/// and a block aligned less strictly than its layout asks is refused.
///
/// # Examples
///
/// ```no_run
/// use stringzilla::sz::{Capabilities, LevenshteinEngine, LevenshteinSymbol, Sequence, Stream, UnifiedAllocator};
///
/// let stream = Stream::new(Capabilities::cuda_enabled(0)?, 0)?;
/// let unified = UnifiedAllocator::new(&stream);
/// let candidates = Sequence::copy(&["sitting", "kitten"], &unified, &stream)?;
/// let mut distances = Vec::new_in(unified);
/// distances.resize(2, 0usize);
///
/// let mut engine = LevenshteinEngine::new(&["kitten"], LevenshteinSymbol::Bytes, &stream)?;
/// stream.scope(|scope| engine.distances(scope, &candidates, &mut distances, 2))?;
/// assert_eq!(distances[..], [3, 0]);
/// # Ok::<(), stringzilla::sz::Status>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct UnifiedAllocator<'stream> {
    stream: &'stream Stream,
}

impl<'stream> UnifiedAllocator<'stream> {
    /// Unified memory on the device of `stream`, the host heap for a CPU stream.
    pub const fn new(stream: &'stream Stream) -> Self {
        UnifiedAllocator { stream }
    }

    /// The capabilities whose memory every block comes from.
    pub fn capabilities(&self) -> Capabilities {
        self.stream.capabilities
    }

    fn c_allocator(&self) -> Result<_SzMemoryAllocator, Status> {
        let mut allocator = _SzMemoryAllocator {
            allocate: None,
            free: None,
            handle: core::ptr::null_mut(),
        };
        unsafe { sz_allocator_init_unified_best(&mut allocator, self.stream.capabilities.0) }.check()?;
        Ok(allocator)
    }
}

unsafe impl core::alloc::Allocator for UnifiedAllocator<'_> {
    fn allocate(&self, layout: core::alloc::Layout) -> Result<core::ptr::NonNull<[u8]>, core::alloc::AllocError> {
        use core::ptr::NonNull;

        if layout.size() == 0 {
            let dangling =
                NonNull::new(core::ptr::without_provenance_mut::<u8>(layout.align())).ok_or(core::alloc::AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(dangling, 0));
        }
        let allocator = self.c_allocator().map_err(|_| core::alloc::AllocError)?;
        let (allocate, free) = allocator.allocate.zip(allocator.free).ok_or(core::alloc::AllocError)?;
        let block = unsafe { allocate(layout.size(), allocator.handle, self.stream.handle) } as *mut u8;
        let block = NonNull::new(block).ok_or(core::alloc::AllocError)?;
        // Each group's runtime promises its own alignment, so a stricter layout is checked here.
        if block.as_ptr().addr() % layout.align() != 0 {
            unsafe {
                free(
                    block.as_ptr() as *mut c_void,
                    layout.size(),
                    allocator.handle,
                    self.stream.handle,
                )
            };
            return Err(core::alloc::AllocError);
        }
        Ok(NonNull::slice_from_raw_parts(block, layout.size()))
    }

    unsafe fn deallocate(&self, block: core::ptr::NonNull<u8>, layout: core::alloc::Layout) {
        if layout.size() == 0 {
            return;
        }
        if let Ok(allocator) = self.c_allocator() {
            if let Some(free) = allocator.free {
                unsafe {
                    free(
                        block.as_ptr() as *mut c_void,
                        layout.size(),
                        allocator.handle,
                        self.stream.handle,
                    )
                };
            }
        }
    }
}

impl From<Capability> for Capabilities {
    fn from(capability: Capability) -> Self {
        Capabilities(capability as sz_capability_t)
    }
}

impl BitOr for Capability {
    type Output = Capabilities;
    fn bitor(self, other: Self) -> Capabilities {
        Capabilities(self as sz_capability_t | other as sz_capability_t)
    }
}

impl BitOr<Capability> for Capabilities {
    type Output = Self;
    fn bitor(self, capability: Capability) -> Self {
        Capabilities(self.0 | capability as sz_capability_t)
    }
}

impl BitOr for Capabilities {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Capabilities(self.0 | other.0)
    }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buffer = [0u8; 256]; // STRINGZILLA_CAPABILITIES_NAME_CAPACITY
        let length = unsafe { sz_capabilities_name(self.0, buffer.as_mut_ptr().cast(), buffer.len()) };
        f.pad(core::str::from_utf8(&buffer[..length]).map_err(|_| fmt::Error)?)
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&Capabilities::from(*self), f)
    }
}

/// Compile-time policy for whether a split keeps or drops empty, zero-length segments.
///
/// A named marker type rather than a raw `bool`, so the choice is branchless and readable
/// at call sites.
pub trait EmptySegments {
    /// Whether zero-length segments are skipped.
    const SKIP: bool;
}
/// Keep empty segments, the default.
pub struct KeepEmpty;
impl EmptySegments for KeepEmpty {
    const SKIP: bool = false;
}
/// Drop empty segments, via `.skip_empty()`.
pub struct SkipEmpty;
impl EmptySegments for SkipEmpty {
    const SKIP: bool = true;
}

/// Compile-time policy for whether overlapping matches are reported - a named marker, not
/// a raw `bool`.
pub trait Overlaps {
    /// Whether overlapping matches are included.
    const OVERLAP: bool;
}
/// Report only non-overlapping matches, the default, like `str::matches`.
pub struct NonOverlapping;
impl Overlaps for NonOverlapping {
    const OVERLAP: bool = false;
}
/// Report overlapping matches too, via `.overlapping()`.
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
    extern crate alloc;
    use alloc::string::ToString;

    use super::*;
    use crate::sz;

    #[test]
    fn metadata() {
        let names = Capabilities::cpu_enabled().to_string();
        assert!(names.split(',').any(|name| name == "serial"), "{names}");
    }

    #[test]
    fn capability_names_match_the_c_table() {
        let names: [&str; CAPABILITIES.len()] = [
            "serial",
            "westmere",
            "goldmont",
            "haswell",
            "skylake",
            "icelake",
            "neon",
            "neonaes",
            "neonsha",
            "sve",
            "sve2",
            "sve2aes",
            "rvv",
            "rvvcrypto",
            "v128",
            "v128relaxed",
            "loongsonasx",
            "powervsx",
            "cuda",
            "rocm",
            "metal",
        ];
        let mut previous = 0;
        for (capability, name) in CAPABILITIES.into_iter().zip(names) {
            assert!(capability as u64 > previous && (capability as u64).is_power_of_two());
            assert_eq!(capability.to_string(), name);
            previous = capability as u64;
        }
        let gpus = Capabilities::GPUS
            .iter()
            .fold(Capabilities::default(), |set, gpu| set | gpu);
        assert_eq!(gpus, Capabilities::GPUS);
        assert_eq!(Capabilities::CPUS.bits() & Capabilities::GPUS.bits(), 0);
    }

    #[test]
    fn producers_report_clamped_capabilities() {
        let enabled = Capabilities::cpu_enabled();
        assert!(enabled.contains(Capability::Serial));
        let detected = Capabilities::cpu_detected();
        assert_eq!(enabled.bits() & !(detected.bits() | Capability::Serial as u64), 0);
        assert_eq!(enabled.bits() & !Capabilities::cpu_compiled().bits(), 0);
        assert_eq!(enabled.configure_thread(), Ok(()));
        assert!(Stream::new(enabled, 1).unwrap().as_raw().is_null());

        type Producers = (
            fn() -> Result<usize, Status>,
            fn(usize) -> Result<Capabilities, Status>,
            fn() -> Capabilities,
            fn(usize) -> Result<Capabilities, Status>,
            Capability,
        );
        let vendors: [Producers; 3] = [
            (
                Capabilities::cuda_count_devices,
                Capabilities::cuda_detected,
                Capabilities::cuda_compiled,
                Capabilities::cuda_enabled,
                Capability::Cuda,
            ),
            (
                Capabilities::rocm_count_devices,
                Capabilities::rocm_detected,
                Capabilities::rocm_compiled,
                Capabilities::rocm_enabled,
                Capability::Rocm,
            ),
            (
                Capabilities::metal_count_devices,
                Capabilities::metal_detected,
                Capabilities::metal_compiled,
                Capabilities::metal_enabled,
                Capability::Metal,
            ),
        ];
        for (count_devices, detected, compiled, enabled, vendor) in vendors {
            let count = match count_devices() {
                Ok(count) => count,
                Err(status) => {
                    assert_eq!(status, Status::MissingGpu);
                    assert_eq!(enabled(0), Err(Status::MissingGpu));
                    assert!(Stream::new(vendor.into(), 0).is_err());
                    continue;
                }
            };
            assert!(count > 0);
            assert_eq!(enabled(count), Err(Status::MissingGpu));
            assert!(Stream::new(vendor.into(), count).is_err());
            for ordinal in 0..count {
                let supported = detected(ordinal).unwrap();
                assert_eq!(supported.bits() & Capabilities::CPUS.bits(), 0);
                assert_eq!(enabled(ordinal).unwrap().bits(), supported.bits() & compiled().bits());
                let stream = Stream::new(enabled(ordinal).unwrap(), ordinal).unwrap();
                assert!(!stream.as_raw().is_null());
                assert_eq!(stream.synchronize(), Ok(()));
            }
        }
    }

    #[test]
    fn enabled_is_what_the_library_widened_at_load() {
        let available = Capabilities::cpu_detected().bits() & Capabilities::cpu_compiled().bits();
        assert_eq!(Capabilities::cpu_enabled().bits(), available);
    }

    #[test]
    fn statuses_read_their_c_codes() {
        let (success, missing_kernel): (sz_status_t, sz_status_t) = (0, -20);
        assert_eq!(success.check(), Ok(()));
        assert_eq!(missing_kernel.check(), Err(Status::MissingKernel));
        assert_eq!(Status::MissingKernel.to_string(), "no kernel for these capabilities");
        // A code the header does not list, like a positive caveat, is a failure named as such.
        let unlisted: [sz_status_t; 3] = [1, -11, -22];
        for code in unlisted {
            assert_eq!(code.check(), Err(Status::Unrecognized));
        }
        assert_eq!(Status::Unrecognized.to_string(), "an unrecognized status");
    }

    #[test]
    fn const_apis() {
        // Each constant is built at compile time, so dropping a `const fn` breaks the build,
        // not this run.
        const SPAN: sz::IndexSpan = sz::IndexSpan::new(6, 5);
        const WHITESPACE: sz::Byteset = sz::Byteset::from_bytes(b" \t\r\n");
        const TOP_TWO_DESCENDING: sz::ArgsortOptions = sz::ArgsortOptions {
            reverse: false,
            uncased: false,
            top: None,
        }
        .reversed()
        .top(2);

        assert_eq!(SPAN.extract(b"Hello World"), Some(&b"World"[..]));
        assert_eq!(sz::IndexSpan::new(usize::MAX, 2).extract(b"Hello World"), None);
        assert_eq!(sz::find_byteset("ab cd", WHITESPACE), Some(2));

        let fruits = ["banana", "apple", "cherry"];
        let mut order = [0; 3];
        sz::argsort(&fruits, &mut order, TOP_TWO_DESCENDING).expect("argsort failed");
        assert_eq!(fruits[order[0]], "cherry");
    }

    /// A tape copied for the CPU reads like the slice it came from, its outputs in the same memory.
    #[test]
    fn cpu_tapes_read_like_their_slices() {
        extern crate alloc;
        use alloc::vec::Vec;

        let cpu = Stream::default(Capabilities::cpu_enabled());
        let unified = UnifiedAllocator::new(&cpu);
        let words = ["sitting", "sunday", "", "kitten"];
        let tape = Sequence::copy(&words, &unified, &cpu).unwrap();
        assert_eq!(tape.len(), words.len());

        let mut engine = sz::LevenshteinEngine::new(&["kitten"], sz::LevenshteinSymbol::Bytes, &cpu).unwrap();
        let mut from_slice = [0usize; 4];
        let mut from_tape = Vec::new_in(unified);
        from_tape.resize(words.len(), usize::MAX);
        cpu.scope(|scope| engine.distances(scope, &words, &mut from_slice, words.len()))
            .unwrap();
        cpu.scope(|scope| engine.distances(scope, &tape, &mut from_tape, words.len()))
            .unwrap();
        assert_eq!(from_tape[..], from_slice[..]);
    }

    /// A verb refuses a stream of another group before queuing anything, and its scope reports that
    /// ahead of its own join, which fails where this build lacks the vendor.
    #[test]
    fn scopes_refuse_engines_of_another_group() {
        let cpu = Stream::default(Capabilities::cpu_enabled());
        let mut engine = sz::LevenshteinEngine::new(&["kitten"], sz::LevenshteinSymbol::Bytes, &cpu).unwrap();
        let mut distances = [0usize; 1];
        let gpu = Stream::default(Capability::Metal.into());
        let refused = gpu.scope(|scope| engine.distances(scope, &["sitting"], &mut distances, 1));
        assert_eq!(refused, Err(Status::DeviceMemoryMismatch));
    }

    /// A CUDA engine reads a tape in unified memory and agrees with the host engine, which reads
    /// the same tape, while host-resident strings are refused rather than read.
    #[cfg(feature = "cuda")]
    #[test]
    fn unified_batches_reach_a_cuda_engine() {
        extern crate alloc;
        use alloc::vec::Vec;

        let Ok(capabilities) = Capabilities::cuda_enabled(0) else {
            return;
        };
        let gpu = Stream::new(capabilities, 0).unwrap();
        let unified = UnifiedAllocator::new(&gpu);
        let queries = ["kitten", "saturday", "Straße"];
        let words = ["sitting", "sunday", "strasse", "", "kitten"];
        let candidates = Sequence::copy(&words, &unified, &gpu).unwrap();
        let mut device_distances = Vec::new_in(unified);
        device_distances.resize(queries.len() * words.len(), usize::MAX);

        let cpu = Stream::default(Capabilities::cpu_enabled());
        let mut host = sz::LevenshteinEngine::new(&queries, sz::LevenshteinSymbol::Bytes, &cpu).unwrap();
        let mut host_distances = [0usize; 15];
        let mut tape_distances = [0usize; 15];
        cpu.scope(|scope| host.distances(scope, &words, &mut host_distances, words.len()))
            .unwrap();
        cpu.scope(|scope| host.distances(scope, &candidates, &mut tape_distances, words.len()))
            .unwrap();
        assert_eq!(tape_distances, host_distances);

        let mut device = sz::LevenshteinEngine::new(&queries, sz::LevenshteinSymbol::Bytes, &gpu).unwrap();
        gpu.scope(|scope| device.distances(scope, &candidates, &mut device_distances, words.len()))
            .unwrap();
        assert_eq!(device_distances[..], host_distances[..]);
        assert_eq!(
            gpu.scope(|scope| device.distances(scope, &words, &mut device_distances, words.len())),
            Err(Status::DeviceMemoryMismatch)
        );
    }
}
