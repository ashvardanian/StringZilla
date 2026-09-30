//! Shared value types, status codes, devices and their capabilities, and library introspection.
//!
//! A [`Device`] is the host CPU or one GPU, and reports its [`Capabilities`] along two axes that
//! have nothing to do with each other, plus the set dispatch uses:
//!
//! - [`Device::capabilities_detected`]: what the device can execute, from CPUID / `getauxval` /
//!   HWCAP on the CPU and from the runtime on a GPU
//! - [`Device::capabilities_compiled`]: what this binary contains, from the build's probes
//! - [`Device::capabilities_enabled`]: what dispatch uses — both axes at once, unless narrowed by
//!   [`Device::capabilities_enable`] on the CPU
//!
//! Reach for [`Device::capabilities_enabled`] unless you specifically mean one of the raw axes.
//!
//! File: rust/stringzilla/types.rs
//! Author: Ash Vardanian

use super::*;
use core::ffi::{c_void, CStr};
use core::fmt;
use core::ops::BitOr;
use core::sync::atomic::{AtomicU64, Ordering};

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
/// [`enabled_cpu_capabilities_mask`], which always holds [`Capability::Serial`] and so always picks
/// a kernel, where every kernel of that dispatch point returns nothing but `sz_success_k`.
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
    Icelake = 1 << 5,      // Intel AVX-512 VBMI2 + VAES
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
    Capability::Icelake,
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
/// use stringzilla::sz::{Capability, Device};
///
/// let enabled = Device::cpu().capabilities_enabled()?;
/// println!("dispatching to {enabled}");
/// assert!(enabled.contains(Capability::Serial));
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

    /// Every GPU capability, which the CPU never detects or enables, C's `sz_cap_devices_k`.
    pub const DEVICES: Self = Capabilities(
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
}

/// Offset of a non-null match `result` inside the haystack starting at `haystack_pointer`.
#[allow(clippy::cast_sign_loss)] // A match never precedes its haystack, so the offset is non-negative.
pub(crate) fn match_offset<T>(result: *const T, haystack_pointer: *const T) -> usize {
    (unsafe { result.offset_from(haystack_pointer) }) as usize
}

/// The CPU capability mask every call passes, zero until first read.
static ENABLED: AtomicU64 = AtomicU64::new(0);

/// The raw `sz_capability_t` mask every call passes, [`Device::capabilities_enabled`] of the CPU.
pub(crate) fn enabled_cpu_capabilities_mask() -> sz_capability_t {
    let mask = ENABLED.load(Ordering::Relaxed);
    if mask != 0 {
        return mask;
    }
    let mut available = 0;
    unsafe { sz_cpu_capabilities_enabled(&mut available) }.infallible();
    match ENABLED.compare_exchange(0, available, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => available,
        Err(current) => current,
    }
}

/// Which runtime a device belongs to, as the `sz_<kind>_*` C functions name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
}

/// One device StringZilla can run kernels on: the host CPU, or a GPU by its runtime's ordinal.
///
/// # Examples
///
/// ```
/// use stringzilla::sz::{Capability, Device, DeviceKind};
///
/// let cpu = Device::cpu();
/// let enabled = cpu.capabilities_enabled()?;
/// cpu.configure_thread(enabled)?;
/// let narrowed = cpu.capabilities_enable(enabled.without(Capability::Haswell))?;
/// assert!(!narrowed.contains(Capability::Haswell));
/// cpu.capabilities_enable(enabled)?;
///
/// // Counting a vendor with no devices fails with `Status::MissingGpu`.
/// for ordinal in 0..Device::count(DeviceKind::Cuda).unwrap_or(0) {
///     let gpu = Device::new(DeviceKind::Cuda, ordinal)?;
///     println!("CUDA device {ordinal} runs {}", gpu.capabilities_enabled()?);
/// }
/// # Ok::<(), stringzilla::sz::Status>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Device {
    kind: DeviceKind,
    ordinal: usize,
}

impl Device {
    /// The host CPU, the one device every process has.
    pub const fn cpu() -> Self {
        Device {
            kind: DeviceKind::Cpu,
            ordinal: 0,
        }
    }

    /// How many devices of `kind` this process sees: one CPU, or at least one GPU, failing with
    /// [`Status::MissingGpu`] where the runtime finds none or this build lacks its kernels.
    pub fn count(kind: DeviceKind) -> Result<usize, Status> {
        let count_devices = match kind {
            DeviceKind::Cpu => return Ok(1),
            DeviceKind::Cuda => sz_cuda_count_devices,
            DeviceKind::Rocm => sz_rocm_count_devices,
            DeviceKind::Metal => sz_metal_count_devices,
        };
        let mut count: sz_size_t = 0;
        unsafe { count_devices(&mut count) }.check()?;
        Ok(count)
    }

    /// Device `ordinal` of `kind`, as its runtime numbers them; the CPU is ordinal zero. Fails with
    /// [`Status::MissingGpu`] past the last device, or for a vendor with none.
    pub fn new(kind: DeviceKind, ordinal: usize) -> Result<Self, Status> {
        if ordinal >= Self::count(kind)? {
            return Err(Status::MissingGpu);
        }
        Ok(Device { kind, ordinal })
    }

    /// Which runtime this device belongs to.
    pub const fn kind(&self) -> DeviceKind {
        self.kind
    }

    /// This device's ordinal in its runtime, always zero for the CPU.
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Capabilities this device supports, whether or not their kernels were compiled in.
    pub fn capabilities_detected(&self) -> Result<Capabilities, Status> {
        let mut mask: sz_capability_t = 0;
        let status = unsafe {
            match self.kind {
                DeviceKind::Cpu => sz_cpu_capabilities_detected(&mut mask),
                DeviceKind::Cuda => sz_cuda_capabilities_detected(self.ordinal, &mut mask),
                DeviceKind::Rocm => sz_rocm_capabilities_detected(self.ordinal, &mut mask),
                DeviceKind::Metal => sz_metal_capabilities_detected(self.ordinal, &mut mask),
            }
        };
        status.check()?;
        Ok(Capabilities(mask))
    }

    /// Capabilities of this device's kind whose kernels were compiled in, whether or not this
    /// device supports them.
    pub fn capabilities_compiled(&self) -> Capabilities {
        let mut mask: sz_capability_t = 0;
        unsafe {
            match self.kind {
                DeviceKind::Cpu => sz_cpu_capabilities_compiled(&mut mask),
                DeviceKind::Cuda => sz_cuda_capabilities_compiled(&mut mask),
                DeviceKind::Rocm => sz_rocm_capabilities_compiled(&mut mask),
                DeviceKind::Metal => sz_metal_capabilities_compiled(&mut mask),
            }
        }
        .infallible();
        Capabilities(mask)
    }

    /// Capabilities kernels run with: [`Device::capabilities_detected`] &
    /// [`Device::capabilities_compiled`], on the CPU narrowed by [`Device::capabilities_enable`]
    /// and always with [`Capability::Serial`].
    pub fn capabilities_enabled(&self) -> Result<Capabilities, Status> {
        let mut mask: sz_capability_t = 0;
        let status = unsafe {
            match self.kind {
                DeviceKind::Cpu => return Ok(Capabilities(enabled_cpu_capabilities_mask())),
                DeviceKind::Cuda => sz_cuda_capabilities_enabled(self.ordinal, &mut mask),
                DeviceKind::Rocm => sz_rocm_capabilities_enabled(self.ordinal, &mut mask),
                DeviceKind::Metal => sz_metal_capabilities_enabled(self.ordinal, &mut mask),
            }
        };
        status.check()?;
        Ok(Capabilities(mask))
    }

    /// Makes `wanted`, clamped to [`Device::capabilities_detected`] &
    /// [`Device::capabilities_compiled`] and with [`Capability::Serial`] kept, what every CPU call
    /// passes, and returns what stuck.
    ///
    /// This is the one piece of process state the crate keeps: calls take no mask, so every thread
    /// dispatches with the set last enabled here. Engines keep the capability they were built with.
    /// It applies to the CPU only, and fails with [`Status::MissingKernel`] on a GPU.
    pub fn capabilities_enable(&self, wanted: Capabilities) -> Result<Capabilities, Status> {
        if self.kind != DeviceKind::Cpu {
            return Err(Status::MissingKernel);
        }
        let mut available = 0;
        unsafe { sz_cpu_capabilities_enabled(&mut available) }.infallible();
        let mask = wanted.0 & available | Capability::Serial as sz_capability_t;
        ENABLED.store(mask, Ordering::Relaxed);
        Ok(Capabilities(mask))
    }

    /// Sets up the calling thread for the kernels in `capabilities`, usually
    /// [`Device::capabilities_enabled`]. Call it once per thread before using those kernels; it is
    /// idempotent. It applies to the CPU only, and fails with [`Status::MissingKernel`] on a GPU.
    pub fn configure_thread(&self, capabilities: Capabilities) -> Result<(), Status> {
        match self.kind {
            DeviceKind::Cpu => unsafe { sz_cpu_configure_thread(capabilities.0) }.check(),
            DeviceKind::Cuda | DeviceKind::Rocm | DeviceKind::Metal => Err(Status::MissingKernel),
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
        let names = Device::cpu().capabilities_enabled().unwrap().to_string();
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
        let devices = Capabilities::DEVICES
            .iter()
            .fold(Capabilities::default(), |set, gpu| set | gpu);
        assert_eq!(devices, Capabilities::DEVICES);
        assert_eq!(Capabilities::CPUS.bits() & Capabilities::DEVICES.bits(), 0);
    }

    #[test]
    fn devices_report_clamped_capabilities() {
        let cpu = Device::cpu();
        let enabled = cpu.capabilities_enabled().unwrap();
        assert!(enabled.contains(Capability::Serial));
        let detected = cpu.capabilities_detected().unwrap();
        assert_eq!(enabled.bits() & !(detected.bits() | Capability::Serial as u64), 0);
        assert_eq!(enabled.bits() & !cpu.capabilities_compiled().bits(), 0);
        assert_eq!(Device::count(DeviceKind::Cpu), Ok(1));
        assert!(Device::new(DeviceKind::Cpu, 1).is_err());

        for kind in [DeviceKind::Cuda, DeviceKind::Rocm, DeviceKind::Metal] {
            let count = match Device::count(kind) {
                Ok(count) => count,
                Err(status) => {
                    assert_eq!(status, Status::MissingGpu);
                    assert_eq!(Device::new(kind, 0), Err(Status::MissingGpu));
                    continue;
                }
            };
            assert!(count > 0);
            assert_eq!(Device::new(kind, count), Err(Status::MissingGpu));
            for ordinal in 0..count {
                let device = Device::new(kind, ordinal).unwrap();
                let detected = device.capabilities_detected().unwrap();
                let compiled = device.capabilities_compiled();
                assert_eq!(detected.bits() & Capabilities::CPUS.bits(), 0);
                assert_eq!(
                    device.capabilities_enabled().unwrap().bits(),
                    detected.bits() & compiled.bits()
                );
            }
        }
        let gpu = Device {
            kind: DeviceKind::Metal,
            ordinal: 0,
        };
        assert_eq!(gpu.capabilities_enable(Capabilities::ANY), Err(Status::MissingKernel));
        assert_eq!(gpu.configure_thread(Capabilities::ANY), Err(Status::MissingKernel));
        assert_eq!(cpu.configure_thread(enabled), Ok(()));
    }

    #[test]
    fn enable_clamps_and_keeps_serial() {
        let cpu = Device::cpu();
        let enabled = cpu.capabilities_enabled().unwrap();
        let available = cpu.capabilities_detected().unwrap().bits() & cpu.capabilities_compiled().bits();
        assert_eq!(cpu.capabilities_enable(Capabilities::ANY).unwrap().bits(), available);
        let serial = Capabilities::from(Capability::Serial);
        assert_eq!(cpu.capabilities_enable(Capabilities::default()), Ok(serial));
        assert_eq!(cpu.capabilities_enabled(), Ok(serial));
        assert!(sz::find("haystack", "st").is_some(), "serial alone still dispatches");
        cpu.capabilities_enable(enabled).unwrap();
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
}
