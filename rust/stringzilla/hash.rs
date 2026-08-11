//! Byte-sums, the 64-bit hasher, SHA-256 states, and HMAC-SHA256.

use super::*;
use core::ffi::c_void;

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

/// Bytes in a SHA256 digest, fixed by FIPS 180-4.
pub const SHA256_DIGEST_LENGTH: usize = 32;

/// Bytes in a SHA256 message block, fixed by FIPS 180-4.
pub const SHA256_BLOCK_LENGTH: usize = 64;

/// One SHA256 digest, as produced by [`Sha256::digest`] and [`sha256_multistate_digest`].
pub type Sha256Digest = [u8; SHA256_DIGEST_LENGTH];

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
/*  Mirrors the C `sz_sha256_state_t` by size and alignment only. Every operation on it is an FFI call, so
 *  restating the C fields here would buy nothing and would let a field change in `hash.h` desynchronize
 *  silently. Deliberately not over-aligned: forcing 64-byte alignment would change the array stride
 *  relative to C and silently misplace every element of a `&mut [Sha256]` handed to the batched kernels. */
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct Sha256([u8; 128]);

/*  Guards the layout the batched kernels depend on: they index a contiguous array with the C stride. */
const _: () = assert!(core::mem::size_of::<Sha256>() == 128);
const _: () = assert!(core::mem::align_of::<Sha256>() == 8);

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
        let mut state = Sha256([0; 128]);
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

    /// Returns the current SHA256 digest, leaving the hasher able to accept more data.
    pub fn digest(&self) -> Sha256Digest {
        let mut digest = [0u8; SHA256_DIGEST_LENGTH];
        unsafe {
            sz_sha256_state_digest(self as *const _ as *const c_void, digest.as_mut_ptr());
        }
        digest
    }

    /// Convenience method to hash data in one call.
    pub fn hash(data: &[u8]) -> Sha256Digest {
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

/// Advances many independent SHA256 states at once, one message per lane.
///
/// Hashing a single message is a serial dependency chain, so no instruction set can speed it up. Independent
/// messages do compress in parallel lanes: sixteen at a time on AVX-512, eight on AVX2. Feed at least a few
/// kilobytes per lane per call, since shorter chunks never reach the lane-parallel path.
///
/// Borrows the caller's storage throughout and allocates nothing. Accepts anything that dereferences to
/// bytes, so a `Vec<String>` or `Vec<Vec<u8>>` needs no intermediate slice-of-slices. See
/// [`sha256_multistate_update_by`] for messages that do not sit in one contiguous slice.
///
/// # Errors
///
/// Returns [`Status::BadAlloc`] if `chunks` does not have exactly one entry per lane.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// let mut states = vec![sz::Sha256::new(); 2];
/// let heads: Vec<String> = vec!["Hello, ".into(), "Goodbye, ".into()];
/// let tails: Vec<String> = vec!["world!".into(), "world!".into()];
/// sz::sha256_multistate_update(&mut states, &heads).unwrap();
/// sz::sha256_multistate_update(&mut states, &tails).unwrap();
///
/// let mut digests = vec![[0u8; 32]; 2];
/// sz::sha256_multistate_digest(&states, &mut digests).unwrap();
/// assert_eq!(digests[0], sz::Sha256::hash(b"Hello, world!"));
/// assert_eq!(digests[1], sz::Sha256::hash(b"Goodbye, world!"));
/// ```
pub fn sha256_multistate_update<Element: AsRef<[u8]>>(states: &mut [Sha256], chunks: &[Element]) -> Result<(), Status> {
    if chunks.len() != states.len() {
        return Err(Status::BadAlloc);
    }
    sha256_multistate_update_by(states, |lane_index| chunks[lane_index].as_ref())
}

/// Advances many independent SHA256 states at once, taking each lane's next chunk from a caller-provided key.
///
/// # Errors
///
/// Returns [`Status::BadAlloc`] if `mapper` cannot serve one chunk per lane.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// struct Record { payload: &'static str }
/// let records = [Record { payload: "alpha" }, Record { payload: "beta" }];
///
/// let mut states = vec![sz::Sha256::new(); 2];
/// sz::sha256_multistate_update_by(&mut states, |lane| records[lane].payload.as_bytes()).unwrap();
///
/// let mut digests = vec![[0u8; 32]; 2];
/// sz::sha256_multistate_digest(&states, &mut digests).unwrap();
/// assert_eq!(digests[0], sz::Sha256::hash(b"alpha"));
/// ```
pub fn sha256_multistate_update_by<Mapper, Key>(states: &mut [Sha256], mapper: Mapper) -> Result<(), Status>
where
    Mapper: Fn(usize) -> Key,
    Key: AsRef<[u8]>,
{
    if states.is_empty() {
        return Ok(());
    }

    // Same adapter as `argsort_by`: relabel each borrowed chunk `'static` so it can cross the C ABI. Safe
    // because the kernel reads the chunks only during this synchronous call.
    let adapter = move |lane_index: usize| -> &'static [u8] {
        let binding = mapper(lane_index);
        let slice = binding.as_ref();
        unsafe { core::mem::transmute(slice) }
    };
    _sha256_multistate_update_impl(adapter, states)
}

/// Helper that takes an adapter (with a concrete type) and performs the FFI call.
fn _sha256_multistate_update_impl<Adapter>(adapter: Adapter, states: &mut [Sha256]) -> Result<(), Status>
where
    Adapter: Fn(usize) -> &'static [u8],
{
    let wrapper = _PunnedSliceLookupView {
        get_slice: unsafe { _get_slice_fn::<Adapter>() },
        data: &adapter as *const Adapter as *const c_void,
    };
    let texts = _SzSequence {
        handle: &wrapper as *const _ as *const c_void,
        count: states.len(),
        get_start: Some(_slice_get_start_punned),
        get_length: Some(_slice_get_length_punned),
    };
    unsafe { sz_sha256_multistate_update(states.as_mut_ptr() as *mut c_void, &texts) };
    Ok(())
}

/// Writes each lane's digest into caller-provided storage, leaving every lane able to accept more data.
///
/// # Errors
///
/// Returns [`Status::BadAlloc`] if `digests` does not have exactly one entry per lane.
pub fn sha256_multistate_digest(states: &[Sha256], digests: &mut [Sha256Digest]) -> Result<(), Status> {
    if digests.len() != states.len() {
        return Err(Status::BadAlloc);
    }
    if states.is_empty() {
        return Ok(());
    }
    unsafe {
        sz_sha256_multistate_digest(
            states.as_ptr() as *const c_void,
            states.len(),
            digests.as_mut_ptr() as *mut u8,
        )
    };
    Ok(())
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
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> Sha256Digest {
    let (mut inner, outer) = _hmac_sha256_primed(key);
    inner.update(message);
    _hmac_sha256_wrap(&inner, &outer)
}

/// Primes the inner and outer states with the HMAC key pads, per FIPS 198-1.
///
/// The message enters only the inner hash, as the suffix of the ipad block, so authenticating many
/// messages under one key is as lane-parallel as digesting them: prime once, then broadcast.
fn _hmac_sha256_primed(key: &[u8]) -> (Sha256, Sha256) {
    // Keys longer than one block are replaced by their digest; shorter ones are zero-padded.
    let mut key_pad = [0u8; SHA256_BLOCK_LENGTH];
    if key.len() > SHA256_BLOCK_LENGTH {
        key_pad[..SHA256_DIGEST_LENGTH].copy_from_slice(&Sha256::hash(key));
    } else {
        key_pad[..key.len()].copy_from_slice(key);
    }

    let mut block = [0u8; SHA256_BLOCK_LENGTH];
    let mut inner = Sha256::new();
    for byte_index in 0..SHA256_BLOCK_LENGTH {
        block[byte_index] = key_pad[byte_index] ^ 0x36;
    }
    inner.update(&block);

    let mut outer = Sha256::new();
    for byte_index in 0..SHA256_BLOCK_LENGTH {
        block[byte_index] = key_pad[byte_index] ^ 0x5c;
    }
    outer.update(&block);

    // The pads are derived from the secret, so don't leave them on the stack for the next frame. The
    // writes are volatile because nothing reads them afterwards and a plain assignment is a dead store.
    unsafe {
        core::ptr::write_volatile(&mut key_pad, [0u8; SHA256_BLOCK_LENGTH]);
        core::ptr::write_volatile(&mut block, [0u8; SHA256_BLOCK_LENGTH]);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    (inner, outer)
}

/// Wraps the inner state's digest with the primed outer state, completing one tag.
fn _hmac_sha256_wrap(inner: &Sha256, outer: &Sha256) -> Sha256Digest {
    let mut wrapping = *outer;
    wrapping.update(&inner.digest());
    wrapping.digest()
}

/// Authenticates many messages under one key at once, one message per lane.
///
/// Authenticating one message is a serial dependency chain, but independent messages compress in
/// parallel lanes - sixteen at a time on AVX-512, eight on AVX2 - and both the message pass and HMAC's
/// outer wrap ride those same kernels. Intended for batches of short messages, such as tokens or
/// webhook bodies sharing a secret; there is no streaming form, since the message never spans calls.
///
/// Borrows the caller's storage throughout and allocates nothing, so it works under `no_std`. `states`
/// is scratch of one state per message, reused for both passes; only `tags` is written for keeps.
///
/// # Errors
///
/// Returns [`Status::BadAlloc`] unless `messages`, `states` and `tags` all have the same length.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// let messages: Vec<String> = vec!["alpha".into(), "beta".into()];
/// let mut states = vec![sz::Sha256::new(); messages.len()];
/// let mut tags = vec![[0u8; 32]; messages.len()];
///
/// sz::hmac_sha256_multistate(b"secret", &messages, &mut states, &mut tags).unwrap();
/// assert_eq!(tags[0], sz::hmac_sha256(b"secret", b"alpha"));
/// ```
pub fn hmac_sha256_multistate<Element: AsRef<[u8]>>(
    key: &[u8],
    messages: &[Element],
    states: &mut [Sha256],
    tags: &mut [Sha256Digest],
) -> Result<(), Status> {
    if messages.len() != states.len() || messages.len() != tags.len() {
        return Err(Status::BadAlloc);
    }
    if messages.is_empty() {
        return Ok(());
    }

    let (inner, outer) = _hmac_sha256_primed(key);
    for state in states.iter_mut() {
        *state = inner;
    }
    sha256_multistate_update(states, messages)?;
    sha256_multistate_digest(states, tags)?;

    // Second pass: every lane absorbs its own inner digest, read straight out of `tags`.
    for state in states.iter_mut() {
        *state = outer;
    }
    {
        let inner_digests = &*tags;
        sha256_multistate_update_by(states, |lane_index| &inner_digests[lane_index][..])?;
    }

    // Safe to overwrite in place: every inner digest has already been absorbed into its lane.
    sha256_multistate_digest(states, tags)
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
pub fn bytesum<Text>(text: Text) -> u64
where
    Text: AsRef<[u8]>,
{
    let text_ref = text.as_ref();
    let text_pointer = text_ref.as_ptr() as _;
    let text_length = text_ref.len();
    unsafe { sz_bytesum(text_pointer, text_length) }
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
pub fn hash_with_seed<Text>(text: Text, seed: u64) -> u64
where
    Text: AsRef<[u8]>,
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
pub fn hash<Text>(text: Text) -> u64
where
    Text: AsRef<[u8]>,
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
pub fn hash_multiseed_into<Text>(text: Text, seeds: &[u64], out: &mut [u64])
where
    Text: AsRef<[u8]>,
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

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::hash::Hasher as _;
    // `HashMap`/`HashSet` have no `alloc`-only equivalent (unlike `Vec`/`String`/`BTreeMap`), so the
    // handful of tests that need them stay behind `feature = "std"`; everything else here runs no_std.
    #[cfg(feature = "std")]
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::sz;

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
    #[cfg(feature = "std")]
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
    #[cfg(feature = "std")]
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
    fn sha256_multistate_matches_single() {
        // Lane counts bracketing both vector widths, with ragged lengths and an empty lane
        for lanes_count in [1usize, 7, 8, 9, 16, 17, 33] {
            let messages: Vec<Vec<u8>> = (0..lanes_count)
                .map(|lane_index| vec![b'a' + (lane_index % 26) as u8; lane_index * 137 % 5000])
                .collect();

            let mut states = vec![sz::Sha256::new(); lanes_count];

            // Feed each lane in three uneven slices, so partial blocks carry across calls
            let mut offsets = vec![0usize; lanes_count];
            for slice_index in 0..3 {
                let ranges: Vec<(usize, usize)> = (0..lanes_count)
                    .map(|lane_index| {
                        let remaining = messages[lane_index].len() - offsets[lane_index];
                        let take = if slice_index == 2 { remaining } else { remaining / 3 };
                        let start = offsets[lane_index];
                        offsets[lane_index] += take;
                        (start, start + take)
                    })
                    .collect();
                sz::sha256_multistate_update_by(&mut states, |lane_index| {
                    let (start, end) = ranges[lane_index];
                    &messages[lane_index][start..end]
                })
                .expect("one chunk per lane");
            }

            let mut digests = vec![[0u8; 32]; lanes_count];
            sz::sha256_multistate_digest(&states, &mut digests).expect("one digest per lane");
            for lane_index in 0..lanes_count {
                assert_eq!(digests[lane_index], sz::Sha256::hash(&messages[lane_index]));
            }
        }
    }

    #[test]
    fn sha256_multistate_borrows_owned_strings() {
        // The generic front door must take `Vec<String>` without an intermediate slice-of-slices.
        let heads: Vec<String> = vec!["Hello, ".into(), "Goodbye, ".into()];
        let tails: Vec<Vec<u8>> = vec![b"world!".to_vec(), b"world!".to_vec()];

        let mut states = vec![sz::Sha256::new(); 2];
        sz::sha256_multistate_update(&mut states, &heads).expect("one chunk per lane");
        sz::sha256_multistate_update(&mut states, &tails).expect("one chunk per lane");

        let mut digests = vec![[0u8; sz::SHA256_DIGEST_LENGTH]; 2];
        sz::sha256_multistate_digest(&states, &mut digests).expect("one digest per lane");
        assert_eq!(digests[0], sz::Sha256::hash(b"Hello, world!"));
        assert_eq!(digests[1], sz::Sha256::hash(b"Goodbye, world!"));
    }

    #[test]
    fn sha256_multistate_size_checks() {
        let mut states = vec![sz::Sha256::new(); 3];
        let too_few: Vec<&[u8]> = vec![b"a".as_slice(), b"b".as_slice()];
        assert_eq!(sz::sha256_multistate_update(&mut states, &too_few), Err(sz::Status::BadAlloc));

        let mut too_few_digests = vec![[0u8; sz::SHA256_DIGEST_LENGTH]; 2];
        assert_eq!(
            sz::sha256_multistate_digest(&states, &mut too_few_digests),
            Err(sz::Status::BadAlloc)
        );
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
        // Degenerate case: an empty key is zero-padded to a full block, an empty message adds nothing.
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
    fn hmac_sha256_multistate_matches_single() {
        // Lane counts bracketing both vector widths, with ragged lengths, an empty message, and keys on
        // either side of the one-block boundary where the key is replaced by its own digest.
        for key_length in [0usize, 16, 64, 65, 200] {
            let key: Vec<u8> = (0..key_length).map(|index| (index % 251) as u8).collect();
            for lanes_count in [1usize, 7, 8, 9, 16, 17, 33] {
                let messages: Vec<Vec<u8>> = (0..lanes_count)
                    .map(|lane_index| vec![b'a' + (lane_index % 26) as u8; lane_index * 137 % 5000])
                    .collect();

                let mut states = vec![sz::Sha256::new(); lanes_count];
                let mut tags = vec![[0u8; sz::SHA256_DIGEST_LENGTH]; lanes_count];
                sz::hmac_sha256_multistate(&key, &messages, &mut states, &mut tags).expect("one tag per message");

                for lane_index in 0..lanes_count {
                    assert_eq!(tags[lane_index], sz::hmac_sha256(&key, &messages[lane_index]));
                }
            }
        }
    }

    #[test]
    fn hmac_sha256_multistate_size_checks() {
        let messages: Vec<&[u8]> = vec![b"a".as_slice(), b"b".as_slice()];
        let mut states = vec![sz::Sha256::new(); 2];
        let mut too_few = vec![[0u8; sz::SHA256_DIGEST_LENGTH]; 1];
        assert_eq!(
            sz::hmac_sha256_multistate(b"k", &messages, &mut states, &mut too_few),
            Err(sz::Status::BadAlloc)
        );
    }

    #[test]
    fn hmac_sha256_rfc4231_case1() {
        // The published vector, so this checks against the standard rather than against ourselves.
        let key = [0x0bu8; 20];
        let mac = sz::hmac_sha256(&key, b"Hi There");
        let expected = [
            0xb0, 0x34, 0x4c, 0x61, 0xd8, 0xdb, 0x38, 0x53, 0x5c, 0xa8, 0xaf, 0xce, 0xaf, 0x0b, 0xf1, 0x2b, 0x88, 0x1d,
            0xc2, 0x00, 0xc9, 0x83, 0x3d, 0xa7, 0x26, 0xe9, 0x37, 0x6c, 0x2e, 0x32, 0xcf, 0xf7,
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
}
