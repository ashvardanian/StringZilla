//! AES-256 in counter mode and in Galois/counter mode.

use super::*;
use core::ffi::c_void;
use core::fmt;

/// Bytes in an AES-256 secret key.
pub const AES256_KEY_LENGTH: usize = 32;

/// Bytes in the nonce both AES-256 modes accept, the width NIST SP 800-38D recommends.
pub const AES256_NONCE_LENGTH: usize = 12;

/// Bytes in a Galois/counter mode, or GCM, authentication tag.
pub const AES256_TAG_LENGTH: usize = 16;

/// Round keys in an AES-256 schedule: 15 of them, four 32-bit words each.
const AES256_ROUND_KEYS: usize = 60;

/// Bytes of Galois hash, or GHASH, subkey powers, holding `H^1` through `H^8` ascending.
const AES256_GALOIS_POWERS: usize = 8 * 16;

/// Why an authenticated decryption refused to hand back plaintext.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthenticationError {
    /// The tag did not match the ciphertext and associated data, so the output holds zeros
    /// rather than the plaintext an attacker chose.
    TagMismatch,
    /// The C core reported a status this binding does not model, so the output is untrusted.
    UnexpectedStatus(i32),
}

/// Expanded AES-256 round-key schedule for counter mode, the construction named AES-256-CTR.
///
/// Counter mode is unauthenticated and seekable: the keystream at a byte offset depends on nothing
/// but the key, the nonce and that offset, so a caller may decrypt the middle of a file without
/// touching its start. Reusing a nonce under one key exposes the exclusive-or of two plaintexts.
///
/// Mirrors `sz_aes256_key_t`, and carries none of the Galois hash powers counter mode never reads.
#[repr(C)]
pub struct Aes256CtrKey {
    round_keys: [u32; AES256_ROUND_KEYS],
}

/// Expanded AES-256 schedule plus the Galois hash subkey powers, the construction named AES-256-GCM.
///
/// Authentication costs the ability to seek, so this mode transforms whole messages or bounded
/// chunks in order. Reusing a nonce under one key exposes the hash subkey, from which every message
/// under that key can be forged.
///
/// Mirrors `sz_aes256_gcm_key_t`.
#[repr(C)]
pub struct Aes256GcmKey {
    block: Aes256CtrKey,
    powers: [u8; AES256_GALOIS_POWERS],
}

/// The payload both directions of a chunked transformation carry.
///
/// Mirrors `sz_aes256_gcm_state_t`. It holds no direction: the Galois hash absorbs ciphertext both
/// ways, the output buffer when sealing and the input when opening, and the C core encodes that in
/// the type of the wrapper rather than in a field. The key is embedded rather than borrowed, so a
/// state cannot outlive it, and wipes with the state.
#[repr(C)]
struct Aes256GcmState {
    key: Aes256GcmKey,
    accumulator: [u8; 16],
    counter: [u8; 16],
    tag_mask: [u8; 16],
    partial: [u8; 16],
    keystream: [u8; 16],
    associated_length: u64,
    text_length: u64,
    buffered: u8,
    keystream_used: u8,
}

/// Seals a message delivered in chunks under Galois/counter mode, or GCM, accumulating its tag as it goes.
///
/// A chunk boundary is invisible to the result, so any chunking produces the ciphertext and tag a
/// single [`Aes256GcmKey::encrypt_into`] would.
///
/// Mirrors `sz_aes256_gcm_encryptor_t`, whose address every call below passes.
#[repr(C)]
pub struct Aes256GcmEncryptor {
    state: Aes256GcmState,
}

/// Opens a message delivered in chunks under Galois/counter mode, or GCM, accumulating its tag as it goes.
///
/// Every emitted byte leaves this type unauthenticated, which is why the methods that emit them say
/// so in their names. Only [`Aes256GcmDecryptor::verify`] tells the caller whether they were genuine.
///
/// Mirrors `sz_aes256_gcm_decryptor_t`, whose address every call below passes.
#[repr(C)]
pub struct Aes256GcmDecryptor {
    state: Aes256GcmState,
}

/// Maps the raw `sz_status_t` of an authenticated path onto a `Result` the caller cannot drop
/// silently. Any status other than success refuses the plaintext rather than guessing at it.
fn authentication_result_from_status(status: i32) -> Result<(), AuthenticationError> {
    const SUCCESS: i32 = Status::Success as i32;
    const AUTHENTICATION_FAILED: i32 = Status::AuthenticationFailed as i32;
    match status {
        SUCCESS => Ok(()),
        AUTHENTICATION_FAILED => Err(AuthenticationError::TagMismatch),
        other => Err(AuthenticationError::UnexpectedStatus(other)),
    }
}

impl Aes256CtrKey {
    /// Expands a 32-byte secret into the AES-256 round-key schedule.
    pub fn new(secret: &[u8; AES256_KEY_LENGTH]) -> Self {
        let mut key = Aes256CtrKey::zeroed();
        unsafe { sz_aes256_key_init(&mut key as *mut _ as *mut c_void, secret.as_ptr()) };
        key
    }

    /// Exclusive-ors `text` against the keystream starting at `byte_offset`, writing `output`.
    /// Counter mode is its own inverse, so this call both encrypts and decrypts.
    ///
    /// # Panics
    ///
    /// Panics if `output` is not as long as `text`.
    pub fn xor_into(&self, nonce: &[u8; AES256_NONCE_LENGTH], byte_offset: u64, text: &[u8], output: &mut [u8]) {
        assert_eq!(text.len(), output.len(), "`output` must be as long as `text`");
        unsafe {
            sz_aes256_ctr_xor(
                self as *const _ as *const c_void,
                nonce.as_ptr(),
                byte_offset,
                text.as_ptr() as *const c_void,
                text.len(),
                output.as_mut_ptr() as *mut c_void,
            )
        }
    }

    /// Exclusive-ors `text` against the keystream starting at `byte_offset`, in place.
    /// Counter mode is its own inverse, so this call both encrypts and decrypts.
    pub fn xor_in_place(&self, nonce: &[u8; AES256_NONCE_LENGTH], byte_offset: u64, text: &mut [u8]) {
        let length = text.len();
        let pointer = text.as_mut_ptr() as *mut c_void;
        unsafe {
            sz_aes256_ctr_xor(
                self as *const _ as *const c_void,
                nonce.as_ptr(),
                byte_offset,
                pointer as *const c_void,
                length,
                pointer,
            )
        }
    }

    /// An all-zero schedule, expanded in place by the C core before any caller sees it.
    const fn zeroed() -> Self {
        Aes256CtrKey {
            round_keys: [0; AES256_ROUND_KEYS],
        }
    }
}

impl Drop for Aes256CtrKey {
    /// Overwrites the schedule so the key does not outlive its use.
    ///
    /// The write is volatile because a plain assignment is a dead store - nothing reads the schedule
    /// after this returns, so the optimizer is free to delete it - and the fence stops the zeros from
    /// sinking past the end of the drop.
    fn drop(&mut self) {
        unsafe { core::ptr::write_volatile(self as *mut Self, Self::zeroed()) };
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl fmt::Debug for Aes256CtrKey {
    /// Prints a placeholder, because a key that reaches a log is a key that has leaked.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Aes256CtrKey(<secret>)")
    }
}

impl Aes256GcmKey {
    /// Expands a 32-byte secret into a schedule and the Galois hash subkey powers.
    pub fn new(secret: &[u8; AES256_KEY_LENGTH]) -> Self {
        let mut key = Aes256GcmKey::zeroed();
        unsafe { sz_aes256_gcm_key_init(&mut key as *mut _ as *mut c_void, secret.as_ptr()) };
        key
    }

    /// Encrypts `text` into `output`, returning the tag over the ciphertext and `associated`.
    /// The associated bytes are authenticated but not encrypted, such as a routing header.
    ///
    /// # Panics
    ///
    /// Panics if `output` is not as long as `text`.
    pub fn encrypt_into(
        &self,
        nonce: &[u8; AES256_NONCE_LENGTH],
        associated: &[u8],
        text: &[u8],
        output: &mut [u8],
    ) -> [u8; AES256_TAG_LENGTH] {
        assert_eq!(text.len(), output.len(), "`output` must be as long as `text`");
        let mut tag = [0u8; AES256_TAG_LENGTH];
        unsafe {
            sz_aes256_gcm_encrypt(
                self as *const _ as *const c_void,
                nonce.as_ptr(),
                associated.as_ptr() as *const c_void,
                associated.len(),
                text.as_ptr() as *const c_void,
                text.len(),
                output.as_mut_ptr() as *mut c_void,
                tag.as_mut_ptr(),
            )
        };
        tag
    }

    /// Encrypts `text` in place, returning the tag over the ciphertext and `associated`.
    pub fn encrypt_in_place(
        &self,
        nonce: &[u8; AES256_NONCE_LENGTH],
        associated: &[u8],
        text: &mut [u8],
    ) -> [u8; AES256_TAG_LENGTH] {
        let mut tag = [0u8; AES256_TAG_LENGTH];
        let length = text.len();
        let pointer = text.as_mut_ptr() as *mut c_void;
        unsafe {
            sz_aes256_gcm_encrypt(
                self as *const _ as *const c_void,
                nonce.as_ptr(),
                associated.as_ptr() as *const c_void,
                associated.len(),
                pointer as *const c_void,
                length,
                pointer,
                tag.as_mut_ptr(),
            )
        };
        tag
    }

    /// Verifies `tag` and decrypts `text` into `output`, which holds the plaintext only on success.
    /// A rejected tag leaves `output` zeroed, so a caller who drops the error still cannot read
    /// forged plaintext. The comparison takes the same time wherever the tag first differs.
    ///
    /// # Panics
    ///
    /// Panics if `output` is not as long as `text`.
    pub fn decrypt_into(
        &self,
        nonce: &[u8; AES256_NONCE_LENGTH],
        associated: &[u8],
        text: &[u8],
        output: &mut [u8],
        tag: &[u8; AES256_TAG_LENGTH],
    ) -> Result<(), AuthenticationError> {
        assert_eq!(text.len(), output.len(), "`output` must be as long as `text`");
        let status = unsafe {
            sz_aes256_gcm_decrypt(
                self as *const _ as *const c_void,
                nonce.as_ptr(),
                associated.as_ptr() as *const c_void,
                associated.len(),
                text.as_ptr() as *const c_void,
                text.len(),
                output.as_mut_ptr() as *mut c_void,
                tag.as_ptr(),
            )
        };
        authentication_result_from_status(status)
    }

    /// Verifies `tag` and decrypts `text` in place, which holds the plaintext only on success.
    /// A rejected tag zeroes `text`, destroying the ciphertext it arrived with.
    pub fn decrypt_in_place(
        &self,
        nonce: &[u8; AES256_NONCE_LENGTH],
        associated: &[u8],
        text: &mut [u8],
        tag: &[u8; AES256_TAG_LENGTH],
    ) -> Result<(), AuthenticationError> {
        let length = text.len();
        let pointer = text.as_mut_ptr() as *mut c_void;
        let status = unsafe {
            sz_aes256_gcm_decrypt(
                self as *const _ as *const c_void,
                nonce.as_ptr(),
                associated.as_ptr() as *const c_void,
                associated.len(),
                pointer as *const c_void,
                length,
                pointer,
                tag.as_ptr(),
            )
        };
        authentication_result_from_status(status)
    }

    /// An all-zero key, expanded in place by the C core before any caller sees it.
    const fn zeroed() -> Self {
        Aes256GcmKey {
            block: Aes256CtrKey::zeroed(),
            powers: [0; AES256_GALOIS_POWERS],
        }
    }
}

impl Drop for Aes256GcmKey {
    /// Overwrites the schedule and the hash powers so the key does not outlive its use.
    ///
    /// The write is volatile for the same reason as in [`Aes256CtrKey`]: a plain assignment is a
    /// dead store the optimizer may delete, and the fence stops the zeros from sinking past the end
    /// of the drop.
    fn drop(&mut self) {
        unsafe { core::ptr::write_volatile(self as *mut Self, Self::zeroed()) };
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl fmt::Debug for Aes256GcmKey {
    /// Prints a placeholder, because a key that reaches a log is a key that has leaked.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Aes256GcmKey(<secret>)")
    }
}

impl Aes256GcmState {
    /// An all-zero payload, filled in by the typed C entry point before any caller sees it.
    const fn zeroed() -> Self {
        Aes256GcmState {
            key: Aes256GcmKey::zeroed(),
            accumulator: [0; 16],
            counter: [0; 16],
            tag_mask: [0; 16],
            partial: [0; 16],
            keystream: [0; 16],
            associated_length: 0,
            text_length: 0,
            buffered: 0,
            keystream_used: 0,
        }
    }
}

impl Drop for Aes256GcmEncryptor {
    /// Overwrites the payload so a streamed message leaves nothing behind.
    ///
    /// The embedded key scrubs itself through [`Aes256GcmKey`]'s drop; the rest does not, and 77 bytes
    /// of tag mask, live keystream and pending hash block survived before this existed.
    fn drop(&mut self) {
        unsafe { core::ptr::write_volatile(&mut self.state as *mut Aes256GcmState, Aes256GcmState::zeroed()) };
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for Aes256GcmDecryptor {
    /// Overwrites the payload for the same reason [`Aes256GcmEncryptor`] does.
    fn drop(&mut self) {
        unsafe { core::ptr::write_volatile(&mut self.state as *mut Aes256GcmState, Aes256GcmState::zeroed()) };
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl Aes256GcmEncryptor {
    /// Begins a chunked encryption under `key` and `nonce`, copying `key` in so this cannot outlive it.
    pub fn new(key: &Aes256GcmKey, nonce: &[u8; AES256_NONCE_LENGTH]) -> Self {
        let mut encryptor = Aes256GcmEncryptor {
            state: Aes256GcmState::zeroed(),
        };
        unsafe {
            sz_aes256_gcm_encryptor_init(
                &mut encryptor as *mut _ as *mut c_void,
                key as *const _ as *const c_void,
                nonce.as_ptr(),
            )
        };
        encryptor
    }

    /// Absorbs associated data, which is authenticated but not encrypted.
    /// All of it must be absorbed before the first chunk of the message.
    pub fn associate(&mut self, associated: &[u8]) -> &mut Self {
        unsafe {
            sz_aes256_gcm_encryptor_associate(
                self as *mut _ as *mut c_void,
                associated.as_ptr() as *const c_void,
                associated.len(),
            )
        };
        self
    }

    /// Encrypts one chunk into `output` and folds its ciphertext into the running tag.
    ///
    /// # Panics
    ///
    /// Panics if `output` is not as long as `text`.
    pub fn encrypt_into(&mut self, text: &[u8], output: &mut [u8]) -> &mut Self {
        assert_eq!(text.len(), output.len(), "`output` must be as long as `text`");
        unsafe {
            sz_aes256_gcm_encryptor_update(
                self as *mut _ as *mut c_void,
                text.as_ptr() as *const c_void,
                text.len(),
                output.as_mut_ptr() as *mut c_void,
            )
        };
        self
    }

    /// Encrypts one chunk in place and folds its ciphertext into the running tag.
    pub fn encrypt_in_place(&mut self, text: &mut [u8]) -> &mut Self {
        let length = text.len();
        let pointer = text.as_mut_ptr() as *mut c_void;
        unsafe {
            sz_aes256_gcm_encryptor_update(self as *mut _ as *mut c_void, pointer as *const c_void, length, pointer)
        };
        self
    }

    /// Returns the tag over everything encrypted so far, leaving the encryptor ready for more.
    pub fn digest(&self) -> [u8; AES256_TAG_LENGTH] {
        let mut tag = [0u8; AES256_TAG_LENGTH];
        unsafe { sz_aes256_gcm_encryptor_digest(self as *const _ as *const c_void, tag.as_mut_ptr()) };
        tag
    }
}

impl Aes256GcmDecryptor {
    /// Begins a chunked decryption under `key` and `nonce`, copying `key` in so this cannot outlive it.
    pub fn new(key: &Aes256GcmKey, nonce: &[u8; AES256_NONCE_LENGTH]) -> Self {
        let mut decryptor = Aes256GcmDecryptor {
            state: Aes256GcmState::zeroed(),
        };
        unsafe {
            sz_aes256_gcm_decryptor_init(
                &mut decryptor as *mut _ as *mut c_void,
                key as *const _ as *const c_void,
                nonce.as_ptr(),
            )
        };
        decryptor
    }

    /// Absorbs associated data, which is authenticated but not encrypted.
    /// All of it must be absorbed before the first chunk of the message.
    pub fn associate(&mut self, associated: &[u8]) -> &mut Self {
        unsafe {
            sz_aes256_gcm_decryptor_associate(
                self as *mut _ as *mut c_void,
                associated.as_ptr() as *const c_void,
                associated.len(),
            )
        };
        self
    }

    /// Decrypts one chunk into `output` and folds its ciphertext into the running tag.
    ///
    /// The emitted bytes are **not authenticated**. Nothing has yet checked that this ciphertext is
    /// genuine, so the caller owns that risk: buffer the output until [`Self::verify`] succeeds and
    /// discard it if it does not. A streaming decryption cannot avoid this, which is why the name
    /// says so. Callers who can hold the whole message should use [`Aes256GcmKey::decrypt_into`],
    /// which emits nothing until the tag has matched.
    ///
    /// # Panics
    ///
    /// Panics if `output` is not as long as `text`.
    pub fn decrypt_unverified_into(&mut self, text: &[u8], output: &mut [u8]) -> &mut Self {
        assert_eq!(text.len(), output.len(), "`output` must be as long as `text`");
        unsafe {
            sz_aes256_gcm_decryptor_update_unverified(
                self as *mut _ as *mut c_void,
                text.as_ptr() as *const c_void,
                text.len(),
                output.as_mut_ptr() as *mut c_void,
            )
        };
        self
    }

    /// Decrypts one chunk in place and folds its ciphertext into the running tag.
    ///
    /// The emitted bytes are **not authenticated**, exactly as in [`Self::decrypt_unverified_into`],
    /// and in place they also overwrite the ciphertext a failed [`Self::verify`] would have to discard.
    pub fn decrypt_unverified_in_place(&mut self, text: &mut [u8]) -> &mut Self {
        let length = text.len();
        let pointer = text.as_mut_ptr() as *mut c_void;
        unsafe {
            sz_aes256_gcm_decryptor_update_unverified(
                self as *mut _ as *mut c_void,
                pointer as *const c_void,
                length,
                pointer,
            )
        };
        self
    }

    /// Checks `tag` against everything absorbed so far, in time independent of where it differs.
    /// Until this returns `Ok`, every byte the decryptor emitted is an attacker's to choose.
    pub fn verify(&self, tag: &[u8; AES256_TAG_LENGTH]) -> Result<(), AuthenticationError> {
        let status = unsafe { sz_aes256_gcm_decryptor_verify(self as *const _ as *const c_void, tag.as_ptr()) };
        authentication_result_from_status(status)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::sz;

    /// Decodes a hexadecimal literal into `output`, returning how many bytes were written.
    fn bytes_from_hex(hex: &str, output: &mut [u8]) -> usize {
        let digits = hex.as_bytes();
        let written = digits.len() / 2;
        for index in 0..written {
            let high = (digits[index * 2] as char).to_digit(16).expect("hexadecimal digit");
            let low = (digits[index * 2 + 1] as char).to_digit(16).expect("hexadecimal digit");
            output[index] = ((high << 4) | low) as u8;
        }
        written
    }

    /// One published Galois/counter mode vector, spelled out rather than derived from any backend.
    struct KnownGcmVector {
        secret: &'static str,
        nonce: &'static str,
        associated: &'static str,
        plaintext: &'static str,
        ciphertext: &'static str,
        tag: &'static str,
    }

    /// Cases 13 through 16 of McGrew and Viega's Galois/counter mode note, the AES-256 vectors
    /// NIST's validation suite is built from.
    const KNOWN_GCM_VECTORS: [KnownGcmVector; 4] = [
        // Empty message, empty associated data.
        KnownGcmVector {
            secret: "0000000000000000000000000000000000000000000000000000000000000000",
            nonce: "000000000000000000000000",
            associated: "",
            plaintext: "",
            ciphertext: "",
            tag: "530f8afbc74536b9a963b4f1c4cb738b",
        },
        // One zero block, empty associated data.
        KnownGcmVector {
            secret: "0000000000000000000000000000000000000000000000000000000000000000",
            nonce: "000000000000000000000000",
            associated: "",
            plaintext: "00000000000000000000000000000000",
            ciphertext: "cea7403d4d606b6e074ec5d3baf39d18",
            tag: "d0d1c8a799996bf0265b98b5d48ab919",
        },
        // Four whole blocks, empty associated data.
        KnownGcmVector {
            secret: "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
            nonce: "cafebabefacedbaddecaf888",
            associated: "",
            plaintext: concat!(
                "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72",
                "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255"
            ),
            ciphertext: concat!(
                "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa",
                "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad"
            ),
            tag: "b094dac5d93471bdec1a502270e3cc6c",
        },
        // A partial trailing block plus associated data, which exercises both zero pads.
        KnownGcmVector {
            secret: "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
            nonce: "cafebabefacedbaddecaf888",
            associated: "feedfacedeadbeeffeedfacedeadbeefabaddad2",
            plaintext: concat!(
                "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72",
                "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"
            ),
            ciphertext: concat!(
                "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa",
                "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662"
            ),
            tag: "76fc6ece0f4e1768cddf8853bb2d551b",
        },
    ];

    #[test]
    fn aes256_counter_round_trip() {
        let secret = [0x2Bu8; sz::AES256_KEY_LENGTH];
        let nonce = [0x07u8; sz::AES256_NONCE_LENGTH];
        let key = sz::Aes256CtrKey::new(&secret);
        let plaintext = b"counter mode is its own inverse, so one call serves both directions";

        let mut ciphertext = vec![0u8; plaintext.len()];
        key.xor_into(&nonce, 0, plaintext, &mut ciphertext);
        assert_ne!(ciphertext.as_slice(), &plaintext[..]);

        let mut recovered = vec![0u8; plaintext.len()];
        key.xor_into(&nonce, 0, &ciphertext, &mut recovered);
        assert_eq!(recovered.as_slice(), &plaintext[..]);

        // Transforming in place must agree with transforming into a separate buffer.
        let mut in_place = plaintext.to_vec();
        key.xor_in_place(&nonce, 0, &mut in_place);
        assert_eq!(in_place, ciphertext);
    }

    #[test]
    fn aes256_counter_seek() {
        let secret = [0x3Cu8; sz::AES256_KEY_LENGTH];
        let nonce = [0x5Du8; sz::AES256_NONCE_LENGTH];
        let key = sz::Aes256CtrKey::new(&secret);
        let whole: Vec<u8> = (0..512u32).map(|index| (index * 31 + 7) as u8).collect();

        let mut from_zero = vec![0u8; whole.len()];
        key.xor_into(&nonce, 0, &whole, &mut from_zero);

        // Seeking is the whole reason counter mode is exposed separately, so every offset - block
        // aligned or not - must land on the keystream the from-zero transformation used.
        for offset in 0..200usize {
            let mut sliced = vec![0u8; whole.len() - offset];
            key.xor_into(&nonce, offset as u64, &whole[offset..], &mut sliced);
            assert_eq!(sliced.as_slice(), &from_zero[offset..]);
        }
    }

    #[test]
    fn aes256_gcm_known_answers() {
        for vector in KNOWN_GCM_VECTORS.iter() {
            let mut secret = [0u8; sz::AES256_KEY_LENGTH];
            let mut nonce = [0u8; sz::AES256_NONCE_LENGTH];
            let mut expected_tag = [0u8; sz::AES256_TAG_LENGTH];
            let mut associated = [0u8; 32];
            let mut plaintext = [0u8; 64];
            let mut expected = [0u8; 64];
            bytes_from_hex(vector.secret, &mut secret);
            bytes_from_hex(vector.nonce, &mut nonce);
            let associated_length = bytes_from_hex(vector.associated, &mut associated);
            let length = bytes_from_hex(vector.plaintext, &mut plaintext);
            bytes_from_hex(vector.ciphertext, &mut expected);
            bytes_from_hex(vector.tag, &mut expected_tag);
            let associated = &associated[..associated_length];

            let key = sz::Aes256GcmKey::new(&secret);
            let mut produced = vec![0u8; length];
            let tag = key.encrypt_into(&nonce, associated, &plaintext[..length], &mut produced);
            assert_eq!(produced.as_slice(), &expected[..length]);
            assert_eq!(tag, expected_tag);

            let mut recovered = vec![0u8; length];
            key.decrypt_into(&nonce, associated, &produced, &mut recovered, &tag)
                .expect("the genuine tag must verify");
            assert_eq!(recovered.as_slice(), &plaintext[..length]);

            // In place must reach the same ciphertext and come back the same way.
            let mut in_place = plaintext[..length].to_vec();
            assert_eq!(key.encrypt_in_place(&nonce, associated, &mut in_place), expected_tag);
            assert_eq!(in_place, produced);
            key.decrypt_in_place(&nonce, associated, &mut in_place, &tag)
                .expect("the genuine tag must verify");
            assert_eq!(in_place.as_slice(), &plaintext[..length]);
        }
    }

    #[test]
    fn aes256_gcm_forged_tag() {
        let secret = [0x11u8; sz::AES256_KEY_LENGTH];
        let nonce = [0x22u8; sz::AES256_NONCE_LENGTH];
        let associated = b"routing header";
        let plaintext = b"the tag is the only thing between the reader and forged plaintext";
        let key = sz::Aes256GcmKey::new(&secret);

        let mut ciphertext = vec![0u8; plaintext.len()];
        let tag = key.encrypt_into(&nonce, associated, plaintext, &mut ciphertext);

        // A single flipped tag bit must be refused, and a caller who drops the error must not find
        // forged plaintext waiting in the buffer.
        let mut forged_tag = tag;
        forged_tag[0] ^= 0x01;
        let mut recovered = vec![0xA5u8; plaintext.len()];
        assert_eq!(
            key.decrypt_into(&nonce, associated, &ciphertext, &mut recovered, &forged_tag),
            Err(sz::AuthenticationError::TagMismatch)
        );
        assert!(recovered.iter().all(|byte| *byte == 0));

        // The associated data is authenticated too, so changing it must be refused just as loudly.
        let mut recovered = vec![0xA5u8; plaintext.len()];
        assert_eq!(
            key.decrypt_into(&nonce, b"forged header", &ciphertext, &mut recovered, &tag),
            Err(sz::AuthenticationError::TagMismatch)
        );
        assert!(recovered.iter().all(|byte| *byte == 0));

        // The genuine tag still recovers the message.
        let mut recovered = vec![0u8; plaintext.len()];
        key.decrypt_into(&nonce, associated, &ciphertext, &mut recovered, &tag)
            .expect("the genuine tag must verify");
        assert_eq!(recovered.as_slice(), &plaintext[..]);
    }

    #[test]
    fn aes256_gcm_streaming() {
        let secret = [0x9Eu8; sz::AES256_KEY_LENGTH];
        let nonce = [0x4Fu8; sz::AES256_NONCE_LENGTH];
        let associated = b"chunked records still authenticate their header";
        let plaintext: Vec<u8> = (0..300u32).map(|index| (index * 17 + 3) as u8).collect();
        let key = sz::Aes256GcmKey::new(&secret);

        let mut whole = vec![0u8; plaintext.len()];
        let whole_tag = key.encrypt_into(&nonce, associated, &plaintext, &mut whole);

        // A chunk boundary must be invisible: the keystream block and the hash block both straddle it.
        for chunk in [1usize, 7, 15, 16, 17, 64] {
            let mut streamed = vec![0u8; plaintext.len()];
            let mut encryptor = sz::Aes256GcmEncryptor::new(&key, &nonce);
            encryptor.associate(associated);
            for offset in (0..plaintext.len()).step_by(chunk) {
                let taken = chunk.min(plaintext.len() - offset);
                encryptor.encrypt_into(
                    &plaintext[offset..offset + taken],
                    &mut streamed[offset..offset + taken],
                );
            }
            assert_eq!(streamed, whole);
            assert_eq!(encryptor.digest(), whole_tag);

            let mut recovered = vec![0u8; plaintext.len()];
            let mut decryptor = sz::Aes256GcmDecryptor::new(&key, &nonce);
            decryptor.associate(associated);
            for offset in (0..plaintext.len()).step_by(chunk) {
                let taken = chunk.min(plaintext.len() - offset);
                decryptor
                    .decrypt_unverified_into(&whole[offset..offset + taken], &mut recovered[offset..offset + taken]);
            }
            decryptor.verify(&whole_tag).expect("the genuine tag must verify");
            assert_eq!(recovered, plaintext);
        }

        // The streaming path emits bytes before anything authenticates them, so the refusal comes
        // from `verify` and the caller is left holding output it must discard itself. That is the
        // risk `decrypt_unverified_into` names, and the reason the one-shot call exists.
        let mut forged_tag = whole_tag;
        forged_tag[15] ^= 0x80;
        let mut recovered = vec![0u8; plaintext.len()];
        let mut decryptor = sz::Aes256GcmDecryptor::new(&key, &nonce);
        decryptor.associate(associated);
        decryptor.decrypt_unverified_into(&whole, &mut recovered);
        assert_eq!(decryptor.verify(&forged_tag), Err(sz::AuthenticationError::TagMismatch));
        assert_eq!(recovered, plaintext);
    }

    /// A dropped encryptor must leave none of its payload behind, which a passing build cannot show.
    ///
    /// The encryptor lives in storage this test owns so the bytes can be read back. Only the named
    /// fields are checked; the six trailing padding bytes are never written by the C core.
    #[test]
    fn aes256_gcm_streaming_scrubs_on_drop() {
        use core::mem::{size_of, MaybeUninit};

        let secret = [0xA5u8; sz::AES256_KEY_LENGTH];
        let nonce = [0x5Au8; sz::AES256_NONCE_LENGTH];
        let key = sz::Aes256GcmKey::new(&secret);

        let mut slot = MaybeUninit::<sz::Aes256GcmEncryptor>::uninit();
        unsafe { slot.as_mut_ptr().write(sz::Aes256GcmEncryptor::new(&key, &nonce)) };

        // Drive it so the keystream and the pending block hold live material rather than zeros.
        let mut text = *b"material that must not survive the drop";
        unsafe { (*slot.as_mut_ptr()).encrypt_in_place(&mut text) };
        unsafe { core::ptr::drop_in_place(slot.as_mut_ptr()) };

        let payload =
            unsafe { core::slice::from_raw_parts(slot.as_ptr() as *const u8, size_of::<sz::Aes256GcmEncryptor>()) };
        let named_fields = &payload[..payload.len() - 6];
        let surviving = named_fields.iter().filter(|byte| **byte != 0).count();
        assert_eq!(
            surviving, 0,
            "{surviving} bytes of a dropped encryptor's payload survived"
        );
    }

    #[test]
    fn aes256_gcm_decryptor_alone() {
        let secret = [0x6Du8; sz::AES256_KEY_LENGTH];
        let nonce = [0x1Au8; sz::AES256_NONCE_LENGTH];
        let associated = b"a header the caller happens to hold in two pieces";
        let plaintext: Vec<u8> = (0..201u32).map(|index| (index * 29 + 11) as u8).collect();
        let key = sz::Aes256GcmKey::new(&secret);

        let mut ciphertext = vec![0u8; plaintext.len()];
        let tag = key.encrypt_into(&nonce, associated, &plaintext, &mut ciphertext);

        // The decryptor hashes the bytes it is given rather than the bytes it emits, so both emit
        // paths must reach the tag the one-shot call made, and the header may arrive in any pieces.
        let mut opened = vec![0u8; plaintext.len()];
        let mut decryptor = sz::Aes256GcmDecryptor::new(&key, &nonce);
        decryptor.associate(&associated[..17]);
        decryptor.associate(&associated[17..]);
        for offset in (0..ciphertext.len()).step_by(23) {
            let taken = 23.min(ciphertext.len() - offset);
            decryptor.decrypt_unverified_into(&ciphertext[offset..offset + taken], &mut opened[offset..offset + taken]);
        }
        decryptor.verify(&tag).expect("the genuine tag must verify");
        assert_eq!(opened, plaintext);

        // In place both buffers live at one address, which is the case the direction of the hash
        // could hide in, so it must agree byte for byte with the separate-buffer opening.
        let mut opened_in_place = ciphertext.clone();
        let mut decryptor = sz::Aes256GcmDecryptor::new(&key, &nonce);
        decryptor.associate(associated);
        for chunk in opened_in_place.chunks_mut(23) {
            decryptor.decrypt_unverified_in_place(chunk);
        }
        decryptor.verify(&tag).expect("the genuine tag must verify");
        assert_eq!(opened_in_place, plaintext);

        // A decryptor that never absorbed the header authenticates a shorter message than the sender
        // sealed, so the same tag must be refused.
        let mut unassociated = ciphertext.clone();
        let mut decryptor = sz::Aes256GcmDecryptor::new(&key, &nonce);
        decryptor.decrypt_unverified_in_place(&mut unassociated);
        assert_eq!(decryptor.verify(&tag), Err(sz::AuthenticationError::TagMismatch));

        // A decryptor started on the wrong nonce emits neither the plaintext nor a matching tag.
        let mut wrong_nonce = nonce;
        wrong_nonce[0] ^= 0x01;
        let mut mismatched = ciphertext.clone();
        let mut decryptor = sz::Aes256GcmDecryptor::new(&key, &wrong_nonce);
        decryptor.associate(associated);
        decryptor.decrypt_unverified_in_place(&mut mismatched);
        assert_eq!(decryptor.verify(&tag), Err(sz::AuthenticationError::TagMismatch));
        assert_ne!(mismatched, plaintext);
    }
}
