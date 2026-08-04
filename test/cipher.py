#!/usr/bin/env python3
"""AES-256 ciphers: the seekable `sz.Aes256CtrKey`, the authenticated `sz.Aes256GcmKey`, and its chunked
`sz.Aes256GcmEncryptor` / `sz.Aes256GcmDecryptor`.

Mirrors the C++ test/cipher.cpp translation unit, sweeping lengths, seek offsets, and chunk sizes far
more densely than a compiled suite comfortably can.

Covers: published known-answer vectors for both modes, the seekable property at every byte offset,
round-trips over every length crossed with every associated-data length, forged tags and mutated
inputs, chunked streaming at every chunk size, and the argument surface.
Compares against: NIST SP 800-38A F.5.5 for counter mode and the McGrew-Viega vectors NIST SP 800-38D
is validated against for Galois/counter mode, the same values `test/cipher.cpp` spells out, plus
serial-versus-dispatched agreement, and - when it is importable - pycryptodome's independent C
implementation, which links only libc rather than wrapping the OpenSSL the C++ suite already uses.

Run:
    uv pip install pytest pytest-repeat
    uv pip install pycryptodome  # Optional, unlocks the third-party differential
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/cipher.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/cipher.py -q
"""

import pytest

import stringzilla as sz

from test.sz_helpers import (
    SEED_VALUES,
    VECTOR_WIDTH_LENGTHS,
    assert_backends_agree,
    capability_sweep,
    differential_bodies,
    forced_capabilities,
    get_random_string,
    run_across_backends,
    scale_iterations,
    seed_random_generators,
)

SECRET_LENGTH = 32
NONCE_LENGTH = 12
TAG_LENGTH = 16

# The secrets and nonces the sweeps reuse, spelled out the same way `test/cipher.cpp` builds them so a
# divergence between the two suites is a real one. Repeating a nonce is only acceptable because every
# case here is compared against a locally computed expectation and no message ever leaves the process.
COUNTER_SECRET = bytes((index * 7 + 1) & 0xFF for index in range(SECRET_LENGTH))
COUNTER_NONCE = bytes((index * 5 + 2) & 0xFF for index in range(NONCE_LENGTH))
AUTHENTICATED_SECRET = bytes((index * 3 + 5) & 0xFF for index in range(SECRET_LENGTH))
AUTHENTICATED_NONCE = bytes((index + 9) & 0xFF for index in range(NONCE_LENGTH))

# NIST SP 800-38A F.5.5, CTR-AES256.Encrypt. That vector names a full 128-bit initial counter block
# incremented across its whole width, while `Aes256CtrKey` builds its counter from a 12-byte nonce and a
# 32-bit block index starting at zero. The two coincide exactly when the stream is entered at the block
# index the published counter spells out, which is what `offset` reaches, and the low 32 bits carry
# nowhere across these four blocks.
COUNTER_VECTOR_SECRET = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
COUNTER_VECTOR_NONCE = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafb")
COUNTER_VECTOR_BLOCK_INDEX = 0xFCFDFEFF
COUNTER_VECTOR_PLAINTEXT = bytes.fromhex(
    "6bc1bee22e409f96e93d7e117393172a"
    "ae2d8a571e03ac9c9eb76fac45af8e51"
    "30c81c46a35ce411e5fbc1191a0a52ef"
    "f69f2445df4f9b17ad2b417be66c3710"
)
COUNTER_VECTOR_CIPHERTEXT = bytes.fromhex(
    "601ec313775789a5b7a7f504bbf3d228"
    "f443e3ca4d62b59aca84e990cacaf5c5"
    "2b0930daa23de94ce87017ba2d84988d"
    "dfc9c58db67aada613c2dd08457941a6"
)

# Cases 13 through 16 of McGrew and Viega's Galois/counter mode note, the AES-256 vectors NIST's own
# validation suite is built from, byte for byte the same literals `test/cipher.cpp` carries.
AUTHENTICATED_VECTORS = [
    # Empty message, empty associated data
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000",
        "",
        "",
        "",
        "530f8afbc74536b9a963b4f1c4cb738b",
    ),
    # One zero block, empty associated data
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000",
        "",
        "00000000000000000000000000000000",
        "cea7403d4d606b6e074ec5d3baf39d18",
        "d0d1c8a799996bf0265b98b5d48ab919",
    ),
    # Four whole blocks, empty associated data
    (
        "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
        "cafebabefacedbaddecaf888",
        "",
        "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
        "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255",
        "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"
        "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad",
        "b094dac5d93471bdec1a502270e3cc6c",
    ),
    # A partial trailing block plus associated data, which exercises both zero pads
    (
        "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
        "cafebabefacedbaddecaf888",
        "feedfacedeadbeeffeedfacedeadbeefabaddad2",
        "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
        "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
        "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"
        "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",
        "76fc6ece0f4e1768cddf8853bb2d551b",
    ),
]

AUTHENTICATED_VECTOR_NAMES = ["empty", "one-block", "four-blocks", "partial-block-with-associated"]

# Associated-data lengths bracketing the 16-byte hash block, crossed with message lengths below.
ASSOCIATED_LENGTHS = [0, 1, 15, 16, 17, 40]


def random_bytes(length: int, seed_value: int) -> bytes:
    """Deterministic body of `length` bytes, seeded so a failing case reproduces from its identifier."""
    seed_random_generators(seed_value)
    return get_random_string(length=length).encode()


# region Unit


@pytest.mark.parametrize("vector", AUTHENTICATED_VECTORS, ids=AUTHENTICATED_VECTOR_NAMES)
def test_gcm_known_answers(vector):
    """Each published Galois/counter mode vector encrypts to its stated ciphertext and tag, decrypts
    back to its plaintext, and refuses a tag with one bit flipped."""
    secret_hex, nonce_hex, associated_hex, plaintext_hex, ciphertext_hex, tag_hex = vector
    secret, nonce = bytes.fromhex(secret_hex), bytes.fromhex(nonce_hex)
    associated, plaintext = bytes.fromhex(associated_hex), bytes.fromhex(plaintext_hex)
    expected_ciphertext, expected_tag = bytes.fromhex(ciphertext_hex), bytes.fromhex(tag_hex)

    key = sz.Aes256GcmKey(secret)
    ciphertext, tag = key.encrypt(plaintext, nonce, associated)
    assert ciphertext == expected_ciphertext
    assert tag == expected_tag

    assert key.decrypt(ciphertext, nonce, tag, associated) == plaintext

    forged_tag = bytes([tag[0] ^ 0x01]) + tag[1:]
    with pytest.raises(sz.AuthenticationError):
        key.decrypt(ciphertext, nonce, forged_tag, associated)


def test_ctr_known_answer():
    """The counter-mode vector encrypts to its stated ciphertext, and the same call run again on that
    ciphertext returns the plaintext, since the mode is its own inverse."""
    key = sz.Aes256CtrKey(COUNTER_VECTOR_SECRET)
    offset = COUNTER_VECTOR_BLOCK_INDEX * 16
    ciphertext = key.xor(COUNTER_VECTOR_PLAINTEXT, COUNTER_VECTOR_NONCE, offset=offset)
    assert ciphertext == COUNTER_VECTOR_CIPHERTEXT
    assert key.xor(ciphertext, COUNTER_VECTOR_NONCE, offset=offset) == COUNTER_VECTOR_PLAINTEXT


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_ctr_every_length_and_offset(seed_value: int):
    """Every byte offset into the keystream must land on the bytes a from-zero encryption produced
    there, which is the whole reason counter mode is exposed apart from the authenticated one, and
    every message length must transform without disturbing that alignment."""
    span = scale_iterations(1024)
    body = random_bytes(span, seed_value)
    key = sz.Aes256CtrKey(COUNTER_SECRET)
    whole = key.xor(body, COUNTER_NONCE)
    assert len(whole) == span

    # A slice taken at any offset, of any length, must equal that slice of the whole stream.
    offsets = range(0, min(span, scale_iterations(257)))
    for offset in offsets:
        assert key.xor(body[offset:], COUNTER_NONCE, offset=offset) == whole[offset:]
        for length in (0, 1, 15, 16, 17, 31, 33, 64):
            end = min(offset + length, span)
            assert key.xor(body[offset:end], COUNTER_NONCE, offset=offset) == whole[offset:end]

    # Decrypting a slice of the ciphertext must return that slice of the plaintext.
    for offset in offsets:
        assert key.xor(whole[offset:], COUNTER_NONCE, offset=offset) == body[offset:]


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_ctr_every_length_round_trips(seed_value: int):
    """Every length from zero up survives an encrypt followed by a decrypt, and yields as many bytes
    as it consumed."""
    key = sz.Aes256CtrKey(COUNTER_SECRET)
    longest = scale_iterations(200)
    for length in range(longest + 1):
        body = random_bytes(length, seed_value + length)
        ciphertext = key.xor(body, COUNTER_NONCE)
        assert len(ciphertext) == length
        assert key.xor(ciphertext, COUNTER_NONCE) == body


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_gcm_every_length_round_trips(seed_value: int):
    """Every length from zero up encrypts and decrypts back, with the associated-data length rotating
    across the sizes that bracket the 16-byte hash block."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    longest = scale_iterations(200)
    for length in range(longest + 1):
        body = random_bytes(length, seed_value + length)
        associated_length = ASSOCIATED_LENGTHS[length % len(ASSOCIATED_LENGTHS)]
        associated = random_bytes(associated_length, seed_value + length + 1)

        ciphertext, tag = key.encrypt(body, AUTHENTICATED_NONCE, associated)
        assert len(ciphertext) == length
        assert len(tag) == TAG_LENGTH
        assert key.decrypt(ciphertext, AUTHENTICATED_NONCE, tag, associated) == body


@pytest.mark.parametrize("associated_length", ASSOCIATED_LENGTHS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_gcm_associated_crossed_with_message(associated_length: int, seed_value: int):
    """Every associated-data length crossed with every message length round-trips, and a message
    authenticated under one associated value is refused under any other."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    associated = random_bytes(associated_length, seed_value)

    for length in VECTOR_WIDTH_LENGTHS:
        body = random_bytes(length, seed_value + length)
        ciphertext, tag = key.encrypt(body, AUTHENTICATED_NONCE, associated)
        assert key.decrypt(ciphertext, AUTHENTICATED_NONCE, tag, associated) == body

        # Associated data is authenticated, so changing it must invalidate the tag.
        other_associated = associated + b"\x00"
        with pytest.raises(sz.AuthenticationError):
            key.decrypt(ciphertext, AUTHENTICATED_NONCE, tag, other_associated)


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_gcm_omitted_associated_matches_empty(seed_value: int):
    """Leaving `associated` out, passing None, and passing empty bytes are the same message."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    body = random_bytes(64, seed_value)
    without = key.encrypt(body, AUTHENTICATED_NONCE)
    assert without == key.encrypt(body, AUTHENTICATED_NONCE, None)
    assert without == key.encrypt(body, AUTHENTICATED_NONCE, b"")


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_gcm_forged_inputs_are_refused(seed_value: int):
    """A tag with any single byte disturbed, a mutated ciphertext, a different nonce, and a truncated
    ciphertext must every one of them raise rather than hand back plaintext."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    body = random_bytes(70, seed_value)
    associated = random_bytes(17, seed_value + 1)
    ciphertext, tag = key.encrypt(body, AUTHENTICATED_NONCE, associated)

    for index in range(TAG_LENGTH):
        forged = bytearray(tag)
        forged[index] ^= 0x80
        with pytest.raises(sz.AuthenticationError):
            key.decrypt(ciphertext, AUTHENTICATED_NONCE, bytes(forged), associated)

    for index in range(len(ciphertext)):
        mutated = bytearray(ciphertext)
        mutated[index] ^= 0x01
        with pytest.raises(sz.AuthenticationError):
            key.decrypt(bytes(mutated), AUTHENTICATED_NONCE, tag, associated)

    other_nonce = bytes([AUTHENTICATED_NONCE[0] ^ 0x01]) + AUTHENTICATED_NONCE[1:]
    with pytest.raises(sz.AuthenticationError):
        key.decrypt(ciphertext, other_nonce, tag, associated)

    with pytest.raises(sz.AuthenticationError):
        key.decrypt(ciphertext[:-1], AUTHENTICATED_NONCE, tag, associated)

    other_key = sz.Aes256GcmKey(bytes([AUTHENTICATED_SECRET[0] ^ 0x01]) + AUTHENTICATED_SECRET[1:])
    with pytest.raises(sz.AuthenticationError):
        other_key.decrypt(ciphertext, AUTHENTICATED_NONCE, tag, associated)


def test_authentication_error_is_a_value_error():
    """The refusal is a `ValueError` subclass, so a caller guarding only against malformed input still
    stops a forgery instead of reading a sentinel."""
    assert issubclass(sz.AuthenticationError, ValueError)
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    ciphertext, _ = key.encrypt(b"payload", AUTHENTICATED_NONCE)
    with pytest.raises(ValueError):
        key.decrypt(ciphertext, AUTHENTICATED_NONCE, bytes(TAG_LENGTH))


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_gcm_streaming_at_every_chunk_size(seed_value: int):
    """A chunk boundary must be invisible: the keystream block and the hash block are both sixteen
    bytes wide and a caller's chunks are not, so every chunk size must reproduce the one-shot
    ciphertext and tag, encrypting and decrypting alike."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    span = scale_iterations(512)
    body = random_bytes(span, seed_value)
    associated = random_bytes(23, seed_value + 1)
    expected_ciphertext, expected_tag = key.encrypt(body, AUTHENTICATED_NONCE, associated)

    for chunk_size in range(1, 41):
        encryptor = sz.Aes256GcmEncryptor(key, AUTHENTICATED_NONCE)
        encryptor.associate(associated)
        pieces = [encryptor.encrypt(body[start : start + chunk_size]) for start in range(0, span, chunk_size)]
        assert b"".join(pieces) == expected_ciphertext
        assert encryptor.digest() == expected_tag

        decryptor = sz.Aes256GcmDecryptor(key, AUTHENTICATED_NONCE)
        decryptor.associate(associated)
        pieces = [
            decryptor.decrypt_unverified(expected_ciphertext[start : start + chunk_size])
            for start in range(0, span, chunk_size)
        ]
        assert b"".join(pieces) == body
        assert decryptor.verify(expected_tag) is None


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_gcm_streaming_associated_in_chunks(seed_value: int):
    """Associated data delivered in chunks of every size authenticates the same as one call."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    body = random_bytes(96, seed_value)
    associated = random_bytes(40, seed_value + 1)
    expected_ciphertext, expected_tag = key.encrypt(body, AUTHENTICATED_NONCE, associated)

    for chunk_size in range(1, 21):
        encryptor = sz.Aes256GcmEncryptor(key, AUTHENTICATED_NONCE)
        for start in range(0, len(associated), chunk_size):
            encryptor.associate(associated[start : start + chunk_size])
        assert encryptor.encrypt(body) == expected_ciphertext
        assert encryptor.digest() == expected_tag


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_gcm_streaming_refuses_a_forged_tag(seed_value: int):
    """A streaming decryption that consumed tampered ciphertext must refuse at `verify`."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    body = random_bytes(100, seed_value)
    ciphertext, tag = key.encrypt(body, AUTHENTICATED_NONCE)

    mutated = bytearray(ciphertext)
    mutated[len(mutated) // 2] ^= 0x40
    decryptor = sz.Aes256GcmDecryptor(key, AUTHENTICATED_NONCE)
    decryptor.decrypt_unverified(bytes(mutated))
    with pytest.raises(sz.AuthenticationError):
        decryptor.verify(tag)


def test_gcm_direction_is_carried_by_the_type():
    """The direction is the type, not a flag, so the operations belonging to the other direction are
    absent rather than guarded: neither object can be pointed the wrong way at all."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)

    encryptor = sz.Aes256GcmEncryptor(key, AUTHENTICATED_NONCE)
    assert not hasattr(encryptor, "decrypt_unverified")
    assert not hasattr(encryptor, "verify")

    decryptor = sz.Aes256GcmDecryptor(key, AUTHENTICATED_NONCE)
    assert not hasattr(decryptor, "encrypt")
    assert not hasattr(decryptor, "digest")

    # The retired flag must not survive as a silently ignored keyword on either type.
    for chunked_type in (sz.Aes256GcmEncryptor, sz.Aes256GcmDecryptor):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            chunked_type(key, AUTHENTICATED_NONCE, decrypting=True)


def test_gcm_chunked_outlives_its_key():
    """Both chunked types copy the key, so dropping the key object cannot leave one reading freed
    memory or a wiped schedule."""
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    expected_ciphertext, expected_tag = key.encrypt(b"payload", AUTHENTICATED_NONCE)

    encryptor = sz.Aes256GcmEncryptor(key, AUTHENTICATED_NONCE)
    decryptor = sz.Aes256GcmDecryptor(key, AUTHENTICATED_NONCE)
    del key
    assert encryptor.encrypt(b"payload") == expected_ciphertext
    assert encryptor.digest() == expected_tag
    assert decryptor.decrypt_unverified(expected_ciphertext) == b"payload"
    assert decryptor.verify(expected_tag) is None


# endregion Unit


# region Argument surface


def test_secret_length_is_checked():
    """Both key types demand exactly 32 secret bytes and name the shortfall."""
    for key_type in (sz.Aes256CtrKey, sz.Aes256GcmKey):
        for wrong_length in (0, 1, 16, 31, 33, 64):
            with pytest.raises(ValueError, match="secret must be exactly 32 bytes"):
                key_type(bytes(wrong_length))
        with pytest.raises(TypeError):
            key_type()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            key_type(secret=bytes(SECRET_LENGTH), unknown=1)
        with pytest.raises(TypeError, match="specified twice"):
            key_type(bytes(SECRET_LENGTH), secret=bytes(SECRET_LENGTH))
        assert key_type(secret=bytes(SECRET_LENGTH)) is not None


def test_nonce_and_tag_lengths_are_checked():
    """A nonce is 12 bytes and a tag is 16, everywhere they are accepted."""
    counter_key = sz.Aes256CtrKey(COUNTER_SECRET)
    authenticated_key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    ciphertext, tag = authenticated_key.encrypt(b"payload", AUTHENTICATED_NONCE)

    for wrong_length in (0, 11, 13, 16):
        with pytest.raises(ValueError, match="nonce must be exactly 12 bytes"):
            counter_key.xor(b"payload", bytes(wrong_length))
        with pytest.raises(ValueError, match="nonce must be exactly 12 bytes"):
            authenticated_key.encrypt(b"payload", bytes(wrong_length))
        with pytest.raises(ValueError, match="nonce must be exactly 12 bytes"):
            authenticated_key.decrypt(ciphertext, bytes(wrong_length), tag)
        with pytest.raises(ValueError, match="nonce must be exactly 12 bytes"):
            sz.Aes256GcmEncryptor(authenticated_key, bytes(wrong_length))
        with pytest.raises(ValueError, match="nonce must be exactly 12 bytes"):
            sz.Aes256GcmDecryptor(authenticated_key, bytes(wrong_length))

    for wrong_length in (0, 15, 17, 32):
        with pytest.raises(ValueError, match="tag must be exactly 16 bytes"):
            authenticated_key.decrypt(ciphertext, AUTHENTICATED_NONCE, bytes(wrong_length))


def test_keyword_and_positional_arguments_agree():
    """Every entry point accepts its arguments positionally, by name, and mixed, and rejects a
    duplicate or unknown name."""
    counter_key = sz.Aes256CtrKey(COUNTER_SECRET)
    positional = counter_key.xor(b"payload", COUNTER_NONCE, 16)
    assert positional == counter_key.xor(b"payload", COUNTER_NONCE, offset=16)
    assert positional == counter_key.xor(b"payload", nonce=COUNTER_NONCE, offset=16)
    assert positional == counter_key.xor(text=b"payload", nonce=COUNTER_NONCE, offset=16)

    with pytest.raises(TypeError, match="nonce specified twice"):
        counter_key.xor(b"payload", COUNTER_NONCE, nonce=COUNTER_NONCE)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        counter_key.xor(b"payload", COUNTER_NONCE, unknown=1)
    with pytest.raises(TypeError):
        counter_key.xor(b"payload")

    authenticated_key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    ciphertext, tag = authenticated_key.encrypt(b"payload", AUTHENTICATED_NONCE, b"header")
    assert (ciphertext, tag) == authenticated_key.encrypt(
        text=b"payload", nonce=AUTHENTICATED_NONCE, associated=b"header"
    )
    assert (
        authenticated_key.decrypt(text=ciphertext, nonce=AUTHENTICATED_NONCE, tag=tag, associated=b"header")
        == b"payload"
    )

    with pytest.raises(TypeError, match="tag specified twice"):
        authenticated_key.decrypt(ciphertext, AUTHENTICATED_NONCE, tag, tag=tag)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        authenticated_key.encrypt(b"payload", AUTHENTICATED_NONCE, unknown=1)

    encryptor = sz.Aes256GcmEncryptor(key=authenticated_key, nonce=AUTHENTICATED_NONCE)
    assert encryptor.encrypt(b"payload") == ciphertext
    decryptor = sz.Aes256GcmDecryptor(authenticated_key, nonce=AUTHENTICATED_NONCE)
    assert decryptor.decrypt_unverified(ciphertext) == b"payload"

    for chunked_type in (sz.Aes256GcmEncryptor, sz.Aes256GcmDecryptor):
        with pytest.raises(TypeError, match="must be an Aes256GcmKey object"):
            chunked_type(sz.Aes256CtrKey(COUNTER_SECRET), AUTHENTICATED_NONCE)
        with pytest.raises(TypeError, match="nonce specified twice"):
            chunked_type(authenticated_key, AUTHENTICATED_NONCE, nonce=AUTHENTICATED_NONCE)


def test_buffer_protocol_inputs_agree():
    """`str`, `bytes`, `bytearray`, and `memoryview` carrying the same bytes are the same message."""
    counter_key = sz.Aes256CtrKey(COUNTER_SECRET)
    authenticated_key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    text = "the second half is decrypted on its own"
    payload = text.encode()

    expected = counter_key.xor(payload, COUNTER_NONCE)
    assert counter_key.xor(text, COUNTER_NONCE) == expected
    assert counter_key.xor(bytearray(payload), COUNTER_NONCE) == expected
    assert counter_key.xor(memoryview(payload), COUNTER_NONCE) == expected
    assert counter_key.xor(sz.Str(text), COUNTER_NONCE) == expected

    expected = authenticated_key.encrypt(payload, AUTHENTICATED_NONCE, payload)
    assert authenticated_key.encrypt(text, AUTHENTICATED_NONCE, text) == expected
    assert authenticated_key.encrypt(bytearray(payload), AUTHENTICATED_NONCE, bytearray(payload)) == expected
    assert authenticated_key.encrypt(memoryview(payload), AUTHENTICATED_NONCE, memoryview(payload)) == expected


def test_empty_inputs_are_accepted():
    """A zero-length message is a valid message in both modes, and still carries a tag."""
    assert sz.Aes256CtrKey(COUNTER_SECRET).xor(b"", COUNTER_NONCE) == b""
    ciphertext, tag = sz.Aes256GcmKey(AUTHENTICATED_SECRET).encrypt(b"", AUTHENTICATED_NONCE)
    assert ciphertext == b""
    assert len(tag) == TAG_LENGTH


# endregion Argument surface


# region Backend differential


@pytest.mark.parametrize("vector", AUTHENTICATED_VECTORS, ids=AUTHENTICATED_VECTOR_NAMES)
def test_unit_backend_differential_known_answers(vector):
    """Every backend must reach the published vector, not merely agree with the others, which is the
    only check a shared mistake in the vectorized kernels cannot survive."""
    secret_hex, nonce_hex, associated_hex, plaintext_hex, ciphertext_hex, tag_hex = vector
    secret, nonce = bytes.fromhex(secret_hex), bytes.fromhex(nonce_hex)
    associated, plaintext = bytes.fromhex(associated_hex), bytes.fromhex(plaintext_hex)
    expected_ciphertext, expected_tag = bytes.fromhex(ciphertext_hex), bytes.fromhex(tag_hex)

    for config in capability_sweep():
        with forced_capabilities(*config):
            # The schedule is expanded when the key is built, so the key must be built in here too.
            key = sz.Aes256GcmKey(secret)
            assert key.encrypt(plaintext, nonce, associated) == (expected_ciphertext, expected_tag)
            assert key.decrypt(expected_ciphertext, nonce, expected_tag, associated) == plaintext

            counter_key = sz.Aes256CtrKey(COUNTER_VECTOR_SECRET)
            offset = COUNTER_VECTOR_BLOCK_INDEX * 16
            transformed = counter_key.xor(COUNTER_VECTOR_PLAINTEXT, COUNTER_VECTOR_NONCE, offset=offset)
            assert transformed == COUNTER_VECTOR_CIPHERTEXT


@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_ctr(length: int, seed_value: int):
    """Counter mode has no standard-library oracle, so every backend must agree with every other, for
    random, tiled, and all-same-character bodies, at offsets that split the first keystream block."""
    for body in differential_bodies(length, seed_value):
        payload = body.encode()
        for offset in (0, 1, 15, 16, 17, 4096):

            def transform(payload=payload, offset=offset):
                return sz.Aes256CtrKey(COUNTER_SECRET).xor(payload, COUNTER_NONCE, offset=offset)

            assert_backends_agree(
                run_across_backends(transform),
                format_inputs=lambda length=length, offset=offset: f"length={length} offset={offset}",
            )


@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_gcm(length: int, seed_value: int):
    """Authenticated encryption and decryption must agree across every backend, for random, tiled, and
    all-same-character bodies crossed with the associated-data lengths that bracket a hash block."""
    for body in differential_bodies(length, seed_value):
        payload = body.encode()
        for associated_length in ASSOCIATED_LENGTHS:
            associated = random_bytes(associated_length, seed_value + associated_length)

            def encrypt(payload=payload, associated=associated):
                return sz.Aes256GcmKey(AUTHENTICATED_SECRET).encrypt(payload, AUTHENTICATED_NONCE, associated)

            assert_backends_agree(
                run_across_backends(encrypt),
                format_inputs=lambda: f"length={length} associated_length={associated_length}",
            )

            ciphertext, tag = encrypt()

            def decrypt(ciphertext=ciphertext, tag=tag, associated=associated):
                key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
                return key.decrypt(ciphertext, AUTHENTICATED_NONCE, tag, associated)

            assert_backends_agree(run_across_backends(decrypt), oracle=payload)


@pytest.mark.parametrize("chunk_size", [1, 7, 15, 16, 17, 31, 64, 100])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_gcm_streaming(chunk_size: int, seed_value: int):
    """The chunked state machine must agree across backends and match the one-shot call, since the
    keystream and hash blocks it carries between calls are where a vectorized tail goes wrong."""
    span = 512
    body = random_bytes(span, seed_value)
    associated = random_bytes(19, seed_value + 1)

    def streamed(body=body, associated=associated):
        key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
        encryptor = sz.Aes256GcmEncryptor(key, AUTHENTICATED_NONCE)
        encryptor.associate(associated)
        pieces = [encryptor.encrypt(body[start : start + chunk_size]) for start in range(0, span, chunk_size)]
        return b"".join(pieces), encryptor.digest()

    def one_shot(body=body, associated=associated):
        return sz.Aes256GcmKey(AUTHENTICATED_SECRET).encrypt(body, AUTHENTICATED_NONCE, associated)

    assert_backends_agree(run_across_backends(streamed), oracle=one_shot())


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_ctr_seek(seed_value: int):
    """The seek property must hold under every backend, not only under the one the host picks."""
    span = 512
    body = random_bytes(span, seed_value)

    for config in capability_sweep():
        with forced_capabilities(*config):
            key = sz.Aes256CtrKey(COUNTER_SECRET)
            whole = key.xor(body, COUNTER_NONCE)
            for offset in range(0, 130):
                assert key.xor(body[offset:], COUNTER_NONCE, offset=offset) == whole[offset:]


# endregion Backend differential


# region Third-party differential

# Chunk sizes for the streaming stress, none of them a divisor of 16, so a boundary lands inside a
# keystream block, inside a hash block, and astride both.
STREAMING_CHUNK_SIZES = [1, 3, 7, 13, 15, 17, 31, 33]


def counter_cipher_from_pycryptodome(aes, secret: bytes, nonce: bytes, byte_offset: int):
    """A pycryptodome counter-mode object entered at `byte_offset`, its partial block already spent.

    `Aes256CtrKey` builds its counter block from the 12 nonce bytes followed by a big-endian 32-bit block
    index, which is what `initial_value` spells once `nonce` is emptied, and a byte offset inside a
    block is reached by discarding that many keystream bytes.
    """
    block_index, inside_block = divmod(byte_offset, 16)
    initial_value = nonce + block_index.to_bytes(4, "big")
    cipher = aes.new(secret, aes.MODE_CTR, nonce=b"", initial_value=initial_value)
    if inside_block:
        cipher.encrypt(bytes(inside_block))
    return cipher


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_stress_ctr_matches_pycryptodome(seed_value: int):
    """Counter mode must reach byte for byte what an independent implementation reaches, at every
    length and every seek offset, including offsets that land mid-block."""
    aes = pytest.importorskip("Crypto.Cipher.AES")
    key = sz.Aes256CtrKey(COUNTER_SECRET)

    for length in VECTOR_WIDTH_LENGTHS:
        body = random_bytes(length, seed_value + length)
        for byte_offset in (0, 1, 15, 16, 17, 63, 4096, 16 * 0xFCFDFEFF):
            oracle = counter_cipher_from_pycryptodome(aes, COUNTER_SECRET, COUNTER_NONCE, byte_offset)
            assert key.xor(body, COUNTER_NONCE, offset=byte_offset) == oracle.encrypt(body)


@pytest.mark.parametrize("associated_length", ASSOCIATED_LENGTHS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_stress_gcm_matches_pycryptodome(associated_length: int, seed_value: int):
    """One-shot authenticated encryption must reach the same ciphertext and the same tag as an
    independent implementation, over every message length crossed with every associated length."""
    aes = pytest.importorskip("Crypto.Cipher.AES")
    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    associated = random_bytes(associated_length, seed_value)

    for length in VECTOR_WIDTH_LENGTHS:
        body = random_bytes(length, seed_value + length)
        oracle = aes.new(AUTHENTICATED_SECRET, aes.MODE_GCM, nonce=AUTHENTICATED_NONCE)
        oracle.update(associated)
        expected_ciphertext, expected_tag = oracle.encrypt_and_digest(body)

        assert key.encrypt(body, AUTHENTICATED_NONCE, associated) == (expected_ciphertext, expected_tag)
        assert key.decrypt(expected_ciphertext, AUTHENTICATED_NONCE, expected_tag, associated) == body


@pytest.mark.parametrize("chunk_size", STREAMING_CHUNK_SIZES)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_stress_gcm_streaming_matches_pycryptodome(chunk_size: int, seed_value: int):
    """Chunk boundaries are where a carried keystream block or a half-filled hash block goes wrong, so
    both chunked types are held against an independent one-shot rather than against our own."""
    aes = pytest.importorskip("Crypto.Cipher.AES")
    span = scale_iterations(300)
    body = random_bytes(span, seed_value)
    associated = random_bytes(29, seed_value + 1)

    oracle = aes.new(AUTHENTICATED_SECRET, aes.MODE_GCM, nonce=AUTHENTICATED_NONCE)
    oracle.update(associated)
    expected_ciphertext, expected_tag = oracle.encrypt_and_digest(body)

    key = sz.Aes256GcmKey(AUTHENTICATED_SECRET)
    encryptor = sz.Aes256GcmEncryptor(key, AUTHENTICATED_NONCE)
    for start in range(0, len(associated), chunk_size):
        encryptor.associate(associated[start : start + chunk_size])
    sealed = [encryptor.encrypt(body[start : start + chunk_size]) for start in range(0, span, chunk_size)]
    assert b"".join(sealed) == expected_ciphertext
    assert encryptor.digest() == expected_tag

    decryptor = sz.Aes256GcmDecryptor(key, AUTHENTICATED_NONCE)
    for start in range(0, len(associated), chunk_size):
        decryptor.associate(associated[start : start + chunk_size])
    opened = [
        decryptor.decrypt_unverified(expected_ciphertext[start : start + chunk_size])
        for start in range(0, span, chunk_size)
    ]
    assert b"".join(opened) == body
    assert decryptor.verify(expected_tag) is None


# endregion Third-party differential
