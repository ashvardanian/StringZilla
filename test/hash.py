#!/usr/bin/env python3
"""Hashing & cryptography tests: sz.hash / Hasher, multiseed, bytesum, SHA-256, HMAC-SHA-256.

Mirrors the C++ test/hash.cpp translation unit.
"""

import hashlib
import hmac

import pytest

import stringzilla as sz
from stringzilla import Str

from test.helpers import SEED_VALUES, seed_random_generators, get_random_string


@pytest.mark.parametrize("body", ["", "hello", "world", "abcdefg", "a" * 32])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hash_basic_equivalence(body: str, seed_value: int):
    # TODO: Add streaming hashers and compare slices vs overall
    hash_seeded = sz.hash(body, seed=seed_value)
    hash_member = sz.Str(body).hash(seed=seed_value)
    assert hash_seeded == hash_member


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hasher_incremental_vs_one_shot(seed_value: int):
    data_full = b"hello world"
    data_prefix = b"hello "
    data_suffix = b"world"

    hasher = sz.Hasher(seed=seed_value)
    hasher.update(data_prefix)
    hasher.update(data_suffix)
    streamed_hash = hasher.digest()

    expected_hash = sz.hash(data_full, seed=seed_value)
    assert isinstance(streamed_hash, int)
    assert streamed_hash == expected_hash


@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hasher_reset_and_hexdigest(seed_value: int):
    data = b"some test payload"
    hasher = sz.Hasher(seed=seed_value)
    hasher.update(data)
    streamed_hash = hasher.digest()
    streamed_hex = hasher.hexdigest()
    assert isinstance(streamed_hex, str) and len(streamed_hex) == 16 and streamed_hex == format(streamed_hash, "016x")

    hasher.reset()
    hasher.update(data)
    re_streamed_hash = hasher.digest()
    re_streamed_hex = hasher.hexdigest()
    assert streamed_hash == re_streamed_hash
    assert streamed_hex == re_streamed_hex


@pytest.mark.parametrize("body", ["", "x", "hello", "abcdefg", "a" * 17, "a" * 32, "a" * 64, "a" * 100])
def test_hash_multiseed_equivalence(body: str):
    from array import array

    seeds_list = [0, 1, 42, 314159, 7, 8, 9, 10, 11]  # > 4 to exercise the 4-wide tail handling
    seeds = array("Q", seeds_list)
    expected = tuple(sz.hash(body, s) for s in seeds_list)

    # Tuple-returning form, standalone and as a member of `Str`
    assert sz.hash_multiseed(body, seeds) == expected
    assert sz.Str(body).hash_multiseed(seeds) == expected

    # Output-buffer form fills in place and returns None
    out = array("Q", [0] * len(seeds))
    assert sz.hash_multiseed(body, seeds, out=out) is None
    assert tuple(out) == expected


def test_hash_multiseed_errors():
    from array import array

    seeds = array("Q", [1, 2, 3])
    with pytest.raises(TypeError):  # A plain list of ints is not a uint64 buffer
        sz.hash_multiseed("x", [1, 2, 3])
    with pytest.raises(TypeError):  # Wrong item size (32-bit) is rejected
        sz.hash_multiseed("x", array("I", [1, 2, 3]))
    with pytest.raises(ValueError):  # Output buffer too small for the seed count
        sz.hash_multiseed("x", seeds, out=array("Q", [0]))


@pytest.mark.parametrize("length", list(range(0, 300)) + [1024, 4096, 100000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_bytesum_random(length: int, seed_value: int):
    def sum_bytes(body: str) -> int:
        return sum([ord(c) for c in body])

    seed_random_generators(seed_value)
    body = get_random_string(length=length)
    assert sum_bytes(body) == sz.bytesum(body)


@pytest.mark.parametrize("length", [0, 1, 3, 7, 15, 31, 63, 64, 65, 127, 128, 129, 255, 256, 1000, 4096, 10000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_sha256(length: int, seed_value: int):

    seed_random_generators(seed_value)
    text = get_random_string(length=length)
    expected = hashlib.sha256(text.encode()).digest()

    # One-shot: standalone function
    assert sz.sha256(text) == expected

    # One-shot: Str method
    assert Str(text).sha256() == expected

    # One-shot: bytes input
    assert sz.sha256(text.encode()) == expected

    # Progressive: single update
    h = sz.Sha256()
    h.update(text)
    assert h.digest() == expected
    assert h.hexdigest() == expected.hex()

    # Progressive: chunked updates
    if length > 0:
        h = sz.Sha256()
        chunk_size = max(1, length // 3)
        for i in range(0, length, chunk_size):
            h.update(text[i : i + chunk_size])
        assert h.digest() == expected
        assert h.hexdigest() == expected.hex()

    # Reset
    h.reset().update(text)
    assert h.digest() == expected

    # Copy
    mid = length // 2
    h1 = sz.Sha256().update(text[:mid])
    h2 = h1.copy()
    h1.update(text[mid:])
    h2.update(text[mid:])
    assert h1.digest() == h2.digest() == expected


@pytest.mark.parametrize("key_length", [0, 1, 16, 32, 64, 65, 128])
@pytest.mark.parametrize("message_length", [0, 1, 63, 64, 65, 127, 128, 1000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hmac_sha256(key_length: int, message_length: int, seed_value: int):

    seed_random_generators(seed_value)
    key = get_random_string(length=key_length).encode()
    message = get_random_string(length=message_length).encode()

    # Test against Python's hmac module
    expected = hmac.new(key, message, hashlib.sha256).digest()
    result = sz.hmac_sha256(key, message)
    assert result == expected

    # Test with string inputs
    key_str = key.decode("latin1")
    message_str = message.decode("latin1")
    result_str = sz.hmac_sha256(key_str, message_str)
    assert result_str == expected


def test_hmac_sha256_kwargs():
    """Test hmac_sha256 with keyword arguments"""
    key = b"secret"
    message = b"Hello, world!"

    # Test against Python's hmac module
    expected = hmac.new(key, message, hashlib.sha256).digest()

    # Test with positional arguments
    result_positional = sz.hmac_sha256(key, message)
    assert result_positional == expected

    # Test with keyword arguments (as shown in README line 483)
    result_kwargs = sz.hmac_sha256(key=key, message=message)
    assert result_kwargs == expected

    # Test with mixed arguments
    result_mixed = sz.hmac_sha256(key, message=message)
    assert result_mixed == expected

    # Test with reversed keyword arguments
    result_reversed = sz.hmac_sha256(message=message, key=key)
    assert result_reversed == expected

    # Missing argument
    with pytest.raises(TypeError, match="expects exactly 2 arguments"):
        sz.hmac_sha256(key=key)

    # Duplicate argument
    with pytest.raises(TypeError, match="key specified twice"):
        sz.hmac_sha256(key, key=key)

    # Unknown keyword argument (only detected when total args == 2)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        sz.hmac_sha256(key=key, unknown=b"test")

    # Too many arguments (3 args)
    with pytest.raises(TypeError, match="expects exactly 2 arguments"):
        sz.hmac_sha256(key=key, message=message, unknown=b"test")
