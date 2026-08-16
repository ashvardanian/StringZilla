#!/usr/bin/env python3
"""Library and version properties, DeviceScope, to_device across Strs layouts, and engine parameter validation.

Mirrors the C++ test/stringzillas.cu translation unit.

Covers: sz and szs version and capability strings, DeviceScope construction from cpu_cores
and gpu_device, to_device on whole, sliced, and Unicode Strs collections including nested
slices, and constructor and call-time parameter validation for LevenshteinDistances and
Fingerprints.
Compares against: cross-consistency between the sz and szs capability reports, and identity
and byte-level equality between the original and device-transferred Strs contents.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat affine-gaps
    SZ_TARGET=stringzillas-cpus uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/stringzillas.py -q
    SZ_TESTS_SEED=42 SZ_TESTS_MULTIPLIER=10 uv run --no-project python -m pytest test/stringzillas.py -q
    SZ_TARGET=stringzillas-cuda uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/stringzillas.py -q

Add -s -vv --maxfail=1 --full-trace -k <pattern> -X faulthandler for deeper diagnostics.
"""

import sys
import threading

import pytest

import stringzilla as sz
import stringzillas as szs
from stringzilla import Strs


# region Unit


def test_library_properties():
    """`sz` and `szs` each report a three-part semantic version and a non-empty capabilities
    string, and both accept their own capabilities back through `reset_capabilities` without raising."""
    assert len(sz.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in sz.__capabilities__, "Serial backend must be present"
    assert isinstance(sz.__capabilities_str__, str) and len(sz.__capabilities_str__) > 0
    sz.reset_capabilities(sz.__capabilities__)  # Should not raise

    # Test StringZillas properties
    assert len(szs.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in szs.__capabilities__, "Serial backend must be present"
    assert isinstance(szs.__capabilities_str__, str) and len(szs.__capabilities_str__) > 0
    sz.reset_capabilities(szs.__capabilities__)  # Should not raise


def test_levenshtein_index_keeps_dictionary_ids():
    words = [b"book", b"back", b"book", b"boon"]
    index = szs.LevenshteinIndex(words, max_distance=2)

    assert repr(index) == "LevenshteinIndex(max_distance=2)"
    query_ids, dictionary_ids, distances = index([b"cook", b"book"], bound=1)
    matches = sorted(zip(query_ids.tolist(), dictionary_ids.tolist(), distances.tolist()))
    assert matches == [(0, 0, 1), (0, 2, 1), (1, 0, 0), (1, 2, 0), (1, 3, 1)]
    assert query_ids.dtype.name == "uint64"
    assert dictionary_ids.dtype.name == "uint32"
    assert distances.dtype.name == "uint8"

    parallel = szs.DeviceScope(cpu_cores=2)
    parallel_matches = index(Strs([b"cook", b"book"]), bound=1, device=parallel)
    assert sorted(zip(*(array.tolist() for array in parallel_matches))) == matches

    keyword_matches = index(queries=[b"cook", b"book"], bound=1)
    assert sorted(zip(*(array.tolist() for array in keyword_matches))) == matches
    with pytest.raises(TypeError):
        index(bound=1)
    with pytest.raises(TypeError):
        index([b"book"], queries=[b"book"])

    # More than eight matches per query exercises the output-capacity retry.
    dense_index = szs.LevenshteinIndex([b"same"] * 20, max_distance=1)
    dense_results = dense_index([b"same"], bound=0)
    assert len(dense_results[0]) == 20
    assert dense_results[1].tolist() == list(range(20))
    assert dense_results[2].tolist() == [0] * 20

    empty_index = szs.LevenshteinIndex([], max_distance=2)
    assert all(len(array) == 0 for array in empty_index([], bound=2))
    assert all(len(array) == 0 for array in empty_index([b"anything"], bound=2))

    with pytest.raises(ValueError):
        index([b"book"], bound=3)
    with pytest.raises(TypeError):
        szs.LevenshteinIndex([b"valid", object()])
    with pytest.raises(TypeError):
        index([b"valid", object()])


def test_levenshtein_index_separates_byte_and_unicode_distance():
    words = ["café", "cafe", "咖啡", "咖非", "café"]
    byte_index = szs.LevenshteinIndex(words, max_distance=2)
    unicode_index = szs.LevenshteinIndexUTF8(Strs(words), max_distance=2)

    byte_results = byte_index(["cafe"], bound=1)
    assert sorted(zip(byte_results[1].tolist(), byte_results[2].tolist())) == [(1, 0)]
    unicode_results = unicode_index(Strs(["cafe", "咖啡"]), bound=1)
    unicode_matches = sorted(zip(*(array.tolist() for array in unicode_results)))
    assert unicode_matches == [(0, 0, 1), (0, 1, 0), (0, 4, 1), (1, 2, 0), (1, 3, 1)]

    with pytest.raises(ValueError):
        unicode_index([b"\xf0\x9f"], bound=1)
    with pytest.raises(ValueError):
        szs.LevenshteinIndexUTF8([b"valid", b"\xf0\x9f"])


def test_device_scope():
    """`DeviceScope` accepts a default scope, `cpu_cores` in {0, 1, N}, and `gpu_device` where CUDA
    is available, and rejects non-numeric arguments and specifying both at once."""

    default_scope = szs.DeviceScope()
    assert default_scope is not None

    scope_multi = szs.DeviceScope(cpu_cores=4)
    assert scope_multi is not None

    if "cuda" in szs.__capabilities__:
        try:
            scope_gpu = szs.DeviceScope(gpu_device=0)
            assert scope_gpu is not None
        except RuntimeError:
            # GPU capability is reported but device initialization failed
            pass
    else:
        with pytest.raises(RuntimeError):
            szs.DeviceScope(gpu_device=0)

    # Test cpu_cores=1 redirects to default scope
    scope_single = szs.DeviceScope(cpu_cores=1)
    assert scope_single is not None

    # Test cpu_cores=0 uses all available cores
    scope_all = szs.DeviceScope(cpu_cores=0)
    assert scope_all is not None

    with pytest.raises(ValueError):
        szs.DeviceScope(cpu_cores=4, gpu_device=0)  # Can't specify both

    with pytest.raises(TypeError):
        szs.DeviceScope(cpu_cores="invalid")

    with pytest.raises(TypeError):
        szs.DeviceScope(gpu_device="invalid")


def test_device_scope_from_another_thread():
    """A `DeviceScope` is usable from a thread that did not create it.

    A CUDA current context is per-thread state that the driver never creates implicitly, so a scope
    scheduled on the creating thread used to leave every other thread with no context - masked only
    because allocating happened to bind one as a side effect. This drives one scope from a worker
    thread that never touched the device, for both the default and the explicit GPU scope."""

    # Each scope needs an engine whose capability matches it: a CPU engine on a GPU scope is a mismatch by design.
    scopes_and_capabilities = [(szs.DeviceScope(), None)]
    if "cuda" in szs.__capabilities__:
        try:
            scopes_and_capabilities.append((szs.DeviceScope(gpu_device=0), ("cuda",)))
        except RuntimeError:
            pass  # GPU capability reported but the device would not initialize

    first = Strs(["hello", "world", "abcdef"])
    second = Strs(["hallo", "word", "abcdxf"])

    for scope, capabilities in scopes_and_capabilities:
        engine = (
            szs.LevenshteinDistances() if capabilities is None else szs.LevenshteinDistances(capabilities=capabilities)
        )
        failures = []

        def run_on_worker():
            try:
                # A cross-product matrix, so the paired distances sit on the diagonal.
                matrix = engine(first, second, device=scope).tolist()
                diagonal = [matrix[index][index] for index in range(len(first))]
                assert diagonal == [1, 1, 1], f"unexpected distances: {diagonal}"
            except BaseException as error:  # noqa: BLE001 - reported through `failures`
                failures.append(error)

        worker = threading.Thread(target=run_on_worker)
        worker.start()
        worker.join()
        assert not failures, f"driving a scope from another thread failed: {failures[0]!r}"


def test_cpu_scope_rejects_gpu_engine():
    """A CPU-only scope cannot drive a GPU engine, and says so rather than failing opaquely.

    The status a scope mismatch reports was inconsistent across engine families, so nothing could
    rely on it; every family now reports the same one, surfaced as a `RuntimeError`."""

    if "cuda" not in szs.__capabilities__:
        pytest.skip("CUDA backend not compiled in")

    cpu_scope = szs.DeviceScope(cpu_cores=2)
    first = Strs(["hello", "world"])
    second = Strs(["hallo", "word"])

    # A CUDA-capability engine paired with a CPU scope must fail.
    engine = szs.LevenshteinDistances(capabilities=("cuda",))
    with pytest.raises(RuntimeError):
        engine(first, second, device=cpu_scope)


# endregion Unit


# region Interop


def test_to_device():
    """`to_device` returns the identical `Strs` object for a full collection, a slice, a
    single-element slice, and an empty slice, each preserving its own length and contents."""

    # Skip test if CUDA is not available
    if "cuda" not in szs.__capabilities__:
        pytest.skip("CUDA not available, skipping to_device test")

    # Create a Strs object with multiple strings, including edge cases
    original_strs = sz.Strs(["hello", "world", "test", "slice", "data", "", "café"])

    # Test full object conversion
    result = szs.to_device(original_strs)
    assert result is original_strs  # Returns same object
    assert len(result) == 7
    assert list(result) == ["hello", "world", "test", "slice", "data", "", "café"]

    # Test with slices, which are non-owning views
    slice_strs = original_strs[1:4]  # Creates a view
    result_slice = szs.to_device(slice_strs)
    assert result_slice is slice_strs  # Returns same object
    assert len(result_slice) == 3
    assert list(result_slice) == ["world", "test", "slice"]

    # Test with single element slice
    single_strs = original_strs[2:3]
    result_single = szs.to_device(single_strs)
    assert result_single is single_strs
    assert len(result_single) == 1
    assert list(result_single) == ["test"]

    # Test with empty slice
    empty_strs = original_strs[3:3]
    result_empty = szs.to_device(empty_strs)
    assert result_empty is empty_strs
    assert len(result_empty) == 0
    assert list(result_empty) == []


def test_to_device_unicode_complex():
    """`to_device` preserves byte-level identity across CJK, RTL Arabic, emoji, and ZWJ Unicode,
    keeps NFC and NFD forms as distinct byte sequences, and holds through nested slices of slices."""

    # Skip test if CUDA is not available
    if "cuda" not in szs.__capabilities__:
        pytest.skip("CUDA not available, skipping to_device test")

    # Complex Unicode test cases as raw byte literals
    unicode_bytes = [
        b"Hello, \xe4\xb8\x96\xe7\x95\x8c!",  # Mixed ASCII + CJK (世界)
        b"\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd8\xa7\xd9\x84\xd8\xb9\xd8\xa7\xd9\x84\xd9\x85",  # Arabic RTL
        b"\xf0\x9f\xa6\x96\xf0\x9f\x94\xa5\xf0\x9f\x9a\x80\xf0\x9f\x92\xbb",  # Emoji sequence 🦖🔥🚀💻
        b"caf\xc3\xa9",  # NFC normalized café
        b"cafe\xcc\x81",  # NFD normalized café (e + combining acute)
        b"\xf0\x9d\x95\xb3\xf0\x9d\x96\x8a\xf0\x9d\x96\x91\xf0\x9d\x96\x91\xf0\x9d\x96\x94",  # Mathematical script 𝕳𝖊𝖑𝖑𝖔
        b"\xe2\x80\x8d\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x92\xbb\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x94\xac",  # ZWJ sequences
        b"\xe1\xbc\x88\xcf\x81\xcf\x87\xce\xb9\xce\xbc\xce\xae\xce\xb4\xce\xb7\xcf\x82",  # Ancient Greek Ἀρχιμήδης
        b"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f\x87\xab\xf0\x9f\x87\xb7\xf0\x9f\x87\xaf\xf0\x9f\x87\xb5",  # Flag sequences 🇺🇸🇫🇷🇯🇵
        b"",  # Empty bytes
        b"\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87",  # Devanagari नमस्ते
        b"\xf0\x9f\xa7\xac\xe2\x9a\x9b\xef\xb8\x8f\xf0\x9f\x94\xac",  # Science emoji 🧬⚛️🔬
    ]

    original_strs = sz.Strs(unicode_bytes)

    # Test full Unicode collection
    result = szs.to_device(original_strs)
    assert result is original_strs
    assert len(result) == len(unicode_bytes)
    assert list(result) == unicode_bytes

    # Test Unicode slice containing various scripts
    slice_strs = original_strs[1:6]  # Arabic, emoji, café forms, mathematical script
    expected_slice = unicode_bytes[1:6]

    result_slice = szs.to_device(slice_strs)
    assert result_slice is slice_strs
    assert len(result_slice) == 5
    assert list(result_slice) == expected_slice

    # Verify byte-level integrity of complex Unicode
    for original, converted in zip(expected_slice, result_slice):
        assert bytes(converted) == original

    # Test edge cases: first element, last element, middle elements
    assert list(szs.to_device(original_strs[0:1])) == [unicode_bytes[0]]  # First element
    assert list(szs.to_device(original_strs[-1:])) == [unicode_bytes[-1]]  # Last element
    assert list(szs.to_device(original_strs[9:10])) == [unicode_bytes[9]]  # Empty bytes

    # Test that NFC and NFD forms are preserved as different byte sequences
    nfc_nfd_slice = original_strs[3:5]  # Both café forms
    result_forms = szs.to_device(nfc_nfd_slice)
    assert len(result_forms) == 2
    assert bytes(result_forms[0]) == unicode_bytes[3]  # NFC form preserved
    assert bytes(result_forms[1]) == unicode_bytes[4]  # NFD form preserved

    # Test a nested slice, a slice of a slice
    nested_slice = slice_strs[1:3]  # Take middle 2 elements from existing slice
    result_nested = szs.to_device(nested_slice)
    assert len(result_nested) == 2
    # slice_strs is [1:6], so slice_strs[1:3] is elements [2:4] from original
    assert bytes(result_nested[0]) == unicode_bytes[2]  # Emoji sequence
    assert bytes(result_nested[1]) == unicode_bytes[3]  # NFC café


# endregion Interop


# region Unit


def test_parameter_validation():
    """`LevenshteinDistances` and `Fingerprints` reject non-numeric constructor keywords and
    non-`Strs` call arguments with `TypeError` or `RuntimeError`, while mismatched query/candidate
    counts are valid and simply produce a rectangular result matrix."""

    # Test constructor parameter type validation
    with pytest.raises(TypeError):
        szs.LevenshteinDistances(open="invalid")  # wrong type

    with pytest.raises(TypeError):
        szs.LevenshteinDistances(extend="invalid")  # wrong type

    with pytest.raises(TypeError):
        szs.LevenshteinDistances(mismatch="invalid")  # wrong type

    with pytest.raises(TypeError):
        szs.Fingerprints(ndim="invalid")  # wrong type

    # Test computation input validation
    engine = szs.LevenshteinDistances()

    # A None query raises `TypeError` or `RuntimeError`, the latter from GPU memory issues.
    # `candidates=None` is valid and requests symmetric self-similarity, so it is not an error.
    with pytest.raises((TypeError, RuntimeError)):
        engine(None, Strs(["test"]))

    # Mismatched query/candidate counts are valid in the cross-product API: they yield a rectangular matrix.
    rectangular = engine(Strs(["a", "b"]), Strs(["c"]))
    assert rectangular.shape == (2, 1)

    # Test with non-Strs inputs
    with pytest.raises((TypeError, RuntimeError)):
        engine(["test"], Strs(["test"]))  # list instead of Strs

    with pytest.raises((TypeError, RuntimeError)):
        engine(Strs(["test"]), ["test"])  # list instead of Strs

    # Test Fingerprints computation validation
    fp_engine = szs.Fingerprints(ndim=5)
    with pytest.raises((TypeError, RuntimeError)):
        fp_engine(None)  # None input

    with pytest.raises((TypeError, RuntimeError)):
        fp_engine(["test"])  # list instead of Strs


# endregion Unit


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", "-s", __file__]))
