#!/usr/bin/env python3
"""
Test suite for StringZillas parallel algorithms module.
Tests with Python lists, NumPy arrays, Apache Arrow columns, and StringZilla Strs types.
To run for the CPU backend:

    uv pip install numpy pyarrow pytest pytest-repeat affine-gaps
    SZ_TARGET=stringzillas-cpus uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -c "import stringzillas; print(stringzillas.__capabilities__)"
    uv run --no-project python -m pytest scripts/test_stringzillas.py -s -x

To run for the CUDA backend:

    uv pip install numpy pyarrow pytest pytest-repeat affine-gaps
    SZ_TARGET=stringzillas-cuda uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -c "import stringzillas; print(stringzillas.__capabilities__)"
    uv run --no-project python -m pytest scripts/test_stringzillas.py -s -x

Recommended flags for better diagnostics:

    -s                  show test output (no capture)
    -vv                 verbose output
    --maxfail=1         stop at first failure
    --full-trace        full Python tracebacks
    -k <pattern>        filter tests by substring
    -X faulthandler     to dump on fatal signals
    --verbose           enable verbose output

Example:

    SZ_TARGET=stringzillas-cpus uv pip install -e . --force-reinstall --no-build-isolation --verbose
    uv run --no-project python -X faulthandler -m pytest scripts/test_stringzillas.py -s -vv --maxfail=1 --full-trace
"""
import os
import sys
import platform
from random import choice, randint, seed
from string import ascii_lowercase
from typing import Optional, Literal

import pytest
import numpy as np  # ! Unlike StringZilla, NumPy is mandatory for StringZillas
import affine_gaps as ag  # ? Provides baseline implementation for NW & SW scoring

import stringzilla as sz
import stringzillas as szs
from stringzilla import Strs


@pytest.fixture(scope="session", autouse=True)
def log_test_environment():
    """Automatically log environment info before running any tests."""

    print()  # New line for better readability
    print("=== StringZillas Test Environment ===")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"StringZilla version: {sz.__version__}")
    print(f"StringZilla capabilities: {sorted(sz.__capabilities__)}")
    print(f"StringZillas version: {szs.__version__}")
    print(f"StringZillas capabilities: {sorted(szs.__capabilities__)}")
    print(f"NumPy version: {np.__version__}")
    print(f"Affine Gaps version: {ag.__version__}")

    # If QEMU is indicated via env (e.g., set by pyproject), mask out SVE/SVE2 to avoid emulation flakiness.
    is_qemu = os.environ.get("SZ_IS_QEMU_", "").lower() in ("1", "true", "yes", "on")
    if is_qemu:
        sve_like = {"sve", "sve2", "sve2+aes"}
        current = list(getattr(sz, "__capabilities__", ()))
        desired = tuple(c for c in current if c.lower() not in sve_like)
        if len(desired) != len(current):
            print(f"QEMU env detected; disabling {sve_like} for stability")
            sz.reset_capabilities(desired)

    print("=" * 40)
    print()  # New line for better readability


def test_library_properties():
    assert len(sz.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in sz.__capabilities__, "Serial backend must be present"
    assert isinstance(sz.__capabilities_str__, str) and len(sz.__capabilities_str__) > 0
    sz.reset_capabilities(sz.__capabilities__)  # Should not raise

    # Test StringZillas properties
    assert len(szs.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in szs.__capabilities__, "Serial backend must be present"
    assert isinstance(szs.__capabilities_str__, str) and len(szs.__capabilities_str__) > 0
    sz.reset_capabilities(szs.__capabilities__)  # Should not raise


DeviceName = Literal["default", "cpu_cores", "gpu_device"]
DEVICE_NAMES = ["default", "cpu_cores", "gpu_device"] if "cuda" in szs.__capabilities__ else ["default", "cpu_cores"]


def device_scope_and_capabilities(device: DeviceName):
    """Create a DeviceScope based on the specified device type."""
    if device == "default":
        return szs.DeviceScope(), ("serial",)
    elif device == "cpu_cores":
        return szs.DeviceScope(cpu_cores=2), ("serial", "parallel")
    elif device == "gpu_device":
        return szs.DeviceScope(gpu_device=0), ("cuda",)
    else:
        raise ValueError(f"Unknown device type: {device}")


InputSizeConfig = Literal["one-large", "few-big", "many-small"]
INPUT_SIZE_CONFIGS = ["one-large", "few-big", "many-small"]

# Reproducible test seeds for consistent CI runs
SEED_VALUES = [
    42,  # Classic test seed
    0,  # Edge case: zero seed
    1,  # Minimal positive seed
    314159,  # Pi digits
]


def generate_string_batches(config: InputSizeConfig):
    """Generate string batches based on the specified configuration.

    Returns:
        tuple: (batch_size, min_length, max_length) parameters for generating test strings
    """
    if config == "one-large":
        return 1, 50, 1024  # Single pair of long strings
    elif config == "few-big":
        return 7, 30, 128  # Few pairs of medium strings
    elif config == "many-small":
        return 1000, 10, 30  # Many pairs of short strings
    else:
        raise ValueError(f"Unknown input size config: {config}")


def seed_random_generators(seed_value: Optional[int] = None):
    """Seed the random number generators for reproducibility."""
    if seed_value is None:
        return
    seed(seed_value)
    # Try to seed NumPy's random number generator
    # This handles both NumPy 1.x and 2.x, and any import issues
    try:
        np.random.seed(seed_value)
    except (ImportError, AttributeError, Exception):
        pass


def test_device_scope():
    """Test DeviceScope for execution context control."""

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


def test_parameter_validation():
    """Test parameter validation and error handling for all engine types."""

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

    # Test None inputs - expect either `TypeError` or `RuntimeError` (GPU memory issues)
    with pytest.raises((TypeError, RuntimeError)):
        engine(None, Strs(["test"]))

    with pytest.raises((TypeError, RuntimeError)):
        engine(Strs(["test"]), None)

    # Test mismatched input sizes
    with pytest.raises((ValueError, RuntimeError)):
        a = Strs(["a", "b"])
        b = Strs(["c"])  # Different size
        engine(a, b)

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


def get_random_string(
    length: Optional[int] = None,
    variability: Optional[int] = None,
    alphabet: Optional[str] = None,
) -> str:
    if length is None:
        length = randint(3, 300)
    if alphabet is None:
        alphabet = ascii_lowercase
    if variability is None:
        variability = len(alphabet)
    return "".join(choice(alphabet[:variability]) for _ in range(length))


def is_equal_strings(native_strings, big_strings):
    for native_slice, big_slice in zip(native_strings, big_strings):
        assert native_slice == big_slice, f"Mismatch between `{native_slice}` and `{str(big_slice)}`"


def baseline_levenshtein_distance(s1, s2) -> int:
    """
    Compute the Levenshtein distance between two strings.
    """

    # Create a matrix of size (len(s1)+1) x (len(s2)+1)
    matrix = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)

    # Initialize the first column and first row of the matrix
    for i in range(len(s1) + 1):
        matrix[i, 0] = i
    for j in range(len(s2) + 1):
        matrix[0, j] = j

    # Compute Levenshtein distance
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,  # Deletion
                matrix[i, j - 1] + 1,  # Insertion
                matrix[i - 1, j - 1] + cost,  # Substitution
            )

    # Return the Levenshtein distance
    return matrix[len(s1), len(s2)]


@pytest.mark.parametrize("max_edit_distance", [150])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_levenshtein_distance_insertions(max_edit_distance: int, seed_value: int):
    """Test Levenshtein distance with sequential insertions using deterministic seeds."""

    # Create a new string by slicing and concatenating
    def insert_char_at(s, char_to_insert, index):
        return s[:index] + char_to_insert + s[index:]

    seed_random_generators(seed_value)
    binary_engine = szs.LevenshteinDistances()

    a = get_random_string(length=20)
    b = a
    for i in range(max_edit_distance):
        source_offset = randint(0, len(ascii_lowercase) - 1)
        target_offset = randint(0, len(b) - 1)
        b = insert_char_at(b, ascii_lowercase[source_offset], target_offset)
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = binary_engine(a_strs, b_strs)
        assert len(results) == 1, "Binary engine should return a single distance"
        assert results[0] == [i + 1], f"Edit distance mismatch after {i + 1} insertions: {a} -> {b}"


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_levenshtein_distances_with_simple_cases(capabilities_mode: str, device_name: DeviceName):

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    binary_engine = szs.LevenshteinDistances(capabilities=base_caps if capabilities_mode == "base" else device_scope)

    def binary_distance(a: str, b: str) -> int:
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = binary_engine(a_strs, b_strs, device=device_scope)
        assert len(results) == 1, "Binary engine should return a single distance"
        return results[0]

    assert binary_distance("hello", "hello") == 0
    assert binary_distance("hello", "hell") == 1
    assert binary_distance("", "") == 0
    assert binary_distance("", "abc") == 3
    assert binary_distance("abc", "") == 3
    assert binary_distance("abc", "ac") == 1, "one deletion"
    assert binary_distance("abc", "a_bc") == 1, "one insertion"
    assert binary_distance("abc", "adc") == 1, "one substitution"
    assert binary_distance("ggbuzgjux{}l", "gbuzgjux{}l") == 1, "one insertion (prepended)"
    assert binary_distance("abcdefgABCDEFG", "ABCDEFGabcdefg") == 14


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_levenshtein_distances_utf8_with_simple_cases(capabilities_mode: str, device_name: DeviceName):

    if device_name == "gpu_device":
        pytest.skip("CUDA backend does not support custom gaps in UTF-8 Levenshtein distances")
        return

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    unicode_engine = szs.LevenshteinDistancesUTF8(
        capabilities=base_caps if capabilities_mode == "base" else device_scope
    )

    def unicode_distance(a: str, b: str) -> int:
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = unicode_engine(a_strs, b_strs, device=device_scope)
        assert len(results) == 1, "Unicode engine should return a single distance"
        return results[0]

    assert unicode_distance("hello", "hell") == 1, "no unicode symbols, just ASCII"
    assert unicode_distance("𠜎 𠜱 𠝹 𠱓", "𠜎𠜱𠝹𠱓") == 3, "add 3 whitespaces in Chinese"
    assert unicode_distance("💖", "💗") == 1

    assert unicode_distance("αβγδ", "αγδ") == 1, "insert Beta"
    assert unicode_distance("école", "école") == 2, "etter 'é' as 1 character vs 'e' + '´'"
    assert unicode_distance("façade", "facade") == 1, "'ç' with cedilla vs. plain"
    assert unicode_distance("Schön", "Scho\u0308n") == 2, "'ö' represented as 'o' + '¨'"
    assert unicode_distance("München", "Muenchen") == 2, "German with umlaut vs. transcription"
    assert unicode_distance("こんにちは世界", "こんばんは世界") == 2, "Japanese greetings"


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_levenshtein_distances_with_custom_gaps(capabilities_mode: str, device_name: DeviceName):

    mismatch: int = 4
    opening: int = 3
    extension: int = 2

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    binary_engine = szs.LevenshteinDistances(
        open=opening,
        extend=extension,
        mismatch=mismatch,
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
    )

    def binary_distance(a: str, b: str) -> int:
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = binary_engine(a_strs, b_strs, device=device_scope)
        assert len(results) == 1, "Binary engine should return a single distance"
        return results[0]

    assert binary_distance("hello", "hello") == 0
    assert binary_distance("hello", "hell") == opening
    assert binary_distance("", "") == 0
    assert binary_distance("", "abc") == opening + 2 * extension
    assert binary_distance("abc", "") == opening + 2 * extension
    assert binary_distance("abc", "ac") == opening, "one deletion"
    assert binary_distance("abc", "a_bc") == opening, "one insertion"
    assert binary_distance("abc", "adc") == mismatch, "one substitution"
    assert binary_distance("ggbuzgjux{}l", "gbuzgjux{}l") == opening, "one insertion (prepended)"
    assert binary_distance("abcdefgABCDEFG", "ABCDEFGabcdefg") == min(14 * mismatch, 2 * opening + 12 * extension)


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
def test_levenshtein_distances_utf8_with_custom_gaps(capabilities_mode: str, device_name: DeviceName):

    if device_name == "gpu_device":
        pytest.skip("CUDA backend does not support custom gaps in UTF-8 Levenshtein distances")
        return

    mismatch: int = 4
    opening: int = 3

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    unicode_engine = szs.LevenshteinDistancesUTF8(
        open=opening,
        extend=opening,
        mismatch=mismatch,
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
    )

    def unicode_distance(a: str, b: str) -> int:
        a_strs = Strs([a])
        b_strs = Strs([b])
        results = unicode_engine(a_strs, b_strs, device=device_scope)
        assert len(results) == 1, "Unicode engine should return a single distance"
        return results[0]

    assert unicode_distance("hello", "hell") == opening, "no unicode symbols, just ASCII"
    assert unicode_distance("𠜎 𠜱 𠝹 𠱓", "𠜎𠜱𠝹𠱓") == 3 * opening, "add 3 whitespaces in Chinese"
    assert unicode_distance("💖", "💗") == 1 * mismatch

    assert unicode_distance("αβγδ", "αγδ") == opening, "insert Beta"
    assert unicode_distance("école", "école") == mismatch + opening, "etter 'é' as 1 character vs 'e' + '´'"
    assert unicode_distance("façade", "facade") == mismatch, "'ç' with cedilla vs. plain"
    assert unicode_distance("Schön", "Scho\u0308n") == mismatch + opening, "'ö' represented as 'o' + '¨'"
    assert unicode_distance("München", "Muenchen") == mismatch + opening, "German with umlaut vs. transcription"
    assert unicode_distance("こんにちは世界", "こんばんは世界") == min(2 * mismatch, 4 * opening), "Japanese greetings"


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("config", INPUT_SIZE_CONFIGS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_levenshtein_distance_random(
    capabilities_mode: str,
    device_name: DeviceName,
    config: InputSizeConfig,
    seed_value: int,
):
    """Test Levenshtein distances with deterministic seeds for reproducibility."""

    seed_random_generators(seed_value)
    batch_size, min_len, max_len = generate_string_batches(config)
    a_batch = [get_random_string(length=randint(min_len, max_len)) for _ in range(batch_size)]
    b_batch = [get_random_string(length=randint(min_len, max_len)) for _ in range(batch_size)]

    baselines = np.array([baseline_levenshtein_distance(a, b) for a, b in zip(a_batch, b_batch)])

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.LevenshteinDistances(capabilities=base_caps if capabilities_mode == "base" else device_scope)

    # Convert to Strs objects
    a_strs, b_strs = Strs(a_batch), Strs(b_batch)
    results = engine(a_strs, b_strs)

    np.testing.assert_array_equal(results, baselines, "Edit distances do not match")


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("config", INPUT_SIZE_CONFIGS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_needleman_wunsch_vs_levenshtein_random(
    capabilities_mode: str,
    device_name: DeviceName,
    config: InputSizeConfig,
    seed_value: int,
):
    """Test Needleman-Wunsch global alignment scores against Levenshtein distances with random strings."""

    seed_random_generators(seed_value)
    batch_size, min_len, max_len = generate_string_batches(config)
    a_batch = [get_random_string(length=randint(min_len, max_len)) for _ in range(batch_size)]
    b_batch = [get_random_string(length=randint(min_len, max_len)) for _ in range(batch_size)]

    character_substitutions = np.zeros((256, 256), dtype=np.int8)
    character_substitutions.fill(-1)
    np.fill_diagonal(character_substitutions, 0)

    baselines = [-baseline_levenshtein_distance(a, b) for a, b in zip(a_batch, b_batch)]

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.NeedlemanWunschScores(
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
        substitution_matrix=character_substitutions,
        open=-1,
        extend=-1,
    )

    # Convert to Strs objects
    a_strs, b_strs = Strs(a_batch), Strs(b_batch)
    results = engine(a_strs, b_strs)

    np.testing.assert_array_equal(results, baselines, "Edit distances do not match")


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("batch_size", [1, 7, 33])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_needleman_wunsch_against_affine_gaps(
    capabilities_mode: str,
    device_name: DeviceName,
    batch_size: int,
    seed_value: int,
):
    """Compare Needleman-Wunsch global alignment scores against affine_gaps baseline."""

    seed_random_generators(seed_value)
    alphabet = ag.default_proteins_alphabet
    a_batch = [get_random_string(length=randint(5, 50), alphabet=alphabet) for _ in range(batch_size)]
    b_batch = [get_random_string(length=randint(5, 50), alphabet=alphabet) for _ in range(batch_size)]

    # Baseline with affine_gaps (Gotoh)
    baseline = np.array(
        [
            int(
                ag.needleman_wunsch_gotoh_score(
                    a,
                    b,
                    substitution_alphabet=alphabet,
                    substitution_matrix=ag.default_proteins_matrix,
                    gap_opening=ag.default_gap_opening,
                    gap_extension=ag.default_gap_extension,
                )
            )
            for a, b in zip(a_batch, b_batch)
        ],
        dtype=np.int64,
    )

    # For StringZillas, blow up the substitution matrix into a 256x256 form
    subs = np.empty((256, 256), dtype=np.int8)
    for i, ci in enumerate(alphabet):
        for j, cj in enumerate(alphabet):
            subs[ord(ci), ord(cj)] = ag.default_proteins_matrix[i, j]

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.NeedlemanWunschScores(
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
        substitution_matrix=subs,
        open=ag.default_gap_opening,
        extend=ag.default_gap_extension,
    )

    results = engine(Strs(a_batch), Strs(b_batch), device=device_scope)
    if not np.array_equal(results, baseline):
        idx = int(np.where(results != baseline)[0][0])
        a, b = a_batch[idx], b_batch[idx]
        aligned_a, aligned_b = ag.needleman_wunsch_gotoh(
            a,
            b,
            substitution_alphabet=alphabet,
            substitution_matrix=ag.default_proteins_matrix,
            gap_open=ag.default_gap_opening,
            gap_extend=ag.default_gap_extension,
        )
        guide_line = "".join("|" if ca == cb else " " for ca, cb in zip(aligned_a, aligned_b))
        pytest.fail(
            "\n".join(
                [
                    f"Needleman-Wunsch mismatch at index {idx}:",
                    f"  a: {a}",
                    f"  b: {b}",
                    f"  szs score:     {int(results[idx])}",
                    f"  affine_gaps:   {int(baseline[idx])}",
                    "  Alignment (affine_gaps):",
                    f"    {aligned_a}",
                    f"    {guide_line}",
                    f"    {aligned_b}",
                ]
            )
        )
    np.testing.assert_array_equal(results, baseline)


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("batch_size", [1, 7, 33])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_smith_waterman_against_affine_gaps(
    capabilities_mode: str,
    device_name: DeviceName,
    batch_size: int,
    seed_value: int,
):
    """Compare Smith-Waterman local alignment scores against affine_gaps baseline."""

    seed_random_generators(seed_value)
    alphabet = ag.default_proteins_alphabet
    a_batch = [get_random_string(length=randint(5, 50), alphabet=alphabet) for _ in range(batch_size)]
    b_batch = [get_random_string(length=randint(5, 50), alphabet=alphabet) for _ in range(batch_size)]

    # Baseline with affine_gaps (Gotoh)
    baseline = np.array(
        [
            int(
                ag.smith_waterman_gotoh_score(
                    a,
                    b,
                    substitution_alphabet=alphabet,
                    substitution_matrix=ag.default_proteins_matrix,
                    gap_opening=ag.default_gap_opening,
                    gap_extension=ag.default_gap_extension,
                )
            )
            for a, b in zip(a_batch, b_batch)
        ],
        dtype=np.int64,
    )

    # For StringZillas, blow up the substitution matrix into a 256x256 form
    subs = np.empty((256, 256), dtype=np.int8)
    for i, ci in enumerate(alphabet):
        for j, cj in enumerate(alphabet):
            subs[ord(ci), ord(cj)] = ag.default_proteins_matrix[i, j]

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.SmithWatermanScores(
        capabilities=base_caps if capabilities_mode == "base" else device_scope,
        substitution_matrix=subs,
        open=ag.default_gap_opening,
        extend=ag.default_gap_extension,
    )

    results = engine(Strs(a_batch), Strs(b_batch), device=device_scope)
    if not np.array_equal(results, baseline):
        idx = int(np.where(results != baseline)[0][0])
        a, b = a_batch[idx], b_batch[idx]
        aligned_a, aligned_b = ag.smith_waterman_gotoh(
            a,
            b,
            substitution_alphabet=alphabet,
            substitution_matrix=ag.default_proteins_matrix,
            gap_open=ag.default_gap_opening,
            gap_extend=ag.default_gap_extension,
        )
        guide_line = "".join("|" if ca == cb else " " for ca, cb in zip(aligned_a, aligned_b))
        pytest.fail(
            "\n".join(
                [
                    f"Smith-Waterman mismatch at index {idx}:",
                    f"  a: {a}",
                    f"  b: {b}",
                    f"  szs score:     {int(results[idx])}",
                    f"  affine_gaps:   {int(baseline[idx])}",
                    "  Alignment (affine_gaps):",
                    f"    {aligned_a}",
                    f"    {guide_line}",
                    f"    {aligned_b}",
                ]
            )
        )
    np.testing.assert_array_equal(results, baseline)


@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("ndim", [1, 7, 64, 1024])
def test_fingerprints(capabilities_mode: str, device_name: str, ndim: int):
    """Test Fingerprints basic functionality."""

    # Create engine with smaller dimensions to avoid memory issues
    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.Fingerprints(ndim=ndim, capabilities=base_caps if capabilities_mode == "base" else device_scope)

    # Basic functionality - empty input should return empty arrays
    hashes, counts = engine(Strs([]), device=device_scope)
    assert hashes.shape == (0, ndim)
    assert counts.shape == (0, ndim)
    assert hashes.dtype == np.uint32
    assert counts.dtype == np.uint32

    test_strings = Strs(["hello", "world", "hello"])
    hashes, counts = engine(test_strings, device=device_scope)

    # Check output shape and types
    assert hashes.shape == (3, ndim), f"Expected (3, {ndim}), got {hashes.shape}"
    assert counts.shape == (3, ndim), f"Expected (3, {ndim}), got {counts.shape}"
    assert hashes.dtype == np.uint32
    assert counts.dtype == np.uint32

    # Identical strings should produce identical fingerprints
    assert np.array_equal(hashes[0], hashes[2]), "Identical strings should produce identical hashes"
    assert np.array_equal(counts[0], counts[2]), "Identical strings should produce identical counts"

    # Different strings should produce different fingerprints, but we can't always expect
    # different counts on very short inputs
    assert not np.array_equal(hashes[0], hashes[1]), "Different strings should produce different hashes"


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("capabilities_mode", ["base", "infer-from-device"])
@pytest.mark.parametrize("device_name", DEVICE_NAMES)
@pytest.mark.parametrize("ndim", [1, 7, 64, 1024])
@pytest.mark.parametrize("seed_value", [42, 123, 1337, 12345, 98765])  # Subset of seeds for this test
def test_fingerprints_random(batch_size: int, capabilities_mode: str, device_name: str, ndim: int, seed_value: int):
    """Test Fingerprints with random strings using deterministic seeds."""

    seed_random_generators(seed_value)
    batch = [get_random_string(length=randint(5, 50)) for _ in range(batch_size)]

    device_scope, base_caps = device_scope_and_capabilities(device_name)
    engine = szs.Fingerprints(ndim=ndim, capabilities=base_caps if capabilities_mode == "base" else device_scope)

    strs = Strs(batch)
    hashes, counts = engine(strs, device=device_scope)
    assert hashes.shape == (batch_size, ndim)
    assert counts.shape == (batch_size, ndim)

    # Verify consistency
    hashes_repeated, counts_repeated = engine(strs, device=device_scope)
    assert np.array_equal(hashes, hashes_repeated), "Same input should produce same hashes"
    assert np.array_equal(counts, counts_repeated), "Same input should produce same counts"


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", "-s", __file__]))
