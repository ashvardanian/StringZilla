"""Capability tests: the sz.capabilities_{detected,compiled,enabled}_{cpu,cuda,rocm,metal} producers,
the sz.stream_{init,free}_{cuda,rocm,metal} producers, configure_thread_cpu, Strings.copy, copy and
synchronize.

Capabilities are reported along two independent axes, `detected` (what a device can execute) and
`compiled` (what the ISA probes baked into this build), plus `enabled` (what dispatch uses, their
intersection, which one call narrows with `capabilities=`).

Conflating the axes is a silent performance cliff rather than a build error: `detected` is true of
the machine no matter what was compiled in.

Run:
    uv pip install numpy pyarrow pytest pytest-repeat
    uv pip install -e . --force-reinstall --no-build-isolation
    uv run --no-project python -m pytest test/capabilities.py -q

File: test/capabilities.py
Author: Ash Vardanian
Date: September 29, 2026
"""

import concurrent.futures
import ctypes
import threading
import time
import weakref

import pytest
from base import SETTINGS, get_random_string, numpy_available, requires_cuda, scaled_iterations

import stringzilla as sz

if numpy_available:
    import numpy as np


def test_capability_names_match_the_c_table():
    """`Capability` has one member per CPU capability, in bit order, and none for the GPU capabilities.

    A name missing or moved here means the names drifted from the `sz_cap_*_k` bits.
    """
    # fmt: off
    expected = [
        "serial",
        "westmere", "goldmont", "haswell", "skylake", "icelake",
        "neon", "neonaes", "neonsha", "sve", "sve2", "sve2aes",
        "rvv", "rvvcrypto", "v128", "v128relaxed", "loongsonasx", "powervsx",
    ]
    # fmt: on
    assert list(sz.Capability.__members__) == [name.upper() for name in expected], "members follow the bit order"


def test_enabled_is_both_detected_and_compiled():
    """What dispatch uses by default is what this CPU runs and this build holds, serial included."""
    detected, compiled, enabled = (
        sz.capabilities_detected_cpu(),
        sz.capabilities_compiled_cpu(),
        sz.capabilities_enabled_cpu(),
    )
    assert enabled & ~(detected | sz.Capability.SERIAL) == 0, "enabled never exceeds what the CPU detects"
    assert enabled & ~compiled == 0, "enabled never exceeds what the build compiled"
    assert sz.Capability.SERIAL in enabled


def test_masks_past_this_cpu_are_clamped():
    """A call asking for every capability runs only what this CPU detects and this build compiled.

    Without the clamp in the library's lookup, an ISA that was compiled in but that this CPU lacks
    would point dispatch at instructions the hardware refuses to execute.
    """
    text = "The quick brown fox jumps over the lazy dog" * 8
    everything = sz.capabilities_detected_cpu() | sz.capabilities_compiled_cpu()
    assert sz.find(text, "lazy", capabilities=everything) == text.find("lazy")
    assert sz.hash(text, capabilities=everything) == sz.hash(text)


@pytest.mark.skipif(not SETTINGS.expect_simd, reason="built without SIMD on purpose: STRINGZILLA_EXPECT_SIMD=0")
def test_compiled_covers_the_baseline_this_machine_detects(detected_baseline: sz.Capability):
    """A build whose ISA probes failed is scalar, and only `compiled` can see it.

    Skips where there is no SIMD to expect, so genuinely serial targets stay green: a 32-bit
    or exotic arch, or a CPU too old for the baseline. Set `STRINGZILLA_EXPECT_SIMD=0` to skip a
    deliberately scalar build on a SIMD-capable machine.
    """
    assert detected_baseline in sz.capabilities_compiled_cpu(), (
        f"this CPU reports {detected_baseline.name} but no {detected_baseline.name} kernels were compiled in - "
        f"the ISA probes failed at build time and this build is scalar"
    )


def test_configure_thread_prepares_another_thread():
    """A thread other than the importing one prepares itself for the CPU's enabled capabilities."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(sz.configure_thread_cpu, sz.capabilities_enabled_cpu()).result() is None


def test_capabilities_keyword_narrows_one_call():
    """`capabilities=` picks the capabilities of one call and leaves the default of every other call alone.

    The keyword keeps no serial fallback, so a mask of no capability finds no kernel.
    """
    text = "The quick brown fox jumps over the lazy dog" * 8
    enabled = sz.capabilities_enabled_cpu()
    assert sz.find(text, "lazy", capabilities=sz.Capability.SERIAL) == sz.find(text, "lazy") == text.find("lazy")
    assert sz.hash(text, capabilities=sz.Capability.SERIAL) == sz.hash(text)
    assert sz.capabilities_enabled_cpu() == enabled
    with pytest.raises(LookupError):
        sz.find(text, "lazy", capabilities=sz.Capability(0))


def test_engine_capabilities_narrow_the_cpu():
    """A CPU engine's `capabilities=` is clamped to what this CPU runs, and it takes no stream."""
    queries = sz.Strings(["kitten", "saturday"])
    everything = sz.capabilities_detected_cpu() | sz.capabilities_compiled_cpu()
    for capabilities in (sz.Capability.SERIAL, everything):
        distances = memoryview(bytearray(4 * 8)).cast("Q", (2, 2))
        sz.LevenshteinEngine(queries, capabilities=capabilities).distances(queries, distances)
        assert distances[0, 0] == distances[1, 1] == 0
        assert distances[0, 1] == distances[1, 0] == 7
    with pytest.raises(TypeError):
        sz.LevenshteinEngine(queries, device=0)
    with pytest.raises(ValueError, match="takes no stream"):
        sz.LevenshteinEngine(queries, stream=1)
    with pytest.raises(LookupError):
        sz.LevenshteinEngine(queries, capabilities=sz.Capability(0))


def test_producers_report_clamped_capabilities():
    """Each GPU vendor counts its devices, or raises without one, enables what is both detected and compiled,
    and makes a stream on its last device that synchronizes and frees."""
    for vendor in ("cuda", "rocm", "metal"):
        count_devices = getattr(sz, f"count_devices_{vendor}")
        detected = getattr(sz, f"capabilities_detected_{vendor}")
        compiled = getattr(sz, f"capabilities_compiled_{vendor}")
        enabled = getattr(sz, f"capabilities_enabled_{vendor}")
        stream_init = getattr(sz, f"stream_init_{vendor}")
        stream_free = getattr(sz, f"stream_free_{vendor}")
        with pytest.raises(ValueError, match="ordinal must be non-negative"):
            enabled(-1)
        with pytest.raises(ValueError, match="ordinal must be non-negative"):
            stream_init(-1)
        try:
            count = count_devices()
        except RuntimeError:  # No device of this vendor answers, which every ordinal reports the same way
            with pytest.raises(RuntimeError):
                enabled(0)
            with pytest.raises(RuntimeError):
                stream_init(0)
            continue
        assert count > 0
        with pytest.raises(RuntimeError):
            enabled(count)
        assert enabled(count - 1) == detected(count - 1) & compiled(count - 1)
        stream = stream_init(count - 1)
        assert sz.synchronize(enabled(count - 1), stream) is None
        assert stream_free(stream) is None


def test_cpu_copies_read_like_their_strings():
    """A tape copied for the CPU reads like the strings it came from, on the host and in an engine."""
    texts = ["kitten", "", "sitting"]
    capabilities = sz.capabilities_enabled_cpu()
    tape = sz.Strings(texts).copy(capabilities)
    assert tape == sz.Strings(texts)
    assert [str(text) for text in tape[1:]] == texts[1:]
    queries = sz.Strings(["kitten"])
    from_strings = memoryview(bytearray(len(texts) * 8)).cast("Q", (1, len(texts)))
    from_tape = memoryview(bytearray(len(texts) * 8)).cast("Q", (1, len(texts)))
    engine = sz.LevenshteinEngine(queries)
    engine.distances(sz.Strings(texts), from_strings)
    engine.distances(tape, from_tape)
    assert from_tape.tolist() == from_strings.tolist()
    assert sz.synchronize(capabilities) is None
    assert sz.synchronize(capabilities, stream=None) is None
    with pytest.raises(TypeError):
        sz.Strings(texts).copy()


def test_cpu_copies_of_buffers_are_writable_bytes():
    """`sz.copy` for the CPU owns the bytes of any buffer, which the host and a host engine write in place."""
    capabilities = sz.capabilities_enabled_cpu()
    source = bytearray(b"kitten")
    copied = sz.copy(source, capabilities)
    assert isinstance(copied, sz.UnifiedBuffer)
    assert bytes(sz.copy(b"kitten", capabilities)) == b"kitten"
    view = memoryview(copied)
    assert not view.readonly
    view[0] = ord("m")
    assert bytes(copied) == b"mitten"
    assert source == b"kitten", "the copy owns its bytes"
    assert bytes(sz.copy(b"", capabilities)) == b""
    assert len(memoryview(sz.copy(bytearray(), capabilities))) == 0

    texts = ["kitten", "sitting"]
    distances = sz.copy(bytearray(len(texts) * 8), capabilities)
    sz.LevenshteinEngine(sz.Strings(["kitten"])).distances(
        sz.Strings(texts), memoryview(distances).cast("Q", (1, len(texts)))
    )
    assert memoryview(distances).cast("Q").tolist() == [0, 3]

    with pytest.raises(TypeError):
        sz.copy(b"kitten", "cpu")
    with pytest.raises(TypeError):
        sz.copy(b"kitten")
    with pytest.raises(TypeError):
        sz.copy("kitten", capabilities)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_cpu_copies_of_arrays_follow_c_order():
    """A strided array is copied in C order, so the bytes read back as the array it views."""
    transposed = np.arange(12, dtype=np.uint32).reshape(3, 4).T
    copied = sz.copy(transposed, sz.capabilities_enabled_cpu())
    assert np.frombuffer(copied, dtype=np.uint32).reshape(4, 3).tolist() == transposed.tolist()
    assert np.frombuffer(sz.copy(np.empty(0), sz.capabilities_enabled_cpu()), dtype=np.uint8).size == 0


def test_levenshtein_bound_clips_distances_past_it():
    """A candidate within the bound or at it is exact, and one past it reports the bound plus one."""
    engine = sz.LevenshteinEngine(sz.Strings(["kitten"]))
    candidates = sz.Strings(["kitten", "mitten", "mittens", "sitting"])

    def distances_for(**kwargs):
        distances = memoryview(bytearray(len(candidates) * 8)).cast("Q", (1, len(candidates)))
        engine.distances(candidates, distances, **kwargs)
        return distances.tolist()[0]

    exact = [0, 1, 2, 3]
    assert distances_for() == exact
    assert distances_for(bound=None) == exact
    assert distances_for(bound=3) == exact
    assert distances_for(bound=2) == exact
    assert distances_for(bound=1) == [0, 1, 2, 2]
    assert distances_for(bound=0) == [0, 1, 1, 1]
    assert distances_for(bound=2**64 - 1) == exact
    with pytest.raises(TypeError):
        distances_for(bound="1")
    with pytest.raises(OverflowError):
        distances_for(bound=-1)


def test_ngrams_radius_zeroes_pairs_past_it():
    """A pair is exact while its misses over the query's distinct N-grams stay in the radius, else 0.0 everywhere."""
    engine = sz.NgramsEngine(sz.Strings(["abcdefgh"]), [3, 4])
    # Misses over the query's eleven distinct N-grams at both widths: none, two, and eight.
    candidates = sz.Strings(["abcdefgh", "abcdefgx", "abcdxxxx"])

    def overlaps_for(**kwargs):
        overlaps = memoryview(bytearray(len(candidates) * 2 * 4)).cast("f", (1, len(candidates), 2))
        engine.overlaps(candidates, overlaps, **kwargs)
        return [list(row) for row in overlaps.tolist()[0]]

    exact = overlaps_for()
    assert exact[0] == [1.0, 1.0]
    assert all(0.0 < share < 1.0 for share in exact[1] + exact[2])
    assert overlaps_for(radius=None) == exact
    assert overlaps_for(radius=1.0) == exact
    assert overlaps_for(radius=2) == exact
    assert overlaps_for(radius=8 / 11) == exact
    assert overlaps_for(radius=7 / 11) == [exact[0], exact[1], [0.0, 0.0]]
    assert overlaps_for(radius=2 / 11) == [exact[0], exact[1], [0.0, 0.0]]
    assert overlaps_for(radius=1 / 11) == [exact[0], [0.0, 0.0], [0.0, 0.0]]
    assert overlaps_for(radius=0) == [exact[0], [0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(TypeError):
        overlaps_for(radius="1")
    with pytest.raises(ValueError, match="radius must be a share"):
        overlaps_for(radius=-0.5)


def test_ngrams_sketches_agree_on_shared_ngrams():
    """Equal texts sketch alike, a near text agrees at more slots than a far one, a short one is 0x4A, the output's
    width sets the slots, and an engine of no queries sketches as one of many does."""
    texts = sz.Strings(["Hello, world", "Hello, world", "Hello, word", "Привет, мир", "Hi"])

    def sketches_for(widths=(5, 8), queries=(), slots=512, **kwargs):
        engine = sz.NgramsEngine(sz.Strings(list(queries)), widths, **kwargs)
        sketches = memoryview(bytearray(len(texts) * slots)).cast("B", (len(texts), slots))
        engine.sketches(texts, sketches)
        return sketches.tolist()

    def agreements(first, second):
        return sum(left == right for left, right in zip(first, second))

    for widths in ((5, 8), [4], (3, 5, 8)):
        for slots in (128, 512, 2048):
            rows = sketches_for(widths, slots=slots)
            assert rows[0] == rows[1]
            assert agreements(rows[0], rows[2]) > agreements(rows[0], rows[3])
            assert rows[4] == [0x4A] * slots
    assert sketches_for(capabilities=sz.Capability.SERIAL) == sketches_for()
    assert sketches_for(queries=["kitten", "sitting"]) == sketches_for()
    with pytest.raises(ValueError, match="disagree on shape"):
        sketches_for(widths=[3, 0])
    with pytest.raises(TypeError):
        sketches_for(widths=["3"])
    engine = sz.NgramsEngine(sz.Strings([]), [5])
    if numpy_available:
        with pytest.raises(ValueError, match="out axis"):
            engine.sketches(texts, np.zeros((len(texts), 0), dtype=np.uint8))
    with pytest.raises(TypeError):
        engine.sketches(["Hello"], memoryview(bytearray(512)).cast("B", (1, 512)))
    with pytest.raises(TypeError):
        engine.sketches(texts)
    with pytest.raises(ValueError, match="no prepared queries"):
        engine.overlaps(texts, memoryview(bytearray(4)).cast("f", (1, 1, 1)))


def test_engine_calls_from_many_threads_agree():
    """Threads scoring through one shared engine and through engines of their own all get the one-thread answer."""
    queries = sz.Strings(["kitten", "saturday", "sunday"])
    candidates = sz.Strings(["sitting", "sunday", "monday", "", "kitten"] * 64)

    def score(engine):
        distances = memoryview(bytearray(len(queries) * len(candidates) * 8)).cast("Q", (len(queries), len(candidates)))
        for _ in range(16):
            engine.distances(candidates, distances)
        return distances.tolist()

    expected = score(sz.LevenshteinEngine(queries))
    shared = sz.LevenshteinEngine(queries)
    engines = [shared] * 4 + [sz.LevenshteinEngine(queries) for _ in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as pool:
        assert all(result == expected for result in pool.map(score, engines))


@pytest.mark.thread_unsafe  # Times one call against a spinning thread, which other test threads would slow down.
def test_engine_verbs_release_the_gil(rng):
    """Another Python thread keeps running through a long engine call, which only holds the engine's own lock."""
    queries = sz.Strings([get_random_string(rng, 60, alphabet="acgt") for _ in range(scaled_iterations(64))])
    candidates = sz.Strings([get_random_string(rng, 60, alphabet="acgt") for _ in range(scaled_iterations(8192))])
    engine = sz.LevenshteinEngine(queries)
    distances = memoryview(bytearray(len(queries) * len(candidates) * 8)).cast("Q", (len(queries), len(candidates)))
    ticks, stop = [0], threading.Event()

    def spin():
        while not stop.is_set():
            ticks[0] += 1

    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        before, started = ticks[0], time.perf_counter()
        time.sleep(0.05)
        rate = (ticks[0] - before) / (time.perf_counter() - started)
        before, started = ticks[0], time.perf_counter()
        engine.distances(candidates, distances)
        elapsed, during = time.perf_counter() - started, ticks[0] - before
    finally:
        stop.set()
        spinner.join()
    # Holding the GIL would leak the spinner one switch interval of 5 ms at most, a sliver of this call.
    assert during > rate * elapsed / 4, f"spinner ran {during} ticks in a {elapsed:.3f} s call, at {rate:.0f}/s"


def unified_array(cudart: ctypes.CDLL, shape, dtype):
    """A NumPy array over CUDA managed memory, which the host and the device both address, freed with the array."""
    dtype = np.dtype(dtype)
    bytes_count = int(np.prod(shape)) * dtype.itemsize
    pointer = ctypes.c_void_p()
    assert cudart.cudaMallocManaged(ctypes.byref(pointer), ctypes.c_size_t(bytes_count), 1) == 0
    memory = (ctypes.c_char * bytes_count).from_address(pointer.value)
    weakref.finalize(memory, cudart.cudaFree, pointer)
    return np.frombuffer(memory, dtype=dtype).reshape(shape)


@requires_cuda
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_gpu_engine_reads_gpu_tapes(cudart: ctypes.CDLL):
    """A CUDA engine scores `Strings.copy` candidates as the CPU engine does, once the stream is synchronized.

    The CPU engine reads the same tape, while the strings in host memory are refused by the device.
    """
    gpu = sz.capabilities_enabled_cuda(0)
    texts = ["sitting", "sunday", "monday", "", "kitten"]
    candidates = sz.Strings(texts).copy(gpu)
    assert len(candidates) == len(texts)
    assert [str(text) for text in candidates] == texts
    assert candidates == sz.Strings(texts)
    assert str(candidates[1]) == "sunday"
    assert str(candidates[-1]) == "kitten"

    queries = sz.Strings(["kitten", "saturday"])
    expected = memoryview(bytearray(len(queries) * len(texts) * 8)).cast("Q", (len(queries), len(texts)))
    from_tape = memoryview(bytearray(len(queries) * len(texts) * 8)).cast("Q", (len(queries), len(texts)))
    sz.LevenshteinEngine(queries).distances(sz.Strings(texts), expected)
    sz.LevenshteinEngine(queries).distances(candidates, from_tape)
    assert from_tape.tolist() == expected.tolist()
    engine = sz.LevenshteinEngine(queries, capabilities=gpu)
    distances = unified_array(cudart, (len(queries), len(texts)), "uintp")
    engine.distances(candidates, distances)
    sz.synchronize(gpu)
    assert distances.tolist() == expected.tolist()
    with pytest.raises(BufferError):
        engine.distances(sz.Strings(texts), distances)
