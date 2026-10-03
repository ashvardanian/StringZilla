"""Capability tests: the sz.{cpu,cuda,rocm,metal}_capabilities_{detected,compiled,enabled} producers,
the sz.{cuda,rocm,metal}_stream_{init,free} producers, cpu_configure_thread, Strs.copy and
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
import ctypes.util
import os
import platform
import sys
import threading
import time
import weakref

import pytest

import stringzilla as sz

BASELINE_BY_MACHINE = {
    ("x86_64", "amd64", "x64"): sz.Capability.HASWELL,
    ("arm64", "aarch64"): sz.Capability.NEON,
}
"""The capability every supported toolchain emits for a given 64-bit architecture. A machine that
detects one of these but did not compile it in has a broken probe, not a slow CPU.
"""


def baseline_for_this_machine():
    """The capability this machine is expected to carry, or None where none is guaranteed."""
    if not sys.maxsize > 2**32:
        return None  # 32-bit targets (i686, armv7) have no guaranteed baseline
    machine = platform.machine().lower()
    for names, baseline in BASELINE_BY_MACHINE.items():
        if machine in names:
            return baseline
    return None


def test_capability_members_are_the_cpu_capabilities():
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
        sz.cpu_capabilities_detected(),
        sz.cpu_capabilities_compiled(),
        sz.cpu_capabilities_enabled(),
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
    everything = sz.cpu_capabilities_detected() | sz.cpu_capabilities_compiled()
    assert sz.find(text, "lazy", capabilities=everything) == text.find("lazy")
    assert sz.hash(text, capabilities=everything) == sz.hash(text)


def test_compiled_covers_the_baseline_this_machine_detects():
    """A build whose ISA probes failed is scalar, and only `compiled` can see it.

    Skips where there is no SIMD to expect, so genuinely serial targets stay green: a 32-bit
    or exotic arch, or a CPU too old for the baseline. Set `STRINGZILLA_EXPECT_SIMD=0` to skip a
    deliberately scalar build on a SIMD-capable machine.
    """
    if os.environ.get("STRINGZILLA_EXPECT_SIMD") == "0":
        pytest.skip("STRINGZILLA_EXPECT_SIMD=0: this build is deliberately scalar")

    baseline = baseline_for_this_machine()
    if baseline is None:
        pytest.skip(f"no SIMD baseline is guaranteed on {platform.machine()}")
    if baseline not in sz.cpu_capabilities_detected():
        pytest.skip(f"this CPU does not report {baseline.name}; nothing to verify")

    assert baseline in sz.cpu_capabilities_compiled(), (
        f"this CPU reports {baseline.name} but no {baseline.name} kernels were compiled in - "
        f"the ISA probes failed at build time and this build is scalar"
    )


def test_configure_thread_prepares_another_thread():
    """A thread other than the importing one prepares itself for the CPU's enabled capabilities."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(sz.cpu_configure_thread, sz.cpu_capabilities_enabled()).result() is None


def test_capabilities_keyword_narrows_one_call():
    """`capabilities=` picks the capabilities of one call and leaves the default of every other call alone.

    The keyword keeps no serial fallback, so a mask of no capability finds no kernel.
    """
    text = "The quick brown fox jumps over the lazy dog" * 8
    enabled = sz.cpu_capabilities_enabled()
    assert sz.find(text, "lazy", capabilities=sz.Capability.SERIAL) == sz.find(text, "lazy") == text.find("lazy")
    assert sz.hash(text, capabilities=sz.Capability.SERIAL) == sz.hash(text)
    assert sz.cpu_capabilities_enabled() == enabled
    with pytest.raises(LookupError):
        sz.find(text, "lazy", capabilities=sz.Capability(0))


def test_engine_capabilities_narrow_the_cpu():
    """A CPU engine's `capabilities=` is clamped to what this CPU runs, and it takes no stream."""
    queries = sz.Strs(["kitten", "saturday"])
    everything = sz.cpu_capabilities_detected() | sz.cpu_capabilities_compiled()
    for capabilities in (sz.Capability.SERIAL, everything):
        distances = memoryview(bytearray(4 * 8)).cast("Q", (2, 2))
        sz.LevenshteinEngine(queries, capabilities=capabilities).distances(queries, distances)
        assert distances[0, 0] == distances[1, 1] == 0 and distances[0, 1] == distances[1, 0] == 7
    with pytest.raises(TypeError):
        sz.LevenshteinEngine(queries, device=0)
    with pytest.raises(ValueError):
        sz.LevenshteinEngine(queries, stream=1)
    with pytest.raises(LookupError):
        sz.LevenshteinEngine(queries, capabilities=sz.Capability(0))


def test_gpu_producers_report_clamped_capabilities():
    """Each GPU vendor counts its devices, or raises without one, enables what is both detected and compiled,
    and makes a stream on its last device that synchronizes and frees."""
    for vendor in ("cuda", "rocm", "metal"):
        count_devices = getattr(sz, f"{vendor}_count_devices")
        detected = getattr(sz, f"{vendor}_capabilities_detected")
        compiled = getattr(sz, f"{vendor}_capabilities_compiled")
        enabled = getattr(sz, f"{vendor}_capabilities_enabled")
        stream_init = getattr(sz, f"{vendor}_stream_init")
        stream_free = getattr(sz, f"{vendor}_stream_free")
        with pytest.raises(ValueError):
            enabled(-1)
        with pytest.raises(ValueError):
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
        assert enabled(count - 1) == detected(count - 1) & compiled()
        stream = stream_init(count - 1)
        assert sz.synchronize(enabled(count - 1), stream) is None and stream_free(stream) is None


def test_cpu_copies_read_like_their_strings():
    """A tape copied for the CPU reads like the strings it came from, on the host and in an engine."""
    texts = ["kitten", "", "sitting"]
    capabilities = sz.cpu_capabilities_enabled()
    tape = sz.Strs(texts).copy(capabilities)
    assert tape == sz.Strs(texts) and [str(text) for text in tape[1:]] == texts[1:]
    queries = sz.Strs(["kitten"])
    from_strs = memoryview(bytearray(len(texts) * 8)).cast("Q", (1, len(texts)))
    from_tape = memoryview(bytearray(len(texts) * 8)).cast("Q", (1, len(texts)))
    engine = sz.LevenshteinEngine(queries)
    engine.distances(sz.Strs(texts), from_strs)
    engine.distances(tape, from_tape)
    assert from_tape.tolist() == from_strs.tolist()
    assert sz.synchronize(capabilities) is None and sz.synchronize(capabilities, stream=None) is None
    with pytest.raises(TypeError):
        sz.Strs(texts).copy()


def test_engine_calls_from_many_threads_agree():
    """Threads scoring through one shared engine and through engines of their own all get the one-thread answer."""
    queries = sz.Strs(["kitten", "saturday", "sunday"])
    candidates = sz.Strs(["sitting", "sunday", "monday", "", "kitten"] * 64)

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
    queries = sz.Strs(["".join(rng.choice("acgt") for _ in range(60)) for _ in range(64)])
    candidates = sz.Strs(["".join(rng.choice("acgt") for _ in range(60)) for _ in range(8192)])
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


def cuda_capabilities_or_skip():
    """The capabilities of CUDA device 0, skipping the test where no CUDA device answers."""
    try:
        return sz.cuda_capabilities_enabled(0)
    except RuntimeError:
        pytest.skip("no CUDA device answers here")


def unified_array(shape, dtype):
    """A NumPy array over CUDA managed memory, which the host and the device both address, freed with the array."""
    np = pytest.importorskip("numpy")
    library = ctypes.util.find_library("cudart") or "libcudart.so"
    try:
        cudart = ctypes.CDLL(library)
    except OSError:
        pytest.skip(f"no CUDA runtime library to allocate managed memory with: {library}")
    dtype = np.dtype(dtype)
    bytes_count = int(np.prod(shape)) * dtype.itemsize
    pointer = ctypes.c_void_p()
    assert cudart.cudaMallocManaged(ctypes.byref(pointer), ctypes.c_size_t(bytes_count), 1) == 0
    memory = (ctypes.c_char * bytes_count).from_address(pointer.value)
    weakref.finalize(memory, cudart.cudaFree, pointer)
    return np.frombuffer(memory, dtype=dtype).reshape(shape)


def test_gpu_engine_reads_gpu_tapes():
    """A CUDA engine scores `Strs.copy` candidates as the CPU engine does, once the stream is synchronized.

    The CPU engine reads the same tape, while the strings in host memory are refused by the device.
    """
    gpu = cuda_capabilities_or_skip()
    texts = ["sitting", "sunday", "monday", "", "kitten"]
    candidates = sz.Strs(texts).copy(gpu)
    assert len(candidates) == len(texts) and [str(text) for text in candidates] == texts
    assert candidates == sz.Strs(texts) and str(candidates[1]) == "sunday" and str(candidates[-1]) == "kitten"

    queries = sz.Strs(["kitten", "saturday"])
    expected = memoryview(bytearray(len(queries) * len(texts) * 8)).cast("Q", (len(queries), len(texts)))
    from_tape = memoryview(bytearray(len(queries) * len(texts) * 8)).cast("Q", (len(queries), len(texts)))
    sz.LevenshteinEngine(queries).distances(sz.Strs(texts), expected)
    sz.LevenshteinEngine(queries).distances(candidates, from_tape)
    assert from_tape.tolist() == expected.tolist()
    engine = sz.LevenshteinEngine(queries, capabilities=gpu)
    distances = unified_array((len(queries), len(texts)), "uintp")
    engine.distances(candidates, distances)
    sz.synchronize(gpu)
    assert distances.tolist() == expected.tolist()
    with pytest.raises(BufferError):
        engine.distances(sz.Strs(texts), distances)
