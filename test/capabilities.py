"""Device and capability tests: sz.Device, its capabilities_{detected,compiled,enabled,enable},
and configure_thread.

Capabilities are reported along two independent axes, `detected` (what a device can execute) and
`compiled` (what the ISA probes baked into this build), plus `enabled` (what dispatch uses, their
intersection unless narrowed by `capabilities_enable`, or for one call by `capabilities=`).

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
import os
import platform
import sys

import pytest
import stringzilla as sz

cpu = sz.Device.cpu()

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


@pytest.fixture(autouse=True)
def restore_enabled_capabilities():
    """Restores the enabled set each test found, as `capabilities_enable` changes it for the whole process."""
    enabled = cpu.capabilities_enabled()
    yield
    cpu.capabilities_enable(enabled)


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
    detected, compiled, enabled = cpu.capabilities_detected(), cpu.capabilities_compiled(), cpu.capabilities_enabled()
    assert enabled & ~(detected | sz.Capability.SERIAL) == 0, "enabled never exceeds what the CPU detects"
    assert enabled & ~compiled == 0, "enabled never exceeds what the build compiled"
    assert sz.Capability.SERIAL in enabled


def test_enabling_everything_keeps_what_runs_here():
    """Asking for every capability leaves exactly the ones both detected and compiled, serial included.

    Without the clamp, enabling an ISA that was compiled in but that this CPU lacks points
    dispatch at instructions the hardware refuses to execute.
    """
    detected, compiled = cpu.capabilities_detected(), cpu.capabilities_compiled()
    enabled = cpu.capabilities_enable(detected | compiled)
    assert enabled == detected & compiled == cpu.capabilities_enabled()
    assert sz.Capability.SERIAL in enabled, "the serial fallback is always both detected and compiled in"


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
    if baseline not in cpu.capabilities_detected():
        pytest.skip(f"this CPU does not report {baseline.name}; nothing to verify")

    assert baseline in cpu.capabilities_compiled(), (
        f"this CPU reports {baseline.name} but no {baseline.name} kernels were compiled in - "
        f"the ISA probes failed at build time and this build is scalar"
    )


@pytest.mark.thread_unsafe  # Its threads would narrow and restore the one process-wide mask under each other.
def test_enable_drops_the_capabilities_left_out():
    """`capabilities_enable` makes `wanted` the enabled set, so a capability left out stops dispatching."""
    available = cpu.capabilities_detected() & cpu.capabilities_compiled()
    capabilities = [
        capability for capability in sz.Capability if capability in available and capability != sz.Capability.SERIAL
    ]
    if not capabilities:
        pytest.skip("scalar build: no capability other than serial to toggle")

    enabled = cpu.capabilities_enable(available ^ capabilities[0])
    assert capabilities[0] not in enabled and enabled == cpu.capabilities_enabled()
    assert cpu.capabilities_enable(available) == available


def test_serial_survives_enabling_nothing():
    """The serial fallback always remains, so a kernel is always found."""
    assert cpu.capabilities_enable(sz.Capability(0)) == sz.Capability.SERIAL
    assert sz.find("haystack", "st") == 3


def test_device_names_one_device_it_sees():
    """A `Device` is a kind and an ordinal, compared by value, refusing an ordinal past the ones this process sees."""
    assert cpu == sz.Device("cpu") == sz.Device(kind="cpu", ordinal=0) and hash(cpu) == hash(sz.Device("cpu"))
    assert (cpu.kind, cpu.ordinal, repr(cpu)) == ("cpu", 0, "Device('cpu', 0)")
    assert sz.Device.count("cpu") == 1
    for kind in ("cpu", "cuda", "rocm", "metal"):
        with pytest.raises(ValueError):
            sz.Device(kind, -1)
        try:
            count = sz.Device.count(kind)
        except RuntimeError:  # No device of this kind answers, which the constructor reports the same way
            with pytest.raises(RuntimeError):
                sz.Device(kind, 0)
            continue
        assert count > 0
        with pytest.raises(ValueError):
            sz.Device(kind, count)
        if kind != "cpu":
            device = sz.Device(kind, count - 1)
            assert device.capabilities_enabled() == device.capabilities_detected() & device.capabilities_compiled()
            with pytest.raises(ValueError):
                device.capabilities_enable(sz.Capability.SERIAL)
            with pytest.raises(ValueError):
                device.configure_thread(sz.Capability.SERIAL)
    with pytest.raises(ValueError):
        sz.Device("tpu")


def test_configure_thread_prepares_another_thread():
    """A thread other than the importing one prepares itself for the CPU's enabled capabilities."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(cpu.configure_thread, cpu.capabilities_enabled()).result() is None


def test_capabilities_keyword_narrows_one_call():
    """`capabilities=` picks the capabilities of one call and leaves the default of every other call alone.

    Unlike `capabilities_enable`, the keyword keeps no serial fallback, so a mask of no capability finds no kernel.
    """
    text = "The quick brown fox jumps over the lazy dog" * 8
    enabled = cpu.capabilities_enabled()
    assert sz.find(text, "lazy", capabilities=sz.Capability.SERIAL) == sz.find(text, "lazy") == text.find("lazy")
    assert sz.hash(text, capabilities=sz.Capability.SERIAL) == sz.hash(text)
    assert cpu.capabilities_enabled() == enabled
    with pytest.raises(LookupError):
        sz.find(text, "lazy", capabilities=sz.Capability(0))


def test_engine_capabilities_narrow_its_device():
    """An engine runs on its `device=`, and `capabilities=` may only narrow that device's enabled set."""
    queries = sz.Strs(["kitten", "saturday"])
    distances = memoryview(bytearray(4 * 8)).cast("Q", (2, 2))
    sz.LevenshteinEngine(queries, device=cpu, capabilities=sz.Capability.SERIAL).distances(queries, distances)
    assert distances[0, 0] == distances[1, 1] == 0 and distances[0, 1] == distances[1, 0]
    with pytest.raises(TypeError):
        sz.LevenshteinEngine(queries, device=0)
    with pytest.raises(ValueError):
        sz.LevenshteinEngine(queries, capabilities=cpu.capabilities_enabled() | (1 << 48))
    with pytest.raises(ValueError):
        sz.LevenshteinEngine(queries, stream=1)
    with pytest.raises(LookupError):
        sz.LevenshteinEngine(queries, capabilities=sz.Capability(0))
