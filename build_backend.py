"""
Build backend and cibuildwheel helpers for StringZilla.

Responsibilities
    -   Backend: delegate to `setuptools.build_meta` and inject `numpy` into
        PEP 517 build requirements only when `SZ_TARGET` is one of
        `stringzillas-cpus` or `stringzillas-cuda`. Also forward PEP 660
        editable hooks so editable installs work in modern tools.
    -   CLI (for CI only): provide compact, cross‑platform commands to prepare the
        wheel test environment and to run the appropriate tests.

CLI Commands
    -   pull-deps [PROJECT_DIR]: when testing parallel targets, installs
        the serial stringzilla into the test venv, and ensures test-only
        deps (NumPy, affine-gaps) are present.
    -   run-tests [PROJECT_DIR]: runs the serial `test/` suite (minus the
        StringZillas modules) for the `stringzilla` target, and the
        StringZillas modules for the parallel targets.
    -   check-wheels [--wheel-dir DIR] SUBSTRING...: fails when no wheel matches
        one of the named platform substrings, so a release cannot silently
        degrade to sdist-only when a wheel job breaks.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _build_meta():
    from setuptools import build_meta as _orig_build_meta  # local import

    return _orig_build_meta


def _detect_target() -> str:
    t = os.environ.get("SZ_TARGET")
    if t:
        return t
    try:
        return Path("SZ_TARGET.env").read_text(encoding="utf-8").strip() or "stringzilla"
    except FileNotFoundError:
        return "stringzilla"


def get_requires_for_build_wheel(config_settings=None):
    """Get build requirements, conditionally including numpy."""
    requirements = _build_meta().get_requires_for_build_wheel(config_settings)
    if _detect_target() in ("stringzillas-cpus", "stringzillas-cuda"):
        requirements.append("numpy")
    return requirements


def get_requires_for_build_editable(config_settings=None):
    """Get build requirements for editable, conditionally including numpy (PEP 660)."""
    bm = _build_meta()
    # Prefer setuptools' own hook if available; otherwise mimic wheel behavior
    if hasattr(bm, "get_requires_for_build_editable"):
        requirements = bm.get_requires_for_build_editable(config_settings)
    else:
        requirements = bm.get_requires_for_build_wheel(config_settings)

    if _detect_target() in ("stringzillas-cpus", "stringzillas-cuda"):
        requirements.append("numpy")
    return requirements


def get_requires_for_build_sdist(config_settings=None):
    """Get build requirements for sdist."""
    return _build_meta().get_requires_for_build_sdist(config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel."""
    return _build_meta().build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    """Build source distribution and embed SZ_TARGET.env for installs from sdist."""
    target = os.environ.get("SZ_TARGET", "stringzilla")
    marker_path = Path("SZ_TARGET.env")
    created = False
    try:
        if not marker_path.exists():
            marker_path.write_text(f"{target}\n", encoding="utf-8")
            created = True
        return _build_meta().build_sdist(sdist_directory, config_settings)
    finally:
        if created and marker_path.exists():
            try:
                marker_path.unlink()
            except OSError:
                pass


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare metadata for wheel build."""
    return _build_meta().prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """Prepare metadata for editable build (PEP 660)."""
    bm = _build_meta()
    if hasattr(bm, "prepare_metadata_for_build_editable"):
        return bm.prepare_metadata_for_build_editable(metadata_directory, config_settings)
    raise RuntimeError(
        "Editable installs require setuptools with PEP 660 support. "  #
        "Please upgrade setuptools (setuptools>=61)."
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable wheel (PEP 660)."""
    bm = _build_meta()
    if hasattr(bm, "build_editable"):
        return bm.build_editable(wheel_directory, config_settings, metadata_directory)
    raise RuntimeError(
        "Editable installs require setuptools with PEP 660 support. "  #
        "Please upgrade setuptools (setuptools>=61)."
    )


# ------------------------------
# CLI utilities for cibuildwheel
# ------------------------------


def _is_parallel_target() -> bool:
    return _detect_target() in ("stringzillas-cpus", "stringzillas-cuda")


def _build_sdist_for(target: str, outdir: str = "dist") -> None:
    """Build a single sdist with the provided SZ_TARGET value."""
    env = os.environ.copy()
    env["SZ_TARGET"] = target
    subprocess.check_call([sys.executable, "-m", "build", "--sdist", "--outdir", outdir], env=env)


def cli_build_sdists(outdir: str = "dist") -> None:
    """Build sdists for stringzilla, stringzillas-cpus, stringzillas-cuda.

    Ensures the PKG-INFO Name matches the intended PyPI package by setting
    SZ_TARGET for each build. Outputs to the provided directory (default: dist).
    """
    Path(outdir).mkdir(exist_ok=True)
    for target in ("stringzilla", "stringzillas-cpus", "stringzillas-cuda"):
        print(f"Building sdist for target: {target}")
        _build_sdist_for(target, outdir)


def cli_prepare_tests(project_dir: Optional[str] = None) -> None:
    """Prepare cibuildwheel test venv.

    - If building/testing a parallel target, install the serial baseline
      from the given project dir using build isolation.
    - Ensure NumPy and affine-gaps are present for test baselines.
    """
    if project_dir is None:
        project_dir = "."

    if _is_parallel_target():
        env = os.environ.copy()
        env["SZ_TARGET"] = "stringzilla"
        subprocess.check_call([sys.executable, "-m", "pip", "install", project_dir], env=env)
        # Install test-only deps required by test/stringzillas.py
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "affine-gaps"])  # noqa: S603


def cli_run_tests(project_dir: Optional[str] = None) -> None:
    """Run the appropriate tests for cibuildwheel."""
    if project_dir is None:
        project_dir = "."
    proj = Path(project_dir).resolve()
    sz_target = os.environ.get("SZ_TARGET", "stringzilla")
    if sz_target == "stringzilla":
        # The same file set the platform CI jobs run: everything under `test/` except the StringZillas modules.
        tests = [
            str(proj / "test"),
            "--ignore=" + str(proj / "test" / "stringzillas.py"),
            "--ignore=" + str(proj / "test" / "similarities.py"),
            "--ignore=" + str(proj / "test" / "fingerprints.py"),
            "--ignore=" + str(proj / "test" / "szs_helpers.py"),
        ]
    else:
        # For stringzillas-cpus and stringzillas-cuda, only test stringzillas
        # but we still need to build stringzilla as a dependency
        tests = [
            str(proj / "test" / "stringzillas.py"),
            str(proj / "test" / "similarities.py"),
            str(proj / "test" / "fingerprints.py"),
        ]
    # A free-threaded interpreter is the only place a data race can surface, and only if the suite runs
    # concurrently. Doctests sit out, asserting against a process-wide stdout capture that threads interleave.
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    parallel = [] if gil_enabled else ["--parallel-threads=4", "--ignore=" + str(proj / "test" / "doctests.py")]
    subprocess.check_call([sys.executable, "-m", "pytest", "-s", "-x", *parallel, *tests])  # noqa: S603


def cli_check_wheels(wheel_dir: str, required: List[str]) -> None:
    """Refuse a release that lost a core platform, while tolerating the long tail.

    The publish jobs run on `always()` so one flaky emulated leg cannot hold a release. That also
    means a wheel job failing outright would publish an sdist-only package with nothing going red.

    - Emulated arches, `musllinux` and Windows-on-Arm are expendable and may be missing.
    - Every glob in `required` must match at least one wheel filename, or this raises.

    Globs rather than substrings because the platform tag carries a version the caller should not
    have to spell out: a `manylinux` x86 wheel is tagged `manylinux_2_17_x86_64`, so the pattern
    has to be `*manylinux*_x86_64*`. Quote them in the shell, or they expand before we see them.
    """
    from fnmatch import fnmatch

    wheels = sorted(Path(wheel_dir).glob("*.whl"))
    missing = []
    for pattern in required:
        matched = [w.name for w in wheels if fnmatch(w.name, pattern)]
        if matched:
            print(f"ok: {len(matched)} wheel(s) matching {pattern!r}")
        else:
            print(f"::error::no wheel matching {pattern!r} - refusing to publish a partial release")
            missing.append(pattern)
    print(f"total wheels: {len(wheels)}")
    if missing:
        raise SystemExit(f"missing core platforms: {', '.join(missing)}")


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="build_backend.py",
        description="Backend wrapper and CI test helpers for StringZilla",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    parser_pull = sub.add_parser("pull-deps", help="Prepare cibuildwheel test environment")
    parser_pull.add_argument("project_dir", nargs="?", default=".")

    parser_run = sub.add_parser("run-tests", help="Run StringZilla test suites")
    parser_run.add_argument("project_dir", nargs="?", default=".")

    parser_sdists = sub.add_parser("build-sdists", help="Build sdists for all targets with correct metadata")
    parser_sdists.add_argument("--outdir", default="dist", help="Output directory for sdists (default: dist)")

    parser_check = sub.add_parser("check-wheels", help="Fail if a core platform produced no wheel")
    parser_check.add_argument("--wheel-dir", default="dist", help="Directory holding the wheels (default: dist)")
    parser_check.add_argument(
        "required", nargs="+", help="Globs a wheel filename must match, e.g. '*win_amd64*' (quote them)"
    )

    namespace = parser.parse_args(argv)
    if namespace.cmd == "pull-deps":
        cli_prepare_tests(namespace.project_dir)
        return 0
    if namespace.cmd == "run-tests":
        cli_run_tests(namespace.project_dir)
        return 0
    if namespace.cmd == "build-sdists":
        cli_build_sdists(namespace.outdir)
        return 0
    if namespace.cmd == "check-wheels":
        cli_check_wheels(namespace.wheel_dir, namespace.required)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
