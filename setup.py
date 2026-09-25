"""Build script for the StringZilla Python package, compiling the CPython extension from C sources.

File: setup.py
Author: Ash Vardanian
Date: June 18, 2023
"""

import os
import sys
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from typing import List, Tuple, Final
import concurrent.futures
import threading
import time


#: Memory budgeted to one compiler pass, so a small many-core box is not driven into the
#: out-of-memory killer.
MEMORY_PER_WORKER_GB: Final[float] = 2.0


def _memory_available_and_total_gb():
    """`(available, total)` in GB from `/proc/meminfo`, or `None` where that file is absent, as on Windows."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            fields = {line.split(":", 1)[0]: line.split()[1] for line in handle if ":" in line}
        return int(fields["MemAvailable"]) / 1024**2, int(fields["MemTotal"]) / 1024**2
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _max_compile_workers() -> int:
    """Concurrency cap for compiling translation units, bounded by cores and by memory alike. Core count
    alone is the wrong bound on a small many-core box: a compiler killed by the out-of-memory killer takes
    the whole build down with no diagnostic. `STRINGZILLA_BUILD_JOBS` overrides both bounds."""
    requested = os.environ.get("STRINGZILLA_BUILD_JOBS", "")
    if requested.isdigit() and int(requested) > 0:
        return int(requested)
    workers = min(os.cpu_count() or 1, 8)
    memory = _memory_available_and_total_gb()
    if memory is not None:
        workers = min(workers, int(memory[0] // MEMORY_PER_WORKER_GB))
    return max(1, workers)


def _log_build_event(message: str) -> None:
    """Timestamped progress line, flushed on every call. A build killed by the host keeps only what already
    reached the log, so each step announces itself before starting rather than reporting after finishing."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _memory_status() -> str:
    """`available of total` in GB, or `unknown` where `/proc/meminfo` is absent."""
    memory = _memory_available_and_total_gb()
    return "unknown" if memory is None else f"{memory[0]:.1f} of {memory[1]:.1f} GB"


def _start_memory_sampler(interval_seconds: float = 10.0):
    """Log memory on an interval while a compile batch runs, returning the callable that stops it. A compiler
    killed by the out-of-memory killer reports nothing itself, so this trajectory is the only evidence left."""
    stop_event = threading.Event()

    def _sample() -> None:
        while not stop_event.wait(interval_seconds):
            _log_build_event(f"memory {_memory_status()}")

    threading.Thread(target=_sample, daemon=True).start()
    return stop_event.set


def _depfile_prerequisites(dep_path: str):
    """Parse a `-MMD/-MF` makefile fragment into its list of prerequisite paths, or `None` if absent."""
    try:
        with open(dep_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    text = text.replace("\\\n", " ")  # un-escape the line continuations make uses
    if ":" in text:
        text = text.split(":", 1)[1]  # drop the `target.o:` prefix, keep the prerequisites
    return [token for token in text.split() if token]


def _object_is_fresh(obj_path: str, dep_path: str) -> bool:
    """True when `obj_path` exists and is newer than every header/source in its depfile (so it can be skipped).
    Conservative: a missing depfile (e.g. a first build) returns False so the object is (re)compiled."""
    if not os.path.exists(obj_path):
        return False
    prerequisites = _depfile_prerequisites(dep_path)
    if not prerequisites:
        return False
    object_mtime = os.path.getmtime(obj_path)
    for prerequisite in prerequisites:
        try:
            if os.path.getmtime(prerequisite) > object_mtime:
                return False
        except OSError:
            return False  # a prerequisite vanished -> rebuild
    return True


def _parallel_compiler_compile(
    self,
    sources,
    output_dir=None,
    macros=None,
    include_dirs=None,
    debug=0,
    extra_preargs=None,
    extra_postargs=None,
    depends=None,
):
    """A parallel drop-in for `distutils.ccompiler.CCompiler.compile`, which compiles the sources of one
    extension serially. Reuses the compiler's own `_setup_compile` / `_get_cc_args` / `_compile`, so the exact
    flags distutils would pass are preserved; only the per-object loop is spread across a thread pool."""
    is_msvc = getattr(self, "compiler_type", "") == "msvc"
    if is_msvc and not self.initialized:
        self.initialize()

    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # `-MMD/-MF` emits a makefile depfile listing the headers each TU pulled in, so an incremental
    # rebuild can skip a translation unit whose object is newer than every source and header
    # (distutils' own check tracks sources only, which is why a header-only edit otherwise needs
    # `--force`). MSVC has no `-MMD`, so it keeps the source-only behavior. `_sz_force` mirrors the
    # `build_ext --force` option.
    use_depfiles = not is_msvc
    force = getattr(self, "_sz_force", False)

    # MSVC's `_compile` is a no-op — it does all work inside its `compile()` override. Build the per-object
    # command template once so each thread can compile independently.
    if is_msvc:
        msvc_compile_opts = list(extra_preargs or [])
        msvc_compile_opts.append("/c")
        if debug:
            msvc_compile_opts.extend(self.compile_options_debug)
        else:
            msvc_compile_opts.extend(self.compile_options)

    def _compile_one(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        if is_msvc:
            args = [self.cc] + msvc_compile_opts + pp_opts
            args.extend(["/Tc" + src, "/Fo" + obj])
            args.extend(extra_postargs)
            self.spawn(args)
        elif use_depfiles:
            dep_path = obj + ".d"
            if not force and _object_is_fresh(obj, dep_path):
                return
            self._compile(obj, src, ext, cc_args, extra_postargs + ["-MMD", "-MF", dep_path], pp_opts)
        else:
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    def _compile_one_logged(obj):
        label = build[obj][0] if obj in build else obj
        if use_depfiles and not force and _object_is_fresh(obj, obj + ".d"):
            _log_build_event(f"skip  {label} (object up to date)")
            return
        _log_build_event(f"start {label}")
        started_at = time.monotonic()
        _compile_one(obj)
        _log_build_event(f"done  {label} in {time.monotonic() - started_at:.1f}s, memory {_memory_status()}")

    workers = max(1, min(_max_compile_workers(), len(objects)))
    _log_build_event(f"compiling {len(objects)} source(s) with {workers} worker(s), memory {_memory_status()}")
    stop_sampler = _start_memory_sampler()
    try:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in pool.map(_compile_one_logged, objects):
                    pass
        else:
            for obj in objects:
                _compile_one_logged(obj)
    finally:
        stop_sampler()
    return objects


class ParallelBuildExt(build_ext):
    """`build_ext` compiling an extension's translation units across cores, where distutils compiles them serially."""

    def build_extension(self, ext):
        import types

        # The hook also skips translation units whose depfile shows nothing changed; `_sz_force` honors `--force`.
        if self.compiler is not None and not getattr(self.compiler, "_sz_parallelized", False):
            self.compiler._sz_force = bool(self.force)
            self.compiler.compile = types.MethodType(_parallel_compiler_compile, self.compiler)
            self.compiler._sz_parallelized = True
        super().build_extension(ext)


def get_compiler() -> str:
    if platform.python_implementation() == "CPython":
        compiler = platform.python_compiler().lower()
        return "gcc" if "gcc" in compiler else "llvm" if "clang" in compiler else ""
    return ""


def is_64bit_x86() -> bool:
    override = os.environ.get("STRINGZILLA_ARCH_X86_64_") if "STRINGZILLA_ARCH_X86_64_" in os.environ else None
    if override is not None:
        if override == "0":
            return False
        elif override == "1":
            return True
        else:
            raise ValueError("Invalid value for STRINGZILLA_ARCH_X86_64_: must be '0' or '1'")

    # Accept common 64-bit x86 identifiers and ensure the Python ABI is 64-bit.
    arch = platform.machine().lower()
    return (arch in ("x86_64", "x64", "amd64")) and (sys.maxsize > 2**32)


def is_64bit_arm() -> bool:
    override = os.environ.get("STRINGZILLA_ARCH_ARM64_") if "STRINGZILLA_ARCH_ARM64_" in os.environ else None
    if override is not None:
        if override == "0":
            return False
        elif override == "1":
            return True
        else:
            raise ValueError("Invalid value for STRINGZILLA_ARCH_ARM64_: must be '0' or '1'")

    # Accept common 64-bit ARM identifiers and ensure the Python ABI is 64-bit.
    arch = platform.machine().lower()
    return (arch in ("arm64", "aarch64")) and (sys.maxsize > 2**32)


def is_big_endian() -> bool:
    return sys.byteorder == "big"


def linux_settings() -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "-std=c99",
        "-D_GNU_SOURCE",  # enable POSIX extensions (sigaction, sigjmp_buf, etc.) when using -std=c99
        "-O2",  # optimization level
        "-fdiagnostics-color=always",  # color console output
        "-Wno-unknown-pragmas",  # like: `pragma region` and some unrolls
        "-Wno-unused-function",  # like: ... declared `static` but never defined
        "-fPIC",  # to enable dynamic dispatch
        "-g",  # include debug symbols for better debugging experience
        "-Wno-incompatible-pointer-types",  # like: passing argument 4 of `sz_export_prefix_u32` from incompatible pointer type
        "-Wno-discarded-qualifiers",  # like: passing argument 1 of `free` discards `const` qualifier from pointer target type
    ]
    link_args = [
        "-fPIC",  # to enable dynamic dispatch
    ]

    # GCC is our primary compiler, so when packaging the library, even if the current machine
    # doesn't support AVX-512 or SVE, still precompile those.
    macros_args = [
        ("STRINGZILLA_ARCH_BIG_ENDIAN_", "1" if is_big_endian() else "0"),
        ("STRINGZILLA_ARCH_X86_64_", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_ARCH_ARM64_", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_WESTMERE", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_GOLDMONT", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_ICELAKE", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_NEONAES", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_SVE", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_SVE2", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_SVE2AES", "1" if is_64bit_arm() else "0"),
    ]

    return compile_args, link_args, macros_args


def darwin_settings() -> Tuple[List[str], List[str], List[Tuple[str]]]:

    min_macos = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")

    # Force single-architecture builds to prevent `universal2`
    if is_64bit_arm():
        current_arch_flags = ["-arch", "arm64"]
    elif is_64bit_x86():
        current_arch_flags = ["-arch", "x86_64"]
    else:
        current_arch_flags = []

    compile_args = [
        "-std=c99",
        "-O2",  # optimization level
        "-fcolor-diagnostics",  # color console output
        "-Wno-unknown-pragmas",  # like: `pragma region` and some unrolls
        "-fPIC",  # to enable dynamic dispatch
        # "-mfloat-abi=hard",  # NEON intrinsics not available with the soft-float ABI
        f"-mmacosx-version-min={min_macos}",  # minimum macOS version (respect env if provided)
        *current_arch_flags,  # force single architecture to prevent universal2 builds
        "-Wno-incompatible-function-pointer-types",
        "-Wno-incompatible-pointer-types",  # like: passing argument 4 of `sz_export_prefix_u32` from incompatible pointer type
        "-Wno-ignored-qualifiers",  # Clang discard qualifiers warning name differs from GCC
    ]
    link_args = [
        "-fPIC",  # to enable dynamic dispatch
        *current_arch_flags,  # force single architecture to prevent universal2 builds
    ]

    # We only support single-arch macOS wheels, but not the Universal builds:
    # - x86_64: enable Westmere (SSE4.2), Goldmont (SHA-NI), and Haswell (AVX2) only
    # - arm64: enable NEON only
    macros_args = [
        ("STRINGZILLA_ARCH_X86_64_", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_ARCH_ARM64_", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_WESTMERE", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_GOLDMONT", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_HASWELL", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_SKYLAKE", "0"),
        ("STRINGZILLA_TARGET_ICELAKE", "0"),
        ("STRINGZILLA_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_NEONAES", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_SVE", "0"),
        ("STRINGZILLA_TARGET_SVE2", "0"),
    ]

    return compile_args, link_args, macros_args


def windows_settings() -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "/std:c11",  # MSVC has no C99
        "/W3",  # use W3 instead of /Wall to avoid excessive warnings
        "/O2",  # optimization level
        "/wd4365",  # disable C4365: signed/unsigned mismatch
        "/wd4820",  # disable C4820: padding added after data member
        "/wd5027",  # disable C5027: move assignment operator implicitly defined as deleted
        "/wd4626",  # disable C4626: assignment operator implicitly defined as deleted
        "/wd4127",  # disable C4127: conditional expression is constant
    ]

    # When packaging the library, even if the current machine doesn't support AVX-512 or SVE, still precompile those.
    macros_args = [
        ("STRINGZILLA_ARCH_BIG_ENDIAN_", "1" if is_big_endian() else "0"),
        ("STRINGZILLA_ARCH_X86_64_", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_ARCH_ARM64_", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_WESTMERE", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_GOLDMONT", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_ICELAKE", "1" if is_64bit_x86() else "0"),
        ("STRINGZILLA_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_NEONAES", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("STRINGZILLA_TARGET_SVE", "0"),
        ("STRINGZILLA_TARGET_SVE2", "0"),
    ]

    # MSVC requires architecture-specific macros for `winnt.h` to work correctly
    if is_64bit_arm():
        macros_args.append(("_ARM64_", "1"))
    elif is_64bit_x86():
        macros_args.append(("_AMD64_", "1"))

    link_args = []
    return compile_args, link_args, macros_args



if sys.platform == "linux" or sys.platform.startswith("freebsd"):
    compile_args, link_args, macros_args = linux_settings()

elif sys.platform == "darwin":
    compile_args, link_args, macros_args = darwin_settings()

elif sys.platform == "win32":
    compile_args, link_args, macros_args = windows_settings()

# TODO: It would be great to infer available compilation flags on FreeBSD. They are likely
# similar to Linux.
else:
    compile_args, link_args, macros_args = [], [], []

# The compiled C shims are split into one translation unit per domain; see c/stringzilla/.
STRINGZILLA_CORE_SOURCES = [
    "c/stringzilla/runtime.c",
    "c/stringzilla/compare.c",
    "c/stringzilla/memory.c",
    "c/stringzilla/hash.c",
    "c/stringzilla/cipher.c",
    "c/stringzilla/find.c",
    "c/stringzilla/sort.c",
    "c/stringzilla/intersect.c",
    "c/stringzilla/levenshtein.c",
    "c/stringzilla/overlap.c",
    "c/stringzilla/substrings.c",
    "c/stringzilla/utf8_norm.c",
    "c/stringzilla/utf8_runes.c",
    "c/stringzilla/utf8_tokens.c",
    "c/stringzilla/utf8_wordbreaks.c",
    "c/stringzilla/utf8_graphemes.c",
    "c/stringzilla/utf8_sentences.c",
    "c/stringzilla/utf8_linebreaks.c",
    "c/stringzilla/utf8_uncased_fold.c",
    "c/stringzilla/utf8_uncased.c",
]
ext_modules = [
    Extension(
        "stringzilla",
        [
            "python/stringzilla/stringzilla.c",
            "python/stringzilla/shared.c",
            "python/stringzilla/file.c",
            "python/stringzilla/str.c",
            "python/stringzilla/strs.c",
            "python/stringzilla/memory.c",
            "python/stringzilla/hash.c",
            "python/stringzilla/cipher.c",
            "python/stringzilla/compare.c",
            "python/stringzilla/find.c",
            "python/stringzilla/sort.c",
            "python/stringzilla/intersect.c",
            "python/stringzilla/levenshtein.c",
            "python/stringzilla/overlap.c",
            "python/stringzilla/substrings.c",
            "python/stringzilla/utf8_runes.c",
            "python/stringzilla/utf8_tokens.c",
            "python/stringzilla/utf8_boundaries.c",
            "python/stringzilla/utf8_wordbreaks.c",
            "python/stringzilla/utf8_graphemes.c",
            "python/stringzilla/utf8_sentences.c",
            "python/stringzilla/utf8_linebreaks.c",
            "python/stringzilla/utf8_uncased_fold.c",
            "python/stringzilla/utf8_uncased.c",
            "python/stringzilla/utf8_norm.c",
        ] + STRINGZILLA_CORE_SOURCES,
        include_dirs=["include", "c/stringzilla"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=[("STRINGZILLA_RUNTIME_DISPATCH", "1")] + macros_args,
    ),
]

__version__ = open("VERSION", "r").read().strip()

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

__description__ = "Search, hash, sort, and process strings faster via SWAR and SIMD"
install_requires = []

setup(
    name="stringzilla",
    version=__version__,
    description=__description__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/StringZilla",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
        "Topic :: File Formats",
        "Topic :: Internet :: Log Analysis",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: System :: Logging",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.10",
    include_dirs=[],
    setup_requires=[],
    ext_modules=ext_modules,
    packages=find_packages(),
    cmdclass={"build_ext": ParallelBuildExt},
    install_requires=install_requires,
)
