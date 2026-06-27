import os
import sys
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from typing import List, Tuple, Final
import subprocess
import concurrent.futures


def _max_compile_workers() -> int:
    """Concurrency cap for compiling translation units. Each `cicc`/`cc1plus` pass on the heavily-templated
    similarity headers needs ~1-2 GB, so we cap below the core count to avoid thrashing or OOM on big boxes."""
    return max(1, min(os.cpu_count() or 1, 8))


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


def _run_compilations_in_parallel(jobs, max_workers: int) -> None:
    """Run `(callable, label)` compile jobs concurrently, surfacing the first failure (and its label)."""
    if not jobs:
        return
    workers = max(1, min(max_workers, len(jobs)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job): label for job, label in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(f"Compilation failed: {futures[future]}") from error


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
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # `-MMD/-MF` emits a makefile depfile listing the headers each TU pulled in, so an incremental rebuild can skip
    # a translation unit whose object is newer than every source AND header (distutils' own check tracks sources
    # only, which is why a header-only edit otherwise needs `--force`). MSVC has no `-MMD`, so it keeps the
    # source-only behavior. `_sz_force` mirrors `build_ext --force`.
    use_depfiles = getattr(self, "compiler_type", "") != "msvc"
    force = getattr(self, "_sz_force", False)

    def _compile_one(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        if use_depfiles:
            dep_path = obj + ".d"
            if not force and _object_is_fresh(obj, dep_path):
                return
            self._compile(obj, src, ext, cc_args, extra_postargs + ["-MMD", "-MF", dep_path], pp_opts)
        else:
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    workers = max(1, min(_max_compile_workers(), len(objects)))
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(_compile_one, objects):
                pass
    else:
        for obj in objects:
            _compile_one(obj)
    return objects


class ParallelBuildExt(build_ext):
    """
    Custom `build_ext` shared by every target: compiles an extension's many C/C++ translation units across cores
    (distutils compiles them serially) and applies per-language flags. Has no third-party dependency, so the base
    `stringzilla` CPython module uses it directly; `NumpyBuildExt` extends it for the numpy-dependent targets.
    """

    def build_extension(self, ext):
        import types

        # Swap in a parallel `compile` so the many C/C++ translation units build across cores like `make -j`; the
        # hook also carries header-depfile staleness skipping (see `_parallel_compiler_compile`). `_sz_force` lets
        # it honor `build_ext --force`.
        if self.compiler is not None and not getattr(self.compiler, "_sz_parallelized", False):
            self.compiler._sz_force = bool(self.force)
            self.compiler.compile = types.MethodType(_parallel_compiler_compile, self.compiler)
            self.compiler._sz_parallelized = True

        # Decide per-language compile flags using our platform helpers
        if sys.platform == "linux" or sys.platform.startswith("freebsd"):
            c_compile_args, _, _ = linux_settings(use_cpp=False)
            cpp_compile_args, _, _ = linux_settings(use_cpp=True)
        elif sys.platform == "darwin":
            c_compile_args, _, _ = darwin_settings(use_cpp=False)
            cpp_compile_args, _, _ = darwin_settings(use_cpp=True)
        elif sys.platform == "win32":
            c_compile_args, _, _ = windows_settings(use_cpp=False)
            cpp_compile_args, _, _ = windows_settings(use_cpp=True)
        else:
            c_compile_args, cpp_compile_args = [], []

        # Separate sources by language
        sources = list(ext.sources or [])
        c_sources = [s for s in sources if s.endswith(".c")]
        cpp_sources = [s for s in sources if s.endswith((".cc", ".cpp", ".cxx"))]

        # Compile sources with per-language flags
        objects: List[str] = []
        if c_sources:
            objects += self.compiler.compile(
                c_sources,
                output_dir=self.build_temp,
                macros=ext.define_macros,
                include_dirs=ext.include_dirs,
                debug=self.debug,
                extra_postargs=c_compile_args,
                depends=ext.depends,
            )
        if cpp_sources:
            objects += self.compiler.compile(
                cpp_sources,
                output_dir=self.build_temp,
                macros=ext.define_macros,
                include_dirs=ext.include_dirs,
                debug=self.debug,
                extra_postargs=cpp_compile_args,
                depends=ext.depends,
            )

        # Add any prebuilt/extra objects
        if getattr(ext, "extra_objects", None):
            objects += list(ext.extra_objects)

        # Link shared object
        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=getattr(ext, "runtime_library_dirs", None),
            extra_postargs=ext.extra_link_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=self.compiler.detect_language(ext.sources),
        )


class NumpyBuildExt(ParallelBuildExt):
    """
    Adds the NumPy include directory for the numpy-dependent targets (the `stringzillas` parallel-algorithm
    modules), deferring the `numpy` import until build time so `cibuildwheel`'s metadata pass — which may run
    before numpy is installed — does not need it. Everything else (parallel per-language compile + link) comes
    from `ParallelBuildExt`.
    """

    def build_extension(self, ext):
        import numpy as np

        numpy_include = np.get_include()
        if numpy_include not in ext.include_dirs:
            ext.include_dirs.append(numpy_include)
        super().build_extension(ext)


class CudaBuildExtension(NumpyBuildExt):
    """
    Custom `build_ext` class for CUDA extensions with deferred NumPy import.

    Compiles `.cu` files with `nvcc`, then delegates C/C++ compilation and
    linking to `NumpyBuildExt` on a per-extension basis.
    """

    def build_extension(self, ext):
        # If this extension has CUDA sources, precompile them with nvcc
        if any(source.endswith(".cu") for source in ext.sources or []):
            self._build_cuda_extension(ext)
        # Now compile remaining C/C++ sources and link
        super().build_extension(ext)

    def _build_cuda_extension(self, ext):
        # Separate CUDA and C sources
        cuda_sources = [s for s in ext.sources if s.endswith(".cu")]
        c_sources = [s for s in ext.sources if not s.endswith(".cu")]

        # Compile the CUDA sources with nvcc, concurrently, skipping any whose object is already up to date.
        # nvcc itself does not parallelize across input files (and `--threads` only splits per-`-gencode` passes,
        # of which we have one), so the only lever is running several nvcc processes at once - mirroring `make -j`.
        os.makedirs(self.build_temp, exist_ok=True)
        # nvcc rejects host compilers newer than it supports (CUDA 12.x caps out at GCC 14). Honor the standard
        # CUDAHOSTCXX so the caller can point nvcc at a compatible host compiler, mirroring CMake and build.rs.
        host_cxx = os.environ.get("CUDAHOSTCXX")
        objects = []
        nvcc_jobs = []
        for cuda_source in cuda_sources:
            obj_name = os.path.splitext(os.path.basename(cuda_source))[0] + ".o"
            obj_path = os.path.join(self.build_temp, obj_name)
            dep_path = obj_path + ".d"
            objects.append(obj_path)

            nvcc_cmd = [
                "nvcc",
                "-c",
                cuda_source,
                "-o",
                obj_path,
                "--compiler-options",
                "-fPIC",
                "-std=c++20",
                "-O2",
                "--use_fast_math",
                "--expt-relaxed-constexpr",  # Allow constexpr functions in device code
                "-arch=sm_90a",  # Default to Hopper
                "-DSZ_DYNAMIC_DISPATCH=1",
                "-DSZ_USE_CUDA=1",
                "-MMD",  # emit a depfile so incremental rebuilds can skip translation units with no changed header
                "-MF",
                dep_path,
            ]
            if host_cxx:
                nvcc_cmd.extend(["-ccbin", host_cxx])
            for inc_dir in ext.include_dirs:
                nvcc_cmd.extend(["-I", inc_dir])
            for define in ext.define_macros:
                if len(define) == 2:
                    nvcc_cmd.append(f"-D{define[0]}={define[1]}")
                else:
                    nvcc_cmd.append(f"-D{define[0]}")

            if not self.force and _object_is_fresh(obj_path, dep_path):
                print(f"Skipping {cuda_source} (object up to date)")
                continue
            nvcc_jobs.append((lambda command=nvcc_cmd: subprocess.check_call(command), cuda_source))

        if nvcc_jobs:
            print(f"Compiling {len(nvcc_jobs)} CUDA source(s) with nvcc ({_max_compile_workers()} parallel workers)...")
        _run_compilations_in_parallel(nvcc_jobs, _max_compile_workers())

        # Update extension: remove .cu sources, add compiled objects
        ext.sources = c_sources
        ext.extra_objects = getattr(ext, "extra_objects", []) + objects

        # After producing CUDA objects, fall through to NumpyBuildExt which
        # will compile C/C++ sources per-language and link everything.


def sz_target_name() -> str:
    # Prefer env var, then a simple marker file, else default
    val = os.environ.get("SZ_TARGET")
    if val:
        return val
    try:
        with open("SZ_TARGET.env", "r", encoding="utf-8") as f:
            v = f.read().strip()
            if v:
                return v
    except FileNotFoundError:
        pass
    return "stringzilla"


sz_target: Final[str] = sz_target_name()


def get_compiler() -> str:
    if platform.python_implementation() == "CPython":
        compiler = platform.python_compiler().lower()
        return "gcc" if "gcc" in compiler else "llvm" if "clang" in compiler else ""
    return ""


def is_64bit_x86() -> bool:
    override = os.environ.get("SZ_IS_64BIT_X86_") if "SZ_IS_64BIT_X86_" in os.environ else None
    if override is not None:
        if override == "0":
            return False
        elif override == "1":
            return True
        else:
            raise ValueError("Invalid value for SZ_IS_64BIT_X86_: must be '0' or '1'")

    # Accept common 64-bit x86 identifiers and ensure the Python ABI is 64-bit.
    arch = platform.machine().lower()
    return (arch in ("x86_64", "x64", "amd64")) and (sys.maxsize > 2**32)


def is_64bit_arm() -> bool:
    override = os.environ.get("SZ_IS_64BIT_ARM_") if "SZ_IS_64BIT_ARM_" in os.environ else None
    if override is not None:
        if override == "0":
            return False
        elif override == "1":
            return True
        else:
            raise ValueError("Invalid value for SZ_IS_64BIT_ARM_: must be '0' or '1'")

    # Accept common 64-bit ARM identifiers and ensure the Python ABI is 64-bit.
    arch = platform.machine().lower()
    return (arch in ("arm64", "aarch64")) and (sys.maxsize > 2**32)


def is_big_endian() -> bool:
    return sys.byteorder == "big"


def linux_settings(use_cpp: bool = False) -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "-std=c++17" if use_cpp else "-std=c99",  # use C++17 for StringZillas, C99 for StringZilla
        "-D_GNU_SOURCE",  # enable POSIX extensions (sigaction, sigjmp_buf, etc.) when using -std=c99
        "-O2",  # optimization level
        "-fdiagnostics-color=always",  # color console output
        "-Wno-unknown-pragmas",  # like: `pragma region` and some unrolls
        "-Wno-unused-function",  # like: ... declared `static` but never defined
        "-fPIC",  # to enable dynamic dispatch
        "-g",  # include debug symbols for better debugging experience
    ]
    # Add C-specific warning suppressions only for C compilation
    if not use_cpp:
        compile_args += [
            "-Wno-incompatible-pointer-types",  # like: passing argument 4 of `sz_export_prefix_u32` from incompatible pointer type
            "-Wno-discarded-qualifiers",  # like: passing argument 1 of `free` discards `const` qualifier from pointer target type
        ]
    link_args = [
        "-fPIC",  # to enable dynamic dispatch
    ]

    # GCC is our primary compiler, so when packaging the library, even if the current machine
    # doesn't support AVX-512 or SVE, still precompile those.
    macros_args = [
        ("SZ_IS_BIG_ENDIAN_", "1" if is_big_endian() else "0"),
        ("SZ_IS_64BIT_X86_", "1" if is_64bit_x86() else "0"),
        ("SZ_IS_64BIT_ARM_", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_WESTMERE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_GOLDMONT", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ICELAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONAES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE2", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE2AES", "1" if is_64bit_arm() else "0"),
    ]

    return compile_args, link_args, macros_args


def darwin_settings(use_cpp: bool = False) -> Tuple[List[str], List[str], List[Tuple[str]]]:

    min_macos = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")

    # Force single-architecture builds to prevent `universal2`
    if is_64bit_arm():
        current_arch_flags = ["-arch", "arm64"]
    elif is_64bit_x86():
        current_arch_flags = ["-arch", "x86_64"]
    else:
        current_arch_flags = []

    compile_args = [
        "-std=c++17" if use_cpp else "-std=c99",  # use C++17 for StringZillas, C99 for StringZilla
        "-O2",  # optimization level
        "-fcolor-diagnostics",  # color console output
        "-Wno-unknown-pragmas",  # like: `pragma region` and some unrolls
        "-fPIC",  # to enable dynamic dispatch
        # "-mfloat-abi=hard",  # NEON intrinsics not available with the soft-float ABI
        f"-mmacosx-version-min={min_macos}",  # minimum macOS version (respect env if provided)
        *current_arch_flags,  # force single architecture to prevent universal2 builds
    ]
    # Add C-specific warning suppressions only for C compilation
    if not use_cpp:
        compile_args += [
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
        ("SZ_IS_64BIT_X86_", "1" if is_64bit_x86() else "0"),
        ("SZ_IS_64BIT_ARM_", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_WESTMERE", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("SZ_USE_GOLDMONT", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("SZ_USE_HASWELL", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "0"),
        ("SZ_USE_ICELAKE", "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONAES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "0"),
        ("SZ_USE_SVE2", "0"),
    ]

    return compile_args, link_args, macros_args


def windows_settings(use_cpp: bool = False) -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "/std:c++17" if use_cpp else "/std:c11",  # use C++17 for StringZillas, C11 for StringZilla, as MSVC has no C99
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
        ("SZ_IS_BIG_ENDIAN_", "1" if is_big_endian() else "0"),
        ("SZ_IS_64BIT_X86_", "1" if is_64bit_x86() else "0"),
        ("SZ_IS_64BIT_ARM_", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_WESTMERE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_GOLDMONT", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ICELAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONAES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "0"),
        ("SZ_USE_SVE2", "0"),
    ]

    # MSVC requires architecture-specific macros for `winnt.h` to work correctly
    if is_64bit_arm():
        macros_args.append(("_ARM64_", "1"))
    elif is_64bit_x86():
        macros_args.append(("_AMD64_", "1"))

    link_args = []
    return compile_args, link_args, macros_args


use_cpp: Final[bool] = sz_target != "stringzilla"

if sys.platform == "linux" or sys.platform.startswith("freebsd"):
    compile_args, link_args, macros_args = linux_settings(use_cpp=use_cpp)

elif sys.platform == "darwin":
    compile_args, link_args, macros_args = darwin_settings(use_cpp=use_cpp)

elif sys.platform == "win32":
    compile_args, link_args, macros_args = windows_settings(use_cpp=use_cpp)

# TODO: It would be great to infer available compilation flags on FreeBSD. They are likely similar to Linux
else:
    compile_args, link_args, macros_args = [], [], []

# The compiled C shims are split into one translation unit per domain (single-CPU) and per
# algorithm (parallel); each binding lists the full set. See c/stringzilla/ and c/stringzillas/.
STRINGZILLA_CORE_SOURCES = [
    "c/stringzilla/runtime.c",
    "c/stringzilla/compare.c",
    "c/stringzilla/memory.c",
    "c/stringzilla/hash.c",
    "c/stringzilla/find.c",
    "c/stringzilla/sort.c",
    "c/stringzilla/intersect.c",
    "c/stringzilla/utf8_norm.c",
    "c/stringzilla/utf8_codepoints.c",
    "c/stringzilla/utf8_tokens.c",
    "c/stringzilla/utf8_words.c",
    "c/stringzilla/utf8_graphemes.c",
    "c/stringzilla/utf8_sentences.c",
    "c/stringzilla/utf8_linebreaks.c",
    "c/stringzilla/utf8_uncased_fold.c",
    "c/stringzilla/utf8_uncased.c",
]
STRINGZILLAS_PARALLEL_STEMS = ["runtime", "levenshtein", "needleman_wunsch", "smith_waterman", "fingerprints"]
# Per-capability instantiation units: each emits one ISA's (CPU) or tier's (CUDA) heavy engine code exactly once, so
# the algorithm entry TUs above only declare them `extern` and link against them. Off-platform files compile to empty
# objects via their internal SZ_USE_* guards. These lists mirror the CMake STRINGZILLAS_*_SOURCES.
STRINGZILLAS_CPU_PROVIDER_STEMS = [
    "levenshtein_serial",
    "levenshtein_icelake",
    "needleman_wunsch_serial",
    "needleman_wunsch_icelake",
    "needleman_wunsch_haswell",
    "needleman_wunsch_neon",
    "smith_waterman_serial",
    "smith_waterman_icelake",
    "smith_waterman_haswell",
    "smith_waterman_neon",
]
STRINGZILLAS_CUDA_PROVIDER_STEMS = [
    "levenshtein_cuda",
    "levenshtein_kepler",
    "levenshtein_hopper",
    "needleman_wunsch_cuda",
    "needleman_wunsch_hopper",
    "smith_waterman_cuda",
    "smith_waterman_hopper",
]
STRINGZILLAS_CPU_SOURCES = [
    f"c/stringzillas/{stem}.cpp" for stem in STRINGZILLAS_PARALLEL_STEMS + STRINGZILLAS_CPU_PROVIDER_STEMS
]
STRINGZILLAS_CUDA_SOURCES = [f"c/stringzillas/{stem}.cu" for stem in STRINGZILLAS_PARALLEL_STEMS] + [
    f"c/stringzillas/{stem}.cu" for stem in STRINGZILLAS_CUDA_PROVIDER_STEMS
]

ext_modules = []
entry_points = {}
command_class = {}

if sz_target == "stringzilla":
    __lib_name__ = "stringzilla"
    ext_modules = [
        Extension(
            "stringzilla",
            ["python/stringzilla.c"] + STRINGZILLA_CORE_SOURCES,
            include_dirs=["include", "c/stringzilla"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1")] + macros_args,
        ),
    ]
    # The `sz_split` / `sz_wc` CLIs moved to the standalone StringZilla-CLI repository.
    entry_points = {"console_scripts": []}
    # Parallel per-language compile + header-depfile incremental rebuilds for the 17 core C TUs (the base module
    # has no numpy dependency, so it uses ParallelBuildExt directly rather than NumpyBuildExt).
    command_class = {"build_ext": ParallelBuildExt}
elif sz_target == "stringzillas-cpus":
    __lib_name__ = "stringzillas-cpus"
    ext_modules = [
        Extension(
            "stringzillas",
            ["python/stringzillas.c"] + STRINGZILLAS_CPU_SOURCES,
            include_dirs=["include", "c/stringzillas", "fork_union/include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1"), ("SZ_USE_CUDA", "0"), ("FU_ENABLE_NUMA", "0")] + macros_args,
        ),
    ]
    command_class = {"build_ext": NumpyBuildExt}
elif sz_target == "stringzillas-cuda":
    __lib_name__ = "stringzillas-cuda"
    # Honor the standard CUDA_HOME / CUDA_PATH so the include + runtime-library paths can be pinned to a
    # toolkit whose `libcudart` the installed driver supports (mismatched runtimes raise
    # `cudaErrorInsufficientDriver` at load). Falls back to the `/usr/local/cuda` symlink.
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    ext_modules = [
        Extension(
            "stringzillas",
            ["python/stringzillas.c"] + STRINGZILLAS_CUDA_SOURCES,
            include_dirs=["include", "c/stringzillas", "fork_union/include", f"{cuda_home}/include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args + [f"-L{cuda_home}/lib64", "-lcudart", "-lcuda", "-lstdc++"],
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1"), ("SZ_USE_CUDA", "1"), ("FU_ENABLE_NUMA", "0")] + macros_args,
            language="c++",  # Force C++ linking
        ),
    ]
    command_class = {"build_ext": CudaBuildExtension}
else:
    raise ValueError("Unknown target specified with SZ_TARGET environment variable.")


__version__ = open("VERSION", "r").read().strip()

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

# Different descriptions for different variants
if sz_target == "stringzilla":
    __description__ = "Search, hash, sort, and process strings faster via SWAR and SIMD"
elif sz_target == "stringzillas-cpus":
    __description__ = (
        "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, on multi-core CPUs"
    )
elif sz_target == "stringzillas-cuda":
    __description__ = (
        "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and CUDA on Nvidia GPUs"
    )
elif sz_target == "stringzillas-rocm":
    __description__ = (
        "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and ROCm on AMD GPUs"
    )
else:
    __description__ = "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU"

# Ensure multi-backend packages depend on the base CPython module
install_requires = []
if sz_target != "stringzilla":
    # Keep versions in lockstep to ensure ABI compatibility
    install_requires = [f"stringzilla=={__version__}"]

setup(
    name=__lib_name__,
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
    entry_points=entry_points,
    cmdclass=command_class,
    install_requires=install_requires,
)
