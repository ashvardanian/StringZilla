import os
import sys
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from typing import List, Tuple, Final
import subprocess
import concurrent.futures
import threading
import time


#: Peak resident set of one host-compiler pass over the templated similarity headers, and of one `cicc` pass
#: over the same headers with the CUDA kernels on top. The CUDA figure is the one that matters: four
#: concurrent `.cu` passes exhausted the 15.6 GB of a 4-core arm64 runner and the kernel killed the build.
CPP_MEMORY_PER_WORKER_GB: Final[float] = 2.0
CUDA_MEMORY_PER_WORKER_GB: Final[float] = 4.0


def _memory_available_and_total_gb():
    """`(available, total)` in GB from `/proc/meminfo`, or `None` where that file is absent, as on Windows."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            fields = {line.split(":", 1)[0]: line.split()[1] for line in handle if ":" in line}
        return int(fields["MemAvailable"]) / 1024**2, int(fields["MemTotal"]) / 1024**2
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _max_compile_workers(memory_per_worker_gb: float = CPP_MEMORY_PER_WORKER_GB) -> int:
    """Concurrency cap for compiling translation units, bounded by cores and by memory alike. Core count
    alone is the wrong bound on a small many-core box: a compiler killed by the out-of-memory killer takes
    the whole build down with no diagnostic. `SZ_MAX_COMPILE_WORKERS` overrides both bounds."""
    requested = os.environ.get("SZ_MAX_COMPILE_WORKERS", "")
    if requested.isdigit() and int(requested) > 0:
        return int(requested)
    workers = min(os.cpu_count() or 1, 8)
    memory = _memory_available_and_total_gb()
    if memory is not None:
        workers = min(workers, int(memory[0] // memory_per_worker_gb))
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


def _run_compilations_in_parallel(jobs, max_workers: int) -> None:
    """Run `(callable, label)` compile jobs concurrently, surfacing the first failure (and its label).
    Each job brackets itself in the log and a sampler records memory throughout, so a batch that dies without
    a compiler diagnostic still shows which translation units were in flight and what memory was doing."""
    if not jobs:
        return
    workers = max(1, min(max_workers, len(jobs)))
    _log_build_event(f"compiling {len(jobs)} source(s) with {workers} worker(s), memory {_memory_status()}")

    def _run_and_log(job, label):
        _log_build_event(f"start {label}")
        started_at = time.monotonic()
        job()
        _log_build_event(f"done  {label} in {time.monotonic() - started_at:.1f}s, memory {_memory_status()}")

    stop_sampler = _start_memory_sampler()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_and_log, job, label): label for job, label in jobs}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    raise RuntimeError(f"Compilation failed: {futures[future]}") from error
    finally:
        stop_sampler()


# CUDA architecture partition, kept in lockstep with the CMake build (`define_stringzillas_cuda_library` in
# CMakeLists.txt) and build.rs. The base tier executes natively on both supported generations (it is the only
# implementation for some ops even on sm_90), so it ships SASS for both plus forward PTX. The Kepler tier is
# superseded by Hopper on sm_90, so it ships sm_80 SASS only, with PTX covering forced-tier corners. The Hopper
# DPX tier starts at sm_90. sm_80/sm_90 are valid on CUDA 12.x and 13.x alike, so no per-toolkit probing.
_CUDA_ARCHES: Final = ["80-real", "90-real", "90-virtual"]
_KEPLER_ARCHES: Final = ["80-real", "80-virtual"]
_HOPPER_ARCHES: Final = ["90-real", "90-virtual"]


def _cuda_gencode_flags(cuda_source: str) -> List[str]:
    """`-gencode` flags for one `.cu`, by tier so no group carries dead SASS. Mirrors the CMake
    `<x>-real` / `<x>-virtual` arch lists: `-real` emits SASS (`code=sm_x`), `-virtual` emits PTX (`code=compute_x`)."""
    stem = os.path.splitext(os.path.basename(cuda_source))[0]
    if stem.endswith("_hopper"):
        arches = _HOPPER_ARCHES
    elif stem.endswith("_kepler"):
        arches = _KEPLER_ARCHES
    else:
        arches = _CUDA_ARCHES
    flags = []
    for arch in arches:
        number, kind = arch.split("-")
        code = f"sm_{number}" if kind == "real" else f"compute_{number}"
        flags += ["-gencode", f"arch=compute_{number},code={code}"]
    return flags


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

    # `-MMD/-MF` emits a makefile depfile listing the headers each TU pulled in, so an incremental rebuild can skip
    # a translation unit whose object is newer than every source AND header (distutils' own check tracks sources
    # only, which is why a header-only edit otherwise needs `--force`). MSVC has no `-MMD`, so it keeps the
    # source-only behavior. `_sz_force` mirrors `build_ext --force`.
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
            if ext in self._c_extensions:
                input_opt = "/Tc" + src
            elif ext in self._cpp_extensions:
                input_opt = "/Tp" + src
            else:
                return
            args = [self.cc] + msvc_compile_opts + pp_opts
            if ext in self._cpp_extensions:
                args.append("/EHsc")
            args.extend([input_opt, "/Fo" + obj])
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
        is_msvc = getattr(self.compiler, "compiler_type", "") == "msvc"
        # Host-compiler flags nvcc must forward to cl.exe / gcc, mirroring CMake and build.rs. On a Windows (MSVC)
        # host, CUDA 13.3 / CCCL 3.x needs the standard-conforming preprocessor, accurate `__cplusplus`, and UTF-8
        # source reading; `/Oi-` (MSVC) and `-fno-builtin-mem*` (GCC/Clang) stop the compiler from substituting
        # builtins for StringZilla's bytewise primitives. `-fPIC` is a GCC/Clang-only flag.
        if is_msvc:
            # nvcc has no host-CRT switch of its own, so the host `cl.exe` falls back to the static runtime while
            # setuptools compiles every C/C++ object against the dynamic one the interpreter itself uses, and the
            # two halves refuse to link with one `LNK2038` per CUDA object. An extension module shares CPython's
            # CRT, so the dynamic runtime is the only correct answer here.
            runtime_library_flag = "/MDd" if self.debug else "/MD"
            host_compiler_flags = ["/Zc:preprocessor", "/Zc:__cplusplus", "/utf-8", "/Oi-", runtime_library_flag]
        else:
            host_compiler_flags = [
                "-fPIC",
                "-fno-builtin-memcmp",
                "-fno-builtin-memchr",
                "-fno-builtin-memcpy",
                "-fno-builtin-memset",
            ]
        # nvcc forwards `-MMD` to a gcc host (Linux) but a cl host rejects it (`nvcc fatal: Unknown option '-MMD'`),
        # so gate the depfile off on MSVC — matching the C/C++ path's `use_depfiles = not is_msvc`.
        use_depfiles = not is_msvc
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
                "-std=c++20",
                "-O2",
                "--use_fast_math",
                "--expt-relaxed-constexpr",  # Allow constexpr functions in device code
                # Per-TU arch partition: one real cubin per GPU-major for the tier this TU serves, plus a single PTX
                # floor on the base tier - a no-overlap manual fatbin (see `_cuda_gencode_flags`).
                *_cuda_gencode_flags(cuda_source),
                "-Xfatbin=--compress-all",  # erases the size cost of multi-arch breadth (compressed ~= single arch)
                "-DSZ_DYNAMIC_DISPATCH=1",
                "-DSZ_USE_CUDA=1",
            ]
            if use_depfiles:
                # Emit a depfile so incremental rebuilds can skip translation units with no changed header.
                nvcc_cmd.extend(["-MMD", "-MF", dep_path])
            if host_cxx:
                nvcc_cmd.extend(["-ccbin", host_cxx])
            for inc_dir in ext.include_dirs:
                nvcc_cmd.extend(["-I", inc_dir])
            for define in ext.define_macros:
                if len(define) == 2:
                    nvcc_cmd.append(f"-D{define[0]}={define[1]}")
                else:
                    nvcc_cmd.append(f"-D{define[0]}")
            for flag in host_compiler_flags:
                nvcc_cmd.extend(["-Xcompiler", flag])

            if use_depfiles and not self.force and _object_is_fresh(obj_path, dep_path):
                _log_build_event(f"skip  {cuda_source} (object up to date)")
                continue
            nvcc_jobs.append((lambda command=nvcc_cmd: subprocess.check_call(command), cuda_source))

        _run_compilations_in_parallel(nvcc_jobs, _max_compile_workers(CUDA_MEMORY_PER_WORKER_GB))

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
    "c/stringzilla/cipher.c",
    "c/stringzilla/find.c",
    "c/stringzilla/sort.c",
    "c/stringzilla/intersect.c",
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
# StringZillas C-API entry units, compiled once per wheel - as C++ into the CPU one, as CUDA into the GPU one.
STRINGZILLAS_API_CPP_SOURCES = [
    "c/stringzillas/runtime.cpp",
    "c/stringzillas/levenshtein.cpp",
    "c/stringzillas/needleman_wunsch.cpp",
    "c/stringzillas/smith_waterman.cpp",
    "c/stringzillas/fingerprints.cpp",
    "c/stringzillas/substrings.cpp",
]
STRINGZILLAS_API_CU_SOURCES = [
    "c/stringzillas/runtime.cu",
    "c/stringzillas/levenshtein.cu",
    "c/stringzillas/needleman_wunsch.cu",
    "c/stringzillas/smith_waterman.cu",
    "c/stringzillas/fingerprints.cu",
    "c/stringzillas/substrings.cu",
]
# Per-ISA CPU instantiation units, host C++ in every wheel; off-platform files compile to empty objects.
STRINGZILLAS_CPUS_SOURCES = [
    "c/stringzillas/levenshtein_serial.cpp",
    "c/stringzillas/levenshtein_icelake.cpp",
    "c/stringzillas/levenshtein_haswell.cpp",
    "c/stringzillas/levenshtein_neon.cpp",
    "c/stringzillas/levenshtein_rvv.cpp",
    "c/stringzillas/needleman_wunsch_serial.cpp",
    "c/stringzillas/needleman_wunsch_icelake.cpp",
    "c/stringzillas/needleman_wunsch_haswell.cpp",
    "c/stringzillas/needleman_wunsch_neon.cpp",
    "c/stringzillas/needleman_wunsch_rvv.cpp",
    "c/stringzillas/smith_waterman_serial.cpp",
    "c/stringzillas/smith_waterman_icelake.cpp",
    "c/stringzillas/smith_waterman_haswell.cpp",
    "c/stringzillas/smith_waterman_neon.cpp",
    "c/stringzillas/smith_waterman_rvv.cpp",
    "c/stringzillas/substrings_serial.cpp",
]
# Per-tier GPU instantiation units, grouped by architecture floor: Hopper DPX needs sm_90, the rest run
# from the base set.
STRINGZILLAS_CUDA_SOURCES = [
    "c/stringzillas/levenshtein_cuda.cu",
    "c/stringzillas/needleman_wunsch_cuda.cu",
    "c/stringzillas/smith_waterman_cuda.cu",
    "c/stringzillas/substrings_cuda.cu",
]
STRINGZILLAS_KEPLER_SOURCES = [
    "c/stringzillas/levenshtein_kepler.cu",
]
STRINGZILLAS_HOPPER_SOURCES = [
    "c/stringzillas/levenshtein_hopper.cu",
    "c/stringzillas/needleman_wunsch_hopper.cu",
    "c/stringzillas/smith_waterman_hopper.cu",
]
# The compiled ForkUnion runtime rides along in every StringZillas extension as one more host C++ unit.
STRINGZILLAS_RUNTIME_SOURCES = ["forkunion/c/forkunion.cpp"]

ext_modules = []
entry_points = {}
command_class = {}

if sz_target == "stringzilla":
    __lib_name__ = "stringzilla"
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
            [
                "python/stringzillas/stringzillas.c",
                "python/stringzillas/device_scope.c",
                "python/stringzillas/similarities.c",
                "python/stringzillas/fingerprints.c",
                "python/stringzillas/substrings.c",
            ]
            + STRINGZILLAS_API_CPP_SOURCES
            + STRINGZILLAS_CPUS_SOURCES
            + STRINGZILLAS_RUNTIME_SOURCES
            # The multi-pattern rewrite path calls the dispatched `sz_copy`, so this wheel carries the core
            # runtime the way every CMake target links `stringzilla_static` rather than leaving it undefined.
            + STRINGZILLA_CORE_SOURCES,
            include_dirs=["include", "c/stringzillas", "forkunion/include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1"), ("SZ_USE_CUDA", "0"), ("FU_WITH_TOPOLOGY", "0")] + macros_args,
        ),
    ]
    command_class = {"build_ext": NumpyBuildExt}
elif sz_target == "stringzillas-cuda":
    __lib_name__ = "stringzillas-cuda"
    # Honor the standard CUDA_HOME / CUDA_PATH so the include + runtime-library paths can be pinned to a
    # toolkit whose `libcudart` the installed driver supports (mismatched runtimes raise
    # `cudaErrorInsufficientDriver` at load). Falls back to the `/usr/local/cuda` symlink.
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    # MSVC and the GNU/Clang toolchains spell library search paths and names differently. On Windows the CUDA
    # runtime + driver import libraries live under `lib\x64` and link.exe wants `/LIBPATH:` plus bare `*.lib`
    # names; elsewhere it is the usual `-L.../lib64 -l...`. The C++ runtime links implicitly under MSVC, so the
    # explicit `-lstdc++` is GNU/Clang-only.
    if os.name == "nt":
        cuda_link_args = link_args + [f"/LIBPATH:{os.path.join(cuda_home, 'lib', 'x64')}", "cudart.lib", "cuda.lib"]
    else:
        # `-lcuda` is the driver API; on a driver-less build host its only `libcuda.so` is the toolkit stub under
        # `lib64/stubs`, so that directory must be on the search path. The real `libcuda.so.1` takes over at load
        # time, and `auditwheel --exclude libcuda.so.1` keeps the stub out of the wheel.
        cuda_link_args = link_args + [
            f"-L{cuda_home}/lib64",
            f"-L{cuda_home}/lib64/stubs",
            "-lcudart",
            "-lcuda",
            "-lstdc++",
        ]
    ext_modules = [
        Extension(
            "stringzillas",
            [
                "python/stringzillas/stringzillas.c",
                "python/stringzillas/device_scope.c",
                "python/stringzillas/similarities.c",
                "python/stringzillas/fingerprints.c",
                "python/stringzillas/substrings.c",
            ]
            + STRINGZILLAS_API_CU_SOURCES
            + STRINGZILLAS_CPUS_SOURCES
            + STRINGZILLAS_CUDA_SOURCES
            + STRINGZILLAS_KEPLER_SOURCES
            + STRINGZILLAS_HOPPER_SOURCES
            + STRINGZILLAS_RUNTIME_SOURCES
            + STRINGZILLA_CORE_SOURCES,
            include_dirs=["include", "c/stringzillas", "forkunion/include", f"{cuda_home}/include"],
            extra_compile_args=compile_args,
            extra_link_args=cuda_link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1"), ("SZ_USE_CUDA", "1"), ("FU_WITH_TOPOLOGY", "0")] + macros_args,
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
