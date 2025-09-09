import os
import sys
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from typing import List, Tuple, Final
import subprocess


class NumpyBuildExt(build_ext):
    """
    Custom build_ext class that defers `numpy` import until build time.

    This is necessary because NumPy may not be available during the initial
    `setup.py` parsing phase (e.g., when `cibuildwheel` is gathering build requirements),
    but we need NumPy's include directories during the actual compilation.
    By deferring the import to `build_extensions()`, we ensure NumPy is only
    required when actually building the extensions, not when querying metadata.
    """

    def build_extension(self, ext):
        import numpy as np

        # Ensure NumPy headers are available
        numpy_include = np.get_include()
        if numpy_include not in ext.include_dirs:
            ext.include_dirs.append(numpy_include)

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

        # Compile CUDA files with nvcc first
        objects = []
        for cuda_source in cuda_sources:
            # Generate object file path
            obj_name = os.path.splitext(os.path.basename(cuda_source))[0] + ".o"
            obj_path = os.path.join(self.build_temp, obj_name)
            os.makedirs(self.build_temp, exist_ok=True)

            # NVCC command
            nvcc_cmd = [
                "nvcc",
                "-c",
                cuda_source,
                "-o",
                obj_path,
                "--compiler-options",
                "-fPIC",
                "-std=c++17",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",  # Allow constexpr functions in device code
                "-arch=sm_90a",  # Default to Hopper
                "-DSZ_DYNAMIC_DISPATCH=1",
                "-DSZ_USE_CUDA=1",
            ]

            # Add include directories
            for inc_dir in ext.include_dirs:
                nvcc_cmd.extend(["-I", inc_dir])

            # Add defines
            for define in ext.define_macros:
                if len(define) == 2:
                    nvcc_cmd.append(f"-D{define[0]}={define[1]}")
                else:
                    nvcc_cmd.append(f"-D{define[0]}")

            print(f"Compiling {cuda_source} with nvcc...")
            subprocess.check_call(nvcc_cmd)
            objects.append(obj_path)

        # Update extension: remove .cu sources, add compiled objects
        ext.sources = c_sources
        ext.extra_objects = getattr(ext, "extra_objects", []) + objects

        # After producing CUDA objects, fall through to NumpyBuildExt which
        # will compile C/C++ sources per-language and link everything.


using_cibuildwheel: Final[str] = os.environ.get("CIBUILDWHEEL", "0") == "1"
sz_target: Final[str] = os.environ.get("SZ_TARGET", "stringzilla")


def get_compiler() -> str:
    if platform.python_implementation() == "CPython":
        compiler = platform.python_compiler().lower()
        return "gcc" if "gcc" in compiler else "llvm" if "clang" in compiler else ""
    return ""


def is_64bit_x86() -> bool:
    if using_cibuildwheel:
        return "SZ_IS_64BIT_X86_" in os.environ
    arch = platform.machine()
    return arch in ["x86_64", "x64", "AMD64"]


def is_64bit_arm() -> bool:
    if using_cibuildwheel:
        return "SZ_IS_64BIT_ARM_" in os.environ
    arch = platform.machine()
    return arch in ["arm64", "aarch64", "ARM64"]


def is_big_endian() -> bool:
    return sys.byteorder == "big"


def linux_settings(use_cpp: bool = False) -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "-std=c++17" if use_cpp else "-std=c99",  # use C++17 for StringZillas, C99 for StringZilla
        "-pedantic",  # stick close to the C language standard, avoid compiler extensions
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
        ("SZ_USE_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ICE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEON_AES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE2", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE2_AES", "1" if is_64bit_arm() else "0"),
        ("SZ_DETECT_BIG_ENDIAN", "1" if is_big_endian() else "0"),
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
        "-pedantic",  # stick close to the C language standard, avoid compiler extensions
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
    # - x86_64: enable Haswell (AVX2) only
    # - arm64: enable NEON only
    macros_args = [
        ("SZ_USE_HASWELL", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "0"),
        ("SZ_USE_ICE", "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEON_AES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "0"),
        ("SZ_USE_SVE2", "0"),
    ]

    return compile_args, link_args, macros_args


def windows_settings(use_cpp: bool = False) -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "/std:c++17" if use_cpp else "/std:c11",  # use C++17 for StringZillas, C11 for StringZilla, as MSVC has no C99
        "/Wall",  # stick close to the C language standard, avoid compiler extensions
        "/O2",  # optimization level
    ]

    # When packaging the library, even if the current machine doesn't support AVX-512 or SVE, still precompile those.
    macros_args = [
        ("SZ_USE_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ICE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEON_AES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "0"),
        ("SZ_USE_SVE2", "0"),
        ("SZ_DETECT_BIG_ENDIAN", "1" if is_big_endian() else "0"),
    ]

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

ext_modules = []
entry_points = {}
command_class = {}

if sz_target == "stringzilla":
    __lib_name__ = "stringzilla"
    ext_modules = [
        Extension(
            "stringzilla",
            ["python/stringzilla.c", "c/stringzilla.c"],
            include_dirs=["include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1")] + macros_args,
        ),
    ]
    entry_points = {
        "console_scripts": [
            "sz_split=cli.split:main",
            "sz_wc=cli.wc:main",
        ],
    }
elif sz_target == "stringzillas-cpus":
    __lib_name__ = "stringzillas-cpus"
    ext_modules = [
        Extension(
            "stringzillas",
            ["python/stringzillas.c", "c/stringzillas.cpp"],
            include_dirs=["include", "c", "fork_union/include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1"), ("SZ_USE_CUDA", "0")] + macros_args,
        ),
    ]
    command_class = {"build_ext": NumpyBuildExt}
elif sz_target == "stringzillas-cuda":
    __lib_name__ = "stringzillas-cuda"
    ext_modules = [
        Extension(
            "stringzillas",
            ["python/stringzillas.c", "c/stringzillas.cu"],
            include_dirs=["include", "c", "fork_union/include", "/usr/local/cuda/include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args + ["-L/usr/local/cuda/lib64", "-lcudart", "-lcuda", "-lstdc++"],
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1"), ("SZ_USE_CUDA", "1")] + macros_args,
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


setup(
    name=__lib_name__,
    version=__version__,
    description="Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU",
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/stringzilla",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
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
    python_requires=">=3.8",
    include_dirs=[],
    setup_requires=[],
    ext_modules=ext_modules,
    packages=find_packages(),
    entry_points=entry_points,
    cmdclass=command_class,
)
