import os
import sys
import platform
from setuptools import setup, find_packages, Extension
from typing import List, Tuple
import sysconfig
import glob


def get_compiler() -> str:
    if platform.python_implementation() == "CPython":
        compiler = platform.python_compiler().lower()
        return "gcc" if "gcc" in compiler else "llvm" if "clang" in compiler else ""
    return ""


using_cibuildwheels = os.environ.get("CIBUILDWHEEL", "0") == "1"


def is_64bit_x86() -> bool:
    if using_cibuildwheels:
        return "SZ_X86_64" in os.environ
    arch = platform.machine()
    return arch in ["x86_64", "x64", "AMD64"]


def is_64bit_arm() -> bool:
    if using_cibuildwheels:
        return "SZ_ARM64" in os.environ
    arch = platform.machine()
    return  arch in ["arm64", "aarch64", "ARM64"]


def is_big_endian() -> bool:
    return sys.byteorder == "big"


def linux_settings() -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "-std=c99",  # use the C 99 language dialect
        "-pedantic",  # stick close to the C language standard, avoid compiler extensions
        "-O3",  # maximum optimization level
        "-fdiagnostics-color=always",  # color console output
        "-Wno-unknown-pragmas",  # like: `pragma region` and some unrolls
        "-Wno-unused-function",  # like: ... declared ‘static’ but never defined
        "-Wno-incompatible-pointer-types",  # like: passing argument 4 of ‘sz_export_prefix_u32’ from incompatible pointer type
        "-Wno-discarded-qualifiers",  # like: passing argument 1 of ‘free’ discards ‘const’ qualifier from pointer target type
        "-fPIC",  # to enable dynamic dispatch
    ]
    link_args = [
        "-fPIC",  # to enable dynamic dispatch
    ]

    # GCC is our primary compiler, so when packaging the library, even if the current machine
    # doesn't support AVX-512 or SVE, still precompile those.
    macros_args = [
        ("SZ_USE_X86_AVX512", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_X86_AVX2", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ARM_SVE", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_ARM_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_DETECT_BIG_ENDIAN", "1" if is_big_endian() else "0"),
    ]

    return compile_args, link_args, macros_args


def darwin_settings() -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "-std=c99",  # use the C 99 language dialect
        "-pedantic",  # stick close to the C language standard, avoid compiler extensions
        "-O3",  # maximum optimization level
        "-fcolor-diagnostics",  # color console output
        "-Wno-unknown-pragmas",  # like: `pragma region` and some unrolls
        "-Wno-incompatible-function-pointer-types",
        "-Wno-incompatible-pointer-types",  # like: passing argument 4 of ‘sz_export_prefix_u32’ from incompatible pointer type
        "-Wno-discarded-qualifiers",  # like: passing argument 1 of ‘free’ discards ‘const’ qualifier from pointer target type
        "-fPIC",  # to enable dynamic dispatch
    ]
    link_args = [
        "-fPIC",  # to enable dynamic dispatch
    ]

    # Apple Clang doesn't support the `-march=native` argument,
    # so we must pre-set the CPU generation. Technically the last Intel-based Apple
    # product was the 2021 MacBook Pro, which had the "Coffee Lake" architecture.
    # During Universal builds, however, even AVX header cause compilation errors.
    can_use_avx2 = is_64bit_x86() and sysconfig.get_platform().startswith("universal")
    macros_args = [
        ("SZ_USE_X86_AVX512", "0"),
        ("SZ_USE_X86_AVX2", "1" if can_use_avx2 else "0"),
        ("SZ_USE_ARM_SVE", "0"),
        ("SZ_USE_ARM_NEON", "1" if is_64bit_arm() else "0"),
    ]

    return compile_args, link_args, macros_args


def windows_settings() -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "/std:c99",  # use the C 99 language dialect
        "/Wall",  # stick close to the C language standard, avoid compiler extensions
        "/O2",  # maximum optimization level
    ]

    # When packaging the library, even if the current machine doesn't support AVX-512 or SVE, still precompile those.
    macros_args = [
        ("SZ_USE_X86_AVX512", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_X86_AVX2", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ARM_SVE", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_ARM_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_DETECT_BIG_ENDIAN", "1" if is_big_endian() else "0"),
    ]

    link_args = []
    return compile_args, link_args, macros_args


if sys.platform == "linux":
    compile_args, link_args, macros_args = linux_settings()

elif sys.platform == "darwin":
    compile_args, link_args, macros_args = darwin_settings()

elif sys.platform == "win32":
    compile_args, link_args, macros_args = windows_settings()


ext_modules = [
    Extension(
        "stringzilla",
        ["python/lib.c"] + glob.glob("c/*.c"),
        # In the past I've used `np.get_include()` to include NumPy headers,
        # but it's not necessary for this library.
        include_dirs=["include"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=[("SZ_DYNAMIC_DISPATCH", "1")] + macros_args,
    ),
]

__version__ = open("VERSION", "r").read().strip()
__lib_name__ = "stringzilla"


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name=__lib_name__,
    version=__version__,
    author="Ash Vardanian",
    description="SIMD-accelerated string search, sort, hashes, fingerprints, & edit distances",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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
    include_dirs=[],
    setup_requires=[],
    ext_modules=ext_modules,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "sz_split=cli.split:main",
            "sz_wc=cli.wc:main",
        ],
    },
)
