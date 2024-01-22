import os
import sys
import platform
from setuptools import setup, Extension
from typing import List, Tuple
import glob

import numpy as np


def get_compiler() -> str:
    if platform.python_implementation() == "CPython":
        compiler = platform.python_compiler().lower()
        return "gcc" if "gcc" in compiler else "llvm" if "clang" in compiler else ""
    return ""


def is_64bit_x86() -> bool:
    arch = platform.machine()
    return arch == "x86_64" or arch == "i386"


def is_64bit_arm() -> bool:
    arch = platform.machine()
    return arch.startswith("arm")


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
    ]

    # GCC is our primary compiler, so when packaging the library, even if the current machine
    # doesn't support AVX-512 or SVE, still precompile those.
    macros_args = [
        ("SZ_USE_X86_AVX512", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_X86_AVX2", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ARM_SVE", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_ARM_NEON", "1" if is_64bit_arm() else "0"),
    ]

    if is_64bit_x86():
        compile_args.append("-march=native")
    elif is_64bit_arm():
        compile_args.append("-march=armv8-a+simd")

    link_args = []
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
    ]

    # GCC is our primary compiler, so when packaging the library, even if the current machine
    # doesn't support AVX-512 or SVE, still precompile those.
    macros_args = [
        ("SZ_USE_X86_AVX512", "0"),
        ("SZ_USE_X86_AVX2", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ARM_SVE", "0"),
        ("SZ_USE_ARM_NEON", "1" if is_64bit_arm() else "0"),
    ]

    # Apple Clang doesn't support the `-march=native` argument,
    # so we must pre-set the CPU generation. Technically the last Intel-based Apple
    # product was the 2021 MacBook Pro, which had the "Coffee Lake" architecture.
    # It's feature-set matches the "skylake" generation code for LLVM and GCC.
    if is_64bit_x86():
        compile_args.append("-march=skylake")
    # None of Apple products support SVE instructions for now.
    elif is_64bit_arm():
        compile_args.append("-march=armv8-a+simd")

    link_args = []
    return compile_args, link_args, macros_args


def windows_settings() -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "/std:c99",  # use the C 99 language dialect
        "/Wall",  # stick close to the C language standard, avoid compiler extensions
        "/O2",  # maximum optimization level
    ]

    # Detect supported architectures for MSVC.
    macros_args = []
    if "AVX512" in platform.processor():
        macros_args.append(("SZ_USE_X86_AVX512", "1"))
        compile_args.append("/arch:AVX512")
    if "AVX2" in platform.processor():
        macros_args.append(("SZ_USE_X86_AVX2", "1"))
        compile_args.append("/arch:AVX2")

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
        include_dirs=["include", np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros_args,
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
    description="Crunch multi-gigabyte strings with ease",
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
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Topic :: File Formats",
        "Topic :: Internet :: Log Analysis",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: System :: Logging",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Indexing",
    ],
    include_dirs=[],
    setup_requires=["numpy"],
    ext_modules=ext_modules,
)
