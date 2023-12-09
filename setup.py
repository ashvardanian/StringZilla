import os
import sys
import platform
from setuptools import setup, Extension
import glob

import numpy as np

compile_args = []
link_args = []
macros_args = [
    ("SZ_USE_X86_AVX512", "0"),
    ("SZ_USE_X86_AVX2", "1"),
    ("SZ_USE_X86_SSE42", "1"),
    ("SZ_USE_ARM_NEON", "0"),
    ("SZ_USE_ARM_CRC32", "0"),
]

if sys.platform == "linux":
    compile_args.append("-std=c99")
    compile_args.append("-O3")
    compile_args.append("-pedantic")
    compile_args.append("-fdiagnostics-color=always")

    compile_args.append("-Wno-unknown-pragmas")

    # Example: passing argument 4 of ‘sz_export_prefix_u32’ from incompatible pointer type
    compile_args.append("-Wno-incompatible-pointer-types")
    # Example: passing argument 1 of ‘free’ discards ‘const’ qualifier from pointer target type
    compile_args.append("-Wno-discarded-qualifiers")

    compile_args.append("-fopenmp")
    link_args.append("-lgomp")

    compiler = ""
    if platform.python_implementation() == "CPython":
        compiler = platform.python_compiler().lower()
        if "gcc" in compiler:
            compiler = "gcc"
        elif "clang" in compiler:
            compiler = "llvm"

    arch = platform.machine()
    if arch == "x86_64" or arch == "i386":
        compile_args.append("-march=native")
    elif arch.startswith("arm"):
        compile_args.append("-march=armv8-a+simd")
        if compiler == "gcc":
            compile_args.extend(["-mfpu=neon", "-mfloat-abi=hard"])


if sys.platform == "darwin":
    compile_args.append("-std=c99")
    compile_args.append("-O3")
    compile_args.append("-pedantic")
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-Wno-incompatible-function-pointer-types")
    compile_args.append("-Wno-incompatible-pointer-types")
    compile_args.append("-fcolor-diagnostics")
    compile_args.append("-Xpreprocessor -fopenmp")
    link_args.append("-Xpreprocessor -lomp")

if sys.platform == "win32":
    compile_args.append("/std:c99")
    compile_args.append("/O2")


ext_modules = [
    Extension(
        "stringzilla",
        ["python/lib.c"] + glob.glob("src/*.c"),
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
