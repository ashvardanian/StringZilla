import os
import sys
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension


compile_args = []
link_args = []
macros_args = []

if sys.platform == "linux":
    compile_args.append("-std=c++17")
    compile_args.append("-O3")
    compile_args.append("-pedantic")
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-fopenmp")
    link_args.append("-lgomp")

if sys.platform == "darwin":
    compile_args.append("-std=c++17")
    compile_args.append("-O3")
    compile_args.append("-pedantic")
    compile_args.append("-Wno-unknown-pragmas")
    compile_args.append("-Xpreprocessor -fopenmp")
    link_args.append("-Xpreprocessor -lomp")

if sys.platform == "win32":
    compile_args.append("/std:c++17")
    compile_args.append("/O2")


ext_modules = [
    Pybind11Extension(
        "stringzilla",
        ["stringzilla/pybind11.cpp"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros_args,
    ),
]

__version__ = open("VERSION", "r").read().strip()
__lib_name__ = "stringzilla"


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()


setup(
    name=__lib_name__,
    version=__version__,
    description="Crunch 100+ GB Strings in Python with ease",
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
    ],
    include_dirs=[],
    ext_modules=ext_modules,
)
