import os
import sys
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension


compile_args = ["-std=c++17", "-O3"]
link_args = []
macros_args = []

if sys.platform == "linux":
    compile_args.append("-fopenmp")
    link_args.append("-lgomp")
if sys.platform == "darwin":
    compile_args.append("-Xpreprocessor -fopenmp")
    link_args.append("-Xpreprocessor -lomp")

compile_args.append("-Wno-unknown-pragmas")


ext_modules = [
    Pybind11Extension(
        "stringzilla.compiled",
        ["stringzilla/stringzilla.cpp"],
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
    packages=["stringzilla"],
    description="Smaller & Faster Single-File Vector Search Engine from Unum",
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
