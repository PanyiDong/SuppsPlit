"""
File Name: setup.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: sPlit
Latest Version: <<projectversion>>
Relative Path: /setup.py
File Created: Thursday, 11th September 2025 2:47:14 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 11th September 2025 5:10:41 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025 - 2025, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# setup.py
from setuptools import setup, Extension
import sys
import pybind11
import os

extra_compile_args = []
extra_link_args = []
include_dirs = [
    pybind11.get_include(),
    pybind11.get_include(user=True),
    # add any additional include paths, e.g. path to nanoflann.hpp if needed
    os.path.join(os.path.dirname(__file__), "src"),
]
library_dirs = []
libraries = []

if sys.platform == "darwin":
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-lomp"]
    # Homebrew paths (adjust if needed)
    include_dirs += ["/usr/local/include", "/opt/homebrew/include"]
    library_dirs += ["/usr/local/lib", "/opt/homebrew/lib"]
    libraries += ["omp"]
elif sys.platform != "win32":
    extra_compile_args += ["-O3", "-std=c++14", "-fopenmp"]
    extra_link_args += ["-fopenmp"]

ext_modules = [
    Extension(
        "splitpy",
        sources=["src/sp.cpp", "src/sPlit.cpp", "src/bindings.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
]

setup(
    name="splitpy",
    version="0.1",
    ext_modules=ext_modules,
)
