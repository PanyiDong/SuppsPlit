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
Last Modified: Thursday, 11th September 2025 4:03:37 pm
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

include_dirs = [
    pybind11.get_include(),
    pybind11.get_include(user=True),
    # add any additional include paths, e.g. path to nanoflann.hpp if needed
    os.path.join(os.path.dirname(__file__), "src"),
]

extra_compile_args = ["-O3", "-std=c++14"]
extra_link_args = []

if sys.platform != "win32":
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]

ext_modules = [
    Extension(
        "splitpy",
        sources=["src/sp.cpp", "src/sPlit.cpp", "src/bindings.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["armadillo"],  # link to Armadillo
    )
]

setup(
    name="splitpy",
    version="0.1",
    ext_modules=ext_modules,
)
