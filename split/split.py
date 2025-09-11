"""
File Name: split.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: split
Latest Version: <<projectversion>>
Relative Path: /split.py
File Created: Thursday, 11th September 2025 3:07:38 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 11th September 2025 4:02:43 pm
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

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import splitpy   # the compiled C++ extension

def compute_sp(n, p, dist_samp, num_subsamp, iter_max=500, tol=1e-10, n_threads=1):
    rnd_flg = num_subsamp < dist_samp.shape[0]
    wts = np.ones(dist_samp.shape[0])
    n0 = float(n * p)
    bd = np.column_stack([dist_samp.min(axis=0), dist_samp.max(axis=0)])

    # initialize design
    idx = np.random.choice(dist_samp.shape[0], size=n, replace=False)
    ini = dist_samp[idx, :] + np.random.normal(0, 1e-8, (n, p))
    for j in range(p):
        ini[:, j] = np.clip(ini[:, j], bd[j, 0], bd[j, 1])

    return splitpy.sp_cpp(ini, dist_samp, True, bd, \
        num_subsamp, iter_max, tol, n_threads, n0, wts, rnd_flg)


def data_format(data):
    if data.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    cols = []
    for col in data.columns:
        if pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == "object":
            enc = OneHotEncoder(drop="first", sparse=False)
            trans = enc.fit_transform(data[[col]])
            cols.append(trans)
        elif np.issubdtype(data[col].dtype, np.number):
            if (data[col] == data[col].iloc[0]).all():
                continue
            cols.append(data[[col]].to_numpy())
        else:
            raise ValueError("Unsupported column type.")

    D = np.hstack(cols)
    return StandardScaler().fit_transform(D)


def SPlit(data, split_ratio=0.2, kappa=None, max_iterations=500,
          tolerance=1e-10, n_threads=1):
    data_ = data_format(data)
    n = round(min(split_ratio, 1 - split_ratio) * data_.shape[0])

    if kappa is None:
        kappa = data_.shape[0]
    else:
        kappa = min(data_.shape[0], int(np.ceil(kappa * n)))

    sp_ = compute_sp(n, data_.shape[1], data_, kappa,
                     iter_max=max_iterations, tol=tolerance,
                     n_threads=n_threads)

    return splitpy.subsample(data_, sp_)
