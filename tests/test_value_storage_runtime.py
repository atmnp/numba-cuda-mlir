# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir import cuda


def test_float16_array_constructors_use_storage_values():
    @cuda.jit
    def kernel(out):
        zeros = np.zeros(4, dtype=np.float16)
        ones = np.ones(4, dtype=np.float16)
        full = np.full(4, 2.5, dtype=np.float16)
        out[0] = zeros[0]
        out[1] = ones[1]
        out[2] = full[2]

    out = np.zeros(3, dtype=np.float16)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.5], dtype=np.float16))


def test_bool_array_constructors_use_storage_values():
    @cuda.jit
    def kernel(out):
        zeros = np.zeros(4, dtype=np.bool_)
        ones = np.ones(4, dtype=np.bool_)
        out[0] = 1 if zeros[0] else 0
        out[1] = 1 if ones[1] else 0

    out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, np.array([0, 1], dtype=np.int32))


def test_float16_reductions_read_value_elements():
    @cuda.jit
    def kernel(sum_out, flags_out, values):
        sum_out[0] = np.sum(values)
        flags_out[0] = 1 if np.any(values) else 0
        flags_out[1] = 1 if np.all(values) else 0

    values = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float16)
    sum_out = np.zeros(1, dtype=np.float16)
    flags_out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](sum_out, flags_out, values)
    np.testing.assert_allclose(sum_out, np.array([6.0], dtype=np.float16))
    np.testing.assert_array_equal(flags_out, np.array([1, 0], dtype=np.int32))


def test_bool_reductions_read_value_elements():
    @cuda.jit
    def kernel(flags_out, values):
        flags_out[0] = 1 if np.any(values) else 0
        flags_out[1] = 1 if np.all(values) else 0

    values = np.array([True, False, True], dtype=np.bool_)
    flags_out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](flags_out, values)
    np.testing.assert_array_equal(flags_out, np.array([1, 0], dtype=np.int32))
