# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int16,
        np.int32,
        np.float16,
        np.complex64,
        np.dtype([("x", np.int16), ("y", np.float64)], align=True),
        np.dtype("S3"),
        np.dtype("U3"),
    ],
)
@pytest.mark.parametrize("layout", ["C", "F", "sliced", "empty", "scalar"])
def test_array_storage_metadata(dtype, layout):
    @cuda.jit
    def kernel(a, out):
        out[0] = a.itemsize
        out[1] = a.nbytes

    if layout == "scalar":
        array = np.zeros((), dtype=dtype)
    elif layout == "empty":
        array = np.zeros((2, 0), dtype=dtype)
    else:
        array = np.zeros((4, 6), dtype=dtype, order="F" if layout == "F" else "C")
    device = cuda.to_device(array)
    if layout == "sliced":
        array = array[::2, 1::2]
        device = device[::2, 1::2]
    out = np.zeros(2, dtype=np.int64)
    kernel[1, 1](device, out)
    np.testing.assert_array_equal(out, [array.itemsize, array.nbytes])


@pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("layout", ["C", "F", "sliced"])
def test_array_transpose_is_view(shape, layout):
    array = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    if layout == "F":
        array = np.asfortranarray(array)
    device = cuda.to_device(array)
    if layout == "sliced":
        slices = (slice(None, None, 2),) * len(shape)
        array = array[slices]
        device = device[slices]
    index = tuple(min(axis, size - 1) for axis, size in enumerate(array.shape))
    transposed_index = index[::-1]

    @cuda.jit
    def kernel(a, dimensions, value):
        view = a.T
        for i in range(len(view.shape)):
            dimensions[i] = view.shape[i]
        value[0] = view[transposed_index]
        view[transposed_index] = 73

    dimensions = np.zeros(array.ndim, dtype=np.int64)
    value = np.zeros(1, dtype=np.int32)
    expected_value = array.T[transposed_index]
    kernel[1, 1](device, dimensions, value)
    np.testing.assert_array_equal(dimensions, array.T.shape)
    assert value[0] == expected_value
    assert device.copy_to_host()[index] == 73


def test_scalar_array_transpose_is_view():
    @cuda.jit
    def kernel(a, out):
        view = a.T
        out[0] = view[()]
        view[()] = 73

    device = cuda.to_device(np.array(9, dtype=np.int32))
    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](device, out)
    assert out[0] == 9
    assert device.copy_to_host()[()] == 73


@pytest.mark.parametrize("space", [cuda.local, cuda.shared])
def test_allocated_array_transpose_is_view(space):
    @cuda.jit
    def kernel(out):
        array = space.array((2, 3), np.int32)
        array[1, 2] = 17
        view = array.T
        out[0] = view[2, 1]
        view[2, 1] = 29
        out[1] = array[1, 2]

    out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, [17, 29])
