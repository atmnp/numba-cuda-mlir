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


@pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 4)])
def test_nested_array_metadata_and_transpose(shape):
    index = tuple(min(axis, size - 1) for axis, size in enumerate(shape))
    transposed_index = index[::-1]

    @cuda.jit
    def kernel(a, out):
        field = a[0].values
        view = field.T
        out[0] = field.itemsize
        out[1] = field.nbytes
        out[2] = view.nbytes
        out[3] = view[transposed_index]
        view[transposed_index] = 73

    array = np.zeros(1, dtype=[("values", np.int16, shape)])
    array["values"][0] = np.arange(np.prod(shape)).reshape(shape)
    device = cuda.to_device(array)
    out = np.zeros(4, dtype=np.int64)
    kernel[1, 1](device, out)
    field = array["values"][0]
    np.testing.assert_array_equal(out, [field.itemsize, field.nbytes, field.nbytes, field[index]])
    assert device.copy_to_host()["values"][0][index] == 73


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.float16, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "sliced", "reversed", "transposed"])
def test_array_byte_strides(dtype, layout):
    transposed = layout == "transposed"

    @cuda.jit
    def kernel(a, out):
        if transposed:
            strides = a.T.strides
        else:
            strides = a.strides
        for i in range(len(strides)):
            out[i] = strides[i]

    array = np.zeros((4, 5, 6), dtype=dtype, order="F" if layout == "F" else "C")
    device = cuda.to_device(array)
    if layout == "sliced":
        device = device[1::2, ::2, 1::3]
    elif layout == "reversed":
        device = device[::-1, ::2, ::-1]
    out = np.zeros(device.ndim, dtype=np.int64)
    kernel[1, 1](device, out)
    expected = device.strides[::-1] if transposed else device.strides
    np.testing.assert_array_equal(out, expected)


def test_scalar_array_empty_strides():
    @cuda.jit
    def kernel(a, out):
        out[0] = len(a.strides)

    out = np.full(1, -1, dtype=np.int64)
    kernel[1, 1](cuda.to_device(np.array(3, dtype=np.int32)), out)
    assert out[0] == 0


@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.dtype([("x", np.int16), ("y", np.float64)], align=True), "S3", "U3"],
)
@pytest.mark.parametrize("empty", [False, True])
def test_explicit_byte_stride_descriptor(dtype, empty):
    device = cuda.to_device(np.zeros((2, 0 if empty else 3), dtype=dtype))

    class ExplicitStrides(type(device)):
        # CAI permits contiguous arrays to omit strides. Supply them here to
        # exercise descriptor byte units independently of that defaulting path.
        @property
        def __cuda_array_interface__(self):
            return {**super().__cuda_array_interface__, "strides": self.strides}

    @cuda.jit
    def kernel(a, out):
        out[0] = a.strides[0]
        out[1] = a.strides[1]

    out = np.zeros(2, dtype=np.int64)
    explicit = ExplicitStrides(device.shape, device.strides, device.dtype, gpu_data=device.gpu_data)
    kernel[1, 1](explicit, out)
    np.testing.assert_array_equal(out, device.strides)


def test_strides_of_kernel_views():
    @cuda.jit
    def kernel(a, out):
        view = a[::2, ::3].T
        out[0] = view.strides[0]
        out[1] = view.strides[1]

    array = np.zeros((6, 9), dtype=np.float64)
    out = np.zeros(2, dtype=np.int64)
    kernel[1, 1](cuda.to_device(array), out)
    np.testing.assert_array_equal(out, array[::2, ::3].T.strides)


@pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 4)])
def test_nested_array_byte_strides(shape):
    @cuda.jit
    def kernel(a, out):
        field = a[0].values
        strides = field.strides
        transposed_strides = field.T.strides
        for i in range(len(strides)):
            out[i] = strides[i]
            out[i + len(strides)] = transposed_strides[i]

    array = np.zeros(1, dtype=[("values", np.int16, shape)])
    out = np.zeros(len(shape) * 2, dtype=np.int64)
    kernel[1, 1](cuda.to_device(array), out)
    field = array["values"][0]
    np.testing.assert_array_equal(out, [*field.strides, *field.T.strides])


@pytest.mark.parametrize("space", [cuda.local, cuda.shared])
def test_allocated_array_byte_strides(space):
    @cuda.jit
    def kernel(out):
        array = space.array((2, 3), np.int32)
        out[0] = array.strides[0]
        out[1] = array.strides[1]

    out = np.zeros(2, dtype=np.int64)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, [12, 4])


def test_dynamic_shared_array_byte_stride():
    @cuda.jit
    def kernel(out):
        array = cuda.shared.array(0, np.int32)
        out[0] = array.strides[0]

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1, 0, 32](out)
    assert out[0] == 4
