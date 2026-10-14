# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose scalar CPU functions through native calls and selected inlining."""

from numba_cuda_mlir.experimental import host


@host.jit
def square(x):
    return x * x


@host.jit(inline="always")
def add_one(x):
    return x + 1


@host.jit
def sum_squares(n):
    total = 0
    for i in range(n):
        total += square(add_one(i))
    return total


if __name__ == "__main__":
    result = sum_squares(5)
    assert result == 55
    print(f"sum_squares(5) = {result}")
