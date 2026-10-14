# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from numba_cuda_mlir import types
from numba_cuda_mlir.experimental import host
from numba_cuda_mlir.numba_cuda.core.errors import TypingError


@host.jit
def square(x):
    return x * x


def test_default_global_and_module_calls_in_loop():
    module = sys.modules[__name__]

    @host.jit
    def caller(n):
        total = 0
        for i in range(n):
            total += square(i) + module.square(i)
        return total

    assert caller(0) == 0
    assert caller(5) == 60
    assert square(3) == 9


@pytest.mark.parametrize(
    "policy",
    ["never", "always", lambda *args: True, lambda *args: False],
    ids=["native", "inline", "policy-inline", "policy-native"],
)
def test_nested_specializations_and_return_types(policy):
    @host.jit(inline=policy)
    def twice(x):
        if x < 0:
            return -x
        return x + x

    @host.jit
    def outer(x):
        return twice(twice(x))

    for value, expected, arg_type, return_type in (
        (np.int32(-3), 6, types.int32, types.int64),
        (np.float32(1.5), 6.0, types.float32, types.float32),
        (np.float64(2.5), 10.0, types.float64, types.float64),
    ):
        assert outer(value) == expected
        assert outer.overloads[(arg_type,)].signature.return_type == return_type


@pytest.mark.parametrize("policy", ["never", "always"])
@pytest.mark.parametrize("failure", ["division", "range", "overflow", "none"])
def test_nested_errors_stop_caller_and_recover(policy, failure):
    @host.jit(inline=policy)
    def leaf(x):
        if failure == "range":
            total = 0
            for i in range(0, 6, x):
                total += i
            return total
        if failure == "overflow":
            return x // -1
        if failure == "none":
            value = 12 // x
            return
        return 12 // x

    @host.jit
    def middle(x):
        return leaf(x)

    @host.jit
    def caller(x):
        middle(x)
        # A different error after the helper exposes a missing checked exit.
        if x == 0 or x == -(2**63):
            for i in range(0, 1, x - x):
                pass
        return 42

    error = (
        ValueError
        if failure == "range"
        else OverflowError
        if failure == "overflow"
        else ZeroDivisionError
    )
    with pytest.raises(error):
        caller(-(2**63) if failure == "overflow" else 0)

    assert caller(2) == 42


@pytest.mark.parametrize("policy", ["never", "always"])
def test_explicit_and_annotated_helper_conversions(policy):
    @host.jit(inline=policy)
    def annotated(x: types.int32):
        return x + 1

    explicit = host.jit(types.float32(types.float32), inline=policy)(lambda x: x * 2)

    @host.jit(inline=policy)
    def narrow_result(x: types.int64) -> types.int32:
        return x

    @host.jit
    def caller(x, y):
        return annotated(x) + explicit(y) + narrow_result(x)

    assert caller(2**32 + 3, np.float32(1.5)) == 10
    with pytest.raises((TypingError, TypeError)):
        caller(3, np.float64(1.5))
    assert caller(2**32 + 3, np.float32(1.5)) == 10


@pytest.mark.parametrize("mutual", [False, True])
def test_recursion_rejected_without_poisoning_other_specializations(mutual):
    @host.jit
    def first(x):
        return second(x - 1) if x else 0

    if mutual:

        @host.jit
        def second(x):
            return first(x - 1) if x else 0
    else:
        second = first

    with pytest.raises(TypingError, match="recursive"):
        first(2)
    assert square(4) == 16


def test_same_named_helpers_and_shared_transitive_dependency():
    def make(offset):
        @host.jit
        def helper(x):
            return square(x) + offset

        return helper

    first, second = make(1), make(2)

    @host.jit
    def caller(x):
        return first(x) + second(x) + first(x)

    assert caller(3) == 31


@pytest.mark.parametrize("kind", ["keyword", "arity", "non-scalar", "return"])
def test_invalid_helper_boundaries(kind):
    helper = host.jit(lambda x: (x, x) if kind == "return" else x)
    if kind == "keyword":

        def caller(x):
            return helper(x=x)
    elif kind == "arity":

        def caller(x):
            return helper(x, x)
    elif kind == "non-scalar":

        def caller(x):
            return helper((x, x))
    else:

        def caller(x):
            return helper(x)

    with pytest.raises((TypingError, TypeError)):
        host.jit(caller)(1)


@pytest.mark.parametrize("inline", ["sometimes", True])
def test_invalid_inline_policy(inline):
    with pytest.raises(ValueError):
        host.jit(inline=inline)(lambda x: x)


def test_native_helper_errors_are_thread_local():
    @host.jit
    def helper(x):
        return 12 // x

    @host.jit
    def caller(x):
        return helper(x) + 1

    assert caller(2) == 7

    def run(x):
        try:
            return caller(x)
        except ZeroDivisionError:
            return "zero"

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(run, [0, 2, 3, 0, 4, 2])) == ["zero", 7, 5, "zero", 4, 7]
