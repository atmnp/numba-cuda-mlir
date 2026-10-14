# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from functools import cached_property, update_wrapper

from numba_cuda_mlir.descriptor import MLIRTargetContext, MLIRTypingContext
from numba_cuda_mlir.experimental.host.execution import scalar_ctype
from numba_cuda_mlir.numba_cuda import types, typing
from numba_cuda_mlir.numba_cuda.core import sigutils
from numba_cuda_mlir.numba_cuda.core.compiler_lock import global_compiler_lock
from numba_cuda_mlir.numba_cuda.core.target_extension import CPU, target_registry
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.typing.typeof import Purpose, typeof


class TargetDescriptor:
    @cached_property
    def typing_context(self):
        return MLIRTypingContext()

    @cached_property
    def target_context(self):
        return MLIRTargetContext(self.typing_context, target="numba_cuda_mlir_host")


class Target(CPU):
    pass


target_registry["numba_cuda_mlir_host"] = Target
target = TargetDescriptor()


class Dispatcher:
    targetdescr = target

    def __init__(self, py_func, targetoptions):
        self.py_func = py_func
        self.targetoptions = targetoptions
        self.overloads = {}
        self._compile_only_overloads = {}
        self._compiling = False
        self._symbol_name = f"host_{id(self):x}"
        self._can_compile = True
        self._signature = inspect.signature(py_func)
        self._arg_annotations = tuple(
            parameter.annotation for parameter in self._signature.parameters.values()
        )
        for parameter in self._signature.parameters.values():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise TypeError("host.jit supports only fixed positional parameters")
        for annotation in self._arg_annotations:
            if annotation is not inspect.Signature.empty:
                if not isinstance(annotation, types.Type):
                    raise TypeError("host.jit annotations must be Numba scalar types")
                scalar_ctype(annotation)
        update_wrapper(self, py_func)

    @property
    def _numba_type_(self):
        return types.Dispatcher(self)

    @property
    def is_compiling(self):
        # Recursion is rejected by compile_only rather than Numba's device recursion path.
        return False

    @property
    def nopython_signatures(self):
        return [result.signature for result in self._compile_only_overloads.values()]

    @global_compiler_lock
    def compile_only(self, sig):
        from numba_cuda_mlir.experimental.host.compiler import compile_for_link

        argtypes, return_type = sigutils.normalize_signature(sig)
        argtypes = tuple(types.unliteral(typ) for typ in argtypes)
        if len(argtypes) != len(self._arg_annotations):
            raise TypeError(
                f"expected {len(self._arg_annotations)} helper arguments, got {len(argtypes)}"
            )
        for typ in argtypes:
            scalar_ctype(typ)
        if argtypes in self._compile_only_overloads:
            result = self._compile_only_overloads[argtypes]
            if return_type is not None and return_type != result.signature.return_type:
                raise TypeError("host.jit specialization already has a different return type")
            return result
        if self._compiling:
            raise TypingError("recursive host.jit helpers are not supported")
        if not self._can_compile:
            raise TypeError(f"no matching host.jit signature for {argtypes}")
        if return_type is None:
            annotation = self._signature.return_annotation
            if annotation is not inspect.Signature.empty:
                return_type = types.none if annotation is None else annotation
        self._compiling = True
        try:
            result = compile_for_link(
                self.py_func, return_type, argtypes, self.targetoptions, self._symbol_name
            )
        finally:
            self._compiling = False
        self._compile_only_overloads[argtypes] = result
        return result

    def get_call_template(self, args, kws):
        if kws:
            raise TypeError("host.jit helpers do not support keyword arguments")
        if len(args) != len(self._arg_annotations):
            raise TypeError(
                f"expected {len(self._arg_annotations)} helper arguments, got {len(args)}"
            )
        argtypes = tuple(
            annotation if isinstance(annotation, types.Type) else types.unliteral(actual)
            for annotation, actual in zip(self._arg_annotations, args, strict=True)
        )
        for typ in args:
            scalar_ctype(typ)
        if self._can_compile:
            result = self.compile_only(argtypes)
        elif argtypes in self._compile_only_overloads:
            result = self._compile_only_overloads[argtypes]
        else:
            raise TypeError(f"no matching host.jit signature for {argtypes}")
        template = typing.make_concrete_template(
            f"CallTemplate({self.py_func.__name__})", self.py_func.__name__, [result.signature]
        )
        return template, self._signature, args, {}

    def get_overload(self, sig):
        return self.compile_only(sig)

    @global_compiler_lock
    def compile(self, sig):
        from numba_cuda_mlir.experimental.host.compiler import execute

        argtypes, return_type = sigutils.normalize_signature(sig)
        argtypes = tuple(argtypes)
        if len(argtypes) != len(self._arg_annotations):
            raise TypeError(
                f"expected {len(self._arg_annotations)} argument types, got {len(argtypes)}"
            )
        if argtypes in self.overloads:
            existing = self.overloads[argtypes]
            if return_type is not None and existing.signature.return_type != return_type:
                raise TypeError(
                    f"host.jit signature {argtypes} already compiled with return type "
                    f"{existing.signature.return_type}, not {return_type}"
                )
            return existing
        if not self._can_compile and argtypes not in self._compile_only_overloads:
            raise TypeError(f"no matching host.jit signature for {argtypes}")
        result = execute(self.compile_only(sig), self.targetoptions)
        self.overloads[argtypes] = result
        return result

    def disable_compile(self):
        self._can_compile = False

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError("host.jit does not support keyword arguments")
        if len(args) != len(self._arg_annotations):
            raise TypeError(f"expected {len(self._arg_annotations)} arguments, got {len(args)}")
        # Validate actual scalars before annotations select the specialization;
        # execution owns their subsequent Python-to-native slot conversion.
        actual_types = tuple(typeof(value, Purpose.argument) for value in args)
        for typ in actual_types:
            scalar_ctype(typ)
        argtypes = tuple(
            annotation if isinstance(annotation, types.Type) else actual_type
            for annotation, actual_type in zip(self._arg_annotations, actual_types, strict=True)
        )
        result = self.compile(argtypes)
        return result.entry_point(*args)

    @property
    def signatures(self):
        return list(self.overloads)

    def inspect_mlir(self, signature=None):
        if signature is None:
            return {sig: result.mlir for sig, result in self.overloads.items()}
        argtypes, _ = sigutils.normalize_signature(signature)
        return self.overloads[tuple(argtypes)].mlir

    def inspect_llvm_dialect(self, signature=None):
        if signature is None:
            return {sig: result.llvm_mlir for sig, result in self.overloads.items()}
        argtypes, _ = sigutils.normalize_signature(signature)
        return self.overloads[tuple(argtypes)].llvm_mlir
