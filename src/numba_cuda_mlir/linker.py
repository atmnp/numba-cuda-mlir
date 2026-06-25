# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import os
import warnings
from numba_cuda_mlir.tools import get_gpu_compute_capability, format_arch
from numba_cuda_mlir.numba_cuda.cudadrv.driver import _Linker, cached_file_read


def _link_item_has_callbacks(link_item) -> bool:
    return bool(getattr(link_item, "setup_callback", None)) or bool(
        getattr(link_item, "teardown_callback", None)
    )


def _link_item_is_ltoir(link_item) -> bool:
    if isinstance(link_item, str):
        return link_item.endswith(".ltoir")
    return type(link_item).__name__ == "LTOIR"


def _link_item_is_cuda_source(link_item) -> bool:
    if isinstance(link_item, str):
        return link_item.endswith(".cu")
    return getattr(link_item, "kind", None) == "cu" or type(link_item).__name__ == "CUSource"


def _link_item_display_name(link_item) -> str:
    if isinstance(link_item, str):
        return link_item
    return getattr(link_item, "name", str(link_item))


@dataclass(frozen=True)
class ResolvedLinkPlan:
    """Resolved JIT link policy.

    ``compile_new_inputs_as_ltoir`` controls the current MLIR module. CUDA
    source link items can use a separate mode because debug NVRTC LTOIR cannot
    always be combined with MLIR-generated debug LTOIR. Existing LTOIR link
    inputs still require an LTO-enabled linker even when CUDA source inputs stay
    as PTX.
    """

    compile_new_inputs_as_ltoir: bool
    compile_cuda_sources_as_ltoir: bool
    linker_uses_lto: bool
    lto_explicit: bool
    requested_lto: bool
    compile_output: str
    has_ltoir_link_items: bool
    has_callback_link_items: bool


def resolve_link_plan(targetoptions: dict, link_items=(), extra_ltoir_items=()) -> ResolvedLinkPlan:
    link_items = tuple(link_items or ())
    extra_ltoir_items = tuple(extra_ltoir_items or ())
    lto_explicit = bool(targetoptions.get("_lto_explicit", "lto" in targetoptions))
    # ``_lto_explicit=False`` marks ``lto`` as derived state from earlier link
    # planning, not a fresh request. Callers that need to force LTO must omit
    # the marker or set it to True.
    requested_lto = bool(targetoptions.get("lto", False)) if lto_explicit else False
    compile_output = targetoptions.get("_compile_output", "ptx")
    has_ltoir_link_items = bool(extra_ltoir_items) or any(
        _link_item_is_ltoir(link_item) for link_item in link_items
    )
    has_cuda_source_link_items = any(
        _link_item_is_cuda_source(link_item) for link_item in link_items
    )
    has_callback_link_items = any(_link_item_has_callbacks(link_item) for link_item in link_items)

    if compile_output not in ("ptx", "ltoir"):
        raise ValueError(f"Unsupported compile output: {compile_output}")

    if compile_output == "ltoir" and lto_explicit and not requested_lto:
        raise ValueError("Cannot produce LTOIR output with lto=False")
    if has_ltoir_link_items and lto_explicit and not requested_lto:
        raise ValueError("Cannot link LTOIR inputs with lto=False")
    if targetoptions.get("debug", False) and lto_explicit and requested_lto:
        if compile_output != "ltoir":
            raise ValueError(
                "debug=True with lto=True is not supported for JIT linking; "
                "disable debug or LTO, or request output='ltoir' for an artifact."
            )

    if compile_output == "ltoir":
        compile_new_inputs_as_ltoir = True
    elif targetoptions.get("debug", False):
        compile_new_inputs_as_ltoir = False
    elif lto_explicit:
        compile_new_inputs_as_ltoir = requested_lto
    elif has_callback_link_items:
        compile_new_inputs_as_ltoir = False
    else:
        compile_new_inputs_as_ltoir = has_ltoir_link_items

    compile_cuda_sources_as_ltoir = compile_new_inputs_as_ltoir
    if targetoptions.get("debug", False) and has_cuda_source_link_items:
        compile_cuda_sources_as_ltoir = False

    linker_uses_lto = compile_new_inputs_as_ltoir or has_ltoir_link_items
    if linker_uses_lto:
        from numba_cuda_mlir.numba_cuda.cudadrv.driver import _have_nvjitlink

        if not _have_nvjitlink():
            raise ValueError(
                "LTO requires nvjitlink, which is not available or not sufficiently recent (>=12.3)"
            )

    return ResolvedLinkPlan(
        compile_new_inputs_as_ltoir=compile_new_inputs_as_ltoir,
        compile_cuda_sources_as_ltoir=compile_cuda_sources_as_ltoir,
        linker_uses_lto=linker_uses_lto,
        lto_explicit=lto_explicit,
        requested_lto=requested_lto,
        compile_output=compile_output,
        has_ltoir_link_items=has_ltoir_link_items,
        has_callback_link_items=has_callback_link_items,
    )


def _link_item_is_ltoir(link_item) -> bool:
    if isinstance(link_item, str):
        return link_item.endswith(".ltoir")
    return getattr(link_item, "kind", None) == "ltoir" or type(link_item).__name__ == "LTOIR"


def _link_item_is_cuda_source(link_item) -> bool:
    if isinstance(link_item, str):
        return link_item.endswith(".cu")
    return getattr(link_item, "kind", None) == "cu" or type(link_item).__name__ == "CUSource"


class Linker(_Linker):
    def __init__(
        self,
        cc: tuple[int, int] | None = None,
        additional_flags: list[str] | None = None,
        lto: bool | None = None,
        arch: str | None = None,
        verbose: bool = False,
        ftz: bool | None = None,
        prec_div: bool | None = None,
        prec_sqrt: bool | None = None,
        fma: bool | None = None,
        debug: bool = False,
        lineinfo: bool = False,
        optimize_unused_variables: bool = True,
        kernels_used: str | tuple[str, ...] | list[str] | None = None,
        variables_used: str | tuple[str, ...] | list[str] | None = None,
        optimization_level: int = 3,
        ptxas_options: str | None = None,
        max_registers: int | None = None,
    ):
        if cc is None:
            cc = get_gpu_compute_capability(tuple)
        if arch is None:
            arch = format_arch(cc)
        super().__init__(
            cc=cc,
            additional_flags=additional_flags,
            lineinfo=lineinfo,
            lto=lto,
            max_registers=max_registers,
            arch=arch,
            verbose=verbose,
            debug=debug,
            ftz=ftz,
            prec_div=prec_div,
            prec_sqrt=prec_sqrt,
            fma=fma,
            optimize_unused_variables=optimize_unused_variables,
            kernels_used=kernels_used,
            variables_used=variables_used,
            optimization_level=optimization_level,
            ptxas_options=ptxas_options,
        )

        self._numba_cuda_mlir_temp_ptx_files: list[str] = []
        self._ltoirs: dict[bytes, str] = {}

    def add_file_guess_ext(
        self, path_or_code, ignore_nonlto=False, compile_cu_as_ltoir: bool | None = None
    ):
        if compile_cu_as_ltoir is None or not _link_item_is_cuda_source(path_or_code):
            return super().add_file_guess_ext(path_or_code, ignore_nonlto=ignore_nonlto)
        if ignore_nonlto and not compile_cu_as_ltoir:
            name = _link_item_display_name(path_or_code)
            warnings.warn(
                f"Not adding {name} as it is not optimizable at link time, and "
                "`ignore_nonlto == True`."
            )
            return None
        if isinstance(path_or_code, str):
            cu = cached_file_read(path_or_code, how="rb")
            self._pending_cu.append(
                ("data", cu, os.path.basename(path_or_code), compile_cu_as_ltoir)
            )
            return None
        self._pending_cu.append(
            ("linkable", path_or_code, _link_item_display_name(path_or_code), compile_cu_as_ltoir)
        )
        return None

    def recreate_with_lto(self, lto: bool = True, ltoir_only: bool = False) -> "Linker":
        """Recreate the linker, re-adding all object codes from raw bytes.

        When *ltoir_only* is True, only LTOIR objects are copied (useful for
        diagnostic ``-ptx`` links that require all inputs to be LTOIR).
        """
        self._materialize_pending_cu()

        existing = list(getattr(self, "_object_codes", []))
        new_linker = Linker(
            cc=self.cc,
            additional_flags=self.additional_flags,
            lto=lto,
            arch=self.arch,
            verbose=self._verbose,
            ftz=self._ftz,
            prec_div=self._prec_div,
            prec_sqrt=self._prec_sqrt,
            fma=self._fma,
            debug=self._debug,
            lineinfo=self.lineinfo,
            optimize_unused_variables=self._optimize_unused_variables,
            kernels_used=self.kernels_used,
            variables_used=self.variables_used,
            optimization_level=self._optimization_level,
            ptxas_options=self._ptxas_options,
            max_registers=self.max_registers,
        )
        for obj in existing:
            code_type = getattr(obj, "code_type", None)
            if ltoir_only and code_type != "ltoir":
                continue
            if code_type == "ptx":
                new_linker.add_ptx(obj.code)
            elif code_type == "cubin":
                new_linker.add_cubin(obj.code)
            elif code_type == "ltoir":
                new_linker.add_ltoir(obj.code)
            else:
                new_linker._object_codes.append(obj)
        new_linker._ltoirs = dict(self._ltoirs)
        return new_linker

    def add_ltoir(self, ltoir: bytes, name: str = "") -> None:
        if ltoir in self._ltoirs:
            return
        if not self.lto:
            new_linker = self.recreate_with_lto()
            self.__dict__.update(new_linker.__dict__)
        super().add_ltoir(ltoir, name)
        self._ltoirs[ltoir] = name

    def _get_linker_options(self, ptx):
        return super()._get_linker_options(ptx)
