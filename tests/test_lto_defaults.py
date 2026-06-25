# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIR JIT LTO option defaults."""

import builtins
import io
import os
import types

import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir import caching
from numba_cuda_mlir import tools
from numba_cuda_mlir.caching import _target_option_key
from numba_cuda_mlir.linker import Linker, resolve_link_plan
from numba_cuda_mlir.mlir_lowering import MLIRLower
from numba_cuda_mlir.numba_cuda import types as nb_types


def _kernel():
    pass


def _device_increment(x):
    return x + 1


def _set_nvjitlink_available(monkeypatch, available):
    from numba_cuda_mlir.numba_cuda.cudadrv import driver

    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: available)


def _install_fake_dispatcher_compile(monkeypatch, on_compile=None):
    from numba_cuda_mlir import mlir_compiler, mlir_optimization

    captured_targetoptions = []

    def compile_mlir(py_func, return_type, argtypes, targetoptions):
        if on_compile is not None:
            on_compile()
        captured_targetoptions.append(targetoptions)
        metadata = {"targetoptions": targetoptions}
        if targetoptions.get("_compile_output") == "ltoir":
            metadata["ltoir"] = b"ltoir"
        elif targetoptions.get("_compile_output") == "ptx":
            metadata["ptx"] = "ptx"
        return types.SimpleNamespace(
            metadata=metadata,
            signature=types.SimpleNamespace(args=argtypes, return_type=return_type),
            target_context=types.SimpleNamespace(insert_user_function=lambda *args: None),
            entry_point=object(),
            fndesc=object(),
            library=object(),
        )

    monkeypatch.setattr(mlir_compiler, "compile_mlir", compile_mlir)
    monkeypatch.setattr(mlir_optimization, "optimize", lambda cres: None)
    return captured_targetoptions


class _RecordingLock:
    def __init__(self):
        self.depth = 0
        self.max_depth = 0

    def __enter__(self):
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1
        return False


class _LoadingCache:
    def __init__(self, cached_result):
        self.cached_result = cached_result
        self.load_count = 0

    def load_overload(self, sig, target_context):
        self.load_count += 1
        return self.cached_result

    def save_overload(self, sig, result):
        pass

    def flush(self):
        pass


def test_mlir_jit_rejects_output_option():
    with pytest.raises(ValueError, match="Got invalid options: output"):
        cuda.jit(output="ltoir")(_kernel)


def test_mlir_jit_defaults_to_ptx_linking():
    dispatcher = cuda.jit(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_link_plan_keeps_implicit_ptx_for_external_ptx_link_items():
    dispatcher = cuda.jit(link=["external.ptx"])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.linker_uses_lto is False
    assert plan.lto_explicit is False


def test_link_plan_keeps_implicit_ptx_for_cu_source_link_items():
    dispatcher = cuda.jit(link=[cuda.CUSource("")])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.linker_uses_lto is False
    assert plan.lto_explicit is False


def test_link_plan_enables_implicit_lto_for_ltoir_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    dispatcher = cuda.jit(link=["external.ltoir"])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.linker_uses_lto is True
    assert plan.lto_explicit is False
    assert plan.has_ltoir_link_items is True


def test_link_plan_rejects_lto_when_nvjitlink_is_unavailable(monkeypatch):
    _set_nvjitlink_available(monkeypatch, False)
    dispatcher = cuda.jit(link=["external.ltoir"])(_kernel)

    with pytest.raises(ValueError, match="LTO requires nvjitlink"):
        resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])


def test_link_plan_preserves_explicit_lto_false_without_ltoir():
    dispatcher = cuda.jit(link=["external.ptx"], lto=False)(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.linker_uses_lto is False
    assert plan.lto_explicit is True
    assert plan.requested_lto is False


def test_link_plan_rejects_explicit_lto_false_with_ltoir(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    dispatcher = cuda.jit(link=["external.ltoir"], lto=False)(_kernel)

    with pytest.raises(ValueError, match="Cannot link LTOIR inputs with lto=False"):
        resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])


def test_link_plan_rejects_ltoir_output_with_explicit_lto_false():
    targetoptions = {"_compile_output": "ltoir", "lto": False, "_lto_explicit": True}

    with pytest.raises(ValueError, match="Cannot produce LTOIR output with lto=False"):
        resolve_link_plan(targetoptions)


def test_link_plan_treats_lto_key_as_explicit_without_marker():
    targetoptions = {"_compile_output": "ltoir", "lto": False}

    with pytest.raises(ValueError, match="Cannot produce LTOIR output with lto=False"):
        resolve_link_plan(targetoptions)


def test_link_plan_allows_debug_ptx_with_implicit_ltoir_input(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    dispatcher = cuda.jit(link=["external.ltoir"], debug=True, opt=False)(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.linker_uses_lto is True
    assert plan.has_ltoir_link_items is True


def test_link_plan_compile_output_ltoir_uses_lto(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    targetoptions = {
        "lto": True,
        "_lto_explicit": False,
        "_compile_output": "ltoir",
        "debug": False,
    }

    plan = resolve_link_plan(targetoptions)

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.linker_uses_lto is True
    assert plan.compile_output == "ltoir"


def test_link_plan_rejects_debug_with_explicit_lto(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    targetoptions = {
        "lto": True,
        "_lto_explicit": True,
        "debug": True,
    }

    with pytest.raises(ValueError, match="debug=True with lto=True"):
        resolve_link_plan(targetoptions, [cuda.CUSource("")])


def test_link_plan_ltoir_output_keeps_debug_cuda_sources_on_ptx(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    targetoptions = {
        "_compile_output": "ltoir",
        "debug": True,
        "_lto_explicit": False,
    }

    plan = resolve_link_plan(targetoptions, [cuda.CUSource("")])

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.compile_cuda_sources_as_ltoir is False
    assert plan.linker_uses_lto is True


def test_link_plan_ignores_legacy_output_option():
    targetoptions = {
        "lto": True,
        "_lto_explicit": False,
        "output": "ltoir",
        "debug": False,
    }

    plan = resolve_link_plan(targetoptions)

    assert plan.compile_output == "ptx"
    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.linker_uses_lto is False


def test_link_plan_ignores_implicit_lto_true_for_callback_link_items():
    targetoptions = {
        "lto": True,
        "_lto_explicit": False,
        "debug": False,
    }
    link_items = [cuda.CUSource("", setup_callback=lambda *args: None)]

    plan = resolve_link_plan(targetoptions, link_items)

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.linker_uses_lto is False
    assert plan.requested_lto is False


def test_link_plan_ignores_nonexplicit_lto_true_without_lto_inputs():
    targetoptions = {
        "lto": True,
        "_lto_explicit": False,
        "debug": False,
    }

    plan = resolve_link_plan(targetoptions)

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.linker_uses_lto is False
    assert plan.requested_lto is False


def test_link_plan_derived_lto_true_without_lto_inputs_ignores_nvjitlink(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, False)
    targetoptions = {
        "lto": True,
        "_lto_explicit": False,
        "debug": False,
    }

    plan = resolve_link_plan(targetoptions)

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.linker_uses_lto is False
    assert plan.requested_lto is False


def test_link_plan_treats_unmarked_lto_as_explicit_for_callback_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    targetoptions = {
        "lto": True,
        "debug": False,
    }
    link_items = [cuda.CUSource("", setup_callback=lambda *args: None)]

    plan = resolve_link_plan(targetoptions, link_items)

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.linker_uses_lto is True
    assert plan.lto_explicit is True


def test_link_plan_lineinfo_preserves_implicit_lto_for_ltoir_inputs(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    dispatcher = cuda.jit(link=["external.ltoir"], lineinfo=True)(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.linker_uses_lto is True


def test_cu_source_uses_resolved_lto_mode_after_lto_enabled():
    cu_source = cuda.CUSource("")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=True)

    linker.add_file_guess_ext(cu_source, compile_cu_as_ltoir=False)

    assert linker.lto is True
    assert linker._pending_cu == [("linkable", cu_source, cu_source.name, False)]


def test_cu_linkable_kind_uses_resolved_lto_mode_after_lto_enabled():
    cu_source = types.SimpleNamespace(kind="cu", name="external.cu")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=True)

    linker.add_file_guess_ext(cu_source, compile_cu_as_ltoir=False)

    assert linker.lto is True
    assert linker._pending_cu == [("linkable", cu_source, "external.cu", False)]


def test_cu_linkable_kind_respects_ignore_nonlto_with_resolved_ptx_mode():
    cu_source = types.SimpleNamespace(kind="cu", name="external.cu")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=True)

    with pytest.warns(UserWarning, match="Not adding external.cu"):
        linker.add_file_guess_ext(cu_source, compile_cu_as_ltoir=False, ignore_nonlto=True)

    assert linker._pending_cu == []


def test_cu_linkable_without_name_warns_with_fallback_display_name():
    cu_source = types.SimpleNamespace(kind="cu")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=True)

    with pytest.warns(UserWarning, match="namespace\\(kind='cu'\\)"):
        linker.add_file_guess_ext(cu_source, compile_cu_as_ltoir=False, ignore_nonlto=True)

    assert linker._pending_cu == []


def test_cu_linkable_without_name_records_fallback_display_name():
    cu_source = types.SimpleNamespace(kind="cu")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=True)

    linker.add_file_guess_ext(cu_source, compile_cu_as_ltoir=False)

    assert linker._pending_cu == [("linkable", cu_source, "namespace(kind='cu')", False)]


def test_cu_path_uses_resolved_lto_mode_after_lto_enabled(tmp_path):
    cu_path = tmp_path / "external.cu"
    cu_path.write_text("// cu")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=True)

    linker.add_file_guess_ext(str(cu_path), compile_cu_as_ltoir=False)

    assert linker.lto is True
    assert linker._pending_cu == [("data", b"// cu", "external.cu", False)]


def test_cu_path_respects_ignore_nonlto_with_resolved_ptx_mode(tmp_path):
    cu_path = tmp_path / "external.cu"
    cu_path.write_text("// cu")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=True)

    with pytest.warns(UserWarning, match=f"Not adding {cu_path}"):
        linker.add_file_guess_ext(str(cu_path), compile_cu_as_ltoir=False, ignore_nonlto=True)

    assert linker._pending_cu == []


def test_non_cu_link_item_ignores_resolved_cu_lto_mode():
    ptx = cuda.PTXSource("// ptx")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=False)

    linker.add_file_guess_ext(ptx, compile_cu_as_ltoir=True)

    assert linker.lto is False
    assert linker._pending_cu == []
    assert len(linker._object_codes) == 1
    assert linker._object_codes[0].code_type == "ptx"


def test_dispatcher_compile_output_bypasses_existing_regular_overload(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    sentinel = object()
    dispatcher.overloads[()] = sentinel

    result = dispatcher.compile(nb_types.void(), output="ltoir")

    assert result is not sentinel
    assert dispatcher.overloads[()] is sentinel
    assert captured_targetoptions[0]["_compile_output"] == "ltoir"
    assert captured_targetoptions[0]["lto"] is True
    assert dispatcher.targetoptions["lto"] is False
    assert "_compile_output" not in dispatcher.targetoptions


def test_dispatcher_compile_ignores_other_thread_temporary_targetoptions_depth():
    dispatcher = cuda.jit(_kernel)
    sentinel = object()
    dispatcher.overloads[()] = sentinel
    dispatcher._temporary_targetoptions_depth = 1

    result = dispatcher.compile(nb_types.void())

    assert result is sentinel


def test_current_thread_temporary_targetoptions_bypass_regular_overload(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    sentinel = object()
    dispatcher.overloads[()] = sentinel

    with dispatcher._temporary_targetoptions({"debug": True}):
        result = dispatcher.compile(nb_types.void())

    assert result is not sentinel
    assert captured_targetoptions[0]["debug"] is True
    assert dispatcher.overloads[()] is sentinel


def test_dispatcher_compile_output_does_not_mutate_after_compile(monkeypatch):
    dispatcher = cuda.jit(_kernel)
    observed_dispatcher_targetoptions = []
    captured_targetoptions = _install_fake_dispatcher_compile(
        monkeypatch,
        on_compile=lambda: observed_dispatcher_targetoptions.append(dispatcher.targetoptions),
    )

    result = dispatcher.compile(nb_types.void(), output="ltoir")

    assert captured_targetoptions[0]["_compile_output"] == "ltoir"
    assert captured_targetoptions[0]["lto"] is True
    assert result.metadata["targetoptions"] is captured_targetoptions[0]
    assert observed_dispatcher_targetoptions == [dispatcher.targetoptions]
    assert "_compile_output" not in observed_dispatcher_targetoptions[0]
    assert dispatcher.targetoptions["lto"] is False
    assert "_compile_output" not in dispatcher.targetoptions


def test_dispatcher_compile_output_ltoir_does_not_leak_into_later_link_plan(monkeypatch):
    _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)

    dispatcher.compile(nb_types.void(), output="ltoir")
    plan = resolve_link_plan(dispatcher.targetoptions)

    assert plan.compile_output == "ptx"
    assert plan.requested_lto is False
    assert plan.linker_uses_lto is False


def test_dispatcher_compile_output_holds_targetoptions_lock(monkeypatch):
    observed_lock_depth = []
    dispatcher = cuda.jit(_kernel)
    lock = _RecordingLock()
    dispatcher._targetoptions_lock = lock
    _install_fake_dispatcher_compile(
        monkeypatch, on_compile=lambda: observed_lock_depth.append(lock.depth)
    )

    dispatcher.compile(nb_types.void(), output="ltoir")

    assert observed_lock_depth == [1]
    assert lock.depth == 0


def test_dispatcher_regular_compile_holds_targetoptions_lock_for_empty_updates(monkeypatch):
    observed_lock_depth = []
    dispatcher = cuda.jit(_kernel)
    lock = _RecordingLock()
    dispatcher._targetoptions_lock = lock
    _install_fake_dispatcher_compile(
        monkeypatch, on_compile=lambda: observed_lock_depth.append(lock.depth)
    )

    dispatcher.compile(nb_types.void())

    assert observed_lock_depth == [1]
    assert lock.depth == 0


def test_dispatcher_compile_output_ltoir_rejects_explicit_lto_false():
    dispatcher = cuda.jit(lto=False)(_kernel)

    with pytest.raises(ValueError, match="Cannot produce LTOIR output with lto=False"):
        dispatcher.compile(nb_types.void(), output="ltoir")


def test_temporary_targetoptions_restores_dispatcher_and_cache_after_exception():
    dispatcher = cuda.jit(_kernel)
    original_targetoptions = dispatcher.targetoptions
    dispatcher._cache = types.SimpleNamespace(_targetoptions=original_targetoptions)

    with pytest.raises(RuntimeError, match="stop"):
        with dispatcher._temporary_targetoptions({"debug": True}) as active_targetoptions:
            assert active_targetoptions is dispatcher.targetoptions
            assert active_targetoptions["debug"] is True
            assert dispatcher._cache._targetoptions is active_targetoptions
            raise RuntimeError("stop")

    assert dispatcher.targetoptions is original_targetoptions
    assert dispatcher._cache._targetoptions is original_targetoptions
    assert dispatcher.targetoptions["debug"] is False


def test_compile_existing_dispatcher_output_does_not_mutate_targetoptions(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    from numba_cuda_mlir import compiler

    compiler._compile(
        dispatcher,
        nb_types.void(),
        {"_compile_output": "ltoir", "lto": True},
        optimized=False,
    )

    assert captured_targetoptions[0]["_compile_output"] == "ltoir"
    assert captured_targetoptions[0]["lto"] is True
    assert dispatcher.targetoptions["lto"] is False
    assert "_compile_output" not in dispatcher.targetoptions


def test_compile_rejects_legacy_output_targetoption_for_plain_function():
    from numba_cuda_mlir import compiler

    with pytest.raises(ValueError, match=r"targetoptions\['output'\] is not supported"):
        compiler._compile(_kernel, nb_types.void(), {"output": "ltoir"}, optimized=False)


def test_compile_rejects_legacy_output_targetoption_for_existing_dispatcher():
    dispatcher = cuda.jit(_kernel)
    from numba_cuda_mlir import compiler

    with pytest.raises(ValueError, match=r"targetoptions\['output'\] is not supported"):
        compiler._compile(dispatcher, nb_types.void(), {"output": "ltoir"}, optimized=False)


def test_compile_rejects_ltoir_output_with_explicit_lto_false_for_plain_function():
    from numba_cuda_mlir import compiler

    with pytest.raises(ValueError, match="Cannot produce LTOIR output with lto=False"):
        compiler._compile(
            _kernel,
            nb_types.void(),
            {"_compile_output": "ltoir", "lto": False},
            optimized=False,
        )


def test_compile_rejects_ltoir_output_with_explicit_lto_false_for_dispatcher():
    dispatcher = cuda.jit(_kernel)
    from numba_cuda_mlir import compiler

    with pytest.raises(ValueError, match="Cannot produce LTOIR output with lto=False"):
        compiler._compile(
            dispatcher,
            nb_types.void(),
            {"_compile_output": "ltoir", "lto": False},
            optimized=False,
        )


def test_compile_dispatcher_lto_override_is_explicit(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    from numba_cuda_mlir import compiler

    compiler._compile(dispatcher, nb_types.void(), {"lto": True}, optimized=False)

    assert captured_targetoptions[0]["lto"] is True
    assert captured_targetoptions[0]["_lto_explicit"] is True
    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_compile_dispatcher_targetoption_override_bypasses_regular_overload(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    from numba_cuda_mlir import compiler

    dispatcher.compile(nb_types.void())
    result = compiler._compile(dispatcher, nb_types.void(), {"debug": True}, optimized=False)

    assert len(captured_targetoptions) == 2
    assert captured_targetoptions[0]["debug"] is False
    assert captured_targetoptions[1]["debug"] is True
    assert result.metadata["targetoptions"] is captured_targetoptions[1]


def test_dispatcher_compile_regular_overload_cache_includes_targetoptions(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)

    first = dispatcher.compile(nb_types.void())
    dispatcher.targetoptions["fastmath"] = True
    second = dispatcher.compile(nb_types.void())

    assert first is not second
    assert len(captured_targetoptions) == 2
    assert captured_targetoptions[1]["fastmath"] is True


def test_compile_plain_function_ltoir_output_preserves_implicit_lto(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    from numba_cuda_mlir import compiler

    compiler._compile(_kernel, nb_types.void(), {"_compile_output": "ltoir"}, optimized=False)

    assert captured_targetoptions[0]["_compile_output"] == "ltoir"
    assert captured_targetoptions[0]["lto"] is True
    assert captured_targetoptions[0]["_lto_explicit"] is False


def test_compile_plain_function_accepts_none_targetoptions(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    from numba_cuda_mlir import compiler

    result = compiler._compile(_kernel, nb_types.void(), None, optimized=False)

    assert result.metadata["targetoptions"] is captured_targetoptions[0]


def test_compile_plain_function_abi_info_does_not_reach_jit(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    from numba_cuda_mlir import compiler

    abi_info = {"abi_name": "kernel_abi"}
    compiler._compile(_kernel, nb_types.void(), {"abi_info": abi_info}, optimized=False)

    assert captured_targetoptions[0]["abi_info"] is abi_info


def test_dispatcher_compile_output_ptx_preserves_explicit_lto(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(lto=True)(_kernel)

    dispatcher.compile(nb_types.void(), output="ptx")

    assert captured_targetoptions[0]["_compile_output"] == "ptx"
    assert captured_targetoptions[0]["lto"] is True
    assert dispatcher.targetoptions["lto"] is True
    assert "_compile_output" not in dispatcher.targetoptions


def test_dispatcher_compile_abi_info_does_not_mutate_targetoptions(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    abi_info = {"abi_name": "kernel_abi"}

    dispatcher.compile(nb_types.void(), abi_info=abi_info)

    assert captured_targetoptions[0]["abi_info"] == abi_info
    assert dispatcher.targetoptions["abi_info"] is None


def test_dispatcher_compile_abi_info_initializes_mlir_target(monkeypatch):
    from numba_cuda_mlir import descriptor

    initialized = False

    def ensure_initialized():
        nonlocal initialized
        initialized = True

    def assert_initialized():
        assert initialized

    monkeypatch.setattr(descriptor.mlir_target, "ensure_initialized", ensure_initialized)
    _install_fake_dispatcher_compile(monkeypatch, on_compile=assert_initialized)
    dispatcher = cuda.jit(_kernel)

    dispatcher.compile(nb_types.void(), abi_info={"abi_name": "kernel_abi"})

    assert initialized


def test_dispatcher_compile_abi_info_bypasses_regular_overload(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    sentinel = object()
    dispatcher.overloads[()] = sentinel
    abi_info = {"abi_name": "kernel_abi"}

    result = dispatcher.compile(nb_types.void(), abi_info=abi_info)

    assert result is not sentinel
    assert dispatcher.overloads[()] is sentinel
    assert captured_targetoptions[0]["abi_info"] == abi_info


def test_dispatcher_compile_abi_info_uses_option_overload_cache(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)

    first = dispatcher.compile(
        nb_types.void(), abi_info={"abi_name": "kernel_abi", "tags": ["a", "b"]}
    )
    second = dispatcher.compile(
        nb_types.void(), abi_info={"abi_name": "kernel_abi", "tags": ["a", "b"]}
    )

    assert first is second
    assert len(captured_targetoptions) == 1
    assert () not in dispatcher.overloads


def test_dispatcher_compile_abi_info_cache_separates_distinct_options(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)

    first = dispatcher.compile(nb_types.void(), abi_info={"abi_name": "first"})
    second = dispatcher.compile(nb_types.void(), abi_info={"abi_name": "second"})

    assert first is not second
    assert len(captured_targetoptions) == 2


def test_dispatcher_compile_option_overload_cache_is_bounded(monkeypatch):
    from numba_cuda_mlir import descriptor

    monkeypatch.setattr(descriptor, "_COMPILE_OPTION_OVERLOAD_CACHE_SIZE", 1)
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)

    first = dispatcher.compile(nb_types.void(), abi_info={"abi_name": "first"})
    dispatcher.compile(nb_types.void(), abi_info={"abi_name": "second"})
    repeated_first = dispatcher.compile(nb_types.void(), abi_info={"abi_name": "first"})

    assert repeated_first is not first
    assert len(dispatcher._compile_option_overloads) == 1
    assert len(captured_targetoptions) == 3


def test_dispatcher_compile_output_uses_option_overload_cache(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)

    first = dispatcher.compile(nb_types.void(), output="ltoir")
    second = dispatcher.compile(nb_types.void(), output="ltoir")

    assert first is second
    assert len(captured_targetoptions) == 1
    assert () not in dispatcher.overloads


def test_dispatcher_compile_output_option_cache_includes_active_targetoptions(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    from numba_cuda_mlir import compiler

    compiler._compile(dispatcher, nb_types.void(), {"_compile_output": "ltoir"}, optimized=False)
    compiler._compile(
        dispatcher,
        nb_types.void(),
        {"_compile_output": "ltoir", "fastmath": True},
        optimized=False,
    )

    assert len(captured_targetoptions) == 2
    assert captured_targetoptions[1]["fastmath"] is True


def test_dispatcher_compile_output_cache_hit_populates_option_overload_cache(monkeypatch):
    from numba_cuda_mlir.descriptor import mlir_target
    from numba_cuda_mlir.numba_cuda.compiler import sigutils

    monkeypatch.setattr(mlir_target, "ensure_initialized", lambda: None)
    argtypes, return_type = sigutils.normalize_signature(nb_types.void())
    cached_result = types.SimpleNamespace(
        metadata={"targetoptions": {"_compile_output": "ltoir"}, "ltoir": b"ltoir"},
        signature=types.SimpleNamespace(args=argtypes, return_type=return_type),
        target_context=types.SimpleNamespace(insert_user_function=lambda *args: None),
        entry_point=object(),
        fndesc=object(),
        library=object(),
    )
    dispatcher = cuda.jit(_kernel)
    dispatcher._cache = _LoadingCache(cached_result)
    applied_carveouts = []
    monkeypatch.setattr(
        dispatcher,
        "_apply_shared_memory_carveout",
        lambda wrapped: applied_carveouts.append(wrapped),
    )

    first = dispatcher.compile(nb_types.void(), output="ltoir")
    second = dispatcher.compile(nb_types.void(), output="ltoir")

    assert first is second
    assert dispatcher._cache.load_count == 1
    assert applied_carveouts == [first]


def test_compile_existing_dispatcher_output_cache_hit_is_not_reoptimized(monkeypatch):
    from numba_cuda_mlir import compiler
    from numba_cuda_mlir.descriptor import mlir_target
    from numba_cuda_mlir.numba_cuda.compiler import sigutils

    monkeypatch.setattr(mlir_target, "ensure_initialized", lambda: None)
    argtypes, return_type = sigutils.normalize_signature(nb_types.void())
    cached_result = types.SimpleNamespace(
        metadata={
            "targetoptions": {"_compile_output": "ltoir"},
            "ltoir": b"ltoir",
            "_optimization_complete": False,
        },
        signature=types.SimpleNamespace(args=argtypes, return_type=return_type),
        target_context=types.SimpleNamespace(insert_user_function=lambda *args: None),
        entry_point=object(),
        fndesc=object(),
        library=object(),
    )
    dispatcher = cuda.jit(_kernel)
    dispatcher._cache = _LoadingCache(cached_result)

    result = compiler._compile(
        dispatcher,
        nb_types.void(),
        {"_compile_output": "ltoir"},
        optimized=True,
    )

    assert result.cres.cres is cached_result


def test_dispatcher_compile_output_ignores_malformed_disk_cache_hit(monkeypatch):
    from numba_cuda_mlir.descriptor import mlir_target
    from numba_cuda_mlir.numba_cuda.compiler import sigutils

    monkeypatch.setattr(mlir_target, "ensure_initialized", lambda: None)
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    argtypes, return_type = sigutils.normalize_signature(nb_types.void())
    cached_result = types.SimpleNamespace(
        metadata={"targetoptions": {"_compile_output": "ltoir"}},
        signature=types.SimpleNamespace(args=argtypes, return_type=return_type),
        target_context=types.SimpleNamespace(insert_user_function=lambda *args: None),
        entry_point=object(),
        fndesc=object(),
        library=object(),
    )
    dispatcher = cuda.jit(_kernel)
    dispatcher._cache = _LoadingCache(cached_result)

    result = dispatcher.compile(nb_types.void(), output="ltoir")

    assert result.cres is not cached_result
    assert dispatcher._cache.load_count == 1
    assert len(captured_targetoptions) == 1
    assert result.metadata["targetoptions"]["_compile_output"] == "ltoir"
    assert result.metadata["ltoir"] == b"ltoir"


def test_dispatcher_compile_output_disk_cache_roundtrip_separates_outputs(monkeypatch):
    from numba_cuda_mlir.descriptor import mlir_target

    class OptionKeyCache:
        def __init__(self, targetoptions):
            self._targetoptions = targetoptions
            self.entries = {}
            self.load_keys = []
            self.save_keys = []

        def _key(self):
            return _target_option_key(self._targetoptions)

        def load_overload(self, sig, target_context):
            key = self._key()
            self.load_keys.append(key)
            return self.entries.get(key)

        def save_overload(self, sig, result):
            key = self._key()
            self.save_keys.append(key)
            self.entries[key] = result

    monkeypatch.setattr(mlir_target, "ensure_initialized", lambda: None)
    monkeypatch.setattr(
        tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (8, 0) if as_type is tuple else "sm_80",
    )
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    dispatcher._cache = OptionKeyCache(dispatcher.targetoptions)

    first_ltoir = dispatcher.compile(nb_types.void(), output="ltoir")
    dispatcher._compile_option_overloads.clear()
    ptx = dispatcher.compile(nb_types.void(), output="ptx")
    dispatcher._compile_option_overloads.clear()
    second_ltoir = dispatcher.compile(nb_types.void(), output="ltoir")

    assert len(captured_targetoptions) == 2
    assert first_ltoir.metadata["targetoptions"]["_compile_output"] == "ltoir"
    assert ptx.metadata["targetoptions"]["_compile_output"] == "ptx"
    assert second_ltoir.metadata["targetoptions"] is first_ltoir.metadata["targetoptions"]
    assert dispatcher._cache.load_keys[0] != dispatcher._cache.load_keys[1]
    assert dispatcher._cache.load_keys[2] == dispatcher._cache.save_keys[0]


def test_dispatcher_recompile_clears_option_overload_cache(monkeypatch):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)

    first = dispatcher.compile(nb_types.void(), abi_info={"abi_name": "kernel_abi"})
    dispatcher.recompile()
    second = dispatcher.compile(nb_types.void(), abi_info={"abi_name": "kernel_abi"})

    assert first is not second
    assert len(captured_targetoptions) == 2


def test_compile_output_participates_in_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    ptx_key = _target_option_key(
        {
            "_compile_output": "ptx",
            "_lto_explicit": True,
            "lto": True,
            "chip": "sm_80",
        }
    )
    ltoir_key = _target_option_key(
        {
            "_compile_output": "ltoir",
            "_lto_explicit": True,
            "lto": True,
            "chip": "sm_80",
        }
    )

    assert ("_compile_output", "ptx") in ptx_key
    assert ("_compile_output", "ltoir") in ltoir_key
    assert ptx_key != ltoir_key


def test_derived_lto_does_not_participate_in_default_ptx_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    absent_key = _target_option_key({"chip": "sm_80", "_lto_explicit": False})
    false_key = _target_option_key({"chip": "sm_80", "_lto_explicit": False, "lto": False})
    true_key = _target_option_key({"chip": "sm_80", "_lto_explicit": False, "lto": True})

    assert absent_key == false_key == true_key
    assert ("lto", None) in true_key


def test_explicit_lto_participates_in_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    implicit_key = _target_option_key({"chip": "sm_80", "_lto_explicit": False, "lto": True})
    explicit_key = _target_option_key({"chip": "sm_80", "_lto_explicit": True, "lto": True})

    assert implicit_key != explicit_key
    assert ("lto", True) in explicit_key


def test_link_items_participate_in_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    no_link_key = _target_option_key({"chip": "sm_80"})
    none_link_key = _target_option_key({"chip": "sm_80", "link": None})
    ptx_link_key = _target_option_key({"chip": "sm_80", "link": ["external.ptx"]})
    ltoir_link_key = _target_option_key({"chip": "sm_80", "link": ["external.ltoir"]})

    assert no_link_key == none_link_key
    assert no_link_key != ptx_link_key
    assert ptx_link_key != ltoir_link_key


@pytest.mark.parametrize(
    "link_metadata",
    [
        {"linked_external_link_items": ("hidden.cu",)},
        {"external_link_items": ["hidden.cu"]},
    ],
)
def test_discovered_link_items_disable_persistent_cache(link_metadata):
    impl = object.__new__(caching.MLIRCacheImpl)
    cres = types.SimpleNamespace(metadata={"targetoptions": {}, **link_metadata})

    assert impl.check_cachable(cres) is False


def test_explicit_link_items_still_reject_persistent_cache():
    impl = object.__new__(caching.MLIRCacheImpl)
    cres = types.SimpleNamespace(metadata={"targetoptions": {"link": ["external.cu"]}})

    with pytest.raises(RuntimeError, match="linking files"):
        impl.check_cachable(cres)


def test_extension_targetoptions_disable_persistent_cache():
    impl = object.__new__(caching.MLIRCacheImpl)
    cres = types.SimpleNamespace(metadata={"targetoptions": {"extensions": [object()]}})

    assert impl.check_cachable(cres) is False


def test_compile_output_artifact_cache_payload_is_not_launch_complete():
    impl = object.__new__(caching.MLIRCacheImpl)
    cres = types.SimpleNamespace(
        signature=types.SimpleNamespace(args=(), return_type=nb_types.void),
        metadata={
            "cubin": b"artifact-cubin",
            "ltoir": b"artifact-ltoir",
            "targetoptions": {"_compile_output": "ltoir"},
        },
    )

    payload = impl.reduce(cres)
    rebuilt = impl.rebuild(None, payload)

    assert payload["_optimization_complete"] is False
    assert rebuilt.metadata["_optimization_complete"] is False


def test_path_link_item_digest_is_memoized(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    link_path = tmp_path / "external.ptx"
    link_path.write_text("// ptx")
    read_count = 0
    real_open = builtins.open
    caching._LINK_PATH_DIGEST_CACHE.clear()

    def counting_open(path, *args, **kwargs):
        nonlocal read_count
        if path == str(link_path):
            read_count += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    first_key = _target_option_key({"chip": "sm_80", "link": [str(link_path)]})
    second_key = _target_option_key({"chip": "sm_80", "link": [str(link_path)]})

    assert first_key == second_key
    assert read_count == 1
    caching._LINK_PATH_DIGEST_CACHE.clear()


def test_path_link_item_digest_cache_evicts_stale_path_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    link_path = tmp_path / "external.ptx"
    caching._LINK_PATH_DIGEST_CACHE.clear()

    link_path.write_text("// first")
    first_key = _target_option_key({"chip": "sm_80", "link": [str(link_path)]})
    next_mtime_ns = link_path.stat().st_mtime_ns + 1_000_000_000
    link_path.write_text("// second")
    os.utime(link_path, ns=(next_mtime_ns, next_mtime_ns))
    second_key = _target_option_key({"chip": "sm_80", "link": [str(link_path)]})

    assert first_key != second_key
    assert len(caching._LINK_PATH_DIGEST_CACHE) == 1
    caching._LINK_PATH_DIGEST_CACHE.clear()


def test_path_link_item_digest_cache_observes_file_identity_changes(monkeypatch):
    path = "external.ptx"
    stats = [
        types.SimpleNamespace(
            st_dev=1,
            st_ino=10,
            st_mtime_ns=100,
            st_ctime_ns=200,
            st_size=5,
        ),
        types.SimpleNamespace(
            st_dev=1,
            st_ino=11,
            st_mtime_ns=100,
            st_ctime_ns=201,
            st_size=5,
        ),
    ]
    payloads = [b"first", b"other"]
    read_count = 0
    caching._LINK_PATH_DIGEST_CACHE.clear()

    def fake_stat(stat_path):
        assert stat_path == path
        return stats.pop(0)

    def fake_open(open_path, mode="r", *args, **kwargs):
        nonlocal read_count
        assert open_path == path
        assert mode == "rb"
        read_count += 1
        return io.BytesIO(payloads.pop(0))

    monkeypatch.setattr(caching.os, "stat", fake_stat)
    monkeypatch.setattr(builtins, "open", fake_open)

    try:
        first_digest = caching._hash_path(path)
        second_digest = caching._hash_path(path)
    finally:
        caching._LINK_PATH_DIGEST_CACHE.clear()

    assert first_digest != second_digest
    assert read_count == 2


def test_path_link_item_key_includes_absolute_path(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    first_path = tmp_path / "first" / "external.ptx"
    second_path = tmp_path / "second" / "external.ptx"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text("// same")
    second_path.write_text("// same")

    first_key = _target_option_key({"chip": "sm_80", "link": [str(first_path)]})
    second_key = _target_option_key({"chip": "sm_80", "link": [str(second_path)]})

    assert first_key != second_key
    assert first_key[-1][1][0][2] == os.path.abspath(str(first_path))
    assert second_key[-1][1][0][2] == os.path.abspath(str(second_path))


def test_path_link_item_digest_cache_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    monkeypatch.setattr(caching, "_LINK_PATH_DIGEST_CACHE_MAX_ENTRIES", 2)
    caching._LINK_PATH_DIGEST_CACHE.clear()
    link_paths = []

    for i in range(3):
        link_path = tmp_path / f"external-{i}.ptx"
        link_path.write_text(f"// ptx {i}")
        link_paths.append(link_path)
        _target_option_key({"chip": "sm_80", "link": [str(link_path)]})

    evicted_path = os.path.abspath(str(link_paths[0]))
    assert len(caching._LINK_PATH_DIGEST_CACHE) == 2
    assert all(cache_key[0] != evicted_path for cache_key in caching._LINK_PATH_DIGEST_CACHE)
    caching._LINK_PATH_DIGEST_CACHE.clear()


def test_debug_options_participate_in_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    default_key = _target_option_key({"chip": "sm_80"})
    debug_key = _target_option_key({"chip": "sm_80", "debug": True})
    lineinfo_key = _target_option_key({"chip": "sm_80", "lineinfo": True})

    assert default_key != debug_key
    assert default_key != lineinfo_key
    assert debug_key != lineinfo_key


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("abi", "c"),
        ("capi", True),
        ("device", True),
        ("fastmath", True),
        ("features", "+ptx80"),
        ("inline", "never"),
        ("max_registers", 32),
        ("opt_level", 0),
        ("ptxas_options", "-v"),
    ),
)
def test_codegen_options_participate_in_cache_key(monkeypatch, option, value):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    default_key = _target_option_key({"chip": "sm_80"})
    option_key = _target_option_key({"chip": "sm_80", option: value})

    assert default_key != option_key


def test_extension_identity_participates_in_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    first_extension = object()
    second_extension = object()

    first_key = _target_option_key({"chip": "sm_80", "extensions": [first_extension]})
    first_key_again = _target_option_key({"chip": "sm_80", "extensions": [first_extension]})
    second_key = _target_option_key({"chip": "sm_80", "extensions": [second_extension]})

    assert first_key == first_key_again
    assert first_key != second_key


def test_in_memory_link_item_content_participates_in_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    first_key = _target_option_key({"chip": "sm_80", "link": [cuda.CUSource("int first;")]})
    second_key = _target_option_key({"chip": "sm_80", "link": [cuda.CUSource("int second;")]})

    assert first_key != second_key


def test_immutable_link_item_data_digest_is_memoized(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    caching._LINK_DATA_DIGEST_CACHE.clear()
    real_sha256 = caching.hashlib.sha256
    hash_count = 0

    def counting_sha256(*args, **kwargs):
        nonlocal hash_count
        hash_count += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(caching.hashlib, "sha256", counting_sha256)
    data = b"external-ltoir"
    link_item = types.SimpleNamespace(data=data, name="external.ltoir")

    first_key = _target_option_key({"chip": "sm_80", "link": [link_item]})
    second_key = _target_option_key({"chip": "sm_80", "link": [link_item]})

    assert first_key == second_key
    assert hash_count == 1
    caching._LINK_DATA_DIGEST_CACHE.clear()


def test_dataless_link_item_uses_explicit_cache_key_sentinel(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    key = _target_option_key({"chip": "sm_80", "link": [types.SimpleNamespace(name="dataless")]})

    assert key[-1][1][0][2] == ("data", None)


def test_file_like_link_item_data_participates_in_cache_key(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    link_path = tmp_path / "external.ptx"
    link_path.write_text("// ptx")

    with open(link_path) as f:
        key = _target_option_key(
            {"chip": "sm_80", "link": [types.SimpleNamespace(data=f, name="external.ptx")]}
        )
        assert f.tell() == 0

    assert key


class _NonSeekableLinkData:
    def __init__(self, data):
        self.data = data
        self.read_count = 0

    def tell(self):
        raise OSError("not seekable")

    def read(self):
        self.read_count += 1
        return self.data


class _FailingSeekableLinkData:
    def __init__(self):
        self.read_count = 0

    def tell(self):
        return 7

    def seek(self, pos):
        return pos

    def read(self):
        self.read_count += 1
        raise OSError("read failed")


def test_nonseekable_file_like_link_item_data_is_not_consumed(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    data = _NonSeekableLinkData(b"external")
    link_item = types.SimpleNamespace(data=data, name="stream.ptx")

    key = _target_option_key({"chip": "sm_80", "link": [link_item]})

    assert data.read_count == 0
    assert key[-1][1][0][2][0] == "opaque"


def test_failing_file_like_link_item_data_falls_back_to_opaque_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    data = _FailingSeekableLinkData()
    link_item = types.SimpleNamespace(data=data, name="stream.ptx")

    key = _target_option_key({"chip": "sm_80", "link": [link_item]})

    assert data.read_count == 1
    assert key[-1][1][0][2][0] == "opaque"


def test_opaque_link_item_data_uses_stable_per_object_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    link_item = types.SimpleNamespace(data=object(), name="bad")

    first_key = _target_option_key({"chip": "sm_80", "link": [link_item]})
    second_key = _target_option_key({"chip": "sm_80", "link": [link_item]})

    assert first_key[-1][1][0][2][0] == "opaque"
    assert first_key == second_key


def test_opaque_link_item_data_gets_fresh_key_after_eviction(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    monkeypatch.setattr(caching, "_OPAQUE_LINK_DATA_KEY_MAX_ENTRIES", 1)
    caching._OPAQUE_LINK_DATA_KEYS.clear()
    link_item = types.SimpleNamespace(data=object(), name="opaque")
    evicting_item = types.SimpleNamespace(data=object(), name="evicting")

    first_key = _target_option_key({"chip": "sm_80", "link": [link_item]})
    _target_option_key({"chip": "sm_80", "link": [evicting_item]})
    second_key = _target_option_key({"chip": "sm_80", "link": [link_item]})

    assert first_key[-1][1][0][2][0] == "opaque"
    assert first_key != second_key
    caching._OPAQUE_LINK_DATA_KEYS.clear()


def test_opaque_link_item_data_key_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    monkeypatch.setattr(caching, "_OPAQUE_LINK_DATA_KEY_MAX_ENTRIES", 2)
    caching._OPAQUE_LINK_DATA_KEYS.clear()
    link_items = [types.SimpleNamespace(data=object(), name=f"opaque-{i}") for i in range(3)]

    for link_item in link_items:
        _target_option_key({"chip": "sm_80", "link": [link_item]})

    assert len(caching._OPAQUE_LINK_DATA_KEYS) == 2
    assert ("opaque", id(link_items[0].data)) not in caching._OPAQUE_LINK_DATA_KEYS
    caching._OPAQUE_LINK_DATA_KEYS.clear()


def test_abi_info_participates_in_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))

    first_key = _target_option_key({"chip": "sm_80", "abi_info": {"abi_name": "first_abi"}})
    second_key = _target_option_key({"chip": "sm_80", "abi_info": {"abi_name": "second_abi"}})

    assert ("abi_info", (("abi_name", "first_abi"),)) in first_key
    assert first_key != second_key


def test_link_item_callback_identity_participates_in_cache_key(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))

    def first_callback(*args):
        pass

    def second_callback(*args):
        pass

    first_key = _target_option_key(
        {
            "chip": "sm_80",
            "link": [
                types.SimpleNamespace(
                    data=b"external", name="external.ptx", setup_callback=first_callback
                )
            ],
        }
    )
    second_key = _target_option_key(
        {
            "chip": "sm_80",
            "link": [
                types.SimpleNamespace(
                    data=b"external", name="external.ptx", setup_callback=second_callback
                )
            ],
        }
    )

    assert first_key[-1][1][0][3][0] == "callback"
    assert first_key != second_key


def test_unhashable_target_option_values_are_cache_key_safe(monkeypatch):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))

    key = _target_option_key(
        {
            "chip": "sm_80",
            "features": ["+ptx80", "+fast"],
            "launch_bounds": [128, 2],
            "ptxas_options": {"warn-spills": {"enabled", "verbose"}},
        }
    )

    hash(key)
    assert ("features", ("+ptx80", "+fast")) in key
    assert ("launch_bounds", (128, 2)) in key


def test_missing_path_link_item_uses_process_stable_uncacheable_key(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (8, 0))
    caching._MISSING_LINK_PATH_KEYS.clear()
    missing_path = tmp_path / "missing.ptx"

    first_key = _target_option_key({"chip": "sm_80", "link": [str(missing_path)]})
    second_key = _target_option_key({"chip": "sm_80", "link": [str(missing_path)]})

    assert first_key[-1][1][0][0] == "missing-path"
    assert first_key == second_key


@pytest.mark.parametrize("method_name", ("_compile_as_device_callee", "compile_device"))
def test_device_compile_paths_neutralize_compile_output(monkeypatch, method_name):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    dispatcher.targetoptions["_compile_output"] = "ltoir"
    dispatcher.targetoptions["lto"] = True

    getattr(dispatcher, method_name)(nb_types.void())

    assert captured_targetoptions[0].get("_compile_output") is None
    assert captured_targetoptions[0]["device"] is True
    assert captured_targetoptions[0]["lto"] is False


@pytest.mark.parametrize("method_name", ("_compile_as_device_callee", "compile_device"))
def test_device_compile_paths_clear_compile_output_on_device_dispatcher(monkeypatch, method_name):
    captured_targetoptions = _install_fake_dispatcher_compile(monkeypatch)
    dispatcher = cuda.jit(_kernel)
    dispatcher.targetoptions["device"] = True
    dispatcher.targetoptions["_compile_output"] = "ltoir"
    dispatcher.targetoptions["lto"] = True

    getattr(dispatcher, method_name)(nb_types.void())

    assert captured_targetoptions[0].get("_compile_output") is None
    assert captured_targetoptions[0]["device"] is True
    assert captured_targetoptions[0]["lto"] is False
    assert "_compile_output" not in dispatcher.targetoptions


def test_debug_ltoir_omits_debug_info_version_module_flag(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    monkeypatch.setattr(Linker, "complete", lambda self: types.SimpleNamespace(code=b""))

    code, resty = cuda.compile(
        _device_increment,
        nb_types.int32(nb_types.int32),
        device=True,
        output="ltoir",
        debug=True,
        opt=False,
    )

    assert resty == nb_types.int32
    assert b"Debug Info Version" not in code


class _RecordingResolvedLinker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.link_items = []
        self.ltoirs = []

    def add_file_guess_ext(self, link_item, compile_cu_as_ltoir=None, ignore_nonlto=False):
        self.link_items.append((link_item, compile_cu_as_ltoir, ignore_nonlto))

    def add_ltoir(self, ltoir, name=""):
        self.ltoirs.append((ltoir, name))


def _fake_lower_for_resolved_linking(targetoptions):
    lower = MLIRLower.__new__(MLIRLower)
    lower.targetoptions = targetoptions
    lower._linker_config = {"cc": (8, 0), "arch": "sm_80"}
    lower._linked_external_link_items = []
    lower._linked_ltoirs = []
    lower._linked_ltoir_keys = set()
    lower.metadata = {}
    return lower


def test_lowering_records_ltoirs_once_by_content():
    lower = _fake_lower_for_resolved_linking({})
    first_linker = types.SimpleNamespace(_ltoirs={b"callee": ""})
    second_linker = types.SimpleNamespace(_ltoirs={b"callee": "", b"other": ""})

    lower._record_ltoirs_from_linker(first_linker)
    lower._record_ltoirs_from_linker(second_linker)

    assert lower._linked_ltoirs == [b"callee", b"other"]


def test_lowering_creates_lto_linker_for_existing_ltoir_but_keeps_new_cu_on_ptx(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    created_linkers = []

    def linker_factory(**kwargs):
        linker = _RecordingResolvedLinker(**kwargs)
        created_linkers.append(linker)
        return linker

    from numba_cuda_mlir import mlir_lowering

    monkeypatch.setattr(mlir_lowering, "Linker", linker_factory)
    lower = _fake_lower_for_resolved_linking(
        {
            "lto": False,
            "debug": True,
            "opt_level": 0,
            "_lto_explicit": False,
        }
    )
    lower._linked_external_link_items = ["external.cu"]
    lower._linked_ltoirs = [b"callee-ltoir"]

    linker = lower._create_resolved_linker()

    assert linker is created_linkers[-1]
    assert linker.kwargs["lto"] is True
    assert linker.link_items == [("external.cu", False, False)]
    assert linker.ltoirs == [(b"callee-ltoir", "")]
    assert lower.metadata["link_plan"].compile_new_inputs_as_ltoir is False
    assert lower.metadata["link_plan"].compile_cuda_sources_as_ltoir is False
    assert lower.metadata["link_plan"].linker_uses_lto is True
    assert lower.metadata["linked_external_link_items"] == ("external.cu",)


def test_lowering_rejects_debug_cu_link_item_with_explicit_lto(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    lower = _fake_lower_for_resolved_linking(
        {
            "lto": True,
            "debug": True,
            "opt_level": 0,
            "_lto_explicit": True,
        }
    )
    lower._linked_external_link_items = ["external.cu"]

    with pytest.raises(ValueError, match="debug=True with lto=True"):
        lower._create_resolved_linker()


def test_dispatcher_compile_rejects_ltoir_link_item_with_explicit_lto_false():
    dispatcher = cuda.jit(link=["external.ltoir"], lto=False)(_kernel)

    with pytest.raises(ValueError, match="Cannot link LTOIR inputs with lto=False"):
        dispatcher.compile(nb_types.void())


class _DiagnosticLinker:
    def __init__(self):
        self.ltoirs = []
        self.link_items = []
        self.additional_flags = []

    def add_ltoir(self, ltoir, name=""):
        self.ltoirs.append((ltoir, name))

    def add_file_guess_ext(self, link_item, compile_cu_as_ltoir=None, ignore_nonlto=False):
        self.link_items.append((link_item, compile_cu_as_ltoir, ignore_nonlto))

    def get_linked_ptx(self):
        return b"linked-lto-ptx"


class _SourceLinker:
    def __init__(self, lto=False):
        self.lto = lto
        self.recreate_args = None
        self.diag_linker = _DiagnosticLinker()

    def recreate_with_lto(self, lto=True, ltoir_only=False):
        self.recreate_args = (lto, ltoir_only)
        return self.diag_linker


class _RetryLinker:
    def __init__(self, name):
        self.name = name
        self.recreate_calls = []
        self.inputs = []

    def recreate_with_lto(self, lto=True, ltoir_only=False):
        self.recreate_calls.append((lto, ltoir_only))
        return _RetryLinker(f"{self.name}-retry-{len(self.recreate_calls)}")


def test_get_lto_ptx_replays_discovered_link_items_for_diagnostic_lto(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    linker = _SourceLinker(lto=True)
    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {"chip": "sm_90", "link": ["decorator.cu"]},
            "linker": linker,
            "linked_external_link_items": ("decorator.cu", "hidden.cu"),
        }
    )
    monkeypatch.setattr(
        mlir_optimization,
        "_compile_ltoir_for_inspection",
        lambda cres, target_options: b"kernel-ltoir",
    )

    assert mlir_optimization.get_lto_ptx(cres) == "linked-lto-ptx"
    assert linker.recreate_args == (True, True)
    assert linker.diag_linker.ltoirs == [(b"kernel-ltoir", "")]
    assert linker.diag_linker.link_items == [
        ("decorator.cu", None, True),
        ("hidden.cu", None, True),
    ]


def test_get_lto_ptx_skips_link_items_already_materialized_as_ltoir(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    linker = _SourceLinker(lto=True)
    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {"chip": "sm_90"},
            "linker": linker,
            "linked_external_link_items": (
                "already-materialized.cu",
                "already-materialized.ltoir",
                "ignored-nonlto.ptx",
            ),
            "link_plan": types.SimpleNamespace(compile_new_inputs_as_ltoir=True),
        }
    )
    monkeypatch.setattr(
        mlir_optimization,
        "_compile_ltoir_for_inspection",
        lambda cres, target_options: b"kernel-ltoir",
    )

    assert mlir_optimization.get_lto_ptx(cres) == "linked-lto-ptx"
    assert linker.recreate_args == (True, True)
    assert linker.diag_linker.link_items == [("ignored-nonlto.ptx", True, True)]


def test_get_lto_ptx_replays_cu_link_items_kept_on_ptx(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    linker = _SourceLinker(lto=True)
    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {"chip": "sm_90"},
            "linker": linker,
            "linked_external_link_items": (
                "debug-source.cu",
                "already-materialized.ltoir",
            ),
            "link_plan": types.SimpleNamespace(
                compile_new_inputs_as_ltoir=True,
                compile_cuda_sources_as_ltoir=False,
            ),
        }
    )
    monkeypatch.setattr(
        mlir_optimization,
        "_compile_ltoir_for_inspection",
        lambda cres, target_options: b"kernel-ltoir",
    )

    assert mlir_optimization.get_lto_ptx(cres) == "linked-lto-ptx"
    assert linker.recreate_args == (True, True)
    assert linker.diag_linker.link_items == []


def test_get_lto_ptx_tolerates_partial_link_plan_for_cu_replay(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    linker = _SourceLinker(lto=True)
    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {"chip": "sm_90"},
            "linker": linker,
            "linked_external_link_items": ("legacy-debug-source.cu",),
            "link_plan": types.SimpleNamespace(compile_cuda_sources_as_ltoir=False),
        }
    )
    monkeypatch.setattr(
        mlir_optimization,
        "_compile_ltoir_for_inspection",
        lambda cres, target_options: b"kernel-ltoir",
    )

    assert mlir_optimization.get_lto_ptx(cres) == "linked-lto-ptx"
    assert linker.diag_linker.link_items == []


def test_get_lto_ptx_uses_decorator_link_items_without_discovered_metadata(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    linker = _SourceLinker(lto=False)
    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {"chip": "sm_90", "link": ["decorator.cu"]},
            "linker": linker,
        }
    )
    monkeypatch.setattr(
        mlir_optimization,
        "_compile_ltoir_for_inspection",
        lambda cres, target_options: b"kernel-ltoir",
    )

    assert mlir_optimization.get_lto_ptx(cres) == "linked-lto-ptx"
    assert linker.diag_linker.link_items == [("decorator.cu", None, True)]


def test_get_lto_ptx_replays_existing_non_lto_linker_items(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    linker = _SourceLinker(lto=False)
    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {"chip": "sm_90"},
            "linker": linker,
            "linked_external_link_items": ("legacy.ptx", "debug-source.cu"),
            "link_plan": types.SimpleNamespace(
                compile_new_inputs_as_ltoir=False,
                compile_cuda_sources_as_ltoir=False,
            ),
        }
    )
    monkeypatch.setattr(
        mlir_optimization,
        "_compile_ltoir_for_inspection",
        lambda cres, target_options: b"kernel-ltoir",
    )

    assert mlir_optimization.get_lto_ptx(cres) == "linked-lto-ptx"
    assert linker.recreate_args == (True, True)
    assert linker.diag_linker.link_items == [
        ("legacy.ptx", False, True),
    ]


def test_get_lto_ptx_replays_all_link_items_when_creating_diagnostic_linker(monkeypatch):
    from numba_cuda_mlir import linker as linker_mod
    from numba_cuda_mlir import mlir_optimization

    created_linkers = []

    class CreatedLinker(_SourceLinker):
        def __init__(self, **kwargs):
            super().__init__(lto=kwargs.get("lto", False))
            created_linkers.append(self)

    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {
                "chip": "sm_90",
                "link": [
                    "materialized.cu",
                    "materialized.ltoir",
                    "ignored-nonlto.ptx",
                ],
            },
            "link_plan": types.SimpleNamespace(compile_new_inputs_as_ltoir=True),
        }
    )
    monkeypatch.setattr(
        mlir_optimization,
        "_compile_ltoir_for_inspection",
        lambda cres, target_options: b"kernel-ltoir",
    )
    monkeypatch.setattr(linker_mod, "Linker", CreatedLinker)

    assert mlir_optimization.get_lto_ptx(cres) == "linked-lto-ptx"
    assert created_linkers[-1].diag_linker.link_items == [
        ("materialized.cu", True, True),
        ("materialized.ltoir", True, True),
        ("ignored-nonlto.ptx", True, True),
    ]


def test_optimize_skips_already_completed_result(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    def fail_context():
        raise AssertionError("optimize should not re-enter MLIR passes after cubin exists")

    cres = types.SimpleNamespace(
        metadata={"cubin": b"already-linked", "_optimization_complete": True}
    )
    monkeypatch.setattr(mlir_optimization.context, "get_context", fail_context)

    mlir_optimization.optimize(cres)


def test_optimize_does_not_skip_placeholder_cubin_without_completion_marker(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    def fail_context():
        raise AssertionError("optimize should not skip placeholder cubin")

    cres = types.SimpleNamespace(metadata={"cubin": b"placeholder"})
    monkeypatch.setattr(mlir_optimization.context, "get_context", fail_context)

    with pytest.raises(AssertionError, match="placeholder cubin"):
        mlir_optimization.optimize(cres)


def test_optimize_retries_use_base_linker_not_mutated_attempt_linker():
    from numba_cuda_mlir import mlir_optimization

    base_linker = _RetryLinker("base")
    cres = types.SimpleNamespace(metadata={"linker": base_linker})

    first = mlir_optimization._fresh_linker_for_optimize(cres, linker_uses_lto=True)
    first.inputs.append("stale-ltoir")
    second = mlir_optimization._fresh_linker_for_optimize(cres, linker_uses_lto=True)

    assert first is not second
    assert first.inputs == ["stale-ltoir"]
    assert second.inputs == []
    assert base_linker.recreate_calls == [(True, False), (True, False)]


def test_root_linker_entry_kernel_preserves_existing_variables_used():
    from numba_cuda_mlir import mlir_optimization

    linker = types.SimpleNamespace(
        variables_used=["retained_global"],
        kernels_used=None,
    )

    mlir_optimization._root_linker_entry_kernel(linker, "kernel_name")

    assert linker.variables_used == ["retained_global"]
    assert linker.kernels_used == ["kernel_name"]


def test_root_linker_entry_kernel_preserves_existing_kernel_roots():
    from numba_cuda_mlir import mlir_optimization

    linker = types.SimpleNamespace(
        variables_used=["retained_global"],
        kernels_used="existing_kernel",
    )

    mlir_optimization._root_linker_entry_kernel(linker, "kernel_name")

    assert linker.kernels_used == ["existing_kernel", "kernel_name"]
