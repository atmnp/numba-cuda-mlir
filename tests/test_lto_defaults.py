# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIR JIT LTO option defaults."""

import types

import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.caching import _targetoptions_cache_key
from numba_cuda_mlir.mlir_lowering import MLIRLower


def _kernel():
    pass


def _set_nvjitlink_available(monkeypatch, available):
    from numba_cuda_mlir.numba_cuda.cudadrv import driver

    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: available)


def test_mlir_jit_defaults_to_ptx_linking():
    dispatcher = cuda.jit(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False
    assert dispatcher.targetoptions["_output_explicit"] is False


def test_mlir_jit_enables_implicit_lto_for_ltoir_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    dispatcher = cuda.jit(link=["external.ltoir"])(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_keeps_cu_link_items_on_ptx_by_default(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    dispatcher = cuda.jit(link=["external.cu"])(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_disables_implicit_lto_without_nvjitlink(monkeypatch):
    _set_nvjitlink_available(monkeypatch, False)

    dispatcher = cuda.jit(link=["external.cu"])(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_rejects_ltoir_link_items_without_nvjitlink(monkeypatch):
    _set_nvjitlink_available(monkeypatch, False)

    with pytest.raises(ValueError, match="LTOIR link inputs require LTO"):
        cuda.jit(link=["external.ltoir"])(_kernel)


def test_mlir_jit_preserves_explicit_lto_false_with_non_lto_link_items():
    dispatcher = cuda.jit(link=["external.cu"], lto=False)(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is True


def test_mlir_jit_rejects_ltoir_link_items_with_explicit_lto_false():
    with pytest.raises(ValueError, match="LTOIR link inputs require LTO"):
        cuda.jit(link=["external.ltoir"], lto=False)(_kernel)


def test_mlir_jit_rejects_ltoir_linkable_objects_with_explicit_lto_false():
    with pytest.raises(ValueError, match="LTOIR link inputs require LTO"):
        cuda.jit(link=[cuda.LTOIR(b"")], lto=False)(_kernel)


def test_mlir_jit_rejects_ltoir_link_items_with_explicit_output_ptx(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    with pytest.raises(ValueError, match="LTOIR link inputs require LTO"):
        cuda.jit(link=["external.ltoir"], output="ptx")(_kernel)


def test_mlir_jit_disables_implicit_lto_for_debug_external_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    # debug=True requires opt=False so verify_target_options resolves opt_level=0.
    dispatcher = cuda.jit(link=["external.cu"], debug=True, opt=False)(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_rejects_ltoir_link_items_when_debug_disables_lto(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    with pytest.raises(ValueError, match="LTOIR link inputs require LTO"):
        cuda.jit(link=["external.ltoir"], debug=True, opt=False)(_kernel)


def test_mlir_jit_enables_implicit_lto_for_lineinfo_external_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    dispatcher = cuda.jit(link=["external.ltoir"], lineinfo=True)(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_rejects_output_ltoir_with_explicit_lto_false():
    with pytest.raises(ValueError, match='output="ltoir" requires LTO'):
        cuda.jit(output="ltoir", lto=False)(_kernel)


def test_mlir_jit_explicit_output_ptx_takes_precedence_over_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    dispatcher = cuda.jit(link=["external.cu"], output="ptx")(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False
    assert dispatcher.targetoptions["_output_explicit"] is True


def test_mlir_jit_preserves_explicit_lto_true():
    dispatcher = cuda.jit(lto=True)(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["_lto_explicit"] is True


def test_mlir_jit_disables_implicit_lto_for_callback_link_items():
    class LinkItem:
        def __init__(self):
            self.setup_callback = lambda: None
            self.teardown_callback = None

    dispatcher = cuda.jit(link=[LinkItem()])(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_output_ltoir_enables_lto():
    dispatcher = cuda.jit(output="ltoir")(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["_lto_explicit"] is False
    assert dispatcher.targetoptions["_output_explicit"] is True


def test_mlir_jit_explicit_output_ltoir_takes_precedence_over_callback_link_items():
    dispatcher = cuda.jit(output="ltoir", link=[_CallbackLinkItem()])(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["output"] == "ltoir"
    assert dispatcher.targetoptions["_lto_explicit"] is False
    assert dispatcher.targetoptions["_output_explicit"] is True


class _RecordingLinker:
    def __init__(self):
        self.link_items = []

    def add_file_guess_ext(self, link_item):
        self.link_items.append(link_item)


class _CallbackLinkItem:
    def setup_callback(self, *args):
        return None


def _fake_lower_for_linking(targetoptions):
    lower = MLIRLower.__new__(MLIRLower)
    lower.targetoptions = targetoptions
    lower._linker_config = {"lto": targetoptions.get("lto", False)}
    lower._linked_external_items = set()
    lower._linked_external_link_items = []
    lower.metadata = {}
    lower._setup_callbacks = []
    lower._teardown_callbacks = []
    linkers = [_RecordingLinker()]
    lower.linker = linkers[0]

    def create_linker(self):
        linkers.append(_RecordingLinker())
        return linkers[-1]

    lower._create_linker = types.MethodType(create_linker, lower)
    return lower, linkers


def test_lowering_keeps_discovered_cu_link_item_on_ptx_by_default(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": False,
        }
    )

    lower.link_external_item("external.cu")

    assert lower.targetoptions["lto"] is False
    assert lower._linker_config["lto"] is False
    assert linkers[-1].link_items == ["external.cu"]


def test_lowering_records_discovered_external_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    lower, _ = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": False,
        }
    )

    lower.link_external_item("external.cu")

    assert lower.metadata["external_link_items"] == ["external.cu"]


def test_lowering_enables_implicit_lto_for_discovered_ltoir_link_item(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": False,
        }
    )

    lower.link_external_item("external.ltoir")

    assert lower.targetoptions["lto"] is True
    assert lower._linker_config["lto"] is True
    assert linkers[-1].link_items == ["external.ltoir"]


def test_lowering_rejects_discovered_ltoir_link_item_with_explicit_lto_false(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    lower, _ = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": True,
            "_output_explicit": False,
        }
    )

    with pytest.raises(ValueError, match="LTOIR link inputs require LTO"):
        lower.link_external_item("external.ltoir")


def test_lowering_rejects_discovered_ltoir_callback_link_item_without_lto(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    ltoir_item = cuda.LTOIR(b"", setup_callback=lambda *args: None)
    lower, _ = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": False,
        }
    )

    with pytest.raises(ValueError, match="LTOIR link inputs require LTO"):
        lower.link_external_item(ltoir_item)


def test_lowering_rejects_discovered_ltoir_link_item_with_explicit_output_ptx(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    lower, _ = _fake_lower_for_linking(
        {
            "lto": False,
            "output": "ptx",
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": True,
        }
    )

    with pytest.raises(ValueError, match="LTOIR link inputs require LTO"):
        lower.link_external_item("external.ltoir")


def test_lowering_preserves_explicit_lto_false_for_discovered_external_link_item(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": True,
            "_output_explicit": False,
        }
    )

    lower.link_external_item("external.cu")

    assert lower.targetoptions["lto"] is False
    assert lower._linker_config["lto"] is False
    assert linkers[-1].link_items == ["external.cu"]


def test_lowering_preserves_explicit_output_ptx_for_discovered_external_link_item(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": True,
        }
    )

    lower.link_external_item("external.cu")

    assert lower.targetoptions["lto"] is False
    assert lower._linker_config["lto"] is False
    assert linkers[-1].link_items == ["external.cu"]


def test_lowering_enables_implicit_lto_for_lineinfo_discovered_external_link_item(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": True,
            "_lto_explicit": False,
            "_output_explicit": False,
        }
    )

    lower.link_external_item("external.ltoir")

    assert lower.targetoptions["lto"] is True
    assert lower._linker_config["lto"] is True
    assert linkers[-1].link_items == ["external.ltoir"]


def test_lowering_preserves_explicit_output_ltoir_for_callback_link_item(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    callback_item = _CallbackLinkItem()
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": True,
            "output": "ltoir",
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": True,
        }
    )

    lower.link_external_item(callback_item)

    assert lower.targetoptions["lto"] is True
    assert lower.targetoptions["output"] == "ltoir"
    assert lower._linker_config["lto"] is True
    assert linkers[-1].link_items == [callback_item]


def test_lowering_preserves_callback_ptx_for_later_external_link_item(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    callback_item = _CallbackLinkItem()
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": False,
        }
    )

    lower.link_external_item(callback_item)
    lower.link_external_item("external.cu")

    assert lower.targetoptions["lto"] is False
    assert lower.targetoptions["output"] == "ptx"
    assert lower._linker_config["lto"] is False
    assert linkers[-1].link_items == [callback_item, "external.cu"]


def test_cache_key_uses_lto_intent_for_implicit_lto():
    targetoptions = {
        "lto": False,
        "output": "ptx",
        "_lto_explicit": False,
        "_output_explicit": False,
    }
    key_before_lowering = _targetoptions_cache_key(targetoptions)

    targetoptions["lto"] = True

    assert _targetoptions_cache_key(targetoptions) == key_before_lowering


def test_cache_key_keeps_explicit_lto_false_distinct():
    explicit_false = {
        "lto": False,
        "output": "ptx",
        "_lto_explicit": True,
        "_output_explicit": False,
    }
    implicit = {
        "lto": False,
        "output": "ptx",
        "_lto_explicit": False,
        "_output_explicit": False,
    }

    assert _targetoptions_cache_key(explicit_false) != _targetoptions_cache_key(implicit)


class _DiagnosticLinker:
    def __init__(self):
        self.ltoirs = []
        self.link_items = []

    def add_ltoir(self, ltoir, name=""):
        self.ltoirs.append((ltoir, name))

    def add_file_guess_ext(self, link_item, ignore_nonlto=False):
        self.link_items.append((link_item, ignore_nonlto))

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


def test_get_lto_ptx_replays_discovered_link_items_for_diagnostic_lto(monkeypatch):
    from numba_cuda_mlir import mlir_optimization

    linker = _SourceLinker(lto=False)
    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {"chip": "sm_90", "link": ["decorator.cu"]},
            "linker": linker,
            "external_link_items": ["decorator.cu", "hidden.cu"],
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
        ("decorator.cu", True),
        ("hidden.cu", True),
    ]


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
    assert linker.diag_linker.link_items == [("decorator.cu", True)]


def test_get_lto_ptx_replays_link_items_when_creating_diagnostic_linker(monkeypatch):
    from numba_cuda_mlir import mlir_optimization
    from numba_cuda_mlir import linker as linker_mod

    created_linkers = []

    class CreatedLinker(_SourceLinker):
        def __init__(self, **kwargs):
            super().__init__(lto=kwargs.get("lto", False))
            created_linkers.append(self)

    cres = types.SimpleNamespace(
        metadata={
            "targetoptions": {"chip": "sm_90", "link": ["decorator.cu"]},
        }
    )
    monkeypatch.setattr(
        mlir_optimization,
        "_compile_ltoir_for_inspection",
        lambda cres, target_options: b"kernel-ltoir",
    )
    monkeypatch.setattr(linker_mod, "Linker", CreatedLinker)

    assert mlir_optimization.get_lto_ptx(cres) == "linked-lto-ptx"
    assert created_linkers[-1].diag_linker.link_items == [("decorator.cu", True)]


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
