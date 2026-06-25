# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Caching support for numba_cuda_mlir.

This module provides cache implementations for MLIRDispatcher, allowing
compiled CUDA kernels to be cached to disk and reloaded in subsequent runs.
"""

import hashlib
import itertools
import os
import threading
import uuid
from collections import OrderedDict

from numba_cuda_mlir.numba_cuda import typing
from numba_cuda_mlir.numba_cuda.core.caching import Cache, CacheImpl, NullCache

_LINK_PATH_DIGEST_CACHE = OrderedDict()
_LINK_PATH_DIGEST_CACHE_LOCK = threading.Lock()
_LINK_PATH_DIGEST_CACHE_MAX_ENTRIES = 128
_LINK_DATA_DIGEST_CACHE = OrderedDict()
_LINK_DATA_DIGEST_CACHE_LOCK = threading.Lock()
_LINK_DATA_DIGEST_CACHE_MAX_ENTRIES = 128
_LINK_DATA_HASH_CHUNK_SIZE = 1024 * 1024
_OPAQUE_LINK_DATA_KEY_COUNTER = itertools.count()
_OPAQUE_LINK_DATA_KEY_LOCK = threading.Lock()
_OPAQUE_LINK_DATA_KEYS = OrderedDict()
_OPAQUE_LINK_DATA_KEY_MAX_ENTRIES = 128
_MISSING_LINK_PATH_KEY_COUNTER = itertools.count()
_MISSING_LINK_PATH_KEY_LOCK = threading.Lock()
_MISSING_LINK_PATH_KEYS = OrderedDict()
_MISSING_LINK_PATH_KEY_MAX_ENTRIES = 128
_NONPERSISTENT_LINK_KEY_SALT = uuid.uuid4().hex


class CachedCompileResult:
    """
    A minimal compile result reconstructed from cached data.

    This provides just enough interface to be used by MLIRDispatcher
    without requiring the full compilation infrastructure.
    """

    def __init__(self, signature, metadata):
        self.signature = signature
        self.metadata = metadata
        # Provide a codegen attribute for cache key computation
        self.codegen = _CachedCodegen()
        # entry_point is needed for recompile() to work
        self.entry_point = metadata.get("func_name")


class _CachedCodegen:
    """Minimal codegen for cache key computation."""

    def magic_tuple(self):
        # Return a consistent tuple for MLIR-compiled code
        from numba_cuda_mlir.tools import (
            get_gpu_compute_capability,
            get_cuda_runtime_version,
        )

        cc = get_gpu_compute_capability(tuple)
        return (get_cuda_runtime_version(), cc)


class MLIRCacheImpl(CacheImpl):
    """
    Cache implementation for MLIR-compiled CUDA kernels.

    Handles serialization and deserialization of compile results.
    """

    def reduce(self, cres):
        """Serialize a compile result for caching."""
        metadata = cres.metadata
        targetoptions = metadata.get("targetoptions", {})
        optimization_complete = metadata.get("_optimization_complete")
        if optimization_complete is None:
            optimization_complete = (
                "cubin" in metadata and targetoptions.get("_compile_output") is None
            )
        return {
            "signature_args": cres.signature.args,
            "signature_return_type": cres.signature.return_type,
            "cubin": metadata.get("cubin"),
            "ptx": metadata.get("ptx"),
            "ltoir": metadata.get("ltoir"),
            "func_name": metadata.get("func_name"),
            "mlir_module_optimized": metadata.get("mlir_module_optimized"),
            "needs_nrt": metadata.get("needs_nrt"),
            "nrt_inline": metadata.get("nrt_inline"),
            "targetoptions": targetoptions,
            "gpu_target": metadata.get("gpu_target"),
            "_optimization_complete": optimization_complete,
        }

    def rebuild(self, target_context, payload):
        """Deserialize a compile result from cache."""
        signature_args = payload["signature_args"]
        signature_return_type = payload["signature_return_type"]
        cubin = payload["cubin"]
        ptx = payload["ptx"]
        ltoir = payload.get("ltoir")
        func_name = payload["func_name"]
        mlir_module_optimized = payload.get("mlir_module_optimized")
        needs_nrt = payload.get("needs_nrt")
        nrt_inline = payload.get("nrt_inline")
        targetoptions = payload.get("targetoptions", {})
        gpu_target = payload.get("gpu_target")
        optimization_complete = payload.get(
            "_optimization_complete",
            cubin is not None and targetoptions.get("_compile_output") is None,
        )

        signature = typing.signature(signature_return_type, *signature_args)

        return CachedCompileResult(
            signature=signature,
            metadata={
                "cubin": cubin,
                "ptx": ptx,
                "ltoir": ltoir,
                "func_name": func_name,
                "mlir_module_optimized": mlir_module_optimized,
                "needs_nrt": needs_nrt,
                "nrt_inline": nrt_inline,
                "targetoptions": targetoptions,
                "gpu_target": gpu_target,
                "_optimization_complete": optimization_complete,
            },
        )

    def check_cachable(self, cres):
        """Check if a compile result can be cached."""
        metadata = cres.metadata
        targetoptions = metadata.get("targetoptions", {})
        link = targetoptions.get("link", [])
        discovered_link = metadata.get("linked_external_link_items") or metadata.get(
            "external_link_items"
        )
        if link:
            raise RuntimeError("Cannot pickle CUDACodeLibrary with linking files")
        if discovered_link:
            return False
        if targetoptions.get("extensions"):
            return False
        return True


def _target_option_key(targetoptions):
    from numba_cuda_mlir.tools import resolve_gpu_target

    gpu_target = resolve_gpu_target(targetoptions)
    compile_output = targetoptions.get("_compile_output")
    lto_explicit = bool(targetoptions.get("_lto_explicit", False))
    if lto_explicit:
        lto_key = targetoptions.get("lto")
    elif compile_output == "ltoir":
        lto_key = True
    else:
        lto_key = None
    link_items = targetoptions.get("link") or ()
    return (
        ("_compile_output", _cache_option_value(compile_output)),
        ("lto_explicit", _cache_option_value(lto_explicit)),
        ("lto", _cache_option_value(lto_key)),
        ("abi", _cache_option_value(targetoptions.get("abi", "numba"))),
        ("abi_info", _cache_option_value(targetoptions.get("abi_info"))),
        ("capi", _cache_option_value(targetoptions.get("capi", False))),
        ("chip", _cache_option_value(gpu_target["chip"])),
        ("device", _cache_option_value(targetoptions.get("device", False))),
        ("debug", _cache_option_value(targetoptions.get("debug", False))),
        ("extensions", _extensions_option_key(targetoptions.get("extensions"))),
        ("fastmath", _cache_option_value(targetoptions.get("fastmath", False))),
        ("features", _cache_option_value(targetoptions.get("features"))),
        ("inline", _cache_option_value(targetoptions.get("inline", "always"))),
        ("lineinfo", _cache_option_value(targetoptions.get("lineinfo", False))),
        ("launch_bounds", _cache_option_value(targetoptions.get("launch_bounds"))),
        ("max_registers", _cache_option_value(targetoptions.get("max_registers"))),
        ("nrt", _cache_option_value(targetoptions.get("_nrt", False))),
        (
            "opt_level",
            _cache_option_value(targetoptions.get("opt_level", targetoptions.get("opt"))),
        ),
        ("ptxas_options", _cache_option_value(targetoptions.get("ptxas_options"))),
        ("link", tuple(_link_item_key(link_item) for link_item in link_items)),
    )


def _extensions_option_key(extensions):
    return tuple(
        _nonpersistent_link_object_key("extension", extension) for extension in extensions or ()
    )


def _normalize_option_cache_value(value, unhashable_key, callable_key=None):
    if callable_key is not None and callable(value):
        return callable_key(value)
    if isinstance(value, (list, tuple)):
        return tuple(
            _normalize_option_cache_value(item, unhashable_key, callable_key) for item in value
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    (
                        _normalize_option_cache_value(key, unhashable_key, callable_key),
                        _normalize_option_cache_value(item, unhashable_key, callable_key),
                    )
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted(
                (
                    _normalize_option_cache_value(item, unhashable_key, callable_key)
                    for item in value
                ),
                key=repr,
            )
        )
    try:
        hash(value)
    except TypeError:
        return unhashable_key(value)
    return value


def _cache_option_value(value):
    return _normalize_option_cache_value(
        value,
        lambda item: _nonpersistent_link_object_key("target-option", item),
        lambda item: _nonpersistent_link_object_key("target-option", item),
    )


def _bytes_for_hash(data):
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if not isinstance(data, bytes):
        raise TypeError(f"Cannot derive stable cache key for link data of type {type(data)!r}")
    return data


def _hash_bytes(data):
    cached_digest = _immutable_link_data_digest(data)
    if cached_digest is not None:
        return cached_digest
    return hashlib.sha256(_bytes_for_hash(data)).hexdigest()


def _immutable_link_data_digest(data):
    if not isinstance(data, (bytes, str)):
        return None

    # Keep a bounded strong reference so id() reuse cannot alias digests; large
    # payloads may remain resident until LRU eviction.
    data_id = id(data)
    key = (type(data), data_id)
    with _LINK_DATA_DIGEST_CACHE_LOCK:
        entry = _LINK_DATA_DIGEST_CACHE.get(key)
        if entry is not None and entry[0] is data:
            _LINK_DATA_DIGEST_CACHE.move_to_end(key)
            return entry[1]

    digest = hashlib.sha256(_bytes_for_hash(data)).hexdigest()

    with _LINK_DATA_DIGEST_CACHE_LOCK:
        entry = _LINK_DATA_DIGEST_CACHE.get(key)
        if entry is not None and entry[0] is data:
            _LINK_DATA_DIGEST_CACHE.move_to_end(key)
            return entry[1]
        _LINK_DATA_DIGEST_CACHE[key] = (data, digest)
        while len(_LINK_DATA_DIGEST_CACHE) > _LINK_DATA_DIGEST_CACHE_MAX_ENTRIES:
            _LINK_DATA_DIGEST_CACHE.popitem(last=False)
    return digest


def _hash_file_obj(file_obj):
    hasher = hashlib.sha256()
    while True:
        try:
            chunk = file_obj.read(_LINK_DATA_HASH_CHUNK_SIZE)
        except TypeError:
            chunk = file_obj.read()
            if chunk:
                hasher.update(_bytes_for_hash(chunk))
            break
        if not chunk:
            break
        hasher.update(_bytes_for_hash(chunk))
    return hasher.hexdigest()


def _hash_path(path):
    try:
        stat = os.stat(path)
    except OSError:
        return None

    cache_key = (
        path,
        getattr(stat, "st_dev", None),
        getattr(stat, "st_ino", None),
        stat.st_mtime_ns,
        getattr(stat, "st_ctime_ns", None),
        stat.st_size,
    )
    with _LINK_PATH_DIGEST_CACHE_LOCK:
        digest = _LINK_PATH_DIGEST_CACHE.get(cache_key)
        if digest is not None:
            _LINK_PATH_DIGEST_CACHE.move_to_end(cache_key)
            return digest

    try:
        with open(path, "rb") as f:
            digest = _hash_file_obj(f)
    except OSError:
        return None

    with _LINK_PATH_DIGEST_CACHE_LOCK:
        cached_digest = _LINK_PATH_DIGEST_CACHE.get(cache_key)
        if cached_digest is not None:
            _LINK_PATH_DIGEST_CACHE.move_to_end(cache_key)
            return cached_digest
        for stale_key in tuple(_LINK_PATH_DIGEST_CACHE):
            if stale_key[0] == path and stale_key != cache_key:
                _LINK_PATH_DIGEST_CACHE.pop(stale_key, None)
        _LINK_PATH_DIGEST_CACHE[cache_key] = digest
        while len(_LINK_PATH_DIGEST_CACHE) > _LINK_PATH_DIGEST_CACHE_MAX_ENTRIES:
            _LINK_PATH_DIGEST_CACHE.popitem(last=False)
    return digest


def _link_item_data_key(data):
    if data is None:
        return ("data", None)
    if hasattr(data, "read"):
        file_obj = data
        try:
            pos = file_obj.tell()
        except (AttributeError, OSError, ValueError):
            return _opaque_link_item_data_key(data)
        try:
            # Computing a content key for file-like objects requires a transient
            # seek/read/restore sequence. Callers must not concurrently consume
            # the same object while compiling.
            file_obj.seek(0)
            digest = _hash_file_obj(file_obj)
        except (AttributeError, OSError, ValueError):
            return _opaque_link_item_data_key(file_obj)
        finally:
            try:
                file_obj.seek(pos)
            except (AttributeError, OSError, ValueError):
                pass
        return ("data", digest)
    try:
        return ("data", _hash_bytes(data))
    except TypeError:
        return _opaque_link_item_data_key(data)


def _opaque_link_item_data_key(data):
    return _nonpersistent_link_object_key("opaque", data)


def _link_item_callback_key(callback):
    if callback is None:
        return None
    return _nonpersistent_link_object_key("callback", callback)


def _nonpersistent_link_object_key(kind, data):
    # Opaque payloads are not safely content-addressable. Assign a stable
    # per-object nonce and a process salt so persistent caches cannot reuse
    # them across processes. Holding the object prevents id reuse while this
    # bounded entry remains cached.
    data_id = id(data)
    key = (kind, data_id)
    with _OPAQUE_LINK_DATA_KEY_LOCK:
        entry = _OPAQUE_LINK_DATA_KEYS.get(key)
        if entry is not None and entry[0] is data:
            _OPAQUE_LINK_DATA_KEYS.move_to_end(key)
            nonce = entry[1]
        else:
            nonce = next(_OPAQUE_LINK_DATA_KEY_COUNTER)
            _OPAQUE_LINK_DATA_KEYS[key] = (data, nonce)
            while len(_OPAQUE_LINK_DATA_KEYS) > _OPAQUE_LINK_DATA_KEY_MAX_ENTRIES:
                _OPAQUE_LINK_DATA_KEYS.popitem(last=False)
    return (kind, type(data).__name__, _NONPERSISTENT_LINK_KEY_SALT, nonce)


def _missing_link_path_key(path):
    with _MISSING_LINK_PATH_KEY_LOCK:
        nonce = _MISSING_LINK_PATH_KEYS.get(path)
        if nonce is not None:
            _MISSING_LINK_PATH_KEYS.move_to_end(path)
        else:
            nonce = next(_MISSING_LINK_PATH_KEY_COUNTER)
            _MISSING_LINK_PATH_KEYS[path] = nonce
            while len(_MISSING_LINK_PATH_KEYS) > _MISSING_LINK_PATH_KEY_MAX_ENTRIES:
                _MISSING_LINK_PATH_KEYS.popitem(last=False)
    return ("missing-path", os.path.basename(path), path, _NONPERSISTENT_LINK_KEY_SALT, nonce)


def _link_item_key(link_item):
    if isinstance(link_item, str):
        path = os.path.abspath(link_item)
        digest = _hash_path(path)
        if digest is None:
            # Keep missing paths process-local, but stable enough for the
            # cache load/save keys within one compile attempt to agree.
            return _missing_link_path_key(path)
        return ("path", os.path.basename(path), path, digest)

    data = getattr(link_item, "data", None)
    return (
        type(link_item).__name__,
        getattr(link_item, "name", None),
        _link_item_data_key(data),
        _link_item_callback_key(getattr(link_item, "setup_callback", None)),
        _link_item_callback_key(getattr(link_item, "teardown_callback", None)),
        bool(getattr(link_item, "nrt", False)),
    )


class MLIRCache(Cache):
    """
    Cache for MLIR-compiled CUDA kernels.

    Uses the standard numba caching infrastructure with MLIR-specific
    serialization.
    """

    _impl_class = MLIRCacheImpl

    def __init__(self, py_func, targetoptions=None):
        self._targetoptions = targetoptions if targetoptions is not None else {}
        super().__init__(py_func)
        self._cache_file._version = (self._cache_file._version, "mlir-target-options-v2")

    def _index_key(self, sig, codegen):
        key = super()._index_key(sig, codegen)
        # Link items that cannot be content-addressed contribute process-salted
        # key components. That deliberately prevents unsafe cross-process cache
        # reuse for callbacks, opaque payloads, and missing paths.
        option_key = _target_option_key(self._targetoptions)
        return (*key, option_key)


# Re-export NullCache for convenience
__all__ = ["MLIRCache", "MLIRCacheImpl", "NullCache", "CachedCompileResult"]
