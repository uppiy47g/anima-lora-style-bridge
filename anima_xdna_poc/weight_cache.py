"""Verified local cache for XDNA-ready Anima weight layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import mmap
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Iterator, Optional
import uuid
import warnings

import torch
from safetensors import safe_open

from .block_checkpoint import (
    AnimaBlockConfig,
    AnimaBlockWeights,
    LINEAR_SHAPES,
    NORM_SHAPES,
)
from .checkpoint_schema import (
    canonical_block_fingerprint,
    canonical_keys,
    detect_checkpoint_schema,
)
from .errors import UnsupportedTensor
from .linear import BLOCK_K, BLOCK_N, PreparedLinear, _aligned


CACHE_ABI = 2
LAYOUT_VERSION = "aie2p-bf16-kn-256x256-k2048-v1"
MODEL_FAMILY = "anima-cosmos-predict2"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "anima-xdna" / "weights"


@dataclass(frozen=True)
class CacheNamespace:
    base_model_fingerprint: str
    adapter_fingerprints: tuple[str, ...] = ()
    merge_strength: Optional[float] = None
    quantization_scheme: str = "none"
    layout_version: str = LAYOUT_VERSION


@dataclass(frozen=True)
class CacheTimings:
    fingerprint_ms: float = 0.0
    pack_ms: float = 0.0
    disk_read_ms: float = 0.0
    verify_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class CacheStatus:
    key: str
    path: Path
    hit: bool
    reason: str
    tensor_count: int
    logical_source_bytes: int
    packed_bytes: int
    padding_bytes: int
    timings: CacheTimings


class CacheIntegrityError(UnsupportedTensor):
    """A named cache entry exists but failed manifest or payload verification."""


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _first_package_version(*names: str) -> str:
    for name in names:
        version = _package_version(name)
        if version != "unavailable":
            return version
    return "unavailable"


def tool_identity(target: str = "npu2") -> dict[str, str]:
    return {
        "triton_xdna": _package_version("triton-xdna"),
        "mlir_aie": _first_package_version("mlir-aie", "mlir-aie-no-rtti"),
        "mlir_air": _package_version("mlir-air"),
        "llvm_aie": _package_version("llvm-aie"),
        "xrt_sdk": Path(os.environ.get("XRT_DEV_DIR", "unconfigured")).name,
        "npu_target": target,
        "cache_abi": str(CACHE_ABI),
    }


def _read_header(path: Path) -> tuple[bytes, dict[str, Any], int]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise UnsupportedTensor(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise UnsupportedTensor(f"truncated safetensors header: {path}")
    try:
        header = json.loads(header_bytes)
    except Exception as error:
        raise UnsupportedTensor(f"invalid safetensors header {path}: {error}") from error
    return header_bytes, header, 8 + header_length


def _expected_keys(blocks: range) -> list[str]:
    return canonical_keys(blocks)


def _source_files(
    checkpoint: Path,
) -> tuple[list[Path], Optional[dict[str, str]]]:
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.suffix.lower() != ".json":
        return [checkpoint], None
    try:
        index = json.loads(checkpoint.read_text(encoding="utf-8"))
        weight_map = {
            key: str(Path(name)).replace("\\", "/")
            for key, name in index["weight_map"].items()
        }
    except Exception as error:
        raise UnsupportedTensor(
            f"cannot read safetensors shard index {checkpoint}: {error}"
        ) from error
    files = sorted({checkpoint.parent / name for name in weight_map.values()})
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise UnsupportedTensor(
            "safetensors shard files are missing: " + ", ".join(missing)
        )
    return files, weight_map


def _source_name(checkpoint: Path, source_file: Path) -> str:
    if checkpoint.suffix.lower() == ".json":
        return str(source_file.relative_to(checkpoint.parent)).replace("\\", "/")
    return source_file.name


def fingerprint_source(
    checkpoint: Path,
    blocks: range,
) -> tuple[dict[str, Any], CacheTimings]:
    started = time.perf_counter()
    checkpoint = Path(checkpoint).resolve()
    source_files, weight_map = _source_files(checkpoint)
    keys = _expected_keys(blocks)
    file_metadata = {}
    headers = {}
    for source_file in source_files:
        source_name = _source_name(checkpoint, source_file)
        header_bytes, header, data_start = _read_header(source_file)
        headers[source_name] = (header, data_start)
        file_metadata[source_name] = {
            "name": source_name,
            "size": source_file.stat().st_size,
            "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        }
    if weight_map is None:
        source_name = _source_name(checkpoint, source_files[0])
        weight_map = {key: source_name for key in headers[source_name][0]}
    schema = detect_checkpoint_schema(weight_map.keys(), blocks)
    canonical_to_source = schema.canonical_to_source
    missing = [
        key
        for key in keys
        if canonical_to_source[key] not in weight_map
        or weight_map[canonical_to_source[key]] not in headers
        or canonical_to_source[key]
        not in headers[weight_map[canonical_to_source[key]]][0]
    ]
    if missing:
        raise UnsupportedTensor(
            "checkpoint is missing cache source tensors: " + ", ".join(missing)
        )
    tensors = []
    for source_file in source_files:
        source_name = _source_name(checkpoint, source_file)
        shard_keys = [
            key
            for key in keys
            if weight_map[canonical_to_source[key]] == source_name
        ]
        header, data_start = headers[source_name]
        with source_file.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                for key in shard_keys:
                    source_key = canonical_to_source[key]
                    metadata = header[source_key]
                    begin, end = metadata["data_offsets"]
                    view = memoryview(mapped)[data_start + begin : data_start + end]
                    digest = hashlib.sha256(view).hexdigest()
                    sample_size = min(4096, len(view))
                    sample_offsets = sorted(
                        {
                            0,
                            max(0, (len(view) - sample_size) // 2),
                            max(0, len(view) - sample_size),
                        }
                    )
                    sample_digest = hashlib.sha256()
                    for sample_offset in sample_offsets:
                        sample_digest.update(
                            view[sample_offset : sample_offset + sample_size]
                        )
                    del view
                    tensors.append(
                        {
                            "key": key,
                            "source_key": source_key,
                            "shard": source_name,
                            "dtype": metadata["dtype"],
                            "shape": metadata["shape"],
                            "byte_length": end - begin,
                            "sha256": digest,
                            "sample_size": sample_size,
                            "sample_offsets": sample_offsets,
                            "sample_sha256": sample_digest.hexdigest(),
                        }
                    )
    identity = {
        "schema": schema.name,
        "index": (
            {
                "name": checkpoint.name,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            if checkpoint.suffix.lower() == ".json"
            else None
        ),
        "files": [
            file_metadata[_source_name(checkpoint, path)]
            for path in source_files
        ],
        "tensors": sorted(tensors, key=lambda item: item["key"]),
    }
    identity["block_fingerprint"] = canonical_block_fingerprint(tensors)
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return identity, CacheTimings(
        fingerprint_ms=(time.perf_counter() - started) * 1000
    )


def _file_guard(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "device": stat.st_dev,
        "file_id": stat.st_ino,
        "modified_ns": stat.st_mtime_ns,
        "changed_ns": stat.st_ctime_ns,
    }


def _source_guard(checkpoint: Path) -> dict[str, Any]:
    files, _ = _source_files(checkpoint)
    return {
        "index": (
            {
                "name": checkpoint.name,
                **_file_guard(checkpoint),
            }
            if checkpoint.suffix.lower() == ".json"
            else None
        ),
        "files": {
            _source_name(checkpoint, path): _file_guard(path)
            for path in files
        },
    }


def _quick_validate_source(
    checkpoint: Path,
    manifest: dict[str, Any],
) -> tuple[bool, float]:
    started = time.perf_counter()
    if manifest.get("source_guard") != _source_guard(checkpoint):
        return False, (time.perf_counter() - started) * 1000
    source_identity = manifest["descriptor"]["source_identity"]
    files, _ = _source_files(checkpoint)
    source_by_name = {
        _source_name(checkpoint, path): path
        for path in files
    }
    tensors_by_shard = {}
    for tensor in source_identity["tensors"]:
        shard = tensor.get("shard")
        if shard is None:
            return False, (time.perf_counter() - started) * 1000
        tensors_by_shard.setdefault(shard, []).append(tensor)
    for file_identity in source_identity["files"]:
        source_file = source_by_name.get(file_identity["name"])
        if source_file is None:
            return False, (time.perf_counter() - started) * 1000
        header_bytes, header, data_start = _read_header(source_file)
        if (
            hashlib.sha256(header_bytes).hexdigest()
            != file_identity["header_sha256"]
        ):
            return False, (time.perf_counter() - started) * 1000
        with source_file.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                for tensor in tensors_by_shard.get(
                    _source_name(checkpoint, source_file), []
                ):
                    metadata = header.get(tensor["source_key"])
                    if metadata is None:
                        return False, (time.perf_counter() - started) * 1000
                    begin, end = metadata["data_offsets"]
                    view = memoryview(mapped)[data_start + begin : data_start + end]
                    digest = hashlib.sha256()
                    for offset in tensor["sample_offsets"]:
                        digest.update(
                            view[offset : offset + tensor["sample_size"]]
                        )
                    del view
                    if digest.hexdigest() != tensor["sample_sha256"]:
                        return False, (time.perf_counter() - started) * 1000
    if source_identity.get("index") is not None:
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != source_identity["index"]["sha256"]
        ):
            return False, (time.perf_counter() - started) * 1000
    return True, (time.perf_counter() - started) * 1000


def _layout_shape(out_features: int, in_features: int) -> tuple[int, int]:
    padded_in = max(_aligned(in_features, BLOCK_K), 256)
    padded_out = _aligned(out_features, BLOCK_N)
    if padded_out == BLOCK_N and padded_in > 256:
        padded_out = 2 * BLOCK_N
    return padded_in, padded_out


def _cache_descriptor(
    source: dict[str, Any],
    config: AnimaBlockConfig,
    blocks: range,
    namespace: CacheNamespace,
) -> dict[str, Any]:
    mapping = [
        {
            "canonical_key": tensor["key"],
            "source_key": tensor["source_key"],
            "shard": tensor["shard"],
        }
        for tensor in source["tensors"]
    ]
    return {
        "schema_version": CACHE_ABI,
        "model_family": MODEL_FAMILY,
        "architecture": asdict(config),
        "block_range": [blocks.start, blocks.stop],
        "weight_key_mapping": mapping,
        "source_identity": source,
        "namespace": asdict(namespace),
        "layout": {
            "version": LAYOUT_VERSION,
            "source_order": "out_in",
            "packed_order": "k_n",
            "packed_dtype": "BF16",
            "k_chunk": 2048,
            "block_k": BLOCK_K,
            "block_n": BLOCK_N,
        },
        "tools": tool_identity(),
    }


def _descriptor_key(descriptor: dict[str, Any]) -> str:
    canonical = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class _EntryLock:
    def __init__(self, path: Path, timeout: float = 300.0):
        self.path = path
        self.timeout = timeout
        self.acquired = False

    def __enter__(self):
        started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(descriptor, f"{os.getpid()}\n".encode())
                os.close(descriptor)
                self.acquired = True
                return self
            except FileExistsError:
                if time.monotonic() - started >= self.timeout:
                    raise CacheIntegrityError(
                        f"timed out waiting for cache lock: {self.path}"
                    )
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.acquired:
            self.path.unlink(missing_ok=True)


class PackedWeightCache:
    """Build, verify, mmap, and exactly prune one packed-weight entry."""

    def __init__(
        self,
        checkpoint: Path,
        config: AnimaBlockConfig,
        start_block: int = 0,
        end_block: int = 28,
        cache_dir: Optional[Path] = None,
        enabled: bool = True,
    ):
        if not 0 <= start_block < end_block <= 28:
            raise UnsupportedTensor(
                f"cache block range must satisfy 0 <= start < end <= 28, "
                f"got [{start_block}, {end_block})"
            )
        self.checkpoint = Path(checkpoint)
        self.config = config
        self.blocks = range(start_block, end_block)
        self.root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_ROOT
        self.enabled = enabled
        self.manifest: Optional[dict[str, Any]] = None
        self.status: Optional[CacheStatus] = None
        self._handle = None
        self._mapped = None

    def identify(self) -> tuple[str, dict[str, Any], CacheTimings]:
        source, timings = fingerprint_source(self.checkpoint, self.blocks)
        namespace = CacheNamespace(source["fingerprint"])
        descriptor = json.loads(
            json.dumps(
                _cache_descriptor(source, self.config, self.blocks, namespace),
                sort_keys=True,
            )
        )
        return _descriptor_key(descriptor), descriptor, timings

    def entry_path(self, key: str) -> Path:
        return self.root / key

    @staticmethod
    def _read_manifest(entry: Path) -> dict[str, Any]:
        try:
            return json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        except Exception as error:
            raise CacheIntegrityError(
                f"cannot read cache manifest {entry}: {error}"
            ) from error

    @staticmethod
    def _verify_entry(
        entry: Path,
        key: str,
        descriptor: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        manifest = PackedWeightCache._read_manifest(entry)
        if manifest.get("cache_key") != key:
            raise CacheIntegrityError("cache manifest key mismatch")
        if manifest.get("descriptor") != descriptor:
            raise CacheIntegrityError("cache manifest descriptor mismatch")
        payload = entry / "weights.bin"
        expected_size = manifest.get("packed_bytes")
        if not payload.is_file() or payload.stat().st_size != expected_size:
            raise CacheIntegrityError(
                f"cache payload size mismatch: expected {expected_size}"
            )
        digest = hashlib.sha256()
        with payload.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest.get("payload_sha256"):
            raise CacheIntegrityError("cache payload SHA-256 mismatch")
        return manifest, (time.perf_counter() - started) * 1000

    def ensure(self, rebuild: bool = False) -> CacheStatus:
        total_started = time.perf_counter()
        if not self.enabled:
            self.status = CacheStatus(
                "", self.root, False, "disabled", 0, 0, 0, 0, CacheTimings()
            )
            return self.status
        candidate = None if rebuild else self._candidate_entry()
        if candidate is not None:
            candidate_manifest = self._read_manifest(candidate)
            with ThreadPoolExecutor(max_workers=2) as executor:
                verify_future = executor.submit(
                    self._verify_payload_only, candidate
                )
                source_future = executor.submit(
                    _quick_validate_source,
                    self.checkpoint,
                    candidate_manifest,
                )
                verified_manifest, verify_ms = verify_future.result()
                source_valid, fingerprint_ms = source_future.result()
            if (
                source_valid
                and verified_manifest == candidate_manifest
                and candidate_manifest.get("cache_key") == candidate.name
            ):
                self.manifest = candidate_manifest
                self.status = self._status(
                    candidate_manifest,
                    True,
                    "verified hit",
                    CacheTimings(fingerprint_ms=fingerprint_ms),
                    verify_ms=verify_ms,
                    total_ms=(time.perf_counter() - total_started) * 1000,
                )
                return self.status
            if (
                source_valid
                and candidate_manifest.get("cache_key") != candidate.name
            ):
                raise CacheIntegrityError("cache manifest key mismatch")
            key, descriptor, timings = self.identify()
        else:
            key, descriptor, timings = self.identify()
        entry = self.entry_path(key)
        if entry.exists() and not rebuild:
            manifest, verify_ms = self._verify_entry(entry, key, descriptor)
            self.manifest = manifest
            self.status = self._status(
                manifest,
                True,
                "verified hit",
                timings,
                verify_ms=verify_ms,
                total_ms=(time.perf_counter() - total_started) * 1000,
            )
            return self.status
        lock = self.root / f"{key}.lock"
        with _EntryLock(lock):
            if entry.exists() and not rebuild:
                manifest, verify_ms = self._verify_entry(entry, key, descriptor)
                self.manifest = manifest
                self.status = self._status(
                    manifest,
                    True,
                    "concurrent builder won",
                    timings,
                    verify_ms=verify_ms,
                    total_ms=(time.perf_counter() - total_started) * 1000,
                )
                return self.status
            if entry.exists():
                shutil.rmtree(entry)
            manifest, pack_ms = self._build_atomic(entry, key, descriptor)
        self.manifest = manifest
        self.status = self._status(
            manifest,
            False,
            "explicit rebuild" if rebuild else "cache miss",
            timings,
            pack_ms=pack_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
        )
        return self.status

    def _candidate_entry(self) -> Optional[Path]:
        if not self.root.is_dir():
            return None
        source_files, _ = _source_files(self.checkpoint)
        current_files = []
        for source_file in source_files:
            source_name = _source_name(self.checkpoint, source_file)
            header_bytes, _, _ = _read_header(source_file)
            current_files.append(
                {
                    "name": source_name,
                    "size": source_file.stat().st_size,
                    "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                }
            )
        current_index = (
            {
                "name": self.checkpoint.name,
                "sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            }
            if self.checkpoint.suffix.lower() == ".json"
            else None
        )
        architecture = json.loads(json.dumps(asdict(self.config)))
        tools = tool_identity()
        candidates = []
        for manifest_path in self.root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                descriptor = manifest["descriptor"]
                source_identity = descriptor["source_identity"]
                if (
                    descriptor["architecture"] == architecture
                    and descriptor["block_range"]
                    == [self.blocks.start, self.blocks.stop]
                    and descriptor["tools"] == tools
                    and descriptor["layout"]["version"] == LAYOUT_VERSION
                    and source_identity["files"] == current_files
                    and source_identity.get("index") == current_index
                ):
                    candidates.append(manifest_path.parent)
            except (KeyError, OSError, ValueError, TypeError):
                continue
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _verify_payload_only(
        entry: Path,
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        manifest = PackedWeightCache._read_manifest(entry)
        payload = entry / "weights.bin"
        expected_size = manifest.get("packed_bytes")
        if not payload.is_file() or payload.stat().st_size != expected_size:
            raise CacheIntegrityError(
                f"cache payload size mismatch: expected {expected_size}"
            )
        digest = hashlib.sha256()
        with payload.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest.get("payload_sha256"):
            raise CacheIntegrityError("cache payload SHA-256 mismatch")
        return manifest, (time.perf_counter() - started) * 1000

    def _status(
        self,
        manifest: dict[str, Any],
        hit: bool,
        reason: str,
        timings: CacheTimings,
        pack_ms: float = 0.0,
        verify_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> CacheStatus:
        return CacheStatus(
            manifest["cache_key"],
            self.entry_path(manifest["cache_key"]),
            hit,
            reason,
            len(manifest["tensors"]),
            manifest["logical_source_bytes"],
            manifest["packed_bytes"],
            manifest["padding_bytes"],
            CacheTimings(
                timings.fingerprint_ms,
                pack_ms,
                0.0,
                verify_ms,
                total_ms,
            ),
        )

    def _build_atomic(
        self,
        entry: Path,
        key: str,
        descriptor: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".{key}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        records = []
        logical_bytes = 0
        payload_hash = hashlib.sha256()
        offset = 0
        try:
            dimensions = {
                "hidden": self.config.hidden_size,
                "head": self.config.head_dim,
                "context": self.config.context_dim,
                "adaln": self.config.adaln_dim,
                "modulation": 3 * self.config.hidden_size,
                "mlp": self.config.mlp_dim,
            }
            source_files, weight_map = _source_files(self.checkpoint)
            with ExitStack() as stack:
                sources = {
                    _source_name(self.checkpoint, path): stack.enter_context(
                        safe_open(path, framework="pt", device="cpu")
                    )
                    for path in source_files
                }
                if weight_map is None:
                    source_name = _source_name(
                        self.checkpoint, source_files[0]
                    )
                    weight_map = {
                        key: source_name
                        for key in sources[source_name].keys()
                    }
                tensor_sources = {
                    tensor["key"]: tensor
                    for tensor in descriptor["source_identity"]["tensors"]
                }
                with (temporary / "weights.bin").open("wb") as payload:
                    for block in self.blocks:
                        prefix = f"transformer_blocks.{block}."
                        for name in LINEAR_SHAPES:
                            canonical_key = prefix + name
                            source = tensor_sources[canonical_key]
                            source_key = source["source_key"]
                            tensor = sources[source["shard"]].get_tensor(source_key)
                            out_name, in_name = LINEAR_SHAPES[name]
                            expected = (dimensions[out_name], dimensions[in_name])
                            if tuple(tensor.shape) != expected:
                                raise UnsupportedTensor(
                                    f"{prefix + name!r} has shape "
                                    f"{tuple(tensor.shape)}, expected {expected}"
                                )
                            if tensor.dtype not in (
                                torch.bfloat16,
                                torch.float16,
                                torch.float32,
                            ):
                                raise UnsupportedTensor(
                                    f"{prefix + name!r} has unsupported dtype "
                                    f"{tensor.dtype}"
                                )
                            logical_bytes += tensor.numel() * tensor.element_size()
                            for start in range(0, tensor.shape[1], 2048):
                                end = min(start + 2048, tensor.shape[1])
                                packed_shape = _layout_shape(tensor.shape[0], end - start)
                                packed = torch.zeros(
                                    packed_shape, dtype=torch.bfloat16
                                )
                                packed[: end - start, : tensor.shape[0]] = (
                                    tensor[:, start:end]
                                    .to(torch.bfloat16)
                                    .T.contiguous()
                                )
                                raw = packed.view(torch.uint8).numpy().tobytes()
                                payload.write(raw)
                                payload_hash.update(raw)
                                records.append(
                                    {
                                        "id": f"{block}:{name}:k{start}",
                                        "canonical_key": canonical_key,
                                        "source_key": source_key,
                                        "source_dtype": str(tensor.dtype).removeprefix("torch."),
                                        "source_shape": list(tensor.shape),
                                        "logical_shape": [end - start, tensor.shape[0]],
                                        "packed_dtype": "bfloat16",
                                        "packed_shape": list(packed_shape),
                                        "offset": offset,
                                        "length": len(raw),
                                        "sha256": hashlib.sha256(raw).hexdigest(),
                                    }
                                )
                                offset += len(raw)
                        for name in NORM_SHAPES:
                            canonical_key = prefix + name
                            source = tensor_sources[canonical_key]
                            source_key = source["source_key"]
                            source_tensor = sources[source["shard"]].get_tensor(
                                source_key
                            )
                            expected = (dimensions[NORM_SHAPES[name]],)
                            if tuple(source_tensor.shape) != expected:
                                raise UnsupportedTensor(
                                    f"{prefix + name!r} has shape "
                                    f"{tuple(source_tensor.shape)}, expected {expected}"
                                )
                            if source_tensor.dtype not in (
                                torch.bfloat16,
                                torch.float16,
                                torch.float32,
                            ):
                                raise UnsupportedTensor(
                                    f"{prefix + name!r} has unsupported dtype "
                                    f"{source_tensor.dtype}"
                                )
                            logical_bytes += (
                                source_tensor.numel() * source_tensor.element_size()
                            )
                            tensor = source_tensor.to(torch.bfloat16).contiguous()
                            raw = tensor.view(torch.uint8).numpy().tobytes()
                            payload.write(raw)
                            payload_hash.update(raw)
                            records.append(
                                {
                                    "id": f"{block}:{name}",
                                    "canonical_key": canonical_key,
                                    "source_key": source_key,
                                    "source_dtype": str(source_tensor.dtype).removeprefix("torch."),
                                    "source_shape": list(source_tensor.shape),
                                    "logical_shape": list(tensor.shape),
                                    "packed_dtype": "bfloat16",
                                    "packed_shape": list(tensor.shape),
                                    "offset": offset,
                                    "length": len(raw),
                                    "sha256": hashlib.sha256(raw).hexdigest(),
                                }
                            )
                            offset += len(raw)
                    payload.flush()
                    os.fsync(payload.fileno())
            manifest = {
                "cache_key": key,
                "descriptor": descriptor,
                "source_guard": _source_guard(self.checkpoint),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tensor_count": len(records),
                "logical_source_bytes": logical_bytes,
                "packed_bytes": offset,
                "padding_bytes": offset - logical_bytes,
                "payload_sha256": payload_hash.hexdigest(),
                "tensors": records,
            }
            manifest_temp = temporary / "manifest.json.tmp"
            manifest_temp.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(manifest_temp, temporary / "manifest.json")
            os.replace(temporary, entry)
            return manifest, (time.perf_counter() - started) * 1000
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def open(self, rebuild: bool = False) -> CacheStatus:
        status = self.ensure(rebuild)
        if not self.enabled:
            return status
        payload = status.path / "weights.bin"
        started = time.perf_counter()
        self._handle = payload.open("rb")
        self._mapped = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        elapsed = (time.perf_counter() - started) * 1000
        self.status = CacheStatus(
            **{
                **asdict(status),
                "path": status.path,
                "timings": CacheTimings(
                    status.timings.fingerprint_ms,
                    status.timings.pack_ms,
                    elapsed,
                    status.timings.verify_ms,
                    status.timings.total_ms + elapsed,
                ),
            }
        )
        return self.status

    def close(self) -> None:
        if self._mapped is not None:
            self._mapped.close()
            self._mapped = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _record(self, identifier: str) -> dict[str, Any]:
        if self.manifest is None or self._mapped is None:
            raise RuntimeError("packed weight cache is not open")
        for record in self.manifest["tensors"]:
            if record["id"] == identifier:
                return record
        raise UnsupportedTensor(f"packed cache tensor not found: {identifier}")

    def tensor(self, identifier: str) -> torch.Tensor:
        record = self._record(identifier)
        begin = record["offset"]
        end = begin + record["length"]
        view = memoryview(self._mapped)[begin:end]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            tensor = torch.frombuffer(view, dtype=torch.uint8).view(torch.bfloat16)
        result = tensor.reshape(record["packed_shape"]).clone()
        del tensor
        del view
        return result

    def prepared(
        self,
        block: int,
        name: str,
        start: int,
        input_tensor: torch.Tensor,
        out_features: int,
    ) -> PreparedLinear:
        packed = self.tensor(f"{block}:{name}:k{start}")
        rows, in_features = input_tensor.shape
        padded_rows = _aligned(rows, 256)
        prepared_input = torch.zeros(
            padded_rows, packed.shape[0], dtype=torch.bfloat16
        )
        prepared_input[:rows, :in_features] = input_tensor.to(torch.bfloat16)
        return PreparedLinear(
            prepared_input,
            packed,
            rows,
            in_features,
            out_features,
        )

    def norm(self, block: int, name: str) -> torch.Tensor:
        return self.tensor(f"{block}:{name}").clone()

    def block_weights(self, block: int) -> AnimaBlockWeights:
        if block not in self.blocks:
            raise UnsupportedTensor(
                f"block {block} is outside cached range "
                f"[{self.blocks.start}, {self.blocks.stop})"
            )
        tensors = {}
        for name in LINEAR_SHAPES:
            record = self._record(f"{block}:{name}:k0")
            tensors[name] = torch.empty(
                record["source_shape"],
                dtype=getattr(torch, record["source_dtype"]),
                device="meta",
            )
        for name in NORM_SHAPES:
            tensors[name] = self.norm(block, name)
        return AnimaBlockWeights(f"transformer_blocks.{block}.", tensors)

    def prune(self) -> Path:
        key, descriptor, _ = self.identify()
        entry = self.entry_path(key)
        if not entry.exists():
            raise CacheIntegrityError(f"cache entry does not exist: {entry}")
        self._verify_entry(entry, key, descriptor)
        shutil.rmtree(entry)
        return entry


def list_entries(cache_dir: Optional[Path] = None) -> Iterator[dict[str, Any]]:
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_ROOT
    if not root.is_dir():
        return
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        yield {
            "key": manifest.get("cache_key"),
            "path": str(manifest_path.parent),
            "block_range": manifest.get("descriptor", {}).get("block_range"),
            "packed_bytes": manifest.get("packed_bytes"),
            "tensor_count": manifest.get("tensor_count"),
            "base_model_fingerprint": manifest.get("descriptor", {})
            .get("namespace", {})
            .get("base_model_fingerprint"),
        }


def prune_entry(key: str, cache_dir: Optional[Path] = None) -> Path:
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise CacheIntegrityError("cache entry key must be 64 lowercase hex characters")
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_ROOT
    entry = root / key
    if not entry.is_dir():
        raise CacheIntegrityError(f"cache entry does not exist: {entry}")
    manifest = PackedWeightCache._read_manifest(entry)
    PackedWeightCache._verify_entry(entry, key, manifest.get("descriptor"))
    shutil.rmtree(entry)
    return entry
