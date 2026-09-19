import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


_CAPTURES = {}


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _activation_rows(tensor, aggregation):
    values = tensor.detach().float()
    if values.ndim == 0:
        return None
    if values.ndim == 1:
        return values.reshape(1, -1)
    if values.ndim == 4:
        values = values.movedim(1, -1)
    values = values.reshape(-1, values.shape[-1])
    if aggregation == "mean":
        values = values.mean(dim=0, keepdim=True)
    return values.cpu()


class Capture:
    def __init__(self, root, pattern, aggregation, max_rows):
        self.root = root
        self.aggregation = aggregation
        self.values = defaultdict(list)
        self.row_counts = defaultdict(int)
        self.handles = []
        expression = re.compile(pattern)
        for name, module in root.named_modules():
            if not name or not expression.search(name):
                continue
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight) or weight.ndim != 2:
                continue
            self.handles.append(
                module.register_forward_hook(
                    self._hook(
                        name,
                        max_rows,
                        input_dim=weight.shape[1],
                        output_dim=weight.shape[0],
                    )
                )
            )
        if not self.handles:
            raise ValueError(f"No linear projection modules matched pattern: {pattern}")

    def _append(self, key, rows, max_rows):
        if rows is None:
            return
        remaining = max_rows - self.row_counts[key]
        if remaining <= 0:
            return
        if rows.shape[0] > remaining:
            indices = torch.linspace(
                0, rows.shape[0] - 1, remaining, dtype=torch.long
            )
            rows = rows[indices]
        self.values[key].append(rows.numpy())
        self.row_counts[key] += rows.shape[0]

    def _hook(self, name, max_rows, input_dim, output_dim):
        def capture(_module, arguments, output):
            input_rows = _activation_rows(
                _first_tensor(arguments), self.aggregation
            )
            output_rows = _activation_rows(_first_tensor(output), self.aggregation)
            if input_rows is not None and input_rows.shape[1] != input_dim:
                raise ValueError(
                    f"{name}.input has width {input_rows.shape[1]}, "
                    f"expected {input_dim}"
                )
            if output_rows is not None and output_rows.shape[1] != output_dim:
                raise ValueError(
                    f"{name}.output has width {output_rows.shape[1]}, "
                    f"expected {output_dim}"
                )
            self._append(f"{name}.input", input_rows, max_rows)
            self._append(f"{name}.output", output_rows, max_rows)

        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, path):
        arrays = {}
        for key, parts in self.values.items():
            if parts:
                arrays[key] = np.concatenate(parts, axis=0)
        if not arrays:
            raise ValueError("No activations were captured during sampling")
        count = len(arrays)
        arrays["__aggregation__"] = np.array(self.aggregation)
        np.savez_compressed(path, **arrays)
        return count


def _model_root(model):
    root = getattr(model, "model", model)
    if not hasattr(root, "named_modules"):
        raise TypeError("MODEL does not expose a torch module tree")
    return root


def _safe_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not name:
        raise ValueError("Session and phase names must contain a safe character")
    return name


class ActivationCaptureStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "session": ("STRING", {"default": "anima_v3"}),
                "phase": (
                    ["source_lora", "target_base"],
                    {"default": "source_lora"},
                ),
                "module_pattern": (
                    "STRING",
                    {"default": ".*attn.*(q|k|v|out).*"},
                ),
                "aggregation": (["mean", "tokens"], {"default": "mean"}),
                "max_rows_per_module": (
                    "INT",
                    {"default": 1024, "min": 2, "max": 1048576},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "capture_token")
    FUNCTION = "start"
    CATEGORY = "Anima Style Bridge"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return time.time_ns()

    def start(
        self, model, session, phase, module_pattern, aggregation, max_rows_per_module
    ):
        patched_model = model.clone() if hasattr(model, "clone") else model
        root = _model_root(patched_model)
        stale_tokens = [
            token
            for token, (capture, _session, _phase) in _CAPTURES.items()
            if capture.root is root
        ]
        for token in stale_tokens:
            capture, _session, _phase = _CAPTURES.pop(token)
            capture.close()
        capture = Capture(root, module_pattern, aggregation, max_rows_per_module)
        token = uuid.uuid4().hex
        _CAPTURES[token] = (capture, _safe_name(session), _safe_name(phase))
        return patched_model, token


class ActivationCaptureFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "capture_token": ("STRING", {"forceInput": True}),
                "append": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "activation_file")
    OUTPUT_NODE = True
    FUNCTION = "finish"
    CATEGORY = "Anima Style Bridge"

    def finish(self, images, capture_token, append):
        if capture_token not in _CAPTURES:
            raise ValueError("Unknown or already finalized activation capture")
        capture, session, phase = _CAPTURES.pop(capture_token)
        try:
            import folder_paths

            output_directory = (
                Path(folder_paths.get_output_directory()) / "activation_deltas"
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / f"{session}.{phase}.npz"
            if append and output_path.exists():
                temporary_path = output_path.with_suffix(".new.npz")
                try:
                    count = capture.save(temporary_path)
                    with np.load(output_path) as previous, np.load(
                        temporary_path
                    ) as current:
                        previous_keys = {
                            key for key in previous.files if not key.startswith("__")
                        }
                        current_keys = {
                            key for key in current.files if not key.startswith("__")
                        }
                        if previous_keys != current_keys:
                            raise ValueError(
                                "Cannot append captures with different module keys"
                            )
                        if (
                            str(previous["__aggregation__"].item())
                            != str(current["__aggregation__"].item())
                        ):
                            raise ValueError(
                                "Cannot append captures with different aggregation"
                            )
                        arrays = {
                            key: np.concatenate(
                                [previous[key], current[key]], axis=0
                            )
                            for key in previous_keys
                        }
                        arrays["__aggregation__"] = previous["__aggregation__"]
                    np.savez_compressed(output_path, **arrays)
                finally:
                    temporary_path.unlink(missing_ok=True)
            else:
                count = capture.save(output_path)
        finally:
            capture.close()
        return {
            "ui": {"text": [f"Saved {count} arrays to {output_path}"]},
            "result": (images, str(output_path)),
        }


class ActivationCaptureClear:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    FUNCTION = "clear"
    CATEGORY = "Anima Style Bridge"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return time.time_ns()

    def clear(self):
        count = len(_CAPTURES)
        for capture, _session, _phase in _CAPTURES.values():
            capture.close()
        _CAPTURES.clear()
        return (f"Cleared {count} activation captures",)


NODE_CLASS_MAPPINGS = {
    "AnimaActivationCaptureStart": ActivationCaptureStart,
    "AnimaActivationCaptureFinish": ActivationCaptureFinish,
    "AnimaActivationCaptureClear": ActivationCaptureClear,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaActivationCaptureStart": "Anima Activation Capture Start",
    "AnimaActivationCaptureFinish": "Anima Activation Capture Finish",
    "AnimaActivationCaptureClear": "Anima Activation Capture Clear",
}
