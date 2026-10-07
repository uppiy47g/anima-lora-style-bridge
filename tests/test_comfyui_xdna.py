import gc
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

from anima_xdna_poc.checkpoint_schema import NATIVE_NAMES, canonical_keys
from comfyui_xdna_nodes.model_wrapper import (
    ATTACHMENT_KEY,
    LoadAnimaBF16,
    AnimaXDNARuntimeStatus,
    LoadAttachAnimaXDNAModel,
    RuntimeAttachment,
    SharedRuntime,
    WRAPPER_KEY,
    _model_storage_profile,
    _validate_patcher,
)


Anima = type("Anima", (), {"__module__": "comfy.ldm.anima.model"})


class FakeBaseModel:
    def __init__(self):
        self.diffusion_model = Anima()


def fake_anima_parameters(dtype):
    parameters = {}
    for key in canonical_keys():
        _, block, name = key.split(".", 2)
        parameter_name = f"blocks.{block}.{NATIVE_NAMES[name]}"
        parameters[parameter_name] = torch.nn.Parameter(
            torch.zeros(1, dtype=dtype)
        )
    parameters["patch_embedding.weight"] = torch.nn.Parameter(
        torch.zeros(1, dtype=dtype)
    )
    return parameters


FakeAnimaParameters = type(
    "Anima",
    (Anima,),
    {
        "__init__": lambda self, parameters: setattr(self, "parameters", parameters),
        "named_parameters": lambda self: iter(self.parameters.items()),
    },
)


class FakePatcher:
    def __init__(self):
        self.model = FakeBaseModel()
        self.patches = {}
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}
        self.attachments = {}

    def clone(self):
        clone = FakePatcher()
        clone.model = self.model
        clone.patches = self.patches.copy()
        clone.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        clone.wrappers = {
            kind: {key: values.copy() for key, values in keys.items()}
            for kind, keys in self.wrappers.items()
        }
        clone.attachments = {
            key: value.on_model_patcher_clone()
            if hasattr(value, "on_model_patcher_clone")
            else value
            for key, value in self.attachments.items()
        }
        return clone

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def get_attachment(self, key):
        return self.attachments.get(key)


class ComfyUIXDNAWrapperTests(unittest.TestCase):
    def setUp(self):
        patcher_extension = types.ModuleType("comfy.patcher_extension")

        class WrappersMP:
            DIFFUSION_MODEL = "diffusion_model"

        patcher_extension.WrappersMP = WrappersMP
        comfy = types.ModuleType("comfy")
        comfy.patcher_extension = patcher_extension
        self.saved = {
            name: sys.modules.get(name)
            for name in ("comfy", "comfy.patcher_extension")
        }
        sys.modules["comfy"] = comfy
        sys.modules["comfy.patcher_extension"] = patcher_extension

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    def test_attach_clones_model_and_registers_diffusion_wrapper(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "model.safetensors")
            checkpoint.write_bytes(b"fixture")
            source = FakePatcher()
            with mock.patch.object(SharedRuntime, "prepare"):
                attached, status = LoadAttachAnimaXDNAModel().attach(
                    source, str(checkpoint), False
                )
            self.assertIsNot(attached, source)
            self.assertIs(attached.model, source.model)
            self.assertEqual(source.wrappers, {})
            self.assertIn("created", status)
            self.assertIn(
                WRAPPER_KEY,
                attached.wrappers["diffusion_model"],
            )
            attachment = attached.get_attachment(ATTACHMENT_KEY)
            self.assertIsInstance(attachment, RuntimeAttachment)
            attachment.cleanup()

    def test_attach_cache_identity_is_stable_until_checkpoint_or_options_change(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "model.safetensors")
            checkpoint.write_bytes(b"first")
            initial = LoadAttachAnimaXDNAModel.IS_CHANGED(
                str(checkpoint), False
            )
            self.assertEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(str(checkpoint), False),
            )
            self.assertNotEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(str(checkpoint), True),
            )
            rebuilt = LoadAttachAnimaXDNAModel.IS_CHANGED(
                str(checkpoint), True
            )
            self.assertEqual(
                rebuilt,
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint), True
                ),
            )
            self.assertNotEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint), False, qkv_chaining=False
                ),
            )
            self.assertNotEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint), False, cache_dir=temporary
                ),
            )
            self.assertEqual(
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint), False, cache_dir=temporary
                ),
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint),
                    False,
                    cache_dir=str(Path(temporary, ".")),
                ),
            )
            checkpoint.write_bytes(b"replacement")
            self.assertNotEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(str(checkpoint), False),
            )

    def test_bf16_loader_cache_identity_tracks_checkpoint_file(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "model.safetensors")
            checkpoint.write_bytes(b"first")
            folder_paths = types.ModuleType("folder_paths")
            folder_paths.get_full_path_or_raise = mock.Mock(
                return_value=str(checkpoint)
            )
            with mock.patch.dict("sys.modules", {"folder_paths": folder_paths}):
                initial = LoadAnimaBF16.IS_CHANGED("model.safetensors")
                self.assertEqual(
                    initial,
                    LoadAnimaBF16.IS_CHANGED("model.safetensors"),
                )
                checkpoint.write_bytes(b"replacement")
                self.assertNotEqual(
                    initial,
                    LoadAnimaBF16.IS_CHANGED("model.safetensors"),
                )

    def test_identity_mismatch_closes_before_runtime_open(self):
        runtime = SharedRuntime(Path("fixture.safetensors"))
        chain = mock.Mock()
        chain.source_identity = {
            "schema": "native-model-diffusion-model",
            "block_fingerprint": "source",
        }
        chain.cache_status = None
        with (
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.AnimaXDNAChainRuntime",
                return_value=chain,
            ),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.fingerprint_model_blocks",
                return_value=("model", "comfyui-anima-module"),
            ),
            self.assertRaisesRegex(RuntimeError, "MODEL/checkpoint mismatch"),
        ):
            runtime.prepare(Anima())
        chain.prepare_weight_cache.assert_called_once()
        chain.close.assert_called_once()

    def test_clones_share_runtime_with_reference_counted_attachments(self):
        runtime = SharedRuntime(Path("fixture.safetensors"))
        first = RuntimeAttachment(runtime)
        second = first.on_model_patcher_clone()
        self.assertEqual(runtime.snapshot()["reference_count"], 2)
        first.cleanup()
        self.assertEqual(runtime.snapshot()["reference_count"], 1)
        second.cleanup()
        self.assertEqual(runtime.snapshot()["state"], "closed")

    def test_attachment_finalizer_releases_reference(self):
        runtime = SharedRuntime(Path("fixture.safetensors"))
        attachment = RuntimeAttachment(runtime)
        del attachment
        gc.collect()
        self.assertEqual(runtime.snapshot()["state"], "closed")

    def test_rejects_lora_and_transformer_patches(self):
        patcher = FakePatcher()
        patcher.patches["diffusion_model.blocks.0.self_attn.q_proj.weight"] = []
        with self.assertRaisesRegex(RuntimeError, "LoRA"):
            _validate_patcher(patcher)
        patcher.patches.clear()
        patcher.model_options["transformer_options"]["patches"] = {
            "mlp_patch": [object()]
        }
        with self.assertRaisesRegex(RuntimeError, "transformer patches"):
            _validate_patcher(patcher)

    def test_status_reports_not_attached(self):
        model, status = AnimaXDNARuntimeStatus().status(FakePatcher())
        self.assertIsInstance(model, FakePatcher)
        self.assertIn("not attached", status)

    def test_runtime_snapshot_names_host_visible_output_bytes(self):
        with TemporaryDirectory() as directory:
            runtime = SharedRuntime(Path(directory) / "checkpoint.safetensors")
            self.assertEqual(
                runtime.snapshot()["last_d2h_host_visible_bytes"],
                0,
            )

    def test_model_storage_profile_separates_blocks_from_nonblock_parameters(self):
        parameters = fake_anima_parameters(torch.bfloat16)
        profile = _model_storage_profile(FakeAnimaParameters(parameters))
        self.assertEqual(profile["block_parameter_count"], 560)
        self.assertEqual(profile["block_dtype_numel"], {"bfloat16": 560})
        self.assertEqual(profile["block_parameter_bytes"], 1120)
        self.assertEqual(profile["block_unique_storage_bytes"], 1120)
        self.assertEqual(profile["nonblock_parameter_bytes"], 2)

    def test_fp32_block_memory_is_reported_without_mutating_parameters(self):
        parameters = fake_anima_parameters(torch.float32)
        diffusion_model = FakeAnimaParameters(parameters)
        runtime = SharedRuntime(Path("fixture.safetensors"))
        chain = mock.Mock()
        chain.source_identity = {
            "schema": "native-model-diffusion-model",
            "block_fingerprint": "same",
        }
        chain.cache_status = None
        with (
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.AnimaXDNAChainRuntime",
                return_value=chain,
            ),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.fingerprint_model_blocks",
                return_value=("same", "comfyui-anima-module"),
            ),
        ):
            runtime.prepare(diffusion_model)
        self.assertEqual(runtime.snapshot()["model_block_parameter_bytes"], 2240)
        self.assertIn(
            "Load Anima (BF16)",
            runtime.snapshot()["model_fp32_block_memory_warning"],
        )
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in parameters.values())
        )
        runtime.close()

    def test_bf16_loader_passes_explicit_dtype_to_comfyui(self):
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                diffusion_model=FakeAnimaParameters(
                    fake_anima_parameters(torch.bfloat16)
                )
            )
        )
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_diffusion_model = mock.Mock(return_value=model)
        comfy = types.ModuleType("comfy")
        comfy.sd = comfy_sd
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_full_path_or_raise = mock.Mock(
            return_value="anima.safetensors"
        )
        with mock.patch.dict(
            "sys.modules",
            {"comfy": comfy, "comfy.sd": comfy_sd, "folder_paths": folder_paths},
        ):
            (loaded,) = LoadAnimaBF16().load("anima.safetensors")
        self.assertIs(loaded, model)
        comfy_sd.load_diffusion_model.assert_called_once_with(
            "anima.safetensors",
            model_options={"dtype": torch.bfloat16},
        )

    def test_bf16_loader_rejects_dtype_expansion(self):
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                diffusion_model=FakeAnimaParameters(
                    fake_anima_parameters(torch.float32)
                )
            )
        )
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_diffusion_model = mock.Mock(return_value=model)
        comfy = types.ModuleType("comfy")
        comfy.sd = comfy_sd
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_full_path_or_raise = mock.Mock(
            return_value="anima.safetensors"
        )
        with (
            mock.patch.dict(
                "sys.modules",
                {"comfy": comfy, "comfy.sd": comfy_sd, "folder_paths": folder_paths},
            ),
            self.assertRaisesRegex(RuntimeError, "refusing to load a memory-expanded"),
        ):
            LoadAnimaBF16().load("anima.safetensors")

    def test_rejects_non_anima_model(self):
        patcher = FakePatcher()
        patcher.model.diffusion_model = object()
        with self.assertRaisesRegex(RuntimeError, "Anima MODEL"):
            _validate_patcher(patcher)


if __name__ == "__main__":
    unittest.main()
