import gc
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from comfyui_xdna_nodes.model_wrapper import (
    ATTACHMENT_KEY,
    AnimaXDNARuntimeStatus,
    LoadAttachAnimaXDNAModel,
    RuntimeAttachment,
    SharedRuntime,
    WRAPPER_KEY,
    _validate_patcher,
)


Anima = type("Anima", (), {"__module__": "comfy.ldm.anima.model"})


class FakeBaseModel:
    def __init__(self):
        self.diffusion_model = Anima()


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
            self.assertEqual(source.wrappers, {})
            self.assertIn("created", status)
            self.assertIn(
                WRAPPER_KEY,
                attached.wrappers["diffusion_model"],
            )
            attachment = attached.get_attachment(ATTACHMENT_KEY)
            self.assertIsInstance(attachment, RuntimeAttachment)
            attachment.cleanup()

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

    def test_rejects_non_anima_model(self):
        patcher = FakePatcher()
        patcher.model.diffusion_model = object()
        with self.assertRaisesRegex(RuntimeError, "Anima MODEL"):
            _validate_patcher(patcher)


if __name__ == "__main__":
    unittest.main()
