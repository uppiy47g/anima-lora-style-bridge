import tempfile
import unittest
import os
import json
from pathlib import Path
import threading
from unittest import mock

import torch
from safetensors.torch import save_file

from anima_xdna_poc.checkpoint import discover_linear_weight, select_weight_key
from anima_xdna_poc.checkpoint_schema import (
    NATIVE_NAMES,
    canonical_keys,
    detect_checkpoint_schema,
    fingerprint_model_blocks,
    validated_variant,
)
from anima_xdna_poc.errors import DependencyUnavailable, NPUUnavailable, UnsupportedTensor
from anima_xdna_poc.linear import cpu_linear, deterministic_input, prepare_linear
from anima_xdna_poc.block import (
    deterministic_block_inputs,
    rotary_embedding,
    run_cpu_block,
)
from anima_xdna_poc.block_checkpoint import (
    AnimaBlockConfig,
    AnimaBlockWeights,
    LINEAR_SHAPES,
    NORM_SHAPES,
    load_block_config,
    load_block_weights,
)
from anima_xdna_poc.chain import AnimaXDNAChainRuntime
from anima_xdna_poc.weight_cache import (
    CacheIntegrityError,
    CacheNamespace,
    LAYOUT_VERSION,
    PackedWeightCache,
    fingerprint_source,
)


def _tiny_block_config():
    return AnimaBlockConfig(
        hidden_size=24,
        num_heads=2,
        head_dim=12,
        context_dim=16,
        adaln_dim=8,
        mlp_dim=48,
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 4.0, 4.0),
        max_size=(8, 8, 8),
    )


def _block_tensors(config, fill=0.0):
    dimensions = {
        "hidden": config.hidden_size,
        "head": config.head_dim,
        "context": config.context_dim,
        "adaln": config.adaln_dim,
        "modulation": 3 * config.hidden_size,
        "mlp": config.mlp_dim,
    }
    tensors = {
        name: torch.full(
            (dimensions[out_name], dimensions[in_name]),
            fill,
            dtype=torch.bfloat16,
        )
        for name, (out_name, in_name) in LINEAR_SHAPES.items()
    }
    tensors.update(
        {
            name: torch.ones(dimensions[size_name], dtype=torch.bfloat16)
            for name, size_name in NORM_SHAPES.items()
        }
    )
    return tensors


def _save_blocks(path, config, count=1, fill=0.0):
    tensors = {}
    for index in range(count):
        tensors.update(
            {
                f"transformer_blocks.{index}.{name}": tensor
                for name, tensor in _block_tensors(config, fill + index).items()
            }
        )
    save_file(tensors, path)


def _save_native_blocks(path, config, prefix, count=1, fill=0.0):
    tensors = {}
    for index in range(count):
        tensors.update(
            {
                f"{prefix}blocks.{index}.{NATIVE_NAMES[name]}": tensor
                for name, tensor in _block_tensors(config, fill + index).items()
            }
        )
    save_file(tensors, path)


class PackedWeightCacheTests(unittest.TestCase):
    def test_model_fingerprint_normalizes_cpu_fp32_upcast(self):
        class Model:
            def __init__(self, dtype):
                self.tensors = {
                    key: torch.full((1,), 1.5, dtype=dtype)
                    for key in canonical_keys()
                }

            def state_dict(self):
                return self.tensors

        bf16, _ = fingerprint_model_blocks(Model(torch.bfloat16))
        fp32, _ = fingerprint_model_blocks(Model(torch.float32))
        self.assertEqual(bf16, fp32)
        self.assertEqual(
            validated_variant(
                "066b4281037504b1b7200ecd65b4182fc765ed882ecd5ca650db7308246a8dee"
            ),
            "Turbo V1.1",
        )

    def test_native_schemas_have_canonical_base_identity_and_separate_turbo(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffusers = root / "diffusers.safetensors"
            native_base = root / "base.safetensors"
            native_turbo = root / "turbo.safetensors"
            _save_blocks(diffusers, config)
            _save_native_blocks(native_base, config, "net.")
            _save_native_blocks(
                native_turbo, config, "model.diffusion_model.", fill=1.0
            )
            diff_identity, _ = fingerprint_source(diffusers, range(1))
            base_identity, _ = fingerprint_source(native_base, range(1))
            turbo_identity, _ = fingerprint_source(native_turbo, range(1))
            cache = PackedWeightCache(
                native_turbo, config, 0, 1, root / "cache"
            )
            status = cache.open()
            manifest = cache.manifest
            cache.close()
        self.assertEqual(diff_identity["schema"], "diffusers-transformer")
        self.assertEqual(base_identity["schema"], "native-net")
        self.assertEqual(
            turbo_identity["schema"], "native-model-diffusion-model"
        )
        self.assertEqual(
            diff_identity["block_fingerprint"],
            base_identity["block_fingerprint"],
        )
        self.assertNotEqual(
            base_identity["block_fingerprint"],
            turbo_identity["block_fingerprint"],
        )
        self.assertEqual(status.tensor_count, 20)
        self.assertTrue(
            all(
                tensor["source_key"].startswith("model.diffusion_model.")
                for tensor in manifest["descriptor"]["source_identity"]["tensors"]
            )
        )

    def test_schema_resolver_rejects_incomplete_checkpoint(self):
        with self.assertRaisesRegex(UnsupportedTensor, "matches=none"):
            detect_checkpoint_schema(["blocks.0.self_attn.q_proj.weight"], range(1))

    def test_deterministic_key_and_future_namespace_fields(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            _save_blocks(checkpoint, config)
            cache = PackedWeightCache(
                checkpoint, config, 0, 1, root / "cache"
            )
            key1, descriptor1, _ = cache.identify()
            key2, descriptor2, _ = cache.identify()
        self.assertEqual(key1, key2)
        self.assertEqual(descriptor1, descriptor2)
        namespace = descriptor1["namespace"]
        self.assertEqual(namespace["adapter_fingerprints"], [])
        self.assertIsNone(namespace["merge_strength"])
        self.assertEqual(namespace["quantization_scheme"], "none")
        self.assertEqual(namespace["layout_version"], LAYOUT_VERSION)
        self.assertEqual(
            CacheNamespace(namespace["base_model_fingerprint"]).quantization_scheme,
            "none",
        )

    def test_source_content_change_invalidates_key(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            _save_blocks(checkpoint, config, fill=0.0)
            first = PackedWeightCache(checkpoint, config, 0, 1, root / "cache")
            key1, _, _ = first.identify()
            _save_blocks(checkpoint, config, fill=1.0)
            second = PackedWeightCache(checkpoint, config, 0, 1, root / "cache")
            key2, _, _ = second.identify()
        self.assertNotEqual(key1, key2)

    def test_header_shape_dtype_layout_and_tool_versions_invalidate_key(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            _save_blocks(checkpoint, config)
            cache = PackedWeightCache(checkpoint, config, 0, 1, root / "cache")
            key, _, _ = cache.identify()
            changed_config = AnimaBlockConfig(
                **{**config.__dict__, "max_size": (9, 8, 8)}
            )
            changed, _, _ = PackedWeightCache(
                checkpoint, changed_config, 0, 1, root / "cache"
            ).identify()
            float_tensors = {
                f"transformer_blocks.0.{name}": tensor.float()
                for name, tensor in _block_tensors(config).items()
            }
            save_file(float_tensors, checkpoint)
            dtype_changed, _, _ = PackedWeightCache(
                checkpoint, config, 0, 1, root / "cache"
            ).identify()
            shape_tensors = _block_tensors(config)
            shape_tensors["attn1.to_q.weight"] = torch.zeros(
                config.hidden_size + 1,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            save_file(
                {
                    f"transformer_blocks.0.{name}": tensor
                    for name, tensor in shape_tensors.items()
                },
                checkpoint,
            )
            shape_changed, _, _ = PackedWeightCache(
                checkpoint, config, 0, 1, root / "cache"
            ).identify()
            _save_blocks(checkpoint, config)
            with mock.patch(
                "anima_xdna_poc.weight_cache.tool_identity",
                return_value={"triton_xdna": "different"},
            ):
                tool_changed, _, _ = cache.identify()
            with mock.patch(
                "anima_xdna_poc.weight_cache.LAYOUT_VERSION",
                "different-layout",
            ):
                layout_changed, _, _ = cache.identify()
        self.assertNotEqual(key, changed)
        self.assertNotEqual(key, dtype_changed)
        self.assertNotEqual(key, shape_changed)
        self.assertNotEqual(key, tool_changed)
        self.assertNotEqual(key, layout_changed)

    def test_build_open_matches_prepare_and_exact_prune(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            _save_blocks(checkpoint, config)
            cache = PackedWeightCache(checkpoint, config, 0, 1, root / "cache")
            status = cache.open()
            source = load_block_weights(checkpoint, config)
            inputs = torch.ones(3, config.hidden_size, dtype=torch.bfloat16)
            cached = cache.prepared(
                0, "attn1.to_q.weight", 0, inputs, config.hidden_size
            )
            regular = prepare_linear(inputs, source["attn1.to_q.weight"])
            self.assertTrue(
                torch.equal(cached.weight_k_n_bf16, regular.weight_k_n_bf16)
            )
            self.assertGreater(status.padding_bytes, 0)
            del cached
            del regular
            cache.close()
            removed = cache.prune()
            self.assertFalse(removed.exists())
            self.assertTrue(root.exists())

    def test_corrupt_payload_and_manifest_are_rejected(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            _save_blocks(checkpoint, config)
            cache = PackedWeightCache(checkpoint, config, 0, 1, root / "cache")
            status = cache.ensure()
            payload = status.path / "weights.bin"
            payload.write_bytes(payload.read_bytes()[:-1])
            with self.assertRaisesRegex(CacheIntegrityError, "size mismatch"):
                cache.ensure()
            status = cache.ensure(rebuild=True)
            manifest_path = status.path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["cache_key"] = "wrong"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CacheIntegrityError, "key mismatch"):
                cache.ensure()

    def test_atomic_failure_cleans_temporary_entry(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            cache_root = root / "cache"
            _save_blocks(checkpoint, config)
            cache = PackedWeightCache(checkpoint, config, 0, 1, cache_root)
            real_replace = os.replace

            def fail_directory_replace(source, destination):
                if Path(source).is_dir():
                    raise OSError("injected atomic failure")
                return real_replace(source, destination)

            with mock.patch(
                "anima_xdna_poc.weight_cache.os.replace",
                side_effect=fail_directory_replace,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    cache.ensure()
            self.assertEqual(
                [path for path in cache_root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_concurrent_build_has_one_atomic_winner(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            _save_blocks(checkpoint, config)
            results = []
            errors = []

            def build():
                try:
                    results.append(
                        PackedWeightCache(
                            checkpoint, config, 0, 1, root / "cache"
                        ).ensure()
                    )
                except Exception as error:
                    errors.append(error)

            threads = [threading.Thread(target=build) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(sum(status.hit for status in results), 1)
            self.assertEqual(len({status.key for status in results}), 1)

    def test_disabled_mode_creates_nothing(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            _save_blocks(checkpoint, config)
            cache_root = root / "cache"
            status = PackedWeightCache(
                checkpoint,
                config,
                0,
                1,
                cache_root,
                enabled=False,
            ).ensure()
            self.assertEqual(status.reason, "disabled")
            self.assertFalse(cache_root.exists())

    def test_shard_index_fingerprints_every_file_and_builds_once(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = {
                f"transformer_blocks.0.{name}": tensor
                for name, tensor in _block_tensors(config).items()
            }
            keys = sorted(tensors)
            first_keys = keys[: len(keys) // 2]
            second_keys = keys[len(keys) // 2 :]
            first_name = "model-00001-of-00002.safetensors"
            second_name = "model-00002-of-00002.safetensors"
            save_file({key: tensors[key] for key in first_keys}, root / first_name)
            save_file({key: tensors[key] for key in second_keys}, root / second_name)
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            **{key: first_name for key in first_keys},
                            **{key: second_name for key in second_keys},
                        }
                    }
                ),
                encoding="utf-8",
            )
            cache = PackedWeightCache(index, config, 0, 1, root / "cache")
            status = cache.ensure()
            self.assertEqual(
                len(cache.manifest["descriptor"]["source_identity"]["files"]),
                2,
            )
            self.assertEqual(status.tensor_count, 20)


class XDNAPocUnitTests(unittest.TestCase):
    def test_selects_canonical_key_before_suffix_fallback(self):
        canonical = "diffusion_model.blocks.0.self_attn.q_proj.weight"
        selected = select_weight_key(
            ["wrapper.blocks.0.self_attn.q_proj.weight", canonical]
        )
        self.assertEqual(selected, canonical)

    def test_selects_diffusers_anima_q_projection(self):
        key = "transformer_blocks.0.attn1.to_q.weight"
        self.assertEqual(select_weight_key([key]), key)

    def test_ambiguous_suffix_requires_explicit_key(self):
        with self.assertRaisesRegex(UnsupportedTensor, "multiple candidate"):
            select_weight_key(
                [
                    "first.blocks.0.self_attn.q_proj.weight",
                    "second.blocks.0.self_attn.q_proj.weight",
                ]
            )

    def test_reader_loads_only_selected_valid_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            key = "diffusion_model.blocks.0.self_attn.q_proj.weight"
            save_file(
                {
                    key: torch.arange(15, dtype=torch.float32).reshape(3, 5),
                    "unrelated.large.weight": torch.zeros(128, 128),
                },
                path,
            )
            selected = discover_linear_weight(path)
        self.assertEqual(selected.key, key)
        self.assertEqual(tuple(selected.tensor.shape), (3, 5))

    def test_reader_rejects_non_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            key = "diffusion_model.blocks.0.self_attn.q_proj.weight"
            save_file({key: torch.zeros(2, 3, 4)}, path)
            with self.assertRaisesRegex(UnsupportedTensor, "rank-2"):
                discover_linear_weight(path)

    def test_layout_transposes_weight_and_pads_to_aie2p_alignment(self):
        inputs = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
        weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        prepared = prepare_linear(inputs, weight)
        self.assertEqual(prepared.padded_shape, (256, 256, 256))
        self.assertTrue(
            torch.equal(prepared.weight_k_n_bf16[:3, :5], weight.to(torch.bfloat16).T)
        )
        self.assertEqual(prepared.input_bf16[1:].count_nonzero().item(), 0)

    def test_cpu_oracle_uses_fp32_accumulation_and_bf16_output(self):
        inputs = deterministic_input(3, 5, seed=17)
        weight = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 7
        prepared = prepare_linear(inputs, weight)
        result = cpu_linear(prepared)
        expected = (inputs.float() @ weight.to(torch.bfloat16).float().T).to(
            torch.bfloat16
        )
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(result, expected))

    def test_layout_widens_single_output_tile_for_large_reduction(self):
        prepared = prepare_linear(
            torch.zeros(1, 512, dtype=torch.bfloat16),
            torch.zeros(256, 512, dtype=torch.bfloat16),
        )
        self.assertEqual(prepared.padded_shape, (256, 512, 512))

    def test_deterministic_input_repeats(self):
        self.assertTrue(
            torch.equal(
                deterministic_input(4, 7, seed=23),
                deterministic_input(4, 7, seed=23),
            )
        )


class AnimaBlockUnitTests(unittest.TestCase):
    def test_loads_and_validates_exact_block_schema(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                """{
                    "num_attention_heads": 2,
                    "attention_head_dim": 12,
                    "cross_attention_dim": 16,
                    "adaln_lora_dim": 8,
                    "mlp_ratio": 2.0,
                    "patch_size": [1, 2, 2],
                    "rope_scale": [1.0, 4.0, 4.0],
                    "max_size": [8, 8, 8]
                }""",
                encoding="utf-8",
            )
            checkpoint = root / "model.safetensors"
            save_file(
                {
                    "transformer_blocks.0." + name: tensor
                    for name, tensor in _block_tensors(config).items()
                }
                | {"unrelated.weight": torch.zeros(128, 128)},
                checkpoint,
            )
            loaded_config = load_block_config(root)
            loaded = load_block_weights(checkpoint, loaded_config)
        self.assertEqual(loaded_config, config)
        self.assertEqual(len(loaded.tensors), 20)
        self.assertEqual(loaded.prefix, "transformer_blocks.0.")

    def test_rejects_wrong_block_weight_shape(self):
        config = _tiny_block_config()
        tensors = _block_tensors(config)
        tensors["attn1.to_q.weight"] = torch.zeros(5, 5)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_file(
                {"transformer_blocks.0." + name: tensor for name, tensor in tensors.items()},
                checkpoint,
            )
            with self.assertRaisesRegex(UnsupportedTensor, "expected"):
                load_block_weights(checkpoint, config)

    def test_cosmos_rope_has_exact_head_shape(self):
        config = _tiny_block_config()
        cos, sin = rotary_embedding(config, 16)
        self.assertEqual(tuple(cos.shape), (16, 12))
        self.assertEqual(tuple(sin.shape), (16, 12))
        self.assertTrue(torch.equal(cos[0], torch.ones(12)))
        self.assertTrue(torch.equal(sin[0], torch.zeros(12)))

    def test_zero_projection_block_preserves_residual(self):
        config = _tiny_block_config()
        weights = AnimaBlockWeights(
            "transformer_blocks.0.",
            _block_tensors(config),
        )
        inputs = deterministic_block_inputs(
            config,
            image_tokens=4,
            context_tokens=3,
            seed=41,
            masked_context_tokens=1,
        )
        self.assertEqual(tuple(inputs.attention_mask.shape), (1, 1, 1, 3))
        self.assertEqual(inputs.attention_mask.count_nonzero().item(), 2)
        result = run_cpu_block(config, weights, inputs)
        self.assertTrue(torch.equal(result.output, inputs.hidden_states))
        self.assertEqual(
            [name for name, _ in result.checkpoints],
            ["self_attention", "cross_attention", "feed_forward"],
        )
        self.assertEqual(result.dispatch_count, 24)

    def test_block_input_validation_rejects_non_square_rope_fixture(self):
        with self.assertRaisesRegex(UnsupportedTensor, "square"):
            deterministic_block_inputs(
                _tiny_block_config(),
                image_tokens=6,
                context_tokens=3,
            )

    def test_cpu_chain_maps_block_indices_and_preserves_zero_residuals(self):
        config = _tiny_block_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                """{
                    "num_attention_heads": 2,
                    "attention_head_dim": 12,
                    "encoder_hidden_states_channels": 16,
                    "adaln_lora_dim": 8,
                    "mlp_ratio": 2.0,
                    "patch_size": [1, 2, 2],
                    "rope_scale": [1.0, 4.0, 4.0],
                    "max_size": [8, 8, 8]
                }""",
                encoding="utf-8",
            )
            checkpoint = root / "model.safetensors"
            tensors = {}
            for index in range(2):
                tensors.update(
                    {
                        f"transformer_blocks.{index}.{name}": tensor
                        for name, tensor in _block_tensors(config).items()
                    }
                )
            save_file(tensors, checkpoint)
            runtime = AnimaXDNAChainRuntime(checkpoint)
            inputs = deterministic_block_inputs(config, 4, 3, seed=91)
            result = runtime.run_cpu_range(inputs, 0, 2)
            self.assertTrue(torch.equal(result.output, inputs.hidden_states))
            self.assertEqual([block.index for block in result.blocks], [0, 1])
            with self.assertRaisesRegex(UnsupportedTensor, r"\[0, 27\]"):
                runtime.weights(28)


class XDNAPocIntegrationTest(unittest.TestCase):
    def test_xdna_matches_cpu_when_available(self):
        from anima_xdna_poc.xdna import execute, probe

        try:
            probe()
        except (DependencyUnavailable, NPUUnavailable) as error:
            self.skipTest(f"{error.category}: {error}")

        prepared = prepare_linear(
            deterministic_input(256, 256, seed=31),
            torch.randn(
                256,
                256,
                generator=torch.Generator().manual_seed(32),
            )
            / (256**0.5),
        )
        reference = cpu_linear(prepared)
        result = execute(prepared, warmup=0, runs=1)
        torch.testing.assert_close(
            result.output_bf16,
            reference,
            rtol=1.6e-2,
            atol=1.5e-3 * (8192 / 256) ** 0.5,
        )

    def test_real_anima_block_matches_cpu_when_available(self):
        checkpoint_value = os.environ.get("ANIMA_XDNA_CHECKPOINT")
        if not checkpoint_value:
            self.skipTest(
                "ANIMA_XDNA_CHECKPOINT is not set to an Anima Base transformer safetensors"
            )
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_file():
            self.skipTest(f"ANIMA_XDNA_CHECKPOINT does not exist: {checkpoint}")

        from anima_xdna_poc.block import (
            compare_block_outputs,
            deterministic_block_inputs,
            run_cpu_block,
            run_xdna_block,
        )
        from anima_xdna_poc.block_checkpoint import (
            load_block_config,
            load_block_weights,
        )

        try:
            from anima_xdna_poc.xdna import probe

            probe()
        except (DependencyUnavailable, NPUUnavailable) as error:
            self.skipTest(f"{error.category}: {error}")
        config = load_block_config(checkpoint.parent)
        weights = load_block_weights(checkpoint, config)
        inputs = deterministic_block_inputs(config, 16, 8, seed=73)
        reference = run_cpu_block(config, weights, inputs)
        actual = run_xdna_block(config, weights, inputs)
        max_error, _ = compare_block_outputs(
            actual.output,
            reference.output,
            rtol=2e-2,
            atol=3e-3,
        )
        self.assertLess(max_error, 0.005)
        legacy = run_xdna_block(
            config,
            weights,
            inputs,
            batched_attention=False,
        )
        torch.testing.assert_close(
            actual.output,
            legacy.output,
            rtol=2e-2,
            atol=3e-3,
        )
        self.assertEqual(actual.dispatch_count, 23)
        self.assertEqual(legacy.dispatch_count, 83)

    def test_real_anima_chain_matches_cpu_when_available(self):
        checkpoint_value = os.environ.get("ANIMA_XDNA_CHECKPOINT")
        fixture_value = os.environ.get("ANIMA_XDNA_CHAIN_FIXTURE")
        if not checkpoint_value or not fixture_value:
            self.skipTest(
                "ANIMA_XDNA_CHECKPOINT and ANIMA_XDNA_CHAIN_FIXTURE are required"
            )
        checkpoint = Path(checkpoint_value)
        fixture = Path(fixture_value)
        if not checkpoint.is_file() or not fixture.is_file():
            self.skipTest(f"chain checkpoint or fixture is missing: {checkpoint}, {fixture}")
        captured = torch.load(fixture, map_location="cpu", weights_only=True)
        from anima_xdna_poc.block import BlockInputs

        inputs = BlockInputs(
            captured["hidden_states"],
            captured["encoder_hidden_states"],
            captured["embedded_timestep"],
            captured["temb"],
            (captured["cos"], captured["sin"]),
            captured.get("attention_mask"),
        )
        with AnimaXDNAChainRuntime(
            checkpoint,
            cache_host_weights=False,
            weight_cache=True,
        ) as runtime:
            validation = runtime.validate_range(inputs)
            steady = runtime.run_range(inputs)
            cache_status = runtime.cache_status
        with AnimaXDNAChainRuntime(
            checkpoint,
            cache_host_weights=False,
            weight_cache=False,
        ) as runtime:
            uncached = runtime.run_range(inputs)
        self.assertEqual(validation.xdna.dispatch_count, 23 * 28)
        self.assertEqual(steady.dispatch_count, 23 * 28)
        self.assertTrue(cache_status.hit)
        self.assertTrue(torch.equal(steady.output, uncached.output))
        self.assertLess(
            max(error.normalized_rms_error for error in validation.errors),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
