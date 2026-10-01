import tempfile
import unittest
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from anima_xdna_poc.checkpoint import discover_linear_weight, select_weight_key
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
        self.assertEqual(actual.dispatch_count, 83)


if __name__ == "__main__":
    unittest.main()
