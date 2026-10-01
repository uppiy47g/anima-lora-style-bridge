import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from anima_xdna_poc.checkpoint import discover_linear_weight, select_weight_key
from anima_xdna_poc.errors import DependencyUnavailable, NPUUnavailable, UnsupportedTensor
from anima_xdna_poc.linear import cpu_linear, deterministic_input, prepare_linear


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
        self.assertEqual(prepared.padded_shape, (256, 256, 64))
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

    def test_deterministic_input_repeats(self):
        self.assertTrue(
            torch.equal(
                deterministic_input(4, 7, seed=23),
                deterministic_input(4, 7, seed=23),
            )
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


if __name__ == "__main__":
    unittest.main()
