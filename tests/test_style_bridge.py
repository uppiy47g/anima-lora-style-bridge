import unittest

import torch

import anima_style_bridge as converter


class ConverterTests(unittest.TestCase):
    def test_parses_diffusers_key(self):
        parsed = converter.parse_sd_base(
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_2_attn1_to_q"
        )

        self.assertEqual(
            parsed,
            ("down_blocks_1", 1, 0, 2, "self_attn.q_proj"),
        )

    def test_parses_sgm_key(self):
        parsed = converter.parse_sd_base(
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k"
        )

        self.assertEqual(
            parsed,
            ("input_blocks", 4, 1, 0, "cross_attn.k_proj"),
        )

    def test_interpolation_reaches_both_endpoints(self):
        items = ["first", "middle", "last"]

        self.assertEqual(converter.interpolation_terms(items, 0), [(1.0, "first")])
        self.assertEqual(
            converter.interpolation_terms(items, converter.ANIMA_BLOCKS - 1),
            [(1.0, "last")],
        )

    def test_combine_factors_preserves_shape_and_delta_norm(self):
        torch.manual_seed(7)
        a = torch.randn(2, 4)
        b = torch.randn(3, 2)

        result_a, result_b = converter.combine_factors(
            [(1.0, a, b)],
            rank=2,
            dtype=torch.float32,
        )

        self.assertEqual(result_a.shape, (2, 4))
        self.assertEqual(result_b.shape, (3, 2))
        self.assertTrue(
            torch.allclose(
                converter.delta_norm(result_a, result_b),
                converter.delta_norm(a, b),
                rtol=1e-5,
                atol=1e-6,
            )
        )

    def test_resample_axis_is_deterministic(self):
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        first = converter.resample_axis(matrix, 4, axis=1)
        second = converter.resample_axis(matrix, 4, axis=1)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
