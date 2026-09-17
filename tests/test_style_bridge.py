import tempfile
import unittest
from pathlib import Path

import numpy as np
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

    def test_linear_cka_is_invariant_to_orthogonal_feature_rotation(self):
        torch.manual_seed(11)
        features = torch.randn(24, 5)
        rotation, _ = torch.linalg.qr(torch.randn(5, 5))

        score = converter.linear_cka(features, features @ rotation)

        self.assertAlmostEqual(score, 1.0, places=5)

    def test_monotonic_cka_matching_preserves_source_order(self):
        first = torch.tensor(
            [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        )
        second = torch.tensor(
            [[0.0, -2.0], [0.0, 1.0], [0.0, -1.0], [0.0, 2.0]]
        )

        matches = converter.monotonic_cka_matches(
            [(0, first), (1, second)],
            [(0, first), (1, second)],
        )

        self.assertEqual(matches[0][0], 0)
        self.assertEqual(matches[1][0], 1)

    def test_depth_matching_reaches_endpoints(self):
        source_items = [(block, None) for block in range(5)]
        target_items = [(block, None) for block in range(3)]

        matches = converter.depth_matches(source_items, target_items)

        self.assertEqual(matches, {0: 0, 1: 2, 2: 4})

    def test_batched_procrustes_recovers_shared_rotation(self):
        torch.manual_seed(12)
        rotation, _ = torch.linalg.qr(torch.randn(4, 4))
        pairs = []
        for _ in range(3):
            source = torch.randn(20, 4)
            pairs.append((source, source @ rotation.T))

        fitted = converter.fit_projection_batches(
            pairs,
            method="procrustes",
            regularization=1e-4,
        )

        self.assertTrue(torch.allclose(fitted, rotation, rtol=1e-4, atol=1e-4))

    def test_ridge_map_recovers_linear_projection(self):
        torch.manual_seed(13)
        inputs = torch.randn(64, 4)
        projection = torch.randn(4, 3)
        outputs = inputs @ projection

        fitted = converter.ridge_map(inputs, outputs, 1e-6)

        self.assertTrue(torch.allclose(fitted, projection, rtol=1e-4, atol=1e-4))

    def test_ridge_transform_maps_directions_without_materializing_projection(self):
        torch.manual_seed(15)
        inputs = torch.randn(32, 4)
        projection = torch.randn(4, 3)
        outputs = inputs @ projection
        directions = torch.randn(12, 4) * 0.1

        transformed = converter.ridge_transform_directions(
            inputs,
            outputs,
            directions,
            1e-6,
        )

        self.assertTrue(
            torch.allclose(
                transformed,
                directions @ projection,
                rtol=1e-3,
                atol=1e-4,
            )
        )

    def test_factorized_delta_calibrates_output_rms(self):
        torch.manual_seed(17)
        inputs = torch.randn(80, 4)
        desired = inputs @ torch.randn(4, 3)

        lora_a, lora_b, scale, normalized_error = converter.factorize_distilled_delta(
            inputs,
            desired,
            rank=3,
            regularization=1e-6,
            dtype=torch.float32,
        )
        predicted = inputs @ (lora_b @ lora_a).T

        self.assertGreater(scale, 0)
        self.assertLess(normalized_error, 1e-4)
        self.assertTrue(
            torch.allclose(
                predicted.square().mean().sqrt(),
                desired.square().mean().sqrt(),
                rtol=1e-5,
                atol=1e-6,
            )
        )

    def test_distill_v3_writes_valid_anima_lora(self):
        torch.manual_seed(19)
        source_base = torch.randn(32, 2)
        target_input = torch.randn(32, 3)
        alignment = torch.tensor([[1.0, 0.2], [0.1, 0.8]])
        target_base = source_base @ alignment
        desired_delta = target_input @ torch.randn(3, 2) * 0.1
        source_delta = desired_delta @ torch.linalg.inv(alignment)
        original_dims = converter.ANIMA_DIMS
        converter.ANIMA_DIMS = {"self_attn.q_proj": (3, 2)}
        try:
            with tempfile.TemporaryDirectory() as directory:
                input_path = Path(directory) / "activations.npz"
                output_path = Path(directory) / "result.safetensors"
                np.savez(
                    input_path,
                    **{
                        "source.0.self_attn.q_proj.base": source_base.numpy(),
                        "source.0.self_attn.q_proj.styled": (
                            source_base + source_delta
                        ).numpy(),
                        "target.0.self_attn.q_proj.base": target_base.numpy(),
                        "target.0.self_attn.q_proj.input": target_input.numpy(),
                    },
                )

                converter.distill_v3(
                    input_path,
                    output_path,
                    rank=2,
                    alignment_regularization=1e-4,
                    distillation_regularization=1e-4,
                    min_cka=0.0,
                    max_scale=10.0,
                    max_nrmse=1.0,
                    validation_fraction=0.2,
                    seed=0,
                    output_dtype="float32",
                )
                converter.validate_output(output_path)

                state = converter.load_file(output_path)
                self.assertEqual(
                    state[
                        "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight"
                    ].shape,
                    (2, 3),
                )
                self.assertEqual(
                    state[
                        "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight"
                    ].shape,
                    (2, 2),
                )
        finally:
            converter.ANIMA_DIMS = original_dims

    def test_hybrid_bridge_distills_valid_anima_lora(self):
        torch.manual_seed(21)
        samples = 48
        source_base = torch.randn(samples, 2)
        rotation, _ = torch.linalg.qr(torch.randn(2, 2))
        target_base = source_base @ rotation.T
        target_input = torch.randn(samples, 3)
        target_delta = target_input @ torch.randn(3, 2) * 0.1
        source_delta = target_delta @ rotation
        original_dims = converter.ANIMA_DIMS
        converter.ANIMA_DIMS = {"self_attn.q_proj": (3, 2)}
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                input_path = directory / "activations.npz"
                bridge_path = directory / "bridge.npz"
                output_path = directory / "result.safetensors"
                np.savez(
                    input_path,
                    **{
                        "source.0.self_attn.q_proj.base": source_base.numpy(),
                        "source.0.self_attn.q_proj.styled": (
                            source_base + source_delta
                        ).numpy(),
                        "target.0.self_attn.q_proj.base": target_base.numpy(),
                        "target.0.self_attn.q_proj.input": target_input.numpy(),
                    },
                )

                converter.fit_v3_hybrid_bridge(
                    input_path,
                    bridge_path,
                    method="procrustes",
                    regularization=1e-4,
                )
                converter.distill_v3_hybrid(
                    input_path,
                    bridge_path,
                    output_path,
                    rank=2,
                    distillation_regularization=1e-4,
                    max_scale=10.0,
                    max_nrmse=0.1,
                    validation_fraction=0.2,
                    seed=0,
                    output_dtype="float32",
                )

                converter.validate_output(output_path)
                state = converter.load_file(output_path)
                self.assertEqual(
                    state[
                        "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight"
                    ].shape,
                    (2, 3),
                )
        finally:
            converter.ANIMA_DIMS = original_dims

    def test_prepare_v3_reconstructs_local_source_base(self):
        torch.manual_seed(23)
        inputs = torch.randn(12, 3)
        base_weight = torch.randn(2, 3)
        down = torch.randn(1, 3)
        up = torch.randn(2, 1)
        delta = (inputs @ down.T) @ up.T
        styled = inputs @ base_weight.T + delta
        target_input = torch.randn(12, 3)
        target_base = torch.randn(12, 2)
        original_dims = converter.ANIMA_DIMS
        converter.ANIMA_DIMS = {"self_attn.q_proj": (3, 2)}
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                source_path = directory / "source.npz"
                target_path = directory / "target.npz"
                lora_path = directory / "style.safetensors"
                output_path = directory / "v3.npz"
                source_prefix = (
                    "diffusion_model.input_blocks.1.1.transformer_blocks.0."
                    "attn1.to_q"
                )
                np.savez(
                    source_path,
                    **{
                        f"{source_prefix}.input": inputs.numpy(),
                        f"{source_prefix}.output": styled.numpy(),
                        "__aggregation__": np.array("mean"),
                    },
                )
                target_prefix = "diffusion_model.blocks.0.self_attn.q_proj"
                np.savez(
                    target_path,
                    **{
                        f"{target_prefix}.input": target_input.numpy(),
                        f"{target_prefix}.output": target_base.numpy(),
                        "__aggregation__": np.array("mean"),
                    },
                )
                converter.save_file(
                    {
                        (
                            "lora_unet_input_blocks_1_1_transformer_blocks_0_"
                            "attn1_to_q.lora_down.weight"
                        ): down,
                        (
                            "lora_unet_input_blocks_1_1_transformer_blocks_0_"
                            "attn1_to_q.lora_up.weight"
                        ): up,
                    },
                    lora_path,
                )

                converter.prepare_v3_activations(
                    source_path,
                    target_path,
                    lora_path,
                    output_path,
                    lora_strength=1.0,
                )

                with np.load(output_path) as prepared:
                    self.assertTrue(
                        np.allclose(
                            prepared["source.0.self_attn.q_proj.base"],
                            (inputs @ base_weight.T).numpy(),
                            rtol=1e-5,
                            atol=1e-6,
                        )
                    )
                    self.assertTrue(
                        np.array_equal(
                            prepared["target.0.self_attn.q_proj.input"],
                            target_input.numpy(),
                        )
                    )
        finally:
            converter.ANIMA_DIMS = original_dims

    def test_prepare_v3_can_truncate_unpaired_tail_rows(self):
        torch.manual_seed(29)
        source_rows = 8
        target_rows = 10
        source_input = torch.randn(source_rows, 3)
        down = torch.randn(1, 3)
        up = torch.randn(2, 1)
        source_output = (source_input @ down.T) @ up.T
        target_input = torch.randn(target_rows, 3)
        target_output = torch.randn(target_rows, 2)
        original_dims = converter.ANIMA_DIMS
        converter.ANIMA_DIMS = {"self_attn.q_proj": (3, 2)}
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                source_path = directory / "source.npz"
                target_path = directory / "target.npz"
                lora_path = directory / "style.safetensors"
                output_path = directory / "v3.npz"
                source_prefix = (
                    "diffusion_model.input_blocks.1.1.transformer_blocks.0."
                    "attn1.to_q"
                )
                np.savez(
                    source_path,
                    **{
                        f"{source_prefix}.input": source_input.numpy(),
                        f"{source_prefix}.output": source_output.numpy(),
                        "__aggregation__": np.array("mean"),
                    },
                )
                target_prefix = "diffusion_model.blocks.0.self_attn.q_proj"
                np.savez(
                    target_path,
                    **{
                        f"{target_prefix}.input": target_input.numpy(),
                        f"{target_prefix}.output": target_output.numpy(),
                        "__aggregation__": np.array("mean"),
                    },
                )
                converter.save_file(
                    {
                        (
                            "lora_unet_input_blocks_1_1_transformer_blocks_0_"
                            "attn1_to_q.lora_down.weight"
                        ): down,
                        (
                            "lora_unet_input_blocks_1_1_transformer_blocks_0_"
                            "attn1_to_q.lora_up.weight"
                        ): up,
                    },
                    lora_path,
                )

                converter.prepare_v3_activations(
                    source_path,
                    target_path,
                    lora_path,
                    output_path,
                    lora_strength=1.0,
                    truncate_unpaired_tail=True,
                )

                with np.load(output_path) as prepared:
                    self.assertEqual(
                        prepared["target.0.self_attn.q_proj.input"].shape[0],
                        source_rows,
                    )
        finally:
            converter.ANIMA_DIMS = original_dims

    def test_pairs_diffusers_lora_by_depth_when_capture_count_matches(self):
        capture_identity = ("input_blocks", 4, 1, 0)
        captures = {
            (capture_identity, "self_attn.q_proj"): {
                "input": "source.input",
                "output": "source.output",
            }
        }
        lora_identity = ("down_blocks_1", 1, 0, 0)
        tensors = {
            "down": torch.randn(1, 3),
            "up": torch.randn(2, 1),
        }
        loras = {(lora_identity, "self_attn.q_proj"): tensors}

        pairs = converter.pair_source_capture_and_lora(captures, loras)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], "self_attn.q_proj")
        self.assertEqual(pairs[0][1], capture_identity)
        self.assertIs(pairs[0][3], tensors)


if __name__ == "__main__":
    unittest.main()
