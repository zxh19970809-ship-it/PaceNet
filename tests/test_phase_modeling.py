import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "opengait"))

from modeling.losses.phase import PhaseRegularizationLoss  # noqa: E402
from modeling.phase_modeling import (  # noqa: E402
    MotionResidualBuilder,
    SoftPhaseTokenizer,
    UniformTemporalTokenizer,
)


class MotionResidualBuilderTests(unittest.TestCase):
    def test_static_sequence_is_zero_and_padding_stays_zero(self):
        silhouettes = torch.ones(2, 1, 5, 8, 6)
        valid_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        motion = MotionResidualBuilder()(silhouettes, valid_mask)
        self.assertEqual(tuple(motion.shape), (2, 2, 5, 8, 6))
        self.assertTrue(torch.equal(motion[0], torch.zeros_like(motion[0])))
        self.assertTrue(torch.equal(motion[1, :, 3:], torch.zeros_like(motion[1, :, 3:])))

    def test_moving_sequence_produces_signed_evidence(self):
        silhouettes = torch.zeros(1, 1, 4, 5, 5)
        silhouettes[:, :, 0, :, :2] = 1
        silhouettes[:, :, 1, :, 1:3] = 1
        silhouettes[:, :, 2, :, 2:4] = 1
        silhouettes[:, :, 3, :, 3:] = 1
        motion = MotionResidualBuilder()(silhouettes)
        self.assertGreater(float(motion[:, 0].abs().sum()), 0.0)
        self.assertGreater(float(motion[:, 1].abs().sum()), 0.0)
        self.assertLess(float(motion[:, 0].min()), 0.0)
        self.assertGreater(float(motion[:, 0].max()), 0.0)


class SoftPhaseTokenizerTests(unittest.TestCase):
    def test_shapes_masking_and_gradients(self):
        torch.manual_seed(7)
        features = torch.randn(3, 9, 3, 16, requires_grad=True)
        valid_mask = torch.tensor(
            [
                [True] * 9,
                [True] * 7 + [False] * 2,
                [True] * 5 + [False] * 4,
            ]
        )
        tokenizer = SoftPhaseTokenizer(
            feature_dim=16,
            num_modalities=3,
            num_phases=4,
            hidden_dim=24,
            dropout=0.0,
        )
        output = tokenizer(features, valid_mask)
        self.assertEqual(tuple(output["tokens"].shape), (3, 3, 4, 16))
        self.assertEqual(tuple(output["assignments"].shape), (3, 9, 4))
        self.assertTrue(
            torch.allclose(
                output["assignments"][valid_mask].sum(dim=-1),
                torch.ones(int(valid_mask.sum())),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                output["assignments"][~valid_mask],
                torch.zeros_like(output["assignments"][~valid_mask]),
            )
        )
        self.assertTrue(
            torch.allclose(output["occupancy"].sum(dim=-1), torch.ones(3), atol=1e-5)
        )

        phase_loss = PhaseRegularizationLoss()
        loss, info = phase_loss(output["assignments"], output["valid_mask"])
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("balance", info)
        (output["tokens"].square().mean() + loss).backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_balance_penalizes_collapsed_assignments(self):
        valid_mask = torch.ones(2, 6, dtype=torch.bool)
        uniform = torch.full((2, 6, 4), 0.25)
        collapsed = torch.zeros(2, 6, 4)
        collapsed[..., 0] = 1.0
        loss_fn = PhaseRegularizationLoss(
            smooth_weight=0.0,
            balance_weight=1.0,
            sharpness_weight=0.0,
            cycle_weight=0.0,
        )
        uniform_loss, _ = loss_fn(uniform, valid_mask)
        collapsed_loss, _ = loss_fn(collapsed, valid_mask)
        self.assertGreater(float(collapsed_loss), float(uniform_loss) + 1.0)

    def test_low_mass_phase_is_attenuated_without_hiding_occupancy(self):
        tokenizer = SoftPhaseTokenizer(
            feature_dim=2, num_modalities=1, num_phases=4,
            hidden_dim=4, dropout=0.0, min_phase_mass_fraction=0.1,
            assignment_balance_strength=0.0,
            normalize_assignment_logits=False,
        )
        features = torch.ones(1, 5, 1, 2)
        with torch.no_grad():
            tokenizer.assignment_network[-1].weight.zero_()
            tokenizer.assignment_network[-1].bias.copy_(
                torch.tensor([8.0, -8.0, -8.0, -8.0])
            )
        output = tokenizer(features)
        self.assertLess(float(output["occupancy"][0, 1].detach()), 1e-5)
        self.assertLess(float(output["tokens"][0, 0, 1].max().detach()), 1e-4)
        self.assertTrue(torch.isfinite(output["tokens"]).all())

    def test_balanced_assignment_mix_prevents_unused_phases(self):
        tokenizer = SoftPhaseTokenizer(
            feature_dim=2, num_modalities=1, num_phases=4,
            hidden_dim=4, dropout=0.0, assignment_balance_strength=0.5,
        )
        features = torch.ones(1, 12, 1, 2)
        with torch.no_grad():
            tokenizer.assignment_network[-1].weight.zero_()
            tokenizer.assignment_network[-1].bias.copy_(
                torch.tensor([12.0, -12.0, -12.0, -12.0])
            )
        occupancy = tokenizer(features)["occupancy"]
        self.assertGreater(float(occupancy.min().detach()), 0.10)
        self.assertLess(float(occupancy.max().detach()), 0.65)


class UniformTemporalTokenizerTests(unittest.TestCase):
    def test_contiguous_bins_respect_each_sequence_length_and_padding(self):
        features = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8, 1, 1)
        valid_mask = torch.tensor(
            [
                [True] * 8,
                [True] * 5 + [False] * 3,
            ]
        )
        tokenizer = UniformTemporalTokenizer(
            feature_dim=1,
            num_modalities=1,
            num_phases=4,
            hidden_dim=4,
            dropout=0.0,
        )
        output = tokenizer(features, valid_mask)

        expected_first = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        expected_second = torch.tensor([0, 0, 1, 2, 3])
        self.assertTrue(
            torch.equal(output["assignments"][0].argmax(dim=-1), expected_first)
        )
        self.assertTrue(
            torch.equal(
                output["assignments"][1, :5].argmax(dim=-1), expected_second
            )
        )
        self.assertTrue(
            torch.equal(
                output["assignments"][1, 5:],
                torch.zeros_like(output["assignments"][1, 5:]),
            )
        )
        self.assertTrue(
            torch.allclose(
                output["tokens"][0, 0, :, 0],
                torch.tensor([0.5, 2.5, 4.5, 6.5]),
            )
        )
        self.assertTrue(
            torch.allclose(
                output["tokens"][1, 0, :, 0],
                torch.tensor([8.5, 10.0, 11.0, 12.0]),
            )
        )

    def test_parameter_count_matches_soft_tokenizer_exactly(self):
        kwargs = dict(
            feature_dim=16,
            num_modalities=3,
            num_phases=4,
            hidden_dim=24,
            dropout=0.1,
        )
        soft = SoftPhaseTokenizer(**kwargs)
        uniform = UniformTemporalTokenizer(**kwargs)
        soft_count = sum(parameter.numel() for parameter in soft.parameters())
        uniform_count = sum(parameter.numel() for parameter in uniform.parameters())
        self.assertEqual(soft_count, uniform_count)
        self.assertEqual(
            list(soft.state_dict().keys()), list(uniform.state_dict().keys())
        )

    def test_capacity_matcher_is_in_graph_but_cannot_change_fixed_bins(self):
        torch.manual_seed(11)
        features = torch.randn(2, 8, 3, 16, requires_grad=True)
        tokenizer = UniformTemporalTokenizer(
            feature_dim=16,
            num_modalities=3,
            num_phases=4,
            hidden_dim=24,
            dropout=0.0,
        )
        first = tokenizer(features)
        with torch.no_grad():
            for parameter in tokenizer.parameters():
                parameter.add_(100.0 * torch.randn_like(parameter))
        second = tokenizer(features)
        self.assertTrue(torch.equal(first["assignments"], second["assignments"]))

        second["tokens"].square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        parameter_gradients = [
            parameter.grad
            for parameter in tokenizer.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in parameter_gradients))
        self.assertTrue(
            all(torch.count_nonzero(gradient) == 0 for gradient in parameter_gradients)
        )


if __name__ == "__main__":
    unittest.main()
