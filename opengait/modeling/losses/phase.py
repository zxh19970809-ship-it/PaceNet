"""Regularisation for end-to-end gait phase discovery."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import BaseLoss


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class PhaseRegularizationLoss(BaseLoss):
    """Prevent phase jitter and collapse while retaining cyclic progression."""

    def __init__(
        self,
        smooth_weight: float = 1.0,
        balance_weight: float = 0.2,
        sharpness_weight: float = 0.02,
        cycle_weight: float = 0.1,
        loss_term_weight: float = 1.0,
    ) -> None:
        super().__init__(loss_term_weight)
        for name, value in {
            "smooth_weight": smooth_weight,
            "balance_weight": balance_weight,
            "sharpness_weight": sharpness_weight,
            "cycle_weight": cycle_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.smooth_weight = float(smooth_weight)
        self.balance_weight = float(balance_weight)
        self.sharpness_weight = float(sharpness_weight)
        self.cycle_weight = float(cycle_weight)

    def forward(
        self, assignments: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if assignments.ndim != 3:
            raise ValueError("assignments must be [B,T,K]")
        if valid_mask.shape != assignments.shape[:2]:
            raise ValueError("valid_mask must align with assignments [B,T]")

        probabilities = assignments.float().clamp_min(0.0)
        mask = valid_mask.to(device=probabilities.device, dtype=torch.bool)
        batch, frames, phases = probabilities.shape
        valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1)

        occupancy = probabilities.sum(dim=1) / valid_count.to(probabilities.dtype)
        occupancy = occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        balance = (
            occupancy * (occupancy * phases).clamp_min(1e-8).log()
        ).sum(dim=-1).mean()

        entropy = -(
            probabilities.clamp_min(1e-8).log() * probabilities
        ).sum(dim=-1) / math.log(phases)
        sharpness = _masked_mean(entropy, mask)

        if frames > 1:
            pair_mask = mask[:, 1:] & mask[:, :-1]
            smooth_per_pair = (
                probabilities[:, 1:] - probabilities[:, :-1]
            ).square().sum(dim=-1)
            smooth = _masked_mean(smooth_per_pair, pair_mask)

            angles = torch.arange(
                phases, dtype=probabilities.dtype, device=probabilities.device
            ) * (2.0 * math.pi / phases)
            phase_x = probabilities @ torch.cos(angles)
            phase_y = probabilities @ torch.sin(angles)
            phase_vector = F.normalize(
                torch.stack([phase_x, phase_y], dim=-1), dim=-1, eps=1e-6
            )
            cross = (
                phase_vector[:, :-1, 0] * phase_vector[:, 1:, 1]
                - phase_vector[:, :-1, 1] * phase_vector[:, 1:, 0]
            )
            signed_sum = (cross * pair_mask.to(cross.dtype)).sum(dim=1)
            direction = torch.where(
                signed_sum >= 0,
                torch.ones_like(signed_sum),
                -torch.ones_like(signed_sum),
            ).detach()
            cycle = _masked_mean(
                F.relu(-direction[:, None] * cross), pair_mask
            )
        else:
            smooth = probabilities.new_zeros(())
            cycle = probabilities.new_zeros(())

        total = (
            self.smooth_weight * smooth
            + self.balance_weight * balance
            + self.sharpness_weight * sharpness
            + self.cycle_weight * cycle
        )
        info = {
            "loss": total.detach(),
            "smooth": smooth.detach(),
            "balance": balance.detach(),
            "sharpness": sharpness.detach(),
            "cycle": cycle.detach(),
            "min_occupancy": occupancy.min().detach(),
            "max_occupancy": occupancy.max().detach(),
        }
        return total, info


__all__ = ["PhaseRegularizationLoss"]
