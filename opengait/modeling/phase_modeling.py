"""Phase-aware building blocks for multimodal scoliosis gait models.

The public seam deliberately contains two deep modules:

``MotionResidualBuilder``
    Converts a silhouette sequence into signed frame-to-frame and
    GEI-relative motion channels while respecting padded frames.

``SoftPhaseTokenizer``
    Learns a shared, end-to-end phase assignment and pools every modality
    with the same assignments.  This keeps temporal alignment behind one
    small interface and makes phase assignments directly inspectable.

``UniformTemporalTokenizer``
    Provides the capacity-matched fixed-bin control for ``SoftPhaseTokenizer``.
    Valid frames are assigned to contiguous, approximately equal temporal
    bins.  The assignment MLP is retained for exact parameter matching but has
    identically zero influence on the fixed assignments.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_mask(
    valid_mask: Optional[torch.Tensor], batch: int, frames: int, device: torch.device
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(batch, frames, dtype=torch.bool, device=device)
    if valid_mask.shape != (batch, frames):
        raise ValueError(
            f"valid_mask must be [B,T]={batch, frames}, got {tuple(valid_mask.shape)}"
        )
    return valid_mask.to(device=device, dtype=torch.bool)


class MotionResidualBuilder(nn.Module):
    """Build explicit motion evidence from normalized silhouette sequences.

    Interface
    ---------
    silhouettes:
        Float tensor ``[B,1,T,H,W]`` in ``[0,1]``.
    valid_mask:
        Optional boolean tensor ``[B,T]``. Invalid/padded frames are excluded
        from the GEI and are exactly zero in the returned tensor.
    returns:
        Tensor ``[B,2,T,H,W]``. Channel 0 is signed temporal difference;
        channel 1 is the signed residual from the valid-frame GEI.
    """

    def forward(
        self,
        silhouettes: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if silhouettes.ndim != 5 or silhouettes.size(1) != 1:
            raise ValueError(
                "silhouettes must have shape [B,1,T,H,W], got "
                f"{tuple(silhouettes.shape)}"
            )
        if not silhouettes.is_floating_point():
            raise TypeError("silhouettes must be a floating-point tensor in [0,1]")

        batch, _, frames, _, _ = silhouettes.shape
        mask = _validate_mask(valid_mask, batch, frames, silhouettes.device)
        spatial_mask = mask[:, None, :, None, None].to(silhouettes.dtype)

        # Signed difference preserves whether a body region is entering or
        # leaving the silhouette. The first frame has no predecessor.
        temporal_difference = torch.zeros_like(silhouettes)
        if frames > 1:
            pair_mask = (mask[:, 1:] & mask[:, :-1])[:, None, :, None, None]
            temporal_difference[:, :, 1:] = (
                silhouettes[:, :, 1:] - silhouettes[:, :, :-1]
            ) * pair_mask.to(silhouettes.dtype)

        valid_count = spatial_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        gait_energy_image = (
            silhouettes * spatial_mask
        ).sum(dim=2, keepdim=True) / valid_count
        gei_residual = (silhouettes - gait_energy_image) * spatial_mask

        motion = torch.cat([temporal_difference, gei_residual], dim=1)
        return torch.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)


class SoftPhaseTokenizer(nn.Module):
    """Pool aligned modality features into shared learnable gait phases.

    All modalities use one assignment distribution. This is the important
    invariant: phase ``k`` in silhouette, motion and pose always describes the
    same set of frames for a sample.

    Input features use ``[B,T,M,D]`` and the returned tokens use
    ``[B,M,K,D]``. The returned dictionary also exposes assignments and
    occupancy for regularisation, diagnostics and phase visualisation.
    """

    def __init__(
        self,
        feature_dim: int,
        num_modalities: int = 3,
        num_phases: int = 4,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        dropout: float = 0.1,
        min_phase_mass_fraction: float = 0.05,
        assignment_balance_strength: float = 0.5,
        balance_iterations: int = 4,
        normalize_assignment_logits: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_modalities <= 0 or num_phases < 2:
            raise ValueError("feature_dim/modalities must be positive and phases >= 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= min_phase_mass_fraction < 1.0 / num_phases:
            raise ValueError("min_phase_mass_fraction must be in [0, 1/num_phases)")
        if not 0.0 <= assignment_balance_strength <= 1.0:
            raise ValueError("assignment_balance_strength must be in [0,1]")
        if balance_iterations < 1:
            raise ValueError("balance_iterations must be positive")

        self.feature_dim = int(feature_dim)
        self.num_modalities = int(num_modalities)
        self.num_phases = int(num_phases)
        self.temperature = float(temperature)
        self.min_phase_mass_fraction = float(min_phase_mass_fraction)
        self.assignment_balance_strength = float(assignment_balance_strength)
        self.balance_iterations = int(balance_iterations)
        self.normalize_assignment_logits = bool(normalize_assignment_logits)
        joint_dim = self.feature_dim * self.num_modalities

        self.modality_norms = nn.ModuleList(
            nn.LayerNorm(self.feature_dim) for _ in range(self.num_modalities)
        )
        self.assignment_network = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_phases),
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError(
                f"features must have shape [B,T,M,D], got {tuple(features.shape)}"
            )
        batch, frames, modalities, feature_dim = features.shape
        if modalities != self.num_modalities or feature_dim != self.feature_dim:
            raise ValueError(
                "feature layout does not match tokenizer configuration: "
                f"expected M,D={self.num_modalities, self.feature_dim}, "
                f"got {modalities, feature_dim}"
            )
        mask = _validate_mask(valid_mask, batch, frames, features.device)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sequence must contain at least one valid frame")

        normalized = torch.stack(
            [self.modality_norms[index](features[:, :, index])
             for index in range(self.num_modalities)],
            dim=2,
        )
        assignment_input = normalized.reshape(batch, frames, -1)
        assignment_logits = self.assignment_network(assignment_input)
        assignment_scores = (
            F.layer_norm(assignment_logits.float(), (self.num_phases,))
            if self.normalize_assignment_logits
            else assignment_logits.float()
        )
        float_mask = mask.unsqueeze(-1).float()
        raw_assignments = torch.softmax(
            assignment_scores / self.temperature, dim=-1
        ) * float_mask
        if self.assignment_balance_strength > 0:
            # A small positive transport seed is necessary once extreme logits
            # underflow a phase to exact zero; multiplicative normalization
            # cannot revive a zero column on its own.
            balanced = raw_assignments + 1e-4 * float_mask
            valid_count_float = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            target_mass = (valid_count_float / self.num_phases).unsqueeze(-1)
            for _ in range(self.balance_iterations):
                balanced = balanced * (
                    target_mass / balanced.sum(dim=1, keepdim=True).clamp_min(1e-6)
                )
                balanced = balanced / balanced.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                balanced = balanced * float_mask
            assignments = (
                (1.0 - self.assignment_balance_strength) * raw_assignments
                + self.assignment_balance_strength * balanced
            )
        else:
            assignments = raw_assignments
        assignments = assignments.to(features.dtype)

        phase_mass = assignments.sum(dim=1)  # [B,K]
        valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        # A nearly empty phase must not amplify its tiny weighted numerator by
        # division through 1e-6. Attenuating under-occupied tokens removes that
        # positive feedback while leaving the true occupancy visible to the
        # balance regularizer and diagnostics.
        minimum_mass = (
            valid_count.to(phase_mass.dtype) * self.min_phase_mass_fraction
        )
        safe_phase_mass = phase_mass.clamp_min(minimum_mass).clamp_min(1e-6)
        # [B,T,K] x [B,T,M,D] -> [B,M,K,D]
        tokens = torch.einsum("btk,btmd->bmkd", assignments, features)
        tokens = tokens / safe_phase_mass[:, None, :, None]

        occupancy = phase_mass / valid_count.to(phase_mass.dtype)
        return {
            "tokens": torch.nan_to_num(tokens),
            "assignments": assignments,
            "assignment_logits": assignment_logits,
            "occupancy": occupancy,
            "valid_mask": mask,
        }


class UniformTemporalTokenizer(SoftPhaseTokenizer):
    """Pool aligned features into fixed contiguous temporal bins.

    This is a parameter-matched control for :class:`SoftPhaseTokenizer`.
    It inherits the same normalization layers and assignment MLP, and executes
    that MLP during every forward pass, so the two tokenizers have identical
    trainable parameter counts and consume randomness in the same places.
    The logits are linked to the returned assignments with a zero multiplier:
    all matching parameters stay in the autograd graph (important for DDP),
    while the fixed temporal partition is mathematically unaffected.

    For a sequence with ``L`` valid frames, valid-frame rank ``r`` is assigned
    to bin ``floor(r * K / L)``.  This makes the control length-aware, excludes
    padding, preserves order, and differs from the learned tokenizer only in
    the frame-to-token assignment rule.
    """

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError(
                f"features must have shape [B,T,M,D], got {tuple(features.shape)}"
            )
        batch, frames, modalities, feature_dim = features.shape
        if modalities != self.num_modalities or feature_dim != self.feature_dim:
            raise ValueError(
                "feature layout does not match tokenizer configuration: "
                f"expected M,D={self.num_modalities, self.feature_dim}, "
                f"got {modalities, feature_dim}"
            )
        mask = _validate_mask(valid_mask, batch, frames, features.device)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sequence must contain at least one valid frame")

        # Execute the same capacity-matching path as SoftPhaseTokenizer.  Its
        # output is deliberately prevented from changing the fixed bins below.
        normalized = torch.stack(
            [
                self.modality_norms[index](features[:, :, index])
                for index in range(self.num_modalities)
            ],
            dim=2,
        )
        assignment_input = normalized.reshape(batch, frames, -1)
        assignment_logits = self.assignment_network(assignment_input)

        valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        valid_rank = mask.to(torch.long).cumsum(dim=1) - 1
        phase_index = torch.div(
            valid_rank * self.num_phases,
            valid_count,
            rounding_mode="floor",
        ).clamp(min=0, max=self.num_phases - 1)
        fixed_assignments = F.one_hot(
            phase_index, num_classes=self.num_phases
        ).to(dtype=torch.float32)
        fixed_assignments = fixed_assignments * mask.unsqueeze(-1).float()

        # Zero-valued dependency keeps every capacity-matching parameter in
        # autograd/DDP without giving it any functional influence.
        assignments = fixed_assignments + assignment_logits.float() * 0.0
        assignments = assignments.to(features.dtype)

        phase_mass = assignments.sum(dim=1)
        minimum_mass = (
            valid_count.to(phase_mass.dtype) * self.min_phase_mass_fraction
        )
        safe_phase_mass = phase_mass.clamp_min(minimum_mass).clamp_min(1e-6)
        tokens = torch.einsum("btk,btmd->bmkd", assignments, features)
        tokens = tokens / safe_phase_mass[:, None, :, None]
        occupancy = phase_mass / valid_count.to(phase_mass.dtype)

        return {
            "tokens": torch.nan_to_num(tokens),
            "assignments": assignments,
            "assignment_logits": assignment_logits,
            "occupancy": occupancy,
            "valid_mask": mask,
        }


__all__ = [
    "MotionResidualBuilder",
    "SoftPhaseTokenizer",
    "UniformTemporalTokenizer",
]
