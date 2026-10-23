"""Complete PACE-Net model implementation.

PACE-Net encodes aligned skeleton maps and silhouettes, optionally constructs
an explicit silhouette-motion stream, assigns frame descriptors to constrained
soft gait phases, and classifies the resulting phase tokens. The A0/A1 and
uniform-bin switches are retained solely to reproduce the reported ablations;
the released final configuration selects the complete A2 architecture.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateBNNecks, SeparateFCs, SetBlockWrapper
from ..phase_modeling import (
    MotionResidualBuilder,
    SoftPhaseTokenizer,
    UniformTemporalTokenizer,
)


CLASS_TO_INDEX = {"negative": 0, "neutral": 1, "positive": 2}


def _as_numpy_batch(modality: object, seq_lengths: Optional[torch.Tensor]) -> np.ndarray:
    array = np.asarray(modality)
    if seq_lengths is not None and array.ndim >= 1 and array.shape[0] == 1:
        array = array[0][None, ...]
    return np.ascontiguousarray(array)


def _pad_aligned_visual_sequences(
    maps: torch.Tensor,
    silhouettes: torch.Tensor,
    seq_lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ordinary or concatenated OpenGait layouts to padded batches."""
    batch, frames = maps.shape[:2]
    if seq_lengths is None:
        mask = torch.ones(batch, frames, dtype=torch.bool, device=maps.device)
        return maps, silhouettes, mask

    lengths = [
        max(1, int(value))
        for value in seq_lengths.detach().cpu().reshape(-1).tolist()
    ]
    if batch == 1 and len(lengths) > 1:
        if sum(lengths) != frames:
            raise ValueError("concatenated seqL does not match the visual frame count")
        max_length = max(lengths)
        padded_maps = maps.new_zeros(len(lengths), max_length, *maps.shape[2:])
        padded_sils = silhouettes.new_zeros(
            len(lengths), max_length, *silhouettes.shape[2:]
        )
        valid_mask = torch.zeros(
            len(lengths), max_length, dtype=torch.bool, device=maps.device
        )
        offset = 0
        for index, length in enumerate(lengths):
            padded_maps[index, :length] = maps[0, offset:offset + length]
            padded_sils[index, :length] = silhouettes[0, offset:offset + length]
            valid_mask[index, :length] = True
            offset += length
        return padded_maps, padded_sils, valid_mask

    if len(lengths) != batch:
        raise ValueError(f"seqL has {len(lengths)} sequences but batch is {batch}")
    if max(lengths) > frames:
        raise ValueError("a seqL entry exceeds the padded visual frame count")
    valid_mask = (
        torch.arange(frames, device=maps.device)[None, :]
        < torch.as_tensor(lengths, device=maps.device)[:, None]
    )
    return maps, silhouettes, valid_mask


class PACENet(BaseModel):
    """Phase-Aware Constrained Encoding Network for scoliosis screening."""

    def build_network(self, model_cfg):
        self.ablation_stage = str(model_cfg.get("ablation_stage", "A2")).upper()
        stage_flags = {
            "A0": (False, False),
            "A1": (True, False),
            "A2": (True, True),
        }
        if self.ablation_stage not in stage_flags:
            raise ValueError("ablation_stage must be A0, A1 or A2")
        self.use_motion, self.use_phase = stage_flags[self.ablation_stage]
        self.motion_descriptor_mode = str(
            model_cfg.get("motion_descriptor_mode", "observed")
        ).lower()
        if self.motion_descriptor_mode not in {"observed", "zero"}:
            raise ValueError(
                "motion_descriptor_mode must be 'observed' or 'zero'"
            )
        if not self.use_motion and self.motion_descriptor_mode != "observed":
            raise ValueError(
                "motion_descriptor_mode can be changed only for a motion stage"
            )
        channels = int(model_cfg.get("feature_channels", 512))
        self.map_backbone = SetBlockWrapper(
            self.get_backbone(model_cfg["map_backbone_cfg"])
        )
        self.sil_backbone = SetBlockWrapper(
            self.get_backbone(model_cfg["sil_backbone_cfg"])
        )
        self.motion_backbone = (
            SetBlockWrapper(self.get_backbone(model_cfg["motion_backbone_cfg"]))
            if self.use_motion else None
        )
        backbone_names = ["map_backbone_cfg", "sil_backbone_cfg"]
        if self.use_motion:
            backbone_names.append("motion_backbone_cfg")
        for name in backbone_names:
            emitted = int(model_cfg[name].get("channels", [channels])[-1])
            if emitted != channels:
                raise ValueError(f"{name} must emit feature_channels={channels}")

        self.motion_builder = MotionResidualBuilder() if self.use_motion else None
        if self.use_phase:
            self.phase_assignment_mode = str(
                model_cfg.get("phase_assignment_mode", "soft")
            ).lower()
            tokenizer_classes = {
                "soft": SoftPhaseTokenizer,
                "uniform_bins": UniformTemporalTokenizer,
            }
            if self.phase_assignment_mode not in tokenizer_classes:
                raise ValueError(
                    "phase_assignment_mode must be 'soft' or 'uniform_bins'"
                )
            tokenizer_class = tokenizer_classes[self.phase_assignment_mode]
            self.phase_tokenizer = tokenizer_class(
                feature_dim=channels,
                num_modalities=3,
                num_phases=int(model_cfg.get("num_phases", 4)),
                hidden_dim=int(model_cfg.get("phase_hidden_dim", 256)),
                temperature=float(model_cfg.get("phase_temperature", 1.0)),
                dropout=float(model_cfg.get("phase_dropout", 0.1)),
                min_phase_mass_fraction=float(
                    model_cfg.get("phase_min_mass_fraction", 0.05)
                ),
                assignment_balance_strength=float(
                    model_cfg.get("phase_assignment_balance_strength", 0.5)
                ),
                balance_iterations=int(
                    model_cfg.get("phase_balance_iterations", 4)
                ),
                normalize_assignment_logits=bool(
                    model_cfg.get("phase_normalize_assignment_logits", True)
                ),
            )
            parts_num = self.phase_tokenizer.num_phases
            phase_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=int(model_cfg.get("phase_attention_heads", 8)),
                dim_feedforward=int(model_cfg.get("phase_ffn_dim", channels * 2)),
                dropout=float(model_cfg.get("phase_dropout", 0.1)),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.phase_context = nn.TransformerEncoder(
                phase_layer, num_layers=int(model_cfg.get("phase_layers", 1))
            )
        else:
            self.phase_assignment_mode = "none"
            self.phase_tokenizer = None
            self.phase_context = None
            parts_num = 1
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
        if int(model_cfg["SeparateFCs"].get("parts_num", -1)) != parts_num:
            raise ValueError(f"SeparateFCs.parts_num must equal {parts_num} for {self.ablation_stage}")
        if int(model_cfg["SeparateBNNecks"].get("parts_num", -1)) != parts_num:
            raise ValueError(
                f"SeparateBNNecks.parts_num must equal {parts_num} for {self.ablation_stage}"
            )

    def inputs_pretreament(self, inputs):
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        if len(seqs_batch) < 2:
            raise ValueError("PACE-Net expects skeleton-map and silhouette roots")
        sampled_seqL = (
            seqL_batch["sampled"] if isinstance(seqL_batch, dict) else seqL_batch
        )
        maps = _as_numpy_batch(seqs_batch[0], sampled_seqL)
        silhouettes = _as_numpy_batch(seqs_batch[1], sampled_seqL)
        if maps.ndim != 5 or maps.shape[2] != 2:
            raise ValueError(f"expected maps [B,T,2,H,W], got {maps.shape}")
        if silhouettes.ndim != 4:
            raise ValueError(f"expected silhouettes [B,T,H,W], got {silhouettes.shape}")
        if maps.shape[:2] != silhouettes.shape[:2]:
            raise ValueError("skeleton maps and silhouettes are not frame-aligned")

        maps_t = torch.from_numpy(maps.astype(np.float32) / 255.0).cuda()
        silhouettes_t = torch.from_numpy(
            silhouettes.astype(np.float32) / 255.0
        ).cuda()
        labs = torch.as_tensor(labs_batch, dtype=torch.long, device=maps_t.device)
        seqL = None if sampled_seqL is None else torch.as_tensor(
            np.asarray(sampled_seqL), dtype=torch.int32, device=maps_t.device
        )
        return [maps_t, silhouettes_t], labs, typs_batch, vies_batch, seqL

    @staticmethod
    def _class_indices(labels, device: torch.device) -> torch.Tensor:
        indices = []
        for label in labels:
            if isinstance(label, str):
                if label not in CLASS_TO_INDEX:
                    raise ValueError(f"unknown scoliosis class: {label}")
                indices.append(CLASS_TO_INDEX[label])
            else:
                value = int(label)
                if value not in (0, 1, 2):
                    raise ValueError(f"class index must be 0, 1 or 2, got {value}")
                indices.append(value)
        return torch.as_tensor(indices, dtype=torch.long, device=device)

    def forward(self, inputs):
        ipts, pids, labels, _, seqL = inputs
        maps, silhouettes = ipts
        if maps.ndim != 5 or silhouettes.ndim != 4:
            raise ValueError("forward expects maps [B,T,2,H,W] and sils [B,T,H,W]")

        maps, silhouettes, valid_mask = _pad_aligned_visual_sequences(
            maps, silhouettes, seqL
        )
        map_input = maps.transpose(1, 2).contiguous()
        sil_input = silhouettes.unsqueeze(1).contiguous()
        motion_input = (
            self.motion_builder(sil_input, valid_mask) if self.use_motion else None
        )

        map_features = self.map_backbone(map_input)
        sil_features = self.sil_backbone(sil_input)
        descriptor_streams = [
            map_features.mean(dim=(-1, -2)).transpose(1, 2),
            sil_features.mean(dim=(-1, -2)).transpose(1, 2),
        ]
        if self.use_motion:
            motion_features = self.motion_backbone(motion_input)
            motion_descriptors = motion_features.mean(dim=(-1, -2)).transpose(1, 2)
            if self.motion_descriptor_mode == "zero":
                # Capacity-matched no-motion control: execute the complete
                # motion path so parameter count, initialization and DDP graph
                # remain unchanged, but prevent motion evidence from reaching
                # the tokenizer or classifier.
                motion_descriptors = motion_descriptors * 0.0
            descriptor_streams.append(motion_descriptors)
        descriptors = torch.stack(descriptor_streams, dim=2)
        descriptors = descriptors * valid_mask[:, :, None, None].to(
            descriptors.dtype
        )

        phase = None
        if self.use_phase:
            phase = self.phase_tokenizer(descriptors, valid_mask)
            # Uniform fusion keeps the three aligned descriptor streams on the
            # same scale before phase-context encoding.
            phase_features = phase["tokens"].mean(dim=1)  # [B,K,D]
            phase_features = self.phase_context(phase_features)
            parts = phase_features.transpose(1, 2).contiguous()  # [B,D,K]
        else:
            # A0/A1 use the same mask-aware global temporal and uniform
            # modality pooling. Their only difference is the motion stream.
            valid_count = valid_mask.sum(dim=1).clamp_min(1).to(descriptors.dtype)
            global_modalities = descriptors.sum(dim=1) / valid_count[:, None, None]
            parts = global_modalities.mean(dim=1).unsqueeze(-1)  # [B,D,1]
        embeddings = self.FCs(parts)
        _, logits = self.BNNecks(embeddings)

        class_indices = self._class_indices(labels, logits.device)
        if pids.numel() != logits.size(0):
            # This can occur only in a concatenated inference batch. Triplet
            # features are ignored in eval, but preserving a valid shape keeps
            # the model interface deterministic.
            pids = torch.arange(logits.size(0), device=logits.device)

        preview_frames = min(sil_input.size(2), 8)
        training_feat = {
            "triplet": {"embeddings": embeddings, "labels": pids},
            "softmax": {"logits": logits, "labels": class_indices},
        }
        visual_summary = {
                "image/silhouette": sil_input[:, :, :preview_frames]
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, 1, sil_input.size(3), sil_input.size(4)),
        }
        inference_feat = {"embeddings": logits}
        if self.use_motion:
            visual_summary.update({
                "image/motion_difference": motion_input[:, :1, :preview_frames]
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, 1, motion_input.size(3), motion_input.size(4)),
            })
        if self.use_phase:
            training_feat["phase_regularization"] = {
                "assignments": phase["assignments"],
                "valid_mask": phase["valid_mask"],
            }
            visual_summary.update({
                "scalar/phase_min_occupancy": phase["occupancy"].min().detach(),
                "scalar/phase_max_occupancy": phase["occupancy"].max().detach(),
            })
            inference_feat.update({
                "phase_occupancy": phase["occupancy"],
                "phase_entropy": -(
                    phase["assignments"].float().clamp_min(1e-8)
                    * phase["assignments"].float().clamp_min(1e-8).log()
                ).sum(dim=-1).sum(dim=1, keepdim=True)
                / (
                    phase["valid_mask"].sum(dim=1, keepdim=True).clamp_min(1)
                    * math.log(self.phase_tokenizer.num_phases)
                ),
            })
            # Raw frame assignments are useful for the fixed-length training
            # smoke test, but cannot be concatenated across variable-length
            # subjects by OpenGait's inference loop.
            if self.training:
                inference_feat["phase_assignments"] = phase["assignments"]
        return {
            "training_feat": training_feat,
            "visual_summary": visual_summary,
            "inference_feat": inference_feat,
        }


__all__ = ["PACENet"]
