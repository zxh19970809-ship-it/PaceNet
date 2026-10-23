"""Two-stream temporal baselines for the LIVE-GaitNeuroKids experiment.

Inputs use the ordinary video layout ``[B,T,C,H,W]``.  Skeleton maps and
silhouettes are encoded by separate OpenGait-compatible ResNet9 frame
encoders.  The three four-token models share the same phase context and heads;
only their frame-to-token assignment rule changes.

The phase tokenizers are imported from the verified OpenGait implementation
when it is available.  ``OPENGAIT_ROOT`` may point at another checkout; the
default is the root of this source package.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet


def _load_phase_tokenizers():
    try:
        from opengait.modeling.phase_modeling import (
            SoftPhaseTokenizer,
            UniformTemporalTokenizer,
        )

        return SoftPhaseTokenizer, UniformTemporalTokenizer
    except ModuleNotFoundError:
        root = Path(
            os.environ.get("OPENGAIT_ROOT", str(Path(__file__).resolve().parents[2]))
        )
        source = root / "opengait" / "modeling" / "phase_modeling.py"
        if not source.is_file():
            raise ModuleNotFoundError(
                "Could not import OpenGait phase_modeling.py. Install OpenGait, "
                "set OPENGAIT_ROOT, or make the verified checkout available at "
                + str(Path(__file__).resolve().parents[2]) + "."
            )
        spec = importlib.util.spec_from_file_location(
            "live_gait_phase_modeling", source
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load phase tokenizer source: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SoftPhaseTokenizer, module.UniformTemporalTokenizer


SoftPhaseTokenizer, UniformTemporalTokenizer = _load_phase_tokenizers()


CLASS_TO_INDEX = {"CP": 0, "ITW": 1, "TD": 2}
NUM_CLASSES = 3
NUM_PHASES = 4
FEATURE_DIM = 512
EMBEDDING_DIM = 256


class ResNet9FrameEncoder(ResNet):
    """OpenGait's ResNet9 layout, with configurable input channels."""

    def __init__(
        self,
        in_channel: int,
        channels: tuple[int, ...] | list[int] = (64, 128, 256, 512),
        layers: tuple[int, ...] | list[int] = (1, 1, 1, 1),
        strides: tuple[int, ...] | list[int] = (1, 2, 2, 1),
        maxpool: bool = False,
    ) -> None:
        channels = tuple(int(value) for value in channels)
        layers = tuple(int(value) for value in layers)
        strides = tuple(int(value) for value in strides)
        if len(channels) != 4 or len(layers) != 4 or len(strides) != 4:
            raise ValueError("ResNet9 requires four channels, layer counts and strides")
        if channels[-1] != FEATURE_DIM:
            raise ValueError("Both ResNet9 streams must emit D=512")
        if any(value < 1 for value in layers):
            raise ValueError("Each ResNet9 stage must contain at least one block")

        self.maxpool_flag = bool(maxpool)
        super().__init__(BasicBlock, list(layers))
        self.fc = None
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = nn.Conv2d(
            int(in_channel), self.inplanes, kernel_size=3, stride=1, padding=1,
            bias=False,
        )
        self.layer1 = self._make_layer(
            BasicBlock, channels[0], layers[0], stride=strides[0]
        )
        self.layer2 = self._make_layer(
            BasicBlock, channels[1], layers[1], stride=strides[1]
        )
        self.layer3 = self._make_layer(
            BasicBlock, channels[2], layers[2], stride=strides[2]
        )
        self.layer4 = self._make_layer(
            BasicBlock, channels[3], layers[3], stride=strides[3]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        if self.maxpool_flag:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class SeparatePartProjection(nn.Module):
    """One independent linear projection per temporal part."""

    def __init__(self, parts: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(parts, in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,Cin,P] x [P,Cin,Cout] -> [B,Cout,P]
        return torch.einsum("bcp,pco->bop", x, self.weight)


class SeparatePartBNClassifier(nn.Module):
    """Match OpenGait's SeparateBNNecks output layout ``[B,C,P]``."""

    def __init__(self, parts: int, in_channels: int, class_num: int) -> None:
        super().__init__()
        self.parts = int(parts)
        self.in_channels = int(in_channels)
        self.class_num = int(class_num)
        self.bn = nn.BatchNorm1d(self.in_channels * self.parts)
        self.weight = nn.Parameter(
            torch.empty(self.parts, self.in_channels, self.class_num)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        x = self.bn(x.reshape(batch, -1)).reshape(
            batch, self.in_channels, self.parts
        )
        feature = F.normalize(x.transpose(1, 2), dim=-1)
        weight = F.normalize(self.weight, dim=1)
        logits = torch.einsum("bpc,pck->bpk", feature, weight)
        return feature.transpose(1, 2).contiguous(), logits.transpose(1, 2).contiguous()


def _make_encoder(config: Optional[dict[str, Any]], expected_channels: int) -> ResNet9FrameEncoder:
    values = dict(config or {})
    kind = values.pop("type", "ResNet9")
    block = values.pop("block", "BasicBlock")
    input_channels = int(values.pop("in_channel", expected_channels))
    channels = values.pop("channels", (64, 128, 256, 512))
    layers = values.pop("layers", (1, 1, 1, 1))
    strides = values.pop("strides", (1, 2, 2, 1))
    maxpool = values.pop("maxpool", False)
    if values:
        raise ValueError(f"Unsupported ResNet9 config keys: {sorted(values)}")
    if kind != "ResNet9" or block != "BasicBlock":
        raise ValueError("The experiment requires ResNet9 with BasicBlock")
    if input_channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} input channels, got {input_channels}"
        )
    return ResNet9FrameEncoder(
        in_channel=input_channels,
        channels=channels,
        layers=layers,
        strides=strides,
        maxpool=maxpool,
    )


class DualStreamTemporalClassifier(nn.Module):
    """CSPT, fixed-bin, unconstrained-assignment, or global-pooling model.

    ``forward`` accepts a skeleton-map tensor and silhouette tensor, both in
    ``[B,T,C,H,W]`` layout, plus an optional boolean ``valid_mask`` of shape
    ``[B,T]``. Inputs should be floating point and already scaled consistently
    by the data pipeline; integer image tensors are converted from uint8 to
    ``[0,1]``. The optional subject/class labels add the OpenGait-compatible
    ``training_feat`` block used by the existing triplet, softmax, and phase
    regularization losses.
    """

    VARIANTS = {"cspt", "uniform", "unconstrained", "global"}

    def __init__(
        self,
        variant: str,
        *,
        map_backbone_cfg: Optional[dict[str, Any]] = None,
        sil_backbone_cfg: Optional[dict[str, Any]] = None,
        feature_channels: int = FEATURE_DIM,
        num_phases: int = NUM_PHASES,
        num_classes: int = NUM_CLASSES,
        phase_hidden_dim: int = 256,
        phase_temperature: float = 1.0,
        phase_dropout: float = 0.1,
        phase_min_mass_fraction: float = 0.05,
        phase_assignment_balance_strength: float = 0.5,
        phase_balance_iterations: int = 4,
        phase_normalize_assignment_logits: bool = True,
        phase_attention_heads: int = 8,
        phase_ffn_dim: int = 1024,
        phase_layers: int = 1,
        embedding_channels: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.variant = str(variant).lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}")
        if feature_channels != FEATURE_DIM or num_phases != NUM_PHASES:
            raise ValueError("The frozen experiment contract is D=512 and K=4")
        if num_classes != NUM_CLASSES:
            raise ValueError("The experiment has CP, ITW and TD (three classes)")
        if embedding_channels <= 0:
            raise ValueError("embedding_channels must be positive")

        self.feature_channels = int(feature_channels)
        self.num_phases = int(num_phases)
        self.num_classes = int(num_classes)
        self.embedding_channels = int(embedding_channels)
        self.map_backbone = _make_encoder(map_backbone_cfg, expected_channels=2)
        self.sil_backbone = _make_encoder(sil_backbone_cfg, expected_channels=1)

        if self.variant == "global":
            parts = 1
            self.phase_tokenizer = None
            self.phase_context = None
        else:
            parts = self.num_phases
            tokenizer_type = (
                UniformTemporalTokenizer
                if self.variant == "uniform"
                else SoftPhaseTokenizer
            )
            unconstrained = self.variant == "unconstrained"
            self.phase_tokenizer = tokenizer_type(
                feature_dim=self.feature_channels,
                num_modalities=2,
                num_phases=self.num_phases,
                hidden_dim=phase_hidden_dim,
                temperature=phase_temperature,
                dropout=phase_dropout,
                min_phase_mass_fraction=phase_min_mass_fraction,
                assignment_balance_strength=(
                    0.0 if unconstrained else phase_assignment_balance_strength
                ),
                balance_iterations=phase_balance_iterations,
                normalize_assignment_logits=(
                    False if unconstrained else phase_normalize_assignment_logits
                ),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=self.feature_channels,
                nhead=phase_attention_heads,
                dim_feedforward=phase_ffn_dim,
                dropout=phase_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.phase_context = nn.TransformerEncoder(
                layer, num_layers=phase_layers
            )

        self.projection = SeparatePartProjection(
            parts, self.feature_channels, self.embedding_channels
        )
        self.classifier = SeparatePartBNClassifier(
            parts, self.embedding_channels, self.num_classes
        )

    @staticmethod
    def _as_float_video(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError(f"{name} must be [B,T,C,H,W], got {tuple(value.shape)}")
        if not value.is_floating_point():
            value = value.float() / 255.0
        return value

    @staticmethod
    def _class_targets(
        labels: Any, batch: int, device: torch.device
    ) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            targets = labels.to(device=device, dtype=torch.long).reshape(-1)
        else:
            values = list(labels)
            targets = torch.as_tensor(
                [CLASS_TO_INDEX[value] if isinstance(value, str) else int(value)
                 for value in values],
                dtype=torch.long,
                device=device,
            )
        if targets.numel() != batch:
            raise ValueError("class_labels must contain one value per sequence")
        if bool(((targets < 0) | (targets >= NUM_CLASSES)).any()):
            raise ValueError("class labels must map to CP=0, ITW=1, TD=2")
        return targets

    @staticmethod
    def _subject_targets(
        subject_ids: Any, batch: int, device: torch.device
    ) -> torch.Tensor:
        targets = torch.as_tensor(subject_ids, dtype=torch.long, device=device).reshape(-1)
        if targets.numel() != batch:
            raise ValueError("subject_ids must contain one value per sequence")
        return targets

    @staticmethod
    def _encode_frames(
        encoder: nn.Module, video: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        spatial = encoder(video.reshape(batch * frames, channels, height, width))
        descriptors = spatial.mean(dim=(-1, -2)).reshape(batch, frames, -1)
        return descriptors * valid_mask.unsqueeze(-1).to(descriptors.dtype)

    def forward(
        self,
        skeleton_map: torch.Tensor,
        silhouette: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        *,
        subject_ids: Optional[Any] = None,
        class_labels: Optional[Any] = None,
    ) -> dict[str, Any]:
        skeleton_map = self._as_float_video(skeleton_map, "skeleton_map")
        silhouette = self._as_float_video(silhouette, "silhouette")
        if skeleton_map.shape[:2] != silhouette.shape[:2]:
            raise ValueError("skeleton maps and silhouettes must be frame-aligned")
        if skeleton_map.size(2) != 2 or silhouette.size(2) != 1:
            raise ValueError("skeleton_map must have C=2 and silhouette must have C=1")
        batch, frames = skeleton_map.shape[:2]
        if valid_mask is None:
            mask = torch.ones(batch, frames, dtype=torch.bool, device=skeleton_map.device)
        else:
            if valid_mask.shape != (batch, frames):
                raise ValueError(
                    f"valid_mask must be [B,T]={batch, frames}, "
                    f"got {tuple(valid_mask.shape)}"
                )
            mask = valid_mask.to(device=skeleton_map.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sequence must contain at least one valid frame")

        map_features = self._encode_frames(self.map_backbone, skeleton_map, mask)
        sil_features = self._encode_frames(self.sil_backbone, silhouette, mask)
        descriptors = torch.stack([map_features, sil_features], dim=2)  # [B,T,2,512]

        phase: Optional[dict[str, torch.Tensor]] = None
        if self.variant == "global":
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(
                descriptors.dtype
            )
            temporal_mean = descriptors.sum(dim=1) / denominator.unsqueeze(-1)
            pooled = temporal_mean.mean(dim=1, keepdim=True)  # [B,1,512]
            parts = pooled.transpose(1, 2).contiguous()  # [B,512,1]
        else:
            phase = self.phase_tokenizer(descriptors, mask)
            # Each modality uses the same assignments; average the two aligned
            # 512-D phase tokens before the common four-token phase context.
            tokens = phase["tokens"].mean(dim=1)  # [B,4,512]
            tokens = self.phase_context(tokens)
            parts = tokens.transpose(1, 2).contiguous()  # [B,512,4]

        projected = self.projection(parts)
        embeddings, logits = self.classifier(projected)
        output: dict[str, Any] = {
            "embeddings": embeddings,
            "logits": logits,
            "prediction_logits": logits.mean(dim=-1),
            "phase": phase,
        }

        if (subject_ids is None) != (class_labels is None):
            raise ValueError("provide both subject_ids and class_labels for training")
        if subject_ids is not None:
            pids = self._subject_targets(subject_ids, batch, logits.device)
            targets = self._class_targets(class_labels, batch, logits.device)
            training_feat: dict[str, Any] = {
                "triplet": {"embeddings": embeddings, "labels": pids},
                "softmax": {"logits": logits, "labels": targets},
            }
            if self.variant in {"cspt", "uniform"}:
                training_feat["phase_regularization"] = {
                    "assignments": phase["assignments"],
                    "valid_mask": phase["valid_mask"],
                }
            output["training_feat"] = training_feat
        return output


class CSPTModel(DualStreamTemporalClassifier):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("cspt", **kwargs)


class UniformPhaseBinningModel(DualStreamTemporalClassifier):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("uniform", **kwargs)


class UnconstrainedAssignmentModel(DualStreamTemporalClassifier):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("unconstrained", **kwargs)


class GlobalPoolingModel(DualStreamTemporalClassifier):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("global", **kwargs)


MODEL_REGISTRY = {
    "cspt": CSPTModel,
    "uniform": UniformPhaseBinningModel,
    "unconstrained": UnconstrainedAssignmentModel,
    "global": GlobalPoolingModel,
}


def build_temporal_model(name: str, **kwargs: Any) -> DualStreamTemporalClassifier:
    """Construct a registered model from a concise variant name."""
    key = str(name).lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown temporal model {name!r}; choose {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)


__all__ = [
    "CLASS_TO_INDEX",
    "CSPTModel",
    "DualStreamTemporalClassifier",
    "GlobalPoolingModel",
    "MODEL_REGISTRY",
    "NUM_CLASSES",
    "NUM_PHASES",
    "UnconstrainedAssignmentModel",
    "UniformPhaseBinningModel",
    "build_temporal_model",
]
