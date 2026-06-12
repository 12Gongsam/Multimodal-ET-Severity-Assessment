"""Configurable encoder and MIL model construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SEG_LEN
from .lstm import LSTM
from .resnet1d import ResNet18_1D
from .timesnet import TimesNet
from .wavenet import MyWaveNet


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SingleModelConfig:
    encoder: EncoderConfig
    num_classes: int = 4
    d_model: int = 128
    time_dropout: float = 0.1
    mil_attn_dim: int = 64
    mil_dropout: float = 0.1
    seq_len: int = SEG_LEN


@dataclass(frozen=True)
class MultimodalModelConfig:
    acc_encoder: EncoderConfig
    traj_encoder: EncoderConfig
    num_classes: int = 4
    traj_encoder_num_classes: int = 3
    d_model: int = 128
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1
    time_dropout: float = 0.1
    mil_attn_dim: int = 64
    time_pool: str = "attn"
    cross_attention_direction: str = "traj_query"
    seq_len: int = SEG_LEN


class GAPTimeReadout(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x.mean(dim=1))


class AttnTimeReadout(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, x)
        return self.dropout(pooled), weights


class AttnMILHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        attention_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.value = nn.Linear(feature_dim, attention_dim)
        self.gate = nn.Linear(feature_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        attention = self.score(
            self.dropout(
                torch.tanh(self.value(features)) * torch.sigmoid(self.gate(features))
            )
        ).squeeze(-1)
        if mask is not None:
            attention = attention.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(attention, dim=1)
        pooled = torch.einsum("bn,bnd->bd", weights, features)
        return self.classifier(self.dropout(pooled)), weights


def _projector(input_dim: int, output_dim: int) -> nn.Module:
    if input_dim == output_dim:
        return nn.Identity()
    return nn.Linear(input_dim, output_dim)


class MIL_MultiModal(nn.Module):
    def __init__(
        self,
        acc_encoder: nn.Module,
        traj_encoder: nn.Module,
        *,
        acc_encoder_dim: int,
        traj_encoder_dim: int,
        d_model: int,
        num_classes: int,
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,
        time_dropout: float = 0.1,
        mil_attn_dim: int = 64,
        time_pool: str = "attn",
        cross_attention_direction: str = "traj_query",
    ):
        super().__init__()
        if d_model % cross_attention_heads != 0:
            raise ValueError("d_model must be divisible by cross_attention_heads")
        if time_pool not in {"attn", "gap"}:
            raise ValueError("time_pool must be 'attn' or 'gap'")
        if cross_attention_direction not in {"traj_query", "acc_query"}:
            raise ValueError(
                "cross_attention_direction must be 'traj_query' or 'acc_query'"
            )

        self.acc_encoder = acc_encoder
        self.traj_encoder = traj_encoder
        self.acc_projection = _projector(acc_encoder_dim, d_model)
        self.traj_projection = _projector(traj_encoder_dim, d_model)
        self.acc_norm = nn.LayerNorm(d_model)
        self.traj_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=cross_attention_heads,
            dropout=cross_attention_dropout,
            batch_first=True,
        )
        self.time_pool = time_pool
        self.cross_attention_direction = cross_attention_direction
        self.time_readout = (
            AttnTimeReadout(d_model, dropout=time_dropout)
            if time_pool == "attn"
            else GAPTimeReadout(dropout=time_dropout)
        )
        self.mil_head = AttnMILHead(
            feature_dim=d_model,
            num_classes=num_classes,
            attention_dim=mil_attn_dim,
            dropout=cross_attention_dropout,
        )

    def forward(
        self,
        acc_bag: torch.Tensor,
        traj_bag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if acc_bag.dim() != 4 or traj_bag.dim() != 4:
            raise ValueError("acc_bag and traj_bag must have shape [B, N, T, C]")
        if acc_bag.shape[:3] != traj_bag.shape[:3]:
            raise ValueError("acc_bag and traj_bag must share [B, N, T]")
        if mask is not None and mask.shape != acc_bag.shape[:2]:
            raise ValueError("mask must have shape [B, N]")

        batch_size, instances, time_steps, acc_channels = acc_bag.shape
        traj_channels = traj_bag.shape[-1]
        acc = acc_bag.reshape(batch_size * instances, time_steps, acc_channels)
        traj = traj_bag.reshape(batch_size * instances, time_steps, traj_channels)

        acc_features = self.acc_norm(
            self.acc_projection(self.acc_encoder.encode(acc))
        )
        traj_features = self.traj_norm(
            self.traj_projection(self.traj_encoder.encode(traj))
        )
        if self.cross_attention_direction == "traj_query":
            query_features = traj_features
            context_features = acc_features
        else:
            query_features = acc_features
            context_features = traj_features

        fused, cross_attention_weights = self.cross_attention(
            query_features,
            context_features,
            context_features,
        )

        if self.time_pool == "attn":
            instance_features, time_weights = self.time_readout(fused)
        else:
            instance_features = self.time_readout(fused)
            time_weights = None

        feature_dim = instance_features.shape[-1]
        instance_features = instance_features.view(
            batch_size, instances, feature_dim
        )
        logits, instance_weights = self.mil_head(instance_features, mask)
        encoded_steps = None if time_weights is None else time_weights.shape[-1]
        return logits, {
            "instance_weights": instance_weights,
            "time_weights": (
                None
                if time_weights is None
                else time_weights.view(batch_size, instances, encoded_steps)
            ),
            "cross_attention_weights": cross_attention_weights,
            "cross_attention_direction": self.cross_attention_direction,
        }


class MIL_Single(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        encoder_dim: int,
        num_classes: int,
        d_model: int = 128,
        time_dropout: float = 0.1,
        mil_attn_dim: int = 64,
        mil_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_projection = _projector(encoder_dim, d_model)
        self.time_readout = AttnTimeReadout(d_model, dropout=time_dropout)
        self.mil_head = AttnMILHead(
            feature_dim=d_model,
            num_classes=num_classes,
            attention_dim=mil_attn_dim,
            dropout=mil_dropout,
        )

    def forward(
        self,
        x_bag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        batch_size, instances, time_steps, channels = x_bag.shape
        flattened = x_bag.reshape(batch_size * instances, time_steps, channels)
        encoded = self.feature_projection(self.encoder.encode(flattened))
        instance_features, time_weights = self.time_readout(encoded)
        instance_features = instance_features.view(batch_size, instances, -1)
        logits, instance_weights = self.mil_head(instance_features, mask)
        return logits, {
            "instance_weights": instance_weights,
            "time_weights": time_weights.view(
                batch_size, instances, time_weights.shape[-1]
            ),
        }


def build_encoder(
    config: EncoderConfig,
    *,
    in_channels: int,
    num_classes: int,
    seq_len: int = SEG_LEN,
    default_feature_dim: int = 128,
) -> Tuple[nn.Module, int]:
    name = config.name.strip().lower()
    params: Dict[str, Any] = dict(config.params)

    if name == "lstm":
        hidden_size = int(params.pop("hidden_size", default_feature_dim))
        num_layers = int(params.pop("num_layers", 2))
        model = LSTM(
            in_c=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(params.pop("dropout", 0.1 if num_layers > 1 else 0.0)),
        )
        output_dim = hidden_size
    elif name == "resnet18":
        output_dim = int(params.pop("feature_dim", default_feature_dim))
        model = ResNet18_1D(
            num_classes=num_classes,
            input_channels=in_channels,
            d_model=output_dim,
        )
    elif name == "timesnet":
        output_dim = int(params.pop("feature_dim", default_feature_dim))
        model = TimesNet(
            num_class=num_classes,
            seq_len=int(params.pop("seq_len", seq_len)),
            enc_in=in_channels,
            d_model=output_dim,
            d_ff=int(params.pop("d_ff", 128)),
            e_layers=int(params.pop("e_layers", 1)),
            top_k=int(params.pop("top_k", 5)),
            f_min=params.pop("f_min", None),
            dropout=float(params.pop("dropout", 0.1)),
        )
    elif name == "mywavenet":
        output_dim = int(params.pop("skip_channels", default_feature_dim))
        model = MyWaveNet(
            in_ch=in_channels,
            res_ch=int(params.pop("residual_channels", default_feature_dim)),
            skip_ch=output_dim,
            dilations=list(params.pop("dilations", (1, 2, 4, 8, 16, 32))),
            n_classes=num_classes,
            n_stacks=int(params.pop("n_stacks", 2)),
        )
    else:
        raise ValueError(
            f"Unknown encoder {config.name!r}. "
            "Choose from LSTM, ResNet18, TimesNet, or MyWaveNet."
        )

    if params:
        raise ValueError(
            f"Unsupported parameters for {config.name}: {sorted(params)}"
        )
    return model, output_dim


def build_single_modality_model(
    config: SingleModelConfig,
    *,
    modality: str,
) -> MIL_Single:
    if modality not in {"acc", "traj"}:
        raise ValueError("modality must be 'acc' or 'traj'")
    in_channels = 3 if modality == "acc" else 2
    encoder, encoder_dim = build_encoder(
        config.encoder,
        in_channels=in_channels,
        num_classes=config.num_classes,
        seq_len=config.seq_len,
        default_feature_dim=config.d_model,
    )
    return MIL_Single(
        encoder,
        encoder_dim=encoder_dim,
        num_classes=config.num_classes,
        d_model=config.d_model,
        time_dropout=config.time_dropout,
        mil_attn_dim=config.mil_attn_dim,
        mil_dropout=config.mil_dropout,
    )


def build_multimodal_model(config: MultimodalModelConfig) -> MIL_MultiModal:
    acc_encoder, acc_dim = build_encoder(
        config.acc_encoder,
        in_channels=3,
        num_classes=config.num_classes,
        seq_len=config.seq_len,
        default_feature_dim=config.d_model,
    )
    traj_encoder, traj_dim = build_encoder(
        config.traj_encoder,
        in_channels=2,
        num_classes=config.traj_encoder_num_classes,
        seq_len=config.seq_len,
        default_feature_dim=config.d_model,
    )
    return MIL_MultiModal(
        acc_encoder,
        traj_encoder,
        acc_encoder_dim=acc_dim,
        traj_encoder_dim=traj_dim,
        d_model=config.d_model,
        num_classes=config.num_classes,
        cross_attention_heads=config.cross_attention_heads,
        cross_attention_dropout=config.cross_attention_dropout,
        time_dropout=config.time_dropout,
        mil_attn_dim=config.mil_attn_dim,
        time_pool=config.time_pool,
        cross_attention_direction=config.cross_attention_direction,
    )
