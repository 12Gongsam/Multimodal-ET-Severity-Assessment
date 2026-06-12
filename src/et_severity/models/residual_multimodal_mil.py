"""Experimental multimodal MIL with residual cross-attention blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..config import SEG_LEN
from .mil_models import (
    AttnMILHead,
    AttnTimeReadout,
    EncoderConfig,
    GAPTimeReadout,
    _projector,
    build_encoder,
)


@dataclass(frozen=True)
class ResidualMultimodalModelConfig:
    acc_encoder: EncoderConfig
    traj_encoder: EncoderConfig
    num_classes: int = 4
    traj_encoder_num_classes: int = 3
    d_model: int = 128
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1
    feedforward_dim: int = 512
    time_dropout: float = 0.1
    mil_attn_dim: int = 64
    time_pool: str = "attn"
    cross_attention_direction: str = "traj_query"
    seq_len: int = SEG_LEN


class ResidualCrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention with residual attention and FFN sublayers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor):
        normalized_context = self.context_norm(context)
        attended, attention_weights = self.cross_attention(
            self.query_norm(query),
            normalized_context,
            normalized_context,
            need_weights=True,
            average_attn_weights=True,
        )
        features = query + self.attention_dropout(attended)
        transformed = self.feedforward(self.feedforward_norm(features))
        features = features + self.feedforward_dropout(transformed)
        return self.output_norm(features), attention_weights


class MILResidualMultiModal(nn.Module):
    """MIL multimodal model with a residual Pre-LN cross-attention block."""

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
        feedforward_dim: int = 512,
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
        self.cross_attention = ResidualCrossAttentionBlock(
            d_model=d_model,
            num_heads=cross_attention_heads,
            feedforward_dim=feedforward_dim,
            dropout=cross_attention_dropout,
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
        traj = traj_bag.reshape(
            batch_size * instances,
            time_steps,
            traj_channels,
        )

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
        )

        if self.time_pool == "attn":
            instance_features, time_weights = self.time_readout(fused)
        else:
            instance_features = self.time_readout(fused)
            time_weights = None

        feature_dim = instance_features.shape[-1]
        instance_features = instance_features.view(
            batch_size,
            instances,
            feature_dim,
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


def build_residual_multimodal_model(
    config: ResidualMultimodalModelConfig,
) -> MILResidualMultiModal:
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
    return MILResidualMultiModal(
        acc_encoder,
        traj_encoder,
        acc_encoder_dim=acc_dim,
        traj_encoder_dim=traj_dim,
        d_model=config.d_model,
        num_classes=config.num_classes,
        cross_attention_heads=config.cross_attention_heads,
        cross_attention_dropout=config.cross_attention_dropout,
        feedforward_dim=config.feedforward_dim,
        time_dropout=config.time_dropout,
        mil_attn_dim=config.mil_attn_dim,
        time_pool=config.time_pool,
        cross_attention_direction=config.cross_attention_direction,
    )
