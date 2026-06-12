"""Experimental multimodal MIL with joint instance-token attention."""

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
    _projector,
    build_encoder,
)


@dataclass(frozen=True)
class JointInstanceAttentionConfig:
    acc_encoder: EncoderConfig
    traj_encoder: EncoderConfig
    num_classes: int = 4
    d_model: int = 128
    attention_heads: int = 8
    joint_attention_layers: int = 2
    feedforward_dim: int = 512
    attention_dropout: float = 0.1
    time_dropout: float = 0.1
    mil_attn_dim: int = 64
    seq_len: int = SEG_LEN


def _zero_padding(
    features: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if padding_mask is None:
        return features
    return features.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class PreNormSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
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

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ):
        normalized = self.attention_norm(features)
        attended, attention_weights = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        features = features + self.attention_dropout(attended)
        features = _zero_padding(features, padding_mask)

        transformed = self.feedforward(self.feedforward_norm(features))
        features = features + self.feedforward_dropout(transformed)
        features = _zero_padding(features, padding_mask)
        return features, attention_weights


class PreNormCrossAttentionBlock(nn.Module):
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

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        normalized_context = self.context_norm(context)
        attended, attention_weights = self.cross_attention(
            self.query_norm(query),
            normalized_context,
            normalized_context,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        features = query + self.attention_dropout(attended)
        features = _zero_padding(features, padding_mask)

        transformed = self.feedforward(self.feedforward_norm(features))
        features = features + self.feedforward_dropout(transformed)
        features = self.output_norm(features)
        features = _zero_padding(features, padding_mask)
        return features, attention_weights


class MILJointInstanceAttention(nn.Module):
    """Pool time per modality, then model joint modality-instance tokens."""

    def __init__(
        self,
        acc_encoder: nn.Module,
        traj_encoder: nn.Module,
        *,
        acc_encoder_dim: int,
        traj_encoder_dim: int,
        d_model: int,
        num_classes: int,
        attention_heads: int = 8,
        joint_attention_layers: int = 2,
        feedforward_dim: int = 512,
        attention_dropout: float = 0.1,
        time_dropout: float = 0.1,
        mil_attn_dim: int = 64,
    ):
        super().__init__()
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if joint_attention_layers not in {2, 3}:
            raise ValueError("joint_attention_layers must be 2 or 3")

        self.acc_encoder = acc_encoder
        self.traj_encoder = traj_encoder
        self.acc_projection = _projector(acc_encoder_dim, d_model)
        self.traj_projection = _projector(traj_encoder_dim, d_model)
        self.acc_norm = nn.LayerNorm(d_model)
        self.traj_norm = nn.LayerNorm(d_model)
        self.acc_time_readout = AttnTimeReadout(
            d_model,
            dropout=time_dropout,
        )
        self.traj_time_readout = AttnTimeReadout(
            d_model,
            dropout=time_dropout,
        )

        self.modality_embeddings = nn.Parameter(torch.empty(2, 1, 1, d_model))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        self.joint_attention = nn.ModuleList(
            [
                PreNormSelfAttentionBlock(
                    d_model=d_model,
                    num_heads=attention_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=attention_dropout,
                )
                for _ in range(joint_attention_layers)
            ]
        )
        self.joint_output_norm = nn.LayerNorm(d_model)
        self.cross_attention = PreNormCrossAttentionBlock(
            d_model=d_model,
            num_heads=attention_heads,
            feedforward_dim=feedforward_dim,
            dropout=attention_dropout,
        )
        self.mil_head = AttnMILHead(
            feature_dim=d_model,
            num_classes=num_classes,
            attention_dim=mil_attn_dim,
            dropout=attention_dropout,
        )

    def forward(
        self,
        acc_bag: torch.Tensor,
        traj_bag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        return_joint_attention: bool = False,
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

        acc_sequence = self.acc_norm(
            self.acc_projection(self.acc_encoder.encode(acc))
        )
        traj_sequence = self.traj_norm(
            self.traj_projection(self.traj_encoder.encode(traj))
        )
        acc_tokens, acc_time_weights = self.acc_time_readout(acc_sequence)
        traj_tokens, traj_time_weights = self.traj_time_readout(traj_sequence)
        acc_tokens = acc_tokens.view(batch_size, instances, -1)
        traj_tokens = traj_tokens.view(batch_size, instances, -1)

        acc_tokens = acc_tokens + self.modality_embeddings[0]
        traj_tokens = traj_tokens + self.modality_embeddings[1]
        joint_tokens = torch.cat([acc_tokens, traj_tokens], dim=1)
        joint_padding_mask = (
            None if mask is None else torch.cat([~mask, ~mask], dim=1)
        )

        joint_attention_weights = []
        for block in self.joint_attention:
            joint_tokens, layer_weights = block(
                joint_tokens,
                joint_padding_mask,
                return_attention=return_joint_attention,
            )
            if return_joint_attention:
                joint_attention_weights.append(layer_weights)
        joint_tokens = self.joint_output_norm(joint_tokens)
        joint_tokens = _zero_padding(joint_tokens, joint_padding_mask)

        acc_context = joint_tokens[:, :instances]
        traj_context = joint_tokens[:, instances:]
        fused, cross_attention_weights = self.cross_attention(
            query=traj_context,
            context=acc_context,
            padding_mask=None if mask is None else ~mask,
        )
        logits, instance_weights = self.mil_head(fused, mask)

        return logits, {
            "instance_weights": instance_weights,
            "acc_time_weights": acc_time_weights.view(
                batch_size,
                instances,
                acc_time_weights.shape[-1],
            ),
            "traj_time_weights": traj_time_weights.view(
                batch_size,
                instances,
                traj_time_weights.shape[-1],
            ),
            "joint_attention_weights": (
                torch.stack(joint_attention_weights)
                if return_joint_attention
                else None
            ),
            "cross_attention_weights": cross_attention_weights,
        }


def build_joint_instance_attention_model(
    config: JointInstanceAttentionConfig,
) -> MILJointInstanceAttention:
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
        num_classes=config.num_classes,
        seq_len=config.seq_len,
        default_feature_dim=config.d_model,
    )
    return MILJointInstanceAttention(
        acc_encoder,
        traj_encoder,
        acc_encoder_dim=acc_dim,
        traj_encoder_dim=traj_dim,
        d_model=config.d_model,
        num_classes=config.num_classes,
        attention_heads=config.attention_heads,
        joint_attention_layers=config.joint_attention_layers,
        feedforward_dim=config.feedforward_dim,
        attention_dropout=config.attention_dropout,
        time_dropout=config.time_dropout,
        mil_attn_dim=config.mil_attn_dim,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
