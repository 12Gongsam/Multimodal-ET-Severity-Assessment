"""WaveNet-style sequence encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnPoolHead(nn.Module):
    def __init__(self, d_model, num_class, p=0.1):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_model, num_class)
        )

    def forward(self, x):
        # x = B T D
        s = self.score(x).squeeze(-1) # B T
        w = F.softmax(s, dim=1) # B T
        z = torch.einsum('bt,btd->bd', w, x) # B d
        return self.proj(z)

class _GatedResidualUnit(nn.Module):
    def __init__(self, res_ch, skip_ch, dilation):
        super().__init__()
        self.conv_filter = nn.Conv1d(res_ch, res_ch, 2,
                                     padding=dilation, dilation=dilation)
        self.conv_gate   = nn.Conv1d(res_ch, res_ch, 2,
                                     padding=dilation, dilation=dilation)
        self.conv_res    = nn.Conv1d(res_ch, res_ch, 1)
        self.conv_skip   = nn.Conv1d(res_ch, skip_ch, 1)

    def forward(self, x):
        # causal padding 보정 (Conv1d 는 causal 지원 X → 수동 crop)
        out_f = self.conv_filter(x)[:, :, :-self.conv_filter.dilation[0]]
        out_g = self.conv_gate  (x)[:, :, :-self.conv_gate.dilation[0]]
        z = torch.tanh(out_f) * torch.sigmoid(out_g)          # gated
        res = self.conv_res(z)
        skip= self.conv_skip(z)
        return (x + res), skip

class MyWaveNet(nn.Module):
    def __init__(
            self,
            in_ch: int,
            res_ch: int,
            skip_ch: int,
            dilations: list,
            n_classes: int = 4,
            n_stacks: int = 2,
    ):
        super().__init__()

        self.skip_ch = skip_ch

        self.ln_pre_cls = nn.LayerNorm(skip_ch)

        self.input_proj = nn.Conv1d(in_ch, res_ch, kernel_size=1)

        def _stack_blocks(nstacks: int, dilations_list, block_cls, res_ch, skip_ch):
            blocks = []
            for _ in range(nstacks):
                for d in dilations_list:
                    blocks.append(block_cls(res_ch, skip_ch, d))
            return nn.ModuleList(blocks)

        self.res_blocks = _stack_blocks(n_stacks, dilations, _GatedResidualUnit, res_ch, skip_ch)

        self.cls_head = AttnPoolHead(skip_ch, n_classes)

    def encode(self, x):
        x = x.permute(0, 2, 1) # B T C -> B C T
        x = self.input_proj(x)

        skip_sum = None
        for blk in self.res_blocks:
            x, skip = blk(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        return skip_sum.permute(0, 2, 1) # B T C

    def forward(self, x):
        # x : B C T
        skip_sum = self.encode(x)

        skip_sum = self.ln_pre_cls(skip_sum) # B T C

        logits_global = self.cls_head(skip_sum)             # (B, n_classes)

        return logits_global
