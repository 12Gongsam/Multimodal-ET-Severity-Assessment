"""TimesNet sequence encoder."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=7, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

def FFT_for_Period(x, k=2, fs=100.0, f_min=None):
    B, T, C = x.shape
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)

    if f_min is None:
        frequency_list[0] = 0
    else:
        # Zero-out all components ≤ f_min (including DC)
        idx_cut = int(math.floor(float(f_min) * T / float(fs)))   # bin index for f_min
        frequency_list[:idx_cut + 1] = 0

    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k, f_min=None):
        super().__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.f_min = f_min

        self.conv = nn.Sequential(
            Inception_Block_V2(d_model, d_ff),
            nn.GELU(),
            Inception_Block_V2(d_ff, d_model),
        )

    def forward(self, x):
        B, T, N = x.size()
        # period_list, period_weight = FFT_for_Period(x, k=self.k, f_min=self.f_min)
        period_list, period_weight = FFT_for_Period(
            x, k=self.k, f_min=self.f_min
        )

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1 ,N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1) # B T N k
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=-1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class TimesNet(nn.Module):
    def __init__(self, num_class, seq_len, enc_in, d_model, d_ff, e_layers, top_k, f_min=None, dropout=0.1):
        super().__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.model = nn.ModuleList([TimesBlock(seq_len, d_model, d_ff, top_k, f_min)
                                    for _ in range(e_layers)])
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * seq_len, num_class)
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_class = num_class

    # ===== 추가: 세그먼트 임베딩만 뽑는 메서드 =====
    def encode(self, x_enc):  # x_enc: [B, T, C]
        enc_out = self.enc_embedding(x_enc)  # [B,T,d_model]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # y = self.act(enc_out)
        # y = self.dropout(y)
        # # 간단하고 안정적인 통계 풀링(평균). 필요하면 mean+std concat으로 확장 가능.
        # feat = y.mean(dim=1)                  # [B, d_model]
        return enc_out

    # 기존 세그먼트 단일 분류도 유지
    def classification(self, x):
        enc_out = self.enc_embedding(x)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)

    def forward(self, x):
        return self.classification(x)  # [B, num_class]
