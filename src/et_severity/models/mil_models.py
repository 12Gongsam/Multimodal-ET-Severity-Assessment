"""Auto-extracted definitions from notebooks for modular use."""

from ..common_imports import *
from ..config import SEG_LEN
from .lstm import LSTM
from .resnet1d import ResNet18_1D
from .timesnet import TimesNet
from .wavenet import MyWaveNet

class GAPTimeReadout(nn.Module):
    """
    x: [B*, T, D] -> z: [B*, D]  (시간축 평균)
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean(dim=1)  # [B*, D]
        return self.drop(z)

class AttnTimeReadout(nn.Module):
    """
    x: [B*, T, D] → (z: [B*, D], w_time: [B*, T])
    """
    def __init__(self, d_model: int, p: float = 0.1):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.score(x).squeeze(-1)          # [B*, T]
        w = F.softmax(s, dim=1)                # [B*, T]
        z = torch.einsum('bt,btd->bd', w, x)   # [B*, D]
        return self.drop(z), w

class AttnMILHead(nn.Module):
    """
    feats: [B, N, D], mask: [B, N] (True=valid)
    반환: logits: [B, C], alpha_inst: [B, N]
    """
    def __init__(self, feat_dim: int, num_classes: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(feat_dim, attn_dim)
        self.U = nn.Linear(feat_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, N, D = feats.shape
        Vh = torch.tanh(self.V(feats))               # [B,N,A]
        Uh = torch.sigmoid(self.U(feats))            # [B,N,A]
        A  = self.w(self.drop(Vh * Uh)).squeeze(-1)  # [B,N]
        if mask is not None:
            A = A.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(A, dim=1)              # [B,N]
        z = torch.einsum('bn,bnd->bd', alpha, feats) # [B,D]
        logits = self.classifier(self.drop(z))       # [B,C]
        return logits, alpha

class MIL_MultiModal(nn.Module):
    """
    입력:
      x_acc_bag  : [B, N, T, C_acc]
      x_traj_bag : [B, N, T, C_traj]
      mask       : [B, N] (True=valid)

    출력:
      sev_logits : [B, num_sclass]
      task_logits: [B, num_tclass]
      aux        : dict (해석용 가중치들)
    """
    def __init__(
        self,
        acc_encoder, traj_encoder,
        d_model: int,
        num_sclass: int,
        *,
        ma_heads: int = 2,
        ma_dropout: float = 0.1,
        time_dropout: float = 0.1,
        mil_attn_dim: int = 64,
        time_pool : str = 'attn',
    ):
        super().__init__()
        self.acc_encoder = acc_encoder
        self.traj_encoder = traj_encoder

        self.acc_ln  = nn.LayerNorm(d_model)
        self.traj_ln = nn.LayerNorm(d_model)

        self.ma = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=ma_heads, dropout=ma_dropout, batch_first=True
        )

        self.time_pool = time_pool
        if self.time_pool == "attn":
        # 시간축 readout (인스턴스 임베딩 만들기)
            self.time_readout_sev  = AttnTimeReadout(d_model, p=time_dropout)  # attn_output → z_sev_inst
        elif self.time_pool == 'gap':
            self.time_readout_sev = GAPTimeReadout(p=time_dropout)
        # 인스턴스→bag MIL (집계 후 최종 로짓)
        self.mil_sev  = AttnMILHead(feat_dim=d_model, num_classes=num_sclass, attn_dim=mil_attn_dim, dropout=ma_dropout)

    @torch.no_grad()
    def _check_shapes(self, x_acc_bag, x_traj_bag, mask):
        assert x_acc_bag.dim() == 4 and x_traj_bag.dim() == 4, "x_* shape must be [B,N,T,C]"
        assert x_acc_bag.shape[:3] == x_traj_bag.shape[:3], "ACC/TRAJ must share [B,N,T]"
        if mask is not None:
            assert mask.shape == x_acc_bag.shape[:2], "mask shape must be [B,N]"

    def forward(self, x_acc_bag: torch.Tensor, x_traj_bag: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self._check_shapes(x_acc_bag, x_traj_bag, mask)
        B, N, T, C_acc = x_acc_bag.shape
        _, _, _, C_traj = x_traj_bag.shape

        # ── (1) 인스턴스 펼치기 ─────────────────────────────────────
        x_acc  = x_acc_bag.reshape(B*N, T, C_acc)      # [B*N, T, C_acc]
        x_traj = x_traj_bag.reshape(B*N, T, C_traj)    # [B*N, T, C_traj]

        # ── (2) 인코더: 시계열 → 시퀀스 임베딩 ─────────────────────
        y_acc  = self.acc_encoder.encode(x_acc)          # [B*N, T, D]
        y_traj = self.traj_encoder.encode(x_traj)      # [B*N, T, D]

        y_acc = self.acc_ln(y_acc)
        y_traj = self.traj_ln(y_traj)

        # ── (3) Cross-Attention: traj(query) ← acc(key/value) ─────
        attn_output, attn_w = self.ma(y_traj, y_acc, y_acc)  # [B*N, T, D], [B*N, num_heads, T, T] (batch_first=True)

        # ── (4) 시간축 어텐션 풀링(인스턴스 임베딩) ─────────────────
        if self.time_pool == "attn":
            z_sev_inst,   w_time_sev  = self.time_readout_sev(attn_output) # [B*N, D], [B*N, T]
        elif self.time_pool == 'gap':
            z_sev_inst = self.time_readout_sev(attn_output) # [B*N, D]
            w_time_sev = None

        # ── (5) [B,N,D]로 재구성 ───────────────────────────────────
        D = z_sev_inst.shape[-1]
        feats_sev  = z_sev_inst.view(B, N, D)    # [B,N,D]

        # ── (6) MIL 집계 (인스턴스 → bag) ──────────────────────────
        sev_logits,  alpha_inst_sev  = self.mil_sev(feats_sev,  mask)  # [B,S], [B,N]
        L = w_time_sev.shape[1] if self.time_pool == "attn" else None
        # 해석용 가중치 (시간/인스턴스)
        aux = {
            "alpha_inst_sev":  alpha_inst_sev,                     # [B,N]
            "w_time_sev":  None if w_time_sev is None else w_time_sev.view(B, N, L),               # [B,N,T]
        }
        return sev_logits, aux

class MIL_Single(nn.Module):
    def __init__(
        self,
        base_encoder: nn.Module,
        num_classes: int,
        *,
        d_model: int = 128,
        time_dropout: float = 0.1,
        attn_dim: int = 128,
        mil_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = base_encoder
        self.time_readout = AttnTimeReadout(d_model=d_model, p=time_dropout)  # x:[B*,T,D]
        self.mil_head = AttnMILHead(feat_dim=d_model, num_classes=num_classes,
                                    attn_dim=attn_dim, dropout=mil_dropout)

    def encode_instances(self, x_bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_bag: [B, N, T, C]
        return:
          z_inst: [B, N, D]
          w_time: [B, N, T]
        """
        B, N, T, C = x_bag.shape
        x_flat = x_bag.reshape(B * N, T, C)
        feat_map = self.encoder.encode(x_flat)
        z_flat, w_flat = self.time_readout(feat_map)  # [B*N, D], [B*N, T]

        T_enc = w_flat.size(-1)         # ★ 인코더가 만든 실제 시간 길이
        D = z_flat.size(-1)             # 임베딩 차원
        # bag 차원 복원
        z_inst = z_flat.view(B, N, -1)    # [B, N, D]
        w_time = w_flat.view(B, N, T_enc)     # [B, N, T]
        return z_inst, w_time

    def forward(
        self,
        x_bag: torch.Tensor,                 # [B, N, T, C]
        mask: Optional[torch.Tensor] = None  # [B, N] (True=valid)
    ):
        z_inst, w_time = self.encode_instances(x_bag)          # [B,N,D], [B,N,T]
        logits_bag, alpha_inst = self.mil_head(z_inst, mask)   # [B,C], [B,N]
        return logits_bag, alpha_inst, w_time

def build_model(
    model_name: str,
    *,
    in_ch: int,
    num_classes: int,
    # 공통/기본 하이퍼파라미터 (필요 모델에만 사용)
    seq_len: int = SEG_LEN,
    d_model: int = 128,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    # MyWaveNet
    dilations: list = (1, 2, 4, 8, 16, 32),
    n_stacks: int = 2,
    # TimesNet
    d_ff: int = 128,
    e_layers: int = 1,
    top_k: int = 5,
    f_min: float = None,
):
    """
    model_name ∈ {'LSTM', 'ResNet18', 'TimesNet', 'MyWaveNet'}
    입력 텐서 형식은 통일하여 [B, T, C]를 기대합니다.
    """
    name = model_name.strip().lower()

    if name == 'lstm':
        model = LSTM(in_ch, hidden_size, num_layers, dropout=dropout)
        return model

    elif name == 'resnet18':
        model = ResNet18_1D(num_classes=num_classes,
                            input_channels=in_ch,
                            d_model=d_model)
        return model

    elif name == 'timesnet':
        assert seq_len is not None, "TimesNet을 쓰려면 seq_len을 지정하세요."
        model = TimesNet(
            num_class=num_classes,
            seq_len=seq_len,
            enc_in=in_ch,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            top_k=top_k,
            f_min=f_min,
            dropout=dropout
        )
        return model

    elif name == 'mywavenet':
        model = MyWaveNet(
            in_ch=in_ch,
            res_ch=d_model,
            skip_ch=d_model,
            dilations=list(dilations),
            n_classes=num_classes,
            n_stacks=n_stacks
        )
        return model

    else:
        raise ValueError(f"알 수 없는 model_name: {model_name!r}. "
                         f"['LSTM','ResNet18','TimesNet','MyWaveNet'] 중에서 선택하세요.")
GlobalAverageTimePool = GAPTimeReadout
TemporalAttentionPool = AttnTimeReadout
GatedAttentionMILHead = AttnMILHead
CrossModalSeverityMILModel = MIL_MultiModal
SingleModalityMILModel = MIL_Single
build_encoder_model = build_model
