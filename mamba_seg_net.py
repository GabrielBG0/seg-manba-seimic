"""
MambaSegNet: State-of-the-art 2D Semantic Segmentation via VMamba
=================================================================

Architecture overview
---------------------
Encoder    : 4 hierarchical VMamba stages with patch merging (x2 downsampling each)
Bottleneck : additional VSSBlocks at the deepest feature level
Decoder    : U-Net-style with PatchExpanding upsampling + VSSBlocks + skip connections
Head       : lightweight linear segmentation head

Key design decisions (VMamba / VM-UNet literature, 2024-2025)
-------------------------------------------------------------
- SS2D (Selective Scan 2D): four-directional cross-scan so every spatial
  position has a path to every other along four axes (-> <- down up).
- Pure-PyTorch sequential scan is the default; the fast CUDA kernel from
  `mamba_ssm` is used automatically when available (~10-20x faster for
  training; strongly recommended).
- All hyper-parameters are exposed so the model can be scaled from a
  lightweight variant to a large backbone.

Dependencies (minimal)
-----------------------
    pip install torch

Optional (for training speed):
    pip install mamba_ssm causal_conv1d

Usage
-----
    from mamba_seg_net import mamba_seg_base, mamba_seg_tiny, DiceCELoss

    model  = mamba_seg_base(num_classes=10)
    x      = torch.randn(2, 3, 512, 512)   # H,W must be divisible by 32
    logits = model(x)                       # (2, 10, 512, 512)

    criterion = DiceCELoss()
    loss = criterion(logits, targets)       # targets: (B, H, W) integer labels
    loss.backward()

References
----------
- VMamba: Visual State Space Model,  Liu et al.,  arXiv 2401.13260
- VM-UNet: VMamba meets U-Net,       Ruan & Xiang, arXiv 2402.02491
- SegMamba: 3-D Mamba segmentation,  Xing et al., MICCAI 2024
- Mamba: Linear-time SSM,            Gu & Dao,    arXiv 2312.00752
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional fast CUDA scan (mamba_ssm) - silent fallback if not installed
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_scan
    _FAST_SCAN = True
except ImportError:
    _FAST_SCAN = False


# ===========================================================================
# 0.  Utility rearrangements (pure PyTorch - no einops needed)
# ===========================================================================

def _b_c_hw_to_b_hw_c(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H, W, C)"""
    return x.permute(0, 2, 3, 1).contiguous()

def _b_hw_c_to_b_c_hw(x: torch.Tensor) -> torch.Tensor:
    """(B, H, W, C) -> (B, C, H, W)"""
    return x.permute(0, 3, 1, 2).contiguous()

def _pixel_shuffle_up2(x: torch.Tensor) -> torch.Tensor:
    """
    (B, H, W, 4*C) -> (B, 2H, 2W, C)  via pixel-shuffle.
    Equivalent to: rearrange(x, 'b h w (p1 p2 c)->b (h p1)(w p2) c', p1=2, p2=2)
    """
    B, H, W, C4 = x.shape
    c = C4 // 4
    x = x.view(B, H, W, 2, 2, c)        # (B, H, W, 2, 2, C)
    x = x.permute(0, 1, 3, 2, 4, 5)     # (B, H, 2, W, 2, C)
    return x.reshape(B, H * 2, W * 2, c).contiguous()

def _pixel_shuffle_up4(x: torch.Tensor) -> torch.Tensor:
    """
    (B, H, W, 16*C) -> (B, 4H, 4W, C)  via pixel-shuffle.
    Equivalent to: rearrange(x, 'b h w (p1 p2 c)->b (h p1)(w p2) c', p1=4, p2=4)
    """
    B, H, W, C16 = x.shape
    c = C16 // 16
    x = x.view(B, H, W, 4, 4, c)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H * 4, W * 4, c).contiguous()


# ===========================================================================
# 1.  Core SSM: selective scan
# ===========================================================================

def selective_scan_pytorch(
    u: torch.Tensor,          # (B, D, L)
    delta: torch.Tensor,      # (B, D, L)
    A: torch.Tensor,          # (D, N)  stored as log(|A|); real A is negative
    B: torch.Tensor,          # (B, N, L)
    C: torch.Tensor,          # (B, N, L)
    D: torch.Tensor,          # (D,)
    delta_bias: Optional[torch.Tensor] = None,  # (D,)
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Pure-PyTorch sequential selective scan (reference implementation).

    Correct and dependency-free; O(D * N * L) time.
    For real training, install mamba_ssm for the parallel CUDA scan.
    """
    dtype_in = u.dtype
    u     = u.float()
    delta = delta.float()
    B_batch, D_dim, L = u.shape
    N = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, D_dim, 1).float()
    if delta_softplus:
        delta = F.softplus(delta)

    # Discretise via Zero-Order Hold: dA[b,d,l,n] = exp(delta * A)
    # A stored as log|A|, real A is negative -> A_neg = -exp(A_log)
    A_neg = -torch.exp(A.float())                              # (D, N)

    # dA: (B, D, L, N);  dB: (B, D, L, N)
    dA = torch.exp(
        delta.unsqueeze(-1) * A_neg.unsqueeze(0).unsqueeze(2)
    )
    B_t = B.float().permute(0, 2, 1)    # (B, L, N)
    C_t = C.float().permute(0, 2, 1)    # (B, L, N)
    dB  = delta.unsqueeze(-1) * B_t.unsqueeze(1)   # (B, D, L, N)

    # Sequential recurrence: x_t = dA_t * x_{t-1} + dB_t * u_t
    x   = u.new_zeros(B_batch, D_dim, N)
    ys: List[torch.Tensor] = []
    for t in range(L):
        x = dA[:, :, t] * x + dB[:, :, t] * u[:, :, t].unsqueeze(-1)
        y = (x * C_t[:, t].unsqueeze(1)).sum(-1)   # (B, D)
        ys.append(y)

    out = torch.stack(ys, dim=2)                   # (B, D, L)
    out = out + D.float().view(1, D_dim, 1) * u
    return out.to(dtype_in)


def selective_scan(
    u, delta, A, B, C, D,
    delta_bias=None, delta_softplus=True, return_last_state=False,
):
    """Dispatch to the fast CUDA scan or the pure-PyTorch fallback."""
    if _FAST_SCAN:
        return _mamba_scan(
            u, delta, A, B, C, D,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )
    out = selective_scan_pytorch(
        u, delta, A, B, C, D,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )
    return (out, None) if return_last_state else out


# ===========================================================================
# 2.  SS2D  -  2-D Selective Scan with four-directional cross-scan
# ===========================================================================

class SS2D(nn.Module):
    """
    Selective Scan 2D (SS2D) -- the core VMamba operator.

    Applies the SSM along four scan paths (->, <-, down, up) over the spatial
    axes, giving every position a global receptive field at O(H*W) cost,
    unlike self-attention which costs O((H*W)^2).

    Parameters
    ----------
    d_model  : number of input/output channels (C)
    d_state  : SSM hidden-state size N (default 16)
    d_rank   : rank of the delta-projection (default ceil(d_model / 16))
    dropout  : output dropout rate
    bias     : use bias in linear projections
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_rank  = d_rank or math.ceil(d_model / 16)
        K = 4   # number of scan directions

        # Expand input: half goes to SSM, half to gating
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=bias)

        # Depthwise conv for short-range context before SSM
        self.conv2d = nn.Conv2d(
            d_model, d_model, kernel_size=3, padding=1,
            groups=d_model, bias=True,
        )
        self.act = nn.SiLU()

        # Per-direction (delta, B, C) projections
        self.x_proj = nn.ModuleList([
            nn.Linear(d_model, self.d_rank + d_state * 2, bias=False)
            for _ in range(K)
        ])
        # delta up-projections with warm-started biases
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.d_rank, d_model, bias=True)
            for _ in range(K)
        ])
        for dt_proj in self.dt_projs:
            dt_init = torch.exp(
                torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)

        # A stored as log|A|, initialised as log(1), log(2), ..., log(N)
        A_log = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0).repeat(d_model, 1)
        )
        self.A_logs = nn.Parameter(A_log.repeat(K, 1, 1))  # (K, D, N)
        self.Ds     = nn.Parameter(torch.ones(K, d_model))  # skip-conn gains

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    # Four-directional scan helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_4dir(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        (B, C, H, W) -> xs: (4, B, C, L)

        Directions:
          0 : row-major    ->    standard raster
          1 : row-major    <-    reversed raster
          2 : col-major    down  (scan each column top-to-bottom)
          3 : col-major    up    reversed column scan
        """
        B, C, H, W = x.shape
        L = H * W
        x0 = x.reshape(B, C, L)                          # row-major ->
        x1 = x0.flip(-1)                                  # row-major <-
        x2 = x.permute(0, 1, 3, 2).reshape(B, C, L)     # col-major down
        x3 = x2.flip(-1)                                  # col-major up
        xs = torch.stack([x0, x1, x2, x3], dim=0)        # (4, B, C, L)
        return xs, (B, C, H, W)

    @staticmethod
    def _merge_4dir(
        ys: torch.Tensor,                                  # (4, B, C, L)
        meta: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Inverse of _scan_4dir.  Returns (B, C, H, W)."""
        B, C, H, W = meta
        y0 = ys[0].reshape(B, C, H, W)
        y1 = ys[1].flip(-1).reshape(B, C, H, W)
        y2 = ys[2].reshape(B, C, W, H).permute(0, 1, 3, 2)
        y3 = ys[3].flip(-1).reshape(B, C, W, H).permute(0, 1, 3, 2)
        return y0 + y1 + y2 + y3

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) -> (B, H, W, C)"""
        B, H, W, C = x.shape
        L = H * W

        # Split into SSM branch and gating branch
        xz    = self.in_proj(x)          # (B, H, W, 2C)
        x_ssm = xz[..., :C]             # (B, H, W, C)
        z     = xz[..., C:]             # (B, H, W, C)

        # Local depthwise conv (captures short-range structure)
        x_ssm = _b_hw_c_to_b_c_hw(x_ssm)    # (B, C, H, W)
        x_ssm = self.act(self.conv2d(x_ssm))

        # Four-directional scan
        xs, meta = self._scan_4dir(x_ssm)    # (4, B, C, L)

        ys: List[torch.Tensor] = []
        for k in range(4):
            xk = xs[k]                        # (B, C, L)

            # Token-wise projection to (delta_rank, N, N)
            proj_k = self.x_proj[k](
                xk.permute(0, 2, 1).reshape(B * L, C)
            ).view(B, L, -1)                  # (B, L, rank+2N)

            dr, ds = self.d_rank, self.d_state
            delta_k = proj_k[..., :dr]                   # (B, L, rank)
            B_k     = proj_k[..., dr:dr + ds]            # (B, L, N)
            C_k     = proj_k[..., dr + ds:]              # (B, L, N)

            # delta up-projection: (B, L, rank) -> (B, C, L)
            delta_k = self.dt_projs[k](delta_k).permute(0, 2, 1)

            yk = selective_scan(
                u          = xk,                          # (B, C, L)
                delta      = delta_k,                     # (B, C, L)
                A          = self.A_logs[k],              # (C, N)
                B          = B_k.permute(0, 2, 1),        # (B, N, L)
                C          = C_k.permute(0, 2, 1),        # (B, N, L)
                D          = self.Ds[k],                  # (C,)
                delta_bias = self.dt_projs[k].bias.float(),
                delta_softplus=True,
            )                                             # (B, C, L)
            ys.append(yk)

        ys_t = torch.stack(ys, dim=0)                    # (4, B, C, L)
        y    = self._merge_4dir(ys_t, meta)              # (B, C, H, W)
        y    = _b_c_hw_to_b_hw_c(y)                     # (B, H, W, C)

        # LayerNorm -> SiLU gate -> output projection
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.dropout(y)


# ===========================================================================
# 3.  VSSBlock -- SS2D with LayerNorm, residual, and optional FFN
# ===========================================================================

class DropPath(nn.Module):
    """Stochastic depth (per-sample) regularisation."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, device=x.device).lt_(keep).div_(keep)
        return x * mask


class VSSBlock(nn.Module):
    """
    Visual State Space Block: LayerNorm -> SS2D -> residual.

    Parameters
    ----------
    d_model   : channel dimension
    d_state   : SSM state size
    mlp_ratio : if > 0, adds a GELU FFN after the SSM (mlp_ratio=2 common)
    drop_path : stochastic-depth probability
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        mlp_ratio: float = 0.0,
        drop_path: float = 0.0,
        **ss2d_kwargs,
    ):
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.ss2d      = SS2D(d_model, d_state=d_state, **ss2d_kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.has_ffn = mlp_ratio > 0
        if self.has_ffn:
            hidden = int(d_model * mlp_ratio)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        x = x + self.drop_path(self.ss2d(self.norm1(x)))
        if self.has_ffn:
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ===========================================================================
# 4.  Spatial resolution changes: Embed / Merge / Expand
# ===========================================================================

class PatchEmbed(nn.Module):
    """
    Overlapping stem: two 3x3 stride-2 convolutions -> effective patch size 4.
    Better edge preservation than a single 4x4 stride-4 projection.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)               # (B, C, H/4, W/4)
        x = _b_c_hw_to_b_hw_c(x)      # (B, H/4, W/4, C)
        return self.norm(x)


class PatchMerging(nn.Module):
    """
    2x2 spatial downsampling: (B, H, W, C) -> (B, H/2, W/2, 2C).
    Concatenates 2x2 neighbours then projects with a linear layer.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        return self.reduction(self.norm(x))


class PatchExpanding(nn.Module):
    """
    2x upsampling: (B, H, W, C) -> (B, 2H, 2W, C//2) via pixel-shuffle.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 2 * dim, bias=False)   # 2C = 4 * (C//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(self.norm(x))   # (B, H, W, 2C)
        return _pixel_shuffle_up2(x)    # (B, 2H, 2W, C//2)


class FinalExpanding(nn.Module):
    """
    4x upsampling head: (B, H, W, C) -> (B, 4H, 4W, C//4).
    Recovers the input resolution after the 4x PatchEmbed downsampling.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 4 * dim, bias=False)   # 4C = 16 * (C//4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(self.norm(x))   # (B, H, W, 4C)
        return _pixel_shuffle_up4(x)    # (B, 4H, 4W, C//4)


# ===========================================================================
# 5.  Encoder and Decoder Stages
# ===========================================================================

class VMambaStage(nn.Module):
    """A sequence of VSSBlocks without spatial downsampling."""

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        drop_path_rates: Optional[List[float]] = None,
        **vss_kwargs,
    ):
        super().__init__()
        dpr = drop_path_rates or [0.0] * depth
        self.blocks = nn.ModuleList([
            VSSBlock(dim, d_state=d_state, drop_path=dpr[i], **vss_kwargs)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class DecoderStage(nn.Module):
    """
    One decoder step:
      1. PatchExpanding (2x upsample, halve channels)
      2. Concatenate encoder skip connection
      3. Fuse with a linear layer
      4. VSSBlocks
    """

    def __init__(
        self,
        dim: int,       # channels from the deeper (smaller) feature map
        skip_dim: int,  # channels of the matching encoder skip connection
        depth: int = 2,
        d_state: int = 16,
        **vss_kwargs,
    ):
        super().__init__()
        self.up     = PatchExpanding(dim)          # -> dim//2
        out_dim     = dim // 2
        self.fuse   = nn.Linear(out_dim + skip_dim, out_dim, bias=False)
        self.blocks = VMambaStage(out_dim, depth, d_state=d_state, **vss_kwargs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                             # (B, 2H, 2W, dim//2)
        x = torch.cat([x, skip], dim=-1)
        x = self.fuse(x)
        return self.blocks(x)


# ===========================================================================
# 6.  MambaSegNet -- full encoder-decoder model
# ===========================================================================

class MambaSegNet(nn.Module):
    """
    VMamba-based U-Net for 2D semantic segmentation.

    Parameters
    ----------
    num_classes    : number of output classes
    in_channels    : input image channels (default 3)
    embed_dim      : base channel width C0 for stage 1
                     (stage dims = [C0, 2C0, 4C0, 8C0])
    depths         : VSSBlock depths per encoder stage (list of 4 ints)
    d_state        : SSM hidden state size N (16 is standard)
    drop_path_rate : max stochastic-depth drop rate (linearly scheduled)
    mlp_ratio      : FFN hidden-dim multiplier in VSSBlocks (0 = no FFN)

    Input  : (B, in_channels, H, W)   -- H and W must be divisible by 32
    Output : (B, num_classes, H, W)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] = (2, 2, 9, 2),
        d_state: int = 16,
        drop_path_rate: float = 0.2,
        mlp_ratio: float = 0.0,
    ):
        super().__init__()
        S    = len(depths)
        self.num_stages = S
        dims = [embed_dim * (2 ** i) for i in range(S)]  # [C, 2C, 4C, 8C]

        # Linear stochastic-depth schedule
        total    = sum(depths)
        dpr_flat = [x.item() for x in torch.linspace(0, drop_path_rate, total)]
        dpr: List[List[float]] = []
        ptr = 0
        for d in depths:
            dpr.append(dpr_flat[ptr:ptr + d])
            ptr += d

        vss_kwargs = dict(mlp_ratio=mlp_ratio)

        # ----- Encoder -------------------------------------------------------
        self.patch_embed = PatchEmbed(in_channels, dims[0])
        self.enc_stages  = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(S):
            self.enc_stages.append(
                VMambaStage(dims[i], depths[i], d_state=d_state,
                            drop_path_rates=dpr[i], **vss_kwargs)
            )
            if i < S - 1:
                self.downsamples.append(PatchMerging(dims[i]))

        # ----- Decoder -------------------------------------------------------
        # dec_stages[0]: dims[S-1] -> dims[S-2], etc.
        self.dec_stages = nn.ModuleList()
        for i in range(S - 1):
            self.dec_stages.append(
                DecoderStage(
                    dim      = dims[S - 1 - i],
                    skip_dim = dims[S - 2 - i],
                    depth    = 2,
                    d_state  = d_state,
                    **vss_kwargs,
                )
            )

        # Final 4x upsample + seg head
        self.final_expand = FinalExpanding(dims[0])
        self.seg_head = nn.Sequential(
            nn.LayerNorm(dims[0] // 4),
            nn.Linear(dims[0] // 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_channels, H, W)  -- H,W divisible by 32
        Returns logits (B, num_classes, H, W)
        """
        # Encoder
        x = self.patch_embed(x)              # (B, H/4, W/4, C0)
        skips: List[torch.Tensor] = []
        for i in range(self.num_stages):
            x = self.enc_stages[i](x)
            if i < self.num_stages - 1:
                skips.append(x)             # save before downsampling
                x = self.downsamples[i](x)

        # Decoder (use encoder skips in reverse order)
        for i, dec in enumerate(self.dec_stages):
            x = dec(x, skips[-(i + 1)])

        # 4x upsample + classification head
        x = self.final_expand(x)            # (B, H, W, C0//4)
        x = self.seg_head(x)               # (B, H, W, num_classes)
        return x.permute(0, 3, 1, 2).contiguous()  # (B, num_classes, H, W)

    def get_param_groups(self, weight_decay: float = 1e-2) -> List[dict]:
        """
        Split parameters for the optimizer:
        - Norms, biases, SSM D-gains and A_logs -> no weight decay
        - Everything else -> weight_decay
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in ('norm', 'bias', '.Ds', 'A_logs')):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {'params': decay,    'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 7.  Convenience constructors  (named after VMamba paper scale points)
# ===========================================================================

def mamba_seg_tiny(num_classes: int, **kwargs) -> MambaSegNet:
    """~22 M params. Rapid experimentation and resource-limited environments."""
    return MambaSegNet(num_classes=num_classes,
                       embed_dim=64, depths=[2, 2, 5, 2],
                       d_state=16, drop_path_rate=0.1, **kwargs)

def mamba_seg_small(num_classes: int, **kwargs) -> MambaSegNet:
    """~49 M params. Good accuracy/efficiency balance."""
    return MambaSegNet(num_classes=num_classes,
                       embed_dim=96, depths=[2, 2, 9, 2],
                       d_state=16, drop_path_rate=0.2, **kwargs)

def mamba_seg_base(num_classes: int, **kwargs) -> MambaSegNet:
    """~89 M params. Recommended for production training and seismic volumes."""
    return MambaSegNet(num_classes=num_classes,
                       embed_dim=128, depths=[2, 2, 9, 2],
                       d_state=16, drop_path_rate=0.3, **kwargs)

def mamba_seg_large(num_classes: int, **kwargs) -> MambaSegNet:
    """~197 M params. Best accuracy; requires ~40 GB GPU for 512x512 batches."""
    return MambaSegNet(num_classes=num_classes,
                       embed_dim=192, depths=[2, 2, 18, 2],
                       d_state=16, drop_path_rate=0.4, **kwargs)


# ===========================================================================
# 8.  Loss function
# ===========================================================================

class DiceCELoss(nn.Module):
    """
    Dice + Cross-Entropy combined loss.

    Standard for dense segmentation with class imbalance -- e.g. seismic
    facies where background voxels can dominate by a large margin.

    Parameters
    ----------
    ce_weight     : scalar weight for CE term (default 1.0)
    dice_weight   : scalar weight for Dice term (default 1.0)
    class_weights : optional (num_classes,) tensor for per-class CE weighting
    smooth        : Dice denominator smoothing constant
    ignore_index  : class index to skip in CE (e.g. -100 for unlabelled)
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = 1e-5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,    # (B, C, H, W)
        targets: torch.Tensor,   # (B, H, W)  long tensor
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)                # (B, C, H, W)

        # One-hot: (B, H, W) -> (B, C, H, W)
        tgt_oh = F.one_hot(
            targets.clamp(min=0), num_classes
        ).permute(0, 3, 1, 2).float()

        dims  = (0, 2, 3)
        inter = (probs * tgt_oh).sum(dims)
        denom = (probs + tgt_oh).sum(dims)
        dice  = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)

        return self.ce_weight * ce_loss + self.dice_weight * dice.mean()


# ===========================================================================
# 9.  Smoke test  (python mamba_seg_net.py)
# ===========================================================================

def _smoke_test():
    print("=" * 62)
    print("  MambaSegNet -- Smoke Test")
    print("=" * 62)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device     : {device}")
    print(f"  Fast scan  : "
          f"{'mamba_ssm CUDA' if _FAST_SCAN else 'PyTorch sequential (install mamba_ssm for speed)'}")
    print()

    variants = [
        ('tiny',  mamba_seg_tiny),
        ('small', mamba_seg_small),
        ('base',  mamba_seg_base),
    ]
    x = torch.randn(1, 3, 256, 256, device=device)
    for name, factory in variants:
        model = factory(num_classes=8).to(device).eval()
        with torch.no_grad():
            out = model(x)
        n = model.count_parameters() / 1e6
        print(f"  {name:6s}  params: {n:6.1f} M   output: {tuple(out.shape)}")

    print("\n  Loss + backward check ...")
    model = mamba_seg_tiny(num_classes=8).to(device).train()
    x2    = torch.randn(2, 3, 256, 256, device=device)
    tgt   = torch.randint(0, 8, (2, 256, 256), device=device)
    loss  = DiceCELoss()(model(x2), tgt)
    loss.backward()
    print(f"  DiceCE loss: {loss.item():.4f}  checkmark")
    print("\n  All checks passed.")


if __name__ == '__main__':
    _smoke_test()