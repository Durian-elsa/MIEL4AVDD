"""
SDEE: Semantic Discrepancy Evidence Extraction

Motivation: since deepfakes preserve temporal smoothness, we cannot rely on
frame discontinuities. Instead, we measure whether each timestep is
SEMANTICALLY consistent with the global video context.

Method:
1. Compute a global semantic prototype `g` from the whole video (weighted
   toward regions the model currently believes are real, via a soft mask).
2. For each timestep, compute the deviation of the local feature `f_t` from `g`.
3. Multi-scale temporal convolutions build the local feature so that
   inconsistencies at different segment lengths are captured before the
   local/global comparison.

Output: enhanced per-timestep features, plus s_t, the semantic-deviation
evidence signal that is one of the three sources aggregated by MLEA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDEE(nn.Module):
    def __init__(self, dim=512, num_heads=8, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.dim = dim

        # Multi-scale temporal convs to capture inconsistency at different segment lengths
        self.tconvs = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim // 64)
            for k in kernel_sizes
        ])
        self.tconv_proj = nn.Linear(dim * len(kernel_sizes), dim)

        # Global prototype via attention pooling (learnable query)
        self.global_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.global_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Local-global comparison
        self.compare_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        # Gated residual fusion of the discrepancy embedding back into features
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, real_mask_soft=None):
        """
        x: (B, T, D) - CMIE-fused features
        real_mask_soft: (B, T) optional soft mask indicating likelihood of being
                        real, used to compute a clean global prototype. During
                        inference this can be the model's own previous
                        prediction (or all-ones when not yet available).

        Returns:
            out:  (B, T, D) - enhanced features encoding semantic discrepancy
            s_t:  (B, T)    - per-timestep raw semantic-deviation evidence
        """
        B, T, D = x.shape

        # 1) Multi-scale temporal context
        x_t = x.transpose(1, 2)  # (B, D, T)
        ms = [tconv(x_t) for tconv in self.tconvs]
        ms = torch.cat(ms, dim=1).transpose(1, 2)   # (B, T, D*K)
        x_local = self.tconv_proj(ms)               # (B, T, D)

        # 2) Global prototype (optionally weighted by real_mask_soft)
        if real_mask_soft is not None:
            # Detach so the prototype cannot be trivially driven toward fakes.
            w = real_mask_soft.detach().unsqueeze(-1).clamp(min=1e-3)
            x_for_global = x * w
        else:
            x_for_global = x

        q = self.global_query.expand(B, -1, -1)                 # (B, 1, D)
        g, _ = self.global_attn(q, x_for_global, x_for_global)  # (B, 1, D)
        g_expand = g.expand(-1, T, -1)                          # (B, T, D)

        # 3) Local-global semantic discrepancy
        cat = torch.cat([x_local, g_expand], dim=-1)  # (B, T, 2D)
        delta = self.compare_mlp(cat)                 # (B, T, D)

        # Cosine-based deviation: low cosine similarity = high discrepancy
        cos_sim = F.cosine_similarity(x_local, g_expand, dim=-1)  # (B, T)
        s_t = 1.0 - cos_sim  # in [0, 2]; higher = more semantically inconsistent

        # 4) Gated residual: blend discrepancy embedding back into features
        gate = self.gate(delta)
        out = self.norm(x + gate * delta)

        return out, s_t
