"""
CMIE: Cross-Modal Inconsistency Encoding

Core idea: real videos have tightly aligned audio-visual semantics. Manipulated
segments break this alignment even when temporal continuity is preserved.

We use bidirectional cross-attention so that:
- Visual tokens attend to audio context (catches lip-sync mismatches at the
  semantic level)
- Audio tokens attend to visual context (catches scene-audio mismatches)

The output is a fused representation per timestep plus a per-timestep
cross-modal inconsistency evidence score, alpha_t, which is one of the three
evidence signals aggregated by MLEA.
"""

import torch
import torch.nn as nn


class CMIEBlock(nn.Module):
    """One block of bidirectional cross-attention with FFN."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_a1 = nn.LayerNorm(dim)
        self.cross_v2a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_a2v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.self_v = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_v3 = nn.LayerNorm(dim)
        self.norm_a3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn_v = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, v, a):
        """
        v: (B, T, D) visual tokens
        a: (B, T, D) audio tokens (already temporally aligned)
        """
        # Cross attention (residual)
        v_norm = self.norm_v1(v)
        a_norm = self.norm_a1(a)
        v_attn, _ = self.cross_v2a(v_norm, a_norm, a_norm)  # visual queries audio
        a_attn, _ = self.cross_a2v(a_norm, v_norm, v_norm)  # audio queries visual
        v = v + v_attn
        a = a + a_attn

        # Self attention (residual)
        v_norm = self.norm_v2(v)
        a_norm = self.norm_a2(a)
        v_self, _ = self.self_v(v_norm, v_norm, v_norm)
        a_self, _ = self.self_a(a_norm, a_norm, a_norm)
        v = v + v_self
        a = a + a_self

        # FFN (residual)
        v = v + self.ffn_v(self.norm_v3(v))
        a = a + self.ffn_a(self.norm_a3(a))
        return v, a


class CMIE(nn.Module):
    """
    Cross-Modal Inconsistency Encoding.

    Stack of CMIE blocks. Outputs:
    - v_out, a_out: refined per-modality features
    - fused:        concatenated and projected feature for downstream modules (SDEE/MLEA)
    - alpha_t:      (B, T) per-timestep cross-modal inconsistency evidence.
                     Trained implicitly via the localization supervision (a
                     misaligned timestep should also be flagged as fake).
    """

    def __init__(self, dim=512, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CMIEBlock(dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.fuse_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(dim * 2, 1)

    def forward(self, v, a):
        for block in self.blocks:
            v, a = block(v, a)
        cat = torch.cat([v, a], dim=-1)          # (B, T, 2D)
        fused = self.fuse_proj(cat)              # (B, T, D)
        alpha_t = self.alpha_head(cat).squeeze(-1)  # (B, T)
        return v, a, fused, alpha_t
