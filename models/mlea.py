"""
MLEA: Multi-Level Evidence Aggregation

Two heads share the same refined features (output of SDEE) but serve
different roles:

- LocalizationHead: an anchor-free, per-timestep head that predicts
  (a) whether a timestep is inside a manipulated segment (frame-level logit,
      l_hat_t -- the third and final evidence signal), and
  (b) the offset from that timestep to the nearest fake-segment boundaries
      (start, end). This follows the design philosophy of anchor-free temporal
      action localization (e.g. ActionFormer): precise boundary recovery even
      when frame-level predictions are noisy.

- MLEAClassifier: a video-level binary classifier that aggregates the THREE
  complementary evidence sources produced across the pipeline, each capturing
  manipulation cues at a different representation depth:
    alpha_t  - cross-modal inconsistency evidence, from CMIE (shallow, modality-specific)
    s_t      - semantic discrepancy evidence, from SDEE (mid-level, context-aware)
    l_hat_t  - contextual/localization evidence, from LocalizationHead (deep, refined)
  All three are detached before being folded into the attention-pooling
  weights, so the classification loss cannot back-propagate through the
  auxiliary supervision paths of the other modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLEAClassifier(nn.Module):
    """Video-level binary classification with multi-level evidence-conditioned attention pooling."""

    def __init__(self, dim=512, num_classes=1, dropout=0.1):
        super().__init__()
        # Per-timestep feature salience scorer
        self.attn_pool = nn.Linear(dim, 1)

        # Learnable weight for each evidence source.
        # Priors: l_hat_t (deep, most direct) starts strongest; alpha_t and
        # s_t start at half weight as shallower/auxiliary signals.
        self.lambda_l     = nn.Parameter(torch.tensor(1.0))   # weight on l_hat_t
        self.lambda_alpha = nn.Parameter(torch.tensor(0.5))   # weight on alpha_t
        self.lambda_s     = nn.Parameter(torch.tensor(0.5))   # weight on s_t

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(self, x, l_hat_t=None, alpha_t=None, s_t=None):
        """
        x:        (B, T, D) refined features from SDEE
        l_hat_t:  (B, T) per-frame contextual/localization evidence (from LocalizationHead)
        alpha_t:  (B, T) cross-modal inconsistency evidence (from CMIE)
        s_t:      (B, T) semantic discrepancy evidence (from SDEE), in [0, 2] range
                         (not already logit-scaled; rescaled below)

        All evidence signals are detached to stop gradients from the
        classification loss flowing back through the auxiliary paths.
        """
        # Feature-based salience
        attn = self.attn_pool(x).squeeze(-1)  # (B, T)

        if l_hat_t is not None:
            attn = attn + self.lambda_l * l_hat_t.detach()

        if alpha_t is not None:
            attn = attn + self.lambda_alpha * alpha_t.detach()

        if s_t is not None:
            # s_t is in [0, 2] (= 1 - cosine_sim); rescale to a logit-like
            # magnitude by centering around 0 and stretching (~3x), so it is
            # comparable in scale to the other logit-valued evidence.
            scaled_s = (s_t - 1.0) * 3.0  # roughly in [-3, 3]
            attn = attn + self.lambda_s * scaled_s.detach()

        # Pool
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = (x * attn).sum(dim=1)                # (B, D)

        # Classify
        logits = self.classifier(pooled)              # (B, 1)
        return logits.squeeze(-1)


class LocalizationHead(nn.Module):
    """
    Anchor-free frame-level localization head.

    Outputs:
      - l_hat_t: (B, T)    per-frame manipulation logit (the third evidence
                            signal consumed by MLEAClassifier)
      - offsets: (B, T, 2) (start_offset, end_offset) regression to the
                            nearest fake-segment boundaries. Valid only where
                            l_hat_t indicates a manipulated frame.
    """

    def __init__(self, dim=512, hidden=256, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv1d(dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.cls_branch = nn.Conv1d(hidden, 1, 1)
        self.reg_branch = nn.Conv1d(hidden, 2, 1)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x_t = x.transpose(1, 2)                          # (B, D, T)
        h = self.shared(x_t)                              # (B, hidden, T)
        l_hat_t = self.cls_branch(h).squeeze(1)            # (B, T)
        offsets_raw = self.reg_branch(h).transpose(1, 2)   # (B, T, 2)
        offsets = F.softplus(offsets_raw)                  # ensure non-negative (distance in frames)
        return l_hat_t, offsets
