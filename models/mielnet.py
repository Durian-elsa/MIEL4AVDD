"""
MIEL: Multi-Level Inconsistency Evidence Learning

Full pipeline:
  Video --> VisualBackbone -----.
                                +--> CMIE --> SDEE --> {LocalizationHead, MLEAClassifier}
  Audio --> AudioBackbone ------'

Two outputs:
  - video-level binary logit (real vs fake)
  - frame-level fake probability + boundary offsets

Three evidence signals flow through the pipeline and are jointly aggregated:
  - alpha_t  (CMIE)              cross-modal inconsistency
  - s_t      (SDEE)              semantic discrepancy (local vs. global)
  - l_hat_t  (LocalizationHead)  contextual/localization inconsistency

This module defines the forward (inference) pass only. Training code
(loss functions, two-stage training schedule, data pipeline) is not part of
this release.
"""

import torch
import torch.nn as nn

from .backbones import VisualBackbone, AudioBackbone, temporal_resample
from .cmie import CMIE
from .sdee import SDEE
from .mlea import MLEAClassifier, LocalizationHead


class MIELNet(nn.Module):
    def __init__(self,
                 feature_dim=512,
                 visual_backbone="vit",
                 audio_backbone="wav2vec2",
                 num_cm_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 freeze_visual_layers=8,
                 num_frames=32,
                 audio_local_ckpt=None):
        """
        audio_local_ckpt: optional path to a local Wav2Vec2 checkpoint. If None,
        falls back to fetching `audio_backbone` (a HF model name) from the hub.
        Set this via your config/CLI rather than hardcoding a path here.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames

        # 1) Backbones
        self.visual_backbone = VisualBackbone(
            backbone_name=visual_backbone,
            pretrained=True,
            feature_dim=feature_dim,
            freeze_layers=freeze_visual_layers,
        )
        self.audio_backbone = AudioBackbone(
            model_name=None if audio_local_ckpt else audio_backbone,
            local_path=audio_local_ckpt,
            feature_dim=feature_dim,
            freeze_feature_extractor=True,
        )

        # 2) CMIE: Cross-Modal Inconsistency Encoding
        self.cmie = CMIE(
            dim=feature_dim, num_layers=num_cm_layers,
            num_heads=num_heads, dropout=dropout,
        )

        # 3) SDEE: Semantic Discrepancy Evidence Extraction
        self.sdee = SDEE(
            dim=feature_dim, num_heads=num_heads,
            kernel_sizes=(3, 5, 7), dropout=dropout,
        )

        # 4) MLEA: Multi-Level Evidence Aggregation
        self.mlea_classifier = MLEAClassifier(dim=feature_dim, dropout=dropout)
        self.loc_head = LocalizationHead(dim=feature_dim, dropout=dropout)

    def forward(self, video, audio, return_features=False):
        """
        video: (B, T, C, H, W)   T = num_frames
        audio: (B, L)            raw waveform at 16kHz
        """
        # 1) Feature extraction
        Fv = self.visual_backbone(video)              # (B, T, D)
        Fa = self.audio_backbone(audio)                # (B, T_a, D)

        # Align temporal axes
        T = Fv.shape[1]
        Fa = temporal_resample(Fa, T)                  # (B, T, D)

        # 2) CMIE: cross-modal reasoning
        v_out, a_out, fused, alpha_t = self.cmie(Fv, Fa)
        # alpha_t: (B, T) - cross-modal inconsistency evidence per timestep

        # 3) First-pass coarse prediction, used to build a clean global prototype
        with torch.no_grad():
            init_logits, _ = self.loc_head(fused)
            real_mask_soft = torch.sigmoid(-init_logits)

        # 4) SDEE: refine features w.r.t. the clean global prototype
        refined, s_t = self.sdee(fused, real_mask_soft=real_mask_soft)
        # s_t: (B, T) - local-vs-global semantic deviation, range [0, 2]

        # 5) MLEA: final predictions
        # Localization head: second call, on refined features (shares params with the first call)
        l_hat_t, offsets = self.loc_head(refined)

        # Classifier: aggregates all three evidence signals
        video_logit = self.mlea_classifier(
            refined,
            l_hat_t=l_hat_t,
            alpha_t=alpha_t,
            s_t=s_t,
        )

        out = {
            "video_logit": video_logit,   # (B,)
            "frame_logits": l_hat_t,      # (B, T)  -- l_hat_t
            "offsets": offsets,           # (B, T, 2)
            "alignment_logits": alpha_t,  # (B, T)  -- alpha_t
            "inconsist_score": s_t,       # (B, T)  -- s_t
        }
        if return_features:
            out["features"] = refined
        return out


