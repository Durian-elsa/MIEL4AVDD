# MIEL: Multi-Level Inconsistency Evidence Learning

A unified framework for **audio-visual deepfake detection** (video-level binary) and
**localization** (frame-level segment prediction), targeting high-fidelity
deepfakes where temporal smoothness is preserved but **semantic** consistency is
violated.

> Formerly developed under the working name `SeMaT-Net`.

> **Scope of this release.** This repository ships the model architecture
> (CMIE / SDEE / MLEA) and an inference-time forward pass. Training code —
> loss functions, the two-stage training schedule, and the data pipeline —
> is not included. Bring your own checkpoint and your own video/audio
> loading to run inference.

## Why this design?

Traditional deepfake detectors rely on temporal artifacts (frame jitter, optical-flow
inconsistencies). State-of-the-art deepfakes preserve temporal continuity, so we
must look for **semantic** breaks instead. MIEL gathers three complementary,
per-timestep evidence signals and aggregates them for the final decision:

1. **CMIE — Cross-Modal Inconsistency Encoding.** Real video + audio are tightly
   aligned semantically. A face-swapped or lip-synced segment breaks this
   alignment even when each modality looks smooth on its own. Bidirectional
   cross-attention between visual and audio tokens surfaces these mismatches
   and produces `alpha_t`, the cross-modal inconsistency evidence.

2. **SDEE — Semantic Discrepancy Evidence Extraction.** Build a global semantic
   prototype from the (likely) real regions of the video. Compare each
   timestep to this prototype: manipulated segments deviate even when their
   immediate neighbors look fine. Multi-scale temporal convolutions capture
   discrepancies at different segment lengths and produce `s_t`, the semantic
   discrepancy evidence.

3. **MLEA — Multi-Level Evidence Aggregation.** An anchor-free localization head
   predicts a per-frame manipulation logit `l_hat_t` plus boundary offsets
   (start, end). A classifier then aggregates all three evidence signals
   (`alpha_t`, `s_t`, `l_hat_t`) — each detached, each weighted by a learnable
   scalar — into the final video-level decision. This is robust to noisy
   frame-level predictions and recovers precise segment boundaries, which is
   critical for LAV-DF / AV-Deepfake1M-style datasets.

The three evidence signals (`alpha_t`, `s_t`, `l_hat_t`) are combined by MLEA's
attention-pooling classifier into a single video-level decision, alongside
per-frame localization output. See the paper for the full training
objective; it is not reproduced in this repository.

## Project Layout

```
MIEL/
├── models/
│   ├── backbones.py                # VisualBackbone (ViT/VideoMAE), AudioBackbone (Wav2Vec2)
│   ├── cmie.py                     # CMIE: bidirectional cross-attention -> alpha_t
│   ├── sdee.py                     # SDEE: local-vs-global semantic discrepancy -> s_t
│   └── mlea.py                     # MLEA: LocalizationHead (-> l_hat_t) + MLEAClassifier
│   └── mielnet.py                  # full model (MIELNet), forward pass only
├── utils/metrics.py                # AUC, Acc, AP@IoU, AR@N (for evaluating your own runs)
└── scripts/
    └── inference.py                # sliding-window inference on a video, given a checkpoint
```

Training code (loss functions, two-stage schedule, dataset/annotation
pipeline) is intentionally not part of this repository.

## Getting Started

### Install
```bash
pip install torch torchvision torchaudio transformers scikit-learn
```

### Inference
Bring your own checkpoint (trained with your own training pipeline) and
video/audio loading code:
```bash
python scripts/inference.py \
       --ckpt checkpoints/mielnet/best.pt \
       --video sample.mp4
```
If you have a local Wav2Vec2 checkpoint, point to it instead of hitting the
Hugging Face hub:
```bash
python scripts/inference.py ... --audio_local_ckpt /path/to/wav2vec2-base-960h
```
`scripts/inference.py` wires up model loading and documents the expected
input tensor shapes; video/audio decoding is left to your own pipeline.

## Evaluation

`utils/metrics.py` provides AUC, Accuracy, AP@IoU, and AR@N helpers
(following the LAV-DF / AV-Deepfake1M protocol) for evaluating outputs from
your own training and inference runs.

## Notes for Contributors / Before Publishing

- `AudioBackbone` accepts an optional `local_path` for a cached Wav2Vec2
  checkpoint — pass it via `--audio_local_ckpt`, never hardcode a machine-local
  path in the model source.
- No checkpoints, dataset files, or annotation jsonl files should be committed;
  keep them out via `.gitignore` (`checkpoints/`, `data/`, `*.pt`).
