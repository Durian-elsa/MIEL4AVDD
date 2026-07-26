"""
MIEL inference script (sliding-window over a full video).

This release ships the model architecture and a forward pass only — training
code, loss functions, and the two-stage training schedule described in the
paper are not included here. Point --ckpt at a checkpoint produced by your
own training run.

Usage:
    python scripts/inference.py --ckpt path/to/checkpoint.pt --video sample.mp4
"""

import argparse

import torch

from models import MIELNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to a trained checkpoint (state_dict).")
    p.add_argument("--video", required=True, help="Path to an input video file.")
    p.add_argument("--feature_dim", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--window_stride", type=int, default=16,
                   help="Frame stride between sliding-window clips.")
    p.add_argument("--audio_local_ckpt", default=None,
                   help="Optional local Wav2Vec2 checkpoint directory.")
    return p.parse_args()


def load_model(args, device):
    model = MIELNet(
        feature_dim=args.feature_dim,
        num_frames=args.num_frames,
        audio_local_ckpt=args.audio_local_ckpt,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    # NOTE: video/audio loading and sliding-window clip extraction are left
    # to your own data pipeline -- plug in your video decoding utility here
    # to produce (video_clip, audio_clip) tensors matching MIELNet.forward's
    # expected shapes: video (1, T, C, H, W), audio (1, L) at 16kHz.
    raise NotImplementedError(
        "Wire up your own video/audio loading here, then call: "
        "out = model(video_clip, audio_clip); "
        "video_prob = torch.sigmoid(out['video_logit']); "
        "frame_prob = torch.sigmoid(out['frame_logits'])"
    )


if __name__ == "__main__":
    main()
