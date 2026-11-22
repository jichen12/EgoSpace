"""Inference: first frame + rendered video/mask with hands + cameras + object views + text prompt.

Example (reproduces the withdraw clip of the project page from its stored control video):

    python examples/inference.py \
        --first-frame   data/withdraw_clip-001995/first_frame.png \
        --control-video data/withdraw_clip-001995/control.mp4 \
        --control-mask  data/withdraw_clip-001995/control_mask.mp4 \
        --camera        data/withdraw_clip-001995/camera.npz \
        --object-images data/withdraw_clip-001995/object_views/view_{1,2,3}.png \
        --prompt        "$(cat data/withdraw_clip-001995/prompt.txt)" \
        --checkpoint    checkpoints/egospace/final.pt \
        --wan-dir       checkpoints/Wan2.1-I2V-14B-720P \
        --output        outputs/withdraw_clip-001995/egospace.mp4

--control-video / --control-mask may also be the outputs of examples/preprocess.py.
The model generates 81 frames at 480x480 (inputs are resized); needs one GPU with ~65 GB.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egospace.inference import generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-frame", type=Path, required=True, help="initial RGB frame (I2V condition)")
    ap.add_argument("--control-video", type=Path, required=True, help="rendered scene with grayscale hands, 81 frames")
    ap.add_argument("--control-mask", type=Path, required=True, help="white = generate, black = known scene")
    ap.add_argument("--camera", type=Path, required=True, help="camera.npz (K, world_to_camera)")
    ap.add_argument("--object-images", type=Path, nargs=3, required=True, help="three views of the manipulated object")
    ap.add_argument("--prompt", required=True, help="text prompt")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--output", type=Path, required=True, help="output .mp4")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/egospace/final.pt")
    ap.add_argument("--wan-dir", type=Path, default=ROOT / "checkpoints/Wan2.1-I2V-14B-720P")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--sampling-steps", type=int, default=40)
    ap.add_argument("--guide-scale", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    out = generate(a.first_frame, a.control_video, a.control_mask, a.camera, a.object_images, a.prompt,
                   a.output, checkpoint=a.checkpoint, wan_dir=a.wan_dir, negative_prompt=a.negative_prompt,
                   seed=a.seed, sampling_steps=a.sampling_steps, guide_scale=a.guide_scale, shift=a.shift,
                   device=a.device)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
