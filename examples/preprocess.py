"""Preprocess: hand MANO + camera trajectory + initial frame -> unified 3D action video.

Example (the withdraw clip of the project page):

    python examples/preprocess.py \
        --first-frame data/withdraw_clip-001995/first_frame.png \
        --camera      data/withdraw_clip-001995/camera.npz \
        --mano        data/withdraw_clip-001995/mano.npz \
        --hand-mask   data/withdraw_clip-001995/hand_mask.png \
        --mano-model-dir checkpoints/mano \
        --depth-pro-checkpoint checkpoints/depth_pro.pt \
        --output-dir  outputs/withdraw_clip-001995

Renders the point cloud and the MANO hands together (shared z-buffer) into control.mp4 +
control_mask.mp4 (the inputs of examples/inference.py); also writes depth.npy and hand_mask.png. File formats of
camera.npz and mano.npz: src/egospace/io.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egospace.preprocess import preprocess  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-frame", type=Path, required=True, help="initial RGB frame")
    ap.add_argument("--camera", type=Path, required=True, help="camera.npz: K, world_to_camera [N,4,4]")
    ap.add_argument("--mano", type=Path, required=True, help="mano.npz: per-hand MANO tracks [N]")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mano-model-dir", type=Path, default=ROOT / "checkpoints/mano",
                    help="folder with MANO_LEFT.pkl / MANO_RIGHT.pkl")
    ap.add_argument("--hand-mask", type=Path, help="first-frame hand mask (white = hand); else SAM 3 is used")
    ap.add_argument("--depth", type=Path, help="metric depth of the first frame (.npy, metres); else Depth Pro")
    ap.add_argument("--depth-pro-checkpoint", type=Path, default=ROOT / "checkpoints/depth_pro.pt")
    ap.add_argument("--sam3-checkpoint", type=Path, default=ROOT / "checkpoints/sam3.pt")
    ap.add_argument("--max-size", type=int, default=640, help="long side of the rendered videos (px)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    report = preprocess(a.first_frame, a.camera, a.mano, a.output_dir, mano_model_dir=a.mano_model_dir,
                        hand_mask=a.hand_mask, depth=a.depth, depth_pro_checkpoint=a.depth_pro_checkpoint,
                        sam3_checkpoint=a.sam3_checkpoint, device=a.device,
                        max_size=a.max_size)
    print(report)
    print(f"control video: {a.output_dir / 'control.mp4'}\ncontrol mask:  {a.output_dir / 'control_mask.mp4'}")


if __name__ == "__main__":
    main()
