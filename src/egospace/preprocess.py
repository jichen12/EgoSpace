"""Build the unified 3D action video (control video + mask) from first frame, camera trajectory and MANO.

  first frame --(hand mask: given or SAM 3)--> hand-free pixels
              --(Depth Pro, or given depth)--> coloured point cloud in the world frame
  point cloud + MANO meshes, rendered together per camera with one shared z-buffer
  (hands in grayscale) ----------------------> control.mp4 / control_mask.mp4

Mask convention (as in training): white = the model must generate (disoccluded pixels and hands),
black = known static scene from the first frame.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .io import load_camera, load_mano, write_video
from .mano import mano_vertices
from .render import render_frame
from .scene_render import estimate_depth, unproject


def _project(verts: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> np.ndarray:
    pc = verts @ w2c[:3, :3].T + w2c[:3, 3]
    pc = pc[pc[:, 2] > 0.05]
    return np.rint((pc @ K.T)[:, :2] / pc[:, 2:3]).astype(np.int64)


def preprocess(first_frame: str | Path, camera: str | Path, mano: str | Path, output_dir: str | Path, *,
               mano_model_dir: str | Path, hand_mask: str | Path | None = None, depth: str | Path | None = None,
               depth_pro_checkpoint: str | Path | None = None, sam3_checkpoint: str | Path | None = None,
               device: str = "cuda", max_size: int = 640, fps: float = 16.0) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(first_frame)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    size = rgb.shape[:2]
    cam = load_camera(camera)
    K, w2c = cam["K"], cam["world_to_camera"]
    n = len(w2c)
    hands = load_mano(mano)
    verts, faces = mano_vertices(hands, mano_model_dir)
    for side, v in verts.items():
        if len(v) != n:
            raise ValueError(f"{side} MANO has {len(v)} frames, camera has {n}")

    # 1. first-frame hand mask: hand pixels must not enter the static point cloud
    if hand_mask is not None:
        mask0 = cv2.imread(str(hand_mask), cv2.IMREAD_GRAYSCALE)
        if mask0 is None:
            raise FileNotFoundError(hand_mask)
        mask0 = cv2.resize(mask0, size[::-1], interpolation=cv2.INTER_NEAREST) > 127
        mask_source = str(hand_mask)
    else:
        if sam3_checkpoint is None:
            raise ValueError("Give --hand-mask, or --sam3-checkpoint to segment the hands with SAM 3")
        from .hand_seg import sam3_hand_mask
        projected = [_project(v[0], w2c[0], K[0]) for v in verts.values() if np.isfinite(v[0]).all()]
        mask0 = sam3_hand_mask(rgb, projected, sam3_checkpoint, device)
        mask_source = "sam3"
    cv2.imwrite(str(out / "hand_mask.png"), mask0.astype(np.uint8) * 255)

    # 2. metric depth of the first frame
    if depth is not None:
        depth_map = np.load(depth).astype(np.float32)
        if depth_map.shape != size:
            raise ValueError(f"depth {depth_map.shape} does not match first frame {size}")
    else:
        if depth_pro_checkpoint is None:
            raise ValueError("Give --depth, or --depth-pro-checkpoint to estimate it with Depth Pro")
        depth_map = estimate_depth(rgb, (K[0, 0, 0] + K[0, 1, 1]) / 2, depth_pro_checkpoint, device)
    np.save(out / "depth.npy", depth_map)

    # 3. static scene point cloud, reprojected along the trajectory. Renders are made at most
    # max_size px on the long side (training controls: HOT3D 608x608, PICO 640x480); camera.npz
    # keeps the intrinsics of the full-resolution first frame, as in training.
    scale = min(1.0, max_size / max(size))
    rsize = (round(size[0] * scale), round(size[1] * scale))
    Kr = K.copy()
    Kr[:, :2] *= np.array([rsize[1] / size[1], rsize[0] / size[0]])[None, :, None]
    if rsize != size:
        rgb_r = cv2.resize(rgb, rsize[::-1], interpolation=cv2.INTER_AREA)
        depth_r = cv2.resize(depth_map, rsize[::-1], interpolation=cv2.INTER_NEAREST)
        mask_r = cv2.resize(mask0.astype(np.uint8), rsize[::-1], interpolation=cv2.INTER_NEAREST) > 0
    else:
        rgb_r, depth_r, mask_r = rgb, depth_map, mask0
    valid = np.isfinite(depth_r) & (depth_r > 0.05) & ~mask_r
    points, colors = unproject(rgb_r, depth_r, Kr[0], w2c[0], valid)

    # 4. point cloud and hand meshes rendered together (shared z-buffer, hands in grayscale)
    ctrl_v, ctrl_m, hand_px = [], [], []
    for t in range(n):
        meshes = [(v[t], faces[s]) for s, v in verts.items() if np.isfinite(v[t]).all()]
        frame, mask, hand = render_frame(points, colors, meshes, w2c[t], Kr[t], rsize)
        ctrl_v.append(frame)
        ctrl_m.append(mask)
        hand_px.append(hand)
        print(f"\rrendered frame {t + 1}/{n}", end="", flush=True)
    print()
    write_video(out / "control.mp4", ctrl_v, fps)
    write_video(out / "control_mask.mp4", ctrl_m, fps)
    np.savez_compressed(out / "control_mask.npz", mask=np.stack(ctrl_m) > 127)
    for src, dst in ((Path(first_frame), out / "first_frame.png"), (Path(camera), out / "camera.npz")):
        if src.resolve() != dst.resolve():
            if dst.suffix == ".png" and src.suffix.lower() != ".png":
                cv2.imwrite(str(dst), bgr)
            else:
                shutil.copy2(src, dst)
    report = {"frames": n, "render_size_hw": list(rsize), "points": int(len(points)),
              "hand_mask": mask_source, "depth": str(depth) if depth is not None else "depth_pro",
              "mean_known_scene_fraction": float(np.mean([(m == 0).mean() for m in ctrl_m])),
              "mean_visible_hand_fraction": float(np.mean([h.mean() for h in hand_px]))}
    (out / "preprocess.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
