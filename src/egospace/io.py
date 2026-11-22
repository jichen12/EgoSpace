"""File formats shared by preprocessing and inference.

camera.npz
    K                      [3,3] or [N,3,3]  pinhole intrinsics in pixels of first_frame
    world_to_camera        [N,4,4]           OpenCV camera (x right, y down, z forward),
                                             image-aligned, same world as the MANO wrists
    model_world_to_camera  [N,4,4] optional  extrinsics handed to the camera encoder instead of
                                             world_to_camera (the HOT3D examples keep the exact
                                             matrices used for the paper; see data/README.md)

mano.npz (smplx MANO, one entry per side; a side may be missing or NaN on frames without that hand)
    betas                  [10]   or {side}_betas [10]
    {side}_hand_pose       [N,15] PCA coefficients  or  [N,45] axis-angle finger joints
    {side}_wrist           [N,6]  global_orient axis-angle (3) + transl (3), world frame, metres
    flat_hand_mean         bool, optional (default False)
    left_shapedirs_fix     bool, optional (default False) negate shapedirs[:,0] of MANO_LEFT,
                           the known smplx left-hand bug that HOT3D's annotations compensate for
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

SIDES = ("left", "right")


def read_video(path: str | Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        return [np.asarray(frame)[..., :3] for frame in reader]
    finally:
        reader.close()


def write_video(path: str | Path, frames, fps: float = 16.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p") as writer:
        for frame in frames:
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=-1)
            writer.append_data(frame.astype(np.uint8))


def load_camera(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        w2c = np.asarray(z["world_to_camera"], np.float64)
        K = np.asarray(z["K"], np.float64)
        model = np.asarray(z["model_world_to_camera"], np.float64) if "model_world_to_camera" in z else None
    if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
        raise ValueError(f"world_to_camera must be [N,4,4], got {w2c.shape}")
    if K.ndim == 2:
        K = np.repeat(K[None], len(w2c), axis=0)
    if K.shape != (len(w2c), 3, 3):
        raise ValueError(f"K must be [3,3] or [N,3,3], got {K.shape}")
    out = {"K": K, "world_to_camera": w2c}
    if model is not None:
        out["model_world_to_camera"] = model
    return out


def load_mano(path: str | Path) -> dict:
    with np.load(path) as z:
        keys = set(z.files)
        out = {"flat_hand_mean": bool(z["flat_hand_mean"]) if "flat_hand_mean" in keys else False,
               "left_shapedirs_fix": bool(z["left_shapedirs_fix"]) if "left_shapedirs_fix" in keys else False,
               "hands": {}}
        for side in SIDES:
            if f"{side}_wrist" not in keys:
                continue
            betas = z[f"{side}_betas"] if f"{side}_betas" in keys else z["betas"]
            out["hands"][side] = {"betas": np.asarray(betas, np.float32).reshape(-1),
                                  "hand_pose": np.asarray(z[f"{side}_hand_pose"], np.float32),
                                  "wrist": np.asarray(z[f"{side}_wrist"], np.float32)}
    if not out["hands"]:
        raise ValueError(f"{path} has no left_wrist / right_wrist entries")
    return out
