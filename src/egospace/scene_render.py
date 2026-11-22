"""First frame -> metric depth -> coloured point cloud (rendered with the hands in render.py)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def estimate_depth(rgb: np.ndarray, focal_px: float, checkpoint: str | Path, device: str) -> np.ndarray:
    """Depth Pro metric depth [H,W] in metres, using the known focal length."""
    import torch
    os.environ["DEPTH_PRO_CHECKPOINT"] = str(checkpoint)
    from scr.utils.depth_pro import depth_pro

    model, transform = depth_pro.create_model_and_transforms(device=torch.device(device))
    model.eval()
    with torch.inference_mode():
        pred = model.infer(transform(rgb.copy()), f_px=torch.tensor(float(focal_px), device=device))
    depth = pred["depth"].float().cpu().numpy()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return depth


def unproject(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, world_to_camera: np.ndarray,
              valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    rays = np.stack((xx, yy, np.ones_like(xx)), -1).reshape(-1, 3) @ np.linalg.inv(K).T
    points_cam = rays * depth.reshape(-1, 1)
    c2w = np.linalg.inv(world_to_camera)
    points = points_cam @ c2w[:3, :3].T + c2w[:3, 3]
    keep = valid.reshape(-1) & np.isfinite(points).all(1)
    return points[keep].astype(np.float32), rgb.reshape(-1, 3)[keep]
