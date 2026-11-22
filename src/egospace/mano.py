"""MANO forward kinematics (smplx) for the hand tracks in mano.npz."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

# chumpy (needed to unpickle the official MANO files) uses names removed from
# recent Python / NumPy.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _value in {"bool": bool, "int": int, "float": float, "complex": complex,
                      "object": object, "str": str, "unicode": str}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import smplx  # noqa: E402
import torch  # noqa: E402


def mano_vertices(mano: dict, model_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return world vertices {side: [N,778,3]} (NaN where absent) and faces {side: [F,3]}."""
    model_dir = Path(model_dir)
    verts, faces = {}, {}
    for side, hand in mano["hands"].items():
        pose, wrist = hand["hand_pose"], hand["wrist"]
        if pose.shape[1] not in (15, 45):
            raise ValueError(f"{side}_hand_pose must have 15 (PCA) or 45 (axis-angle) columns")
        path = model_dir / f"MANO_{side.upper()}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"MANO model missing: {path} (download from https://mano.is.tue.mpg.de)")
        layer = smplx.create(str(path), "mano", use_pca=pose.shape[1] == 15, num_pca_comps=15,
                             is_rhand=side == "right", flat_hand_mean=mano["flat_hand_mean"]).eval()
        if side == "left" and mano["left_shapedirs_fix"]:
            layer.shapedirs[:, 0, :] *= -1
        out = np.full((len(wrist), 778, 3), np.nan)
        ok = np.isfinite(wrist).all(1) & np.isfinite(pose).all(1)
        if ok.any():
            w = torch.as_tensor(wrist[ok])
            with torch.no_grad():
                result = layer(betas=torch.as_tensor(hand["betas"][:10])[None].expand(int(ok.sum()), -1),
                               global_orient=w[:, :3], hand_pose=torch.as_tensor(pose[ok]),
                               transl=w[:, 3:], return_verts=True)
            out[ok] = result.vertices.numpy()
        verts[side] = out
        faces[side] = np.asarray(layer.faces, np.int64)
    return verts, faces
