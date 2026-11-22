"""Joint rendering of the static point cloud and the MANO hand meshes into one frame.

Points and hand triangles share a single z-buffer, so hands are hidden behind scene geometry
that is closer to the camera and vice versa.

Points: nearest-point splat; holes up to fill_px pixels take the colour and depth of the nearest
point, larger disocclusions stay empty (gray 127 in the image, white in the mask).
Hands: perspective-correct triangle rasterization with the shading of the training renderer
(PyTorch3D SoftPhongShader, faces_per_pixel=1): skin (0.95, 0.72, 0.56), point light at the world
origin, ambient 0.7 / diffuse 0.6 / specular 0.1, shininess 64; written in grayscale with luminance
weights [0.2989, 0.5870, 0.1140], as in the training controls.

Mask: 255 = the model must generate (empty pixels and visible hand pixels), 0 = known scene.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

SKIN = np.array([0.95, 0.72, 0.56])
AMBIENT, DIFFUSE, SPECULAR, SHININESS = 0.7, 0.6, 0.1, 64.0
LUMA = np.array([0.2989, 0.5870, 0.1140])


def _vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-6)


def _splat_points(points, colors, w2c, K, size, fill_px):
    h, w = size
    pc = points @ w2c[:3, :3].T.astype(np.float32) + w2c[:3, 3].astype(np.float32)
    z = pc[:, 2]
    ok = np.isfinite(pc).all(1) & (z > 0.05) & (z < 20.0)
    pc, z, rgb = pc[ok], z[ok], colors[ok]
    u = np.rint(K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(np.int64)
    v = np.rint(K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    linear, z, rgb = v[inside] * w + u[inside], z[inside], rgb[inside]
    image = np.zeros((h * w, 3), np.float64)
    zbuf = np.full(h * w, np.inf)
    if len(linear):
        order = np.lexsort((z, linear))
        first = np.r_[True, linear[order][1:] != linear[order][:-1]]
        picked = order[first]
        image[linear[picked]] = rgb[picked]
        zbuf[linear[picked]] = z[picked]
    image, zbuf = image.reshape(h, w, 3), zbuf.reshape(h, w)
    covered = np.isfinite(zbuf)
    if covered.any():
        dist, (iy, ix) = distance_transform_edt(~covered, return_indices=True)
        fill = ~covered & (dist <= fill_px)
        image[fill] = image[iy[fill], ix[fill]]
        zbuf[fill] = zbuf[iy[fill], ix[fill]]
    return image, zbuf


def _raster_meshes(image, zbuf, hand, meshes, w2c, K):
    """Rasterize hand triangles into image/zbuf in place; hand marks pixels won by a hand."""
    h, w = zbuf.shape
    R, t = w2c[:3, :3], w2c[:3, 3]
    cam_center = -R.T @ t
    for vw, f in meshes:
        vn = _vertex_normals(vw, f)
        vc = vw @ R.T + t
        uv = (vc @ K.T)[:, :2] / vc[:, 2:3]
        for tri in f:
            z = vc[tri, 2]
            if (z <= 0.01).any():
                continue
            p = uv[tri]
            x0, x1 = int(max(np.floor(p[:, 0].min()), 0)), int(min(np.ceil(p[:, 0].max()), w - 1))
            y0, y1 = int(max(np.floor(p[:, 1].min()), 0)), int(min(np.ceil(p[:, 1].max()), h - 1))
            if x0 > x1 or y0 > y1:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.0, np.arange(y0, y1 + 1) + 0.0)
            a, b, c = p
            den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(den) < 1e-12:
                continue
            l0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
            l1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            q = np.stack([l0 / z[0], l1 / z[1], l2 / z[2]], -1)  # perspective-correct barycentrics
            zz = 1.0 / q.sum(-1)
            q = q * zz[..., None]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (zz < sub)  # shared depth test against points and other triangles
            if not upd.any():
                continue
            sub[upd] = zz[upd]
            n = q[upd] @ vn[tri]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
            P = q[upd] @ vw[tri]
            L = -P
            L /= np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-6)
            V = cam_center - P
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-6)
            cos = (n * L).sum(1)
            refl = -L + 2 * cos[:, None] * n
            spec = np.clip((V * refl).sum(1), 0, None) ** SHININESS * (cos > 0)
            col = np.clip((AMBIENT + DIFFUSE * np.clip(cos, 0, None))[:, None] * SKIN + SPECULAR * spec[:, None], 0, 1)
            gray = np.clip((col * 255.0) @ LUMA, 0, 255)
            image[y0:y1 + 1, x0:x1 + 1][upd] = gray[:, None]
            hand[y0:y1 + 1, x0:x1 + 1][upd] = True


def render_frame(points: np.ndarray, colors: np.ndarray, meshes, world_to_camera: np.ndarray, K: np.ndarray,
                 size: tuple[int, int], fill_px: float = 2.0,
                 hole_value: int = 127) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render points + hand meshes together. Returns RGB uint8, mask uint8, visible-hand bool."""
    image, zbuf = _splat_points(points, colors, world_to_camera, K, size, fill_px)
    hand = np.zeros(size, bool)
    _raster_meshes(image, zbuf, hand, meshes, world_to_camera, K)
    empty = ~np.isfinite(zbuf)
    image[empty] = hole_value
    mask = np.where(empty | hand, 255, 0).astype(np.uint8)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8), mask, hand
