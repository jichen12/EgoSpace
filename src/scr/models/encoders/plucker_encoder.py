"""Plucker ray helpers and trainable Plucker-token encoder.

This file owns both dense Plucker ray construction helpers and the trainable
path that concatenates dense rays with the rendered mask, downsamples by
``(4, 8, 8)`` then ``(1, 2, 2)``, and returns Wan-width Plucker tokens.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    """Return a module whose parameters are initialized to exact zero."""

    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class PluckerEncoder(nn.Module):
    """Encode dense Plucker rays and render masks into Wan-token Plucker controls."""

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 256,
        out_dim: int = 5120,
    ) -> None:
        """Create the trainable Plucker-mask downsampler and zero output projection."""

        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)

        # Left temporal padding maps T=81 to T=21 with stride-(4, 8, 8)
        # while preserving causal alignment with the video latents.
        self.raw_padding = (0, 0, 0, 0, 3, 0)
        self.mask_proj = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8)),
            nn.GroupNorm(max(1, self.hidden_dim // 8), self.hidden_dim),
            nn.SiLU(),
        )
        self.mask_zero_proj = zero_module(
            nn.Conv3d(self.hidden_dim, self.out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

    def forward(self, dense_plucker: torch.Tensor, rendered_mask: torch.Tensor) -> torch.Tensor:
        """Return Plucker tokens shaped ``[B, T_lat * H/16 * W/16, out_dim]``."""

        if dense_plucker.ndim != 5:
            raise ValueError(f"dense_plucker must be [B, 6, T, H, W], got {tuple(dense_plucker.shape)}.")
        if rendered_mask.ndim != 5:
            raise ValueError(f"rendered_mask must be [B, 1, T, H, W], got {tuple(rendered_mask.shape)}.")
        if dense_plucker.shape[0] != rendered_mask.shape[0]:
            raise ValueError("dense_plucker and rendered_mask batch sizes must match.")
        if dense_plucker.shape[2:] != rendered_mask.shape[2:]:
            raise ValueError(
                "Dense Plucker rays and rendered mask must share [T, H, W]. "
                f"Got Plucker {tuple(dense_plucker.shape[2:])} and mask {tuple(rendered_mask.shape[2:])}."
            )

        plucker_embedding = torch.cat([dense_plucker, rendered_mask.to(dtype=dense_plucker.dtype)], dim=1)
        if plucker_embedding.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} Plucker input channels, got {plucker_embedding.shape[1]}.")

        plucker_embedding = F.pad(plucker_embedding, self.raw_padding, mode="constant", value=0.0)
        plucker_volume = self.mask_proj(plucker_embedding)
        plucker_volume = self.mask_zero_proj(plucker_volume)
        return plucker_volume.flatten(2).transpose(1, 2).contiguous()


@torch.amp.autocast("cuda", enabled=False)
def batch_sample_rays(
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    image_h: int | None = None,
    image_w: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample camera ray origins and directions for every pixel in a batch."""

    device = intrinsic.device
    batch = intrinsic.shape[0]
    c2w = torch.inverse(extrinsic)[:, :3, :4].to(device)
    x = torch.arange(image_w, device=device).float() - 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1)
    points = einops.repeat(points, "w h c -> b (h w) c", b=batch)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic.inverse().to(device).transpose(-1, -2)
    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d)
    return rays_o, rays_d


@torch.amp.autocast("cuda", enabled=False)
def embed_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, nframe: int) -> torch.Tensor:
    """Convert ray origins and directions to six-channel Plucker embeddings."""

    if len(rays_o.shape) == 4:
        rays_o = einops.rearrange(rays_o, "b f n c -> (b f) n c")
        rays_d = einops.rearrange(rays_d, "b f n c -> (b f) n c")
    cross_od = torch.cross(rays_o, rays_d, dim=-1)
    cam_emb = torch.cat([rays_d, cross_od], dim=-1)
    return einops.rearrange(cam_emb, "(b f) n c -> b f n c", f=nframe)


@torch.amp.autocast("cuda", enabled=False)
def camera_center_normalization(w2c: torch.Tensor, nframe: int, camera_scale: float = 2.0) -> torch.Tensor:
    """Normalize camera centers relative to the first view, matching the prior helper."""

    c2w_view0 = w2c[::nframe].inverse()
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)
    w2c = c2w_view0 @ w2c
    c2w = torch.linalg.inv(w2c)
    camera_dist_2med = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    return w2c


def get_camera_embedding(
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    f: int,
    h: int,
    w: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Return dense Plucker rays shaped ``[B, 6, F, H, W]``."""

    if normalize:
        extrinsic = camera_center_normalization(extrinsic, nframe=f)
    rays_o, rays_d = batch_sample_rays(intrinsic, extrinsic, image_h=h, image_w=w)
    dense_plucker = embed_rays(rays_o, rays_d, nframe=f)
    return einops.rearrange(dense_plucker, "b f (h w) c -> b c f h w", h=h, w=w)
