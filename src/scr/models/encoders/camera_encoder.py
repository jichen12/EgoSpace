"""Uni3C-style camera-control encoder architecture.

This file intentionally contains only the neural architecture for the camera
control branch. It does not build dataloaders, choose losses, run optimizers, or
inject into Wan DiT. The training/assembly code owns those decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def zero_module(module: nn.Module) -> nn.Module:
    """Return a module whose parameters are initialized to exact zero."""

    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


@dataclass
class CameraEncoderOutput:
    """Named output tensors from the camera-control branch.

    Attributes:
        render_latent: VAE-encoded rendered point-cloud video,
            shaped ``[B, C_lat, T_lat, H_lat, W_lat]``.
        control_volume: Concatenation of the leading ``hidden_states`` motion
            channels, I2V mask channels, and ``render_latent``;
            shaped ``[B, condition_channels + C_lat, T_lat, H_lat, W_lat]``.
        patch_tokens: Patchified render/control tokens before mask/camera fusion,
            shaped ``[B, N, conv_out_dim]``.
        plucker_tokens: Embedded dense Plucker rays plus render mask, shaped
            ``[B, N, wan_hidden_dim]`` before any local projection.
        fused_tokens: Elementwise sum of ``patch_tokens`` and projected
            ``plucker_tokens``, shaped ``[B, N, conv_out_dim]``.
        residuals: One zero-projected residual tensor per external DiT block,
            each shaped ``[B, N, wan_hidden_dim]`` for direct Wan injection.
        token_grid: Token grid after Wan patching, as ``(T_patch, H_patch, W_patch)``.
    """

    render_latent: torch.Tensor
    control_volume: torch.Tensor
    patch_tokens: torch.Tensor
    plucker_tokens: torch.Tensor
    fused_tokens: torch.Tensor
    residuals: list[torch.Tensor]
    token_grid: tuple[int, int, int]


class RMSNormAcrossHeads(nn.Module):
    """Apply Uni3C's affine RMS normalization across the complete Q/K width."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Create the all-head normalization used by Diffusers' Wan attention."""

        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension in float32 and preserve the input dtype."""

        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=input_dtype) * self.weight.to(dtype=input_dtype)

class RotarySelfAttention(nn.Module):
    """Self-attention over 3D video tokens with Wan-style rotary position encoding."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope_max_seq_len: int = 1024,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
    ) -> None:
        """Create query/key/value projections and cached 1D frequency tables."""

        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Rotary attention requires an even per-head dimension.")
        self.rope_max_seq_len = int(rope_max_seq_len)
        self.q = nn.Linear(self.dim, self.dim)
        self.k = nn.Linear(self.dim, self.dim)
        self.v = nn.Linear(self.dim, self.dim)
        self.out = nn.Linear(self.dim, self.dim)
        if qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNormAcrossHeads(self.dim, eps=eps)
            self.norm_k = RMSNormAcrossHeads(self.dim, eps=eps)
        elif qk_norm is None:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            raise ValueError(f"Unsupported camera attention qk_norm={qk_norm!r}.")

    def forward(self, x: torch.Tensor, token_grid: tuple[int, int, int]) -> torch.Tensor:
        """Attend over ``[B, N, D]`` tokens while preserving the input shape."""

        batch, token_count, _ = x.shape
        if token_count != math.prod(token_grid):
            raise ValueError(f"Token count {token_count} does not match grid {token_grid}.")

        # Uni3C normalizes each complete Q/K vector before splitting it into
        # heads, matching Diffusers' ``rms_norm_across_heads`` behavior.
        q = (
            self.norm_q(self.q(x))
            .view(batch, token_count, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.norm_k(self.k(x))
            .view(batch, token_count, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = self.v(x).view(batch, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        rotary = build_3d_rotary_frequencies(
            token_grid,
            self.head_dim,
            x.device,
            max_seq_len=self.rope_max_seq_len,
        )
        q = apply_rotary(q, rotary)
        k = apply_rotary(k, rotary)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, token_count, self.dim)
        return self.out(attended)


class AdaLayerNormZero(nn.Module):
    """LayerNorm plus timestep-generated shift, scale, and residual gate."""

    def __init__(self, time_dim: int, dim: int) -> None:
        """Create the modulation projection used before attention and FFN blocks."""

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 3 * dim))

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return modulated activations and a residual gate, both Wan-token aligned."""

        shift, scale, gate = self.modulation(temb).chunk(3, dim=-1)
        x = self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return x, gate[:, None, :]


class CameraControlBlock(nn.Module):
    """One Uni3C external DiT block for camera-control residual generation."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        time_dim: int,
        rope_max_seq_len: int = 1024,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
    ) -> None:
        """Create timestep-modulated self-attention and feed-forward sublayers."""

        super().__init__()
        self.norm_attn = AdaLayerNormZero(time_dim, dim)
        self.self_attn = RotarySelfAttention(
            dim,
            num_heads,
            rope_max_seq_len=rope_max_seq_len,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm_ffn = AdaLayerNormZero(time_dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor, token_grid: tuple[int, int, int]) -> torch.Tensor:
        """Return one external DiT update for control tokens shaped ``[B, N, D]``."""

        attn_input, attn_gate = self.norm_attn(x, temb)
        x = x + attn_gate * self.self_attn(attn_input, token_grid)
        ffn_input, ffn_gate = self.norm_ffn(x, temb)
        x = x + ffn_gate * self.ffn(ffn_input)
        return x


class CameraEncoder(nn.Module):
    """Faithful Uni3C camera-control branch, renamed from ``PCDController``.

    Inputs:
        ``rendered_video``:
            Raw rendered point-cloud RGB video, ``[B, 3, T, H, W]``.
        ``rendered_mask``:
            Binary rendered point-cloud visibility mask, ``[B, 1, T, H, W]``.
        ``hidden_states``:
            CameraEncoder latent input volume before patchifying, usually
            ``concat(zt, first_frame_mask_latents, first_frame_latents)`` with
            shape ``[B, 36, T_lat, H_lat, W_lat]``. The leading
            ``condition_channels`` are consumed as zt plus the I2V mask.
        ``plucker_tokens``:
            Trainable Plucker-mask tokens from ``PluckerEncoder``, shaped
            ``[B, N_tokens, wan_hidden_dim]``.

    Output:
        ``CameraEncoderOutput.residuals`` is a list of tensors shaped
        ``[B, N_tokens, wan_hidden_dim]``. The trainer explicitly adds these to
        selected Wan DiT layers.
    """

    def __init__(
        self,
        render_vae: Any | None,
        latent_channels: int = 16,
        condition_channels: int = 20,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        conv_out_dim: int = 5120,
        control_dim: int = 1024,
        control_ffn_dim: int = 4096,
        control_num_heads: int = 16,
        num_control_layers: int = 20,
        wan_hidden_dim: int = 5120,
        time_frequency_dim: int = 256,
        time_dim: int = 5120,
        rope_max_seq_len: int = 1024,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        plucker_token_dim: int | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        """Create the render encoder, mask/camera fusion, and external DiT branch."""

        super().__init__()
        self.render_vae = render_vae
        self.latent_channels = int(latent_channels)
        self.condition_channels = int(condition_channels)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.conv_out_dim = int(conv_out_dim)
        self.control_dim = int(control_dim)
        self.num_control_layers = int(num_control_layers)
        self.wan_hidden_dim = int(wan_hidden_dim)
        self.plucker_token_dim = int(plucker_token_dim or self.wan_hidden_dim)
        # ``time_frequency_dim`` remains accepted for checkpoint/config
        # compatibility. Released Uni3C reuses the frozen Wan timestep
        # embedding rather than owning another trainable timestep MLP here.
        self.time_frequency_dim = int(time_frequency_dim)
        self.time_dim = int(time_dim)
        self.qk_norm = qk_norm
        self.eps = float(eps)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # Uni3C concatenates first-frame conditioning channels from the Wan
        # hidden-state input with the rendered-video latent before patchifying.
        self.controlnet_patch_embedding = nn.Conv3d(
            self.condition_channels + self.latent_channels,
            self.conv_out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # PluckerEncoder now owns dense Plucker + rendered-mask fusion. This
        # projection keeps the add path compatible with smaller control widths.
        self.plucker_token_projection = (
            nn.Linear(self.plucker_token_dim, self.conv_out_dim)
            if self.plucker_token_dim != self.conv_out_dim
            else nn.Identity()
        )
        self.controlnet_input_projection = (
            nn.Linear(self.conv_out_dim, self.control_dim)
            if self.conv_out_dim != self.control_dim
            else nn.Identity()
        )
        self.control_blocks = nn.ModuleList(
            [
                CameraControlBlock(
                    dim=self.control_dim,
                    ffn_dim=control_ffn_dim,
                    num_heads=control_num_heads,
                    time_dim=self.time_dim,
                    rope_max_seq_len=rope_max_seq_len,
                    qk_norm=self.qk_norm,
                    eps=self.eps,
                )
                for _ in range(self.num_control_layers)
            ]
        )
        self.residual_projections = nn.ModuleList(
            [zero_module(nn.Linear(self.control_dim, self.wan_hidden_dim)) for _ in range(self.num_control_layers)]
        )

    def forward(
        self,
        rendered_video: torch.Tensor,
        rendered_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        plucker_tokens: torch.Tensor,
        timestep: torch.Tensor | None = None,
        render_latent: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> CameraEncoderOutput:
        """Generate camera residuals using the frozen Wan timestep embedding."""

        if hidden_states.ndim != 5:
            raise ValueError(f"hidden_states must be [B, C, T_lat, H_lat, W_lat], got {tuple(hidden_states.shape)}.")
        if hidden_states.shape[1] < self.condition_channels:
            raise ValueError(
                f"hidden_states must contain at least {self.condition_channels} condition channels, got {hidden_states.shape[1]}."
            )

        # rendered_video: [B, 3, T, H, W]
        # render_latent:  [B, 16, T_lat, H_lat, W_lat]
        render_latent = self.encode_rendered_video(rendered_video, render_latent)
        if render_latent.shape[0] != hidden_states.shape[0]:
            raise ValueError("render_latent and hidden_states batch sizes must match.")
        if render_latent.shape[2:] != hidden_states.shape[2:]:
            raise ValueError(
                f"render_latent grid {tuple(render_latent.shape[2:])} must match hidden_states grid {tuple(hidden_states.shape[2:])}."
            )

        # The trainer passes CameraEncoder hidden_states as zt-first:
        # 16 noisy latent channels + 4 I2V mask channels, then first-frame latents.
        motion_and_mask_condition = hidden_states[:, : self.condition_channels]

        # control_volume: [B, 36, T_lat, H_lat, W_lat] for Wan I2V defaults.
        control_volume = torch.cat([motion_and_mask_condition, render_latent], dim=1)

        # controlnet_patch_embedding:
        # [B, 36, T_lat, H_lat, W_lat]
        # -> [B, conv_out_dim, T_lat, H_lat/2, W_lat/2]
        patch_volume = self.controlnet_patch_embedding(control_volume)
        token_grid = tuple(int(value) for value in patch_volume.shape[2:])

        # patch_tokens: [B, N, conv_out_dim], where N = T_lat * H_lat/2 * W_lat/2.
        patch_tokens = patch_volume.flatten(2).transpose(1, 2).contiguous()

        # plucker_tokens: [B, N, wan_hidden_dim], where N matches token_grid.
        if plucker_tokens.ndim != 3:
            raise ValueError(f"plucker_tokens must be [B, N, C], got {tuple(plucker_tokens.shape)}.")
        if plucker_tokens.shape[0] != patch_tokens.shape[0]:
            raise ValueError("plucker_tokens and patch_tokens batch sizes must match.")
        if plucker_tokens.shape[1] != patch_tokens.shape[1]:
            raise ValueError(
                f"Plucker token count {plucker_tokens.shape[1]} does not match control token count {patch_tokens.shape[1]}."
            )
        if plucker_tokens.shape[2] != self.plucker_token_dim:
            raise ValueError(
                f"Expected Plucker token dim {self.plucker_token_dim}, got {plucker_tokens.shape[2]}."
            )
        projected_plucker_tokens = self.plucker_token_projection(plucker_tokens.to(dtype=patch_tokens.dtype))

        # Fused external DiT input, matching Uni3C-style elementwise addition.
        fused_tokens = patch_tokens + projected_plucker_tokens
        control_tokens = self.controlnet_input_projection(fused_tokens)

        # Released Uni3C supplies the frozen Wan transformer's timestep
        # embedding to every external AdaLayerNorm. Keeping that embedding out
        # of this trainable module avoids a second, non-Uni3C timestep MLP.
        if temb is None:
            raise ValueError("CameraEncoder requires the frozen Wan timestep embedding via temb.")
        if temb.ndim != 2 or temb.shape != (control_tokens.shape[0], self.time_dim):
            raise ValueError(
                f"temb must be [B, {self.time_dim}], got {tuple(temb.shape)} for batch {control_tokens.shape[0]}."
            )

        residuals: list[torch.Tensor] = []
        x = control_tokens
        for block, projection in zip(self.control_blocks, self.residual_projections):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    lambda tokens, time_embedding, control_block=block: control_block(tokens, time_embedding, token_grid),
                    x,
                    temb,
                    use_reentrant=False,
                )
            else:
                x = block(x, temb, token_grid)
            residuals.append(projection(x))

        return CameraEncoderOutput(
            render_latent=render_latent,
            control_volume=control_volume,
            patch_tokens=patch_tokens,
            plucker_tokens=plucker_tokens,
            fused_tokens=fused_tokens,
            residuals=residuals,
            token_grid=token_grid,
        )

    def encode_rendered_video(
        self,
        rendered_video: torch.Tensor,
        render_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode rendered RGB video to Wan VAE latents shaped ``[B, C, T_lat, H_lat, W_lat]``."""

        if render_latent is not None:
            return render_latent
        if self.render_vae is None:
            raise RuntimeError("CameraEncoder requires render_vae or a precomputed render_latent.")
        if rendered_video.ndim != 5:
            raise ValueError(f"rendered_video must be [B, 3, T, H, W], got {tuple(rendered_video.shape)}.")

        # Local WanVAE expects a Python list of [C, T, H, W] videos and returns
        # a list of [C_lat, T_lat, H_lat, W_lat] latents.
        with torch.no_grad():
            if self.render_vae.__class__.__name__ == "WanVAE":
                encoded = self.render_vae.encode([sample for sample in rendered_video])
                return torch.stack(encoded, dim=0).to(device=rendered_video.device)

            # Diffusers-style VAEs typically accept batched tensors and return
            # an object with ``latent_dist`` or ``latents``.
            encoded = self.render_vae.encode(rendered_video)
            if hasattr(encoded, "latent_dist"):
                encoded = encoded.latent_dist.mode()
            elif hasattr(encoded, "latents"):
                encoded = encoded.latents
            if isinstance(encoded, (list, tuple)):
                encoded = torch.stack(list(encoded), dim=0)
            return encoded.to(device=rendered_video.device)


def build_3d_rotary_frequencies(
    token_grid: tuple[int, int, int],
    head_dim: int,
    device: torch.device,
    max_seq_len: int = 1024,
) -> torch.Tensor:
    """Build Wan-compatible complex RoPE frequencies for a ``[T, H, W]`` grid."""

    frames, height, width = token_grid
    if max(token_grid) > max_seq_len:
        raise ValueError(
            f"Camera token grid {token_grid} exceeds rope_max_seq_len={max_seq_len}."
        )
    complex_dim = head_dim // 2
    frame_dim = complex_dim - 2 * (complex_dim // 3)
    height_dim = complex_dim // 3
    width_dim = complex_dim // 3

    freqs_f = (
        rope_1d(frames, frame_dim, device)
        .view(frames, 1, 1, frame_dim)
        .expand(frames, height, width, frame_dim)
    )
    freqs_h = (
        rope_1d(height, height_dim, device)
        .view(1, height, 1, height_dim)
        .expand(frames, height, width, height_dim)
    )
    freqs_w = (
        rope_1d(width, width_dim, device)
        .view(1, 1, width, width_dim)
        .expand(frames, height, width, width_dim)
    )
    return torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, frames * height * width, complex_dim)


def rope_1d(length: int, complex_dim: int, device: torch.device) -> torch.Tensor:
    """Return the float64 complex frequencies used by Wan rotary embedding."""

    if complex_dim <= 0:
        return torch.empty(length, 0, device=device, dtype=torch.complex128)
    positions = torch.arange(length, device=device, dtype=torch.float64)
    indices = torch.arange(complex_dim, device=device, dtype=torch.float64)
    theta = torch.pow(torch.tensor(10000.0, device=device, dtype=torch.float64), -indices / complex_dim)
    angles = torch.outer(positions, theta)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply Wan's float64 complex rotation to ``[B, H, N, D]`` attention tensors."""

    dtype = x.dtype
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.to(dtype=dtype)
