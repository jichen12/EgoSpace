"""Train Stand-In object conditioning on a frozen native Wan 2.1 I2V DiT.

Reference-image latents use Wan's existing patch embedding, a fixed zero
diffusion timestep, Restricted Self-Attention (RSA), and Conditional Position
Mapping (CPM). Only rank-128 image-branch QKV LoRA matrices are trainable; the
video QKV path and every original Wan parameter remain frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import types
from typing import Any, Iterable, Sequence
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


STANDIN_ROPE_MODES = (
    "tss-rope",
    "vanilla-rope",
    "ts-rope",
    "ss-rope",
)


@dataclass(frozen=True)
class StandInLoRAConfig:
    """Configuration for image-only QKV LoRA adapters in Wan DiT blocks."""

    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    train_attn_blocks: tuple[int, ...] | None = None
    dropout: float = 0.0
    init: str = "zero"


@dataclass
class StandInForwardOutput:
    """Structured tensors returned by one Stand-In training forward pass."""

    prediction: torch.Tensor
    target: torch.Tensor
    loss: torch.Tensor
    diffusion_loss: torch.Tensor
    z0: torch.Tensor
    zt_video: torch.Tensor
    reference_latents: torch.Tensor
    model_latents: torch.Tensor
    i2v_mask: torch.Tensor
    clean_condition_latents: torch.Tensor
    i2v_condition: torch.Tensor
    timesteps: torch.Tensor
    reference_timesteps: torch.Tensor
    video_patch_grid: tuple[int, int, int]
    image_patch_grid: tuple[int, int, int]
    token_seq_len: int
    injected_lora_modules: list[str]


@dataclass(frozen=True)
class StandInImageRopeCoordinates:
    """Describe the temporal positions and spatial offsets for object tokens."""

    temporal_positions: torch.Tensor
    height_offset: int
    width_offset: int


class ImageBranchLoRA(nn.Module):
    """Compute a trainable low-rank residual for image tokens only."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        device: torch.device,
        dropout: float = 0.0,
        init: str = "zero",
        trainable: bool = True,
    ) -> None:
        """Create a residual adapter without owning or altering the base QKV layer."""

        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.lora_down = nn.Linear(int(in_features), self.rank, bias=False, device=device, dtype=torch.float32)
        self.lora_up = nn.Linear(self.rank, int(out_features), bias=False, device=device, dtype=torch.float32)
        self.reset_lora_parameters(init)
        self.set_lora_trainable(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the LoRA residual, cast back to the input dtype."""

        lora_dtype = self.lora_down.weight.dtype
        lora_in = self.dropout(x).to(dtype=lora_dtype)
        lora_out = self.lora_up(self.lora_down(lora_in)) * self.scale
        return lora_out.to(device=x.device, dtype=x.dtype)

    def reset_lora_parameters(self, init: str) -> None:
        """Initialize LoRA weights with a standard identity-preserving default."""

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        normalized = str(init).lower()
        if normalized in {"zero", "zeros", "identity_preserving"}:
            nn.init.zeros_(self.lora_up.weight)
        elif normalized in {"gaussian", "normal"}:
            nn.init.normal_(self.lora_up.weight, mean=0.0, std=1.0e-4)
        elif normalized in {"kaiming", "kaiming_uniform"}:
            nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"Unsupported LoRA init={init!r}; use zero, gaussian, or kaiming.")

    def set_lora_trainable(self, trainable: bool) -> None:
        """Toggle the two low-rank matrices together."""
        self.lora_down.requires_grad_(bool(trainable))
        self.lora_up.requires_grad_(bool(trainable))


class WanObjectStandInModel(nn.Module):
    """Assemble frozen Wan 2.1 I2V modules with a Stand-In image branch.

    Architecture:
        1. The frozen Wan VAE encodes the ground-truth RGB video to
           ``z0_video: [B, C, T_lat, H_lat, W_lat]``.
        2. The same frozen VAE encodes each canonical object view as a one-frame
           video, then concatenates those latents on the temporal axis:
           ``reference_latents: [B, C, R, H_lat, W_lat]``.
        3. Wan flow-matching noise is applied only to video frames:
           ``zt_video = (1 - sigma) * z0_video + sigma * noise``.
        4. Video and canonical views each form their own 36-channel Wan I2V
           input. Canonical views use zero noisy-latent channels, an all-one
           mask, their clean VAE latents, and a fixed timestep of zero.
        5. Wan's shared patch embedding tokenizes the video and reference inputs
           independently. Their resulting ``[B, N, D]`` sequences are then
           concatenated along the token axis.
        6. CPM leaves video coordinates unchanged and maps every reference
           token to temporal coordinate -1 and a disjoint spatial grid.
        7. RSA lets video queries read both branches while reference queries
           can read only reference keys and values.

    Frozen modules:
        - Wan VAE, including its encoder and decoder weights.
        - Wan DiT base weights, including self-attention base projections.
        - Optional Wan T5 text encoder.

    Trainable modules:
        - Rank-128 LoRA residuals applied to image-token Q, K, and V only.
    """

    def __init__(
        self,
        wan_checkpoint_dir: str | Path = "wan2_1/I2V-14B-720P",
        wan_upstream_path: str | Path = "wan2_1",
        device: torch.device | str = "cuda",
        num_train_timesteps: int = 1000,
        vae_checkpoint: str = "Wan2.1_VAE.pth",
        load_text_encoder: bool = True,
        load_image_encoder: bool = True,
        clip_checkpoint: str = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        clip_tokenizer: str = "xlm-roberta-large",
        transformer_dtype: torch.dtype | None = torch.bfloat16,
        gradient_checkpointing: bool = True,
        lora_rank: int = 128,
        lora_alpha: float = 128.0,
        lora_target_modules: Sequence[str] = ("self_attn.q", "self_attn.k", "self_attn.v"),
        train_attn_blocks: Sequence[int] | None = None,
        lora_dropout: float = 0.0,
        lora_init: str = "zero",
        expected_attn_blocks: int | None = 40,
    ) -> None:
        """Load frozen Wan modules and attach image-only QKV LoRA adapters."""

        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.wan_checkpoint_dir = Path(wan_checkpoint_dir)
        self.wan_upstream_path = Path(wan_upstream_path)
        self.num_train_timesteps = int(num_train_timesteps)
        self.vae_checkpoint = str(vae_checkpoint)
        self.load_text_encoder = bool(load_text_encoder)
        self.load_image_encoder = bool(load_image_encoder)
        self.clip_checkpoint = str(clip_checkpoint)
        self.clip_tokenizer = str(clip_tokenizer)
        self.transformer_dtype = transformer_dtype if self.device.type == "cuda" else None
        self.gradient_checkpointing = bool(gradient_checkpointing)
        # OOM fix: Track whether frozen encoder weights currently live on CPU between forward and backward.
        self._frozen_encoders_offloaded = False
        self.vae_stride = (4, 8, 8)
        self.sp_size = 1

        required_targets = ("self_attn.q", "self_attn.k", "self_attn.v")
        configured_targets = tuple(str(name) for name in lora_target_modules)
        if int(lora_rank) != 128:
            raise ValueError(f"Stand-In requires image-branch QKV LoRA rank 128, got {lora_rank}.")
        if configured_targets != required_targets:
            raise ValueError(
                "Stand-In LoRA must target image-token QKV only; expected "
                f"{required_targets}, got {configured_targets}."
            )
        self.lora_config = StandInLoRAConfig(
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            target_modules=configured_targets,
            train_attn_blocks=tuple(int(idx) for idx in train_attn_blocks) if train_attn_blocks is not None else None,
            dropout=float(lora_dropout),
            init=str(lora_init),
        )

        wan_model_module, WanModel, WanVAE = import_wan21_modules(self.wan_upstream_path)
        self._wan_model_module = wan_model_module
        self.wan_dit = WanModel.from_pretrained(str(self.wan_checkpoint_dir))
        self._validate_wan21_i2v_model()
        self.patch_size = tuple(int(value) for value in self.wan_dit.patch_size)
        self.loaded_dit_checkpoint = str(self.wan_checkpoint_dir)
        if self.transformer_dtype is None:
            torch.nn.Module.to(self.wan_dit, device=self.device)
        else:
            torch.nn.Module.to(self.wan_dit, device=self.device, dtype=self.transformer_dtype)
        self.wan_dit.eval().requires_grad_(False)

        self.vae = WanVAE(vae_pth=str(self.wan_checkpoint_dir / self.vae_checkpoint), device=self.device)
        self.vae.model.eval().requires_grad_(False)
        if not self.load_text_encoder:
            warn_condition_disabled(
                "T5 encoder is disabled by configuration; text_context must be supplied or text conditioning will be zero."
            )
        if not self.load_image_encoder:
            warn_condition_disabled(
                "CLIP encoder is disabled by configuration; image conditioning will be zero."
            )
        self.text_encoder = self._load_text_encoder() if self.load_text_encoder else None
        self.image_encoder = self._load_image_encoder() if self.load_image_encoder else None
        self._freeze_optional_encoder(self.text_encoder)
        self._freeze_optional_encoder(self.image_encoder)

        self._install_standin_self_attention()

        self.injected_lora_modules = self._inject_image_qkv_lora(expected_attn_blocks=expected_attn_blocks)
        self._set_trainable_modules(training=True)
        self._assert_only_lora_trainable()

    def _validate_wan21_i2v_model(self) -> None:
        """Require the native Wan 2.1 I2V architecture used by the checkpoint."""

        model_type = str(getattr(self.wan_dit, "model_type", ""))
        in_dim = int(getattr(self.wan_dit, "in_dim", 0))
        out_dim = int(getattr(self.wan_dit, "out_dim", 0))
        if model_type != "i2v" or in_dim != 36 or out_dim != 16:
            raise ValueError(
                "Stand-In training requires a native Wan 2.1 I2V DiT "
                f"with model_type='i2v', in_dim=36, and out_dim=16; found "
                f"model_type={model_type!r}, in_dim={in_dim}, out_dim={out_dim}."
            )

    def _install_standin_self_attention(self) -> None:
        """Attach image-only LoRA, RSA, and CPM attention to native Wan blocks."""

        for block in self.wan_dit.blocks:
            # The standalone Stand-In trainer retains its original TSS-RoPE
            # coordinates. Integrated training can select an ablation mode.
            block.self_attn._standin_rope = "tss-rope"
            block.self_attn.forward = types.MethodType(wan21_self_attention_forward_standin, block.self_attn)

    def forward(
        self,
        video: torch.Tensor,
        reference_images: torch.Tensor,
        text: str | Sequence[str] | None = None,
        text_context: Sequence[torch.Tensor] | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> StandInForwardOutput:
        """Run Wan flow matching with a clean, fixed-timestep reference branch.

        Args:
            video: RGB video as ``[B, T, 3, H, W]`` or ``[B, 3, T, H, W]`` in
                ``[0, 1]`` or ``[-1, 1]``.
            reference_images: Canonical object views as ``[B, R, 3, H, W]`` or
                ``[B, R, H, W, 3]``.  These are clean conditioning frames.
            text: Optional fixed prompt string or prompt batch.
            text_context: Optional precomputed Wan T5 contexts, one ``[L, D]``
                tensor per batch item.
            timesteps: Optional Wan training timesteps, ``[B]``.
            noise: Optional video-latent noise matching ``z0_video``.
        """

        # OOM fix: Restore frozen runtime encoders only when a preceding training forward offloaded them.
        self.restore_frozen_encoders()

        # video_bcthw: [B, 3, T, H, W] in Wan VAE's [-1, 1] image range.
        video_bcthw = normalize_video_for_vae(video).to(self.device)
        # OOM fix: The normalized tensor is now independent, so this local raw-input reference can be dropped.
        del video

        # z0_video path: video_bcthw is [B, 3, T, H, W]; each sample passed to
        # WanVAE.encode is [3, T, H, W], returns [16, T_lat, H_lat, W_lat], and
        # torch.stack restores the batch as [B, 16, T_lat, H_lat, W_lat].
        # Wan VAE weights are frozen and the no-grad guard enforces the
        # non-negotiable VAE constraint.
        with torch.no_grad():
            z0_video = torch.stack(self.vae.encode([sample for sample in video_bcthw]), dim=0).to(self.device)

        batch, _, video_latent_frames, latent_height, latent_width = z0_video.shape

        # first_frame_video: [B, 3, 1, H, W] in [-1, 1]. It is the only image
        # used by Wan's original CLIP cross-attention path.
        first_frame_video = video_bcthw[:, :, :1].contiguous()
        # OOM fix: Save only shape metadata needed later, then release the full normalized RGB video.
        video_frame_count = int(video_bcthw.shape[2])
        video_spatial_size = tuple(video_bcthw.shape[-2:])
        del video_bcthw

        # reference_images_brcHW starts as [B, R, 3, H, W] in [-1, 1], then is
        # resized to video [H, W] so its VAE latents share [H_lat, W_lat] with
        # z0_video.
        reference_images_brcHW = normalize_reference_images(reference_images).to(self.device)
        # OOM fix: The normalized canonical views no longer need the original raw reference tensor.
        del reference_images
        reference_images_brcHW = resize_reference_images(reference_images_brcHW, size=video_spatial_size)
        # This is where reference images pass through the VAE. Inside
        # _encode_reference_latents, each canonical image [3, H, W] is
        # unsqueezed to a one-frame video [3, 1, H, W], WanVAE.encode returns
        # [16, 1, H_lat, W_lat], the R views are concatenated on latent time to
        # [16, R, H_lat, W_lat], and the batch stack returns
        # reference_latents: [B, 16, R, H_lat, W_lat].
        with torch.no_grad():
            reference_latents = self._encode_reference_latents(reference_images_brcHW).to(
                device=self.device,
                dtype=z0_video.dtype,
            )
        # OOM fix: Canonical RGB tensors are unused after their frozen VAE latents have been produced.
        del reference_images_brcHW

        if reference_latents.shape[-2:] != z0_video.shape[-2:]:
            # Spatial-only latent resize keeps reference_latents as
            # [B, 16, R, H_lat, W_lat]; it does not change channels or the R
            # reference-frame axis.
            reference_latents = resize_latent_spatial(reference_latents, z0_video.shape[-2:])

        # Encode the complete normalized-domain [I, 0, ..., 0] sequence. The
        # causal VAE must retain its native 1,4,4,... temporal cache behavior.
        # Outputs are video_mask: [B, 4, T_lat, H_lat, W_lat] and
        # clean_video_latents: [B, 16, T_lat, H_lat, W_lat].
        with torch.no_grad():
            video_mask, clean_video_latents = self._encode_first_frame_i2v_condition(
                first_frame_video=first_frame_video,
                frame_count=video_frame_count,
                expected_latent_shape=tuple(z0_video.shape),
            )

        # Wan 2.1 flow-matching training target.  The trainable LoRA predicts
        # the velocity/noise residual for video frames; reference frames remain
        # clean conditioning frames and are excluded from the MSE.
        if timesteps is None:
            timesteps = torch.randint(0, self.num_train_timesteps, (batch,), device=self.device, dtype=torch.long)
        else:
            timesteps = timesteps.to(device=self.device, dtype=torch.long)
        # Stand-In keeps the reference identity branch clean at every denoising step.
        reference_timesteps = torch.zeros_like(timesteps)
        if noise is None:
            noise = torch.randn_like(z0_video)
        else:
            noise = noise.to(device=self.device, dtype=z0_video.dtype)
        sigma = timesteps.float().view(batch, 1, 1, 1, 1) / float(self.num_train_timesteps)
        # zt_video and target stay on the video timeline only:
        # zt_video: [B, 16, T_lat, H_lat, W_lat],
        # target: [B, 16, T_lat, H_lat, W_lat].
        zt_video = (1.0 - sigma) * z0_video + sigma * noise
        target = noise - z0_video
        # OOM fix: Noise and sigma are no longer needed after the detached flow-matching tensors are built.
        del noise, sigma

        # Build extended diagnostic tensors used by the public forward output
        # and inference cache. Canonical slots are zero in x_t, one in the
        # mask, and populated only in the clean condition. These branches stay
        # separate here; _wan_forward_wan21_i2v splits them back at
        # video_latent_frames before constructing either 36-channel Wan input.
        # model_latents = cat([zt_video, zeros_like_ref], dim=2)
        #     [B, 16, T_lat, H_lat, W_lat] + [B, 16, R, H_lat, W_lat]
        #     -> [B, 16, T_lat + R, H_lat, W_lat]
        # clean_condition_latents = cat([clean_video_latents, reference_latents], dim=2)
        #     [B, 16, T_lat, H_lat, W_lat] + [B, 16, R, H_lat, W_lat]
        #     -> [B, 16, T_lat + R, H_lat, W_lat]
        # i2v_mask = cat([video_mask, ones_for_ref], dim=2)
        #     [B, 4, T_lat, H_lat, W_lat] + [B, 4, R, H_lat, W_lat]
        #     -> [B, 4, T_lat + R, H_lat, W_lat]
        # i2v_condition = cat([i2v_mask, clean_condition_latents], dim=1)
        #     [B, 4, T_lat + R, H_lat, W_lat] + [B, 16, T_lat + R, H_lat, W_lat]
        #     -> [B, 20, T_lat + R, H_lat, W_lat].
        model_latents, i2v_mask, clean_condition_latents, i2v_condition = self._build_i2v_condition(
            zt_video=zt_video,
            video_mask=video_mask,
            clean_video_latents=clean_video_latents,
            reference_latents=reference_latents,
        )
        # OOM fix: The extended condition tensors own their data, so release their smaller source tensors.
        del video_mask, clean_video_latents
        video_patch_grid = expected_wan_patch_grid(zt_video, self.patch_size)
        image_patch_grid = expected_wan_patch_grid(reference_latents, self.patch_size)
        token_seq_len = math.prod(video_patch_grid) + math.prod(image_patch_grid)

        contexts = self._resolve_text_context(batch=batch, dtype=z0_video.dtype, text=text, text_context=text_context)
        clip_context = self._resolve_clip_context(first_frame_video)
        # OOM fix: Caption and first-frame inputs are unused once their frozen encoder contexts exist.
        del text, text_context, first_frame_video
        # The DiT path creates separate 36-channel video and reference inputs,
        # applies the same Wan patch embedding to each, and concatenates only
        # their flattened token sequences:
        # [B, N_video, 5120] + [B, N_reference, 5120]
        # -> [B, N_video + N_reference, 5120].
        prediction = self._wan_forward_wan21_i2v(
            model_latents=model_latents,
            i2v_condition=i2v_condition,
            video_latent_frames=video_latent_frames,
            timesteps=timesteps,
            reference_timesteps=reference_timesteps,
            text_context=contexts,
            clip_context=clip_context,
            token_seq_len=token_seq_len,
        )
        # OOM fix: Python references to encoder contexts can be dropped; autograd retains only tensors required by DiT.
        del contexts, clip_context
        diffusion_loss = F.mse_loss(prediction, target)
        return StandInForwardOutput(
            prediction=prediction,
            target=target,
            loss=diffusion_loss,
            diffusion_loss=diffusion_loss,
            z0=z0_video,
            zt_video=zt_video,
            reference_latents=reference_latents,
            model_latents=model_latents,
            i2v_mask=i2v_mask,
            clean_condition_latents=clean_condition_latents,
            i2v_condition=i2v_condition,
            timesteps=timesteps,
            reference_timesteps=reference_timesteps,
            video_patch_grid=video_patch_grid,
            image_patch_grid=image_patch_grid,
            token_seq_len=token_seq_len,
            injected_lora_modules=list(self.injected_lora_modules),
        )

    def offload_frozen_encoders(self) -> None:
        """Move frozen VAE, T5, and CLIP weights to CPU until the next forward."""

        # OOM fix: Encoder weights have no backward graph because all encoder calls run under no_grad.
        if self.device.type != "cuda" or self._frozen_encoders_offloaded:
            return
        cpu_device = torch.device("cpu")
        self.vae.model.to(cpu_device)
        self.vae.mean = self.vae.mean.to(cpu_device)
        self.vae.std = self.vae.std.to(cpu_device)
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        self.vae.device = cpu_device
        if self.text_encoder is not None:
            self.text_encoder.model.to(cpu_device)
            self.text_encoder.device = cpu_device
        if self.image_encoder is not None:
            self.image_encoder.model.to(cpu_device)
            self.image_encoder.device = cpu_device
        self._frozen_encoders_offloaded = True

    def restore_frozen_encoders(self) -> None:
        """Restore frozen VAE, T5, and CLIP weights before runtime encoding."""

        # OOM fix: Runtime encoding remains unchanged; weights return to their original CUDA rank only when needed.
        if not self._frozen_encoders_offloaded:
            return
        self.vae.model.to(self.device)
        self.vae.mean = self.vae.mean.to(self.device)
        self.vae.std = self.vae.std.to(self.device)
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        self.vae.device = self.device
        if self.text_encoder is not None:
            self.text_encoder.model.to(self.device)
            self.text_encoder.device = self.device
        if self.image_encoder is not None:
            self.image_encoder.model.to(self.device)
            self.image_encoder.device = self.device
        self._frozen_encoders_offloaded = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only LoRA parameters with ``requires_grad=True``."""

        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_modules(self) -> dict[str, nn.Module]:
        """Return LoRA modules for logging, inspection, and WandB watching."""

        return {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, ImageBranchLoRA) and any(parameter.requires_grad for parameter in module.parameters())
        }

    def frozen_modules(self) -> dict[str, nn.Module | Any | None]:
        """Return frozen Wan modules for explicit status checks."""

        return {
            "wan_vae": self.vae.model,
            "wan_dit": self.wan_dit,
            "wan_text_encoder": self.text_encoder,
        }

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a checkpoint payload containing LoRA tensors only."""

        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if is_lora_parameter_name(name)
        }

    def load_lora_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        """Load LoRA-only checkpoint tensors into the injected DiT modules."""

        lora_keys = {name for name, _ in self.named_parameters() if is_lora_parameter_name(name)}
        incoming = {name: tensor for name, tensor in state_dict.items() if is_lora_parameter_name(name)}
        missing = sorted(lora_keys.difference(incoming))
        unexpected = sorted(set(incoming).difference(lora_keys))
        if strict and (missing or unexpected):
            raise RuntimeError(f"LoRA checkpoint mismatch: missing={missing}, unexpected={unexpected}.")
        self.load_state_dict(incoming, strict=False)

    def _encode_reference_latents(self, reference_images: torch.Tensor) -> torch.Tensor:
        """Encode canonical views as clean one-frame VAE latents and concatenate."""

        if reference_images.ndim != 5:
            raise ValueError(f"reference_images must be [B, R, 3, H, W], got {tuple(reference_images.shape)}.")
        latents_by_sample = []
        for sample in reference_images:
            view_latents = []
            for ref_image in sample:
                # WanVAE.encode expects a list of [C, T, H, W] videos.  Each
                # canonical image is treated as a one-frame video, producing
                # [16, 1, H_lat, W_lat].
                view_latents.append(self.vae.encode([ref_image.unsqueeze(1)])[0])
            latents_by_sample.append(torch.cat(view_latents, dim=1))
        return torch.stack(latents_by_sample, dim=0)

    def _encode_first_frame_i2v_condition(
        self,
        first_frame_video: torch.Tensor,
        frame_count: int,
        expected_latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode Wan's native first-frame mask and clean video condition."""

        assert first_frame_video.ndim == 5 and tuple(first_frame_video.shape[1:3]) == (3, 1), (
            f"first_frame_video must be [B,3,1,H,W], got {tuple(first_frame_video.shape)}."
        )
        batch, _, _, height, width = first_frame_video.shape
        assert frame_count > 0 and (int(frame_count) - 1) % int(self.vae_stride[0]) == 0, (
            f"Wan I2V requires F=1+k*{self.vae_stride[0]}, got F={frame_count}."
        )

        condition_video = first_frame_video.new_zeros(
            int(batch),
            3,
            int(frame_count),
            int(height),
            int(width),
        )
        condition_video[:, :, :1] = first_frame_video
        clean_video_latents = torch.stack(
            self.vae.encode([sample for sample in condition_video]),
            dim=0,
        ).to(device=first_frame_video.device)
        assert tuple(clean_video_latents.shape) == tuple(expected_latent_shape), (
            f"Wan clean condition shape {tuple(clean_video_latents.shape)} does not match video latent "
            f"shape {tuple(expected_latent_shape)}."
        )
        clean_video_latents = clean_video_latents.to(dtype=first_frame_video.dtype)
        mask = build_wan_i2v_first_frame_mask(
            batch=int(batch),
            frame_count=int(frame_count),
            latent_frames=int(clean_video_latents.shape[2]),
            latent_height=int(clean_video_latents.shape[3]),
            latent_width=int(clean_video_latents.shape[4]),
            device=clean_video_latents.device,
            dtype=clean_video_latents.dtype,
        )
        return mask, clean_video_latents

    def _build_i2v_condition(
        self,
        zt_video: torch.Tensor,
        video_mask: torch.Tensor,
        clean_video_latents: torch.Tensor,
        reference_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extend x_t, mask, and z_clean independently with canonical slots."""

        return extend_wan_i2v_with_canonical_latents(
            zt_video=zt_video,
            video_mask=video_mask,
            clean_video_latents=clean_video_latents,
            reference_latents=reference_latents,
        )

    def _wan_forward_wan21_i2v(
        self,
        model_latents: torch.Tensor,
        i2v_condition: torch.Tensor,
        video_latent_frames: int,
        timesteps: torch.Tensor,
        reference_timesteps: torch.Tensor,
        text_context: Sequence[torch.Tensor],
        clip_context: torch.Tensor,
        token_seq_len: int,
    ) -> torch.Tensor:
        """Run fixed-timestep image tokens through CPM and RSA Wan blocks."""

        assert_wan_i2v_branch_compatibility(model_latents, i2v_condition)
        if tuple(reference_timesteps.shape) != tuple(timesteps.shape):
            raise ValueError(
                f"Reference timesteps must match video timesteps {tuple(timesteps.shape)}, "
                f"got {tuple(reference_timesteps.shape)}."
            )
        if torch.count_nonzero(reference_timesteps).item() != 0:
            raise ValueError("Stand-In reference timesteps must remain fixed at zero.")
        total_latent_frames = int(model_latents.shape[2])
        video_latent_frames = int(video_latent_frames)
        if not 0 < video_latent_frames < total_latent_frames:
            raise ValueError(
                "video_latent_frames must split a non-empty video and reference timeline; "
                f"got video={video_latent_frames}, total={total_latent_frames}."
            )

        dit_dtype = self.wan_dit.patch_embedding.weight.dtype
        video_wan_inputs = torch.cat(
            [
                model_latents[:, :, :video_latent_frames].to(dtype=dit_dtype),
                i2v_condition[:, :, :video_latent_frames].to(dtype=dit_dtype),
            ],
            dim=1,
        )
        reference_wan_inputs = torch.cat(
            [
                model_latents[:, :, video_latent_frames:].to(dtype=dit_dtype),
                i2v_condition[:, :, video_latent_frames:].to(dtype=dit_dtype),
            ],
            dim=1,
        )
        if (
            int(video_wan_inputs.shape[1]) != int(self.wan_dit.in_dim)
            or int(reference_wan_inputs.shape[1]) != int(self.wan_dit.in_dim)
        ):
            raise ValueError(
                f"Wan 2.1 I2V patch embedding expects {self.wan_dit.in_dim} channels, "
                f"got video={video_wan_inputs.shape[1]} and reference={reference_wan_inputs.shape[1]}."
            )

        tokens, video_patch_grid, image_patch_grid = patchify_wan_video_and_image_inputs(
            patch_embedding=self.wan_dit.patch_embedding,
            video_inputs=video_wan_inputs,
            image_inputs=reference_wan_inputs,
            patch_size=self.patch_size,
            model_dim=int(self.wan_dit.dim),
        )
        video_token_length = math.prod(video_patch_grid)
        image_token_length = math.prod(image_patch_grid)
        expected_token_length = video_token_length + image_token_length
        if int(token_seq_len) != expected_token_length:
            raise ValueError(
                f"Configured token length {token_seq_len} does not match CPM branches {expected_token_length}."
            )
        grid_sizes = torch.tensor([video_patch_grid] * len(tokens), dtype=torch.long, device=self.device)
        seq_lens = torch.tensor([sample.size(1) for sample in tokens], dtype=torch.long, device=self.device)
        if int(seq_lens.max().item()) > int(token_seq_len):
            raise ValueError(f"Wan token count {int(seq_lens.max().item())} exceeds seq_len={token_seq_len}.")
        hidden = torch.cat(
            [
                torch.cat([sample, sample.new_zeros(1, int(token_seq_len) - sample.size(1), sample.size(2))], dim=1)
                for sample in tokens
            ],
            dim=0,
        )

        with self._wan_model_module.amp.autocast(dtype=torch.float32):
            timestep_embedding = self.wan_dit.time_embedding(
                self._wan_model_module.sinusoidal_embedding_1d(self.wan_dit.freq_dim, timesteps).float()
            )
            timestep_projection = self.wan_dit.time_projection(timestep_embedding).unflatten(1, (6, self.wan_dit.dim))
            reference_timestep_embedding = self.wan_dit.time_embedding(
                self._wan_model_module.sinusoidal_embedding_1d(self.wan_dit.freq_dim, reference_timesteps).float()
            )
            reference_timestep_projection = self.wan_dit.time_projection(
                reference_timestep_embedding
            ).unflatten(1, (6, self.wan_dit.dim))

        padded_text = []
        for context in text_context:
            context = context[: self.wan_dit.text_len]
            padded_text.append(
                torch.cat([context, context.new_zeros(self.wan_dit.text_len - context.size(0), context.size(1))])
            )
        text_dtype = next(self.wan_dit.text_embedding.parameters()).dtype
        text_tokens = self.wan_dit.text_embedding(torch.stack(padded_text).to(dtype=text_dtype))
        image_dtype = next(self.wan_dit.img_emb.parameters()).dtype
        image_tokens = self.wan_dit.img_emb(clip_context.to(device=self.device, dtype=image_dtype))
        cross_attention_context = torch.cat([image_tokens, text_tokens], dim=1)

        if self.wan_dit.freqs.device != self.device:
            self.wan_dit.freqs = self.wan_dit.freqs.to(self.device)
        for block in self.wan_dit.blocks:
            block.self_attn._standin_video_grid = video_patch_grid
            block.self_attn._standin_image_grid = image_patch_grid
            block.self_attn._standin_video_token_length = video_token_length
            block.self_attn._standin_image_token_length = image_token_length
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    lambda value, wan_block=block: wan21_dit_block_forward_standin(
                        wan_block,
                        value,
                        timestep_projection=timestep_projection,
                        reference_timestep_projection=reference_timestep_projection,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.wan_dit.freqs,
                        context=cross_attention_context,
                        context_lens=None,
                        video_token_length=video_token_length,
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = wan21_dit_block_forward_standin(
                    block,
                    hidden,
                    timestep_projection=timestep_projection,
                    reference_timestep_projection=reference_timestep_projection,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=self.wan_dit.freqs,
                    context=cross_attention_context,
                    context_lens=None,
                    video_token_length=video_token_length,
                )

        hidden = hidden[:, :video_token_length]
        hidden = self.wan_dit.head(hidden, timestep_embedding)
        return torch.stack(self.wan_dit.unpatchify(hidden, grid_sizes), dim=0)

    def _resolve_clip_context(self, first_frame_video: torch.Tensor) -> torch.Tensor:
        """Encode only the target video first frame through native Wan CLIP."""

        assert first_frame_video.ndim == 5 and tuple(first_frame_video.shape[1:3]) == (3, 1), (
            f"first_frame_video must be [B,3,1,H,W], got {tuple(first_frame_video.shape)}."
        )
        batch = int(first_frame_video.shape[0])
        image_dtype = next(self.wan_dit.img_emb.parameters()).dtype
        if self.image_encoder is None:
            warn_condition_disabled(
                "CLIP encoder is empty or uninitialized; image condition was not encoded and zero tokens are being used."
            )
            return torch.zeros(batch, 257, 1280, device=self.device, dtype=image_dtype)
        videos = [sample.contiguous() for sample in first_frame_video]
        try:
            with torch.no_grad():
                context = self.image_encoder.visual(videos)
        except Exception as exc:
            warn_condition_disabled(f"CLIP image encoding failed: {type(exc).__name__}: {exc}")
            raise RuntimeError("Wan CLIP image condition could not be injected.") from exc
        if context is None or context.numel() == 0:
            warn_condition_disabled("CLIP returned an empty condition, so image cross-attention cannot be injected.")
            raise RuntimeError("Wan CLIP returned an empty image condition.")
        assert tuple(context.shape[:2]) == (batch, 257) and int(context.shape[-1]) == 1280, (
            f"Wan CLIP context must be [B,257,1280], got {tuple(context.shape)}."
        )
        return context.to(device=self.device, dtype=image_dtype)

    def _resolve_text_context(
        self,
        batch: int,
        dtype: torch.dtype,
        text: str | Sequence[str] | None,
        text_context: Sequence[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        """Return Wan T5 context tensors, or zero context when T5 is disabled."""

        if text_context is not None:
            if len(text_context) != batch:
                raise ValueError(f"text_context length {len(text_context)} does not match batch {batch}.")
            if any(context is None or context.numel() == 0 for context in text_context):
                warn_condition_disabled("A supplied T5 text_context is empty; text condition cannot be injected.")
                raise ValueError("text_context entries must be non-empty tensors.")
            return [context.to(device=self.device, dtype=dtype) for context in text_context]
        prompts = normalize_prompt_batch(text, batch)
        if self.text_encoder is not None:
            try:
                with torch.no_grad():
                    contexts = self.text_encoder(prompts, self.device)
            except Exception as exc:
                warn_condition_disabled(f"T5 text encoding failed: {type(exc).__name__}: {exc}")
                raise RuntimeError("Wan T5 text condition could not be injected.") from exc
            if not contexts or any(context is None or context.numel() == 0 for context in contexts):
                warn_condition_disabled("T5 returned an empty condition, so text cross-attention cannot be injected.")
                raise RuntimeError("Wan T5 returned an empty text condition.")
            return [context.to(device=self.device, dtype=dtype) for context in contexts]
        warn_condition_disabled(
            "T5 encoder is empty or uninitialized; text condition was not encoded and zero tokens are being used."
        )
        return [torch.zeros(1, self.wan_dit.text_dim, device=self.device, dtype=dtype) for _ in range(batch)]

    def _load_text_encoder(self) -> Any | None:
        """Load the frozen Wan 2.1 T5 encoder when checkpoint files exist."""

        from wan.modules.t5 import T5EncoderModel

        checkpoint = self.wan_checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer = self.wan_checkpoint_dir / "google/umt5-xxl"
        if not checkpoint.exists() or not tokenizer.exists():
            warn_condition_disabled(
                f"T5 checkpoint or tokenizer is missing: checkpoint={checkpoint}, tokenizer={tokenizer}."
            )
            return None
        try:
            return T5EncoderModel(
                text_len=self.wan_dit.text_len,
                dtype=torch.bfloat16,
                device=self.device,
                checkpoint_path=str(checkpoint),
                tokenizer_path=str(tokenizer),
            )
        except Exception as exc:
            warn_condition_disabled(f"T5 checkpoint loading failed: {type(exc).__name__}: {exc}")
            return None

    def _load_image_encoder(self) -> Any | None:
        """Load the frozen native Wan 2.1 CLIP image encoder when available."""

        from wan.modules.clip import CLIPModel

        checkpoint = self.wan_checkpoint_dir / self.clip_checkpoint
        tokenizer = self.wan_checkpoint_dir / self.clip_tokenizer
        if not checkpoint.exists() or not tokenizer.exists():
            warn_condition_disabled(
                f"CLIP checkpoint or tokenizer is missing: checkpoint={checkpoint}, tokenizer={tokenizer}."
            )
            return None
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            return CLIPModel(
                dtype=dtype,
                device=self.device,
                checkpoint_path=str(checkpoint),
                tokenizer_path=str(tokenizer),
            )
        except Exception as exc:
            warn_condition_disabled(f"CLIP checkpoint loading failed: {type(exc).__name__}: {exc}")
            return None

    def _freeze_optional_encoder(self, encoder: Any | None) -> None:
        """Freeze an optional Wan encoder object if it is present."""

        if encoder is None:
            return
        model = getattr(encoder, "model", encoder)
        if isinstance(model, nn.Module):
            model.eval().requires_grad_(False)

    def _set_trainable_modules(self, training: bool) -> None:
        """Keep Wan frozen and toggle only configured LoRA modules."""

        self.wan_dit.eval()
        self.vae.model.eval().requires_grad_(False)
        self._freeze_optional_encoder(self.text_encoder)
        self._freeze_optional_encoder(self.image_encoder)
        train_blocks = self._train_block_set()
        for name, module in self.named_modules():
            if not isinstance(module, ImageBranchLoRA):
                continue
            block_idx = parse_lora_block_index(name)
            module.train(bool(training and block_idx in train_blocks))
            module.set_lora_trainable(bool(training and block_idx in train_blocks))

    def _inject_image_qkv_lora(self, expected_attn_blocks: int | None) -> list[str]:
        """Attach residual LoRA modules beside frozen QKV projections."""

        total_blocks = len(self.wan_dit.blocks)
        if expected_attn_blocks is not None and total_blocks != int(expected_attn_blocks):
            raise ValueError(f"Expected {expected_attn_blocks} Wan attention blocks, found {total_blocks}.")
        train_blocks = self._train_block_set(total_blocks=total_blocks)
        injected: list[str] = []
        for block_idx, block in enumerate(self.wan_dit.blocks):
            for target in self.lora_config.target_modules:
                parent, child_name = resolve_child_module(block, target)
                child = getattr(parent, child_name)
                if not isinstance(child, nn.Linear):
                    raise TypeError(
                        f"LoRA target blocks.{block_idx}.{target} is {type(child).__name__}, not nn.Linear."
                    )
                adapter_name = f"{child_name}_image_lora"
                if hasattr(parent, adapter_name):
                    raise RuntimeError(f"LoRA adapter blocks.{block_idx}.{target} was already installed.")
                setattr(
                    parent,
                    adapter_name,
                    ImageBranchLoRA(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        rank=self.lora_config.rank,
                        alpha=self.lora_config.alpha,
                        device=child.weight.device,
                        dropout=self.lora_config.dropout,
                        init=self.lora_config.init,
                        trainable=block_idx in train_blocks,
                    ),
                )
                injected.append(f"blocks.{block_idx}.{target}_image_lora")
        return injected

    def _train_block_set(self, total_blocks: int | None = None) -> set[int]:
        """Return the zero-based Wan block indices whose LoRA matrices train."""

        total = int(total_blocks if total_blocks is not None else len(self.wan_dit.blocks))
        all_blocks = set(range(total))
        if self.lora_config.train_attn_blocks is None:
            return all_blocks
        selected = {int(idx) for idx in self.lora_config.train_attn_blocks}
        invalid = sorted(idx for idx in selected if idx < 0 or idx >= total)
        if invalid:
            raise ValueError(f"train_attn_blocks contains invalid zero-based block indices: {invalid}.")
        if selected != all_blocks:
            raise ValueError("Stand-In requires image-token QKV LoRA training in every Wan DiT block.")
        return selected

    def _assert_only_lora_trainable(self) -> None:
        """Raise if any non-LoRA parameter is accidentally trainable."""

        offenders = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not is_lora_parameter_name(name)
        ]
        if offenders:
            raise RuntimeError(
                f"Only LoRA parameters may require gradients; found trainable non-LoRA tensors: {offenders}."
            )
        if not self.trainable_parameters():
            raise RuntimeError("No LoRA parameters are trainable; check train_attn_blocks and target_modules.")


def _select_signed_rope_frequencies(
    frequency_table: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Select RoPE multipliers while supporting CPM's temporal coordinate -1."""

    positions = positions.to(device=frequency_table.device, dtype=torch.long)
    absolute_positions = positions.abs()
    if absolute_positions.numel() and int(absolute_positions.max().item()) >= int(frequency_table.shape[0]):
        raise ValueError(
            f"CPM coordinate {int(absolute_positions.max().item())} exceeds "
            f"Wan's RoPE table length {frequency_table.shape[0]}."
        )
    selected = frequency_table.index_select(0, absolute_positions)
    return torch.where(positions.unsqueeze(-1) < 0, selected.conj(), selected)


def normalize_standin_rope_mode(rope: str) -> str:
    """Validate and return one mutually exclusive Stand-In RoPE mode."""

    normalized = str(rope).strip().lower()
    if normalized not in STANDIN_ROPE_MODES:
        choices = ", ".join(STANDIN_ROPE_MODES)
        raise ValueError(f"Unknown obj_encoder.rope={rope!r}; use one of: {choices}.")
    return normalized


def build_standin_image_rope_coordinates(
    rope: str,
    video_grid: tuple[int, int, int],
    image_grid: tuple[int, int, int],
    device: torch.device | str,
) -> StandInImageRopeCoordinates:
    """Build object coordinates by independently selecting temporal and spatial RoPE.

    Wan's native grid starts every axis at zero. Because Stand-In patchifies the
    object frames separately and appends their tokens after the video tokens,
    vanilla temporal coordinates continue at ``video_grid[0]`` while vanilla
    spatial coordinates reuse the video's zero-based spatial grid.
    """

    mode = normalize_standin_rope_mode(rope)
    video_frames, video_height, video_width = (int(value) for value in video_grid)
    image_frames = int(image_grid[0])

    if mode in {"tss-rope", "ts-rope"}:
        # Preserve Stand-In's original temporal CPM coordinate exactly.
        temporal_positions = torch.full(
            (image_frames,),
            -1,
            device=device,
            dtype=torch.long,
        )
    else:
        # Native Wan positions are consecutive, so appended object frames are
        # the trailing frames of the main video timeline.
        temporal_positions = torch.arange(
            video_frames,
            video_frames + image_frames,
            device=device,
            dtype=torch.long,
        )

    if mode in {"tss-rope", "ss-rope"}:
        # Preserve Stand-In's original disjoint spatial grid exactly.
        height_offset = video_height
        width_offset = video_width
    else:
        # Native Wan spatial positions are shared with the video patch grid.
        height_offset = 0
        width_offset = 0

    return StandInImageRopeCoordinates(
        temporal_positions=temporal_positions,
        height_offset=height_offset,
        width_offset=width_offset,
    )


def wan21_rope_apply_cpm(
    x: torch.Tensor,
    grid_size: tuple[int, int, int],
    freqs: torch.Tensor,
    temporal_positions: torch.Tensor,
    height_offset: int,
    width_offset: int,
) -> torch.Tensor:
    """Apply Wan 3D RoPE in an explicit CPM coordinate domain."""

    frames, height, width = (int(value) for value in grid_size)
    sequence_length = frames * height * width
    if int(x.shape[1]) != sequence_length:
        raise ValueError(
            f"CPM branch has {x.shape[1]} tokens, but grid {grid_size} requires {sequence_length}."
        )
    if int(temporal_positions.numel()) != frames:
        raise ValueError(
            f"CPM requires one temporal coordinate per frame; got {temporal_positions.numel()} for {frames}."
        )

    complex_dim = int(x.shape[-1]) // 2
    temporal_freqs, height_freqs, width_freqs = freqs.to(
        device=x.device,
        dtype=torch.complex64,
    ).split(
        [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3],
        dim=1,
    )
    temporal = _select_signed_rope_frequencies(temporal_freqs, temporal_positions)
    vertical = _select_signed_rope_frequencies(
        height_freqs,
        torch.arange(height, device=x.device, dtype=torch.long) + int(height_offset),
    )
    horizontal = _select_signed_rope_frequencies(
        width_freqs,
        torch.arange(width, device=x.device, dtype=torch.long) + int(width_offset),
    )
    rotary = torch.cat(
        [
            temporal.view(frames, 1, 1, -1).expand(frames, height, width, -1),
            vertical.view(1, height, 1, -1).expand(frames, height, width, -1),
            horizontal.view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(sequence_length, 1, -1)
    complex_x = torch.view_as_complex(
        x.float().reshape(x.shape[0], sequence_length, x.shape[2], -1, 2)
    )
    return torch.view_as_real(complex_x * rotary).flatten(3).to(dtype=x.dtype)


def wan21_dit_block_forward_standin(
    block: nn.Module,
    hidden: torch.Tensor,
    timestep_projection: torch.Tensor,
    reference_timestep_projection: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    context: torch.Tensor,
    context_lens: torch.Tensor | None,
    video_token_length: int,
) -> torch.Tensor:
    """Run one frozen Wan block with separate video and reference modulation."""

    video_hidden = hidden[:, :video_token_length]
    image_hidden = hidden[:, video_token_length:]
    if image_hidden.shape[1] <= 0:
        raise ValueError("Stand-In requires at least one reference-image token.")

    video_modulation = (block.modulation.float() + timestep_projection.float()).chunk(6, dim=1)
    image_modulation = (block.modulation.float() + reference_timestep_projection.float()).chunk(6, dim=1)
    video_shift_attn, video_scale_attn, video_gate_attn = video_modulation[:3]
    image_shift_attn, image_scale_attn, image_gate_attn = image_modulation[:3]

    video_attn_input = block.norm1(video_hidden).float() * (1 + video_scale_attn) + video_shift_attn
    image_attn_input = block.norm1(image_hidden).float() * (1 + image_scale_attn) + image_shift_attn
    attention_output = block.self_attn(
        torch.cat([video_attn_input, image_attn_input], dim=1),
        seq_lens,
        grid_sizes,
        freqs,
    )
    video_attention = attention_output[:, :video_token_length]
    image_attention = attention_output[:, video_token_length:]
    video_hidden = video_hidden + video_attention * video_gate_attn
    image_hidden = image_hidden + image_attention * image_gate_attn

    # Preserve Wan's original text/CLIP cross-attention on generated video tokens.
    video_hidden = video_hidden + block.cross_attn(block.norm3(video_hidden), context, context_lens)

    video_shift_ffn, video_scale_ffn, video_gate_ffn = video_modulation[3:]
    image_shift_ffn, image_scale_ffn, image_gate_ffn = image_modulation[3:]
    video_ffn_input = block.norm2(video_hidden).float() * (1 + video_scale_ffn) + video_shift_ffn
    image_ffn_input = block.norm2(image_hidden).float() * (1 + image_scale_ffn) + image_shift_ffn
    video_hidden = video_hidden + block.ffn(video_ffn_input) * video_gate_ffn
    image_hidden = image_hidden + block.ffn(image_ffn_input) * image_gate_ffn
    return torch.cat([video_hidden, image_hidden], dim=1)


def wan21_self_attention_forward_standin(
    self: nn.Module,
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Run RSA/CPM with optional image LoRA and gated object K/V injection."""

    from wan.modules.attention import flash_attention

    required_attributes = (
        "_standin_video_grid",
        "_standin_image_grid",
        "_standin_video_token_length",
        "_standin_image_token_length",
        "_standin_rope",
    )
    missing = [name for name in required_attributes if not hasattr(self, name)]
    if missing:
        raise RuntimeError(f"Stand-In self-attention is missing initialized attributes: {missing}.")

    video_grid = tuple(int(value) for value in self._standin_video_grid)
    image_grid = tuple(int(value) for value in self._standin_image_grid)
    video_length = int(self._standin_video_token_length)
    image_length = int(self._standin_image_token_length)
    if int(x.shape[1]) != video_length + image_length:
        raise ValueError(
            f"RSA expected {video_length + image_length} tokens, got {x.shape[1]}."
        )

    batch = int(x.shape[0])
    heads = int(self.num_heads)
    head_dim = int(self.head_dim)
    video_x = x[:, :video_length]
    image_x = x[:, video_length:]

    video_query = self.norm_q(self.q(video_x)).view(batch, video_length, heads, head_dim)
    video_key = self.norm_k(self.k(video_x)).view(batch, video_length, heads, head_dim)
    video_value = self.v(video_x).view(batch, video_length, heads, head_dim)

    q_delta = self.q_image_lora(image_x) if hasattr(self, "q_image_lora") else 0
    k_delta = self.k_image_lora(image_x) if hasattr(self, "k_image_lora") else 0
    v_delta = self.v_image_lora(image_x) if hasattr(self, "v_image_lora") else 0
    image_query = self.norm_q(self.q(image_x) + q_delta).view(
        batch, image_length, heads, head_dim
    )
    image_key = self.norm_k(self.k(image_x) + k_delta).view(
        batch, image_length, heads, head_dim
    )
    image_value = (self.v(image_x) + v_delta).view(batch, image_length, heads, head_dim)

    video_temporal = torch.arange(video_grid[0], device=x.device, dtype=torch.long)
    image_coordinates = build_standin_image_rope_coordinates(
        rope=self._standin_rope,
        video_grid=video_grid,
        image_grid=image_grid,
        device=x.device,
    )
    video_query = wan21_rope_apply_cpm(video_query, video_grid, freqs, video_temporal, 0, 0)
    video_key = wan21_rope_apply_cpm(video_key, video_grid, freqs, video_temporal, 0, 0)
    image_query = wan21_rope_apply_cpm(
        image_query,
        image_grid,
        freqs,
        image_coordinates.temporal_positions,
        image_coordinates.height_offset,
        image_coordinates.width_offset,
    )
    image_key = wan21_rope_apply_cpm(
        image_key,
        image_grid,
        freqs,
        image_coordinates.temporal_positions,
        image_coordinates.height_offset,
        image_coordinates.width_offset,
    )

    image_lens = torch.full((batch,), image_length, device=x.device, dtype=torch.long)
    image_output = flash_attention(
        q=image_query,
        k=image_key,
        v=image_value,
        k_lens=image_lens,
        window_size=self.window_size,
    )
    inject_object_kv = bool(getattr(self, "_standin_inject_object_kv", True))
    if inject_object_kv:
        video_keys = torch.cat([video_key, image_key], dim=1)
        video_values = torch.cat([video_value, image_value], dim=1)
        video_lens = seq_lens
    else:
        video_keys = video_key
        video_values = video_value
        video_lens = torch.full((batch,), video_length, device=x.device, dtype=torch.long)
    video_output = flash_attention(
        q=video_query,
        k=video_keys,
        v=video_values,
        k_lens=video_lens,
        window_size=self.window_size,
    )

    # An inference-only probe may reuse the exact normalized and RoPE-transformed
    # Q/K tensors from the real operation. The callback is deliberately absent
    # during training and ordinary inference, so this hook has zero extra tensor
    # allocation or attention work unless an experiment installs it.
    reference_probe = getattr(self, "_reference_attention_probe", None)
    if inject_object_kv and reference_probe is not None:
        reference_probe(
            video_query=video_query,
            concatenated_keys=video_keys,
            key_lengths=video_lens,
            window_size=self.window_size,
            video_grid=video_grid,
            image_grid=image_grid,
        )
    return self.o(torch.cat([video_output, image_output], dim=1).flatten(2))


def warn_condition_disabled(message: str) -> None:
    """Emit a visible warning whenever a Wan CLIP or T5 condition is unavailable."""

    warnings.warn(message, RuntimeWarning, stacklevel=2)


def build_wan_i2v_first_frame_mask(
    batch: int,
    frame_count: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pack Wan's pixel-time first-frame mask into four latent mask channels."""

    assert frame_count > 0 and (int(frame_count) - 1) % 4 == 0, (
        f"Wan first-frame mask requires frame_count=1+4k, got {frame_count}."
    )
    assert int(latent_frames) == 1 + (int(frame_count) - 1) // 4, (
        f"latent_frames={latent_frames} does not match frame_count={frame_count}."
    )
    frame_mask = torch.zeros(
        int(batch),
        int(frame_count),
        int(latent_height),
        int(latent_width),
        device=device,
        dtype=dtype,
    )
    frame_mask[:, :1] = 1
    packed = torch.cat([frame_mask[:, :1].repeat_interleave(4, dim=1), frame_mask[:, 1:]], dim=1)
    mask = packed.view(int(batch), int(latent_frames), 4, int(latent_height), int(latent_width))
    return mask.transpose(1, 2).contiguous()


def extend_wan_i2v_with_canonical_latents(
    zt_video: torch.Tensor,
    video_mask: torch.Tensor,
    clean_video_latents: torch.Tensor,
    reference_latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append zero x_t slots, one mask slots, and clean canonical latents."""

    tensors = {
        "zt_video": zt_video,
        "video_mask": video_mask,
        "clean_video_latents": clean_video_latents,
        "reference_latents": reference_latents,
    }
    for name, tensor in tensors.items():
        assert tensor.ndim == 5, f"{name} must be 5D, got {tuple(tensor.shape)}."
    assert int(zt_video.shape[1]) == int(clean_video_latents.shape[1]) == int(reference_latents.shape[1]) == 16
    assert int(video_mask.shape[1]) == 4
    base_metadata = (zt_video.shape[0], zt_video.shape[3], zt_video.shape[4], zt_video.device, zt_video.dtype)
    for name, tensor in tensors.items():
        metadata = (tensor.shape[0], tensor.shape[3], tensor.shape[4], tensor.device, tensor.dtype)
        assert metadata == base_metadata, f"{name} metadata {metadata} does not match x_t metadata {base_metadata}."
    assert tuple(video_mask.shape[2:]) == (
        int(zt_video.shape[2]),
        int(zt_video.shape[3]),
        int(zt_video.shape[4]),
    )
    assert tuple(clean_video_latents.shape) == tuple(zt_video.shape)

    reference_frames = int(reference_latents.shape[2])
    x_ref_placeholder = zt_video.new_zeros(
        int(zt_video.shape[0]),
        16,
        reference_frames,
        int(zt_video.shape[3]),
        int(zt_video.shape[4]),
    )
    mask_ref = video_mask.new_ones(
        int(video_mask.shape[0]),
        4,
        reference_frames,
        int(video_mask.shape[3]),
        int(video_mask.shape[4]),
    )
    xt_extended = torch.cat([zt_video, x_ref_placeholder], dim=2)
    mask_extended = torch.cat([video_mask, mask_ref], dim=2)
    clean_extended = torch.cat([clean_video_latents, reference_latents], dim=2)
    assert tuple(xt_extended.shape) == tuple(clean_extended.shape)
    assert tuple(mask_extended.shape[0:1] + mask_extended.shape[2:]) == tuple(
        xt_extended.shape[0:1] + xt_extended.shape[2:]
    )
    i2v_condition = torch.cat([mask_extended, clean_extended], dim=1)
    assert_wan_i2v_branch_compatibility(xt_extended, i2v_condition)
    return xt_extended, mask_extended, clean_extended, i2v_condition


def assert_wan_i2v_branch_compatibility(
    model_latents: torch.Tensor,
    i2v_condition: torch.Tensor,
) -> None:
    """Assert shape, device, and dtype compatibility before 36-channel concat."""

    assert model_latents.ndim == i2v_condition.ndim == 5
    assert int(model_latents.shape[1]) == 16
    assert int(i2v_condition.shape[1]) == 20
    assert tuple(model_latents.shape[0:1] + model_latents.shape[2:]) == tuple(
        i2v_condition.shape[0:1] + i2v_condition.shape[2:]
    )
    assert model_latents.device == i2v_condition.device
    assert model_latents.dtype == i2v_condition.dtype


def patchify_wan_video_and_image_inputs(
    patch_embedding: nn.Module,
    video_inputs: torch.Tensor,
    image_inputs: torch.Tensor,
    patch_size: Sequence[int],
    model_dim: int,
) -> tuple[list[torch.Tensor], tuple[int, int, int], tuple[int, int, int]]:
    """Patchify both branches identically and concatenate video then image tokens."""

    assert video_inputs.ndim == image_inputs.ndim == 5
    assert int(video_inputs.shape[0]) == int(image_inputs.shape[0])
    assert int(video_inputs.shape[1]) == int(image_inputs.shape[1])
    assert tuple(video_inputs.shape[3:]) == tuple(image_inputs.shape[3:])
    assert video_inputs.device == image_inputs.device
    assert video_inputs.dtype == image_inputs.dtype

    video_grid = expected_wan_patch_grid(video_inputs, patch_size)
    image_grid = expected_wan_patch_grid(image_inputs, patch_size)
    if video_grid[1:] != image_grid[1:]:
        raise ValueError(
            f"Video patch grid {video_grid} and image patch grid {image_grid} "
            "must have the same spatial dimensions."
        )

    token_sequences = []
    for video_sample, image_sample in zip(video_inputs, image_inputs):
        # Both branches share the native 36->D Wan Conv3D projection. Keeping
        # these calls separate prevents reference frames from being patchified
        # as part of one combined latent-time volume.
        video_volume = patch_embedding(video_sample.unsqueeze(0))
        image_volume = patch_embedding(image_sample.unsqueeze(0))
        expected_video_shape = (1, int(model_dim), *video_grid)
        expected_image_shape = (1, int(model_dim), *image_grid)
        assert tuple(video_volume.shape) == expected_video_shape, (
            f"Wan video patch embedding returned {tuple(video_volume.shape)}, expected {expected_video_shape}."
        )
        assert tuple(image_volume.shape) == expected_image_shape, (
            "Wan reference patch embedding returned "
            f"{tuple(image_volume.shape)}, expected {expected_image_shape}."
        )
        video_tokens = video_volume.flatten(2).transpose(1, 2)
        image_tokens = image_volume.flatten(2).transpose(1, 2)
        token_sequences.append(torch.cat([video_tokens, image_tokens], dim=1))
    return token_sequences, video_grid, image_grid


def expected_wan_patch_grid(
    model_latents: torch.Tensor,
    patch_size: Sequence[int],
) -> tuple[int, int, int]:
    """Return the exact Conv3D output grid after validating patch divisibility."""

    assert model_latents.ndim == 5
    temporal_patch, height_patch, width_patch = (int(value) for value in patch_size)
    frames, height, width = (int(value) for value in model_latents.shape[2:])
    assert frames % temporal_patch == height % height_patch == width % width_patch == 0, (
        f"Latent grid {(frames, height, width)} is not divisible by patch size {tuple(patch_size)}."
    )
    return frames // temporal_patch, height // height_patch, width // width_patch


def import_wan21_modules(upstream_path: str | Path) -> tuple[Any, type[nn.Module], Any]:
    """Import native Wan 2.1 modules from the configured checkout."""

    resolved = Path(upstream_path).resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))

    # Clear a previously imported Wan package when it came from another checkout.
    existing = sys.modules.get("wan.modules.model")
    if existing is not None:
        module_file = Path(getattr(existing, "__file__", "")).resolve()
        if resolved not in module_file.parents:
            for name in list(sys.modules):
                if name == "wan" or name.startswith("wan."):
                    del sys.modules[name]

    from wan.modules.model import WanModel
    from wan.modules import model as wan_model_module
    from wan.modules.vae import WanVAE

    module_file = Path(wan_model_module.__file__).resolve()
    if resolved not in module_file.parents:
        raise RuntimeError(f"Imported wan.modules.model from {module_file}, expected it under {resolved}.")
    return wan_model_module, WanModel, WanVAE


def resolve_child_module(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Resolve a dotted child module path and return its parent plus leaf name."""

    parts = dotted_name.split(".")
    if not parts:
        raise ValueError("LoRA target module path cannot be empty.")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise TypeError(f"LoRA target component {part!r} is not an nn.Module.")
    return parent, parts[-1]


def parse_lora_block_index(module_name: str) -> int:
    """Parse ``wan_dit.blocks.<idx>`` from a named LoRA module path."""

    parts = module_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "blocks" and idx + 1 < len(parts):
            return int(parts[idx + 1])
    raise ValueError(f"Could not parse Wan block index from module name {module_name!r}.")


def is_lora_parameter_name(name: str) -> bool:
    """Return True when a parameter name belongs to an injected LoRA matrix."""

    return ".lora_down." in name or ".lora_up." in name


def normalize_video_for_vae(video: torch.Tensor) -> torch.Tensor:
    """Return video as ``[B, 3, T, H, W]`` in Wan VAE ``[-1, 1]`` range."""

    if video.ndim != 5:
        raise ValueError(f"video must be 5D [B,T,3,H,W] or [B,3,T,H,W], got {tuple(video.shape)}.")
    if video.shape[1] == 3:
        video_bcthw = video.float()
    elif video.shape[2] == 3:
        video_bcthw = video.permute(0, 2, 1, 3, 4).contiguous().float()
    else:
        raise ValueError(f"Could not identify RGB channel in video shape {tuple(video.shape)}.")
    if video_bcthw.max() > 2.0:
        video_bcthw = video_bcthw.div(255.0)
    if video_bcthw.min() >= 0.0 and video_bcthw.max() <= 1.0:
        video_bcthw = video_bcthw.mul(2.0).sub(1.0)
    return video_bcthw


def normalize_reference_images(images: torch.Tensor) -> torch.Tensor:
    """Return canonical views as ``[B, R, 3, H, W]`` in ``[-1, 1]`` range."""

    if images.ndim == 4:
        images = images.unsqueeze(1)
    if images.ndim != 5:
        raise ValueError(f"reference_images must be [B,R,3,H,W] or [B,R,H,W,3], got {tuple(images.shape)}.")
    if images.shape[2] == 3:
        output = images.float()
    elif images.shape[-1] == 3:
        output = images.permute(0, 1, 4, 2, 3).contiguous().float()
    else:
        raise ValueError(f"Could not identify RGB channel in reference_images shape {tuple(images.shape)}.")
    if output.max() > 2.0:
        output = output.div(255.0)
    if output.min() >= 0.0 and output.max() <= 1.0:
        output = output.mul(2.0).sub(1.0)
    return output


def resize_reference_images(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize ``[B, R, 3, H, W]`` canonical views to the video spatial size."""

    batch, refs, channels, height, width = images.shape
    if (height, width) == tuple(size):
        return images
    flat = images.reshape(batch * refs, channels, height, width)
    resized = F.interpolate(flat, size=tuple(size), mode="bicubic", align_corners=False)
    return resized.reshape(batch, refs, channels, int(size[0]), int(size[1])).contiguous()


def resize_latent_spatial(latents: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize latent videos on the spatial dimensions while preserving time."""

    batch, channels, frames, height, width = latents.shape
    flat = latents.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    resized = F.interpolate(flat, size=tuple(size), mode="bilinear", align_corners=False)
    return resized.reshape(batch, frames, channels, int(size[0]), int(size[1])).permute(0, 2, 1, 3, 4).contiguous()


def normalize_prompt_batch(text: str | Sequence[str] | None, batch: int) -> list[str]:
    """Normalize optional prompt input to one string per batch element."""

    if text is None:
        return [""] * int(batch)
    if isinstance(text, str):
        return [text] * int(batch)
    prompts = [str(item) for item in text]
    if len(prompts) != int(batch):
        raise ValueError(f"Prompt count {len(prompts)} does not match batch {batch}.")
    return prompts


def iter_lora_modules(module: nn.Module) -> Iterable[tuple[str, ImageBranchLoRA]]:
    """Yield all injected LoRA modules by name."""

    for name, child in module.named_modules():
        if isinstance(child, ImageBranchLoRA):
            yield name, child
