"""EgoHOI camera, hand, and Stand-In object assembly for Wan 2.1 I2V.

The camera path is mirrored from ``scr/engine/camera_trainer.py`` and the hand
path is mirrored from ``v1_based_modifyv3_seplr_web.py``. This file owns the
frozen Wan base modules, trainable condition/LoRA modules, forward tensor flow,
and the exact locations where hand and camera controls enter Wan DiT.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import types
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from scr.models.encoders.camera_encoder import CameraEncoder, CameraEncoderOutput
from scr.models.encoders.egohoi_hand_encoder import (
    EgoHOIHandEncoder,
    EgoHOIReferenceHandEncoder,
)
from scr.models.encoders.plucker_encoder import PluckerEncoder, camera_center_normalization
from scr.engine.obj_stand_trainer import (
    ImageBranchLoRA,
    StandInLoRAConfig,
    expected_wan_patch_grid,
    normalize_reference_images,
    normalize_standin_rope_mode,
    patchify_wan_video_and_image_inputs,
    parse_lora_block_index,
    resize_latent_spatial,
    resize_reference_images,
    resolve_child_module,
    wan21_dit_block_forward_standin,
    wan21_self_attention_forward_standin,
)


DEFAULT_HAND_LORA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"


def prepare_optional_deepspeed_import() -> None:
    """Preload optional DeepSpeed without requiring a local CUDA toolkit.

    Transformers imports DeepSpeed when it is installed, even though this
    regular trainer does not use DeepSpeed. DeepSpeed 0.14.5 probes optional
    CUDA extensions at import time and raises on runtime-only CUDA hosts where
    PyTorch works but ``CUDA_HOME``/``nvcc`` is absent.
    """

    if "deepspeed" in sys.modules:
        return
    os.environ.setdefault("DS_ACCELERATOR", "cuda")
    original_is_available = torch.cuda.is_available
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
        try:
            import deepspeed  # noqa: F401
        except ModuleNotFoundError as error:
            if error.name != "deepspeed":
                raise
    finally:
        torch.cuda.is_available = original_is_available  # type: ignore[method-assign]


def install_wan_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 4.0,
    target_modules: str | list[str] | tuple[str, ...] = DEFAULT_HAND_LORA_TARGET_MODULES,
    init_lora_weights: bool | str = "kaiming",
) -> tuple[nn.Module, set[int]]:
    """Inject v1's PEFT LoRA adapters and return their parameter identities.

    The original Wan parameters must already be frozen. PEFT then marks only
    the inserted LoRA tensors trainable; those tensors are kept in float32 as
    in ``v1_based_modifyv3_seplr_web.py``.
    """

    prepare_optional_deepspeed_import()
    from peft import LoraConfig, inject_adapter_in_model

    if int(rank) <= 0:
        raise ValueError(f"hand_lora_rank must be positive, got {rank}.")
    targets = (
        [part.strip() for part in target_modules.split(",") if part.strip()]
        if isinstance(target_modules, str)
        else [str(part).strip() for part in target_modules if str(part).strip()]
    )
    if not targets:
        raise ValueError("hand_lora_target_modules must contain at least one module name.")

    # Architecture-spy tests use a parameter-free Wan stand-in. A real Wan
    # checkpoint must expose at least one requested suffix or configuration is
    # wrong and training should stop loudly.
    module_names = [name for name, _ in model.named_modules()]
    has_target = any(
        name == target or name.endswith(f".{target}")
        for name in module_names
        for target in targets
    )
    if not has_target:
        if next(model.parameters(), None) is None:
            return model, set()
        raise ValueError(
            "None of the Wan LoRA targets matched the loaded DiT: " + ", ".join(targets)
        )

    resolved_init = True if init_lora_weights == "kaiming" else init_lora_weights
    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=float(alpha),
        init_lora_weights=resolved_init,
        target_modules=targets,
    )
    existing_parameter_ids = {id(parameter) for parameter in model.parameters()}
    model = inject_adapter_in_model(lora_config, model)
    lora_parameter_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        # Select only tensors inserted by this PEFT call. Stand-In object LoRA
        # modules already use lora_down/lora_up names on the same Wan model and
        # must remain outside the independent hand checkpoint/optimizer view.
        if id(parameter) in existing_parameter_ids or not (
            ".lora_A." in name or ".lora_B." in name
        ):
            continue
        parameter.requires_grad_(True)
        parameter.data = parameter.data.to(dtype=torch.float32)
        lora_parameter_ids.add(id(parameter))
    if not lora_parameter_ids:
        raise RuntimeError("PEFT returned without creating any trainable Wan LoRA tensors.")
    return model, lora_parameter_ids


class WanLoRAState(nn.Module):
    """Expose only injected Wan LoRA tensors to optimizers and checkpoints.

    The referenced parameters remain registered by ``wan_dit`` so DDP follows
    the actual forward graph. This lightweight view prevents a compact control
    checkpoint from serializing the frozen 14B base model.
    """

    def __init__(self, model: nn.Module, parameter_ids: set[int]) -> None:
        """Collect stable named references to the inserted LoRA parameters."""

        super().__init__()
        references = OrderedDict(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if id(parameter) in parameter_ids
        )
        object.__setattr__(self, "_parameter_references", references)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Iterate over referenced LoRA parameters without registering Wan."""

        del recurse
        return iter(self._parameter_references.values())

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Iterate over LoRA names exactly as they appear in ``wan_dit``."""

        del recurse, remove_duplicate
        for name, parameter in self._parameter_references.items():
            yield f"{prefix}.{name}" if prefix else name, parameter

    def state_dict(
        self,
        destination: OrderedDict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> OrderedDict[str, torch.Tensor]:
        """Serialize only LoRA tensors under their stable Wan parameter names."""

        destination = destination if destination is not None else OrderedDict()
        for name, parameter in self._parameter_references.items():
            destination[prefix + name] = parameter if keep_vars else parameter.detach()
        return destination

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> nn.modules.module._IncompatibleKeys:
        """Restore compact LoRA tensors without replacing DiT references."""

        if assign:
            raise ValueError("WanLoRAState cannot replace referenced parameters with assign=True.")
        normalized = {
            key.removeprefix("pipe.dit.").removeprefix("wan_dit."): value
            for key, value in state_dict.items()
        }
        expected = set(self._parameter_references)
        supplied = set(normalized)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        errors: list[str] = []
        with torch.no_grad():
            for name in sorted(expected & supplied):
                parameter = self._parameter_references[name]
                value = normalized[name]
                if tuple(value.shape) != tuple(parameter.shape):
                    errors.append(
                        f"size mismatch for {name}: checkpoint {tuple(value.shape)} "
                        f"!= model {tuple(parameter.shape)}"
                    )
                    continue
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        if errors or (strict and (missing or unexpected)):
            details = errors
            if missing:
                details.append("missing keys: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected keys: " + ", ".join(unexpected))
            raise RuntimeError("Error(s) loading Wan LoRA state:\n  " + "\n  ".join(details))
        return nn.modules.module._IncompatibleKeys(missing, unexpected)


@dataclass
class EgoHOIControlForwardOutput:
    """Structured outputs from one multi-encoder training forward pass."""

    prediction: torch.Tensor
    target: torch.Tensor
    loss: torch.Tensor
    z0: torch.Tensor
    zt: torch.Tensor
    first_frame_condition: torch.Tensor
    reference_hand_condition: torch.Tensor | None
    reference_latents: torch.Tensor | None
    reference_timesteps: torch.Tensor | None
    hidden_states: torch.Tensor | None
    dense_plucker: torch.Tensor | None
    plucker_tokens: torch.Tensor | None
    hand_tokens: torch.Tensor | None
    hand_token_grid: tuple[int, int, int] | None
    camera: CameraEncoderOutput | None
    timesteps: torch.Tensor
    video_patch_grid: tuple[int, int, int]
    image_patch_grid: tuple[int, int, int] | None
    token_seq_len: int
    injected_layers: list[int]
    injected_obj_lora_modules: list[str]


class EgoHOIControlModel(nn.Module):
    """Assemble Wan 2.1 with independent camera, hand, and object paths.

    Frozen modules:
        - Wan 2.1 VAE: encodes/decodes videos and first-frame conditioning.
        - Wan 2.1 DiT base weights: predict diffusion velocity/noise while
          v1-compatible LoRA adapters provide the only hand-driven updates.
        - Wan T5 text encoder: optional prompt encoder, frozen when loaded.
        - Wan CLIP image encoder: optional first-frame image encoder, frozen when loaded.

    Trainable modules, constructed only when their branch is enabled:
        - ``PluckerEncoder``: converts dense Plucker rays and masks into Plucker tokens.
        - ``CameraEncoder``: builds Uni3C control residuals from render/video/camera inputs.
        - ``EgoHOIHandEncoder``: maps rendered hand RGB video to 5120D Wan tokens.
        - ``EgoHOIReferenceHandEncoder``: injects the first hand frame into the
          20-channel Wan I2V image condition before patch embedding.
        - Wan LoRA adapters on ``q,k,v,o,ffn.0,ffn.2`` when hand control is enabled.
        - Stand-In image-only QKV LoRA adapters when object control is enabled.

    The forward pass intentionally spells out the Wan block loop:

    ``wan_hidden = block(wan_hidden, ...)``
    ``wan_hidden = wan_hidden + camera_residual``

    Hand tokens are added to the main Wan token sequence immediately after
    patch embedding, matching the legacy hand-only training implementation.
    """

    def __init__(
        self,
        wan_checkpoint_dir: str | Path = "wan2_1/I2V-14B-720P",
        wan_upstream_path: str | Path = "wan2_1",
        device: torch.device | str = "cuda",
        injected_layers: list[int] | tuple[int, ...] | None = None,
        num_train_timesteps: int = 1000,
        vae_checkpoint: str = "Wan2.1_VAE.pth",
        camera_encoder_enabled: bool = True,
        hand_encoder_enabled: bool = True,
        obj_encoder_enabled: bool = False,
        camera_encoder_trainable: bool | None = None,
        hand_encoder_trainable: bool | None = None,
        obj_encoder_trainable: bool | None = None,
        plucker_hidden_dim: int = 256,
        plucker_feature_channels: int = 6,
        camera_num_layers: int = 20,
        camera_control_dim: int | None = 1024,
        camera_conv_out_dim: int | None = 5120,
        camera_ffn_dim: int | None = 4096,
        camera_num_heads: int | None = 16,
        camera_time_frequency_dim: int = 256,
        camera_time_dim: int | None = 5120,
        camera_rope_max_seq_len: int = 1024,
        camera_qk_norm: str | None = "rms_norm_across_heads",
        camera_eps: float = 1e-6,
        hand_concat_dim: int = 4,
        hand_reference_enabled: bool = True,
        hand_reference_concat_dim: int = 4,
        hand_lora_enabled: bool = True,
        hand_lora_rank: int = 4,
        hand_lora_alpha: float = 4.0,
        hand_lora_target_modules: str | list[str] | tuple[str, ...] = DEFAULT_HAND_LORA_TARGET_MODULES,
        hand_lora_init_weights: bool | str = "kaiming",
        obj_lora_rank: int = 128,
        obj_lora_alpha: float = 128.0,
        obj_lora_target_modules: Sequence[str] = ("self_attn.q", "self_attn.k", "self_attn.v"),
        obj_train_attn_blocks: Sequence[int] | None = None,
        obj_injected_layers: Sequence[int] | None = None,
        obj_lora_dropout: float = 0.0,
        obj_lora_init: str = "zero",
        obj_rope: str = "tss-rope",
        obj_expected_attn_blocks: int | None = 40,
        load_text_encoder: bool = False,
        load_image_encoder: bool = False,
        clip_checkpoint: str = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        clip_tokenizer: str = "xlm-roberta-large",
        transformer_dtype: torch.dtype | None = torch.bfloat16,
        gradient_checkpointing: bool = True,
    ) -> None:
        """Load frozen Wan modules and create only the enabled control modules."""

        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.wan_checkpoint_dir = Path(wan_checkpoint_dir)
        self.wan_upstream_path = Path(wan_upstream_path)
        self.num_train_timesteps = int(num_train_timesteps)
        self.vae_checkpoint = vae_checkpoint
        self.camera_encoder_enabled = bool(camera_encoder_enabled)
        self.hand_encoder_enabled = bool(hand_encoder_enabled)
        self.obj_encoder_enabled = bool(obj_encoder_enabled)
        self.camera_encoder_trainable = (
            self.camera_encoder_enabled
            if camera_encoder_trainable is None
            else bool(camera_encoder_trainable)
        )
        self.hand_encoder_trainable = (
            self.hand_encoder_enabled
            if hand_encoder_trainable is None
            else bool(hand_encoder_trainable)
        )
        self.obj_encoder_trainable = (
            self.obj_encoder_enabled
            if obj_encoder_trainable is None
            else bool(obj_encoder_trainable)
        )
        if self.camera_encoder_trainable and not self.camera_encoder_enabled:
            raise ValueError("camera_encoder_trainable requires camera_encoder_enabled.")
        if self.hand_encoder_trainable and not self.hand_encoder_enabled:
            raise ValueError("hand_encoder_trainable requires hand_encoder_enabled.")
        if self.obj_encoder_trainable and not self.obj_encoder_enabled:
            raise ValueError("obj_encoder_trainable requires obj_encoder_enabled.")
        self.obj_rope = normalize_standin_rope_mode(obj_rope)
        self.obj_injected_layers = (
            tuple(int(index) for index in obj_injected_layers)
            if obj_injected_layers is not None
            else None
        )
        self.hand_reference_enabled = self.hand_encoder_enabled and bool(hand_reference_enabled)
        self.hand_lora_enabled = self.hand_encoder_enabled and bool(hand_lora_enabled)
        self.hand_lora_rank = int(hand_lora_rank)
        self.hand_lora_alpha = float(hand_lora_alpha)
        self.hand_lora_target_modules = hand_lora_target_modules
        self.hand_lora_init_weights = hand_lora_init_weights
        self.clip_checkpoint = str(clip_checkpoint)
        self.clip_tokenizer = str(clip_tokenizer)
        self.load_text_encoder = bool(load_text_encoder)
        self.load_image_encoder = bool(load_image_encoder)
        self.transformer_dtype = transformer_dtype if self.device.type == "cuda" else None
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # Make the local Wan 2.1 package importable without hiding which
        # backbone is being used.
        if str(self.wan_upstream_path.resolve()) not in sys.path:
            sys.path.insert(0, str(self.wan_upstream_path.resolve()))

        from wan.modules import model as wan_model_module
        from wan.modules.model import WanModel
        from wan.modules.vae import WanVAE

        self._wan_model_module = wan_model_module

        self.wan_dit = WanModel.from_pretrained(str(self.wan_checkpoint_dir))
        if self.transformer_dtype is None:
            torch.nn.Module.to(self.wan_dit, device=self.device)
        else:
            torch.nn.Module.to(self.wan_dit, device=self.device, dtype=self.transformer_dtype)
        self.wan_dit.eval().requires_grad_(False)

        # The object branch is the exact Stand-In CPM/RSA image-token path.
        # Install it before PEFT can wrap the base QKV Linear modules for the
        # independent hand branch.
        self._obj_lora_parameter_ids: set[int] = set()
        self.injected_obj_lora_modules: list[str] = []
        self.obj_lora_config: StandInLoRAConfig | None = None
        if self.obj_encoder_enabled:
            configured_targets = tuple(str(name) for name in obj_lora_target_modules)
            required_targets = (
                ("self_attn.q", "self_attn.k", "self_attn.v")
                if self.obj_injected_layers is None
                else ("self_attn.k", "self_attn.v")
            )
            # if int(obj_lora_rank) != 128:
            #     raise ValueError(
            #         f"Stand-In requires image-branch QKV LoRA rank 128, got {obj_lora_rank}."
            #     )
            # if configured_targets != required_targets:
            #     raise ValueError(
            #         "Stand-In LoRA must target image-token QKV only; expected "
            #         f"{required_targets}, got {configured_targets}."
            #     )
            import warnings
            if int(obj_lora_rank) != 128:
                warnings.warn(
                    f"Stand-In requires image-branch QKV LoRA rank 128, got {obj_lora_rank}.",
                    UserWarning
                )
            if configured_targets != required_targets:
                warnings.warn(
                    "Stand-In LoRA must target image-token QKV only; expected "
                    f"{required_targets}, got {configured_targets}.",
                    UserWarning
                )
            self.obj_lora_config = StandInLoRAConfig(
                rank=int(obj_lora_rank),
                alpha=float(obj_lora_alpha),
                target_modules=configured_targets,
                train_attn_blocks=(
                    tuple(int(index) for index in obj_train_attn_blocks)
                    if obj_train_attn_blocks is not None
                    else None
                ),
                dropout=float(obj_lora_dropout),
                init=str(obj_lora_init),
            )
            self._validate_wan21_i2v_model()
            self._install_standin_self_attention()
            self.injected_obj_lora_modules = self._inject_image_qkv_lora(
                expected_attn_blocks=obj_expected_attn_blocks
            )
            self._obj_lora_parameter_ids = {
                id(parameter)
                for module in self.wan_dit.modules()
                if isinstance(module, ImageBranchLoRA)
                for parameter in module.parameters()
            }

        # Mirror v1: freeze every base DiT tensor, inject rank-4/alpha-4 LoRA
        # into attention and FFN projections, and keep only adapters trainable.
        self._hand_lora_parameter_ids: set[int] = set()
        if self.hand_lora_enabled:
            self.wan_dit, self._hand_lora_parameter_ids = install_wan_lora(
                self.wan_dit,
                rank=self.hand_lora_rank,
                alpha=self.hand_lora_alpha,
                target_modules=self.hand_lora_target_modules,
                init_lora_weights=self.hand_lora_init_weights,
            )
        self.wan_lora = (
            WanLoRAState(self.wan_dit, self._hand_lora_parameter_ids)
            if self.hand_lora_enabled
            else None
        )
        self.obj_encoder = (
            WanLoRAState(self.wan_dit, self._obj_lora_parameter_ids)
            if self.obj_encoder_enabled
            else None
        )

        self.vae = WanVAE(
            vae_pth=str(self.wan_checkpoint_dir / self.vae_checkpoint),
            device=self.device,
        )
        self.vae.model.eval().requires_grad_(False)

        # Optional frozen encoders. Training jobs can pass precomputed
        # ``text_context`` and ``clip_context`` to forward and leave these off.
        self.text_encoder = self._load_text_encoder() if self.load_text_encoder else None
        self.image_encoder = self._load_image_encoder() if self.load_image_encoder else None
        self._freeze_optional_encoder(self.text_encoder)
        self._freeze_optional_encoder(self.image_encoder)

        self.patch_size = tuple(int(value) for value in self.wan_dit.patch_size)
        self.vae_stride = (4, 8, 8)
        wan_hidden_dim = int(self.wan_dit.dim)
        camera_control_dim = int(camera_control_dim or wan_hidden_dim)
        camera_conv_out_dim = int(camera_conv_out_dim or camera_control_dim)
        plucker_token_dim = wan_hidden_dim
        camera_ffn_dim = int(camera_ffn_dim or camera_control_dim * 4)
        camera_num_heads = int(camera_num_heads or choose_attention_heads(camera_control_dim))
        camera_time_frequency_dim = int(camera_time_frequency_dim)
        camera_time_dim = int(camera_time_dim or wan_hidden_dim)
        if self.camera_encoder_enabled and camera_time_frequency_dim != int(self.wan_dit.freq_dim):
            raise ValueError(
                "Uni3C camera_time_frequency_dim must match the frozen Wan "
                f"freq_dim={self.wan_dit.freq_dim}, got {camera_time_frequency_dim}."
            )
        if self.camera_encoder_enabled and camera_time_dim != wan_hidden_dim:
            raise ValueError(
                "Uni3C camera_time_dim must match the frozen Wan timestep "
                f"embedding width {wan_hidden_dim}, got {camera_time_dim}."
            )
        self.camera_time_frequency_dim = camera_time_frequency_dim

        # Trainable PluckerEncoder: concat(dense rays, rendered mask)
        # [B, 7, 81, 480, 480] -> Plucker tokens [B, 21 * 30 * 30, Wan_dim].
        self.plucker_encoder = (
            PluckerEncoder(
                in_channels=1 + int(plucker_feature_channels),
                hidden_dim=int(plucker_hidden_dim),
                out_dim=plucker_token_dim,
            ).to(self.device)
            if self.camera_encoder_enabled
            else None
        )

        # Trainable CameraEncoder: rendered video + Plucker tokens
        # become residual tokens [B, N, Wan_dim] for explicit DiT injection.
        self.camera_encoder = (
            CameraEncoder(
                render_vae=self.vae,
                latent_channels=16,
                condition_channels=20,
                patch_size=self.patch_size,
                conv_out_dim=camera_conv_out_dim,
                control_dim=camera_control_dim,
                control_ffn_dim=camera_ffn_dim,
                control_num_heads=camera_num_heads,
                num_control_layers=int(camera_num_layers),
                wan_hidden_dim=wan_hidden_dim,
                time_frequency_dim=self.camera_time_frequency_dim,
                time_dim=camera_time_dim,
                rope_max_seq_len=int(camera_rope_max_seq_len),
                qk_norm=camera_qk_norm,
                eps=float(camera_eps),
                plucker_token_dim=plucker_token_dim,
                gradient_checkpointing=self.gradient_checkpointing,
            ).to(self.device)
            if self.camera_encoder_enabled
            else None
        )

        # Trainable hand encoder mirrored from the legacy ``dwpose_embedding``
        # stack. For [B,3,81,480,480], it emits a volume
        # [B,Wan_dim,21,30,30] and flattened tokens [B,18900,Wan_dim].
        self.hand_encoder = (
            EgoHOIHandEncoder(
                embedding_dim=wan_hidden_dim,
                concat_dim=int(hand_concat_dim),
            ).to(self.device)
            if self.hand_encoder_enabled
            else None
        )
        # V1 encodes the first rendered hand frame to 20 channels and adds it
        # to image_emb["y"] before the 36-channel Wan input is patchified.
        self.reference_hand_encoder = (
            EgoHOIReferenceHandEncoder(
                out_channels=20,
                concat_dim=int(hand_reference_concat_dim),
            ).to(self.device)
            if self.hand_reference_enabled
            else None
        )
        if self.camera_encoder_enabled:
            assert self.camera_encoder is not None
            if injected_layers is None:
                # Match Uni3C's one-residual-per-ControlNet-block behavior by
                # injecting into the earliest Wan blocks unless configured otherwise.
                injected_layers = list(range(int(camera_num_layers)))
            self.injected_layers = [int(layer_idx) for layer_idx in injected_layers]
            if len(self.injected_layers) > self.camera_encoder.num_control_layers:
                raise ValueError("Cannot inject more Wan layers than CameraEncoder residual blocks.")
        else:
            self.injected_layers = []

        self._set_trainable_modules()

    def forward(
        self,
        video: torch.Tensor,
        rendered_video: torch.Tensor | None = None,
        rendered_mask: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        hand_video: torch.Tensor | None = None,
        reference_images: torch.Tensor | None = None,
        first_frame: torch.Tensor | None = None,
        text_context: list[torch.Tensor] | None = None,
        clip_context: torch.Tensor | None = None,
        prompts: list[str] | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> EgoHOIControlForwardOutput:
        """Run the enabled camera, hand, and object training dataflow.

        Args:
            video: Ground-truth RGB video, ``[B, T, 3, H, W]`` in ``[0, 1]`` or
                already-normalized ``[-1, 1]`` range.
            rendered_video: Rendered point-cloud RGB video, ``[B, 3, T, H, W]``.
            rendered_mask: Rendered point-cloud mask, ``[B, 1, T, H, W]``.
            intrinsics: Camera intrinsics, ``[B, T, 3, 3]``.
            extrinsics: Camera-to-world or world-to-camera matrices,
                ``[B, T, 4, 4]``. The Plucker convention follows the local
                preprocessing utilities: rays are built in the supplied camera
                trajectory frame.
            hand_video: Rendered RGB hand video, ``[B,3,T,H,W]`` or
                ``[B,T,3,H,W]``. When omitted, camera-only conditioning is
                retained for backward compatibility.
            reference_images: Canonical object views as ``[B,R,3,H,W]`` or
                ``[B,R,H,W,3]`` when the Stand-In object branch is enabled.
            first_frame: Optional first frame ``[B, 3, H, W]``. When omitted,
                ``video[:, 0]`` is used.
            text_context: Optional precomputed Wan T5 context list, each item
                shaped ``[L, 4096]``.
            clip_context: Optional precomputed Wan CLIP visual context,
                ``[B, 257, 1280]``.
            prompts: Optional text prompts used only when the frozen text
                encoder was loaded.
            timesteps: Optional Wan diffusion timesteps, ``[B]``.
            noise: Optional noise tensor matching VAE latents.
        """

        if video.ndim != 5:
            raise ValueError(
                "video must be [B,T,3,H,W], "
                f"got {tuple(video.shape)}."
            )
        batch, frame_count, _, image_height, image_width = (
            int(value) for value in video.shape
        )
        if first_frame is None:
            # Select the small first-frame view before preparing the full VAE
            # input so the source video itself can remain in host memory.
            first_frame = video[:, 0]

        if self.camera_encoder_enabled:
            camera_inputs = {
                "rendered_video": rendered_video,
                "rendered_mask": rendered_mask,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            }
            missing_inputs = [name for name, value in camera_inputs.items() if value is None]
            if missing_inputs:
                raise ValueError(
                    "Camera control is enabled but these inputs are missing: "
                    + ", ".join(missing_inputs)
                )
            assert rendered_video is not None
            assert rendered_mask is not None
            assert intrinsics is not None
            assert extrinsics is not None
        if self.hand_encoder_enabled:
            if hand_video is None:
                raise ValueError("Hand control is enabled but hand_video is missing.")
        if self.obj_encoder_enabled and reference_images is None:
            raise ValueError("Object control is enabled but reference_images is missing.")

        # 1. input video -> Wan VAE encode
        # video: [B, T, 3, H, W]
        # z0:    [B, 16, T_lat, H_lat, W_lat]
        # Build one private channel-first copy and normalize it in place. This
        # avoids retaining separate full-resolution CUDA tensors for the raw
        # video and its normalized VAE input.
        video_for_vae = video.permute(0, 2, 1, 3, 4).contiguous().to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        with torch.no_grad():
            normalize_video_in_place(video_for_vae)
            encoded_video = self.vae.encode(list(video_for_vae.unbind(dim=0)))
            z0 = torch.stack(encoded_video, dim=0)
        # WanVAE.encode clears its internal feature cache. Drop the caller-side
        # list and full-resolution input before allocating diffusion tensors.
        del encoded_video, video_for_vae

        # 2. diffusion noise addition
        # zt and target follow Wan flow matching:
        # zt = (1 - sigma) * z0 + sigma * noise
        # target = noise - z0
        if timesteps is None:
            timesteps = torch.randint(0, self.num_train_timesteps, (batch,), device=self.device, dtype=torch.long)
        else:
            timesteps = timesteps.to(device=self.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(z0)
        else:
            noise = noise.to(device=self.device, dtype=z0.dtype)
        sigma = timesteps.float().view(batch, 1, 1, 1, 1) / float(self.num_train_timesteps)
        zt = (1.0 - sigma) * z0 + sigma * noise
        target = noise - z0
        # zt and target now own the required values; the sampled noise is not
        # needed by any later layer or by backward through frozen VAE latents.
        del noise, sigma

        # 3. first-frame I2V condition
        # first_frame_condition: [B, 20, T_lat, H_lat, W_lat]
        # first_frame_condition = concat(first_frame_mask_latents, first_frame_image_latents)
        # concat([B, 4, F_lat, H_lat, W_lat], [B, 16, F_lat, H_lat, W_lat])
        # first_frame_mask_latents: [B, 4, F_lat, H_lat, W_lat] (nothing related to rendered mask)
        # first_frame_image_latents: [B, 16, F_lat, H_lat, W_lat] (first frame + zero padding)
        # Note: first_frame_mask_latents: 
        # to tell the conditioning image exists only at the first video frame temporal group. 
        # It is not a segmentation mask, not the rendered mask, and not the actual first-frame image. 
        first_frame = first_frame.to(self.device)
        first_frame_condition = self._encode_first_frame_condition(
            first_frame,
            frame_count=frame_count,
            latent_grid=z0.shape[2:],
        )
        camera_i2v_mask = first_frame_condition[:, :4]

        # 4. V1 REFERENCE-HAND INJECTION INTO image_emb["y"] BEFORE PATCHIFY.
        # The first rendered hand frame produces [B,20,1,H/8,W/8], whose time
        # axis broadcasts over the complete [B,20,T_lat,H/8,W/8] I2V condition.
        reference_hand_condition: torch.Tensor | None = None
        if self.hand_reference_enabled:
            assert hand_video is not None
            reference_hand_condition = self.encode_reference_hand_condition(
                hand_video=hand_video,
                target_condition=first_frame_condition,
            )
            first_frame_condition = first_frame_condition + reference_hand_condition

        # 5. STAND-IN OBJECT-REFERENCE LATENTS. Canonical views use the same
        # frozen VAE as the video but remain a separate clean token branch.
        reference_latents: torch.Tensor | None = None
        reference_timesteps: torch.Tensor | None = None
        if self.obj_encoder_enabled:
            assert reference_images is not None
            reference_latents = self.encode_object_reference_latents(
                reference_images=reference_images,
                video_spatial_size=(image_height, image_width),
                target_latent=z0,
            )
            reference_timesteps = torch.zeros_like(timesteps)

        # 6. CameraEncoder hidden_states are zt-first because
        # CameraEncoder consumes hidden_states[:, :20] as zt plus the I2V mask.
        # zt:                    [B, 16, T_lat, H_lat, W_lat]
        # camera_i2v_mask:       [B, 4, T_lat, H_lat, W_lat]
        # first-frame latents:   [B, 16, T_lat, H_lat, W_lat]
        # hidden_states:         [B, 36, T_lat, H_lat, W_lat]
        hidden_states: torch.Tensor | None = None
        if self.camera_encoder_enabled:
            hidden_states = torch.cat([zt, camera_i2v_mask, first_frame_condition[:, 4:]], dim=1)
        # print(f"hidden_states: {hidden_states.shape}, should be [B, 36, 21, 60, 60]")

        # 7. Plucker ray construction and trainable Plucker-token encoding
        # dense_plucker:  [B, 6, T, H, W]
        # rendered_mask:  [B, 1, T, H, W]
        # concat(dense_plucker, rendered_mask),
        # and downsampled with Conv3d (4,8,8) then (1,2,2)
        # plucker_tokens: [B, T_lat * H/16 * W/16, Wan_dim]
        # H/16 is H_lat, W/16 is W_lat.
        dense_plucker: torch.Tensor | None = None
        plucker_tokens: torch.Tensor | None = None
        if self.camera_encoder_enabled:
            assert self.plucker_encoder is not None
            assert rendered_mask is not None
            assert intrinsics is not None
            assert extrinsics is not None
            rendered_mask = rendered_mask.to(self.device)
            intrinsics = intrinsics.to(self.device)
            extrinsics = extrinsics.to(self.device)
            dense_plucker = build_dense_plucker_rays(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                height=int(image_height),
                width=int(image_width),
            ).to(device=self.device, dtype=z0.dtype)
            plucker_tokens = self.plucker_encoder(dense_plucker, rendered_mask)

        # 8. HAND SEQUENCE ENCODER CALL
        # hand_video:      [B, 3, T, H, W] or [B, T, 3, H, W]
        # prepared inside: [B, 3, T+3, H, W] (legacy first-frame padding)
        # hand volume:     [B, 5120, T_hand, H/16, W/16]
        # hand_tokens:     [B, T_hand * H/16 * W/16, 5120]
        hand_tokens: torch.Tensor | None = None
        hand_token_grid: tuple[int, int, int] | None = None
        if self.hand_encoder_enabled:
            assert self.hand_encoder is not None
            assert hand_video is not None
            hand_tokens, hand_token_grid = self.hand_encoder.encode_tokens(hand_video)

        # 9. CameraEncoder forward
        # rendered_video is encoded inside CameraEncoder:
        # rendered_video -> render_latent [B, 16, T_lat, H_lat, W_lat]
        # plucker_tokens are added to control tokens inside CameraEncoder.
        # external DiT branch -> residuals[i] [B, N, Wan_dim]
        camera: CameraEncoderOutput | None = None
        if self.camera_encoder_enabled:
            assert self.camera_encoder is not None
            assert rendered_video is not None
            assert rendered_mask is not None
            assert plucker_tokens is not None
            assert hidden_states is not None
            rendered_video = rendered_video.to(self.device)
            camera = self.camera_encoder(
                rendered_video=normalize_rendered_video(rendered_video),
                rendered_mask=rendered_mask,
                hidden_states=hidden_states,
                plucker_tokens=plucker_tokens,
                timestep=timesteps,
                temb=self._camera_timestep_embedding(timesteps),
            )
            # CameraEncoder has already converted the rendered RGB video to
            # latents. Its autograd graph retains only tensors needed for
            # trainable camera layers, so these outer input references can go.
            del rendered_video, rendered_mask, intrinsics, extrinsics

        # Both hand encoders have consumed the source render. Saved autograd
        # tensors keep only the values required by their trainable convolutions.
        if hand_video is not None:
            del hand_video

        # 10. frozen text/image conditions for Wan DiT
        # context:      list of [L, 4096] before Wan text_embedding
        # clip_context: [B, 257, 1280] before Wan img_emb
        text_context = self._resolve_text_context(batch, z0.dtype, prompts, text_context)
        clip_context = self._resolve_clip_context(batch, z0.dtype, first_frame, clip_context)

        # 11. Add hand tokens and camera residuals only to video tokens; the
        # object branch independently contributes reference K/V through RSA.
        prediction = self._wan_forward_with_controls(
            zt=zt,
            first_frame_condition=first_frame_condition,
            timesteps=timesteps,
            text_context=text_context,
            clip_context=clip_context,
            camera=camera,
            hand_tokens=hand_tokens,
            hand_token_grid=hand_token_grid,
            reference_latents=reference_latents,
        )

        loss = F.mse_loss(prediction, target)
        video_patch_grid = expected_wan_patch_grid(zt, self.patch_size)
        image_patch_grid = (
            expected_wan_patch_grid(reference_latents, self.patch_size)
            if reference_latents is not None
            else None
        )
        token_seq_len = math.prod(video_patch_grid) + (
            math.prod(image_patch_grid) if image_patch_grid is not None else 0
        )
        return EgoHOIControlForwardOutput(
            prediction=prediction,
            target=target,
            loss=loss,
            z0=z0,
            zt=zt,
            first_frame_condition=first_frame_condition,
            reference_hand_condition=reference_hand_condition,
            reference_latents=reference_latents,
            reference_timesteps=reference_timesteps,
            hidden_states=hidden_states,
            dense_plucker=dense_plucker,
            plucker_tokens=plucker_tokens,
            hand_tokens=hand_tokens,
            hand_token_grid=hand_token_grid,
            camera=camera,
            timesteps=timesteps,
            video_patch_grid=video_patch_grid,
            image_patch_grid=image_patch_grid,
            token_seq_len=token_seq_len,
            injected_layers=list(self.injected_layers),
            injected_obj_lora_modules=list(self.injected_obj_lora_modules),
        )

    def train(self, mode: bool = True) -> "EgoHOIControlModel":
        """Train controls/LoRA while keeping every Wan base tensor frozen."""

        super().train(mode)
        # V1 places the LoRA-adapted DiT in train mode. Only adapter tensors
        # require gradients; a hand-disabled or LoRA-disabled Wan remains eval.
        self.wan_dit.train(mode if self.hand_lora_enabled and self.hand_encoder_trainable else False)
        self.vae.model.eval()
        self._freeze_optional_encoder(self.text_encoder)
        self._freeze_optional_encoder(self.image_encoder)
        trainable_names = set(self.trainable_modules())
        for name, module in self.active_modules().items():
            if name in trainable_names:
                module.train(mode)
            else:
                module.eval()
        self._set_obj_lora_training(mode)
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that should receive optimizer updates."""

        return [
            parameter
            for parameters in self.trainable_parameter_groups().values()
            for parameter in parameters
        ]

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Separate camera, hand encoders, and Wan LoRA for LR policies."""

        groups: dict[str, list[nn.Parameter]] = {}
        if self.camera_encoder_trainable:
            assert self.plucker_encoder is not None
            assert self.camera_encoder is not None
            groups["camera"] = [
                parameter
                for module in (self.plucker_encoder, self.camera_encoder)
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
        if self.hand_encoder_trainable:
            assert self.hand_encoder is not None
            hand_modules: list[nn.Module] = [self.hand_encoder]
            if self.reference_hand_encoder is not None:
                hand_modules.append(self.reference_hand_encoder)
            groups["hand"] = [
                parameter
                for module in hand_modules
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
        if self.hand_encoder_trainable and self.wan_lora is not None:
            lora_parameters = [
                parameter
                for parameter in self.wan_lora.parameters()
                if parameter.requires_grad
            ]
            if lora_parameters:
                groups["wan_lora"] = lora_parameters
        if self.obj_encoder_trainable and self.obj_encoder is not None:
            obj_parameters = [
                parameter
                for parameter in self.obj_encoder.parameters()
                if parameter.requires_grad
            ]
            if obj_parameters:
                groups["obj"] = obj_parameters
        return groups

    def frozen_modules(self) -> dict[str, nn.Module | Any | None]:
        """Return the frozen backbone modules for logging and inspection."""

        return {
            "wan_vae": self.vae.model,
            "wan_dit": self.wan_dit,
            "wan_text_encoder": self.text_encoder,
            "wan_image_encoder": self.image_encoder,
        }

    def trainable_modules(self) -> dict[str, nn.Module]:
        """Return the trainable condition modules for logging and inspection."""

        modules: dict[str, nn.Module] = {}
        if self.camera_encoder_trainable:
            assert self.plucker_encoder is not None
            assert self.camera_encoder is not None
            modules["plucker_encoder"] = self.plucker_encoder
            modules["camera_encoder"] = self.camera_encoder
        if self.hand_encoder_trainable:
            assert self.hand_encoder is not None
            modules["hand_encoder"] = self.hand_encoder
        if self.hand_encoder_trainable and self.hand_reference_enabled:
            assert self.reference_hand_encoder is not None
            modules["reference_hand_encoder"] = self.reference_hand_encoder
        if self.hand_encoder_trainable and self.hand_lora_enabled:
            assert self.wan_lora is not None
            modules["wan_lora"] = self.wan_lora
        if self.obj_encoder_trainable:
            assert self.obj_encoder is not None
            modules["obj_encoder"] = self.obj_encoder
        return modules

    def active_modules(self) -> dict[str, nn.Module]:
        """Return active compact control modules, including frozen branches."""

        modules: dict[str, nn.Module] = {}
        if self.camera_encoder_enabled:
            assert self.plucker_encoder is not None
            assert self.camera_encoder is not None
            modules["plucker_encoder"] = self.plucker_encoder
            modules["camera_encoder"] = self.camera_encoder
        if self.hand_encoder_enabled:
            assert self.hand_encoder is not None
            modules["hand_encoder"] = self.hand_encoder
        if self.hand_reference_enabled:
            assert self.reference_hand_encoder is not None
            modules["reference_hand_encoder"] = self.reference_hand_encoder
        if self.hand_lora_enabled:
            assert self.wan_lora is not None
            modules["wan_lora"] = self.wan_lora
        if self.obj_encoder_enabled:
            assert self.obj_encoder is not None
            modules["obj_encoder"] = self.obj_encoder
        return modules

    def _wan_forward_with_controls(
        self,
        zt: torch.Tensor,
        first_frame_condition: torch.Tensor,
        timesteps: torch.Tensor,
        text_context: list[torch.Tensor],
        clip_context: torch.Tensor,
        camera: CameraEncoderOutput | None,
        hand_tokens: torch.Tensor | None,
        hand_token_grid: tuple[int, int, int] | None,
        reference_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Wan DiT with every enabled control on its intended token branch."""

        from wan.modules.model import sinusoidal_embedding_1d

        video_wan_inputs = torch.cat(
            [zt, first_frame_condition],
            dim=1,
        )
        if int(video_wan_inputs.shape[1]) != int(self.wan_dit.in_dim):
            raise ValueError(
                f"Wan I2V patch embedding expects {self.wan_dit.in_dim} channels, "
                f"got video={video_wan_inputs.shape[1]}."
            )

        image_patch_grid: tuple[int, int, int] | None = None
        if self.obj_encoder_enabled:
            dit_dtype = self.wan_dit.patch_embedding.weight.dtype
            video_wan_inputs = video_wan_inputs.to(dtype=dit_dtype)
            if reference_latents is None:
                raise ValueError("Object control requires reference_latents in the Wan forward.")
            reference_mask = reference_latents.new_ones(
                int(reference_latents.shape[0]),
                4,
                int(reference_latents.shape[2]),
                int(reference_latents.shape[3]),
                int(reference_latents.shape[4]),
            )
            reference_model_latents = torch.zeros_like(reference_latents)
            reference_condition = torch.cat([reference_mask, reference_latents], dim=1)
            reference_wan_inputs = torch.cat(
                [reference_model_latents, reference_condition], dim=1
            ).to(dtype=dit_dtype)
            wan_tokens_per_sample, video_patch_grid, image_patch_grid = (
                patchify_wan_video_and_image_inputs(
                    patch_embedding=self.wan_dit.patch_embedding,
                    video_inputs=video_wan_inputs,
                    image_inputs=reference_wan_inputs,
                    patch_size=self.patch_size,
                    model_dim=int(self.wan_dit.dim),
                )
            )
            del reference_mask, reference_model_latents, reference_condition
        else:
            if reference_latents is not None:
                raise ValueError("reference_latents were supplied while object control is disabled.")
            patch_volumes = [
                self.wan_dit.patch_embedding(sample.unsqueeze(0))
                for sample in video_wan_inputs
            ]
            video_patch_grid = tuple(int(value) for value in patch_volumes[0].shape[2:])
            if any(
                tuple(int(value) for value in volume.shape[2:]) != video_patch_grid
                for volume in patch_volumes
            ):
                raise ValueError("All video samples must share one Wan patch grid.")
            wan_tokens_per_sample = [
                volume.flatten(2).transpose(1, 2) for volume in patch_volumes
            ]
            del patch_volumes

        video_token_length = math.prod(video_patch_grid)
        image_token_length = math.prod(image_patch_grid) if image_patch_grid is not None else 0
        seq_lens = torch.tensor(
            [tokens.size(1) for tokens in wan_tokens_per_sample],
            dtype=torch.long,
            device=self.device,
        )
        seq_len = int(seq_lens.max().item())
        expected_seq_len = video_token_length + image_token_length
        if seq_len != expected_seq_len:
            raise ValueError(
                f"Wan token length {seq_len} does not match video/object grids "
                f"{expected_seq_len}."
            )
        grid_sizes = torch.tensor(
            [video_patch_grid] * len(wan_tokens_per_sample),
            dtype=torch.long,
            device=self.device,
        )
        wan_hidden = torch.cat(
            [
                torch.cat(
                    [tokens, tokens.new_zeros(1, seq_len - tokens.size(1), tokens.size(2))],
                    dim=1,
                )
                for tokens in wan_tokens_per_sample
            ],
            dim=0,
        )
        del video_wan_inputs, wan_tokens_per_sample

        # HAND TOKEN ADDITION TO THE MAIN TOKEN SEQUENCE
        # wan_hidden before:   [B, seq_len, Wan_dim]
        # hand_tokens before:  [B, N_hand, Wan_dim]
        # aligned hand tokens: [B, N_video, Wan_dim]
        # Object tokens are never altered by the hand sequence encoder.
        # For 81x480x480, both grids are [21,30,30], so seq_len=18900
        # and this matches the legacy elementwise x = x + condition_tokens.
        if hand_tokens is not None:
            if hand_token_grid is None:
                raise ValueError("hand_token_grid is required when hand_tokens are provided.")
            aligned_hand_tokens = align_hand_tokens_to_wan_tokens(
                hand_tokens=hand_tokens,
                source_grid=hand_token_grid,
                target_grid_sizes=grid_sizes,
                target_seq_len=video_token_length,
            ).to(device=wan_hidden.device, dtype=wan_hidden.dtype)
            if image_token_length:
                wan_hidden = torch.cat(
                    [
                        wan_hidden[:, :video_token_length] + aligned_hand_tokens,
                        wan_hidden[:, video_token_length:],
                    ],
                    dim=1,
                )
            else:
                wan_hidden = wan_hidden + aligned_hand_tokens
            # The sum's autograd node retains what backward needs; this outer
            # reference would otherwise keep another full Wan-width token grid.
            del aligned_hand_tokens

        # Wan timestep embeddings:
        # e:  [B, Wan_dim]
        # e0: [B, 6, Wan_dim]
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = self.wan_dit.time_embedding(sinusoidal_embedding_1d(self.wan_dit.freq_dim, timesteps).float())
            e0 = self.wan_dit.time_projection(e).unflatten(1, (6, self.wan_dit.dim))
            reference_e0: torch.Tensor | None = None
            if self.obj_encoder_enabled:
                reference_timesteps = torch.zeros_like(timesteps)
                reference_embedding = self.wan_dit.time_embedding(
                    sinusoidal_embedding_1d(
                        self.wan_dit.freq_dim, reference_timesteps
                    ).float()
                )
                reference_e0 = self.wan_dit.time_projection(reference_embedding).unflatten(
                    1, (6, self.wan_dit.dim)
                )

        # Wan text and optional image context.
        text_context = [
            torch.cat([ctx, ctx.new_zeros(self.wan_dit.text_len - ctx.size(0), ctx.size(1))])
            for ctx in text_context
        ]
        context_tokens = self.wan_dit.text_embedding(torch.stack(text_context, dim=0))
        if clip_context is not None:
            context_clip = self.wan_dit.img_emb(clip_context)
            context_tokens = torch.cat([context_clip, context_tokens], dim=1)

        if self.wan_dit.freqs.device != self.device:
            self.wan_dit.freqs = self.wan_dit.freqs.to(self.device)

        standard_block_kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.wan_dit.freqs,
            "context": context_tokens,
            "context_lens": None,
        }

        residual_by_layer: dict[int, torch.Tensor] = {}
        if camera is not None:
            residual_by_layer = {
                layer_idx: camera.residuals[residual_idx]
                for residual_idx, layer_idx in enumerate(self.injected_layers)
            }

        for layer_idx, block in enumerate(self.wan_dit.blocks):
            if self.obj_encoder_enabled:
                assert image_patch_grid is not None
                assert reference_e0 is not None
                block.self_attn._standin_video_grid = video_patch_grid
                block.self_attn._standin_image_grid = image_patch_grid
                block.self_attn._standin_video_token_length = video_token_length
                block.self_attn._standin_image_token_length = image_token_length
                block.self_attn._standin_inject_object_kv = layer_idx in self._obj_injected_layer_set()
                block_forward = lambda hidden, wan_block=block: wan21_dit_block_forward_standin(
                    wan_block,
                    hidden,
                    timestep_projection=e0,
                    reference_timestep_projection=reference_e0,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=self.wan_dit.freqs,
                    context=context_tokens,
                    context_lens=None,
                    video_token_length=video_token_length,
                )
            else:
                block_forward = lambda hidden, wan_block=block: wan_block(
                    hidden, **standard_block_kwargs
                )
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                wan_hidden = checkpoint(block_forward, wan_hidden, use_reentrant=False)
            else:
                wan_hidden = block_forward(wan_hidden)

            if layer_idx in residual_by_layer:
                assert camera is not None
                # Explicit Uni3C injection point:
                # camera_residual[layer_idx]: [B, N_camera, Wan_dim]
                # aligned_residual:           [B, seq_len, Wan_dim]
                # wan_hidden[layer_idx] += camera_residual[layer_idx]
                camera_residual = residual_by_layer[layer_idx]
                aligned_residual = align_camera_residual_to_wan_tokens(
                    camera_residual=camera_residual,
                    source_grid=camera.token_grid,
                    target_grid_sizes=grid_sizes,
                    target_seq_len=video_token_length,
                ).to(device=wan_hidden.device, dtype=wan_hidden.dtype)
                if image_token_length:
                    wan_hidden = torch.cat(
                        [
                            wan_hidden[:, :video_token_length] + aligned_residual,
                            wan_hidden[:, video_token_length:],
                        ],
                        dim=1,
                    )
                else:
                    wan_hidden = wan_hidden + aligned_residual

        # Wan prediction head and unpatchify:
        # hidden tokens -> [B, seq_len, out_dim * prod(patch_size)]
        # unpatchified list -> stack [B, 16, T_lat, H_lat, W_lat]
        if image_token_length:
            wan_hidden = wan_hidden[:, :video_token_length]
        wan_hidden = self.wan_dit.head(wan_hidden, e)
        prediction_list = self.wan_dit.unpatchify(wan_hidden, grid_sizes)
        return torch.stack(prediction_list, dim=0)

    def encode_object_reference_latents(
        self,
        reference_images: torch.Tensor,
        video_spatial_size: tuple[int, int],
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Encode canonical images as clean one-frame Stand-In VAE latents."""

        images = normalize_reference_images(reference_images).to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        images = resize_reference_images(images, size=video_spatial_size)
        latents_by_sample: list[torch.Tensor] = []
        with torch.no_grad():
            for sample in images:
                view_latents = [
                    self.vae.encode([reference.unsqueeze(1)])[0]
                    for reference in sample
                ]
                latents_by_sample.append(torch.cat(view_latents, dim=1))
        reference_latents = torch.stack(latents_by_sample, dim=0).to(
            device=target_latent.device,
            dtype=target_latent.dtype,
        )
        if int(reference_latents.shape[0]) != int(target_latent.shape[0]):
            raise ValueError(
                "Object-reference batch does not match video batch: "
                f"{reference_latents.shape[0]} != {target_latent.shape[0]}."
            )
        if reference_latents.shape[-2:] != target_latent.shape[-2:]:
            reference_latents = resize_latent_spatial(
                reference_latents, tuple(int(value) for value in target_latent.shape[-2:])
            )
        return reference_latents

    def _validate_wan21_i2v_model(self) -> None:
        """Require the native Wan 2.1 I2V dimensions used by Stand-In."""

        model_type = str(getattr(self.wan_dit, "model_type", ""))
        in_dim = int(getattr(self.wan_dit, "in_dim", 0))
        out_dim = int(getattr(self.wan_dit, "out_dim", 0))
        if model_type != "i2v" or in_dim != 36 or out_dim != 16:
            raise ValueError(
                "Stand-In object control requires native Wan 2.1 I2V with "
                "model_type='i2v', in_dim=36, and out_dim=16; found "
                f"model_type={model_type!r}, in_dim={in_dim}, out_dim={out_dim}."
            )

    def _install_standin_self_attention(self) -> None:
        """Install Stand-In RSA/CPM attention without changing base weights."""

        for block in self.wan_dit.blocks:
            # One immutable mode controls both object RoPE axes for this run.
            block.self_attn._standin_rope = self.obj_rope
            block.self_attn.forward = types.MethodType(
                wan21_self_attention_forward_standin, block.self_attn
            )

    def _inject_image_qkv_lora(self, expected_attn_blocks: int | None) -> list[str]:
        """Attach image-token LoRA modules for the integrated object branch."""

        if self.obj_lora_config is None:
            return []
        total_blocks = len(self.wan_dit.blocks)
        if expected_attn_blocks is not None and total_blocks != int(expected_attn_blocks):
            raise ValueError(
                f"Expected {expected_attn_blocks} Wan attention blocks, found {total_blocks}."
            )
        train_blocks = self._obj_injected_layer_set(total_blocks)
        injected: list[str] = []
        for block_index, block in enumerate(self.wan_dit.blocks):
            if block_index not in train_blocks:
                continue
            for target in self.obj_lora_config.target_modules:
                parent, child_name = resolve_child_module(block, target)
                child = getattr(parent, child_name)
                if not isinstance(child, nn.Linear):
                    raise TypeError(
                        f"Object LoRA target blocks.{block_index}.{target} is "
                        f"{type(child).__name__}, not nn.Linear."
                    )
                adapter_name = f"{child_name}_image_lora"
                if hasattr(parent, adapter_name):
                    raise RuntimeError(
                        f"Object LoRA blocks.{block_index}.{target} is already installed."
                    )
                setattr(
                    parent,
                    adapter_name,
                    ImageBranchLoRA(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        rank=self.obj_lora_config.rank,
                        alpha=self.obj_lora_config.alpha,
                        device=child.weight.device,
                        dropout=self.obj_lora_config.dropout,
                        init=self.obj_lora_config.init,
                        trainable=True,
                    ),
                )
                injected.append(
                    f"blocks.{block_index}.{target}_image_lora"
                )
        return injected

    def _obj_injected_layer_set(self, total_blocks: int | None = None) -> set[int]:
        """Return object layers where video tokens may read object K/V."""

        total = int(total_blocks if total_blocks is not None else len(self.wan_dit.blocks))
        if self.obj_injected_layers is None:
            return set(range(total))
        selected = {int(index) for index in self.obj_injected_layers}
        invalid = sorted(index for index in selected if index < 0 or index >= total)
        if invalid:
            raise ValueError(
                f"obj_encoder.injected_layers contains invalid zero-based layer indices: {invalid}."
            )
        return selected

    def _obj_train_block_set(self, total_blocks: int | None = None) -> set[int]:
        """Validate and return legacy object LoRA block selection."""

        total = int(total_blocks if total_blocks is not None else len(self.wan_dit.blocks))
        all_blocks = set(range(total))
        if self.obj_lora_config is None or self.obj_lora_config.train_attn_blocks is None:
            return all_blocks
        selected = {int(index) for index in self.obj_lora_config.train_attn_blocks}
        invalid = sorted(index for index in selected if index < 0 or index >= total)
        if invalid:
            raise ValueError(
                f"obj_encoder.train_attn_blocks contains invalid indices: {invalid}."
            )
        # if selected != all_blocks:
        #     raise ValueError(
        #         "Stand-In requires image-token QKV LoRA training in every Wan DiT block."
        #     )
        import warnings
        if selected != all_blocks:
            warnings.warn(
                "Stand-In requires image-token QKV LoRA training in every Wan DiT block. "
                "Continuing may cause unexpected behavior.",
                UserWarning
            )
        return selected

    def _set_obj_lora_training(self, training: bool) -> None:
        """Toggle only installed object-image LoRA modules."""

        if not self.obj_encoder_enabled:
            return
        for name, module in self.wan_dit.named_modules():
            if not isinstance(module, ImageBranchLoRA):
                continue
            block_index = parse_lora_block_index(name)
            enabled = bool(
                training
                and self.obj_encoder_trainable
                and block_index in self._obj_injected_layer_set()
            )
            module.train(enabled)
            module.set_lora_trainable(enabled)

    def _camera_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return the frozen 5120D Wan timestep embedding consumed by PCDController."""

        from wan.modules.model import sinusoidal_embedding_1d

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
            return self.wan_dit.time_embedding(
                sinusoidal_embedding_1d(self.camera_time_frequency_dim, timesteps).float()
            )

    def _encode_first_frame_condition(
        self,
        first_frame: torch.Tensor,
        frame_count: int,
        latent_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Build Wan I2V first-frame condition shaped ``[B, 20, T_lat, H_lat, W_lat]``.
        concat([B, 4, F_lat, H_lat, W_lat], [B, 16, F_lat, H_lat, W_lat]). """

        batch = int(first_frame.shape[0])
        latent_frames, latent_height, latent_width = (int(value) for value in latent_grid)

        # Mask path copied from Wan I2V:
        # mask before view: [1, T, H_lat, W_lat]
        # mask after view:  [4, T_lat, H_lat, W_lat]
        mask = torch.ones(1, frame_count, latent_height, latent_width, device=self.device)
        mask[:, 1:] = 0
        mask = torch.cat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, latent_height, latent_width).transpose(1, 2)[0]

        height = latent_height * self.vae_stride[1]
        width = latent_width * self.vae_stride[2]
        first_frame_videos = []
        for sample in normalize_image_batch(first_frame):
            resized_first = F.interpolate(
                sample.unsqueeze(0),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            blank_frames = resized_first.new_zeros(3, frame_count - 1, height, width)
            first_frame_videos.append(torch.cat([resized_first[:, None], blank_frames], dim=1))

        with torch.no_grad():
            first_frame_latents = torch.stack(
                self.vae.encode(first_frame_videos),
                dim=0,
            ).to(self.device)
        # The frozen VAE has already cleared its temporal cache. Release the
        # full mostly-zero conditioning videos before assembling latent masks.
        del first_frame_videos
        masks = mask.to(dtype=first_frame_latents.dtype).unsqueeze(0).expand(batch, -1, -1, -1, -1)
        return torch.cat([masks, first_frame_latents], dim=1)

    def encode_reference_hand_condition(
        self,
        hand_video: torch.Tensor,
        target_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the first hand frame and align it to ``image_emb[\"y\"]``.

        V1 supplies a one-frame reference feature. Expanding its temporal axis
        is numerically identical to the broadcasting performed by
        ``image_emb[\"y\"] + random_ref_dwpose_data`` and makes the resulting
        tensor shape explicit for camera-control and inference consumers.
        """

        if self.hand_encoder is None or self.reference_hand_encoder is None:
            raise RuntimeError("Reference-hand encoding is not enabled for this model.")
        if target_condition.ndim != 5 or target_condition.shape[1] != 20:
            raise ValueError(
                "target_condition must be [B,20,T,H,W], "
                f"got {tuple(target_condition.shape)}."
            )
        reference_frame = self.hand_encoder.first_frame(hand_video)
        reference_parameter = next(self.reference_hand_encoder.parameters())
        # Transfer only the selected frame. The sequence encoder independently
        # transfers its prepared channel-first tensor, so the full source hand
        # video never needs a redundant CUDA copy.
        reference_frame = reference_frame.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        reference = self.reference_hand_encoder(reference_frame)
        if reference.shape[0] != target_condition.shape[0]:
            raise ValueError(
                "Reference-hand batch does not match image condition: "
                f"{reference.shape[0]} != {target_condition.shape[0]}."
            )
        target_height, target_width = (int(value) for value in target_condition.shape[-2:])
        if tuple(reference.shape[-2:]) != (target_height, target_width):
            reference_image = F.interpolate(
                reference[:, :, 0],
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            reference = reference_image.unsqueeze(2)
        reference = reference.expand(-1, -1, int(target_condition.shape[2]), -1, -1)
        return reference.to(device=target_condition.device, dtype=target_condition.dtype)

    def _resolve_text_context(
        self,
        batch: int,
        dtype: torch.dtype,
        prompts: list[str] | None,
        text_context: list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        """Return Wan text context as a list of ``[L, text_dim]`` tensors."""

        if text_context is not None:
            return [context.to(device=self.device, dtype=dtype) for context in text_context]
        if self.text_encoder is not None and prompts is not None:
            with torch.no_grad():
                return [context.to(device=self.device, dtype=dtype) for context in self.text_encoder(prompts, self.device)]
        return [torch.zeros(1, self.wan_dit.text_dim, device=self.device, dtype=dtype) for _ in range(batch)]

    def _resolve_clip_context(
        self,
        batch: int,
        dtype: torch.dtype,
        first_frame: torch.Tensor,
        clip_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return Wan image context shaped ``[B, 257, 1280]`` for I2V cross-attention."""

        if clip_context is not None:
            return clip_context.to(device=self.device, dtype=dtype)
        if self.image_encoder is not None:
            with torch.no_grad():
                # Wan CLIP visual expects a list containing [3, 1, H, W] images.
                images = [sample[:, None] for sample in normalize_image_batch(first_frame)]
                return self.image_encoder.visual(images).to(device=self.device, dtype=dtype)
        return torch.zeros(batch, 257, 1280, device=self.device, dtype=dtype)

    def _load_text_encoder(self) -> Any | None:
        """Load the frozen Wan T5 encoder when checkpoint files are available."""

        from wan.modules.t5 import T5EncoderModel

        checkpoint = self.wan_checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer = self.wan_checkpoint_dir / "google/umt5-xxl"
        if not checkpoint.exists() or not tokenizer.exists():
            return None
        return T5EncoderModel(
            text_len=self.wan_dit.text_len,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=str(checkpoint),
            tokenizer_path=str(tokenizer),
        )

    def _load_image_encoder(self) -> Any | None:
        """Load the frozen Wan CLIP encoder when checkpoint files are available."""

        from wan.modules.clip import CLIPModel

        checkpoint = self.wan_checkpoint_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        tokenizer = self.wan_checkpoint_dir / "xlm-roberta-large"
        if not checkpoint.exists() or not tokenizer.exists():
            return None
        return CLIPModel(
            dtype=torch.float16,
            device=self.device,
            checkpoint_path=str(checkpoint),
            tokenizer_path=str(tokenizer),
        )

    def _freeze_optional_encoder(self, encoder: Any | None) -> None:
        """Freeze an optional Wan text or image encoder object."""

        if encoder is None:
            return
        model = getattr(encoder, "model", encoder)
        if isinstance(model, nn.Module):
            model.eval().requires_grad_(False)

    def _set_trainable_modules(self) -> None:
        """Make trainable/frozen status explicit after all modules are created."""

        # Never call wan_dit.requires_grad_(False) after PEFT injection: doing
        # so silently freezes the adapters as well. Freeze by identity so only
        # v1 LoRA A/B tensors remain trainable inside the DiT.
        trainable_wan_parameter_ids: set[int] = set()
        if self.hand_encoder_trainable:
            trainable_wan_parameter_ids.update(self._hand_lora_parameter_ids)
        if self.obj_encoder_trainable:
            trainable_wan_parameter_ids.update(self._obj_lora_parameter_ids)
        for parameter in self.wan_dit.parameters():
            parameter.requires_grad_(id(parameter) in trainable_wan_parameter_ids)
        self.wan_dit.train(self.hand_lora_enabled and self.hand_encoder_trainable)
        self.vae.model.eval().requires_grad_(False)
        self._freeze_optional_encoder(self.text_encoder)
        self._freeze_optional_encoder(self.image_encoder)
        for module in self.active_modules().values():
            module.eval().requires_grad_(False)
        for module in self.trainable_modules().values():
            module.train().requires_grad_(True)
        self._set_obj_lora_training(training=self.obj_encoder_trainable)


def align_hand_tokens_to_wan_tokens(
    hand_tokens: torch.Tensor,
    source_grid: tuple[int, int, int],
    target_grid_sizes: torch.Tensor,
    target_seq_len: int,
) -> torch.Tensor:
    """Resize or pad hand tokens to Wan's active padded token sequence."""

    return align_camera_residual_to_wan_tokens(
        camera_residual=hand_tokens,
        source_grid=source_grid,
        target_grid_sizes=target_grid_sizes,
        target_seq_len=target_seq_len,
    )


def align_camera_residual_to_wan_tokens(
    camera_residual: torch.Tensor,
    source_grid: tuple[int, int, int],
    target_grid_sizes: torch.Tensor,
    target_seq_len: int,
) -> torch.Tensor:
    """Resize or pad camera residual tokens to Wan's active padded token sequence."""

    if camera_residual.ndim != 3:
        raise ValueError(f"camera_residual must be [B, N, D], got {tuple(camera_residual.shape)}.")
    batch, _, dim = camera_residual.shape
    aligned_samples = []
    for sample_idx in range(batch):
        target_grid = tuple(int(value) for value in target_grid_sizes[sample_idx].tolist())
        sample_tokens = camera_residual[sample_idx]
        if source_grid != target_grid:
            sample_volume = sample_tokens.transpose(0, 1).reshape(1, dim, *source_grid)
            sample_volume = F.interpolate(sample_volume, size=target_grid, mode="trilinear", align_corners=False)
            sample_tokens = sample_volume.flatten(2).transpose(1, 2).squeeze(0)
        if sample_tokens.shape[0] < target_seq_len:
            padding = sample_tokens.new_zeros(target_seq_len - sample_tokens.shape[0], dim)
            sample_tokens = torch.cat([sample_tokens, padding], dim=0)
        else:
            sample_tokens = sample_tokens[:target_seq_len]
        aligned_samples.append(sample_tokens)
    return torch.stack(aligned_samples, dim=0)


def build_dense_plucker_rays(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build dense Plucker rays shaped ``[B, 6, T, H, W]`` from camera matrices."""

    if intrinsics.ndim != 4 or extrinsics.ndim != 4:
        raise ValueError(
            f"intrinsics/extrinsics must be [B, T, 3, 3]/[B, T, 4, 4], got {tuple(intrinsics.shape)} and {tuple(extrinsics.shape)}."
        )
    batch, frames = intrinsics.shape[:2]
    flat_intrinsics = intrinsics.reshape(batch * frames, 3, 3)
    # Uni3C makes every trajectory relative to its first view, recenters the
    # camera positions, and scales translation before constructing Plucker
    # rays. Apply that operation independently per batch item so trajectories
    # from different clips cannot affect one another.
    normalized_extrinsics = torch.stack(
        [camera_center_normalization(sample, nframe=frames, camera_scale=2.0) for sample in extrinsics],
        dim=0,
    )
    flat_extrinsics = normalized_extrinsics.reshape(batch * frames, 4, 4)

    # Pixel coordinates: [H*W, 3] homogeneous image-plane points.
    y = torch.arange(height, device=intrinsics.device, dtype=torch.float32) + 0.5
    x = torch.arange(width, device=intrinsics.device, dtype=torch.float32) - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)

    # Camera rays:
    # directions_camera: [B*T, H*W, 3]
    # c2w:               [B*T, 3, 4]
    # ray directions and origins are then expressed in the supplied trajectory frame.
    c2w = torch.linalg.inv(flat_extrinsics)[:, :3, :4]
    directions_camera = pixels.unsqueeze(0) @ torch.linalg.inv(flat_intrinsics).transpose(-1, -2)
    ray_directions = F.normalize(directions_camera @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)
    ray_origins = c2w[:, None, :3, 3].expand_as(ray_directions)
    ray_moments = torch.cross(ray_origins, ray_directions, dim=-1)

    # camera embedding order is [direction, origin x direction].
    plucker = torch.cat([ray_directions, ray_moments], dim=-1)
    plucker = plucker.reshape(batch, frames, height, width, 6)
    return plucker.permute(0, 4, 1, 2, 3).contiguous()


def normalize_video_in_place(video: torch.Tensor) -> torch.Tensor:
    """Normalize a private floating-point video tensor without another allocation."""

    if not video.is_floating_point():
        raise TypeError(f"video must be floating point, got {video.dtype}.")
    if video.max() <= 1.0 and video.min() >= 0.0:
        video.mul_(2.0).sub_(1.0)
    return video


def normalize_video_batch(video: torch.Tensor) -> torch.Tensor:
    """Return ``[B, T, 3, H, W]`` video in Wan VAE ``[-1, 1]`` range."""

    if video.max() <= 1.0 and video.min() >= 0.0:
        return video.mul(2.0).sub(1.0)
    return video


def normalize_rendered_video(rendered_video: torch.Tensor) -> torch.Tensor:
    """Return ``[B, 3, T, H, W]`` rendered video in Wan VAE ``[-1, 1]`` range."""

    if rendered_video.max() <= 1.0 and rendered_video.min() >= 0.0:
        return rendered_video.mul(2.0).sub(1.0)
    return rendered_video


def normalize_image_batch(image: torch.Tensor) -> torch.Tensor:
    """Return ``[B, 3, H, W]`` images in Wan encoder ``[-1, 1]`` range."""

    if image.max() <= 1.0 and image.min() >= 0.0:
        return image.mul(2.0).sub(1.0)
    return image


def choose_attention_heads(channels: int) -> int:
    """Choose a valid multi-head attention count for camera-control width."""

    for heads in (16, 8, 4, 2, 1):
        if channels % heads == 0:
            return heads
    return 1
