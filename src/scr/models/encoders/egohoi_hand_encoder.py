"""EgoHOI hand-video encoder mirrored from the legacy hand-training script.

This module is intentionally self-contained.  It does not import the legacy
training script or any other hand encoder, so it can be trained, checkpointed,
and reused as a normal model component.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


class EgoHOIHandEncoder(nn.Module):
    """Convert rendered RGB hand videos into Wan-width spatiotemporal tokens.

    Architecture (the same convolution stack as
    ``v1_based_modifyv3_seplr_web_hand.py``):

    1. Prepend three copies of the first frame.  Wan's VAE maps 81 RGB frames
       to 21 latent frames; changing 81 to 84 before two stride-2 temporal
       convolutions makes the hand branch produce the same 21-frame grid.
    2. Apply three 3x3x3 Conv3d + SiLU blocks at full resolution.
    3. Downsample space by 2, then time and space by 2 twice.
    4. Project and downsample space once more with a 1x2x2 convolution.

    For the standard input ``[B, 3, 81, 480, 480]``, the shape path is:

    ``[B, 3, 84, 480, 480] -> [B, 16, 84, 240, 240]``
    ``-> [B, 16, 42, 120, 120] -> [B, 16, 21, 60, 60]``
    ``-> [B, 5120, 21, 30, 30] -> [B, 18900, 5120]``.

    Args:
        embedding_dim: Output token width.  Wan 2.1 I2V-14B uses 5120.
        concat_dim: Legacy base channel multiplier; ``4`` gives 16 hidden
            channels and exactly reproduces the original architecture.
    """

    def __init__(self, embedding_dim: int = 5120, concat_dim: int = 4) -> None:
        """Build the legacy seven-convolution hand embedding network."""

        super().__init__()
        hidden_channels = int(concat_dim) * 4
        self.embedding_dim = int(embedding_dim)
        self.concat_dim = int(concat_dim)

        # Keep this Sequential ordering stable so old ``dwpose_embedding.N``
        # checkpoint tensors can be mapped to ``encoder.N`` without ambiguity.
        self.encoder = nn.Sequential(
            nn.Conv3d(3, hidden_channels, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(
                hidden_channels,
                self.embedding_dim,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                padding=0,
            ),
        )

    def forward(self, hand_video: torch.Tensor) -> torch.Tensor:
        """Return flattened hand tokens shaped ``[B, N_hand, embedding_dim]``."""

        tokens, _ = self.encode_tokens(hand_video)
        return tokens

    def encode_volume(self, hand_video: torch.Tensor) -> torch.Tensor:
        """Return the 3D hand embedding volume ``[B, embedding_dim, F, H, W]``."""

        prepared_video = self.prepare_hand_video(hand_video)
        parameter = next(self.parameters())
        prepared_video = prepared_video.to(device=parameter.device, dtype=parameter.dtype)
        return self.encoder(prepared_video)

    def encode_tokens(
        self,
        hand_video: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Encode and flatten a hand video, returning tokens and its ``(F,H,W)`` grid."""

        # hand_volume: [B, embedding_dim, F_hand, H_hand, W_hand].
        hand_volume = self.encode_volume(hand_video)
        token_grid = tuple(int(size) for size in hand_volume.shape[2:])

        # hand_tokens: [B, F_hand * H_hand * W_hand, embedding_dim].
        hand_tokens = hand_volume.flatten(2).transpose(1, 2).contiguous()
        return hand_tokens, token_grid

    def prepare_hand_video(self, hand_video: torch.Tensor) -> torch.Tensor:
        """Normalize layout/range and prepend the three legacy alignment frames.

        Accepted layouts are ``[B,3,T,H,W]``, ``[B,T,3,H,W]``, and their
        unbatched equivalents.  Integer or ``[0,255]`` floating-point pixels
        are scaled to ``[0,1]``; already normalized ``[0,1]`` tensors are kept.
        """

        hand_video = self._as_bcthw(hand_video)
        hand_video = hand_video.float()
        if hand_video.numel() and hand_video.detach().amax() > 1.0:
            hand_video = hand_video / 255.0

        # Legacy logic: [B,3,T,H,W] -> [B,3,T+3,H,W].
        first_frame_padding = hand_video[:, :, :1].repeat(1, 1, 3, 1, 1)
        return torch.cat([first_frame_padding, hand_video], dim=2).contiguous()

    def first_frame(self, hand_video: torch.Tensor) -> torch.Tensor:
        """Return the first rendered hand frame in ``[B, 3, H, W]`` layout.

        The proven v1 path uses the first pose frame as
        ``random_ref_dwpose_data``. Keeping extraction beside the sequence
        layout conversion guarantees that both hand branches see the same
        frame even when the dataset supplies ``[B,T,3,H,W]`` tensors.
        """

        return self._as_bcthw(hand_video)[:, :, 0].contiguous()

    def load_legacy_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> nn.modules.module._IncompatibleKeys:
        """Load tensors saved under the legacy ``dwpose_embedding`` prefix."""

        converted: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = key
            for prefix in ("module.", "model."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            if normalized_key.startswith("dwpose_embedding."):
                normalized_key = "encoder." + normalized_key.removeprefix("dwpose_embedding.")
            elif normalized_key and normalized_key[0].isdigit():
                normalized_key = "encoder." + normalized_key
            if normalized_key.startswith("encoder."):
                converted[normalized_key] = value
        return self.load_state_dict(converted, strict=strict)

    @staticmethod
    def _as_bcthw(hand_video: torch.Tensor) -> torch.Tensor:
        """Convert supported batched/unbatched hand layouts to ``[B,3,T,H,W]``."""

        if hand_video.ndim == 4:
            # [3,T,H,W] or [T,3,H,W].
            if hand_video.shape[0] == 3:
                hand_video = hand_video.unsqueeze(0)
            elif hand_video.shape[1] == 3:
                hand_video = hand_video.permute(1, 0, 2, 3).unsqueeze(0)
            else:
                raise ValueError(
                    "Unbatched hand_video must be [3,T,H,W] or [T,3,H,W], "
                    f"got {tuple(hand_video.shape)}."
                )
        elif hand_video.ndim == 5 and hand_video.shape[1] != 3 and hand_video.shape[2] == 3:
            # Dataset batches use [B,T,3,H,W].
            hand_video = hand_video.permute(0, 2, 1, 3, 4)

        if hand_video.ndim != 5 or hand_video.shape[1] != 3:
            raise ValueError(
                "hand_video must be [B,3,T,H,W] or [B,T,3,H,W], "
                f"got {tuple(hand_video.shape)}."
            )
        if hand_video.shape[2] < 1:
            raise ValueError("hand_video must contain at least one frame.")
        return hand_video.contiguous()


class EgoHOIReferenceHandEncoder(nn.Module):
    """Encode v1's reference hand frame into Wan's 20-channel I2V condition.

    This is the exact ``randomref_embedding_pose`` Conv2d stack from
    ``v1_based_modifyv3_seplr_web.py``. Its singleton temporal dimension is
    broadcast across ``image_emb[\"y\"]`` before Wan patchifies the combined
    noisy-latent and image-condition tensor.
    """

    def __init__(self, out_channels: int = 20, concat_dim: int = 4) -> None:
        """Build the six-convolution v1 reference-hand encoder."""

        super().__init__()
        self.out_channels = int(out_channels)
        self.concat_dim = int(concat_dim)
        hidden_channels = self.concat_dim * 4

        # Keep this ordering identical to v1 so
        # ``randomref_embedding_pose.N`` checkpoints map one-to-one.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, reference_hand: torch.Tensor) -> torch.Tensor:
        """Return a reference condition shaped ``[B, 20, 1, H/8, W/8]``."""

        prepared = self.prepare_reference_hand(reference_hand)
        parameter = next(self.parameters())
        prepared = prepared.to(device=parameter.device, dtype=parameter.dtype)
        return self.encoder(prepared).unsqueeze(2)

    def prepare_reference_hand(self, reference_hand: torch.Tensor) -> torch.Tensor:
        """Normalize BCHW/BHWC reference images to float ``[B,3,H,W]``."""

        if reference_hand.ndim == 3:
            reference_hand = reference_hand.unsqueeze(0)
        if reference_hand.ndim != 4:
            raise ValueError(
                "reference_hand must be [B,3,H,W], [B,H,W,3], [3,H,W], "
                f"or [H,W,3], got {tuple(reference_hand.shape)}."
            )
        if reference_hand.shape[1] != 3 and reference_hand.shape[-1] == 3:
            reference_hand = reference_hand.permute(0, 3, 1, 2)
        if reference_hand.shape[1] != 3:
            raise ValueError(
                "reference_hand must contain three RGB channels, "
                f"got {tuple(reference_hand.shape)}."
            )
        reference_hand = reference_hand.float()
        if reference_hand.numel() and reference_hand.detach().amax() > 1.0:
            reference_hand = reference_hand / 255.0
        return reference_hand.contiguous()

    def load_legacy_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> nn.modules.module._IncompatibleKeys:
        """Load tensors stored under v1's ``randomref_embedding_pose`` name."""

        converted: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = key
            for prefix in ("module.", "model."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            if normalized_key.startswith("randomref_embedding_pose."):
                normalized_key = "encoder." + normalized_key.removeprefix(
                    "randomref_embedding_pose."
                )
            elif normalized_key and normalized_key[0].isdigit():
                normalized_key = "encoder." + normalized_key
            if normalized_key.startswith("encoder."):
                converted[normalized_key] = value
        return self.load_state_dict(converted, strict=strict)
