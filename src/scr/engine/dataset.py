"""Dataset utilities for camera-encoder training on preprocessed HOT3D clips.

The current implementation targets ``outputs/hot3d_preprocessed`` and keeps the
sample schema explicit so future datasets can be added beside HOT3D without
changing the trainer interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CameraTrainingSample:
    """Single camera-training item before dataloader collation."""

    first_frame: torch.Tensor
    raw_video: torch.Tensor
    intrinsic: torch.Tensor
    extrinsic: torch.Tensor
    rendered_novel_views: torch.Tensor
    rendered_novel_view_masks: torch.Tensor
    rendered_hands: torch.Tensor
    rendered_object: torch.Tensor
    clip_name: str


class Hot3DPreprocessedDataset(Dataset):
    """Load preprocessed HOT3D clips produced by ``scripts/preprocess.py``."""

    def __init__(
        self,
        root: str | Path = "outputs/hot3d_preprocessed",
        clip_names: list[str] | None = None,
        num_frames: int = 81,
        image_size: int | tuple[int, int] | None = 480,
        stream_id: str = "214-1",
        load_rendered_views: bool = True,
        load_hands: bool = True,
        require_hands: bool = False,
    ) -> None:
        """Discover clips and store frame-loading settings."""

        self.root = Path(root)
        self.clip_names = clip_names
        self.num_frames = int(num_frames)
        self.image_size = normalize_image_size(image_size)
        self.stream_id = stream_id
        self.load_rendered_views = bool(load_rendered_views)
        self.load_hands = bool(load_hands)
        self.require_hands = bool(require_hands)
        self.clip_dirs = self._discover_clip_dirs()
        if not self.clip_dirs:
            raise FileNotFoundError(f"No preprocessed HOT3D clips found under {self.root}.")

    def __len__(self) -> int:
        """Return the number of discovered preprocessed clips."""

        return len(self.clip_dirs)

    def __getitem__(self, index: int) -> CameraTrainingSample:
        """Load one clip with RGB, rendered views, masks, and camera matrices."""

        clip_dir = self.clip_dirs[index % len(self.clip_dirs)]
        raw_video = self._load_frame_sequence(clip_dir / "undistortion" / f"rgb_undistorted_{self.stream_id}", mode="rgb")
        frame_count, _, height, width = raw_video.shape
        if self.load_rendered_views:
            rendered_novel_views = self._load_frame_sequence(clip_dir / "view_render" / "render_frames", mode="rgb")
            rendered_novel_view_masks = self._load_frame_sequence(clip_dir / "view_render" / "render_mask_frames", mode="mask")
        else:
            # Hand training does not consume novel-view supervision, so skipping
            # these images cuts per-sample IO while preserving the shared schema.
            rendered_novel_views = torch.zeros_like(raw_video)
            rendered_novel_view_masks = torch.zeros(frame_count, 1, height, width, dtype=raw_video.dtype)
        intrinsic, extrinsic = self._load_camera_info(clip_dir / f"camera_info_{self.stream_id}.json")
        hand_dir = clip_dir / "hand"
        if self.load_hands and self.require_hands:
            rendered_hands = self._load_frame_sequence(hand_dir, mode="rgb").to(dtype=raw_video.dtype)
        elif self.load_hands:
            rendered_hands = self._load_optional_frame_sequence(
                hand_dir,
                mode="rgb",
                fallback_shape=(frame_count, 3, height, width),
                dtype=raw_video.dtype,
            )
        else:
            rendered_hands = torch.zeros_like(raw_video)
        rendered_object = torch.zeros(frame_count, 1, height, width, dtype=raw_video.dtype)
        return CameraTrainingSample(
            first_frame=raw_video[0],
            raw_video=raw_video,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            rendered_novel_views=rendered_novel_views,
            rendered_novel_view_masks=rendered_novel_view_masks,
            rendered_hands=rendered_hands,
            rendered_object=rendered_object,
            clip_name=clip_dir.name,
        )

    def _discover_clip_dirs(self) -> list[Path]:
        """Return clip directories containing the required HOT3D outputs."""

        candidates = [self.root / name for name in self.clip_names] if self.clip_names else sorted(self.root.glob("clip-*"))
        clip_dirs = []
        for path in candidates:
            if not path.is_dir():
                continue
            # Camera extraction is independent from undistortion outputs and
            # exposes one consolidated file directly in each clip directory.
            camera_info = path / f"camera_info_{self.stream_id}.json"
            rgb_dir = path / "undistortion" / f"rgb_undistorted_{self.stream_id}"
            render_dir = path / "view_render" / "render_frames"
            has_render_inputs = render_dir.is_dir() or not self.load_rendered_views
            if camera_info.exists() and rgb_dir.is_dir() and has_render_inputs:
                clip_dirs.append(path)
        return clip_dirs

    def _load_frame_sequence(self, frame_dir: Path, mode: str) -> torch.Tensor:
        """Load a fixed-length image sequence as ``[F, C, H, W]`` float tensors."""

        if not frame_dir.is_dir():
            raise FileNotFoundError(f"Frame directory is missing: {frame_dir}")
        frame_paths = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in {frame_dir}.")
        selected = frame_paths[: self.num_frames]
        if len(selected) < self.num_frames:
            selected.extend([selected[-1]] * (self.num_frames - len(selected)))
        frames = [load_image_tensor(path, mode=mode, image_size=self.image_size) for path in selected]
        return torch.stack(frames, dim=0)

    def _load_optional_frame_sequence(
        self,
        frame_dir: Path,
        mode: str,
        fallback_shape: tuple[int, int, int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load optional frames or return zeros when that preprocessing stage is absent."""

        if not frame_dir.is_dir():
            return torch.zeros(*fallback_shape, dtype=dtype)
        frame_paths = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not frame_paths:
            return torch.zeros(*fallback_shape, dtype=dtype)
        selected = frame_paths[: self.num_frames]
        if len(selected) < self.num_frames:
            selected.extend([selected[-1]] * (self.num_frames - len(selected)))
        frames = [load_image_tensor(path, mode=mode, image_size=self.image_size) for path in selected]
        return torch.stack(frames, dim=0).to(dtype=dtype)

    def _load_camera_info(self, camera_info_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Read intrinsic and extrinsic matrices from the preprocessed JSON file."""

        with camera_info_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        extrinsic = torch.as_tensor(payload["extrinsic"], dtype=torch.float32)
        intrinsic_payload = payload["intrinsic"]
        intrinsic = torch.as_tensor(intrinsic_payload, dtype=torch.float32)
        if intrinsic.ndim == 2:
            intrinsic = intrinsic.unsqueeze(0).repeat(extrinsic.shape[0], 1, 1)
        intrinsic = intrinsic[: self.num_frames]
        extrinsic = extrinsic[: self.num_frames]
        if intrinsic.shape[0] < self.num_frames:
            intrinsic = pad_time(intrinsic, self.num_frames)
        if extrinsic.shape[0] < self.num_frames:
            extrinsic = pad_time(extrinsic, self.num_frames)
        return intrinsic, extrinsic


class PlaceholderCameraDataset(Dataset):
    """Small synthetic dataset used when no HOT3D files are available."""

    def __init__(self, length: int = 4, num_frames: int = 8, image_size: int = 64) -> None:
        """Store deterministic placeholder dimensions for smoke training."""

        self.length = int(length)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        """Return the configured placeholder length."""

        return self.length

    def __getitem__(self, index: int) -> CameraTrainingSample:
        """Create one deterministic synthetic camera-training sample."""

        coords = torch.linspace(0.0, 1.0, self.image_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        frames = []
        for frame_idx in range(self.num_frames):
            phase = frame_idx / max(1, self.num_frames - 1)
            frame = torch.stack([torch.remainder(xx + phase, 1.0), yy, torch.full_like(xx, index / max(1, self.length))])
            frames.append(frame)
        raw_video = torch.stack(frames, dim=0)
        masks = torch.ones(self.num_frames, 1, self.image_size, self.image_size)
        intrinsic = torch.eye(3).repeat(self.num_frames, 1, 1)
        intrinsic[:, 0, 0] = intrinsic[:, 1, 1] = float(self.image_size)
        intrinsic[:, 0, 2] = intrinsic[:, 1, 2] = float(self.image_size) / 2.0
        extrinsic = torch.eye(4).repeat(self.num_frames, 1, 1)
        return CameraTrainingSample(raw_video[0], raw_video, intrinsic, extrinsic, raw_video, masks, masks * 0.0, masks * 0.0, f"placeholder-{index:04d}")


def build_camera_dataset(config: dict[str, Any]) -> Dataset:
    """Build the configured camera-training dataset, currently HOT3D or placeholder."""

    dataset_name = str(config.get("name", "hot3d")).lower()
    if dataset_name in {"hot3d", "hot3d_preprocessed"}:
        return Hot3DPreprocessedDataset(
            root=config.get("root", "outputs/hot3d_preprocessed"),
            clip_names=config.get("clip_names"),
            num_frames=int(config.get("num_frames", 81)),
            image_size=config.get("image_size", config.get("resolution", 480)),
            stream_id=str(config.get("stream_id", "214-1")),
            load_rendered_views=bool(config.get("load_rendered_views", True)),
            load_hands=bool(config.get("load_hands", True)),
            require_hands=bool(config.get("require_hands", False)),
        )
    if dataset_name in {"placeholder", "synthetic"}:
        return PlaceholderCameraDataset(
            length=int(config.get("length", 4)),
            num_frames=int(config.get("num_frames", 8)),
            image_size=int(config.get("image_size", config.get("resolution", 64))),
        )
    raise ValueError(f"Unsupported camera dataset: {dataset_name}.")


def collate_camera_training_samples(samples: list[CameraTrainingSample]) -> dict[str, Any]:
    """Stack camera-training samples into the trainer batch dictionary."""

    return {
        "first_frame": torch.stack([sample.first_frame for sample in samples], dim=0),
        "raw_video": torch.stack([sample.raw_video for sample in samples], dim=0),
        "video": torch.stack([sample.raw_video for sample in samples], dim=0),
        "intrinsic": torch.stack([sample.intrinsic for sample in samples], dim=0),
        "extrinsic": torch.stack([sample.extrinsic for sample in samples], dim=0),
        "rendered_novel_views": torch.stack([sample.rendered_novel_views for sample in samples], dim=0),
        "rendered_novel_view_masks": torch.stack([sample.rendered_novel_view_masks for sample in samples], dim=0),
        "rendered_hands": torch.stack([sample.rendered_hands for sample in samples], dim=0),
        "rendered_object": torch.stack([sample.rendered_object for sample in samples], dim=0),
        "clip_name": [sample.clip_name for sample in samples],
    }


def load_image_tensor(path: Path, mode: str, image_size: tuple[int, int] | None) -> torch.Tensor:
    """Load one image with PIL and return a normalized CHW float tensor."""

    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        if mode == "mask":
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        if image_size is not None:
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            image = image.resize((image_size[1], image_size[0]), resampling)
        # NumPy keeps image decoding in C and avoids building one Python int per
        # pixel, which is a major bottleneck for 81-frame 480px training clips.
        array = np.array(image, copy=True)
        tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        tensor = tensor.permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor


def normalize_image_size(image_size: int | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    """Normalize config image-size values to ``(height, width)`` or ``None``."""

    if image_size is None:
        return None
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError(f"image_size must be an int or pair, got {image_size}.")
    return (int(image_size[0]), int(image_size[1]))


def pad_time(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Pad a time-major tensor by repeating its final item."""

    if tensor.shape[0] >= length:
        return tensor[:length]
    padding = tensor[-1:].repeat(length - tensor.shape[0], *([1] * (tensor.ndim - 1)))
    return torch.cat((tensor, padding), dim=0)


def resize_video_to_latent(video: torch.Tensor, latent_size: tuple[int, int]) -> torch.Tensor:
    """Resize ``[B, F, C, H, W]`` video to latent spatial size and return ``[B, C, F, H, W]``."""

    batch, frames, channels, _, _ = video.shape
    flat = video.reshape(batch * frames, channels, video.shape[-2], video.shape[-1])
    resized = F.interpolate(flat, size=latent_size, mode="bilinear", align_corners=False)
    return resized.reshape(batch, frames, channels, *latent_size).permute(0, 2, 1, 3, 4).contiguous()
