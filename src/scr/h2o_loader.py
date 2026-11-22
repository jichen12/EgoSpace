"""Load prepared H2O control windows and staged object-reference images.

The loader mirrors the batch schema used by :mod:`scr.hot3d_loader` so H2O
steps can be interleaved with HOT3D without changing the model forward path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scr.hot3d_loader import (
    DEFAULT_CAPTION,
    Hot3DControlDataset,
    Hot3DControlSample,
    ObjectRopeSample,
    _load_normalized_video_frames,
    _normalize_image_size,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("outputs/h2o_preprocessed")
DEFAULT_RAW_ROOT = DEFAULT_ROOT
DEFAULT_CAPTION_PATH = DEFAULT_ROOT / "h2o_action_label_captions.json"
DEFAULT_TRAIN_SUBJECTS = ("subject1_ego", "subject2_ego", "subject3_ego")
DEFAULT_VAL_SUBJECTS = ("subject4_ego",)
H2O_SOURCE_WIDTH = 1280
H2O_SOURCE_HEIGHT = 720
H2O_CROP_SIZE = 720
H2O_INTRINSIC_VALUE_COUNT = 6
H2O_POSE_VALUE_COUNT = 16


@dataclass(frozen=True)
class H2OControlWindow:
    """Describe one complete H2O RGB/control interval."""

    clip_name: str
    start_frame: int
    end_frame: int
    video_path: Path
    rendered_video_path: Path
    rendered_mask_path: Path
    hand_path: Path
    raw_cam_dir: Path


@dataclass(frozen=True)
class H2OObjectFrame:
    """Describe one rendered H2O object observation."""

    frame_index: int
    object_index: int
    object_name: str
    image_path: Path


@dataclass(frozen=True)
class H2OObjectWindow:
    """Attach a fixed training interval to its clip-wide object view bank."""

    metadata_path: Path
    video_path: Path
    clip_name: str
    start_frame: int
    end_frame: int
    in_interval_views: tuple[H2OObjectFrame, ...]
    out_of_interval_views: tuple[H2OObjectFrame, ...]


def _resolve_path(path: str | Path) -> Path:
    """Resolve repository-relative configuration and metadata paths."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPOSITORY_ROOT / candidate


def _natural_sort_key(path: Path) -> tuple[Any, ...]:
    """Return a stable numeric-aware key for H2O clip paths."""

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.as_posix())
    )


def _normalize_start_frames(
    start_frames: int | str | Sequence[int] | None,
) -> tuple[int, ...]:
    """Normalize H2O starts without imposing a maximum video frame."""

    if start_frames is None:
        values = [0]
    elif isinstance(start_frames, str):
        values = [int(value.strip()) for value in start_frames.split(",") if value.strip()]
    elif isinstance(start_frames, int):
        values = [int(start_frames)]
    else:
        values = [int(value) for value in start_frames]
    if not values:
        raise ValueError("H2O start_frames must contain at least one frame index.")
    if min(values) < 0:
        raise ValueError(f"H2O start_frames cannot be negative, got {values}.")
    return tuple(dict.fromkeys(values))


def _load_caption_index(path: str | Path | None) -> dict[tuple[str, int, int], str]:
    """Load H2O captions by source clip and exact inclusive frame range."""

    if path is None:
        return {}
    caption_path = _resolve_path(path)
    if not caption_path.is_file():
        return {}
    with caption_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON caption mapping in {caption_path}.")

    captions: dict[tuple[str, int, int], str] = {}
    for key, entry in payload.items():
        if not isinstance(entry, dict):
            raise ValueError(f"H2O caption entry {key!r} must be an object.")
        source_clip = str(entry.get("source_clip", "")).strip().strip("/")
        frame_range = entry.get("frame_range")
        caption = str(entry.get("caption", "")).strip()
        if (
            not source_clip
            or not isinstance(frame_range, (list, tuple))
            or len(frame_range) != 2
        ):
            raise ValueError(f"H2O caption entry {key!r} is incomplete.")
        # Some no-action intervals intentionally carry an empty caption; keep
        # those on the configured generic prompt instead of failing discovery.
        if caption:
            captions[(source_clip, int(frame_range[0]), int(frame_range[1]))] = caption
    return captions


def _caption_for_window(
    captions: dict[tuple[str, int, int], str],
    clip_name: str,
    start_frame: int,
    video_frames: int,
    fallback: str,
) -> str:
    """Return an exact-window H2O caption or the configured fallback."""

    end_frame = int(start_frame) + int(video_frames) - 1
    return captions.get((clip_name, int(start_frame), end_frame), fallback)


def _load_text_vector(path: Path, expected_size: int) -> np.ndarray:
    """Load and validate one whitespace-delimited H2O calibration row."""

    values = np.loadtxt(path, dtype=np.float32).reshape(-1)
    if values.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} values in {path}, got {values.size}."
        )
    return values


def _load_h2o_camera_window(
    raw_cam_dir: Path,
    start_frame: int,
    frame_count: int,
    image_size: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load crop-adjusted intrinsics and world-to-camera H2O poses."""

    fx, fy, cx, cy, width, height = _load_text_vector(
        raw_cam_dir / "cam_intrinsics.txt", H2O_INTRINSIC_VALUE_COUNT
    )
    if (int(round(float(width))), int(round(float(height)))) != (
        H2O_SOURCE_WIDTH,
        H2O_SOURCE_HEIGHT,
    ):
        raise ValueError(
            f"Expected H2O calibration for 1280x720 at {raw_cam_dir}, "
            f"got {width}x{height}."
        )
    output_height, output_width = image_size or (H2O_CROP_SIZE, H2O_CROP_SIZE)
    scale_x = float(output_width) / H2O_CROP_SIZE
    scale_y = float(output_height) / H2O_CROP_SIZE
    crop_left = (H2O_SOURCE_WIDTH - H2O_CROP_SIZE) / 2.0
    intrinsic = torch.tensor(
        [
            [float(fx) * scale_x, 0.0, (float(cx) - crop_left) * scale_x],
            [0.0, float(fy) * scale_y, float(cy) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    camera_to_worlds = []
    for frame_index in range(int(start_frame), int(start_frame) + int(frame_count)):
        pose_path = raw_cam_dir / "cam_pose" / f"{frame_index:06d}.txt"
        camera_to_worlds.append(
            torch.from_numpy(
                _load_text_vector(pose_path, H2O_POSE_VALUE_COUNT).reshape(4, 4)
            )
        )
    camera_to_world = torch.stack(camera_to_worlds, dim=0).float()
    extrinsic = torch.linalg.inv(camera_to_world)
    intrinsics = intrinsic.unsqueeze(0).repeat(int(frame_count), 1, 1)
    return intrinsics, extrinsic


def _center_crop_resize_video(
    video: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Center-crop decoded H2O frames to a square and resize to ``image_size``."""

    if video.ndim != 4:
        raise ValueError(
            f"Expected H2O video tensor [F, C, H, W], got {tuple(video.shape)}."
        )
    height, width = int(video.shape[-2]), int(video.shape[-1])
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    cropped = video[..., top : top + crop_size, left : left + crop_size]
    if (crop_size, crop_size) == tuple(int(value) for value in image_size):
        return cropped.contiguous()
    resized = F.interpolate(
        cropped,
        size=(int(image_size[0]), int(image_size[1])),
        mode="bilinear",
        align_corners=False,
    )
    return resized.contiguous()


def _center_crop_pil_image(image: Image.Image) -> Image.Image:
    """Return the centered square crop of one loaded H2O RGB image."""

    width, height = image.size
    crop_size = min(int(width), int(height))
    left = (int(width) - crop_size) // 2
    top = (int(height) - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def _load_h2o_video_frames(
    path: str | Path,
    frame_count: int,
    image_size: tuple[int, int],
    start_frame: int = 0,
) -> torch.Tensor:
    """Load H2O video frames with square center-crop before fixed-size resize."""

    video = _load_normalized_video_frames(
        path,
        frame_count=frame_count,
        image_size=None,
        start_frame=start_frame,
    )
    return _center_crop_resize_video(video, image_size)


def _load_h2o_image_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Load one H2O reference image with square center-crop and resize."""

    with Image.open(path) as image:
        image = _center_crop_pil_image(image.convert("RGB"))
        resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        image = image.resize((int(image_size[1]), int(image_size[0])), resampling)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


class H2OControlDataset(Dataset[Hot3DControlSample]):
    """Load H2O windows using the shared HOT3D control sample schema."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        raw_root: str | Path = DEFAULT_RAW_ROOT,
        subjects: Sequence[str] = DEFAULT_TRAIN_SUBJECTS,
        video_frames: int = 81,
        image_size: int | Sequence[int] | None = 480,
        start_frames: int | str | Sequence[int] | None = None,
        caption: str = DEFAULT_CAPTION,
        caption_annotations_path: str | Path | None = DEFAULT_CAPTION_PATH,
        load_rendered_views: bool = True,
        load_hands: bool = True,
        require_hands: bool = True,
        length: int = 0,
    ) -> None:
        """Store H2O settings and discover complete windows in requested order."""

        self.root = _resolve_path(root)
        self.raw_root = _resolve_path(raw_root)
        self.subjects = tuple(str(subject) for subject in subjects)
        self.video_frames = int(video_frames)
        if self.video_frames <= 0:
            raise ValueError("H2O video_frames must be positive.")
        self.image_size = _normalize_image_size(image_size)
        self.start_frames = _normalize_start_frames(start_frames)
        self.caption = str(caption)
        self.caption_index = _load_caption_index(caption_annotations_path)
        self.load_rendered_views = bool(load_rendered_views)
        self.load_hands = bool(load_hands)
        self.require_hands = bool(require_hands)
        self.requested_length = int(length)
        self.items = self._discover_items()
        if self.requested_length > 0:
            self.items = self.items[: self.requested_length]
        if not self.items:
            raise FileNotFoundError(
                f"No complete H2O windows found under {self.root} for "
                f"subjects={list(self.subjects)} and start_frames={list(self.start_frames)}."
            )

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> H2OControlDataset:
        """Build an H2O control dataset from the training configuration."""

        return cls(
            root=data_config.get("root", DEFAULT_ROOT),
            raw_root=data_config.get("raw_root", DEFAULT_RAW_ROOT),
            subjects=data_config.get("subjects", DEFAULT_TRAIN_SUBJECTS),
            video_frames=int(data_config.get("video_frames", 81)),
            image_size=data_config.get("resolution", data_config.get("image_size", 480)),
            start_frames=data_config.get("start_frames"),
            caption=str(data_config.get("caption", DEFAULT_CAPTION)),
            caption_annotations_path=data_config.get(
                "caption_annotations_path", DEFAULT_CAPTION_PATH
            ),
            load_rendered_views=bool(data_config.get("load_rendered_views", True)),
            load_hands=bool(data_config.get("load_hands", True)),
            require_hands=bool(data_config.get("require_hands", True)),
            length=int(data_config.get("length", 0) or 0),
        )

    def __len__(self) -> int:
        """Return the number of selected complete H2O windows."""

        return len(self.items)

    def __getitem__(self, index: int) -> Hot3DControlSample:
        """Decode one H2O control window."""

        item = self.items[index % len(self.items)]
        raw_video = _load_h2o_video_frames(
            item.video_path,
            frame_count=self.video_frames,
            image_size=self.image_size or (480, 480),
            start_frame=item.start_frame,
        )
        if self.load_rendered_views:
            rendered_views = _load_h2o_video_frames(
                item.rendered_video_path,
                frame_count=self.video_frames,
                image_size=self.image_size or (480, 480),
                start_frame=0,
            ).to(dtype=raw_video.dtype)
            rendered_mask_rgb = _load_h2o_video_frames(
                item.rendered_mask_path,
                frame_count=self.video_frames,
                image_size=self.image_size or (480, 480),
                start_frame=0,
            ).to(dtype=raw_video.dtype)
            rendered_masks = rendered_mask_rgb.mean(dim=1, keepdim=True)
            intrinsic, extrinsic = _load_h2o_camera_window(
                item.raw_cam_dir,
                item.start_frame,
                self.video_frames,
                self.image_size,
            )
        else:
            rendered_views = None
            rendered_masks = None
            intrinsic = None
            extrinsic = None

        if self.load_hands and item.hand_path.is_file():
            rendered_hands = _load_h2o_video_frames(
                item.hand_path,
                frame_count=self.video_frames,
                image_size=self.image_size or (480, 480),
                start_frame=item.start_frame,
            ).to(dtype=raw_video.dtype)
        elif self.load_hands and self.require_hands:
            raise FileNotFoundError(f"Required H2O hand video is missing: {item.hand_path}")
        else:
            rendered_hands = None

        caption = _caption_for_window(
            self.caption_index,
            item.clip_name,
            item.start_frame,
            self.video_frames,
            self.caption,
        )
        return Hot3DControlSample(
            first_frame=raw_video[0],
            raw_video=raw_video,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            rendered_novel_views=rendered_views,
            rendered_novel_view_masks=rendered_masks,
            rendered_hands=rendered_hands,
            rendered_object=None,
            clip_name=item.clip_name,
            caption=caption,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            video_path=str(item.video_path),
            rendered_video_path=str(item.rendered_video_path),
            rendered_mask_path=str(item.rendered_mask_path),
            hand_path=str(item.hand_path),
            camera_info_path=str(item.raw_cam_dir),
        )

    @staticmethod
    def collate_samples(samples: Sequence[Hot3DControlSample]) -> dict[str, Any]:
        """Collate H2O controls with the shared control batch implementation."""

        return Hot3DControlDataset.collate_samples(samples)

    def describe_item_files(self, index: int) -> dict[str, Any]:
        """Describe one H2O window without decoding tensors."""

        item = self.items[index % len(self.items)]
        return {
            "dataset": "h2o",
            "clip_name": item.clip_name,
            "interval": [item.start_frame, item.end_frame],
            "caption": _caption_for_window(
                self.caption_index,
                item.clip_name,
                item.start_frame,
                self.video_frames,
                self.caption,
            ),
            "video_path": str(item.video_path),
            "rendered_video_path": str(item.rendered_video_path),
            "rendered_mask_path": str(item.rendered_mask_path),
            "hand_path": str(item.hand_path),
            "camera_info_path": str(item.raw_cam_dir),
        }

    def _discover_items(self) -> list[H2OControlWindow]:
        """Return complete windows grouped by start frame, then clip order."""

        clip_dirs: list[Path] = []
        for subject in self.subjects:
            clip_dirs.extend((self.root / subject).glob("*/*/cam4"))
        clip_dirs = sorted(
            (path for path in clip_dirs if path.is_dir()), key=_natural_sort_key
        )

        complete_clips: list[tuple[str, Path, Path, Path, Path]] = []
        for clip_dir in clip_dirs:
            clip_name = clip_dir.relative_to(self.root).parent.as_posix()
            video_path = clip_dir / f"{clip_dir.parent.name}_rgb.mp4"
            hand_path = clip_dir / "hand.mp4"
            raw_cam_dir = self.raw_root / clip_name / "cam4"
            if not video_path.is_file():
                continue
            if self.load_rendered_views and not (
                (raw_cam_dir / "cam_intrinsics.txt").is_file()
                and (raw_cam_dir / "cam_pose").is_dir()
            ):
                continue
            if self.load_hands and self.require_hands and not hand_path.is_file():
                continue
            complete_clips.append(
                (clip_name, video_path, hand_path, raw_cam_dir, clip_dir)
            )

        items: list[H2OControlWindow] = []
        for start_frame in self.start_frames:
            end_frame = start_frame + self.video_frames - 1
            range_label = f"{start_frame}-{end_frame}"
            for clip_name, video_path, hand_path, raw_cam_dir, clip_dir in complete_clips:
                rendered_video_path = clip_dir / "view_render" / f"render_{range_label}.mp4"
                rendered_mask_path = (
                    clip_dir / "view_render" / f"render_mask_{range_label}.mp4"
                )
                # A configured start may exceed this clip's last valid window;
                # its absent interval render simply omits that clip/start pair.
                if self.load_rendered_views and not (
                    rendered_video_path.is_file() and rendered_mask_path.is_file()
                ):
                    continue
                if not (raw_cam_dir / "cam_pose" / f"{end_frame:06d}.txt").is_file():
                    continue
                items.append(
                    H2OControlWindow(
                        clip_name=clip_name,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        video_path=video_path,
                        rendered_video_path=rendered_video_path,
                        rendered_mask_path=rendered_mask_path,
                        hand_path=hand_path,
                        raw_cam_dir=raw_cam_dir,
                    )
                )
        return items


class H2OObjectDataset(Dataset[ObjectRopeSample]):
    """Load H2O object views for one shared object curriculum stage."""

    def __init__(
        self,
        control_dataset: H2OControlDataset,
        stage: str,
        reference_image_size: int | Sequence[int] | None = 480,
        load_video: bool = False,
        length: int = 0,
    ) -> None:
        """Index clip object renders as in- and out-of-interval view pools."""

        if stage not in {"obj_stageA", "obj_stageB", "canonical"}:
            raise ValueError(f"Unknown H2O object-reference stage {stage!r}.")
        self.control_dataset = control_dataset
        self.stage = stage
        self.reference_image_size = _normalize_image_size(reference_image_size)
        self.load_video = bool(load_video)
        self.items = self._discover_items()
        requested_length = int(length)
        if requested_length > 0:
            self.items = self.items[:requested_length]
        if not self.items:
            raise FileNotFoundError(
                "No H2O control windows have usable in-interval object renders."
            )

    def __len__(self) -> int:
        """Return the number of H2O windows with object references."""

        return len(self.items)

    def __getitem__(self, index: int) -> ObjectRopeSample:
        """Decode three object images following the configured curriculum stage."""

        item = self.items[index % len(self.items)]
        selected = self._select_views(item)
        reference_images = torch.stack(
            [
                _load_h2o_image_tensor(
                    view.image_path,
                    self.reference_image_size or (480, 480),
                )
                for view in selected
            ],
            dim=0,
        )
        if self.load_video:
            video = _load_h2o_video_frames(
                item.video_path,
                frame_count=self.control_dataset.video_frames,
                image_size=self.control_dataset.image_size or (480, 480),
                start_frame=item.start_frame,
            )
        else:
            # Multi-encoder collation discards the object's duplicate RGB copy.
            video = torch.empty(0, dtype=torch.float32)
        primary = selected[0]
        caption = _caption_for_window(
            self.control_dataset.caption_index,
            item.clip_name,
            item.start_frame,
            self.control_dataset.video_frames,
            self.control_dataset.caption,
        )
        return ObjectRopeSample(
            clip_name=item.clip_name,
            caption=caption,
            video=video,
            reference_images=reference_images,
            object_uid=f"h2o:{primary.object_index}:{primary.object_name}",
            object_id=str(primary.object_index),
            object_name=primary.object_name,
            object_num=1,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            video_path=str(item.video_path),
            metadata_path=str(item.metadata_path),
            reference_paths=[str(view.image_path) for view in selected],
        )

    def describe_item_files(self, index: int) -> dict[str, Any]:
        """Describe one staged H2O object selection without image decoding."""

        item = self.items[index % len(self.items)]
        selected = self._select_views(item)
        primary = selected[0]
        return {
            "dataset": "h2o",
            "clip_name": item.clip_name,
            "interval": [item.start_frame, item.end_frame],
            "caption": _caption_for_window(
                self.control_dataset.caption_index,
                item.clip_name,
                item.start_frame,
                self.control_dataset.video_frames,
                self.control_dataset.caption,
            ),
            "video_path": str(item.video_path),
            "metadata_path": str(item.metadata_path),
            "object_id": str(primary.object_index),
            "object_uid": f"h2o:{primary.object_index}:{primary.object_name}",
            "object_name": primary.object_name,
            "object_view_stage": self.stage,
            "curated_views": [view.frame_index for view in selected],
            "reference_paths": [str(view.image_path) for view in selected],
        }

    def _select_views(self, item: H2OObjectWindow) -> list[H2OObjectFrame]:
        """Randomly choose the stage-specific mix of in/out interval views."""

        current = list(item.in_interval_views)
        outside = list(item.out_of_interval_views)
        if self.stage == "obj_stageA":
            return self._sample(current, 3)
        if self.stage == "obj_stageB":
            return [*self._sample(current, 2), *self._sample(outside or current, 1)]
        return self._sample(outside or current, 3)

    @staticmethod
    def _sample(views: Sequence[H2OObjectFrame], count: int) -> list[H2OObjectFrame]:
        """Randomly sample views, reusing frames only when a pool is too small."""

        if not views:
            raise ValueError("Cannot sample H2O object views from an empty pool.")
        if len(views) >= count:
            indices = torch.randperm(len(views))[:count].tolist()
        else:
            indices = torch.randint(len(views), (count,)).tolist()
        return [views[int(index)] for index in indices]

    def _discover_items(self) -> list[H2OObjectWindow]:
        """Pair every control interval with its clip-wide rendered-object bank."""

        frame_banks: dict[str, tuple[Path, tuple[H2OObjectFrame, ...]]] = {}
        for control_item in self.control_dataset.items:
            if control_item.clip_name in frame_banks:
                continue
            metadata_path = control_item.video_path.parent / "object_render_metadata.json"
            if not metadata_path.is_file():
                continue
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            frames: list[H2OObjectFrame] = []
            for frame in payload.get("frames", []):
                if not isinstance(frame, dict) or not frame.get("image_path"):
                    continue
                image_path = _resolve_path(frame["image_path"])
                if not image_path.is_file():
                    continue
                frames.append(
                    H2OObjectFrame(
                        frame_index=int(frame["frame_key"]),
                        object_index=int(frame.get("object_index", -1)),
                        object_name=str(frame.get("object_name", "object")),
                        image_path=image_path,
                    )
                )
            frame_banks[control_item.clip_name] = (
                metadata_path,
                tuple(sorted(frames, key=lambda frame: frame.frame_index)),
            )

        items: list[H2OObjectWindow] = []
        for control_item in self.control_dataset.items:
            bank = frame_banks.get(control_item.clip_name)
            if bank is None:
                continue
            metadata_path, frames = bank
            current = tuple(
                frame
                for frame in frames
                if control_item.start_frame <= frame.frame_index <= control_item.end_frame
            )
            if not current:
                continue
            outside = tuple(
                frame
                for frame in frames
                if frame.frame_index < control_item.start_frame
                or frame.frame_index > control_item.end_frame
            )
            items.append(
                H2OObjectWindow(
                    metadata_path=metadata_path,
                    video_path=control_item.video_path,
                    clip_name=control_item.clip_name,
                    start_frame=control_item.start_frame,
                    end_frame=control_item.end_frame,
                    in_interval_views=current,
                    out_of_interval_views=outside,
                )
            )
        return items
