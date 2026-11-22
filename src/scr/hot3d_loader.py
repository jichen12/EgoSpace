"""Load HOT3D control windows and curated or canonical object-view datasets.

To test the loader, use:
```
uv run python -c "
import json
from scr.hot3d_loader import Hot3DObjectRopeDataset

dataset = Hot3DObjectRopeDataset(
    hot3d_start_from=1859,
    video_image_size=64,
    reference_image_size=64,
)

print(f'discovered_items={len(dataset.items)}')

for index in range(10):
    sample = dataset[index]
    print(json.dumps({
        'index': index,
        'clip_name': sample.clip_name,
        'interval': [sample.start_frame, sample.end_frame],
        'object_id': sample.object_id,
        'object_name': sample.object_name,
        'caption': sample.caption,
        'video_shape': list(sample.video.shape),
        'reference_shape': list(sample.reference_images.shape),
        'reference_paths': sample.reference_paths,
    }))
```

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
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("outputs/hot3d_preprocessed")
DEFAULT_VAL_ROOT = Path("outputs/hot3d_preprocessed_val")
DEFAULT_CAPTION_PATH = Path("outputs/hot3d_videollava_captions.json")
# Canonical-view defaults stay importable for checkpoints, inference, and configs.
DEFAULT_CAPTION = "An egocentric video showing hands interacting with an object."
DEFAULT_REFERENCE_VIEW_LABELS = ("front", "elev_000_az_0120", "elev_000_az_0240")
# Supported HOT3D camera-render directories produced by the preprocessing jobs.
HOT3D_VIEW_RENDER_ROOTS = ("hand_view_render", "view_render_old")
DEFAULT_VIEW_RENDER_ROOT = HOT3D_VIEW_RENDER_ROOTS[0]
MAX_START_FRAME = 70
DEFAULT_VIEW_SOURCE = "curated"
DEFAULT_CANONICAL_METADATA_GLOB = "clip-*/canonical_views/*_canonical_views.json"
DEFAULT_CURATED_METADATA_GLOB = "clip-*/curated_views/curated_views.json"
CURATED_REFERENCE_VIEWS_PER_OBJECT = 3
CLIP_NAME_PATTERN = re.compile(r"clip-\d+")
CAPTION_WINDOW_PATTERN = re.compile(r"^(clip-\d+)_s(\d+)_e(\d+)$")
TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
OBJECT_NAME_STOPWORDS = frozenset({"food", "obj", "object"})

# Fixed Video-LLaVA captions are addressed by their exact inclusive video window.
CaptionIndex = dict[tuple[str, int, int], str]


class IncompleteCuratedIntervalError(ValueError):
    """Mark a curated interval that cannot supply its three reference images."""


@dataclass(frozen=True)
class ObjectRopeSample:
    """Store one decoded HOT3D object-reference sample before collation."""

    clip_name: str
    caption: str
    video: torch.Tensor
    reference_images: torch.Tensor
    object_uid: str
    object_id: str
    object_name: str
    object_num: int
    start_frame: int
    end_frame: int
    video_path: str
    metadata_path: str
    reference_paths: list[str]


@dataclass(frozen=True)
class CuratedObjectClip:
    """Describe one object interval and its ordered curated reference renders."""

    metadata_path: Path
    video_path: Path
    clip_name: str
    start_frame: int
    end_frame: int
    object_id: str
    object_uid: str
    object_name: str
    curated_view_frames: tuple[int, ...]
    reference_paths: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        """Return the inclusive interval length."""

        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class CanonicalObjectClip:
    """Describe one fixed video window and its canonical-view metadata."""

    metadata_path: Path
    video_path: Path
    clip_name: str
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        """Return the inclusive fixed-window length."""

        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class Hot3DControlWindow:
    """Describe one fixed HOT3D window and every control-training input file."""

    clip_name: str
    start_frame: int
    end_frame: int
    video_path: Path
    rendered_video_path: Path
    rendered_mask_path: Path
    hand_path: Path
    camera_info_path: Path


@dataclass(frozen=True)
class Hot3DControlSample:
    """Store one decoded camera-and-hand control sample before collation."""

    first_frame: torch.Tensor
    raw_video: torch.Tensor
    intrinsic: torch.Tensor | None
    extrinsic: torch.Tensor | None
    rendered_novel_views: torch.Tensor | None
    rendered_novel_view_masks: torch.Tensor | None
    rendered_hands: torch.Tensor | None
    rendered_object: torch.Tensor | None
    clip_name: str
    caption: str
    start_frame: int
    end_frame: int
    video_path: str
    rendered_video_path: str
    rendered_mask_path: str
    hand_path: str
    camera_info_path: str


class Hot3DControlDataset(Dataset[Hot3DControlSample]):
    """Load fixed RGB, rendered-view, and hand windows for EgoHOI training."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        clip_names: Sequence[str] | None = None,
        video_frames: int = 81,
        image_size: int | Sequence[int] | None = 480,
        stream_id: str = "214-1",
        start_frames: int | str | Sequence[int] | None = None,
        caption: str = DEFAULT_CAPTION,
        caption_annotations_path: str | Path | None = DEFAULT_CAPTION_PATH,
        load_rendered_views: bool = True,
        load_hands: bool = True,
        require_hands: bool = True,
        view_render_root: str = DEFAULT_VIEW_RENDER_ROOT,
    ) -> None:
        """Store loader settings and discover one item per available frame window."""

        self.root = _resolve_workspace_path(root)
        self.clip_names = tuple(str(name) for name in clip_names) if clip_names else None
        self.video_frames = int(video_frames)
        if self.video_frames <= 0:
            raise ValueError(f"video_frames must be positive, got {self.video_frames}.")
        self.image_size = _normalize_image_size(image_size)
        self.stream_id = str(stream_id)
        self.start_frames = _normalize_start_frames(start_frames)
        self.caption = str(caption)
        self.caption_index = _load_caption_index(caption_annotations_path)
        self.load_rendered_views = bool(load_rendered_views)
        self.load_hands = bool(load_hands)
        self.require_hands = bool(require_hands)
        self.view_render_root = str(view_render_root)
        if self.view_render_root not in HOT3D_VIEW_RENDER_ROOTS:
            raise ValueError(
                "view_render_root must be one of "
                f"{HOT3D_VIEW_RENDER_ROOTS}, got {self.view_render_root!r}."
            )
        self.items = self._discover_items()
        if not self.items:
            expected_starts = ", ".join(str(frame) for frame in self.start_frames)
            raise FileNotFoundError(
                f"No complete HOT3D control windows found under {self.root} for "
                f"start_frames=[{expected_starts}]."
            )

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> Hot3DControlDataset:
        """Build the HOT3D control dataset from the training data section."""

        return cls(
            root=data_config.get("root", DEFAULT_ROOT),
            clip_names=data_config.get("clip_names"),
            video_frames=int(data_config.get("video_frames", data_config.get("num_frames", 81))),
            image_size=data_config.get(
                "image_size",
                data_config.get("resolution", data_config.get("video_image_size", 480)),
            ),
            stream_id=str(data_config.get("stream_id", "214-1")),
            start_frames=data_config.get("start_frames"),
            caption=str(data_config.get("caption", data_config.get("prompt", DEFAULT_CAPTION))),
            caption_annotations_path=data_config.get(
                "caption_annotations_path",
                DEFAULT_CAPTION_PATH,
            ),
            load_rendered_views=bool(data_config.get("load_rendered_views", True)),
            load_hands=bool(data_config.get("load_hands", True)),
            require_hands=bool(data_config.get("require_hands", True)),
            view_render_root=str(data_config.get("view_render_root", DEFAULT_VIEW_RENDER_ROOT)),
        )

    def _resolve_render_window_paths(
        self,
        clip_dir: Path,
        range_label: str,
    ) -> tuple[Path, Path] | None:
        """Return the first complete render and mask pair for one window.

        Prepared HOT3D clips use both the legacy ``render_*`` naming convention
        and the newer ``render_hand_*`` convention inside ``hand_view_render``.
        Accept both forms so inference works with existing preprocessed clips.
        """

        render_root = self.view_render_root
        # Preserve compatibility with fixtures and older clips using view_render.
        if (
            render_root == DEFAULT_VIEW_RENDER_ROOT
            and not (clip_dir / render_root).is_dir()
            and (clip_dir / "view_render").is_dir()
        ):
            render_root = "view_render"
        render_dir = clip_dir / render_root
        candidate_pairs = (
            (
                render_dir / f"render_{range_label}.mp4",
                render_dir / f"render_mask_{range_label}.mp4",
            ),
            (
                render_dir / f"render_hand_{range_label}.mp4",
                render_dir / f"render_mask_hand_{range_label}.mp4",
            ),
        )
        for rendered_video_path, rendered_mask_path in candidate_pairs:
            if rendered_video_path.is_file() and rendered_mask_path.is_file():
                return rendered_video_path, rendered_mask_path
        return None

    @staticmethod
    def collate_samples(samples: Sequence[Hot3DControlSample]) -> dict[str, Any]:
        """Stack only controls that were loaded for the enabled model branches."""

        if not samples:
            raise ValueError("Cannot collate an empty HOT3D control sample list.")
        raw_video = torch.stack([sample.raw_video for sample in samples], dim=0)
        batch: dict[str, Any] = {
            # Derive the first frame from the stacked video and avoid stacking
            # the complete RGB video a second time under a redundant key.
            "first_frame": raw_video[:, 0],
            "raw_video": raw_video,
            "clip_name": [sample.clip_name for sample in samples],
            "caption": [sample.caption for sample in samples],
            "start_frame": [sample.start_frame for sample in samples],
            "end_frame": [sample.end_frame for sample in samples],
            "interval": [(sample.start_frame, sample.end_frame) for sample in samples],
            "video_path": [sample.video_path for sample in samples],
            "rendered_video_path": [sample.rendered_video_path for sample in samples],
            "rendered_mask_path": [sample.rendered_mask_path for sample in samples],
            "hand_path": [sample.hand_path for sample in samples],
            "camera_info_path": [sample.camera_info_path for sample in samples],
        }

        camera_values = [
            (
                sample.intrinsic,
                sample.extrinsic,
                sample.rendered_novel_views,
                sample.rendered_novel_view_masks,
            )
            for sample in samples
        ]
        if all(all(value is not None for value in values) for values in camera_values):
            batch["intrinsic"] = torch.stack(
                [sample.intrinsic for sample in samples if sample.intrinsic is not None],
                dim=0,
            )
            batch["extrinsic"] = torch.stack(
                [sample.extrinsic for sample in samples if sample.extrinsic is not None],
                dim=0,
            )
            batch["rendered_novel_views"] = torch.stack(
                [
                    sample.rendered_novel_views
                    for sample in samples
                    if sample.rendered_novel_views is not None
                ],
                dim=0,
            )
            batch["rendered_novel_view_masks"] = torch.stack(
                [
                    sample.rendered_novel_view_masks
                    for sample in samples
                    if sample.rendered_novel_view_masks is not None
                ],
                dim=0,
            )
        elif any(any(value is not None for value in values) for values in camera_values):
            raise ValueError("Cannot mix loaded and disabled camera controls in one batch.")

        hand_values = [sample.rendered_hands for sample in samples]
        if all(value is not None for value in hand_values):
            batch["rendered_hands"] = torch.stack(
                [value for value in hand_values if value is not None],
                dim=0,
            )
        elif any(value is not None for value in hand_values):
            raise ValueError("Cannot mix loaded and disabled hand controls in one batch.")
        return batch

    def __len__(self) -> int:
        """Return the number of complete clip-window combinations."""

        return len(self.items)

    def __getitem__(self, index: int) -> Hot3DControlSample:
        """Decode one source window and its matching rendered controls."""

        item = self.items[index % len(self.items)]
        raw_video = _load_normalized_video_frames(
            item.video_path,
            frame_count=self.video_frames,
            image_size=self.image_size,
            start_frame=item.start_frame,
        )
        frame_count, _, height, width = raw_video.shape
        if self.load_rendered_views:
            # Rendered videos are already trimmed to the named interval, so
            # their local frame zero corresponds to the source start frame.
            rendered_novel_views = _load_normalized_video_frames(
                item.rendered_video_path,
                frame_count=self.video_frames,
                image_size=self.image_size,
                start_frame=0,
            ).to(dtype=raw_video.dtype)
            rendered_mask_rgb = _load_normalized_video_frames(
                item.rendered_mask_path,
                frame_count=self.video_frames,
                image_size=self.image_size,
                start_frame=0,
            ).to(dtype=raw_video.dtype)
            rendered_novel_view_masks = rendered_mask_rgb.mean(dim=1, keepdim=True)
        else:
            # Disabled camera control has no placeholder volume. Omitting it
            # prevents a hand-only batch from allocating two unused videos.
            rendered_novel_views = None
            rendered_novel_view_masks = None

        if self.load_hands and item.hand_path.is_file():
            # hand.mp4 contains the full 150-frame source clip and therefore
            # uses the absolute frame-window start rather than local frame zero.
            rendered_hands = _load_normalized_video_frames(
                item.hand_path,
                frame_count=self.video_frames,
                image_size=self.image_size,
                start_frame=item.start_frame,
            ).to(dtype=raw_video.dtype)
        elif self.load_hands and self.require_hands:
            raise FileNotFoundError(f"Required HOT3D hand video is missing: {item.hand_path}")
        else:
            rendered_hands = None

        if self.load_rendered_views:
            intrinsic, extrinsic = _load_camera_window(
                item.camera_info_path,
                start_frame=item.start_frame,
                frame_count=self.video_frames,
            )
        else:
            # Camera matrices are not opened or materialized for hand-only runs.
            intrinsic = None
            extrinsic = None
        caption = _caption_for_exact_window(
            self.caption_index,
            clip_name=item.clip_name,
            start_frame=item.start_frame,
            frame_count=self.video_frames,
            default_caption=self.caption,
        )
        # No current camera/hand model consumes the legacy object-mask slot.
        rendered_object = None
        return Hot3DControlSample(
            first_frame=raw_video[0],
            raw_video=raw_video,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            rendered_novel_views=rendered_novel_views,
            rendered_novel_view_masks=rendered_novel_view_masks,
            rendered_hands=rendered_hands,
            rendered_object=rendered_object,
            clip_name=item.clip_name,
            caption=caption,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            video_path=str(item.video_path),
            rendered_video_path=str(item.rendered_video_path),
            rendered_mask_path=str(item.rendered_mask_path),
            hand_path=str(item.hand_path),
            camera_info_path=str(item.camera_info_path),
        )

    def describe_item_files(self, index: int) -> dict[str, Any]:
        """Return one window's selected files and caption without decoding video."""

        item = self.items[index % len(self.items)]
        return {
            "clip_name": item.clip_name,
            "interval": [item.start_frame, item.end_frame],
            "caption": _caption_for_exact_window(
                self.caption_index,
                clip_name=item.clip_name,
                start_frame=item.start_frame,
                frame_count=self.video_frames,
                default_caption=self.caption,
            ),
            "video_path": str(item.video_path),
            "rendered_video_path": str(item.rendered_video_path),
            "rendered_mask_path": str(item.rendered_mask_path),
            "hand_path": str(item.hand_path),
            "camera_info_path": str(item.camera_info_path),
        }

    def _discover_items(self) -> list[Hot3DControlWindow]:
        """Return complete windows grouped by start frame, then by clip."""

        if self.clip_names:
            clip_dirs = [self.root / clip_name for clip_name in self.clip_names]
        else:
            clip_dirs = sorted(self.root.glob("clip-*"))

        complete_clip_inputs: list[tuple[str, Path, Path, Path, Path]] = []
        for clip_dir in clip_dirs:
            if not clip_dir.is_dir():
                continue
            clip_name = clip_dir.name
            video_path = clip_dir / f"{clip_name}_rgb.mp4"
            hand_path = clip_dir / "hand.mp4"
            # Camera metadata is consolidated independently from RGB
            # undistortion and now lives directly in the prepared clip root.
            camera_info_path = clip_dir / f"camera_info_{self.stream_id}.json"
            if not video_path.is_file():
                continue
            if self.load_rendered_views and not camera_info_path.is_file():
                continue
            if self.load_hands and self.require_hands and not hand_path.is_file():
                continue

            complete_clip_inputs.append(
                (clip_name, video_path, hand_path, camera_info_path, clip_dir)
            )

        items: list[Hot3DControlWindow] = []
        # Keep the configured start-frame order authoritative across the whole
        # dataset: visit every eligible clip at one start before advancing to
        # the next start. This makes sequential loading produce, for example,
        # all 0-80 windows before any 34-114 window.
        for start_frame in self.start_frames:
            for (
                clip_name,
                video_path,
                hand_path,
                camera_info_path,
                clip_dir,
            ) in complete_clip_inputs:
                end_frame = start_frame + self.video_frames - 1
                range_label = f"{start_frame}-{end_frame}"
                rendered_paths = self._resolve_render_window_paths(clip_dir, range_label)
                if self.load_rendered_views:
                    if rendered_paths is None:
                        continue
                    rendered_video_path, rendered_mask_path = rendered_paths
                else:
                    # Keep stable placeholder paths when rendered views are disabled.
                    rendered_video_path = (
                        clip_dir / self.view_render_root / f"render_{range_label}.mp4"
                    )
                    rendered_mask_path = (
                        clip_dir / self.view_render_root / f"render_mask_{range_label}.mp4"
                    )
                items.append(
                    Hot3DControlWindow(
                        clip_name=clip_name,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        video_path=video_path,
                        rendered_video_path=rendered_video_path,
                        rendered_mask_path=rendered_mask_path,
                        hand_path=hand_path,
                        camera_info_path=camera_info_path,
                    )
                )
        return items


class Hot3DObjectRopeDataset(Dataset[ObjectRopeSample]):
    """Load HOT3D videos with either curated or canonical object references."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        clip_names: Sequence[str] | None = None,
        view_source: str | None = None,
        curated_metadata_glob: str = DEFAULT_CURATED_METADATA_GLOB,
        metadata_glob: str | None = None,
        video_frames: int = 81,
        video_image_size: int | Sequence[int] | None = 480,
        reference_image_size: int | Sequence[int] | None = 480,
        reference_view_labels: Sequence[str] | None = DEFAULT_REFERENCE_VIEW_LABELS,
        num_reference_views: int | None = None,
        object_uid: str | None = None,
        object_bop_id: int | None = None,
        object_name: str | None = None,
        max_object_num: int | None = 1,
        caption: str = DEFAULT_CAPTION,
        caption_annotations_path: str | Path | None = DEFAULT_CAPTION_PATH,
        start_frames: int | str | Sequence[int] | None = None,
        require_video: bool = True,
        strict_view_labels: bool = True,
        hot3d_start_from: int | None = None,
        length: int = 0,
    ) -> None:
        """Store view-loader settings and discover all matching samples.

        Direct legacy callers that provide ``metadata_glob`` but omit
        ``view_source`` are treated as canonical-view callers.
        """

        self.root = Path(root)
        self.clip_names = [str(name) for name in clip_names] if clip_names else None
        inferred_view_source = "canonical" if view_source is None and metadata_glob is not None else view_source
        self.view_source = _normalize_view_source(inferred_view_source)
        self.curated_metadata_glob = str(curated_metadata_glob)
        self.metadata_glob = str(metadata_glob or DEFAULT_CANONICAL_METADATA_GLOB)
        self.video_frames = int(video_frames)
        if self.video_frames <= 0 or (self.video_frames - 1) % 4 != 0:
            raise ValueError(f"Wan 2.1 requires video_frames=1+4k, got {self.video_frames}.")
        self.video_image_size = _normalize_image_size(video_image_size)
        if self.video_image_size is not None and any(size % 16 for size in self.video_image_size):
            raise ValueError(
                f"Wan 2.1 I2V requires video height/width divisible by 16, got {self.video_image_size}."
            )
        self.reference_image_size = _normalize_image_size(reference_image_size)
        self.reference_view_labels = (
            tuple(str(label) for label in reference_view_labels) if reference_view_labels else None
        )
        self.num_reference_views = int(num_reference_views) if num_reference_views is not None else None
        if self.num_reference_views is not None and self.num_reference_views <= 0:
            raise ValueError(f"num_reference_views must be positive or None, got {self.num_reference_views}.")
        self.object_uid = str(object_uid) if object_uid else None
        self.object_bop_id = int(object_bop_id) if object_bop_id is not None else None
        self.object_name = str(object_name) if object_name else None
        self.max_object_num = int(max_object_num) if max_object_num is not None else None
        if self.max_object_num is not None and self.max_object_num <= 0:
            raise ValueError(f"max_object_num must be positive or None, got {self.max_object_num}.")
        self.caption = str(caption)
        self.caption_index = _load_caption_index(caption_annotations_path)
        self.start_frames = _normalize_start_frames(start_frames)
        self.require_video = bool(require_video)
        self.strict_view_labels = bool(strict_view_labels)
        self.hot3d_start_from = int(hot3d_start_from) if hot3d_start_from is not None else None
        self.requested_length = int(length)

        self.items = self._discover_items()
        if not self.items:
            metadata_pattern = (
                self.curated_metadata_glob if self.view_source == "curated" else self.metadata_glob
            )
            raise FileNotFoundError(
                f"No {self.view_source} object-rope samples found under {self.root}. Expected metadata "
                f"matching {metadata_pattern!r} plus clip RGB mp4 files."
            )

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> Hot3DObjectRopeDataset:
        """Build the selected HOT3D view dataset from a training data config."""

        return cls(
            root=data_config.get("root", DEFAULT_ROOT),
            clip_names=data_config.get("clip_names"),
            view_source=data_config.get("view_source"),
            curated_metadata_glob=str(
                data_config.get("curated_metadata_glob", DEFAULT_CURATED_METADATA_GLOB)
            ),
            metadata_glob=data_config.get("metadata_glob"),
            video_frames=int(data_config.get("video_frames", data_config.get("num_frames", 81))),
            video_image_size=data_config.get("resolution", data_config.get("video_image_size", 480)),
            reference_image_size=data_config.get(
                "reference_image_size",
                data_config.get("canonical_image_size", 480),
            ),
            reference_view_labels=data_config.get(
                "reference_view_labels",
                list(DEFAULT_REFERENCE_VIEW_LABELS),
            ),
            num_reference_views=data_config.get("num_reference_views"),
            object_uid=data_config.get("object_uid"),
            object_bop_id=data_config.get("object_bop_id"),
            object_name=data_config.get("object_name"),
            max_object_num=data_config.get("max_object_num", 1),
            caption=str(data_config.get("caption", DEFAULT_CAPTION)),
            caption_annotations_path=data_config.get(
                "caption_annotations_path",
                DEFAULT_CAPTION_PATH,
            ),
            start_frames=data_config.get("start_frames"),
            require_video=bool(data_config.get("require_video", True)),
            strict_view_labels=bool(data_config.get("strict_view_labels", True)),
            hot3d_start_from=data_config.get("hot3d_start_from"),
            length=int(data_config.get("length", 0) or 0),
        )

    @staticmethod
    def collate_samples(samples: Sequence[ObjectRopeSample]) -> dict[str, Any]:
        """Stack object-reference samples into the model batch dictionary."""

        return {
            "clip_name": [sample.clip_name for sample in samples],
            "caption": [sample.caption for sample in samples],
            "video": torch.stack([sample.video for sample in samples], dim=0),
            "reference_images": torch.stack([sample.reference_images for sample in samples], dim=0),
            "object_uid": [sample.object_uid for sample in samples],
            "object_id": [sample.object_id for sample in samples],
            "object_name": [sample.object_name for sample in samples],
            "object_num": [sample.object_num for sample in samples],
            "start_frame": [sample.start_frame for sample in samples],
            "end_frame": [sample.end_frame for sample in samples],
            "interval": [(sample.start_frame, sample.end_frame) for sample in samples],
            "video_path": [sample.video_path for sample in samples],
            "metadata_path": [sample.metadata_path for sample in samples],
            "reference_paths": [sample.reference_paths for sample in samples],
        }

    def __len__(self) -> int:
        """Return discovered sample count or a configured overfit-debug length."""

        return max(1, self.requested_length) if self.requested_length > 0 else len(self.items)

    def __getitem__(self, index: int) -> ObjectRopeSample:
        """Decode one sample using the configured reference-view source."""

        item = self.items[index % len(self.items)]
        if isinstance(item, CanonicalObjectClip):
            return self._load_canonical_sample(item)
        return self._load_curated_sample(item)

    def _load_curated_sample(self, item: CuratedObjectClip) -> ObjectRopeSample:
        """Decode one inclusive RGB interval and its ordered curated renders."""

        reference_images = self._load_curated_reference_images(item)
        video = _load_normalized_video_frames(
            item.video_path,
            frame_count=item.frame_count,
            image_size=self.video_image_size,
            start_frame=item.start_frame,
        )
        caption = self._caption_for_interval(
            clip_name=item.clip_name,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            object_name=item.object_name,
        )
        return ObjectRopeSample(
            clip_name=item.clip_name,
            caption=caption,
            video=video,
            reference_images=reference_images,
            object_uid=item.object_uid,
            object_id=item.object_id,
            object_name=item.object_name,
            object_num=1,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            video_path=str(item.video_path),
            metadata_path=str(item.metadata_path),
            reference_paths=[str(path) for path in item.reference_paths],
        )

    def _load_canonical_sample(self, item: CanonicalObjectClip) -> ObjectRopeSample:
        """Decode one fixed RGB window and its selected canonical renders."""

        (
            reference_images,
            object_uid,
            object_id,
            object_name,
            object_num,
            reference_paths,
        ) = self._load_canonical_reference_images(item.metadata_path)
        video = _load_normalized_video_frames(
            item.video_path,
            frame_count=item.frame_count,
            image_size=self.video_image_size,
            start_frame=item.start_frame,
        )
        caption = self._format_caption(
            self._caption_for_window(clip_name=item.clip_name, start_frame=item.start_frame),
            object_name=object_name,
            clip_name=item.clip_name,
        )
        return ObjectRopeSample(
            clip_name=item.clip_name,
            caption=caption,
            video=video,
            reference_images=reference_images,
            object_uid=object_uid,
            object_id=object_id,
            object_name=object_name,
            object_num=object_num,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            video_path=str(item.video_path),
            metadata_path=str(item.metadata_path),
            reference_paths=[str(path) for path in reference_paths],
        )

    def _discover_items(self) -> list[CuratedObjectClip | CanonicalObjectClip]:
        """Dispatch sample discovery to the configured view source."""

        if self.view_source == "canonical":
            return self._discover_canonical_items()
        return self._discover_curated_items()

    def _discover_curated_items(self) -> list[CuratedObjectClip]:
        """Return complete curated items aligned with configured video windows."""

        allowed = set(self.clip_names or [])
        requested_intervals = {
            (start_frame, start_frame + self.video_frames - 1)
            for start_frame in self.start_frames
        }
        items: list[CuratedObjectClip] = []
        for metadata_path in sorted(self.root.glob(self.curated_metadata_glob)):
            clip_name = metadata_path.parent.parent.name
            if allowed and clip_name not in allowed:
                continue
            if self.hot3d_start_from is not None and _hot3d_clip_index(clip_name) < self.hot3d_start_from:
                continue

            video_path = self.root / clip_name / f"{clip_name}_rgb.mp4"
            if self.require_video and not video_path.is_file():
                continue
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object in {metadata_path}.")

            objects = payload.get("objects", [])
            if not isinstance(objects, list):
                raise ValueError(f"Expected an objects list in {metadata_path}.")
            for object_index, obj in enumerate(objects):
                if not isinstance(obj, dict) or not self._curated_object_matches(obj):
                    continue
                # A manifest can omit a fixed window when no moving object
                # produced reference renders. Entries for another window
                # configuration must not create or interrupt a training step.
                interval = obj.get("interval")
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    raise ValueError(
                        f"Object {object_index} in {metadata_path} has invalid interval {interval!r}."
                    )
                if (int(interval[0]), int(interval[1])) not in requested_intervals:
                    continue

                try:
                    item = self._curated_item_from_payload(
                        metadata_path=metadata_path,
                        video_path=video_path,
                        clip_name=clip_name,
                        object_index=object_index,
                        obj=obj,
                    )
                except IncompleteCuratedIntervalError:
                    continue
                # Resolve files before workers begin so an incomplete interval
                # is omitted instead of failing in the middle of training.
                if not all(path.is_file() for path in item.reference_paths):
                    continue
                items.append(item)

        start_order = {
            start_frame: position
            for position, start_frame in enumerate(self.start_frames)
        }
        return sorted(
            items,
            key=lambda item: (start_order[item.start_frame], item.clip_name),
        )

    def _curated_object_matches(self, obj: dict[str, Any]) -> bool:
        """Return whether one curated object passes optional identity filters."""

        if self.object_uid is not None and str(obj.get("object_uid", "")) != self.object_uid:
            return False
        if self.object_bop_id is not None:
            try:
                object_id = int(obj.get("object_id", -1))
            except (TypeError, ValueError):
                return False
            if object_id != self.object_bop_id:
                return False
        if self.object_name is not None and str(obj.get("object_name", "")) != self.object_name:
            return False
        return True

    def _curated_item_from_payload(
        self,
        metadata_path: Path,
        video_path: Path,
        clip_name: str,
        object_index: int,
        obj: dict[str, Any],
    ) -> CuratedObjectClip:
        """Validate and convert one curated JSON entry into a dataset item."""

        interval = obj.get("interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"Object {object_index} in {metadata_path} has invalid interval {interval!r}.")
        start_frame, end_frame = int(interval[0]), int(interval[1])
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(f"Object {object_index} in {metadata_path} has invalid interval {interval!r}.")
        interval_frames = end_frame - start_frame + 1
        if interval_frames != self.video_frames:
            raise ValueError(
                f"Object {object_index} in {metadata_path} has {interval_frames} interval frames; "
                f"configured video_frames={self.video_frames}."
            )

        curated_view_frames, reference_paths = _curated_reference_paths(obj, metadata_path)
        if len(reference_paths) != CURATED_REFERENCE_VIEWS_PER_OBJECT:
            raise IncompleteCuratedIntervalError(
                f"Object {object_index} in {metadata_path} must provide "
                f"{CURATED_REFERENCE_VIEWS_PER_OBJECT} curated views, got {len(reference_paths)}."
            )
        if any(frame < start_frame or frame > end_frame for frame in curated_view_frames):
            raise ValueError(
                f"Object {object_index} in {metadata_path} has curated views outside interval {interval!r}."
            )

        object_name = str(obj.get("object_name", "")).strip()
        if not object_name:
            raise ValueError(f"Object {object_index} in {metadata_path} has no object_name for caption fallback.")
        return CuratedObjectClip(
            metadata_path=metadata_path,
            video_path=video_path,
            clip_name=clip_name,
            start_frame=start_frame,
            end_frame=end_frame,
            object_id=str(obj.get("object_id", "")),
            object_uid=str(obj.get("object_uid", "")),
            object_name=object_name,
            curated_view_frames=curated_view_frames,
            reference_paths=reference_paths,
        )

    def _load_curated_reference_images(self, item: CuratedObjectClip) -> torch.Tensor:
        """Load an item's curated renders as [R, 3, H, W] float tensors."""

        images = [
            _load_image_tensor(path, image_size=self.reference_image_size)
            for path in item.reference_paths
        ]
        return torch.stack(images, dim=0)

    def _caption_for_interval(
        self,
        clip_name: str,
        start_frame: int,
        end_frame: int,
        object_name: str,
    ) -> str:
        """Return the caption for this exact interval or the selected object name."""

        return self.caption_index.get(
            (clip_name, int(start_frame), int(end_frame)),
            object_name,
        )

    def describe_item_files(self, index: int) -> dict[str, Any]:
        """Return one selected sample's files without decoding its tensors."""

        item = self.items[index % len(self.items)]
        if isinstance(item, CanonicalObjectClip):
            return self._describe_canonical_item(item)
        return {
            "view_source": self.view_source,
            "clip_name": item.clip_name,
            "interval": [item.start_frame, item.end_frame],
            "start_frame": item.start_frame,
            "end_frame": item.end_frame,
            "caption": self._caption_for_interval(
                clip_name=item.clip_name,
                start_frame=item.start_frame,
                end_frame=item.end_frame,
                object_name=item.object_name,
            ),
            "video_path": str(item.video_path),
            "metadata_path": str(item.metadata_path),
            "object_id": item.object_id,
            "object_uid": item.object_uid,
            "object_name": item.object_name,
            "curated_views": list(item.curated_view_frames),
            "reference_paths": [str(path) for path in item.reference_paths],
        }

    def _describe_canonical_item(self, item: CanonicalObjectClip) -> dict[str, Any]:
        """Return one canonical sample selection without decoding its tensors."""

        reference_paths, object_uids, object_ids, object_names = self._selected_canonical_reference_paths(
            item.metadata_path
        )
        return {
            "view_source": self.view_source,
            "clip_name": item.clip_name,
            "interval": [item.start_frame, item.end_frame],
            "start_frame": item.start_frame,
            "end_frame": item.end_frame,
            "caption": self._format_caption(
                self._caption_for_window(item.clip_name, item.start_frame),
                object_name=", ".join(object_names),
                clip_name=item.clip_name,
            ),
            "video_path": str(item.video_path),
            "metadata_path": str(item.metadata_path),
            "object_id": ",".join(object_ids),
            "object_uid": ",".join(object_uids),
            "object_name": ", ".join(object_names),
            "reference_paths": [str(path) for path in reference_paths],
        }

    # Caption-aware object selection applies to canonical metadata only.
    def _select_canonical_objects(
        self,
        canonical_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Select canonical-metadata objects whose names occur in the fixed caption."""

        candidates = [
            obj
            for obj in canonical_payload.get("objects", [])
            if obj.get("views")
        ]
        if self.object_uid is not None:
            candidates = [obj for obj in candidates if str(obj.get("object_uid", "")) == self.object_uid]
        if self.object_bop_id is not None:
            candidates = [obj for obj in candidates if int(obj.get("object_bop_id", -1)) == self.object_bop_id]
        if self.object_name is not None:
            candidates = [obj for obj in candidates if str(obj.get("object_name", "")) == self.object_name]
        default_selection = self._default_object_selection(candidates)
        caption = self._caption_for_canonical_metadata(canonical_payload)
        if _is_default_caption(caption):
            return default_selection
        matched = _select_caption_matched_objects(caption, candidates)
        if not matched:
            return default_selection
        return self._default_object_selection(matched)

    def _default_object_selection(self, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the deterministic canonical object subset."""

        selected = list(candidates)
        if self.max_object_num is not None:
            selected = selected[: int(self.max_object_num)]
        return selected

    def _caption_for_canonical_metadata(self, canonical_payload: dict[str, Any]) -> str:
        """Return the exact-window caption associated with canonical-view metadata."""

        clip_name = _clip_name_from_payload(canonical_payload)
        start_frame = _payload_frame_index(canonical_payload)
        if clip_name is None or start_frame is None:
            return self.caption
        return self._caption_for_window(clip_name=clip_name, start_frame=start_frame)

    def _caption_for_window(self, clip_name: str, start_frame: int) -> str:
        """Return the caption for one exact fixed-size canonical window."""

        clip_start = int(start_frame)
        clip_end = clip_start + self.video_frames - 1
        return self.caption_index.get((clip_name, clip_start, clip_end), self.caption)

    def _discover_canonical_items(self) -> list[CanonicalObjectClip]:
        """Return fixed canonical-view windows containing requested objects."""

        allowed = set(self.clip_names or [])
        items: list[CanonicalObjectClip] = []
        for metadata_path in sorted(self.root.glob(self.metadata_glob)):
            clip_name = metadata_path.parent.parent.name
            if allowed and clip_name not in allowed:
                continue
            if self.hot3d_start_from is not None and _hot3d_clip_index(clip_name) < self.hot3d_start_from:
                continue
            start_frame = _metadata_frame_index(metadata_path)
            if start_frame is None or start_frame not in self.start_frames:
                continue
            video_path = self.root / clip_name / f"{clip_name}_rgb.mp4"
            if self.require_video and not video_path.is_file():
                continue
            if self._canonical_metadata_has_requested_object(metadata_path):
                items.append(
                    CanonicalObjectClip(
                        metadata_path=metadata_path,
                        video_path=video_path,
                        clip_name=clip_name,
                        start_frame=start_frame,
                        end_frame=start_frame + self.video_frames - 1,
                    )
                )
        return items

    def _canonical_metadata_has_requested_object(self, metadata_path: Path) -> bool:
        """Return whether canonical metadata has a usable requested object."""

        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for obj in self._select_canonical_objects(payload):
            try:
                if self._select_canonical_views(list(obj.get("views", [])), metadata_path):
                    return True
            except ValueError:
                if self.strict_view_labels:
                    return False
                raise
        return False

    def _load_canonical_reference_images(
        self,
        metadata_path: Path,
    ) -> tuple[torch.Tensor, str, str, str, int, list[Path]]:
        """Load selected canonical renders and their joined object metadata."""

        paths, object_uids, object_ids, object_names = self._selected_canonical_reference_paths(metadata_path)
        images = [_load_image_tensor(path, image_size=self.reference_image_size) for path in paths]
        if not images:
            raise ValueError(f"No canonical reference images selected from {metadata_path}.")
        return (
            torch.stack(images, dim=0),
            ",".join(object_uids),
            ",".join(object_ids),
            ", ".join(object_names),
            len(object_names),
            paths,
        )

    def _selected_canonical_reference_paths(
        self,
        metadata_path: Path,
    ) -> tuple[list[Path], list[str], list[str], list[str]]:
        """Resolve selected canonical image paths and object identifiers."""

        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        objects = self._select_canonical_objects(payload)
        if not objects:
            raise ValueError(f"No requested object found in {metadata_path}.")

        paths: list[Path] = []
        object_uids: list[str] = []
        object_ids: list[str] = []
        object_names: list[str] = []
        for obj in objects:
            selected_views = self._select_canonical_views(list(obj.get("views", [])), metadata_path)
            if not selected_views:
                continue
            object_uids.append(str(obj.get("object_uid", "")))
            object_ids.append(str(obj.get("object_bop_id", obj.get("object_id", ""))))
            object_names.append(str(obj.get("object_name", "object")))
            paths.extend(_resolve_workspace_path(view["image_path"]) for view in selected_views)
        if not paths:
            raise ValueError(f"No canonical reference images selected from {metadata_path}.")
        return paths, object_uids, object_ids, object_names

    def _select_canonical_views(
        self,
        views: list[dict[str, Any]],
        metadata_path: Path,
    ) -> list[dict[str, Any]]:
        """Select canonical views in their configured TS-RoPE order."""

        if self.reference_view_labels is not None:
            by_label = {str(view.get("label", "")): view for view in views}
            selected: list[dict[str, Any]] = []
            missing: list[str] = []
            for label in self.reference_view_labels:
                if label in by_label:
                    selected.append(by_label[label])
                else:
                    missing.append(label)
            if missing and self.strict_view_labels:
                raise ValueError(f"Missing canonical view labels {missing} in {metadata_path}.")
            if selected:
                return selected[: self.num_reference_views] if self.num_reference_views else selected
        count = self.num_reference_views or len(views)
        return views[: int(count)]

    def _format_caption(self, caption: str, object_name: str, clip_name: str) -> str:
        """Apply optional object and clip placeholders to canonical captions."""

        try:
            return caption.format(object_name=object_name.replace("_", " "), clip_name=clip_name)
        except (KeyError, IndexError, ValueError):
            return caption


def _curated_reference_paths(
    obj: dict[str, Any],
    metadata_path: Path,
) -> tuple[tuple[int, ...], tuple[Path, ...]]:
    """Resolve render paths in the exact order declared by curated_views."""

    raw_curated_views = obj.get("curated_views")
    if not isinstance(raw_curated_views, (list, tuple)) or not raw_curated_views:
        raise IncompleteCuratedIntervalError(f"Missing curated_views in {metadata_path}.")
    curated_view_frames = tuple(int(frame) for frame in raw_curated_views)

    renders_by_frame: dict[int, dict[str, Any]] = {}
    renders = obj.get("renders", [])
    if not isinstance(renders, list):
        raise ValueError(f"Expected a renders list in {metadata_path}.")
    for render in renders:
        if not isinstance(render, dict) or "frame_position" not in render:
            continue
        frame_position = int(render["frame_position"])
        if frame_position in renders_by_frame:
            raise ValueError(f"Duplicate curated render frame {frame_position} in {metadata_path}.")
        renders_by_frame[frame_position] = render

    reference_paths: list[Path] = []
    for frame_position in curated_view_frames:
        render = renders_by_frame.get(frame_position)
        if render is None or not render.get("image_path"):
            raise IncompleteCuratedIntervalError(
                f"Missing render for curated frame {frame_position} in {metadata_path}."
            )
        reference_paths.append(_resolve_workspace_path(render["image_path"]))
    return curated_view_frames, tuple(reference_paths)


def _select_caption_matched_objects(caption: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return objects whose normalized names share keywords with the caption."""

    caption_tokens = _text_tokens(caption)
    if not caption_tokens:
        return []
    scored: list[tuple[int, int, int, int, dict[str, Any]]] = []
    for object_index, obj in enumerate(candidates):
        object_tokens = _object_name_tokens(str(obj.get("object_name", "")))
        match = _caption_object_match(caption_tokens, object_tokens)
        if match is None:
            continue
        phrase_bonus, matched_count, first_position = match
        scored.append((phrase_bonus, matched_count, -first_position, -object_index, obj))
    scored.sort(key=lambda item: item[:4], reverse=True)
    return [obj for *_, obj in scored]


def _caption_object_match(caption_tokens: tuple[str, ...], object_tokens: tuple[str, ...]) -> tuple[int, int, int] | None:
    """Score how strongly one object name is mentioned by caption tokens."""

    if not object_tokens:
        return None
    token_positions: dict[str, list[int]] = {}
    for position, token in enumerate(caption_tokens):
        token_positions.setdefault(token, []).append(position)
    matched_positions = [token_positions[token][0] for token in object_tokens if token in token_positions]
    if not matched_positions:
        return None
    phrase_bonus = int(_has_multi_token_phrase(caption_tokens, object_tokens))
    return (phrase_bonus, len(matched_positions), min(matched_positions))


def _has_multi_token_phrase(caption_tokens: tuple[str, ...], object_tokens: tuple[str, ...]) -> bool:
    """Return whether a multi-word object name appears contiguously in the caption."""

    if len(object_tokens) <= 1 or len(caption_tokens) < len(object_tokens):
        return False
    variants = {object_tokens, tuple(reversed(object_tokens))}
    width = len(object_tokens)
    for start in range(0, len(caption_tokens) - width + 1):
        if caption_tokens[start : start + width] in variants:
            return True
    return False


def _text_tokens(text: str) -> tuple[str, ...]:
    """Normalize free text into lowercase alphanumeric tokens."""

    return tuple(TEXT_TOKEN_PATTERN.findall(str(text).lower()))


def _object_name_tokens(object_name: str) -> tuple[str, ...]:
    """Normalize an available HOT3D object name into useful matching tokens."""

    return tuple(token for token in _text_tokens(object_name) if token not in OBJECT_NAME_STOPWORDS)


def _is_default_caption(caption: str) -> bool:
    """Return whether a caption is the generic fallback prompt."""

    return _text_tokens(caption) == _text_tokens(DEFAULT_CAPTION)


def _clip_name_from_payload(payload: dict[str, Any]) -> str | None:
    """Extract a HOT3D ``clip-xxxxxx`` name from canonical-view metadata."""

    for key in ("clip_name", "source_clip", "clip", "output_dir", "mask_output_dir"):
        value = payload.get(key)
        if value is None:
            continue
        match = CLIP_NAME_PATTERN.search(str(value))
        if match:
            return match.group(0)
    return None


def _payload_frame_index(payload: dict[str, Any]) -> int | None:
    """Extract the selected source-frame index from canonical-view metadata."""

    for key in ("frame_key", "selected_frame_key", "frame_index"):
        value = payload.get(key)
        if value is None:
            continue
        value_text = str(value)
        if value_text.isdigit():
            return int(value_text)
    return None


def _resolve_workspace_path(path: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return REPO_ROOT / candidate


def _load_image_tensor(path: Path, image_size: tuple[int, int] | None) -> torch.Tensor:
    """Load one RGB image as normalized ``[3, H, W]`` float data."""

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            image = image.resize((int(image_size[1]), int(image_size[0])), resampling)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


def _normalize_image_size(image_size: int | Sequence[int] | None) -> tuple[int, int] | None:
    """Normalize image-size config values to ``(height, width)`` or ``None``."""

    if image_size is None:
        return None
    if isinstance(image_size, int):
        return (int(image_size), int(image_size))
    values = list(image_size)
    if len(values) != 2:
        raise ValueError(f"image_size must be an int or pair, got {image_size}.")
    return (int(values[0]), int(values[1]))


def _normalize_view_source(view_source: str | None) -> str:
    """Normalize and validate the configured reference-view source."""

    normalized = str(view_source or DEFAULT_VIEW_SOURCE).strip().lower()
    if normalized not in {"curated", "canonical"}:
        raise ValueError(
            f"view_source must be 'curated' or 'canonical', got {view_source!r}."
        )
    return normalized


def _metadata_frame_index(metadata_path: Path) -> int | None:
    """Return the six-digit frame index encoded in canonical metadata."""

    suffix = "_canonical_views.json"
    name = metadata_path.name
    if name.endswith(suffix):
        frame_key = name[: -len(suffix)]
        if frame_key.isdigit():
            return int(frame_key)
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            frame_key = str(json.load(handle).get("frame_key", ""))
    except (OSError, json.JSONDecodeError):
        return None
    return int(frame_key) if frame_key.isdigit() else None


def _normalize_start_frames(start_frames: int | str | Sequence[int] | None) -> tuple[int, ...]:
    """Normalize fixed-window starts and enforce the HOT3D maximum index."""

    if start_frames is None:
        values = [0]
    elif isinstance(start_frames, str):
        values = [int(item.strip()) for item in start_frames.split(",") if item.strip()]
    elif isinstance(start_frames, int):
        values = [int(start_frames)]
    else:
        values = [int(value) for value in start_frames]
    if not values:
        raise ValueError("start_frames must contain at least one frame index.")
    if min(values) < 0 or max(values) > MAX_START_FRAME:
        raise ValueError(f"start_frames must be within [0, {MAX_START_FRAME}], got {values}.")
    return tuple(dict.fromkeys(values))


def _load_caption_index(path: str | Path | None) -> CaptionIndex:
    """Load Video-LLaVA captions keyed by their exact inclusive video window."""

    if path is None:
        return {}
    annotation_path = _resolve_workspace_path(path)
    if not annotation_path.is_file():
        return {}
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {annotation_path}.")
    caption_index: CaptionIndex = {}
    for key, entry in payload.items():
        match = CAPTION_WINDOW_PATTERN.fullmatch(str(key))
        if match is None:
            raise ValueError(
                f"Caption key {key!r} in {annotation_path} must match "
                "'clip-xxxxxx_sNNN_eNNN'."
            )
        if not isinstance(entry, dict):
            raise ValueError(f"Caption entry {key!r} in {annotation_path} must be an object.")
        caption = str(entry.get("caption", "")).strip()
        if not caption:
            raise ValueError(f"Caption entry {key!r} in {annotation_path} has no caption.")
        clip_name, start_text, end_text = match.groups()
        start_frame, end_frame = int(start_text), int(end_text)
        if end_frame < start_frame:
            raise ValueError(f"Caption entry {key!r} has an invalid descending frame range.")
        caption_index[(clip_name, start_frame, end_frame)] = caption
    return caption_index



def _caption_for_exact_window(
    caption_index: CaptionIndex,
    clip_name: str,
    start_frame: int,
    frame_count: int,
    default_caption: str,
) -> str:
    """Return only the caption assigned to the exact inclusive frame window."""

    window_start = int(start_frame)
    window_end = window_start + int(frame_count) - 1
    return caption_index.get((clip_name, window_start, window_end), str(default_caption))


def _load_camera_window(
    camera_info_path: str | Path,
    start_frame: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and slice intrinsic and extrinsic matrices for one source window."""

    resolved_path = _resolve_workspace_path(camera_info_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    extrinsic = torch.as_tensor(payload["extrinsic"], dtype=torch.float32)
    if extrinsic.ndim == 2:
        extrinsic = extrinsic.unsqueeze(0)
    intrinsic = torch.as_tensor(payload["intrinsic"], dtype=torch.float32)
    if intrinsic.ndim == 2:
        intrinsic = intrinsic.unsqueeze(0).repeat(extrinsic.shape[0], 1, 1)
    start = int(start_frame)
    stop = start + int(frame_count)
    intrinsic = _slice_and_pad_time(intrinsic, start=start, stop=stop)
    extrinsic = _slice_and_pad_time(extrinsic, start=start, stop=stop)
    return intrinsic, extrinsic


def _slice_and_pad_time(tensor: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Slice a time-major tensor and repeat its final row if the source is short."""

    selected = tensor[int(start) : int(stop)]
    if selected.shape[0] == 0:
        raise ValueError(
            f"Cannot select frame window [{start}, {stop - 1}] from {tensor.shape[0]} camera rows."
        )
    required = int(stop) - int(start)
    if selected.shape[0] < required:
        repeat_shape = [required - selected.shape[0]] + [1] * (selected.ndim - 1)
        selected = torch.cat((selected, selected[-1:].repeat(*repeat_shape)), dim=0)
    return selected


def _load_normalized_video_frames(
    path: str | Path,
    frame_count: int,
    image_size: tuple[int, int] | None,
    start_frame: int = 0,
) -> torch.Tensor:
    """Load a contiguous MP4 window as ``[F, 3, H, W]`` float RGB values in ``[0, 1]``."""

    video_path = _resolve_workspace_path(path)
    try:
        import cv2
    except ImportError:
        return _load_normalized_video_frames_imageio(
            video_path,
            frame_count=frame_count,
            image_size=image_size,
            start_frame=start_frame,
        )

    start_frame = int(start_frame)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return _load_normalized_video_frames_imageio(
            video_path,
            frame_count=frame_count,
            image_size=image_size,
            start_frame=start_frame,
        )
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            decode_count = min(int(frame_count), max(0, total_frames - start_frame))
        else:
            decode_count = int(frame_count)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = _read_video_frames_sequential(capture, decode_count, image_size)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames could be decoded from {video_path}.")
    while len(frames) < int(frame_count):
        frames.append(frames[-1].clone())
    return torch.stack(frames[: int(frame_count)], dim=0)


def _load_normalized_video_frames_imageio(
    path: str | Path,
    frame_count: int,
    image_size: tuple[int, int] | None,
    start_frame: int = 0,
) -> torch.Tensor:
    """Load a contiguous normalized RGB frame window through imageio."""

    import imageio.v2 as imageio

    video_path = _resolve_workspace_path(path)
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
    except Exception as exc:
        raise FileNotFoundError(f"Could not open raw video clip: {video_path}") from exc
    try:
        try:
            total_frames = int(reader.count_frames())
        except Exception:
            total_frames = 0
        start_frame = int(start_frame)
        if total_frames > 0:
            stop_frame = min(total_frames, start_frame + int(frame_count))
            frames = [
                _normalize_rgb_frame(reader.get_data(frame_idx), image_size)
                for frame_idx in range(start_frame, stop_frame)
            ]
        else:
            frames = []
            for frame_idx, frame_rgb in enumerate(reader):
                if frame_idx < start_frame:
                    continue
                frames.append(_normalize_rgb_frame(frame_rgb, image_size))
                if len(frames) >= int(frame_count):
                    break
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames could be decoded from {video_path}.")
    while len(frames) < int(frame_count):
        frames.append(frames[-1].clone())
    return torch.stack(frames[: int(frame_count)], dim=0)


def _read_video_frame_at(capture: Any, frame_idx: int, image_size: tuple[int, int] | None) -> torch.Tensor:
    """Decode one OpenCV frame by index and return RGB CHW float data."""

    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not decode frame {frame_idx} from video.")
    return _normalize_bgr_frame(frame_bgr, image_size)


def _read_video_frames_sequential(capture: Any, frame_count: int, image_size: tuple[int, int] | None) -> list[torch.Tensor]:
    """Decode up to ``frame_count`` frames sequentially from OpenCV."""

    frames: list[torch.Tensor] = []
    while len(frames) < int(frame_count):
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            break
        frames.append(_normalize_bgr_frame(frame_bgr, image_size))
    return frames


def _normalize_bgr_frame(frame_bgr: np.ndarray, image_size: tuple[int, int] | None) -> torch.Tensor:
    """Convert an OpenCV BGR frame to normalized RGB ``[3, H, W]``."""

    import cv2

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        height, width = int(image_size[0]), int(image_size[1])
        frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    array = np.ascontiguousarray(frame_rgb)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


def _normalize_rgb_frame(frame_rgb: np.ndarray, image_size: tuple[int, int] | None) -> torch.Tensor:
    """Convert an RGB ndarray to normalized ``[3, H, W]`` float data."""

    array = np.asarray(frame_rgb)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    image = Image.fromarray(array.astype(np.uint8, copy=False)).convert("RGB")
    if image_size is not None:
        height, width = int(image_size[0]), int(image_size[1])
        resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        image = image.resize((width, height), resampling)
    normalized = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(normalized.copy()).permute(2, 0, 1).contiguous().float().div_(255.0)


def _hot3d_clip_index(clip_name: str) -> int:
    """Return the numeric HOT3D id from names like ``clip-001849``."""

    prefix = "clip-"
    if not clip_name.startswith(prefix):
        raise ValueError(f"Expected HOT3D clip name to start with {prefix!r}, got {clip_name!r}.")
    return int(clip_name[len(prefix) :])
