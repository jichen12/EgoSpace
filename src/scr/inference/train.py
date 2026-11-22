"""Train the combined EgoHOI camera and hand control model.

Run from the repository root with the managed uv environment:

``uv run python scripts/model/train.py``

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python scripts/model/train.py

uv run python scripts/model/train.py --set train.steps=20000

uv run python scripts/model/train.py --dry-run --set train.steps=3000
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Sampler, Subset
import yaml


# Direct script execution puts ``scripts`` rather than the repository root on
# sys.path. Add the root once so local ``scr`` imports work from any cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scr.engine.dataset import build_camera_dataset, collate_camera_training_samples
from scr.engine.loaded_data_logging import LoadedDataLogger
from scr.hot3d_loader import (
    DEFAULT_CAPTION as HOT3D_DEFAULT_CAPTION,
    DEFAULT_CAPTION_PATH as HOT3D_DEFAULT_CAPTION_PATH,
    DEFAULT_ROOT as HOT3D_DEFAULT_ROOT,
    Hot3DControlDataset,
    Hot3DControlWindow,
    Hot3DObjectRopeDataset,
    ObjectRopeSample,
    _load_image_tensor,
)
from scr.h2o_loader import (
    DEFAULT_CAPTION_PATH as H2O_DEFAULT_CAPTION_PATH,
    DEFAULT_RAW_ROOT as H2O_DEFAULT_RAW_ROOT,
    DEFAULT_ROOT as H2O_DEFAULT_ROOT,
    DEFAULT_TRAIN_SUBJECTS as H2O_DEFAULT_TRAIN_SUBJECTS,
    H2OControlDataset,
    H2OControlWindow,
    H2OObjectDataset,
)
from scr.engine.seed import seed_everything
from scr.engine.trainer import EgoHOIControlModel
from scr.utils.config import load_config


LOGGER = logging.getLogger("egohoi.train")
DEFAULT_CAMERA_RENDER_ROOT = "hand_view_render"
HOT3D_CAMERA_RENDER_SUBDIR = DEFAULT_CAMERA_RENDER_ROOT
H2O_CAMERA_RENDER_SUBDIR = DEFAULT_CAMERA_RENDER_ROOT


@dataclass(frozen=True)
class DistributedContext:
    """Process identity and device information for single-process or DDP runs."""

    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        """Return whether this process owns logging and checkpoint writes."""

        return self.rank == 0


class TrainHot3DControlDataset(Hot3DControlDataset):
    """Load HOT3D control windows from the train-time hand-view renders.

    Camera-encoder training now consumes the prepared render videos stored
    under ``clip-XXXXXX/hand_view_render`` instead of the older shared
    ``view_render`` directory used by the generic HOT3D loader.
    """

    def __init__(
        self,
        *args: Any,
        render_subdir: str = HOT3D_CAMERA_RENDER_SUBDIR,
        **kwargs: Any,
    ) -> None:
        """Store the render directory name before discovering clip windows."""

        self.render_subdir = str(render_subdir)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> TrainHot3DControlDataset:
        """Build the HOT3D training dataset with a configurable render folder."""

        return cls(
            root=data_config.get("root", HOT3D_DEFAULT_ROOT),
            clip_names=data_config.get("clip_names"),
            video_frames=int(data_config.get("video_frames", data_config.get("num_frames", 81))),
            image_size=data_config.get(
                "image_size",
                data_config.get("resolution", data_config.get("video_image_size", 480)),
            ),
            stream_id=str(data_config.get("stream_id", "214-1")),
            start_frames=data_config.get("start_frames"),
            caption=str(data_config.get("caption", data_config.get("prompt", HOT3D_DEFAULT_CAPTION))),
            caption_annotations_path=data_config.get(
                "caption_annotations_path", HOT3D_DEFAULT_CAPTION_PATH
            ),
            load_rendered_views=bool(data_config.get("load_rendered_views", True)),
            load_hands=bool(data_config.get("load_hands", True)),
            require_hands=bool(data_config.get("require_hands", True)),
            render_subdir=str(data_config.get("view_render_root", HOT3D_CAMERA_RENDER_SUBDIR)),
        )

    def __getitem__(self, index: int) -> Any:
        """Decode one sample and convert ``hand.mp4`` RGB content to grayscale."""

        sample = super().__getitem__(index)
        if sample.rendered_hands is None:
            return sample
        return replace(
            sample,
            rendered_hands=_rgb_video_tensor_to_gray_video(sample.rendered_hands),
        )

    def _resolve_render_window_paths(
        self,
        clip_dir: Path,
        range_label: str,
    ) -> tuple[Path, Path] | None:
        """Return the first complete render/mask pair for one interval.

        Older training jobs expected ``render_*`` names inside
        ``hand_view_render``. The current compositor writes
        ``render_hand_*`` and ``render_mask_hand_*`` instead. Accept both
        conventions so existing datasets do not need to be renamed.
        """

        render_dir = clip_dir / self.render_subdir
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

    def _discover_items(self) -> list[Hot3DControlWindow]:
        """Return complete windows using the configured render subdirectory."""

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
                    # Keep the legacy placeholder paths stable when the camera
                    # render branch is disabled and no render videos are read.
                    rendered_video_path = (
                        clip_dir / self.render_subdir / f"render_{range_label}.mp4"
                    )
                    rendered_mask_path = (
                        clip_dir / self.render_subdir / f"render_mask_{range_label}.mp4"
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


class TrainH2OControlDataset(H2OControlDataset):
    """Load H2O control windows from a configurable render-video folder."""

    def __init__(
        self,
        *args: Any,
        render_subdir: str = H2O_CAMERA_RENDER_SUBDIR,
        **kwargs: Any,
    ) -> None:
        """Store the render directory name before H2O window discovery."""

        self.render_subdir = str(render_subdir)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> TrainH2OControlDataset:
        """Build the H2O training dataset with a configurable render folder."""

        return cls(
            root=data_config.get("root", H2O_DEFAULT_ROOT),
            raw_root=data_config.get("raw_root", H2O_DEFAULT_RAW_ROOT),
            subjects=data_config.get("subjects", H2O_DEFAULT_TRAIN_SUBJECTS),
            video_frames=int(data_config.get("video_frames", 81)),
            image_size=data_config.get("resolution", data_config.get("image_size", 480)),
            start_frames=data_config.get("start_frames"),
            caption=str(data_config.get("caption", HOT3D_DEFAULT_CAPTION)),
            caption_annotations_path=data_config.get(
                "caption_annotations_path", H2O_DEFAULT_CAPTION_PATH
            ),
            load_rendered_views=bool(data_config.get("load_rendered_views", True)),
            load_hands=bool(data_config.get("load_hands", True)),
            require_hands=bool(data_config.get("require_hands", True)),
            length=int(data_config.get("length", 0) or 0),
            render_subdir=str(data_config.get("view_render_root", H2O_CAMERA_RENDER_SUBDIR)),
        )

    def _resolve_render_window_paths(
        self,
        clip_dir: Path,
        range_label: str,
    ) -> tuple[Path, Path] | None:
        """Return the first complete H2O render/mask pair for one interval."""

        render_dir = clip_dir / self.render_subdir
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

    def _discover_items(self) -> list[H2OControlWindow]:
        """Return complete H2O windows using the configured render folder."""

        clip_dirs: list[Path] = []
        for subject in self.subjects:
            clip_dirs.extend((self.root / subject).glob("*/*/cam4"))
        clip_dirs = sorted(path for path in clip_dirs if path.is_dir())

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
                rendered_paths = self._resolve_render_window_paths(clip_dir, range_label)
                if self.load_rendered_views:
                    if rendered_paths is None:
                        continue
                    rendered_video_path, rendered_mask_path = rendered_paths
                else:
                    # Keep stable placeholder paths when camera render IO is disabled.
                    rendered_video_path = clip_dir / self.render_subdir / f"render_{range_label}.mp4"
                    rendered_mask_path = (
                        clip_dir / self.render_subdir / f"render_mask_{range_label}.mp4"
                    )
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


def _rgb_video_tensor_to_gray_video(video: torch.Tensor) -> torch.Tensor:
    """Convert normalized RGB video frames to repeated-channel grayscale frames."""

    if video.ndim != 4 or video.shape[1] != 3:
        return video
    if torch.allclose(video[:, 0], video[:, 1]) and torch.allclose(
        video[:, 1], video[:, 2]
    ):
        return video

    rgb_float = (
        video.detach()
        .to(device="cpu", dtype=torch.float32)
        .mul(255.0)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    gray = np.dot(rgb_float[..., :3], [0.2989, 0.5870, 0.1140])
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    gray_rgb = np.repeat(gray_uint8[..., None], 3, axis=-1)
    return (
        torch.from_numpy(gray_rgb)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device=video.device, dtype=video.dtype)
        .div_(255.0)
    )


class MultiEncoderHot3DDataset(Dataset[tuple[Any, Any]]):
    """Pair camera/hand windows with exact object-reference windows.

    The class name is retained for checkpoint and test compatibility, but the
    shared item schema also supports prepared H2O control/object datasets.
    """

    def __init__(
        self,
        control_dataset: Hot3DControlDataset,
        object_dataset: Hot3DObjectRopeDataset,
    ) -> None:
        """Index samples by clip and inclusive frame interval for co-training."""

        self.control_dataset = control_dataset
        self.object_dataset = object_dataset
        object_indices = {
            (item.clip_name, int(item.start_frame), int(item.end_frame)): index
            for index, item in enumerate(object_dataset.items)
        }
        self.pairs = [
            (control_index, object_indices[key])
            for control_index, item in enumerate(control_dataset.items)
            if (
                key := (item.clip_name, int(item.start_frame), int(item.end_frame))
            ) in object_indices
        ]
        if not self.pairs:
            raise FileNotFoundError(
                "No exact clip/start/end matches exist between HOT3D controls "
                "and object-reference metadata."
            )

    def __len__(self) -> int:
        """Return the number of exact multi-encoder sample pairs."""

        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Decode one control sample and its matching object references."""

        control_index, object_index = self.pairs[index % len(self.pairs)]
        return self.control_dataset[control_index], self.object_dataset[object_index]

    def describe_item_files(self, index: int) -> dict[str, Any]:
        """Merge paired control and object metadata without decoding tensors."""

        control_index, object_index = self.pairs[index % len(self.pairs)]
        metadata = self.control_dataset.describe_item_files(control_index)
        object_metadata = self.object_dataset.describe_item_files(object_index)
        # Match collate_samples: the object-aware caption and object fields
        # override the generic control caption for the exact paired interval.
        metadata.update(object_metadata)
        return metadata

    @staticmethod
    def collate_samples(samples: list[tuple[Any, Any]]) -> dict[str, Any]:
        """Merge control tensors with the three interval-matched object images."""

        control_samples = [sample[0] for sample in samples]
        object_samples = [sample[1] for sample in samples]
        batch = Hot3DControlDataset.collate_samples(control_samples)
        object_batch = Hot3DObjectRopeDataset.collate_samples(object_samples)
        batch["reference_images"] = object_batch["reference_images"]
        # Object-aware captions name the selected interaction object while all
        # camera and hand tensors remain those from the exact matched window.
        batch["caption"] = object_batch["caption"]
        for key in (
            "object_uid",
            "object_id",
            "object_name",
            "object_num",
            "metadata_path",
            "reference_paths",
        ):
            batch[key] = object_batch[key]
        return batch


class DescribedConcatDataset(ConcatDataset[Any]):
    """Concatenate ordered validation datasets while preserving file audits."""

    def describe_item_files(self, index: int) -> dict[str, Any]:
        """Delegate metadata lookup to the child dataset owning ``index``."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = next(
            position
            for position, cumulative_size in enumerate(self.cumulative_sizes)
            if index < cumulative_size
        )
        previous_size = (
            0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        )
        child_dataset = self.datasets[dataset_index]
        describe_item = getattr(child_dataset, "describe_item_files", None)
        if not callable(describe_item):
            raise TypeError(
                f"{type(child_dataset).__name__} must implement describe_item_files()."
            )
        return dict(describe_item(index - previous_size))


class CuratedSelectedCanonicalDataset(Hot3DObjectRopeDataset):
    """Load canonical renders for the object selected by each curated interval."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Index curated object identities before canonical item discovery."""

        root = Path(kwargs.get("root", "outputs/hot3d_preprocessed"))
        curated_metadata_glob = str(
            kwargs.get(
                "curated_metadata_glob",
                "clip-*/curated_views/curated_views.json",
            )
        )
        self.curated_objects_by_interval = self._load_curated_object_index(
            root,
            curated_metadata_glob,
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(
        cls,
        data_config: dict[str, Any],
    ) -> CuratedSelectedCanonicalDataset:
        """Build the canonical dataset while retaining shared config parsing."""

        dataset = super().from_config(data_config)
        if not isinstance(dataset, cls):
            raise TypeError("Canonical dataset construction returned an unexpected type.")
        return dataset

    @staticmethod
    def _load_curated_object_index(
        root: Path,
        metadata_glob: str,
    ) -> dict[tuple[str, int, int], tuple[dict[str, str], ...]]:
        """Map each curated interval to its ordered object identities."""

        index: dict[tuple[str, int, int], tuple[dict[str, str], ...]] = {}
        for metadata_path in sorted(root.glob(metadata_glob)):
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object in {metadata_path}.")

            clip_name = metadata_path.parent.parent.name
            interval_objects: dict[tuple[str, int, int], list[dict[str, str]]] = {}
            objects = payload.get("objects", [])
            if not isinstance(objects, list):
                raise ValueError(f"Expected an objects list in {metadata_path}.")
            for object_index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    continue
                interval = obj.get("interval")
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    raise ValueError(
                        f"Object {object_index} in {metadata_path} has invalid "
                        f"interval {interval!r}."
                    )
                key = (clip_name, int(interval[0]), int(interval[1]))
                interval_objects.setdefault(key, []).append(
                    {
                        "object_uid": str(obj.get("object_uid", "")),
                        "object_id": str(obj.get("object_id", "")),
                        "object_name": str(obj.get("object_name", "")),
                    }
                )
            index.update(
                {
                    key: tuple(object_identities)
                    for key, object_identities in interval_objects.items()
                }
            )
        return index

    def _select_canonical_objects(
        self,
        canonical_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Select canonical objects matching the same window's curated objects."""

        clip_value = str(canonical_payload.get("clip", ""))
        clip_name = Path(clip_value).stem if clip_value else ""
        frame_value = canonical_payload.get(
            "frame_index",
            canonical_payload.get("frame_key"),
        )
        try:
            start_frame = int(frame_value)
        except (TypeError, ValueError):
            return []
        interval_key = (
            clip_name,
            start_frame,
            start_frame + self.video_frames - 1,
        )
        curated_identities = self.curated_objects_by_interval.get(interval_key, ())
        if not curated_identities:
            return []

        candidates = [
            obj
            for obj in canonical_payload.get("objects", [])
            if isinstance(obj, dict) and obj.get("views")
        ]
        selected: list[dict[str, Any]] = []
        for identity in curated_identities:
            for candidate in candidates:
                candidate_uid = str(candidate.get("object_uid", ""))
                candidate_id = str(
                    candidate.get("object_bop_id", candidate.get("object_id", ""))
                )
                candidate_name = str(candidate.get("object_name", ""))
                matches = (
                    bool(identity["object_uid"])
                    and candidate_uid == identity["object_uid"]
                ) or (
                    bool(identity["object_id"])
                    and candidate_id == identity["object_id"]
                ) or (
                    bool(identity["object_name"])
                    and candidate_name == identity["object_name"]
                )
                if matches and candidate not in selected:
                    selected.append(candidate)
                    break

        # Preserve explicit object filters after replacing caption matching
        # with curated-interval identity matching.
        if self.object_uid is not None:
            selected = [
                obj
                for obj in selected
                if str(obj.get("object_uid", "")) == self.object_uid
            ]
        if self.object_bop_id is not None:
            selected = [
                obj
                for obj in selected
                if int(obj.get("object_bop_id", obj.get("object_id", -1)))
                == self.object_bop_id
            ]
        if self.object_name is not None:
            selected = [
                obj
                for obj in selected
                if str(obj.get("object_name", "")) == self.object_name
            ]
        return self._default_object_selection(selected)


class ObjectReferenceStageDataset(Dataset[ObjectRopeSample]):
    """Expose one fixed curated object-reference training stage."""

    def __init__(
        self,
        curated_dataset: Hot3DObjectRopeDataset,
        stage: str,
        *,
        length: int = 0,
    ) -> None:
        """Retain stage-compatible curated items and index cross-interval views."""

        if stage not in {"obj_stageA", "obj_stageB", "canonical"}:
            raise ValueError(f"Unknown object-reference stage {stage!r}.")
        self.curated_dataset = curated_dataset
        self.stage = stage
        self.requested_length = int(length)
        if not curated_dataset.items:
            raise FileNotFoundError(
                "No curated object intervals exist for the staged "
                "object-reference schedule."
            )
        self.view_bank = self._build_curated_view_bank()
        # MultiEncoderHot3DDataset uses items to align object and control
        # windows without decoding either dataset.
        self.items = list(curated_dataset.items)

    def __len__(self) -> int:
        """Return aligned sample count or the configured debug-cycle length."""

        if self.requested_length > 0:
            return max(1, self.requested_length)
        return len(self.items)

    def __getitem__(self, index: int) -> ObjectRopeSample:
        """Decode references from the source combination for this stage."""

        curated_index = index % len(self.items)
        if self.stage == "obj_stageA":
            return self.curated_dataset[curated_index]

        curated_sample = self.curated_dataset[curated_index]
        current_refs = self._current_interval_view_refs(curated_index)
        out_refs = self._out_of_interval_view_refs(curated_index)
        if self.stage == "obj_stageB":
            selected_refs = [
                *self._sample_view_refs(current_refs, 2),
                *self._sample_view_refs(out_refs or current_refs, 1),
            ]
        else:
            selected_refs = self._sample_view_refs(out_refs or current_refs, 3)

        reference_images, reference_paths = self._load_selected_reference_views(
            selected_refs
        )
        return replace(
            curated_sample,
            reference_images=reference_images,
            reference_paths=reference_paths,
        )

    def describe_item_files(self, index: int) -> dict[str, Any]:
        """Describe the staged reference selection without image decoding."""

        curated_index = index % len(self.items)
        if self.stage == "obj_stageA":
            metadata = self.curated_dataset.describe_item_files(curated_index)
        else:
            metadata = dict(self.curated_dataset.describe_item_files(curated_index))
            current_refs = self._current_interval_view_refs(curated_index)
            out_refs = self._out_of_interval_view_refs(curated_index)
            if self.stage == "obj_stageB":
                selected_refs = [
                    *self._sample_view_refs(current_refs, 2),
                    *self._sample_view_refs(out_refs or current_refs, 1),
                ]
            else:
                selected_refs = self._sample_view_refs(out_refs or current_refs, 3)
            selected_items = [self._view_ref_metadata(ref) for ref in selected_refs]
            metadata["view_source"] = "curated"
            metadata["curated_views"] = [item["frame"] for item in selected_items]
            metadata["reference_paths"] = [item["path"] for item in selected_items]
        metadata["object_view_stage"] = self.stage
        return metadata

    @staticmethod
    def collate_samples(samples: list[ObjectRopeSample]) -> dict[str, Any]:
        """Collate staged samples with the shared object-reference schema."""

        return Hot3DObjectRopeDataset.collate_samples(samples)

    def _build_curated_view_bank(
        self,
    ) -> dict[tuple[str, str, str, str], list[tuple[int, int]]]:
        """Group all curated views by clip and object identity for sampling."""

        bank: dict[tuple[str, str, str, str], list[tuple[int, int]]] = {}
        for item_index, item in enumerate(self.curated_dataset.items):
            key = self._curated_identity_key(item)
            for view_index, _ in enumerate(item.curated_view_frames):
                bank.setdefault(key, []).append((item_index, view_index))
        return bank

    def _curated_identity_key(self, item: Any) -> tuple[str, str, str, str]:
        """Return the same-object key used to collect cross-interval views."""

        return (
            str(item.clip_name),
            str(item.object_uid),
            str(item.object_id),
            str(item.object_name),
        )

    def _current_interval_view_refs(self, curated_index: int) -> list[tuple[int, int]]:
        """Return every curated view that belongs to the current interval."""

        item = self.curated_dataset.items[curated_index]
        return [
            (curated_index, view_index)
            for view_index, _ in enumerate(item.curated_view_frames)
        ]

    def _out_of_interval_view_refs(self, curated_index: int) -> list[tuple[int, int]]:
        """Return same-object curated views outside the current frame interval."""

        item = self.curated_dataset.items[curated_index]
        refs = self.view_bank.get(self._curated_identity_key(item), [])
        return [
            (item_index, view_index)
            for item_index, view_index in refs
            if (
                self.curated_dataset.items[item_index].curated_view_frames[view_index]
                < item.start_frame
                or self.curated_dataset.items[item_index].curated_view_frames[view_index]
                > item.end_frame
            )
        ]

    def _sample_view_refs(
        self,
        refs: list[tuple[int, int]],
        count: int,
    ) -> list[tuple[int, int]]:
        """Randomly sample view references, reusing views only when required."""

        if not refs:
            raise ValueError("Cannot sample object-reference views from an empty pool.")
        if len(refs) >= count:
            indices = torch.randperm(len(refs))[:count].tolist()
        else:
            indices = torch.randint(len(refs), (count,)).tolist()
        return [refs[int(index)] for index in indices]

    def _load_selected_reference_views(
        self,
        refs: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, list[str]]:
        """Load selected curated render tensors and return their source paths."""

        images: list[torch.Tensor] = []
        paths: list[str] = []
        for item_index, view_index in refs:
            path = self.curated_dataset.items[item_index].reference_paths[view_index]
            images.append(
                _load_image_tensor(
                    path,
                    image_size=self.curated_dataset.reference_image_size,
                )
            )
            paths.append(str(path))
        return torch.stack(images, dim=0), paths

    def _view_ref_metadata(self, ref: tuple[int, int]) -> dict[str, Any]:
        """Return the frame number and path for one curated view reference."""

        item_index, view_index = ref
        item = self.curated_dataset.items[item_index]
        return {
            "frame": int(item.curated_view_frames[view_index]),
            "path": str(item.reference_paths[view_index]),
        }


def build_hot3d_object_dataset(
    data_config: dict[str, Any],
) -> Dataset[ObjectRopeSample]:
    """Build object data, including staged curated-reference schedules."""

    reference_stage = data_config.get("_object_reference_stage")
    if reference_stage is not None:
        requested_length = int(data_config.get("length", 0) or 0)
        shared_config = dict(data_config)
        shared_config.pop("_object_reference_stage", None)
        shared_config["length"] = 0
        curated_config = dict(shared_config)
        curated_config["view_source"] = "curated"
        curated_dataset = Hot3DObjectRopeDataset.from_config(curated_config)
        return ObjectReferenceStageDataset(
            curated_dataset,
            str(reference_stage),
            length=requested_length,
        )

    configured_source = data_config.get("view_source")
    if configured_source is None:
        configured_source = "canonical" if data_config.get("metadata_glob") else "curated"
    view_source = str(configured_source).strip().lower()
    dataset_type = (
        CuratedSelectedCanonicalDataset
        if view_source == "canonical"
        else Hot3DObjectRopeDataset
    )
    return dataset_type.from_config(data_config)


class StartFrameGroupedSampler(Sampler[int]):
    """Sample fixed-window dataset indices start-frame-first."""

    def __init__(
        self,
        dataset: Hot3DControlDataset | H2OControlDataset,
        *,
        shuffle: bool,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Build ordered index groups and configure deterministic DDP sharding."""

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}."
            )

        self.groups = tuple(
            tuple(
                index
                for index, item in enumerate(dataset.items)
                if item.start_frame == start_frame
            )
            for start_frame in dataset.start_frames
        )
        grouped_length = sum(len(group) for group in self.groups)
        if grouped_length != len(dataset):
            raise ValueError("Every dataset item must belong to a configured start frame.")
        self.group_num_samples = tuple(
            (
                len(group) // self.num_replicas
                if self.drop_last
                else math.ceil(len(group) / self.num_replicas)
            )
            for group in self.groups
        )
        self.num_samples = sum(self.group_num_samples)

    def __iter__(self) -> Iterator[int]:
        """Yield one rank's indices without crossing start-frame group order."""

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        rank_indices: list[int] = []
        for group, group_num_samples in zip(
            self.groups,
            self.group_num_samples,
            strict=True,
        ):
            ordered_group = list(group)
            if self.shuffle and len(group) > 1:
                permutation = torch.randperm(len(group), generator=generator).tolist()
                ordered_group = [group[position] for position in permutation]

            group_total_size = group_num_samples * self.num_replicas
            if self.drop_last:
                ordered_group = ordered_group[:group_total_size]
            elif len(ordered_group) < group_total_size:
                # Balance each group independently so every DDP rank advances
                # to the next start frame on the same training step.
                ordered_group.extend(
                    [ordered_group[-1]] * (group_total_size - len(ordered_group))
                )
            rank_indices.extend(
                ordered_group[self.rank : group_total_size : self.num_replicas]
            )
        return iter(rank_indices)

    def __len__(self) -> int:
        """Return the number of samples assigned to this process."""

        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic within-group shuffle for one epoch."""

        self.epoch = int(epoch)


class TrainingLogger:
    """Fan scalar metrics out to stdout, TensorBoard, and optional Weights & Biases."""

    def __init__(self, config: dict[str, Any], output_dir: Path, enabled: bool) -> None:
        """Initialize only the logging backends enabled by configuration."""

        self.enabled = bool(enabled)
        self.tensorboard = None
        self.wandb = None
        if not self.enabled:
            return

        logging_config = dict(config.get("logging", config.get("visualization", {})))
        tensorboard_config = dict(logging_config.get("tensorboard", {}))
        if bool(tensorboard_config.get("enabled", True)):
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = Path(tensorboard_config.get("log_dir", output_dir / "tensorboard"))
                self.tensorboard = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                LOGGER.warning("TensorBoard is enabled but not installed; continuing without it.")

        wandb_config = dict(logging_config.get("wandb", {}))
        if bool(wandb_config.get("enabled", False)):
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=str(wandb_config.get("project", "conditional-egoworld")),
                    entity=wandb_config.get("entity"),
                    name=wandb_config.get("name"),
                    dir=str(output_dir),
                    config=config,
                    mode=str(wandb_config.get("mode", "online")),
                    resume="allow",
                )
            except ImportError:
                LOGGER.warning("Weights & Biases is enabled but not installed; continuing without it.")

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Write one scalar metric dictionary at an optimizer step."""

        if not self.enabled:
            return
        if self.tensorboard is not None:
            for name, value in metrics.items():
                self.tensorboard.add_scalar(name, value, step)
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self) -> None:
        """Flush and close active logging backends."""

        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
        if self.wandb is not None:
            self.wandb.finish()


def parse_args() -> argparse.Namespace:
    """Parse the config, clip boundary, resume, and override options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML training configuration; defaults to default_config().",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument(
        "--requested-gpus",
        type=int,
        default=None,
        help="Override distributed.requested_gpus and auto-launch that many local workers.",
    )
    parser.add_argument(
        "--start-clip",
        type=int,
        default=None,
        help="Skip HOT3D clips below this numeric id (for example, 1859).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted YAML key; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit every scheduled training and validation batch without "
        "allocating Wan 2.1.",
    )
    parser.add_argument(
        "--preview-steps",
        type=int,
        default=0,
        help="Print loaded dataset metadata for this many steps before training.",
    )
    parser.add_argument(
        "--no-auto-launch",
        action="store_true",
        help="Disable automatic torchrun relaunch for multi-GPU configurations.",
    )
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted ``key=value`` CLI overrides using YAML value parsing."""

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --set value {override!r}; expected KEY=VALUE.")
        dotted_key, raw_value = override.split("=", 1)
        keys = [part for part in dotted_key.split(".") if part]
        if not keys:
            raise ValueError(f"Invalid empty override key in {override!r}.")
        target = config
        for key in keys[:-1]:
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Cannot place {dotted_key!r} below non-mapping key {key!r}.")
            target = nested
        target[keys[-1]] = yaml.safe_load(raw_value)
    return config


def configure_python_logging(output_dir: Path, is_main: bool) -> None:
    """Configure concise console logging and a persistent rank-zero log file."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def hot3d_clip_number(clip_name: str | int) -> int:
    """Return the numeric id from ``1859``, ``001859``, or ``clip-001859``."""

    value = str(clip_name).strip()
    if value.startswith("clip-"):
        value = value[len("clip-") :]
    if not value.isdigit():
        raise ValueError(
            f"Invalid HOT3D clip {clip_name!r}; expected an integer or 'clip-NNNNNN'."
        )
    return int(value)


def select_clip_names_from_start(
    data_config: dict[str, Any],
    *,
    start_clip_key: str = "start_clip",
    max_clips: int | None = None,
) -> dict[str, Any]:
    """Restrict a HOT3D config to a numeric clip suffix and optional count.

    Explicit ``clip_names`` retain their configured order. Otherwise, clip
    directories are discovered under ``hot3d_data.root`` and sorted numerically.
    """

    if data_config.get(start_clip_key) is None:
        return data_config

    start_clip = hot3d_clip_number(data_config[start_clip_key])
    configured_names = data_config.get("clip_names")
    if configured_names:
        if isinstance(configured_names, (str, int)):
            configured_names = [configured_names]
        clip_names = [str(name) for name in configured_names]
    else:
        root = Path(data_config.get("root", "outputs/hot3d_preprocessed"))
        if not root.is_absolute() and not root.exists():
            root = REPOSITORY_ROOT / root
        clip_names = [path.name for path in root.glob("clip-*") if path.is_dir()]
        clip_names.sort(key=hot3d_clip_number)

    selected_names = [
        clip_name
        for clip_name in clip_names
        if hot3d_clip_number(clip_name) >= start_clip
    ]
    if max_clips is not None and max_clips > 0:
        selected_names = selected_names[:max_clips]
    if not selected_names:
        raise FileNotFoundError(
            f"No HOT3D clip directories at or after clip-{start_clip:06d}."
        )

    data_config["clip_names"] = selected_names
    return data_config


def select_validation_clip_names(data_config: dict[str, Any]) -> dict[str, Any]:
    """Restrict validation to explicit numeric HOT3D clip ids."""

    val_clips = data_config.get("val_clips", [])
    if isinstance(val_clips, (str, int)):
        val_clips = [val_clips]
    selected_names = [
        f"clip-{hot3d_clip_number(clip_id):06d}" for clip_id in val_clips
    ]
    if not selected_names:
        raise FileNotFoundError("Validation requires at least one hot3d_data.val_clips entry.")

    data_config["clip_names"] = selected_names
    return data_config


def initialize_distributed(device_setting: str) -> DistributedContext:
    """Initialize torchrun/NCCL when WORLD_SIZE is greater than one."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed EgoHOI training requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return DistributedContext(torch.device("cuda", local_rank), rank, local_rank, world_size)

    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
    if device.type == "cuda":
        # torch.cuda.set_device requires an explicit index; plain "cuda" means
        # the current process's default device, which is index zero here.
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    return DistributedContext(device=device)


def build_dataloader(
    config: dict[str, Any],
    distributed: DistributedContext,
    *,
    validation: bool = False,
    data_overrides: dict[str, Any] | None = None,
) -> tuple[DataLoader, Sampler[int] | None]:
    """Build a train or validation loader from the shared ``hot3d_data`` section."""

    data_config = dict(config.get("hot3d_data", {}))
    if data_overrides:
        data_config.update(data_overrides)
    camera_enabled = encoder_is_enabled(config, "camera_encoder")
    hand_enabled = encoder_is_enabled(config, "hand_encoder")
    obj_enabled = encoder_is_enabled(config, "obj_encoder")
    if validation and data_config.get("val_root"):
        data_config["root"] = data_config["val_root"]
    # Resolve clip boundaries before constructing the loader because the
    # dataset's clip_names option is the authoritative clip allowlist.
    if validation:
        data_config = select_validation_clip_names(data_config)
        data_config["start_frames"] = data_config.get("val_start_frames", [0, 69])
    else:
        data_config = select_clip_names_from_start(data_config)
    # Encoder switches are authoritative for loader IO. Disabled controls keep
    # zero placeholders in the shared sample schema but never open their files.
    data_config["load_rendered_views"] = camera_enabled
    data_config["load_hands"] = hand_enabled
    data_config["require_hands"] = hand_enabled and bool(data_config.get("require_hands", True))
    dataset_name = str(data_config.get("name", "hot3d_preprocessed")).lower()
    is_hot3d_dataset = dataset_name in {"hot3d", "hot3d_preprocessed"}
    if is_hot3d_dataset:
        if obj_enabled and not (camera_enabled or hand_enabled):
            # Object-only loading is exactly the standalone Stand-In dataset
            # path, including curated intervals and canonical-view cycling.
            dataset = build_hot3d_object_dataset(data_config)
            collate_fn = Hot3DObjectRopeDataset.collate_samples
        else:
            control_dataset = TrainHot3DControlDataset.from_config(data_config)
            if obj_enabled:
                object_config = dict(data_config)
                # Pair against discovered metadata rather than applying the
                # standalone object's artificial debug cycling length.
                object_config["length"] = 0
                object_dataset = build_hot3d_object_dataset(object_config)
                dataset = MultiEncoderHot3DDataset(control_dataset, object_dataset)
                collate_fn = MultiEncoderHot3DDataset.collate_samples
            else:
                dataset = control_dataset
                collate_fn = Hot3DControlDataset.collate_samples
    else:
        # Keep the existing synthetic dataset path available for smoke tests.
        if obj_enabled:
            raise ValueError("Object-reference training requires a HOT3D dataset.")
        dataset = build_camera_dataset(data_config)
        collate_fn = collate_camera_training_samples

    sampler: Sampler[int] | None = None
    drop_last = bool(data_config.get("drop_last", False))
    shuffle = bool(data_config.get("shuffle", not is_hot3d_dataset))
    # HOT3D ordering is start-frame-major. Its sampler may randomize clips
    # within each start-frame group, but it never mixes the groups themselves.
    if not validation and isinstance(dataset, Hot3DControlDataset) and (
        shuffle or distributed.world_size > 1
    ):
        sampler = StartFrameGroupedSampler(
            dataset,
            shuffle=shuffle,
            seed=int(config.get("seed", 23)),
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            drop_last=drop_last,
        )
    elif distributed.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=not validation and shuffle,
            seed=int(config.get("seed", 23)),
            drop_last=drop_last,
        )

    num_workers = int(data_config.get("num_workers", 4))
    dataloader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 1)),
        shuffle=(
            sampler is None
            and not validation
            and shuffle
        ),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(data_config.get("pin_memory", distributed.device.type == "cuda")),
        persistent_workers=bool(data_config.get("persistent_workers", num_workers > 0)) and num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    if len(dataloader) == 0:
        split_name = "validation" if validation else "training"
        raise RuntimeError(
            f"The {split_name} DataLoader is empty; check hot3d_data.root and batch_size."
        )
    return dataloader, sampler


def h2o_data_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge H2O overrides onto HOT3D defaults for shared loader settings."""

    data_config = dict(config.get("hot3d_data", {}))
    h2o_overrides = dict(config.get("h2o_data", {}))
    data_config.update(h2o_overrides)
    # H2O raw RGB is 1280x720, and its calibration loader applies a centered
    # 720x720 crop before scaling. Keep every H2O training image/video at the
    # fixed 480 square expected by this entrypoint.
    data_config["resolution"] = 480
    data_config["image_size"] = 480
    data_config["reference_image_size"] = 480
    return data_config


def h2o_interleaving_is_enabled(config: dict[str, Any]) -> bool:
    """Return whether periodic H2O optimizer steps are configured."""

    data_config = h2o_data_config(config)
    interval = int(data_config.get("hot3d_steps_per_h2o", 0) or 0)
    if interval < 0:
        raise ValueError("h2o_data.hot3d_steps_per_h2o must be non-negative.")
    return interval > 0


def training_dataset_name(config: dict[str, Any], optimizer_step: int) -> str:
    """Choose HOT3D or H2O for one zero-based optimizer step."""

    if not h2o_interleaving_is_enabled(config):
        return "hot3d"
    hot3d_steps = int(h2o_data_config(config)["hot3d_steps_per_h2o"])
    return (
        "h2o"
        if int(optimizer_step) % (hot3d_steps + 1) == hot3d_steps
        else "hot3d"
    )


def build_h2o_dataloader(
    config: dict[str, Any],
    distributed: DistributedContext,
    *,
    validation: bool = False,
    data_overrides: dict[str, Any] | None = None,
) -> tuple[DataLoader, Sampler[int] | None]:
    """Build one prepared H2O train or validation loader."""

    data_config = h2o_data_config(config)
    if data_overrides:
        data_config.update(data_overrides)
    if validation:
        data_config["subjects"] = data_config.get("val_subjects", ["subject4_ego"])
        data_config["start_frames"] = data_config.get("val_start_frames", [0])
        data_config["length"] = int(data_config.get("val_length", 0) or 0)
    else:
        data_config["subjects"] = data_config.get(
            "subjects", ["subject1_ego", "subject2_ego", "subject3_ego"]
        )
        # HOT3D debug-length settings must not leak into H2O training.
        data_config["length"] = int(data_config.get("train_length", 0) or 0)

    camera_enabled = encoder_is_enabled(config, "camera_encoder")
    hand_enabled = encoder_is_enabled(config, "hand_encoder")
    obj_enabled = encoder_is_enabled(config, "obj_encoder")
    data_config["load_rendered_views"] = camera_enabled
    data_config["load_hands"] = hand_enabled
    data_config["require_hands"] = hand_enabled and bool(
        data_config.get("require_hands", True)
    )
    requested_length = int(data_config.get("length", 0) or 0)
    control_config = dict(data_config)
    if obj_enabled:
        control_config["length"] = 0
    control_dataset = TrainH2OControlDataset.from_config(control_config)
    if obj_enabled:
        object_stage = str(data_config.get("_object_reference_stage", "canonical"))
        object_dataset = H2OObjectDataset(
            control_dataset,
            object_stage,
            reference_image_size=data_config.get("reference_image_size", 480),
            load_video=not (camera_enabled or hand_enabled),
            length=requested_length,
        )
        if camera_enabled or hand_enabled:
            dataset: Dataset[Any] = MultiEncoderHot3DDataset(
                control_dataset, object_dataset
            )
            collate_fn = MultiEncoderHot3DDataset.collate_samples
        else:
            dataset = object_dataset
            collate_fn = Hot3DObjectRopeDataset.collate_samples
    else:
        dataset = control_dataset
        collate_fn = H2OControlDataset.collate_samples

    sampler: Sampler[int] | None = None
    drop_last = bool(data_config.get("drop_last", False))
    shuffle = bool(data_config.get("shuffle", False))
    if not validation and isinstance(dataset, H2OControlDataset) and (
        shuffle or distributed.world_size > 1
    ):
        sampler = StartFrameGroupedSampler(
            dataset,
            shuffle=shuffle,
            seed=int(config.get("seed", 23)),
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            drop_last=drop_last,
        )
    elif distributed.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=not validation and shuffle,
            seed=int(config.get("seed", 23)),
            drop_last=drop_last,
        )

    num_workers = int(data_config.get("num_workers", 4))
    dataloader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 1)),
        shuffle=sampler is None and not validation and shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(data_config.get("pin_memory", distributed.device.type == "cuda")),
        persistent_workers=bool(data_config.get("persistent_workers", num_workers > 0))
        and num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    if len(dataloader) == 0:
        raise RuntimeError("The H2O DataLoader is empty; check h2o_data settings.")
    return dataloader, sampler


def combine_validation_dataloaders(
    config: dict[str, Any],
    distributed: DistributedContext,
    hot3d_loader: DataLoader,
    h2o_loader: DataLoader | None,
) -> DataLoader:
    """Append ordered H2O validation samples after the HOT3D validation grid."""

    if h2o_loader is None:
        return hot3d_loader
    dataset = DescribedConcatDataset([hot3d_loader.dataset, h2o_loader.dataset])
    sampler: Sampler[int] | None = None
    if distributed.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=False,
            seed=int(config.get("seed", 23)),
            drop_last=False,
        )
    data_config = dict(config.get("hot3d_data", {}))
    num_workers = int(data_config.get("num_workers", 4))
    if encoder_is_enabled(config, "obj_encoder") and (
        encoder_is_enabled(config, "camera_encoder")
        or encoder_is_enabled(config, "hand_encoder")
    ):
        collate_fn = MultiEncoderHot3DDataset.collate_samples
    elif encoder_is_enabled(config, "obj_encoder"):
        collate_fn = Hot3DObjectRopeDataset.collate_samples
    else:
        collate_fn = Hot3DControlDataset.collate_samples
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 1)),
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(data_config.get("pin_memory", distributed.device.type == "cuda")),
        persistent_workers=bool(data_config.get("persistent_workers", num_workers > 0))
        and num_workers > 0,
        drop_last=False,
        collate_fn=collate_fn,
    )


def object_reference_stage(config: dict[str, Any], optimizer_step: int) -> str:
    """Return the object-reference source for a zero-based optimizer step."""

    train_config = dict(config.get("train", {}))
    stage_a_steps = int(train_config.get("obj_stageA_steps", 0) or 0)
    stage_b_steps = int(train_config.get("obj_stageB_steps", 0) or 0)
    if stage_a_steps < 0 or stage_b_steps < 0:
        raise ValueError("Object-reference stage step counts must be non-negative.")
    if optimizer_step < stage_a_steps:
        return "obj_stageA"
    if optimizer_step < stage_a_steps + stage_b_steps:
        return "obj_stageB"
    return "canonical"


def build_object_stage_dataloaders(
    config: dict[str, Any],
    distributed: DistributedContext,
) -> dict[str, tuple[DataLoader, Sampler[int] | None]]:
    """Build fixed-source loaders for the optimizer-step object schedule."""

    train_config = dict(config.get("train", {}))
    has_stage_schedule = (
        "obj_stageA_steps" in train_config or "obj_stageB_steps" in train_config
    )
    if not encoder_is_enabled(config, "obj_encoder") or not has_stage_schedule:
        loader, sampler = build_dataloader(config, distributed)
        return {"default": (loader, sampler)}
    return {
        stage: build_dataloader(
            config,
            distributed,
            data_overrides={"_object_reference_stage": stage},
        )
        for stage in ("obj_stageA", "obj_stageB", "canonical")
    }


def build_h2o_object_stage_dataloaders(
    config: dict[str, Any],
    distributed: DistributedContext,
) -> dict[str, tuple[DataLoader, Sampler[int] | None]]:
    """Build H2O loaders for every shared object-reference stage."""

    train_config = dict(config.get("train", {}))
    has_stage_schedule = (
        "obj_stageA_steps" in train_config or "obj_stageB_steps" in train_config
    )
    if not encoder_is_enabled(config, "obj_encoder") or not has_stage_schedule:
        loader, sampler = build_h2o_dataloader(config, distributed)
        return {"default": (loader, sampler)}
    return {
        stage: build_h2o_dataloader(
            config,
            distributed,
            data_overrides={"_object_reference_stage": stage},
        )
        for stage in ("obj_stageA", "obj_stageB", "canonical")
    }


def training_loader_key(config: dict[str, Any], optimizer_step: int) -> str:
    """Return the dataset/stage loader key for one optimizer step."""

    dataset_name = training_dataset_name(config, optimizer_step)
    train_config = dict(config.get("train", {}))
    has_stage_schedule = (
        "obj_stageA_steps" in train_config or "obj_stageB_steps" in train_config
    )
    stage = (
        object_reference_stage(config, optimizer_step)
        if encoder_is_enabled(config, "obj_encoder") and has_stage_schedule
        else "default"
    )
    return f"{dataset_name}:{stage}"


def build_training_dataloaders(
    config: dict[str, Any],
    distributed: DistributedContext,
) -> dict[str, tuple[DataLoader, Sampler[int] | None]]:
    """Build all HOT3D and enabled H2O loaders used by the step schedule."""

    loaders = {
        f"hot3d:{stage}": loader
        for stage, loader in build_object_stage_dataloaders(
            config, distributed
        ).items()
    }
    if h2o_interleaving_is_enabled(config):
        loaders.update(
            {
                f"h2o:{stage}": loader
                for stage, loader in build_h2o_object_stage_dataloaders(
                    config, distributed
                ).items()
            }
        )
    return loaders


def warn_if_object_stages_exceed_steps(config: dict[str, Any]) -> None:
    """Warn when the configured object curriculum extends past training."""

    train_config = dict(config.get("train", {}))
    stage_steps = int(train_config.get("obj_stageA_steps", 0) or 0) + int(
        train_config.get("obj_stageB_steps", 0) or 0
    )
    total_steps = int(train_config.get("steps", train_config.get("max_steps", 1000)))
    if stage_steps > total_steps:
        LOGGER.warning(
            "Object-reference stages total %d steps "
            "(train.obj_stageA_steps + train.obj_stageB_steps), exceeding "
            "train.steps=%d; Stage C out-of-interval curated training will not be reached.",
            stage_steps,
            total_steps,
        )


def validation_is_enabled(config: dict[str, Any]) -> bool:
    """Return whether validation has a cadence and explicit clip/frame grid."""

    train_config = dict(config.get("train", {}))
    data_config = dict(config.get("hot3d_data", config.get("data", {})))
    val_clips = data_config.get("val_clips", [])
    val_start_frames = data_config.get("val_start_frames", [])
    return (
        int(train_config.get("val_every_n_steps", 0) or 0) > 0
        and bool(val_clips)
        and bool(val_start_frames)
    )


def log_batch_shapes(batch: dict[str, Any]) -> None:
    """Log the tensor schema produced by the configured dataset."""

    shapes = {
        key: tuple(value.shape)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    LOGGER.info("Batch tensor shapes: %s", shapes)


def preview_loaded_dataset_steps(dataloader: DataLoader, steps: int) -> dict[str, Any]:
    """Load and print sample selections for the requested number of training steps."""

    iterator = iter(dataloader)
    first_batch: dict[str, Any] | None = None
    for step in range(int(steps)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        if first_batch is None:
            first_batch = batch
        tensor_shapes = {
            key: tuple(value.shape)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        LOGGER.info(
            "Loaded dataset step %d: clip=%s interval=%s caption=%s "
            "rgb=%s rendered=%s hand=%s tensor_shapes=%s",
            step + 1,
            batch.get("clip_name"),
            batch.get("interval"),
            batch.get("caption"),
            batch.get("video_path"),
            batch.get("rendered_video_path"),
            batch.get("hand_path"),
            tensor_shapes,
        )
    if first_batch is None:
        raise ValueError("preview steps must be positive.")
    return first_batch


def describe_loaded_data_item(dataset: Dataset[Any], index: int) -> dict[str, Any]:
    """Describe one selected sample without calling the decoding ``__getitem__``."""

    if isinstance(dataset, Subset):
        source_index = int(dataset.indices[index % len(dataset)])
        return describe_loaded_data_item(dataset.dataset, source_index)
    describe_item = getattr(dataset, "describe_item_files", None)
    if not callable(describe_item):
        raise TypeError(
            f"{type(dataset).__name__} must implement describe_item_files() "
            "for metadata-only dry runs."
        )
    metadata = dict(describe_item(index))
    interval = metadata.get("interval")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        metadata.setdefault("start_frame", int(interval[0]))
        metadata.setdefault("end_frame", int(interval[1]))
    return metadata


def loaded_data_metadata_batch(
    dataset: Dataset[Any],
    indices: Any,
    cache: dict[int, dict[str, Any]],
) -> dict[str, list[Any]]:
    """Collate cached file descriptions for one sampler-selected batch."""

    if isinstance(indices, torch.Tensor):
        selected_indices = [int(index) for index in indices.flatten().tolist()]
    elif isinstance(indices, (list, tuple)):
        selected_indices = [int(index) for index in indices]
    else:
        selected_indices = [int(indices)]
    samples = []
    for index in selected_indices:
        if index not in cache:
            cache[index] = describe_loaded_data_item(dataset, index)
        samples.append(cache[index])
    keys = set().union(*(sample.keys() for sample in samples))
    return {key: [sample.get(key) for sample in samples] for key in keys}


def log_dry_run_loaded_data(
    config: dict[str, Any],
    dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    sampler: Sampler[int] | None,
    loaded_data_logger: LoadedDataLogger,
    staged_loaders: dict[str, tuple[DataLoader, Sampler[int] | None]] | None = None,
) -> tuple[int, int]:
    """Audit sampler-selected metadata without decoding any dataset tensors."""

    train_config = dict(config.get("train", {}))
    total_steps = int(train_config.get("steps", train_config.get("max_steps", 1000)))
    accumulation_steps = max(
        1,
        int(train_config.get("gradient_accumulation_steps", 1)),
    )
    val_every = max(0, int(train_config.get("val_every_n_steps", 0)))
    if total_steps < 0:
        raise ValueError("train.steps must be non-negative.")

    loaded_data_logger.reset()
    epoch = 0
    data_step = 0
    validation_batches = 0
    train_cache: dict[str, dict[int, dict[str, Any]]] = {}
    validation_cache: dict[int, dict[str, Any]] = {}
    active_loaders = staged_loaders or {"default": (dataloader, sampler)}
    stage_epochs = {stage: 0 for stage in active_loaders}
    batch_indices_iterators = {}
    for stage, (stage_loader, stage_sampler) in active_loaders.items():
        if stage_sampler is not None:
            stage_sampler.set_epoch(0)
        batch_indices_iterators[stage] = iter(stage_loader.batch_sampler)

    for optimizer_step in range(1, total_steps + 1):
        stage = (
            training_loader_key(config, optimizer_step - 1)
            if staged_loaders
            else "default"
        )
        stage_loader, stage_sampler = active_loaders[stage]
        for _ in range(accumulation_steps):
            try:
                batch_indices = next(batch_indices_iterators[stage])
            except StopIteration:
                stage_epochs[stage] += 1
                epoch = max(epoch, stage_epochs[stage])
                if stage_sampler is not None:
                    stage_sampler.set_epoch(stage_epochs[stage])
                batch_indices_iterators[stage] = iter(stage_loader.batch_sampler)
                batch_indices = next(batch_indices_iterators[stage])
            batch = loaded_data_metadata_batch(
                stage_loader.dataset,
                batch_indices,
                train_cache.setdefault(stage, {}),
            )
            data_step += 1
            loaded_data_logger.log(batch, data_step)

        if (
            val_dataloader is not None
            and val_every
            and optimizer_step % val_every == 0
        ):
            for validation_batch, batch_indices in enumerate(
                val_dataloader.batch_sampler,
                start=1,
            ):
                batch = loaded_data_metadata_batch(
                    val_dataloader.dataset,
                    batch_indices,
                    validation_cache,
                )
                loaded_data_logger.log_validation(
                    batch,
                    optimizer_step,
                    validation_batch,
                )
                validation_batches += 1

    return data_step, validation_batches


def model_kwargs_from_config(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Translate YAML sections into ``EgoHOIControlModel`` constructor options."""

    wan_config = dict(config.get("wan", {}))
    camera_config = dict(config.get("camera_encoder", {}))
    hand_config = dict(config.get("hand_encoder", {}))
    obj_config = dict(config.get("obj_encoder", {}))
    train_config = dict(config.get("train", {}))
    allow_legacy_obj_lora_config = bool(config.get("_allow_legacy_obj_lora_config", False))
    if encoder_is_enabled(config, "obj_encoder") and not allow_legacy_obj_lora_config:
        forbidden_obj_keys = sorted(
            key for key in ("target_modules", "train_attn_blocks") if key in obj_config
        )
        if forbidden_obj_keys:
            raise ValueError(
                "obj_encoder target_modules and train_attn_blocks are fixed internals; "
                f"remove these stale config keys: {forbidden_obj_keys}."
            )
    obj_injected_layers = obj_config.get("injected_layers", list(range(20, 40)))
    obj_lora_targets = (
        obj_config.get("target_modules", ["self_attn.q", "self_attn.k", "self_attn.v"])
        if allow_legacy_obj_lora_config
        else ["self_attn.k", "self_attn.v"]
    )
    obj_train_attn_blocks = (
        obj_config.get("train_attn_blocks") if allow_legacy_obj_lora_config else None
    )
    precision = str(train_config.get("mixed_precision", "bf16")).lower()
    transformer_dtype = None
    if device.type == "cuda" and precision == "fp16":
        transformer_dtype = torch.float16
    elif device.type == "cuda" and precision == "bf16":
        transformer_dtype = torch.bfloat16

    return {
        "wan_checkpoint_dir": wan_config.get("checkpoint_dir", "wan2_1/I2V-14B-720P"),
        "wan_upstream_path": wan_config.get("upstream_path", "wan2_1"),
        "device": device,
        "injected_layers": camera_config.get("injected_layers"),
        "num_train_timesteps": int(wan_config.get("num_train_timesteps", 1000)),
        "vae_checkpoint": str(wan_config.get("vae_checkpoint", "Wan2.1_VAE.pth")),
        "camera_encoder_enabled": encoder_is_enabled(config, "camera_encoder"),
        "hand_encoder_enabled": encoder_is_enabled(config, "hand_encoder"),
        "obj_encoder_enabled": encoder_is_enabled(config, "obj_encoder"),
        "camera_encoder_trainable": encoder_train_is_enabled(config, "camera_encoder"),
        "hand_encoder_trainable": encoder_train_is_enabled(config, "hand_encoder"),
        "obj_encoder_trainable": encoder_train_is_enabled(config, "obj_encoder"),
        "plucker_hidden_dim": int(camera_config.get("plucker_hidden_dim", 256)),
        "plucker_feature_channels": int(camera_config.get("plucker_feature_channels", 6)),
        "camera_num_layers": int(camera_config.get("num_layers", 20)),
        "camera_control_dim": int(camera_config.get("control_dim", 1024)),
        "camera_conv_out_dim": int(camera_config.get("conv_out_dim", 5120)),
        "camera_ffn_dim": int(camera_config.get("ffn_dim", 4096)),
        "camera_num_heads": int(camera_config.get("num_heads", 16)),
        "camera_time_frequency_dim": int(camera_config.get("time_frequency_dim", 256)),
        "camera_time_dim": int(camera_config.get("time_dim", 5120)),
        "camera_rope_max_seq_len": int(camera_config.get("rope_max_seq_len", 1024)),
        "camera_qk_norm": camera_config.get("qk_norm", "rms_norm_across_heads"),
        "camera_eps": float(camera_config.get("eps", 1e-6)),
        "hand_concat_dim": int(hand_config.get("concat_dim", 4)),
        "hand_reference_enabled": bool(hand_config.get("reference_enable", True)),
        "hand_reference_concat_dim": int(hand_config.get("reference_concat_dim", 4)),
        "hand_lora_enabled": bool(hand_config.get("lora_enable", True)),
        "hand_lora_rank": int(hand_config.get("lora_rank", 4)),
        "hand_lora_alpha": float(hand_config.get("lora_alpha", 4.0)),
        "hand_lora_target_modules": hand_config.get(
            "lora_target_modules", "q,k,v,o,ffn.0,ffn.2"
        ),
        "hand_lora_init_weights": hand_config.get("init_lora_weights", "kaiming"),
        "obj_lora_rank": int(obj_config.get("rank", 128)),
        "obj_lora_alpha": float(obj_config.get("alpha", 128.0)),
        "obj_lora_target_modules": obj_lora_targets,
        "obj_train_attn_blocks": obj_train_attn_blocks,
        "obj_injected_layers": obj_injected_layers,
        "obj_lora_dropout": float(obj_config.get("dropout", 0.0)),
        "obj_lora_init": str(obj_config.get("init", "zero")),
        "obj_rope": str(obj_config.get("rope", "tss-rope")),
        "obj_expected_attn_blocks": wan_config.get("expected_attn_blocks", 40),
        "load_text_encoder": bool(wan_config.get("load_text_encoder", True)),
        "load_image_encoder": bool(wan_config.get("load_image_encoder", True)),
        "clip_checkpoint": str(
            wan_config.get(
                "clip_checkpoint",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            )
        ),
        "clip_tokenizer": str(wan_config.get("clip_tokenizer", "xlm-roberta-large")),
        "transformer_dtype": transformer_dtype,
        "gradient_checkpointing": bool(train_config.get("gradient_checkpointing", True)),
    }


def build_optimizer(
    model: EgoHOIControlModel,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Create AdamW groups with v1-constant hand and LoRA learning rates.

    Camera parameters retain their existing base LR and scheduler label. The
    hand sequence/reference encoders and Wan LoRA use constant LR, matching
    v1's scheduler-free AdamW configuration.
    """

    optimizer_config = dict(config.get("optimizer", {}))
    hand_config = dict(config.get("hand_encoder", {}))
    betas = tuple(float(value) for value in optimizer_config.get("betas", [0.9, 0.999]))
    base_lr = float(optimizer_config.get("lr", 1e-5))
    lora_lr_value = hand_config.get("lora_learning_rate")
    lora_lr = base_lr if lora_lr_value is None else float(lora_lr_value)
    parameter_groups: list[dict[str, Any]] = []
    for group_name, parameters in model.trainable_parameter_groups().items():
        if not parameters:
            continue
        parameter_groups.append(
            {
                "params": parameters,
                "lr": lora_lr if group_name == "wan_lora" else base_lr,
                "group_name": group_name,
                "lr_schedule": "camera" if group_name == "camera" else "constant",
            }
        )
    if not parameter_groups:
        raise ValueError(
            "Training requires camera_encoder.enable_train, hand_encoder.enable_train, "
            "or obj_encoder.enable_train to be true; "
            "the frozen Wan backbone has no optimizer parameters."
        )
    return torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        betas=betas,
        eps=float(optimizer_config.get("eps", 1e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    """Schedule camera groups only; keep v1 hand and LoRA groups constant."""

    train_config = dict(config.get("train", {}))
    total_steps = max(1, int(train_config.get("steps", train_config.get("max_steps", 1000))))
    warmup_steps = max(0, int(train_config.get("warmup_steps", 0)))
    min_ratio = float(train_config.get("min_lr_ratio", 0.0))

    def lr_multiplier(step: int) -> float:
        """Return the existing camera warmup/cosine multiplier."""

        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    def constant_multiplier(step: int) -> float:
        """Return v1's invariant hand/LoRA learning-rate multiplier."""

        del step
        return 1.0

    multipliers = [
        lr_multiplier
        if parameter_group.get("lr_schedule") == "camera"
        else constant_multiplier
        for parameter_group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multipliers)


def maybe_load_hand_weights(model: EgoHOIControlModel, config: dict[str, Any]) -> None:
    """Optionally initialize the independent hand encoder from a legacy checkpoint."""

    hand_config = dict(config.get("hand_encoder", {}))
    if model.hand_encoder is None:
        return
    checkpoint_path = hand_config.get("pretrained_path")
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "trainable_modules" in payload:
        payload = payload["trainable_modules"].get("hand_encoder", payload)
    if isinstance(payload, dict) and "hand_encoder" in payload:
        payload = payload["hand_encoder"]
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Hand checkpoint must contain a state dictionary: {checkpoint_path}")
    model.hand_encoder.load_legacy_state_dict(payload, strict=bool(hand_config.get("strict", True)))
    LOGGER.info("Loaded hand encoder initialization from %s", checkpoint_path)


def branch_module_names(model: EgoHOIControlModel, section_name: str) -> tuple[str, ...]:
    """Return compact module names owned by one configured encoder branch."""

    active_modules = model.active_modules()
    if section_name == "camera_encoder":
        names = ("plucker_encoder", "camera_encoder")
    elif section_name == "hand_encoder":
        names = ("hand_encoder", "reference_hand_encoder", "wan_lora")
    elif section_name == "obj_encoder":
        names = ("obj_encoder",)
    else:
        raise ValueError(f"Unknown encoder section {section_name!r}.")
    return tuple(name for name in names if name in active_modules)


def _looks_like_state_dict(payload: Any) -> bool:
    """Return whether a mapping appears to be a raw module state dictionary."""

    return isinstance(payload, dict) and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    )


def _module_state_from_payload(
    payload: Any,
    module_name: str,
    module_names: tuple[str, ...],
) -> dict[str, torch.Tensor] | None:
    """Find one module state in compact or direct checkpoint payloads."""

    if not isinstance(payload, dict):
        return None
    for container_name in ("active_modules", "trainable_modules"):
        container = payload.get(container_name)
        if isinstance(container, dict) and isinstance(container.get(module_name), dict):
            return container[module_name]
    if isinstance(payload.get(module_name), dict):
        return payload[module_name]
    state_dict = payload.get("state_dict")
    if isinstance(state_dict, dict):
        if isinstance(state_dict.get(module_name), dict):
            return state_dict[module_name]
        if len(module_names) == 1 and _looks_like_state_dict(state_dict):
            return state_dict
    if len(module_names) == 1 and _looks_like_state_dict(payload):
        return payload
    return None


def load_encoder_branch_from_ckpt(
    model: EgoHOIControlModel,
    section_name: str,
    checkpoint_path: str | Path,
) -> None:
    """Strictly load one active encoder branch from a checkpoint path."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"{section_name}.load_from_ckpt does not exist: {path}")
    module_names = branch_module_names(model, section_name)
    if not module_names:
        raise ValueError(f"{section_name} is not active, so it cannot load {path}.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    active_modules = model.active_modules()
    missing_modules: list[str] = []
    for module_name in module_names:
        module_state = _module_state_from_payload(payload, module_name, module_names)
        if module_state is None:
            missing_modules.append(module_name)
            continue
        try:
            active_modules[module_name].load_state_dict(module_state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                f"Failed to load {section_name}.{module_name} from {path}: {error}"
            ) from error
    if missing_modules:
        raise KeyError(
            f"Checkpoint {path} is missing {section_name} module state(s): "
            + ", ".join(missing_modules)
        )
    LOGGER.info("Loaded %s initialization from %s", section_name, path)


def load_configured_encoder_branches(model: EgoHOIControlModel, config: dict[str, Any]) -> None:
    """Load every enabled encoder branch with a configured checkpoint path."""

    for section_name in ("camera_encoder", "hand_encoder", "obj_encoder"):
        section = dict(config.get(section_name, {}))
        if not encoder_is_enabled(config, section_name):
            continue
        checkpoint_path = section.get("load_from_ckpt")
        if checkpoint_path in (None, ""):
            continue
        load_encoder_branch_from_ckpt(model, section_name, checkpoint_path)
    model._set_trainable_modules()


def compact_checkpoint_modules(model: EgoHOIControlModel) -> dict[str, torch.nn.Module]:
    """Return compact modules to save without the frozen Wan backbone."""

    active_modules = getattr(model, "active_modules", None)
    if callable(active_modules):
        return active_modules()
    return model.trainable_modules()


def checkpoint_payload(
    model: EgoHOIControlModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    step: int,
    epoch: int,
    best_loss: float,
    best_average_loss: float,
    cumulative_loss_sum: float,
) -> dict[str, Any]:
    """Build a compact checkpoint without duplicating the frozen Wan backbone."""

    return {
        # Version 4 records the normalized object RoPE mode explicitly.
        "format_version": 4,
        "global_step": int(step),
        "epoch": int(epoch),
        # Save object metadata actually installed on the model, not just raw config.
        "obj_rope": model.obj_rope,
        "obj_injected_layers": (
            None
            if getattr(model, "obj_injected_layers", None) is None
            else list(model.obj_injected_layers)
        ),
        "best_loss": float(best_loss),
        "best_average_loss": float(best_average_loss),
        "cumulative_loss_sum": float(cumulative_loss_sum),
        "active_modules": {
            name: module.state_dict()
            for name, module in compact_checkpoint_modules(model).items()
        },
        # Keep the historical key populated for inference/checkpoint readers.
        "trainable_modules": {
            name: module.state_dict()
            for name, module in compact_checkpoint_modules(model).items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
    }


def save_checkpoint(
    path: Path,
    model: EgoHOIControlModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    step: int,
    epoch: int,
    best_loss: float,
    best_average_loss: float,
    cumulative_loss_sum: float,
) -> None:
    """Atomically save trainable modules and all state needed for resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            step,
            epoch,
            best_loss,
            best_average_loss,
            cumulative_loss_sum,
        ),
        temporary_path,
    )
    temporary_path.replace(path)
    LOGGER.info("Saved checkpoint: %s", path)


def load_checkpoint(
    path: Path,
    model: EgoHOIControlModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float, float, float]:
    """Restore trainable model, optimizer, scheduler, scaler, and loop counters."""

    payload = torch.load(path, map_location=device, weights_only=False)
    module_states = payload.get("active_modules", payload.get("trainable_modules", {}))
    for name, module in model.active_modules().items():
        if name not in module_states:
            raise KeyError(f"Checkpoint {path} is missing active module {name!r}.")
        module.load_state_dict(module_states[name], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler_state = payload.get("scaler")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    cumulative_loss_sum = float(payload.get("cumulative_loss_sum", 0.0))
    global_step = int(payload.get("global_step", 0))
    best_average_loss = payload.get("best_average_loss")
    if best_average_loss is None:
        if global_step > 0 and cumulative_loss_sum == 0.0:
            LOGGER.warning(
                "Checkpoint %s predates cumulative-average tracking; resumed "
                "train/cumulative_average will restart from this step.",
                path,
            )
        best_average_loss = float("inf")
    LOGGER.info("Resumed training state from %s", path)
    return (
        global_step,
        int(payload.get("epoch", 0)),
        float(payload.get("best_loss", float("inf"))),
        float(best_average_loss),
        cumulative_loss_sum,
    )


def remove_old_step_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    """Retain only the newest numbered checkpoints when retention is enabled."""

    if keep_last <= 0:
        return
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink()


def autocast_context(device: torch.device, precision: str):
    """Return the configured mixed-precision context for the current device."""

    precision = precision.lower()
    if device.type != "cuda" or precision not in {"bf16", "fp16"}:
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def prepare_forward_kwargs(batch: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Map dataset fields and per-window captions to model inputs."""

    # Dataset videos are [B,T,C,H,W]. The Wan target keeps that layout, while
    # camera renders/masks are converted to [B,C,T,H,W]. The hand encoder
    # accepts [B,T,3,H,W] directly and performs its own legacy preprocessing.
    video = batch.get("raw_video")
    if video is None:
        video = batch.get("video")
    if not isinstance(video, torch.Tensor):
        raise KeyError("Training batch must contain raw_video or video.")
    batch_size = int(video.shape[0])
    data_config = dict(config.get("hot3d_data", {}))
    default_prompt = str(data_config.get("caption", data_config.get("prompt", "")))
    batch_captions = batch.get("caption")
    if batch_captions is None:
        prompts = [default_prompt] * batch_size
    else:
        prompts = [
            str(caption).strip() if str(caption).strip() else default_prompt
            for caption in batch_captions
        ]
    forward_kwargs = {
        "video": video,
        "first_frame": batch.get("first_frame", video[:, 0]),
        "prompts": prompts,
    }
    if encoder_is_enabled(config, "camera_encoder"):
        forward_kwargs.update(
            {
                "rendered_video": batch["rendered_novel_views"]
                .permute(0, 2, 1, 3, 4)
                .contiguous(),
                "rendered_mask": batch["rendered_novel_view_masks"]
                .permute(0, 2, 1, 3, 4)
                .contiguous(),
                "intrinsics": batch["intrinsic"],
                "extrinsics": batch["extrinsic"],
            }
        )
    if encoder_is_enabled(config, "hand_encoder"):
        forward_kwargs["hand_video"] = batch["rendered_hands"]
    if encoder_is_enabled(config, "obj_encoder"):
        forward_kwargs["reference_images"] = batch["reference_images"]
    return forward_kwargs


def validation_timestep_values(config: dict[str, Any]) -> list[int]:
    """Return the validated fixed-timestep cycle from the train config."""

    train_config = dict(config.get("train", {}))
    num_train_timesteps = int(
        config.get("wan", {}).get("num_train_timesteps", 1000)
    )
    configured_timesteps = train_config.get(
        "val_timestep", num_train_timesteps // 2
    )
    if isinstance(configured_timesteps, (list, tuple)):
        timesteps = [int(timestep) for timestep in configured_timesteps]
    else:
        # Keep scalar configurations backward compatible by treating them as
        # a one-value cycle.
        timesteps = [int(configured_timesteps)]
    if not timesteps:
        raise ValueError("train.val_timestep must contain at least one timestep.")
    if len(set(timesteps)) != len(timesteps):
        raise ValueError("train.val_timestep must not contain duplicate values.")
    for timestep in timesteps:
        if timestep < 0 or timestep >= num_train_timesteps:
            raise ValueError(
                "train.val_timestep values must be in "
                f"[0, {num_train_timesteps}), got {timestep}."
            )
    return timesteps


def fixed_validation_timestep(
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
    sample_offset: int = 0,
    sample_stride: int = 1,
    dataset_length: int | None = None,
) -> torch.Tensor:
    """Build cyclic fixed timesteps for one ordered validation batch."""

    timesteps = validation_timestep_values(config)
    batch_size = int(batch_size)
    sample_offset = int(sample_offset)
    sample_stride = int(sample_stride)
    timestep_cycle = torch.tensor(timesteps, device=device, dtype=torch.long)
    sample_indices = sample_offset + sample_stride * torch.arange(
        batch_size,
        device=device,
        dtype=torch.long,
    )
    # DistributedSampler pads with samples from the start when the dataset
    # length is not divisible by world size. Wrap those padded indices so they
    # receive the same timestep as the original validation sample.
    if dataset_length is not None:
        sample_indices.remainder_(int(dataset_length))
    cycle_indices = sample_indices.remainder(len(timesteps))
    return timestep_cycle[cycle_indices]


def fixed_validation_noise(
    video: torch.Tensor,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    batch_index: int,
) -> torch.Tensor:
    """Generate repeatable latent noise for one ordered validation batch.

    A batch-specific seed keeps the noise assigned to each validation batch
    stable across validation passes without retaining a large GPU-side cache.
    """

    if video.ndim != 5:
        raise ValueError(
            "Validation video must have shape [B, T, C, H, W], "
            f"got {tuple(video.shape)}."
        )
    stride = tuple(int(value) for value in getattr(model, "vae_stride", (4, 8, 8)))
    if len(stride) != 3 or any(value <= 0 for value in stride):
        raise ValueError(
            f"Model vae_stride must contain three positive values, got {stride}."
        )

    batch, frames, _, height, width = (int(value) for value in video.shape)
    latent_channels = int(config.get("wan", {}).get("latent_channels", 16))
    latent_shape = (
        batch,
        latent_channels,
        1 + max(0, frames - 1) // stride[0],
        height // stride[1],
        width // stride[2],
    )
    noise_seed = int(
        config.get("train", {}).get("val_noise_seed", config.get("seed", 23))
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed + int(batch_index))
    return torch.randn(latent_shape, generator=generator, device=device)


def validation_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: dict[str, Any],
    distributed: DistributedContext,
    precision: str,
    *,
    return_timestep_losses: bool = False,
) -> float | tuple[float, dict[int, float]]:
    """Measure average and optional per-timestep validation losses."""

    model.eval()
    configured_timesteps = validation_timestep_values(config)
    loss_sum = 0.0
    sample_count = 0
    timestep_loss_sums = {timestep: 0.0 for timestep in configured_timesteps}
    timestep_sample_counts = {timestep: 0 for timestep in configured_timesteps}
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                forward_kwargs = prepare_forward_kwargs(batch, config)
                video = forward_kwargs["video"]
                batch_size = int(video.shape[0])
                timesteps = fixed_validation_timestep(
                    config,
                    batch_size=batch_size,
                    device=distributed.device,
                    sample_offset=(
                        distributed.rank + sample_count * distributed.world_size
                    ),
                    sample_stride=distributed.world_size,
                    dataset_length=len(dataloader.dataset),
                )
                forward_kwargs["timesteps"] = timesteps
                forward_kwargs["noise"] = fixed_validation_noise(
                    video,
                    config,
                    model,
                    distributed.device,
                    batch_index,
                )
                with autocast_context(distributed.device, precision):
                    output = model(**forward_kwargs)
                # The model exposes a scalar batch loss, but a batch may span
                # several configured timesteps. Compute the same MSE per
                # sample so every timestep receives its own exact statistic.
                # Converting one sample at a time avoids retaining full-batch
                # float32 copies of the large video-latent tensors.
                per_sample_losses = torch.stack(
                    [
                        prediction.detach()
                        .float()
                        .sub_(target.detach())
                        .square_()
                        .mean()
                        for prediction, target in zip(
                            output.prediction,
                            output.target,
                            strict=True,
                        )
                    ]
                )
                loss_sum += float(per_sample_losses.sum().item())
                sample_count += batch_size
                for timestep, sample_loss in zip(
                    timesteps.detach().cpu().tolist(),
                    per_sample_losses.cpu().tolist(),
                    strict=True,
                ):
                    timestep = int(timestep)
                    timestep_loss_sums[timestep] = (
                        timestep_loss_sums.get(timestep, 0.0) + float(sample_loss)
                    )
                    timestep_sample_counts[timestep] = (
                        timestep_sample_counts.get(timestep, 0) + 1
                    )
                # Validation has no backward pass. Release every CUDA output
                # and generated noise tensor before evaluating the next batch.
                del output, per_sample_losses, timesteps, forward_kwargs, video, batch
    finally:
        model.train()

    totals = torch.tensor(
        [loss_sum, float(sample_count)]
        + [timestep_loss_sums[timestep] for timestep in configured_timesteps]
        + [float(timestep_sample_counts[timestep]) for timestep in configured_timesteps],
        device=distributed.device,
        dtype=torch.float64,
    )
    if distributed.world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_samples = int(totals[1].item())
    if total_samples == 0:
        raise RuntimeError("The validation DataLoader did not yield any batches.")
    mean_loss = float(totals[0].item() / total_samples)
    if not return_timestep_losses:
        return mean_loss

    timestep_count = len(configured_timesteps)
    timestep_losses = {
        timestep: float(
            totals[2 + index].item()
            / totals[2 + timestep_count + index].item()
        )
        for index, timestep in enumerate(configured_timesteps)
        if totals[2 + timestep_count + index].item() > 0
    }
    return mean_loss, timestep_losses


def train(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Run data loading, distributed setup, optimization, logging, and checkpoints."""

    distributed = initialize_distributed(str(config.get("device", "auto")))
    output_dir = Path(config.get("output_dir", "outputs/egohoi_control"))
    configure_python_logging(output_dir, distributed.is_main)
    loaded_data_logger = LoadedDataLogger(
        config,
        output_dir,
        distributed.is_main,
        force_all=args.dry_run,
    )
    validate_encoder_training_config(config)
    seed_everything(int(config.get("seed", 23)) + distributed.rank)
    if distributed.is_main:
        warn_if_object_stages_exceed_steps(config)

    staged_loaders = build_training_dataloaders(config, distributed)
    initial_stage = training_loader_key(config, 0)
    dataloader, sampler = staged_loaders[initial_stage]
    val_dataloader = None
    if validation_is_enabled(config):
        val_overrides = (
            {"_object_reference_stage": "canonical"}
            if encoder_is_enabled(config, "obj_encoder")
            else None
        )
        hot3d_val_loader, _ = build_dataloader(
            config,
            distributed,
            validation=True,
            data_overrides=val_overrides,
        )
        h2o_val_loader = None
        if int(h2o_data_config(config).get("val_length", 0) or 0) > 0:
            h2o_val_loader, _ = build_h2o_dataloader(
                config,
                distributed,
                validation=True,
                data_overrides=val_overrides,
            )
        val_dataloader = combine_validation_dataloaders(
            config,
            distributed,
            hot3d_val_loader,
            h2o_val_loader,
        )
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    if args.dry_run:
        data_steps, validation_batches = log_dry_run_loaded_data(
            config,
            dataloader,
            val_dataloader,
            sampler,
            loaded_data_logger,
            staged_loaders,
        )
        LOGGER.info(
            "Dry run complete; recorded %d training data steps and %d "
            "validation batches in %s without constructing the model.",
            data_steps,
            validation_batches,
            loaded_data_logger.log_path,
        )
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    preview_steps = max(0, int(getattr(args, "preview_steps", 0)))
    if preview_steps and distributed.is_main:
        first_batch = preview_loaded_dataset_steps(dataloader, preview_steps)
    else:
        first_batch = next(iter(dataloader))
    if distributed.is_main:
        log_batch_shapes(first_batch)
    # The startup batch is used only for schema validation. Keeping it here
    # would retain a complete CPU/pinned video batch for the entire model run.
    del first_batch

    model = EgoHOIControlModel(**model_kwargs_from_config(config, distributed.device))
    maybe_load_hand_weights(model, config)
    load_configured_encoder_branches(model, config)
    model.train()
    train_model: torch.nn.Module = model
    if distributed.world_size > 1:
        train_model = DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    train_config = dict(config.get("train", {}))
    precision = str(train_config.get("mixed_precision", "bf16"))
    scaler = torch.cuda.amp.GradScaler(enabled=distributed.device.type == "cuda" and precision == "fp16")
    total_steps = int(train_config.get("steps", train_config.get("max_steps", 1000)))
    accumulation_steps = max(1, int(train_config.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(train_config.get("max_grad_norm", 1.0))
    log_every = max(1, int(train_config.get("log_every_n_steps", 1)))
    save_every = max(0, int(train_config.get("save_every_n_steps", 100)))
    val_every = max(0, int(train_config.get("val_every_n_steps", 0)))
    keep_last = max(0, int(train_config.get("keep_last_checkpoints", 3)))
    checkpoint_dir = output_dir / "checkpoints"
    metrics_logger = TrainingLogger(config, output_dir, distributed.is_main)

    global_step, epoch, best_loss = 0, 0, float("inf")
    best_average_loss = float("inf")
    resume_path = args.resume or train_config.get("resume_from")
    sum_loss = 0.0
    if resume_path:
        global_step, epoch, best_loss, best_average_loss, sum_loss = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, scaler, distributed.device
        )

    trainable_parameters = model.trainable_parameters()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    running_loss = 0.0
    optimizer_step_timesteps: list[int] = []
    stage_epochs = {stage: epoch for stage in staged_loaders}
    data_iterators: dict[str, Iterator[Any]] = {}
    for stage, (stage_loader, stage_sampler) in staged_loaders.items():
        if stage_sampler is not None:
            stage_sampler.set_epoch(epoch)
        data_iterators[stage] = iter(stage_loader)
    data_step = global_step * accumulation_steps

    try:
        while global_step < total_steps:
            stage = training_loader_key(config, global_step)
            dataloader, sampler = staged_loaders[stage]
            try:
                batch = next(data_iterators[stage])
            except StopIteration:
                stage_epochs[stage] += 1
                epoch = max(epoch, stage_epochs[stage])
                if sampler is not None:
                    sampler.set_epoch(stage_epochs[stage])
                data_iterators[stage] = iter(dataloader)
                batch = next(data_iterators[stage])

            data_step += 1
            loaded_data_logger.log(batch, data_step)
            forward_kwargs = prepare_forward_kwargs(batch, config)
            with autocast_context(distributed.device, precision):
                output = train_model(**forward_kwargs)
                loss = output.loss
                scaled_loss = loss / accumulation_steps
            loss_value = float(loss.detach().item())
            # Retain only scalar timestep values so the optimizer-step log can
            # report every sample across gradient-accumulation microbatches.
            optimizer_step_timesteps.extend(
                int(timestep) for timestep in output.timesteps.detach().cpu().tolist()
            )
            # The loss graph retains only tensors required by backward. Drop
            # diagnostic output fields and host batch references before the
            # backward peak and, crucially, before constructing the next step.
            del output, forward_kwargs, batch
            scaler.scale(scaled_loss).backward()
            del scaled_loss, loss
            running_loss += loss_value
            micro_step += 1
            if micro_step % accumulation_steps:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            mean_loss = running_loss / accumulation_steps
            running_loss = 0.0
            if distributed.world_size > 1:
                reduced_loss = torch.tensor(mean_loss, device=distributed.device)
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                mean_loss = float(reduced_loss.item() / distributed.world_size)
            sum_loss += mean_loss
            cumulative_average_loss = sum_loss / global_step
            is_best_step = mean_loss < best_loss
            if is_best_step:
                best_loss = mean_loss
            is_best_average_step = cumulative_average_loss < best_average_loss
            if is_best_average_step:
                best_average_loss = cumulative_average_loss
            metrics = {
                "train/loss": mean_loss,
                "train/cumulative_average": cumulative_average_loss,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/grad_norm": float(grad_norm),
                "train/epoch": float(epoch),
            }
            for parameter_group in optimizer.param_groups:
                group_name = str(parameter_group.get("group_name", "parameters"))
                metrics[f"train/lr_{group_name}"] = float(parameter_group["lr"])
            if val_dataloader is not None and val_every and global_step % val_every == 0:
                mean_val_loss, timestep_val_losses = validation_loss(
                    model,
                    val_dataloader,
                    config,
                    distributed,
                    precision,
                    return_timestep_losses=True,
                )
                metrics["val/loss"] = mean_val_loss
                for timestep, timestep_loss in timestep_val_losses.items():
                    metrics[f"val/loss_timestep_{timestep}"] = timestep_loss
                if distributed.is_main:
                    LOGGER.info(
                        "step=%d/%d validation_loss=%.6f "
                        "validation_timestep_losses=%s validation_samples=%d",
                        global_step,
                        total_steps,
                        mean_val_loss,
                        ",".join(
                            f"{timestep}:{loss:.6f}"
                            for timestep, loss in timestep_val_losses.items()
                        ),
                        len(val_dataloader.dataset),
                    )

            should_log = global_step % log_every == 0 or "val/loss" in metrics
            if distributed.is_main and should_log:
                LOGGER.info(
                    "step=%d/%d timestep=%s epoch=%d loss=%.6f "
                    "cumulative_average=%.6f lr=%.3e grad_norm=%.4f",
                    global_step,
                    total_steps,
                    ",".join(str(timestep) for timestep in optimizer_step_timesteps),
                    epoch,
                    mean_loss,
                    metrics["train/cumulative_average"],
                    metrics["train/lr"],
                    metrics["train/grad_norm"],
                )
                metrics_logger.log(metrics, global_step)
            optimizer_step_timesteps.clear()

            # Disabled per training preference: do not write a moving
            # best-loss checkpoint during training.
            # if distributed.is_main and is_best_step:
            #     save_checkpoint(
            #         checkpoint_dir / "best.pt",
            #         model,
            #         optimizer,
            #         scheduler,
            #         scaler,
            #         config,
            #         global_step,
            #         epoch,
            #         best_loss,
            #         best_average_loss,
            #         sum_loss,
            #     )

            # Disabled per training preference: do not write a moving
            # best-running-average checkpoint during training.
            # if distributed.is_main and is_best_average_step:
            #     save_checkpoint(
            #         checkpoint_dir / "best_average.pt",
            #         model,
            #         optimizer,
            #         scheduler,
            #         scaler,
            #         config,
            #         global_step,
            #         epoch,
            #         best_loss,
            #         best_average_loss,
            #         sum_loss,
            #     )

            if distributed.is_main and save_every and global_step % save_every == 0:
                checkpoint_path = checkpoint_dir / f"step_{global_step:08d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    global_step,
                    epoch,
                    best_loss,
                    best_average_loss,
                    sum_loss,
                )
                remove_old_step_checkpoints(checkpoint_dir, keep_last)

        if distributed.is_main:
            save_checkpoint(
                checkpoint_dir / "final.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                global_step,
                epoch,
                best_loss,
                best_average_loss,
                sum_loss,
            )
    finally:
        metrics_logger.close()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


# DO NOT delete comments here!
def default_config() -> dict[str, Any]:
    """Return a runnable camera, hand, and Stand-In object configuration."""
    # DO NOT delete comments here!
    return {
        # Legacy trainer name; the three enable flags select any branch combination.
        "trainer": "camera_hand",
        "seed": 23,
        "device": "auto",
        "output_dir": "outputs/smoke_dry_run",
        "distributed": {
            "requested_gpus": 1,
        },
        "wan": {
            "checkpoint_dir": "wan2_1/I2V-14B-720P",
            "upstream_path": "wan2_1",
            "vae_checkpoint": "Wan2.1_VAE.pth",
            "num_train_timesteps": 1000,
            "load_text_encoder": True,
            "load_image_encoder": True,
            "clip_checkpoint": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "clip_tokenizer": "xlm-roberta-large",
            "expected_attn_blocks": 40,
        },
        "camera_encoder": {
            # Disable to skip rendered-view IO, construction, training, and
            # camera residual injection into Wan blocks.
            "enable": True,
            "enable_train": True,
            "load_from_ckpt": None,
            # Omit injected_layers to place one residual in each configured
            # camera-ControlNet layer, starting from the first Wan block.
            # First 20 blocks.
            "injected_layers": list(range(20)),
            # Plucker rays and masks are encoded at Wan token width before
            # entering the compact camera-ControlNet residual branch.
            "plucker_hidden_dim": 256,
            "plucker_feature_channels": 6,
            # Uni3C fuses point-render and Plucker/mask features at Wan width
            # before the shared projection into the compact 1024D controller.
            "conv_out_dim": 5120,
            "control_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16, # head dimension=64=1024/16
            "num_layers": 20,
            "time_frequency_dim": 256,
            "time_dim": 5120,
            "rope_max_seq_len": 1024,
            "qk_norm": "rms_norm_across_heads",
            "eps": 1.0e-6,
        },
        "hand_encoder": {
            # Disable to skip rendered-hand IO and every v1 hand adaptation.
            "enable": False,
            "enable_train": True,
            "load_from_ckpt": None,
            # concat_dim=4 mirrors the legacy 16-channel Conv3d hand stack.
            "concat_dim": 4, # Conv3d internal width 4 × concat_dim = 4*4 = 16
            # V1's first hand frame is injected into the 20-channel
            # image_emb["y"] condition before Wan patch embedding.
            "reference_enable": True,
            "reference_concat_dim": 4, # Conv2d internal width 4 × reference_concat_dim = 4*4 = 16
            # V1 adapts the frozen Wan DiT base through these rank-4 adapters.
            "lora_enable": True,
            "lora_rank": 128,
            "lora_alpha": 128,
            "lora_target_modules": "q,k,v,o,ffn.0,ffn.2",
            "init_lora_weights": "kaiming",
            # Null uses optimizer.lr, exactly like v1's fallback behavior.
            "lora_learning_rate": None,
            "pretrained_path": None,
            "strict": True,
        },
        "obj_encoder": {
            # Disabled by default so existing camera/hand jobs retain their
            # original architecture and dataset IO.
            "enable": False,
            "enable_train": True,
            "load_from_ckpt": None,
            # tss-rope / vanilla-rope / ts-rope / ss-rope
            # tss-rope: temporal -1, shifted spatial grid
            # vanilla-rope: trailing temporal frames, shared spatial positions
            # ts-rope: temporal -1, shared spatial positions
            # ss-rope: trailing temporal frames, shifted spatial grid
            "rope": "tss-rope",
            # Object tokens update in every Wan block, while video tokens read
            # object K/V only on these zero-based self-attention layers.
            "injected_layers": list(range(20, 40)),
            "rank": 128, # 128
            "alpha": 128.0, # 128.0
            "dropout": 0.0,
            "init": "zero",
        },
        "hot3d_data": {
            "name": "hot3d_preprocessed",
            "root": "outputs/hot3d_preprocessed",
            # "val_root": "outputs/hot3d_preprocessed_val",
            "val_root": "outputs/hot3d_preprocessed",
            # Set to 1949 to include clip-001949 and every later clip.
            "start_clip": None,
            "start_frames": [0, 10, 20, 30, 40, 50, 60, 69],
            # Validation uses every explicit clip id at every validation start frame.
            # generated from
            # random.seed(23)
            # random_integers = [random.randint(1849, 3364) for _ in range(100)]
            "val_clips": [1877, 1883, 1889, 1899, 1921, 1928, 1966,
                          2017, 2020, 2024, 2050, 2054, 2065,
                          2068, 2072, 2083, 2093, 2115, 2140, 2155,
                          2184, 2191, 2201, 2212, 2214, 2221, 2242,
                          2249, 2270, 2291, 2296, 2301, 2336, 2378,
                          2396, 2413, 2423, 2425, 2439, 2442, 2473,
                          2477, 2492, 2511, 2521, 2526, 2582, 2584,
                          2591, 2604, 2606, 2625, 2633, 2711,
                          2716, 2721, 2732, 2759, 2765, 2780,
                          2819, 2838, 2858, 2869, 2898, 2914, 2934,
                          2941, 2959, 2966, 2973, 3007, 3012, 3020,
                          3022, 3041, 3055, 3057, 3058, 3061,
                          3062, 3118, 3119, 3120, 3134, 3164, 3196,
                          3201, 3228, 3232, 3297, 3303, 3319, 3335,
                          3352, 3354],
            "val_start_frames": [0], # [0,69]
            "clip_names": None,
            "stream_id": "214-1",
            "video_frames": 81,
            "caption_annotations_path": "outputs/hot3d_qwen_captions.json",
            "view_render_root": "hand_view_render", # "hand_view_render" or "view_render_old"
            "resolution": 480,
            # Object-reference loader settings are inert unless
            # obj_encoder.enable is true.
            "view_source": "curated", # "curated" or "canonical"
            # Each available start-frame interval contributes its own three
            # curated images; intervals without a complete triplet are skipped.
            "curated_metadata_glob": "clip-*/curated_views/curated_views.json",
            "metadata_glob": "clip-*/canonical_views/*_canonical_views.json",
            "reference_image_size": 480,
            "reference_view_labels": [
                "elev_000_az_0120",
                "elev_000_az_0240",
                "front",
            ],
            # First train with three curated object views, then two curated
            # views plus one out-of-interval curated view, then three
            # out-of-interval curated views.
            # Stage C, still internally keyed as "canonical" for schedule compatibility,
            # If no out-of-interval curated views exist,
            # Stage B/C fall back to the current interval’s curated views.
            # Staged loading no longer requires canonical metadata for Stage B/C.
            "num_reference_views": 3,
            "max_object_num": 1,
            "strict_view_labels": False,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": True,
            # "drop_last": False,
            "load_rendered_views": True,
            "require_hands": True,
            "caption": "An egocentric video showing human hands interacting with everyday objects.",
            # Preserve start-frame groups; true only shuffles clips per group.
            "shuffle": False,
        },
        "h2o_data": {
            "name": "h2o_preprocessed",
            "root": "outputs/h2o_preprocessed",
            "raw_root": "outputs/h2o_preprocessed",
            # After this many HOT3D optimizer steps, train the next step on H2O.
            # A value of 19 produces the repeating schedule 19 HOT3D + 1 H2O.
            "hot3d_steps_per_h2o": 19,
            "subjects": [
                "subject1_ego",
                "subject2_ego",
                # "subject3_ego",
            ],
            "val_subjects": ["subject3_ego"],
            # Starts are intentionally not clamped globally. Each short clip
            # contributes only starts whose complete 81-frame files exist.
            "start_frames": [
                0, 20, 40, 60, 80, 100, 120, 140,
                160, 180, 200, 220, 240, 260, 280, 300,
                320, 340, 360, 380, 400, 420, 440, 460,
                480, 500, 520, 540, 560, 580, 600, 620,
                640, 660, 680, 700, 720, 740, 760, 780,
                800, 820, 840, 860, 880, 900, 920, 940,
                960, 980, 1000
            ],
            # Validation takes this many ordered subject-4 windows. H2O uses a
            # length because it does not have HOT3D's explicit val_clips grid.
            "val_length": 10,
            "val_start_frames": [0],
            "caption_annotations_path": (
                "outputs/h2o_preprocessed/h2o_action_label_captions.json"
            ),
            "view_render_root": "hand_view_render",
            "resolution": 480,
            "reference_image_size": 480,
        },
        "optimizer": {
            "lr": 1.0e-5,
            # "betas": [0.9, 0.999],
            # "eps": 1.0e-8,
            "weight_decay": 0.0,
        },
        "train": {
            "steps": 15000,
            # Shared by HOT3D and H2O: three in-interval object views, then
            # two in-interval plus one out-of-interval view, then three
            # out-of-interval views. Stage B/C fall back to current-interval
            # views when an object has no cross-interval observations.
            "obj_stageA_steps": 1500,
            "obj_stageB_steps": 1500,
            # "warmup_steps": 50,
            # "min_lr_ratio": 0.1,
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            # "gradient_accumulation_steps": 1,
            # "max_grad_norm": 1.0,
            "log_every_n_steps": 1,
            # Run no-gradient validation every N optimizer steps; 0 disables validation.
            "val_every_n_steps": 500,
            # Cycle these timesteps over ordered validation samples and reuse
            # deterministic noise so losses stay comparable across passes.
            "val_timestep": [357, 682, 833, 921, 978],
            "val_noise_seed": 23,
            "save_every_n_steps": 1000,
            # "keep_last_checkpoints": 3,
            "resume_from": None,
        },
        "logging": {
            # Disabled by default. Enable it to record source paths, intervals, and
            # captions for every loaded data step.
            "loaded_data": {
                "enabled": True,
                "every_n_steps": 1,
            },
            "tensorboard": {
                "enabled": False,
                "log_dir": "outputs/egohoi_control/tensorboard",
            },
            "wandb": {
                "enabled": True,
                "project": "conditional-egoworld",
                "name": "hand",
                "entity": None,
                "mode": "offline", # or online
            },
        },
    }


def encoder_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a validated encoder config mapping."""

    section = config.get(section_name, {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise TypeError(f"{section_name} must be a mapping, got {type(section).__name__}.")
    return section


def encoder_is_enabled(config: dict[str, Any], section_name: str) -> bool:
    """Return whether an encoder branch participates in the run."""

    section = encoder_section(config, section_name)
    enabled = section.get("enable", section_name != "obj_encoder")
    if not isinstance(enabled, bool):
        raise TypeError(
            f"{section_name}.enable must be true or false, got {enabled!r}."
        )
    return enabled


def encoder_train_is_enabled(config: dict[str, Any], section_name: str) -> bool:
    """Return whether an active encoder branch should receive optimizer updates."""

    if not encoder_is_enabled(config, section_name):
        return False
    section = encoder_section(config, section_name)
    enable_train = section.get("enable_train", True)
    if not isinstance(enable_train, bool):
        raise TypeError(
            f"{section_name}.enable_train must be true or false, got {enable_train!r}."
        )
    return enable_train


def validate_encoder_training_config(config: dict[str, Any]) -> None:
    """Validate training-only encoder checkpoint requirements."""

    for section_name in ("camera_encoder", "hand_encoder", "obj_encoder"):
        if not encoder_is_enabled(config, section_name):
            continue
        section = encoder_section(config, section_name)
        if encoder_train_is_enabled(config, section_name):
            continue
        if section.get("load_from_ckpt") in (None, ""):
            raise ValueError(
                f"{section_name}.enable=true with enable_train=false requires "
                f"{section_name}.load_from_ckpt to point to a checkpoint."
            )


def maybe_auto_launch(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Relaunch under torchrun when the config requests multiple GPUs."""

    if args.no_auto_launch or os.environ.get("WORLD_SIZE"):
        return
    requested_gpus = int(config.get("distributed", {}).get("requested_gpus", 1))
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if requested_gpus <= 1 or available_gpus < requested_gpus:
        return
    forwarded_args = [arg for arg in sys.argv[1:] if arg != "--no-auto-launch"]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(requested_gpus),
        str(Path(__file__).resolve()),
        *forwarded_args,
        "--no-auto-launch",
    ]
    raise SystemExit(subprocess.call(command))


def limit_default_config_for_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Shrink the built-in config so bare ``--dry-run`` is a quick smoke audit."""

    train_config = config.setdefault("train", {})
    train_config.update(
        {
            "steps": 20,
            "obj_stageA_steps": 5,
            "obj_stageB_steps": 5,
            "val_every_n_steps": 5,
            "save_every_n_steps": 0,
        }
    )

    hot3d_config = config.setdefault("hot3d_data", {})
    hot3d_config["start_frames"] = list(hot3d_config.get("start_frames", [0]))[:2]
    hot3d_config["val_clips"] = list(hot3d_config.get("val_clips", []))[:2]
    hot3d_config["val_start_frames"] = list(
        hot3d_config.get("val_start_frames", [0])
    )[:1]

    h2o_config = config.setdefault("h2o_data", {})
    h2o_config["subjects"] = list(h2o_config.get("subjects", []))[:1]
    h2o_config["val_subjects"] = list(h2o_config.get("val_subjects", []))[:1]
    h2o_config["start_frames"] = list(h2o_config.get("start_frames", [0]))[:2]
    h2o_config["val_start_frames"] = list(h2o_config.get("val_start_frames", [0]))[:1]
    h2o_config["val_length"] = min(2, int(h2o_config.get("val_length", 0) or 0))
    return config


def main() -> None:
    """Load defaults, optionally relaunch for multi-GPU, and train."""

    args = parse_args()
    config = load_config(args.config) if args.config else default_config()
    if args.dry_run and args.config is None:
        config = limit_default_config_for_dry_run(config)
    config = apply_overrides(config, args.overrides)
    if args.requested_gpus is not None:
        config.setdefault("distributed", {})["requested_gpus"] = int(args.requested_gpus)
    if args.start_clip is not None:
        config.setdefault("hot3d_data", {})["start_clip"] = int(args.start_clip)
    trainer_name = str(config.get("trainer", "camera_hand"))
    if trainer_name != "camera_hand":
        raise ValueError(
            f"scripts/model/train.py expects trainer='camera_hand', got {trainer_name!r}."
        )
    maybe_auto_launch(config, args)
    train(config, args)


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    main()
