"""Metadata-only logging for all training-loader steps in one JSON file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch


LOGGER = logging.getLogger("egohoi.train")


class LoadedDataLogger:
    """Record source paths, intervals, and captions without copying loaded data."""

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: Path,
        enabled: bool,
        *,
        force_all: bool = False,
    ) -> None:
        """Use the output directory and optionally audit every dry-run batch."""

        logging_config = dict(config.get("logging", {}))
        loaded_data_config = dict(logging_config.get("loaded_data", {}))
        self.enabled = bool(
            enabled and (force_all or loaded_data_config.get("enabled", False))
        )
        self.log_path = output_dir / "loaded_data.json"
        self.every_n_steps = (
            1
            if force_all
            else max(1, int(loaded_data_config.get("every_n_steps", 1)))
        )

    def reset(self) -> Path | None:
        """Start a fresh audit file when a dry run must describe only this run."""

        if not self.enabled:
            return None
        self._write_payload({"steps": [], "validations": []})
        return self.log_path

    def log(self, batch: dict[str, Any], step: int) -> Path | None:
        """Add or replace one training data step in the shared audit file."""

        if not self.enabled or int(step) % self.every_n_steps:
            return None

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        batch_size = _loaded_batch_size(batch)
        step_payload = {
            "data_step": int(step),
            "batch_size": batch_size,
            "samples": [
                _sample_metadata(batch, sample_index)
                for sample_index in range(batch_size)
            ],
        }
        payload = self._read_payload()
        retained_steps = [
            item
            for item in payload["steps"]
            if int(item.get("data_step", -1)) != int(step)
        ]
        payload["steps"] = sorted(
            [*retained_steps, step_payload],
            key=lambda item: int(item["data_step"]),
        )
        self._write_payload(payload)
        LOGGER.info("Recorded loaded data step %d in %s", step, self.log_path)
        return self.log_path

    def log_validation(
        self,
        batch: dict[str, Any],
        optimizer_step: int,
        batch_index: int,
    ) -> Path | None:
        """Add or replace one batch from a scheduled validation pass."""

        if not self.enabled:
            return None

        batch_size = _loaded_batch_size(batch)
        batch_payload = {
            "validation_batch": int(batch_index),
            "batch_size": batch_size,
            "samples": [
                _sample_metadata(batch, sample_index)
                for sample_index in range(batch_size)
            ],
        }
        payload = self._read_payload()
        validation_payload = next(
            (
                item
                for item in payload["validations"]
                if int(item.get("optimizer_step", -1)) == int(optimizer_step)
            ),
            None,
        )
        if validation_payload is None:
            validation_payload = {
                "optimizer_step": int(optimizer_step),
                "batches": [],
            }
            payload["validations"].append(validation_payload)
        retained_batches = [
            item
            for item in validation_payload["batches"]
            if int(item.get("validation_batch", -1)) != int(batch_index)
        ]
        validation_payload["batches"] = sorted(
            [*retained_batches, batch_payload],
            key=lambda item: int(item["validation_batch"]),
        )
        payload["validations"] = sorted(
            payload["validations"],
            key=lambda item: int(item["optimizer_step"]),
        )
        self._write_payload(payload)
        LOGGER.info(
            "Recorded validation batch %d at optimizer step %d in %s",
            batch_index,
            optimizer_step,
            self.log_path,
        )
        return self.log_path

    def _read_payload(self) -> dict[str, list[dict[str, Any]]]:
        """Load existing records so resumed training preserves prior steps."""

        if not self.log_path.is_file():
            return {"steps": [], "validations": []}
        with self.log_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError(f"{self.log_path} must contain a JSON 'steps' list.")
        validations = payload.get("validations", [])
        if not isinstance(validations, list):
            raise ValueError(
                f"{self.log_path} must contain a JSON 'validations' list."
            )
        return {"steps": steps, "validations": validations}

    def _write_payload(self, payload: dict[str, list[dict[str, Any]]]) -> None:
        """Atomically persist the complete metadata-only audit payload."""

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.log_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary_path.replace(self.log_path)


def _loaded_batch_size(batch: dict[str, Any]) -> int:
    """Infer batch size from a tensor or collated metadata sequence."""

    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    for value in batch.values():
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Cannot log loaded data from a batch with no samples.")


def _sample_metadata(batch: dict[str, Any], sample_index: int) -> dict[str, Any]:
    """Build source-path metadata for one collated loader sample."""

    start_frame = int(_batch_item(batch, "start_frame", sample_index, 0))
    end_frame = int(
        _batch_item(
            batch,
            "end_frame",
            sample_index,
            start_frame + _video_frame_count(batch) - 1,
        )
    )
    return {
        "sample_index": sample_index,
        "clip_name": str(_batch_item(batch, "clip_name", sample_index, "")),
        "caption": str(_batch_item(batch, "caption", sample_index, "")),
        "video": {
            "path": _path_item(batch, "video_path", sample_index),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "interval": [start_frame, end_frame],
        },
        "hand_clip": {
            "path": _path_item(batch, "hand_path", sample_index),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "interval": [start_frame, end_frame],
        },
        "camera_rendered_view": {
            "video_path": _path_item(batch, "rendered_video_path", sample_index),
            "mask_path": _path_item(batch, "rendered_mask_path", sample_index),
            "camera_info_path": _path_item(batch, "camera_info_path", sample_index),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "interval": [start_frame, end_frame],
        },
        "object_images": _path_list_item(batch, "reference_paths", sample_index),
        "object_view_stage": _batch_item(
            batch,
            "object_view_stage",
            sample_index,
            None,
        ),
        "object_metadata_path": _path_item(batch, "metadata_path", sample_index),
        "object_uid": _batch_item(batch, "object_uid", sample_index, None),
        "object_id": _batch_item(batch, "object_id", sample_index, None),
        "object_name": _batch_item(batch, "object_name", sample_index, None),
    }


def _video_frame_count(batch: dict[str, Any]) -> int:
    """Return the loaded video length for interval fallback metadata."""

    for key in ("raw_video", "video"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return int(value.shape[1])
    return 1


def _batch_item(
    batch: dict[str, Any],
    key: str,
    sample_index: int,
    default: Any,
) -> Any:
    """Return one collated metadata value without indexing scalar strings."""

    value = batch.get(key)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        selected = value[sample_index]
        return selected.item() if selected.numel() == 1 else selected.tolist()
    if isinstance(value, (list, tuple)):
        return value[sample_index] if sample_index < len(value) else default
    return value


def _path_item(batch: dict[str, Any], key: str, sample_index: int) -> str | None:
    """Return one source path as a string or null when it was not loaded."""

    value = _batch_item(batch, key, sample_index, None)
    return None if value is None else str(value)


def _path_list_item(
    batch: dict[str, Any],
    key: str,
    sample_index: int,
) -> list[str]:
    """Return one sample's source image paths as strings."""

    value = _batch_item(batch, key, sample_index, [])
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(path) for path in value]
