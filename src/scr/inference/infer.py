"""Infer with any camera, hand, and object controls saved by scripts/model/train.py.

The checkpoint is the source of truth for enabled encoders, architecture widths,
injection layers, Wan paths, and precision. This module performs exactly one
inference call; use scripts/model/infer_batch.py for HOT3D/H2O batches.

Example:
    uv run python scripts/model/infer.py --device cuda:7
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import nullcontext, suppress
from copy import deepcopy
from dataclasses import dataclass, replace
import gc
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping
import warnings

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Make repository imports work when this file is executed from any directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scr.engine.seed import seed_everything
from scr.engine.trainer import (
    EgoHOIControlModel,
    build_dense_plucker_rays,
    normalize_rendered_video,
)
from scr.engine.obj_stand_trainer import normalize_standin_rope_mode
from scr.h2o_loader import H2OControlDataset, H2OObjectDataset
from scr.hot3d_loader import (
    Hot3DControlDataset,
    Hot3DObjectRopeDataset,
    _load_camera_window,
    _load_normalized_video_frames,
)
from scr.models.encoders.camera_encoder import CameraEncoderOutput
from scr.inference.train import MultiEncoderHot3DDataset, model_kwargs_from_config

DEFAULT_CHECKPOINT_PATH = Path("outputs/second_train_ckpt/train_camera_obj/checkpoints/final.pt")
# DEFAULT_CHECKPOINT_PATH = Path("outputs/ckpt/train_hand_only/checkpoints/final.pt")
# DEFAULT_CHECKPOINT_PATH = Path("outputs/ckpt/train_camera_hand/checkpoints/final.pt")
DEFAULT_DATA_ROOT = Path("outputs/hot3d_preprocessed_test")
DEFAULT_OUTPUT_DIR = Path("outputs/second_train_ckpt/train_camera_obj/infer")
DEFAULT_CAPTION_ANNOTATIONS_PATH = Path("outputs/hot3d_qwen_captions.json")
DEFAULT_CLIP = "1877"
# DEFAULT_START_FRAMES may be a single int, such as (0), or a tuple/list.
# DEFAULT_START_FRAMES = (0, 69)
DEFAULT_START_FRAMES = (0,)
DEFAULT_MAX_CLIP = 1
DEFAULT_ZERO2 = False

# override default obj loading
obj_reference_1 = "outputs/hot3d/clip-001906/rotated_object_views/elev_000_az_090/000033_obj_000019_flask_uid-4111539686391.png"
obj_reference_2 = "outputs/hot3d/clip-001906/rotated_object_views/front/000033_obj_000019_flask_uid-4111539686391.png"
obj_reference_2 = "outputs/hot3d/clip-001906/rotated_object_views/elev_p60_az_000/000033_obj_000019_flask_uid-4111539686391.png"
# DEFAULT_REFERENCE_PATHS: list[str | Path] = [obj_reference_1,obj_reference_2,obj_reference_2]
DEFAULT_REFERENCE_PATHS = []

# Set this to a string to override captions loaded from the annotation JSON.
# DEFAULT_CAPTION: str | None = "hold the juice carton"
DEFAULT_CAPTION = None
DEFAULT_OUTPUT_NAME = "inference.mp4"

_RUNNER_CACHE: dict[tuple[str, str, str, bool], tuple[CheckpointSpec, AdaptiveInferenceRunner]] = {}

def parse_args() -> argparse.Namespace:
    """Parse checkpoint, data, sampling, device, and output options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--caption-annotations-path",
        type=Path,
        default=DEFAULT_CAPTION_ANNOTATIONS_PATH,
        help=(
            "Window-caption JSON used for inference prompts. Defaults to the "
            "HOT3D Qwen captions instead of the checkpoint's training caption "
            "path."
        ),
    )
    parser.add_argument(
        "--clip",
        default=DEFAULT_CLIP,
        help="First HOT3D clip id to infer, for example 1849 or clip-001849.",
    )
    parser.add_argument(
        "--max-clip",
        type=int,
        default=DEFAULT_MAX_CLIP,
        help="Number of consecutive source clips to process starting at --clip.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Backward-compatible single start frame. Overrides --start-frames.",
    )
    parser.add_argument(
        "--start-frames",
        type=int,
        nargs="+",
        default=list(normalize_default_start_frames(DEFAULT_START_FRAMES)),
        help="Frame starts for each 150-frame hand.mp4 source video.",
    )
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example cuda:6. Auto chooses CUDA when available.",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default=None,
        help="Override the precision stored under train.mixed_precision.",
    )
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--sample-solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--camera-scale", type=float, default=1.0)
    parser.add_argument("--hand-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Defaults to the annotation selected for the clip window.",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--max-area", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--zero2",
        action="store_true",
        default=DEFAULT_ZERO2,
        help="Load a DeepSpeed ZeRO-2 checkpoint through scripts/model/infer_zero2.py.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Resolve checkpoint and data metadata without loading Wan.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class CheckpointSpec:
    """Describe the adaptive architecture stored in one training checkpoint."""

    path: Path
    payload: dict[str, Any]
    config: dict[str, Any]
    module_states: dict[str, Mapping[str, torch.Tensor]]
    camera_enabled: bool
    hand_enabled: bool
    reference_hand_enabled: bool
    hand_lora_enabled: bool
    obj_enabled: bool = False
    obj_rope: str = "tss-rope"
    obj_injected_layers: list[int] | None = None

    @property
    def encoder_names(self) -> list[str]:
        """Return enabled control branch names in deterministic order."""

        names: list[str] = []
        if self.camera_enabled:
            names.append("camera_encoder")
        if self.hand_enabled:
            names.append("hand_encoder")
        if self.obj_enabled:
            names.append("obj_encoder")
        return names


@dataclass
class ControlCache:
    """Hold control tensors that do not change across denoising timesteps."""

    first_frame_condition: torch.Tensor
    clip_context: torch.Tensor
    camera_i2v_mask: torch.Tensor | None = None
    hand_tokens: torch.Tensor | None = None
    hand_token_grid: tuple[int, int, int] | None = None
    render_latent: torch.Tensor | None = None
    rendered_mask: torch.Tensor | None = None
    plucker_tokens: torch.Tensor | None = None
    reference_latents: torch.Tensor | None = None


@dataclass
class InferenceResult:
    """Store a generated video and metadata used in the saved summary."""

    video: torch.Tensor
    seed: int
    timesteps: list[int]
    encoder_names: list[str]
    injected_layers: list[int]
    hand_token_grid: tuple[int, int, int] | None
    camera_token_grid: tuple[int, int, int] | None


def resolve_workspace_path(path: str | Path) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    value = Path(path).expanduser()
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def normalize_image_size(image_size: int | Sequence[int] | None) -> tuple[int, int] | None:
    """Normalize an image-size config value to ``(height, width)``."""

    if image_size is None:
        return None
    if isinstance(image_size, int):
        return (int(image_size), int(image_size))
    values = list(image_size)
    if len(values) != 2:
        raise ValueError(f"image_size must be an int or pair, got {image_size}.")
    return (int(values[0]), int(values[1]))


def active_default_reference_paths() -> tuple[Path, ...]:
    """Return configured object-reference image paths, resolving repo relatives."""

    return tuple(
        resolve_workspace_path(path)
        for path in DEFAULT_REFERENCE_PATHS
        if str(path).strip()
    )


def load_reference_image(path: Path, image_size: tuple[int, int] | None) -> torch.Tensor:
    """Load one object-reference image as a ``[3, H, W]`` float tensor."""

    if not path.is_file():
        raise FileNotFoundError(f"Default object reference image not found: {path}")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            image = image.resize((int(image_size[1]), int(image_size[0])), resampling)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


def load_default_reference_images(data_config: Mapping[str, Any]) -> torch.Tensor | None:
    """Load configured default object references or return None for legacy loading."""

    reference_paths = active_default_reference_paths()
    if not reference_paths:
        return None
    image_size = normalize_image_size(
        data_config.get(
            "reference_image_size", data_config.get("canonical_image_size", 480)
        )
    )
    images = [
        load_reference_image(path, image_size=image_size)
        for path in reference_paths
    ]
    return torch.stack(images, dim=0)


def apply_default_reference_images(
    batch: dict[str, Any],
    reference_images: torch.Tensor | None,
) -> None:
    """Replace dataset object references in a collated batch when configured."""

    if reference_images is None:
        return
    batch_size = int(batch["first_frame"].shape[0])
    batch["reference_images"] = reference_images.unsqueeze(0).repeat(
        batch_size, 1, 1, 1, 1
    )
    batch["reference_paths"] = [
        [str(path) for path in active_default_reference_paths()]
        for _ in range(batch_size)
    ]


def describe_default_references(
    item: dict[str, Any],
    reference_images: torch.Tensor | None,
) -> dict[str, Any]:
    """Return an item description updated with configured default references."""

    if reference_images is None:
        return item
    updated = dict(item)
    updated["metadata_path"] = None
    updated["reference_paths"] = [str(path) for path in active_default_reference_paths()]
    updated["reference_source"] = "DEFAULT_REFERENCE_PATHS"
    return updated


def normalize_clip_name(value: str | int) -> str:
    """Normalize numeric and clip-prefixed HOT3D identifiers."""

    text = str(value).strip()
    if text.startswith("clip-"):
        text = text[len("clip-") :]
    if not text.isdigit():
        raise ValueError(f"Invalid HOT3D clip {value!r}.")
    return f"clip-{int(text):06d}"


def hot3d_clip_number(value: str | int) -> int:
    """Return the integer suffix of a normalized HOT3D clip identifier."""

    return int(normalize_clip_name(value).removeprefix("clip-"))


def resolve_clip_names(start_clip: str | int, max_clip: int | None) -> list[str]:
    """Return consecutive HOT3D clip names beginning with ``start_clip``."""

    clip_count = 1 if max_clip is None else int(max_clip)
    if clip_count <= 0:
        raise ValueError("--max-clip must be greater than zero.")
    first_clip = hot3d_clip_number(start_clip)
    return [normalize_clip_name(first_clip + offset) for offset in range(clip_count)]


def normalize_default_start_frames(start_frames: int | Sequence[int]) -> tuple[int, ...]:
    """Return configured default frame starts as an ordered tuple of integers."""

    if isinstance(start_frames, int):
        return (int(start_frames),)
    return tuple(int(frame) for frame in start_frames)


def resolve_start_frames(args: argparse.Namespace) -> list[int]:
    """Return the ordered frame windows requested by the inference CLI."""

    if args.start_frame is not None:
        return [int(args.start_frame)]
    return [int(frame) for frame in args.start_frames]


def resolve_checkpoint_obj_rope(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    """Resolve and cross-check object RoPE metadata with legacy fallback."""

    obj_section = config.get("obj_encoder", {})
    if obj_section is None:
        obj_section = {}
    if not isinstance(obj_section, Mapping):
        raise TypeError(
            "Checkpoint config section 'obj_encoder' must be a mapping."
        )

    config_has_rope = "rope" in obj_section
    config_rope = normalize_standin_rope_mode(obj_section.get("rope", "tss-rope"))
    metadata_rope = payload.get("obj_rope")
    if metadata_rope is None:
        # Version 3 and older checkpoints fall back to config; configs that
        # predate this option retain the original TSS-RoPE behavior.
        return config_rope

    saved_rope = normalize_standin_rope_mode(metadata_rope)
    if config_has_rope and saved_rope != config_rope:
        raise ValueError(
            "Checkpoint obj_rope metadata disagrees with config obj_encoder.rope: "
            f"metadata={saved_rope!r}, config={config_rope!r}."
        )
    return saved_rope


def normalize_obj_injected_layers(value: Any) -> list[int] | None:
    """Normalize object injected-layer metadata, preserving None as legacy all-layer."""

    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            "obj_encoder.injected_layers must be a sequence of zero-based layer indices or None."
        )
    return [int(index) for index in value]


def resolve_checkpoint_obj_injected_layers(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[int] | None:
    """Resolve object injected layers with top-level checkpoint metadata priority."""

    if "obj_injected_layers" in payload:
        return normalize_obj_injected_layers(payload.get("obj_injected_layers"))
    obj_section = config.get("obj_encoder", {})
    if obj_section is None:
        obj_section = {}
    if not isinstance(obj_section, Mapping):
        raise TypeError(
            "Checkpoint config section 'obj_encoder' must be a mapping."
        )
    if "injected_layers" in obj_section:
        return normalize_obj_injected_layers(obj_section.get("injected_layers"))
    return None


def load_checkpoint_spec(path: Path) -> CheckpointSpec:
    """Load and validate a compact scripts/model/train.py checkpoint on CPU."""

    resolved = resolve_workspace_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Inference checkpoint not found: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a mapping, got {type(payload).__name__}.")
    config = payload.get("config")
    # New checkpoints serialize every enabled compact branch, including frozen
    # controls. Fall back to the historical key for older training outputs.
    states = payload.get("active_modules", payload.get("trainable_modules"))
    if not isinstance(config, dict):
        raise KeyError("Checkpoint is missing its resolved config mapping.")
    if not isinstance(states, dict):
        raise KeyError(
            "Checkpoint is missing its active_modules/trainable_modules mapping."
        )

    has_plucker = "plucker_encoder" in states
    has_camera = "camera_encoder" in states
    if has_plucker != has_camera:
        raise KeyError(
            "Camera checkpoints require both plucker_encoder and camera_encoder."
        )
    camera_enabled = has_plucker and has_camera
    hand_enabled = "hand_encoder" in states
    reference_hand_enabled = "reference_hand_encoder" in states
    hand_lora_enabled = "wan_lora" in states
    obj_enabled = "obj_encoder" in states
    obj_rope = resolve_checkpoint_obj_rope(payload, config)
    obj_injected_layers = resolve_checkpoint_obj_injected_layers(payload, config)
    if (reference_hand_enabled or hand_lora_enabled) and not hand_enabled:
        raise KeyError(
            "Reference-hand and Wan LoRA checkpoint states require hand_encoder."
        )
    if not camera_enabled and not hand_enabled and not obj_enabled:
        raise ValueError(
            "Checkpoint contains no camera, hand, or object encoder state."
        )

    module_names = ["plucker_encoder", "camera_encoder"] if camera_enabled else []
    if hand_enabled:
        module_names.append("hand_encoder")
    if reference_hand_enabled:
        module_names.append("reference_hand_encoder")
    if hand_lora_enabled:
        module_names.append("wan_lora")
    if obj_enabled:
        module_names.append("obj_encoder")
    typed_states: dict[str, Mapping[str, torch.Tensor]] = {}
    for name in module_names:
        state = states[name]
        if not isinstance(state, Mapping):
            raise TypeError(f"Checkpoint module {name!r} is not a state dictionary.")
        typed_states[name] = state
    return CheckpointSpec(
        path=resolved,
        payload=payload,
        config=deepcopy(config),
        module_states=typed_states,
        camera_enabled=camera_enabled,
        hand_enabled=hand_enabled,
        reference_hand_enabled=reference_hand_enabled,
        hand_lora_enabled=hand_lora_enabled,
        obj_enabled=obj_enabled,
        obj_rope=obj_rope,
        obj_injected_layers=obj_injected_layers,
    )


def resolve_checkpoint_loader(
    args: argparse.Namespace,
) -> Callable[[Path], CheckpointSpec]:
    """Select the compact or ZeRO-2 checkpoint adapter for this run."""

    if not args.zero2:
        return load_checkpoint_spec

    from scr.inference import infer_zero2

    infer_zero2.prepare_optional_deepspeed_import()
    if args.checkpoint == DEFAULT_CHECKPOINT_PATH:
        args.checkpoint = infer_zero2.DEFAULT_CHECKPOINT_PATH
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = infer_zero2.DEFAULT_OUTPUT_DIR
    return infer_zero2.load_zero2_checkpoint_spec


def checkpoint_precision(config: dict[str, Any], override: str | None) -> str:
    """Return validated inference precision from CLI or checkpoint metadata."""

    value = override or str(config.get("train", {}).get("mixed_precision", "bf16"))
    aliases = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "none": "fp32",
        "no": "fp32",
    }
    precision = aliases.get(value.lower(), value.lower())
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported checkpoint precision {value!r}.")
    return precision


def transformer_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    """Translate precision metadata to the Wan transformer dtype."""

    if device.type != "cuda" or precision == "fp32":
        return None
    return torch.bfloat16 if precision == "bf16" else torch.float16


def model_kwargs_from_spec(
    spec: CheckpointSpec,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    """Use training's model factory with checkpoint-state branch adaptation.

    Calling ``scripts.train.model_kwargs_from_config`` keeps inference on the
    exact same constructor mapping as training. Serialized module names remain
    authoritative so legacy and partial checkpoints can enable any one, two,
    or all three control encoders even if their saved enable flags are stale.
    """

    config = deepcopy(spec.config)
    for section_name, enabled in (
        ("camera_encoder", spec.camera_enabled),
        ("hand_encoder", spec.hand_enabled),
        ("obj_encoder", spec.obj_enabled),
    ):
        section = config.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise TypeError(
                f"Checkpoint config section {section_name!r} must be a mapping."
            )
        section["enable"] = enabled

    # Make explicit checkpoint metadata authoritative for model construction,
    # including legacy checkpoints whose config omitted the field entirely.
    obj = config["obj_encoder"]
    obj["rope"] = spec.obj_rope
    obj["injected_layers"] = spec.obj_injected_layers
    if spec.obj_injected_layers is None:
        config["_allow_legacy_obj_lora_config"] = True
    else:
        obj.pop("target_modules", None)
        obj.pop("train_attn_blocks", None)
    # These optional hand modules have their own checkpoint entries. Derive
    # their construction from serialized state rather than current defaults.
    hand = config["hand_encoder"]
    hand["reference_enable"] = spec.reference_hand_enabled
    hand["lora_enable"] = spec.hand_lora_enabled
    train = config.setdefault("train", {})
    if not isinstance(train, dict):
        raise TypeError("Checkpoint config section 'train' must be a mapping.")
    train["mixed_precision"] = precision
    # Checkpointing changes execution/memory only and is unnecessary for eval.
    train["gradient_checkpointing"] = False

    kwargs = model_kwargs_from_config(config, device)
    # Training normally runs from the repository root. Resolve the two Wan
    # paths here so direct inference remains cwd-independent.
    # EgoSpace: the Wan package is vendored at src/wan (REPOSITORY_ROOT = src/),
    # and the Wan2.1 I2V-14B-720P weights are located by EGOSPACE_WAN_DIR.
    import os
    wan_dir = os.environ.get("EGOSPACE_WAN_DIR")
    kwargs["wan_checkpoint_dir"] = (
        Path(wan_dir).expanduser().resolve() if wan_dir
        else resolve_workspace_path(kwargs["wan_checkpoint_dir"])
    )
    kwargs["wan_upstream_path"] = REPOSITORY_ROOT
    return kwargs


def build_model(
    spec: CheckpointSpec,
    device: torch.device,
    precision: str,
) -> EgoHOIControlModel:
    """Instantiate the training architecture and strictly restore its controls."""

    model = EgoHOIControlModel(**model_kwargs_from_spec(spec, device, precision))
    # Inference must restore enabled frozen branches as well as branches that
    # were trainable. The saved module names are the authoritative topology.
    modules = model.active_modules()
    expected_names = set(modules)
    checkpoint_names = set(spec.module_states)
    if expected_names != checkpoint_names:
        raise KeyError(
            "Checkpoint/model active module mismatch: "
            f"missing={sorted(expected_names - checkpoint_names)}, "
            f"unexpected={sorted(checkpoint_names - expected_names)}."
        )
    for name, module in modules.items():
        module.load_state_dict(spec.module_states[name], strict=True)
    model.eval().requires_grad_(False)
    return model


def build_dataset(
    spec: CheckpointSpec,
    args: argparse.Namespace,
) -> tuple[Dataset[Any], dict[str, Any]]:
    """Build requested HOT3D windows using checkpoint data hyperparameters."""

    checkpoint_data = checkpoint_data_config(spec.config)
    frame_num = int(
        args.frame_num
        or checkpoint_data.get("video_frames", checkpoint_data.get("num_frames", 81))
    )
    resolution = args.resolution
    if resolution is None:
        resolution = checkpoint_data.get(
            "image_size",
            checkpoint_data.get(
                "resolution", checkpoint_data.get("video_image_size", 480)
            ),
        )
    data_config = deepcopy(checkpoint_data)
    data_config.update(
        {
            "root": str(resolve_workspace_path(args.data_root)),
            "clip_names": resolve_clip_names(args.clip, args.max_clip),
            "video_frames": frame_num,
            "image_size": resolution,
            "resolution": resolution,
            "start_frames": resolve_start_frames(args),
            "load_rendered_views": spec.camera_enabled,
            "load_hands": spec.hand_enabled,
            "require_hands": spec.hand_enabled,
            # Inference visits each discovered window once.
            "length": 0,
        }
    )
    caption_annotations_path = getattr(args, "caption_annotations_path", None)
    if caption_annotations_path is not None:
        data_config["caption_annotations_path"] = str(
            resolve_workspace_path(caption_annotations_path)
        )
    # Explicit inference clip selection supersedes training range selection.
    data_config.pop("start_clip", None)
    data_config.pop("hot3d_start_from", None)
    if spec.obj_enabled and not (spec.camera_enabled or spec.hand_enabled):
        dataset: Dataset[Any] = build_hot3d_dataset_component(
            Hot3DObjectRopeDataset, data_config
        )
    else:
        control_dataset = build_hot3d_dataset_component(
            Hot3DControlDataset, data_config
        )
        if spec.obj_enabled:
            object_dataset = build_hot3d_dataset_component(
                Hot3DObjectRopeDataset, data_config
            )
            dataset = MultiEncoderHot3DDataset(control_dataset, object_dataset)
        else:
            dataset = control_dataset
    return dataset, data_config


def build_hot3d_dataset_component(
    dataset_type: type[Hot3DControlDataset] | type[Hot3DObjectRopeDataset],
    data_config: dict[str, Any],
) -> Hot3DControlDataset | Hot3DObjectRopeDataset:
    """Build one HOT3D loader, tolerating legacy checkpoint caption schemas."""

    try:
        return dataset_type.from_config(data_config)
    except ValueError as error:
        message = str(error)
        if "Caption key " not in message or "must match" not in message:
            raise
        legacy_path = data_config.get("caption_annotations_path")
        warnings.warn(
            "Ignoring incompatible checkpoint caption annotations at "
            f"{legacy_path!r}; using the checkpoint's default caption. Pass "
            "--caption-annotations-path to select a current window-caption JSON.",
            stacklevel=2,
        )
        data_config["caption_annotations_path"] = None
        return dataset_type.from_config(data_config)


def build_loader(dataset: Dataset[Any], device: torch.device) -> DataLoader:
    """Create the ordered window inference loader."""

    if isinstance(dataset, (Hot3DControlDataset, H2OControlDataset)):
        collate_fn = Hot3DControlDataset.collate_samples
    elif isinstance(dataset, (Hot3DObjectRopeDataset, H2OObjectDataset)):
        collate_fn = Hot3DObjectRopeDataset.collate_samples
    elif isinstance(dataset, MultiEncoderHot3DDataset):
        collate_fn = MultiEncoderHot3DDataset.collate_samples
    else:
        raise TypeError(f"Unsupported inference dataset {type(dataset).__name__}.")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )


def collate_inference_item(dataset: Dataset[Any], index: int) -> dict[str, Any]:
    """Collate one item and expose a uniform raw-video/first-frame schema."""

    if isinstance(dataset, (Hot3DControlDataset, H2OControlDataset)):
        batch = Hot3DControlDataset.collate_samples([dataset[index]])
    elif isinstance(dataset, (Hot3DObjectRopeDataset, H2OObjectDataset)):
        batch = Hot3DObjectRopeDataset.collate_samples([dataset[index]])
    elif isinstance(dataset, MultiEncoderHot3DDataset):
        batch = MultiEncoderHot3DDataset.collate_samples([dataset[index]])
    else:
        raise TypeError(f"Unsupported inference dataset {type(dataset).__name__}.")
    if "raw_video" not in batch:
        batch["raw_video"] = batch["video"]
    if "first_frame" not in batch:
        batch["first_frame"] = batch["raw_video"][:, 0]
    return batch


def validate_inference_batch(
    spec: CheckpointSpec,
    batch: Mapping[str, Any],
) -> None:
    """Validate decoded source, render, hand, and object tensors before sampling."""

    required_tensors = {
        "first_frame": 4,
        "raw_video": 5,
    }
    if spec.camera_enabled:
        required_tensors.update(
            {
                "intrinsic": 4,
                "extrinsic": 4,
                "rendered_novel_views": 5,
                "rendered_novel_view_masks": 5,
            }
        )
    if spec.hand_enabled:
        required_tensors["rendered_hands"] = 5
    if spec.obj_enabled:
        required_tensors["reference_images"] = 5

    for name, dimensions in required_tensors.items():
        value = batch.get(name)
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                f"Inference with {', '.join(spec.encoder_names)} requires "
                f"decoded tensor {name!r}."
            )
        if value.ndim != dimensions:
            raise ValueError(
                f"Inference tensor {name!r} must be {dimensions}D, "
                f"got shape {tuple(value.shape)}."
            )

    batch_size = int(batch["first_frame"].shape[0])
    for name in required_tensors:
        if int(batch[name].shape[0]) != batch_size:
            raise ValueError(
                f"Inference tensor {name!r} has batch size "
                f"{batch[name].shape[0]}, expected {batch_size}."
            )
    if spec.obj_enabled and int(batch["reference_images"].shape[2]) != 3:
        raise ValueError(
            "reference_images must use [B, R, 3, H, W] RGB layout, got "
            f"{tuple(batch['reference_images'].shape)}."
        )


def describe_inference_item(dataset: Dataset[Any], index: int) -> dict[str, Any]:
    """Describe selected source/reference files without assuming one loader."""

    if isinstance(dataset, (Hot3DControlDataset, H2OControlDataset)):
        return dataset.describe_item_files(index)
    if isinstance(dataset, MultiEncoderHot3DDataset):
        control_index, object_index = dataset.pairs[index % len(dataset.pairs)]
        description = dataset.control_dataset.describe_item_files(control_index)
        object_item = dataset.object_dataset.items[object_index]
    elif isinstance(dataset, (Hot3DObjectRopeDataset, H2OObjectDataset)):
        object_item = dataset.items[index % len(dataset.items)]
        description = {
            "clip_name": object_item.clip_name,
            "interval": [int(object_item.start_frame), int(object_item.end_frame)],
            "video_path": str(object_item.video_path),
        }
    else:
        raise TypeError(f"Unsupported inference dataset {type(dataset).__name__}.")
    description["metadata_path"] = str(object_item.metadata_path)
    description["reference_paths"] = [
        str(path) for path in getattr(object_item, "reference_paths", ())
    ]
    return description


class AdaptiveInferenceRunner:
    """Sample Wan videos using every control encoder stored in a checkpoint."""

    def __init__(
        self,
        spec: CheckpointSpec,
        device: torch.device,
        precision: str,
    ) -> None:
        """Load the exact training architecture and freeze all modules."""

        self.spec = spec
        self.device = device
        self.precision = precision
        self.model = build_model(spec, device, precision)

    def close(self) -> None:
        """Release the model and reclaim cached CUDA allocations."""

        if hasattr(self, "model"):
            del self.model
        if self.device.type == "cuda":
            with suppress(Exception):
                torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
        gc.collect()

    def generate(
        self,
        batch: dict[str, Any],
        *,
        prompt: str | list[str],
        negative_prompt: str | list[str],
        frame_num: int,
        max_area: int | None,
        sampling_steps: int,
        shift: float,
        sample_solver: str,
        guide_scale: float,
        camera_scale: float,
        hand_scale: float,
        seed: int,
        show_progress: bool,
        attention_collector: Any | None = None,
    ) -> InferenceResult:
        """Denoise one batch with all checkpoint control branches."""

        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be greater than zero.")
        validate_inference_batch(self.spec, batch)
        first_frame = center_crop_image_tensor(batch["first_frame"].to(self.device))
        prepared = self._prepare_spatial_inputs(
            batch, first_frame, frame_num, max_area
        )
        target_height, target_width = prepared["target_size"]
        latent_grid = (
            (frame_num - 1) // int(self.model.vae_stride[0]) + 1,
            target_height // int(self.model.vae_stride[1]),
            target_width // int(self.model.vae_stride[2]),
        )
        batch_size = int(first_frame.shape[0])
        prompts = expand_prompt(prompt, batch_size)
        negative_prompts = expand_prompt(negative_prompt, batch_size)
        resolved_seed = resolve_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(resolved_seed)
        latent = torch.randn(
            batch_size,
            16,
            *latent_grid,
            dtype=torch.float32,
            generator=generator,
            device=self.device,
        )

        with self.autocast_context(), torch.no_grad():
            cache = self._prepare_controls(
                first_frame,
                prepared,
                latent,
                frame_num,
                hand_scale,
            )
            text_context = self.model._resolve_text_context(
                batch_size, latent.dtype, prompts, text_context=None
            )
            negative_context = self.model._resolve_text_context(
                batch_size, latent.dtype, negative_prompts, text_context=None
            )
            timesteps, scheduler = self._build_scheduler(
                sample_solver, sampling_steps, shift
            )
            last_camera: CameraEncoderOutput | None = None
            for step_index, timestep in enumerate(tqdm(
                timesteps, disable=not show_progress, desc="Denoising"
            )):
                timestep_batch = torch.full(
                    (batch_size,),
                    int(timestep.item()),
                    device=self.device,
                    dtype=torch.long,
                )
                camera = self._camera_residual(
                    latent, timestep_batch, cache, camera_scale
                )
                last_camera = camera
                if attention_collector is not None:
                    attention_collector.begin_conditional_evaluation(
                        step_index, int(timestep.item())
                    )
                try:
                    conditional = self.model._wan_forward_with_controls(
                        zt=latent,
                        first_frame_condition=cache.first_frame_condition,
                        timesteps=timestep_batch,
                        text_context=text_context,
                        clip_context=cache.clip_context,
                        camera=camera,
                        hand_tokens=cache.hand_tokens,
                        hand_token_grid=cache.hand_token_grid,
                        reference_latents=cache.reference_latents,
                    )
                finally:
                    if attention_collector is not None:
                        attention_collector.end_conditional_evaluation()
                unconditional = self.model._wan_forward_with_controls(
                    zt=latent,
                    first_frame_condition=cache.first_frame_condition,
                    timesteps=timestep_batch,
                    text_context=negative_context,
                    clip_context=cache.clip_context,
                    camera=camera,
                    hand_tokens=cache.hand_tokens,
                    hand_token_grid=cache.hand_token_grid,
                    reference_latents=cache.reference_latents,
                )
                prediction = unconditional + float(guide_scale) * (
                    conditional - unconditional
                )
                latent = scheduler.step(
                    prediction,
                    timestep,
                    latent,
                    return_dict=False,
                    generator=generator,
                )[0]
            videos = self.model.vae.decode([sample for sample in latent])

        return InferenceResult(
            video=torch.stack(videos, dim=0).detach().cpu(),
            seed=resolved_seed,
            timesteps=[int(value.item()) for value in timesteps],
            encoder_names=self.spec.encoder_names,
            injected_layers=list(self.model.injected_layers),
            hand_token_grid=cache.hand_token_grid,
            camera_token_grid=(
                last_camera.token_grid if last_camera is not None else None
            ),
        )

    def _prepare_spatial_inputs(
        self,
        batch: dict[str, Any],
        first_frame: torch.Tensor,
        frame_num: int,
        max_area: int | None,
    ) -> dict[str, Any]:
        """Move enabled controls to the device and align size and layout."""

        target_size = target_image_size(first_frame, max_area, self.model)
        prepared: dict[str, Any] = {"target_size": target_size}
        if self.spec.obj_enabled:
            prepared["reference_images"] = batch["reference_images"]
        if self.spec.hand_enabled:
            hand_video = ensure_bcthw(
                batch["rendered_hands"].to(self.device), 3, "rendered_hands"
            )
            hand_video = center_crop_video_tensor(rgb_hand_video_to_gray(hand_video))
            prepared["hand_video"] = resize_video(
                pad_or_trim_frames(hand_video, frame_num),
                target_size,
                mode="bilinear",
            )

        if self.spec.camera_enabled:
            rendered_video = ensure_bcthw(
                batch["rendered_novel_views"].to(self.device),
                3,
                "rendered_novel_views",
            )
            rendered_mask = ensure_bcthw(
                batch["rendered_novel_view_masks"].to(self.device),
                1,
                "rendered_novel_view_masks",
            )
            rendered_video = pad_or_trim_frames(rendered_video, frame_num)
            rendered_mask = pad_or_trim_frames(rendered_mask, frame_num)
            source_size = tuple(int(value) for value in rendered_video.shape[-2:])
            cropped_intrinsics, cropped_size = center_crop_intrinsics(
                batch["intrinsic"].to(self.device), source_size
            )
            rendered_video = center_crop_video_tensor(rendered_video)
            rendered_mask = center_crop_video_tensor(rendered_mask)
            prepared["rendered_video"] = resize_video(
                rendered_video, target_size, mode="bilinear"
            )
            prepared["rendered_mask"] = resize_video(
                rendered_mask, target_size, mode="nearest"
            )
            prepared["intrinsics"] = scale_intrinsics(
                cropped_intrinsics, cropped_size, target_size
            )
            prepared["extrinsics"] = batch["extrinsic"].to(self.device)
        return prepared

    def _prepare_controls(
        self,
        first_frame: torch.Tensor,
        prepared: dict[str, Any],
        latent: torch.Tensor,
        frame_num: int,
        hand_scale: float,
    ) -> ControlCache:
        """Encode all timestep-invariant I2V and control inputs."""

        first_condition = self.model._encode_first_frame_condition(
            first_frame=first_frame,
            frame_count=frame_num,
            latent_grid=tuple(int(value) for value in latent.shape[2:]),
        )
        camera_i2v_mask = first_condition[:, :4]
        if self.spec.reference_hand_enabled:
            reference_condition = self.model.encode_reference_hand_condition(
                hand_video=prepared["hand_video"],
                target_condition=first_condition,
            )
            first_condition = first_condition + reference_condition * float(hand_scale)
        clip_context = self.model._resolve_clip_context(
            batch=int(first_frame.shape[0]),
            dtype=first_condition.dtype,
            first_frame=first_frame,
            clip_context=None,
        )
        cache = ControlCache(
            first_frame_condition=first_condition,
            clip_context=clip_context,
            camera_i2v_mask=camera_i2v_mask,
        )

        if self.spec.obj_enabled:
            cache.reference_latents = self.model.encode_object_reference_latents(
                reference_images=prepared["reference_images"],
                video_spatial_size=tuple(prepared["target_size"]),
                target_latent=latent,
            )

        if self.spec.hand_enabled:
            assert self.model.hand_encoder is not None
            hand_tokens, hand_grid = self.model.hand_encoder.encode_tokens(
                prepared["hand_video"]
            )
            cache.hand_tokens = hand_tokens * float(hand_scale)
            cache.hand_token_grid = hand_grid

        if self.spec.camera_enabled:
            assert self.model.camera_encoder is not None
            assert self.model.plucker_encoder is not None
            rendered_video = normalize_rendered_video(prepared["rendered_video"])
            render_latent = self.model.camera_encoder.encode_rendered_video(
                rendered_video
            )
            dense_plucker = build_dense_plucker_rays(
                intrinsics=prepared["intrinsics"],
                extrinsics=prepared["extrinsics"],
                height=int(rendered_video.shape[-2]),
                width=int(rendered_video.shape[-1]),
            ).to(device=self.device, dtype=render_latent.dtype)
            rendered_mask = prepared["rendered_mask"].to(dtype=render_latent.dtype)
            cache.render_latent = render_latent
            cache.rendered_mask = rendered_mask
            cache.plucker_tokens = self.model.plucker_encoder(
                dense_plucker, rendered_mask
            )
        return cache

    def _camera_residual(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        cache: ControlCache,
        camera_scale: float,
    ) -> CameraEncoderOutput | None:
        """Compute timestep-dependent camera residuals when enabled."""

        if not self.spec.camera_enabled:
            return None
        assert self.model.camera_encoder is not None
        assert cache.render_latent is not None
        assert cache.rendered_mask is not None
        assert cache.plucker_tokens is not None
        assert cache.camera_i2v_mask is not None
        hidden_states = torch.cat(
            [
                latent,
                cache.camera_i2v_mask.to(dtype=latent.dtype),
                cache.first_frame_condition[:, 4:].to(dtype=latent.dtype),
            ],
            dim=1,
        )
        camera = self.model.camera_encoder(
            rendered_video=torch.empty(0, device=self.device),
            rendered_mask=cache.rendered_mask,
            hidden_states=hidden_states,
            plucker_tokens=cache.plucker_tokens,
            timestep=timestep,
            render_latent=cache.render_latent,
            temb=self.model._camera_timestep_embedding(timestep),
        )
        if float(camera_scale) == 1.0:
            return camera
        return replace(
            camera,
            residuals=[
                residual * float(camera_scale) for residual in camera.residuals
            ],
        )

    def _build_scheduler(
        self,
        sample_solver: str,
        sampling_steps: int,
        shift: float,
    ) -> tuple[torch.Tensor, Any]:
        """Create the selected Wan flow-matching scheduler."""

        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.model.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift
            )
            return scheduler.timesteps, scheduler
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=self.model.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(
            scheduler, device=self.device, sigmas=sigmas
        )
        return timesteps, scheduler

    def autocast_context(self) -> Any:
        """Return the mixed-precision context selected by the checkpoint."""

        dtype = transformer_dtype(self.device, self.precision)
        return nullcontext() if dtype is None else amp.autocast(dtype=dtype)


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve and validate a CPU or CUDA device."""

    text = str(requested)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if text == "auto"
        else torch.device(text)
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} requested, but CUDA is unavailable.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA index {index} is unavailable.")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    return device


def target_image_size(
    first_frame: torch.Tensor,
    max_area: int | None,
    model: EgoHOIControlModel,
) -> tuple[int, int]:
    """Compute the VAE-and-patch-aligned Wan output size."""

    height, width = (int(value) for value in first_frame.shape[-2:])
    area = int(max_area or height * width)
    if area <= 0:
        raise ValueError("max_area must be greater than zero.")
    aspect = height / width
    height_alignment = int(model.vae_stride[1]) * int(model.patch_size[1])
    width_alignment = int(model.vae_stride[2]) * int(model.patch_size[2])
    target_height = int(np.sqrt(area * aspect) // height_alignment * height_alignment)
    target_width = int(np.sqrt(area / aspect) // width_alignment * width_alignment)
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"max_area={area} is too small for Wan.")
    return target_height, target_width


def ensure_bcthw(
    tensor: torch.Tensor,
    channels: int,
    name: str,
) -> torch.Tensor:
    """Normalize a video tensor to batched B,C,T,H,W layout."""

    if tensor.ndim != 5:
        raise ValueError(f"{name} must be 5D, got {tuple(tensor.shape)}.")
    if tensor.shape[1] == channels:
        return tensor.contiguous()
    if tensor.shape[2] == channels:
        return tensor.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"{name} does not contain {channels} channels.")


def pad_or_trim_frames(tensor: torch.Tensor, frame_num: int) -> torch.Tensor:
    """Trim a video or repeat its last frame to the requested length."""

    if frame_num <= 0:
        raise ValueError("frame_num must be greater than zero.")
    if tensor.shape[2] >= frame_num:
        return tensor[:, :, :frame_num].contiguous()
    padding = tensor[:, :, -1:].repeat(
        1, 1, frame_num - tensor.shape[2], 1, 1
    )
    return torch.cat([tensor, padding], dim=2).contiguous()


def center_crop_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Center-crop a BCHW image tensor to its largest square region."""

    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW image tensor, got {tuple(tensor.shape)}.")
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return tensor[..., top : top + crop_size, left : left + crop_size].contiguous()


def center_crop_video_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Center-crop a BCTHW video tensor to its largest square frame region."""

    if tensor.ndim != 5:
        raise ValueError(f"Expected BCTHW video tensor, got {tuple(tensor.shape)}.")
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return tensor[..., top : top + crop_size, left : left + crop_size].contiguous()


def rgb_hand_video_to_gray(video: torch.Tensor) -> torch.Tensor:
    """Convert BCTHW RGB hand controls to repeated-channel grayscale frames."""

    if video.ndim != 5 or int(video.shape[1]) != 3:
        return video
    if torch.allclose(video[:, 0], video[:, 1]) and torch.allclose(video[:, 1], video[:, 2]):
        return video
    weights = video.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1, 1)
    gray = (video * weights).sum(dim=1, keepdim=True)
    return gray.repeat(1, 3, 1, 1, 1).contiguous()


def resize_image_tensor(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Resize every BCHW image tensor to a fixed spatial size."""

    if tuple(tensor.shape[-2:]) == tuple(target_size):
        return tensor
    return F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).contiguous()


def resize_video(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    """Resize every frame in a B,C,T,H,W tensor."""

    if tuple(tensor.shape[-2:]) == tuple(target_size):
        return tensor
    batch, channels, frames, source_height, source_width = tensor.shape
    flat = tensor.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, source_height, source_width
    )
    kwargs: dict[str, Any] = {"size": target_size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(flat, **kwargs)
    return (
        resized.reshape(batch, frames, channels, *target_size)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def center_crop_intrinsics(
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Shift camera intrinsics to match a centered square crop."""

    source_height, source_width = (int(value) for value in source_size)
    crop_size = min(source_height, source_width)
    top = (source_height - crop_size) // 2
    left = (source_width - crop_size) // 2
    cropped = intrinsics.clone()
    cropped[..., 0, 2] -= float(left)
    cropped[..., 1, 2] -= float(top)
    return cropped, (crop_size, crop_size)


def scale_intrinsics(
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Scale camera intrinsics to a resized render."""

    if source_size == target_size:
        return intrinsics
    source_height, source_width = source_size
    target_height, target_width = target_size
    scaled = intrinsics.clone()
    scaled[..., 0, :] *= float(target_width) / float(source_width)
    scaled[..., 1, :] *= float(target_height) / float(source_height)
    return scaled


def expand_prompt(prompt: str | list[str], batch_size: int) -> list[str]:
    """Return exactly one prompt per inference sample."""

    if isinstance(prompt, str):
        return [prompt] * batch_size
    if len(prompt) != batch_size:
        raise ValueError(f"Expected {batch_size} prompts, got {len(prompt)}.")
    return [str(value) for value in prompt]


def resolve_inference_prompt(
    args: argparse.Namespace,
    batch: Mapping[str, Any],
) -> str | list[str]:
    """Select the prompt, allowing configured defaults to replace JSON captions."""

    if args.prompt is not None:
        return args.prompt
    if DEFAULT_CAPTION is not None:
        return DEFAULT_CAPTION
    return batch["caption"]


def resolve_seed(seed: int) -> int:
    """Return a deterministic seed or generate a random nonnegative seed."""

    return int(seed) if int(seed) >= 0 else random.randint(0, sys.maxsize)


def video_to_uint8_frames(video: torch.Tensor) -> list[np.ndarray]:
    """Convert one decoded Wan video to uint8 RGB frames."""

    normalized = (
        video.detach().cpu().clamp(-1.0, 1.0).add(1.0).mul(127.5).to(torch.uint8)
    )
    array = normalized.permute(1, 2, 3, 0).contiguous().numpy()
    return list(array)


def save_video(video: torch.Tensor, output_path: Path, fps: int) -> None:
    """Encode one generated video as an H.264 MP4."""

    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=8
    )
    try:
        for frame in video_to_uint8_frames(video):
            writer.append_data(frame)
    finally:
        writer.close()


def save_frames(video: torch.Tensor, output_dir: Path) -> None:
    """Write generated RGB frames as numbered PNG files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(video_to_uint8_frames(video)):
        imageio.imwrite(output_dir / f"frame_{index:04d}.png", frame)


def summary_video_paths(payload: Any) -> set[str]:
    """Collect every ``video_path`` value recursively from a JSON payload."""

    paths: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == "video_path" and isinstance(value, str):
                paths.add(value)
            else:
                paths.update(summary_video_paths(value))
    elif isinstance(payload, list):
        for value in payload:
            paths.update(summary_video_paths(value))
    return paths


def load_summary_json(summary_path: Path) -> dict[str, Any]:
    """Read an existing inference summary or return an empty summary."""

    if not summary_path.is_file():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Inference summary must be a JSON object: {summary_path}")
    return payload


def summary_has_video(summary_path: Path, video_path: Path) -> bool:
    """Return whether a summary JSON already records the inferred video."""

    try:
        payload = load_summary_json(summary_path)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    candidates = {str(video_path), str(video_path.resolve())}
    return bool(summary_video_paths(payload) & candidates)


def append_summary_result(
    summary_path: Path,
    base_summary: dict[str, Any],
    result_summary: dict[str, Any],
) -> None:
    """Append one inference result to a per-clip JSON summary."""

    existing = load_summary_json(summary_path)
    if not existing:
        existing = {
            **base_summary,
            "item": base_summary.get("item"),
            "items": [],
            "results": [],
        }
    elif "results" not in existing:
        previous_result = dict(existing) if "video_path" in existing else None
        previous_item = existing.get("item")
        existing = {
            **base_summary,
            "item": base_summary.get("item"),
            "items": [previous_item] if isinstance(previous_item, dict) else [],
            "results": [previous_result] if previous_result is not None else [],
        }
    results = existing.setdefault("results", [])
    if not isinstance(results, list):
        raise TypeError(f"Summary results must be a list: {summary_path}")
    items = existing.setdefault("items", [])
    if isinstance(items, list):
        items.append(result_summary["item"])
    results.append(result_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8"
    )


def checkpoint_summary(
    spec: CheckpointSpec,
    data_config: dict[str, Any],
    item: dict[str, Any],
    precision: str,
) -> dict[str, Any]:
    """Build a JSON-safe summary before the heavyweight model load."""

    camera = dict(spec.config.get("camera_encoder", {}))
    hand = dict(spec.config.get("hand_encoder", {}))
    obj = dict(spec.config.get("obj_encoder", {}))
    architecture: dict[str, Any] = {}
    if spec.camera_enabled:
        keys = (
            "injected_layers",
            "plucker_hidden_dim",
            "plucker_feature_channels",
            "conv_out_dim",
            "control_dim",
            "ffn_dim",
            "num_heads",
            "num_layers",
            "time_frequency_dim",
            "time_dim",
            "rope_max_seq_len",
            "qk_norm",
            "eps",
        )
        architecture["camera_encoder"] = {key: camera.get(key) for key in keys}
    if spec.hand_enabled:
        architecture["hand_encoder"] = {
            "concat_dim": hand.get("concat_dim", 4),
            "reference_enabled": spec.reference_hand_enabled,
            "reference_concat_dim": hand.get("reference_concat_dim", 4),
            "wan_lora_enabled": spec.hand_lora_enabled,
            "lora_rank": hand.get("lora_rank", 4),
            "lora_alpha": hand.get("lora_alpha", 4.0),
            "lora_target_modules": hand.get(
                "lora_target_modules", "q,k,v,o,ffn.0,ffn.2"
            ),
        }
    if spec.obj_enabled:
        architecture["obj_encoder"] = {
            "rope": spec.obj_rope,
            "rank": obj.get("rank", 128),
            "alpha": obj.get("alpha", 128.0),
            "dropout": obj.get("dropout", 0.0),
            "init": obj.get("init", "zero"),
            "injected_layers": spec.obj_injected_layers,
            "target_modules": obj.get(
                "target_modules",
                ["self_attn.q", "self_attn.k", "self_attn.v"]
                if spec.obj_injected_layers is None
                else ["self_attn.k", "self_attn.v"],
            ),
            "train_attn_blocks": obj.get("train_attn_blocks")
            if spec.obj_injected_layers is None
            else None,
        }
    return {
        "checkpoint": str(spec.path),
        "format_version": spec.payload.get("format_version"),
        "global_step": spec.payload.get("global_step"),
        "best_loss": spec.payload.get("best_loss"),
        "encoders": spec.encoder_names,
        "precision": precision,
        "architecture": architecture,
        "data": {
            "root": data_config["root"],
            "clip_names": data_config["clip_names"],
            "video_frames": data_config["video_frames"],
            "image_size": data_config["image_size"],
            "start_frames": data_config["start_frames"],
        },
        "item": item,
    }


def output_video_path(
    output_dir: Path,
    clip_name: str,
    start_frame: int,
    end_frame: int,
    encoder_names: list[str],
) -> Path:
    """Return a descriptive path for one generated MP4."""

    branch = "_".join(
        name.removesuffix("_encoder") for name in encoder_names
    )
    filename = f"{clip_name}_{start_frame}-{end_frame}_{branch}_inference.mp4"
    return resolve_workspace_path(output_dir) / clip_name / filename



def checkpoint_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the checkpoint's inference data section with legacy fallback."""

    if isinstance(config.get("hot3d_data"), Mapping):
        return dict(config["hot3d_data"])
    if isinstance(config.get("data"), Mapping):
        return dict(config["data"])
    return {}


def image_size_from_config(config: Mapping[str, Any]) -> tuple[int, int] | None:
    """Resolve the image size used when direct file inputs need decoding."""

    data_config = checkpoint_data_config(config)
    return normalize_image_size(
        data_config.get(
            "image_size",
            data_config.get("resolution", data_config.get("video_image_size", 480)),
        )
    )


def tensor_first_frame(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize an image or video tensor to batched ``[B, 3, H, W]``."""

    tensor = value.detach().float()
    if tensor.ndim == 3 and int(tensor.shape[0]) == 3:
        return tensor.unsqueeze(0).contiguous()
    if tensor.ndim == 4 and int(tensor.shape[1]) == 3:
        return tensor[:1].contiguous()
    if tensor.ndim == 4 and int(tensor.shape[0]) == 3:
        return tensor.unsqueeze(0).contiguous()
    if tensor.ndim == 5 and int(tensor.shape[2]) == 3:
        return tensor[:, 0].contiguous()
    if tensor.ndim == 5 and int(tensor.shape[1]) == 3:
        return tensor[:, :, 0].contiguous()
    raise ValueError(f"{name} must be an RGB image or video tensor, got {tuple(tensor.shape)}.")


def load_image_or_first_video_frame(
    source: str | Path | torch.Tensor,
    *,
    image_size: tuple[int, int] | None,
    frame_num: int,
) -> torch.Tensor:
    """Load an RGB image/video path or tensor as a batched first-frame image."""

    if isinstance(source, torch.Tensor):
        return tensor_first_frame(source, "reference_img")
    path = resolve_workspace_path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Reference image/video not found: {path}")
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        video = _load_normalized_video_frames(
            path, frame_count=frame_num, image_size=None, start_frame=0
        )
        frame = center_crop_image_tensor(video[:1].contiguous())
        if image_size is not None:
            frame = resize_image_tensor(frame, image_size)
        return frame.contiguous()
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)
    tensor = center_crop_image_tensor(tensor.unsqueeze(0))
    if image_size is not None:
        tensor = resize_image_tensor(tensor, image_size)
    return tensor.contiguous()


def tensor_video_batched(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a video tensor to batched ``[B, T, C, H, W]`` layout."""

    tensor = value.detach().float()
    if tensor.ndim == 4 and int(tensor.shape[1]) in {1, 3}:
        return tensor.unsqueeze(0).contiguous()
    if tensor.ndim == 4 and int(tensor.shape[0]) in {1, 3}:
        return tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    if tensor.ndim == 5 and int(tensor.shape[2]) in {1, 3}:
        return tensor.contiguous()
    if tensor.ndim == 5 and int(tensor.shape[1]) in {1, 3}:
        return tensor.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"{name} must be a video tensor, got {tuple(tensor.shape)}.")


def load_video_input(
    source: str | Path | torch.Tensor | None,
    *,
    image_size: tuple[int, int] | None,
    frame_num: int,
    name: str,
) -> torch.Tensor | None:
    """Load a video path or tensor as batched ``[B, T, C, H, W]`` controls."""

    if source is None:
        return None
    if isinstance(source, torch.Tensor):
        return tensor_video_batched(source, name)
    path = resolve_workspace_path(source)
    if not path.is_file():
        raise FileNotFoundError(f"{name} video not found: {path}")
    video = _load_normalized_video_frames(
        path, frame_count=frame_num, image_size=None, start_frame=0
    )
    return video.unsqueeze(0).contiguous()


def tensor_reference_images(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize object-reference tensors to ``[B, R, 3, H, W]``."""

    tensor = value.detach().float()
    if tensor.ndim == 4 and int(tensor.shape[1]) == 3:
        return tensor.unsqueeze(0).contiguous()
    if tensor.ndim == 5 and int(tensor.shape[2]) == 3:
        return tensor.contiguous()
    raise ValueError(f"{name} must use [R, 3, H, W] or [B, R, 3, H, W], got {tuple(tensor.shape)}.")


def load_reference_images_input(
    sources: Sequence[str | Path | torch.Tensor] | torch.Tensor | None,
    *,
    image_size: tuple[int, int] | None,
) -> torch.Tensor | None:
    """Load three object-reference images or return a provided tensor batch."""

    if sources is None:
        return None
    if isinstance(sources, torch.Tensor):
        return tensor_reference_images(sources, "three_obj_imgs")
    images: list[torch.Tensor] = []
    for source in sources:
        if isinstance(source, torch.Tensor):
            images.append(tensor_first_frame(source, "three_obj_imgs")[0])
        else:
            images.append(load_reference_image(resolve_workspace_path(source), image_size))
    if len(images) != 3:
        raise ValueError(f"three_obj_imgs must contain exactly three images, got {len(images)}.")
    return torch.stack(images, dim=0).unsqueeze(0).contiguous()


def normalize_mask_video(mask: torch.Tensor) -> torch.Tensor:
    """Convert RGB or single-channel masks to ``[B, T, 1, H, W]``."""

    batched = tensor_video_batched(mask, "view_render mask")
    if int(batched.shape[2]) == 1:
        return batched
    return batched.mean(dim=2, keepdim=True).contiguous()


def normalize_camera_matrix_batch(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize camera matrices to batched time-major tensor layout."""

    tensor = value.detach().float() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0).contiguous()
    if tensor.ndim == 4:
        return tensor.contiguous()
    raise ValueError(f"{name} must be [T, M, M] or [B, T, M, M], got {tuple(tensor.shape)}.")


def load_view_render_input(
    source: Mapping[str, Any] | Sequence[Any] | None,
    *,
    image_size: tuple[int, int] | None,
    frame_num: int,
) -> dict[str, torch.Tensor] | None:
    """Load rendered-view video, mask, and camera matrices for one sample."""

    if source is None:
        return None
    if not isinstance(source, Mapping):
        values = list(source)
        if len(values) != 4:
            raise ValueError("view_render sequences must be (video, mask, intrinsic, extrinsic).")
        source = {
            "video": values[0],
            "mask": values[1],
            "intrinsic": values[2],
            "extrinsic": values[3],
        }
    video_source = source.get("rendered_novel_views", source.get("video"))
    mask_source = source.get("rendered_novel_view_masks", source.get("mask"))
    rendered_video = load_video_input(
        video_source, image_size=image_size, frame_num=frame_num, name="view_render"
    )
    rendered_mask = load_video_input(
        mask_source, image_size=image_size, frame_num=frame_num, name="view_render mask"
    )
    if rendered_video is None or rendered_mask is None:
        raise KeyError("view_render must provide video and mask controls.")
    intrinsic = source.get("intrinsic", source.get("intrinsics"))
    extrinsic = source.get("extrinsic", source.get("extrinsics"))
    if intrinsic is None or extrinsic is None:
        camera_info_path = source.get("camera_info_path")
        if camera_info_path is None:
            raise KeyError("view_render must provide intrinsic/extrinsic or camera_info_path.")
        start_frame = int(source.get("start_frame", 0))
        intrinsic, extrinsic = _load_camera_window(camera_info_path, start_frame, frame_num)
    return {
        "rendered_novel_views": rendered_video,
        "rendered_novel_view_masks": normalize_mask_video(rendered_mask),
        "intrinsic": normalize_camera_matrix_batch(intrinsic, "intrinsic"),
        "extrinsic": normalize_camera_matrix_batch(extrinsic, "extrinsic"),
    }


def build_direct_inference_batch(
    spec: CheckpointSpec,
    reference_img: str | Path | torch.Tensor,
    hand_video: str | Path | torch.Tensor | None,
    view_render: Mapping[str, Any] | Sequence[Any] | None,
    three_obj_imgs: Sequence[str | Path | torch.Tensor] | torch.Tensor | None,
    *,
    frame_num: int,
) -> dict[str, Any]:
    """Build the single-sample batch consumed by ``AdaptiveInferenceRunner``."""

    image_size = image_size_from_config(spec.config)
    first_frame = load_image_or_first_video_frame(
        reference_img, image_size=image_size, frame_num=frame_num
    )
    batch: dict[str, Any] = {
        "first_frame": first_frame,
        "raw_video": first_frame.unsqueeze(1).repeat(1, frame_num, 1, 1, 1),
        "clip_name": ["direct"],
        "caption": [DEFAULT_CAPTION or ""],
        "start_frame": [0],
        "end_frame": [frame_num - 1],
    }
    if spec.hand_enabled:
        loaded_hand = load_video_input(
            hand_video, image_size=image_size, frame_num=frame_num, name="hand_video"
        )
        if loaded_hand is None:
            raise KeyError("hand_video is required by this checkpoint.")
        batch["rendered_hands"] = loaded_hand
    if spec.camera_enabled:
        loaded_view = load_view_render_input(
            view_render, image_size=image_size, frame_num=frame_num
        )
        if loaded_view is None:
            raise KeyError("view_render is required by this checkpoint.")
        batch.update(loaded_view)
    if spec.obj_enabled:
        reference_images = load_reference_images_input(
            three_obj_imgs,
            image_size=normalize_image_size(
                checkpoint_data_config(spec.config).get("reference_image_size", 480)
            ),
        )
        if reference_images is None:
            raise KeyError("three_obj_imgs is required by this checkpoint.")
        batch["reference_images"] = reference_images
    return batch


def get_cached_runner(
    ckpt: str | Path,
    *,
    device: str | torch.device,
    precision: str | None,
    zero2: bool,
) -> tuple[CheckpointSpec, str, torch.device, AdaptiveInferenceRunner]:
    """Return a checkpoint runner, reusing it across batch API calls."""

    args = argparse.Namespace(
        checkpoint=Path(ckpt),
        output_dir=DEFAULT_OUTPUT_DIR,
        zero2=zero2,
    )
    checkpoint_loader = resolve_checkpoint_loader(args)
    spec = checkpoint_loader(args.checkpoint)
    resolved_precision = checkpoint_precision(spec.config, precision)
    resolved_device = resolve_device(device)
    key = (str(spec.path), str(resolved_device), resolved_precision, bool(zero2))
    cached = _RUNNER_CACHE.get(key)
    if cached is not None:
        cached_spec, runner = cached
        return cached_spec, resolved_precision, resolved_device, runner
    runner = AdaptiveInferenceRunner(spec, resolved_device, resolved_precision)
    _RUNNER_CACHE[key] = (spec, runner)
    return spec, resolved_precision, resolved_device, runner


def clear_inference_cache() -> None:
    """Close every cached runner created through the public inference API."""

    for _, runner in list(_RUNNER_CACHE.values()):
        runner.close()
    _RUNNER_CACHE.clear()


def infer(
    reference_img: str | Path | torch.Tensor,
    hand_video: str | Path | torch.Tensor | None,
    view_render: Mapping[str, Any] | Sequence[Any] | None,
    three_obj_imgs: Sequence[str | Path | torch.Tensor] | torch.Tensor | None,
    ckpt: str | Path,
    caption: str,
    output_dir: str | Path,
    *,
    device: str | torch.device = "auto",
    precision: str | None = None,
    negative_prompt: str = "",
    frame_num: int = 81,
    max_area: int | None = None,
    sampling_steps: int = 40,
    shift: float = 3.0,
    sample_solver: str = "unipc",
    guide_scale: float = 5.0,
    camera_scale: float = 1.0,
    hand_scale: float = 1.0,
    seed: int = 23,
    fps: int = 16,
    output_name: str = DEFAULT_OUTPUT_NAME,
    zero2: bool = False,
    show_progress: bool = True,
) -> torch.Tensor:
    """Run exactly one inference call and save its generated MP4.

    ``reference_img`` supplies the first RGB frame. ``hand_video`` and
    ``view_render`` may be ``None`` only when the checkpoint does not enable
    those branches. ``three_obj_imgs`` must provide exactly three object images
    when the object encoder is enabled. The returned tensor is the generated
    video in Wan's decoded ``[C, T, H, W]`` range before MP4 quantization.
    """

    spec, _, _, runner = get_cached_runner(
        ckpt,
        device=device,
        precision=precision,
        zero2=zero2,
    )
    batch = build_direct_inference_batch(
        spec,
        reference_img,
        hand_video,
        view_render,
        three_obj_imgs,
        frame_num=int(frame_num),
    )
    batch["caption"] = [str(caption)]
    result = runner.generate(
        batch,
        prompt=str(caption),
        negative_prompt=negative_prompt,
        frame_num=int(frame_num),
        max_area=max_area,
        sampling_steps=int(sampling_steps),
        shift=float(shift),
        sample_solver=str(sample_solver),
        guide_scale=float(guide_scale),
        camera_scale=float(camera_scale),
        hand_scale=float(hand_scale),
        seed=int(seed),
        show_progress=bool(show_progress),
    )
    video = result.video[0]
    save_video(video, resolve_workspace_path(output_dir) / output_name, fps=int(fps))
    return video


def view_render_from_batch(batch: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the direct-API view-render mapping for a collated batch."""

    required = (
        "rendered_novel_views",
        "rendered_novel_view_masks",
        "intrinsic",
        "extrinsic",
    )
    if not all(key in batch for key in required):
        return None
    return {key: batch[key] for key in required}


def single_inference_requested(args: argparse.Namespace) -> None:
    """Reject CLI requests that would produce more than one output video."""

    start_frames = resolve_start_frames(args)
    if len(start_frames) != 1 or int(args.max_clip) != 1:
        raise ValueError(
            "scripts/model/infer.py runs exactly one inference. Use "
            "scripts/model/infer_batch.py for multiple clips or start frames."
        )


def main() -> None:
    """Resolve one dataset window, call ``infer()``, and save its video."""

    args = parse_args()
    single_inference_requested(args)
    checkpoint_loader = resolve_checkpoint_loader(args)
    spec = checkpoint_loader(args.checkpoint)
    precision = checkpoint_precision(spec.config, args.precision)
    dataset, data_config = build_dataset(spec, args)
    if len(dataset) != 1:
        raise ValueError(
            f"scripts/model/infer.py expected one decoded item, discovered {len(dataset)}. "
            "Use scripts/model/infer_batch.py for batch inference."
        )

    default_reference_images = (
        load_default_reference_images(data_config) if spec.obj_enabled else None
    )
    item = describe_default_references(
        describe_inference_item(dataset, 0), default_reference_images
    )
    inspection = checkpoint_summary(spec, data_config, item, precision)
    print(json.dumps(inspection, indent=2, sort_keys=True), flush=True)
    if args.inspect:
        return

    video_path = output_video_path(
        args.output_dir,
        item["clip_name"],
        int(item["interval"][0]),
        int(item["interval"][1]),
        spec.encoder_names,
    )
    summary_path = video_path.parent / "inference_summary.json"
    if summary_has_video(summary_path, video_path):
        print(f"Skipping recorded inference: {video_path}", flush=True)
        return

    seed_everything(args.seed)
    batch = collate_inference_item(dataset, 0)
    apply_default_reference_images(batch, default_reference_images)
    prompt = resolve_inference_prompt(args, batch)
    video = infer(
        batch["first_frame"],
        batch.get("rendered_hands"),
        view_render_from_batch(batch),
        batch.get("reference_images"),
        args.checkpoint,
        expand_prompt(prompt, 1)[0],
        video_path.parent,
        device=args.device,
        precision=args.precision,
        negative_prompt=args.negative_prompt,
        frame_num=int(data_config["video_frames"]),
        max_area=args.max_area,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        sample_solver=args.sample_solver,
        guide_scale=args.guide_scale,
        camera_scale=args.camera_scale,
        hand_scale=args.hand_scale,
        seed=args.seed,
        fps=args.fps,
        output_name=video_path.name,
        zero2=bool(args.zero2),
        show_progress=not args.no_progress,
    )
    frame_dir: Path | None = None
    if args.save_frames:
        frame_dir = video_path.parent / f"frames_{batch['start_frame'][0]}-{batch['end_frame'][0]}"
        save_frames(video, frame_dir)
    summary = {
        **inspection,
        "item": item,
        "device": str(args.device),
        "seed": int(args.seed),
        "sampling_steps": int(args.sampling_steps),
        "frame_count": int(video.shape[1]),
        "video_path": str(video_path),
        "frame_dir": str(frame_dir) if frame_dir is not None else None,
    }
    append_summary_result(summary_path, inspection, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    clear_inference_cache()


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    main()
