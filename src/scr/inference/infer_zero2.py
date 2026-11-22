"""Infer from the DeepSpeed ZeRO-2 checkpoint produced by train_zero2.py.

The sampling, dataset, model construction, and output logic intentionally
reuse ``scripts/model/infer.py``. This adapter only translates DeepSpeed's directory
checkpoint into the compact per-module state expected by that inference code.

Example:
    uv run python scripts/model/infer_zero2.py --device cuda:6
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
import sys
from typing import Any

import torch


# Make repository imports work when this file is executed from any directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scr.inference import infer as base_infer


DEFAULT_CHECKPOINT_PATH = Path("outputs/egohoi_control_zero2/checkpoints_zero2/best")
DEFAULT_OUTPUT_DIR = Path("outputs/egohoi_control_zero2/inference/best")
DEEPSPEED_MODEL_STATE_NAME = "mp_rank_00_model_states.pt"


def prepare_optional_deepspeed_import() -> None:
    """Preload DeepSpeed while skipping its unused CUDA extension probe.

    Transformers detects the installed DeepSpeed package while PEFT installs
    LoRA. DeepSpeed 0.14.5 otherwise probes an optional FP quantizer and raises
    when this runtime-only CUDA host has no CUDA_HOME/nvcc. Inference never uses
    that extension, so this mirrors the safe import used by train_zero2.py.
    """

    if "deepspeed" in sys.modules:
        return
    os.environ.setdefault("DS_ACCELERATOR", "cuda")
    original_is_available = torch.cuda.is_available
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
        import deepspeed  # noqa: F401
    finally:
        torch.cuda.is_available = original_is_available  # type: ignore[method-assign]


def resolve_model_state_path(path: str | Path) -> tuple[Path, Path]:
    """Resolve a DeepSpeed tag directory or its rank-zero model-state file."""

    resolved = base_infer.resolve_workspace_path(path)
    if resolved.is_dir():
        model_state_path = resolved / DEEPSPEED_MODEL_STATE_NAME
        checkpoint_path = resolved
    else:
        model_state_path = resolved
        checkpoint_path = resolved.parent
    if not model_state_path.is_file():
        raise FileNotFoundError(
            "DeepSpeed rank-zero model checkpoint not found: "
            f"{model_state_path}"
        )
    return checkpoint_path, model_state_path


def state_below_prefix(
    state: Mapping[str, torch.Tensor],
    prefix: str,
) -> OrderedDict[str, torch.Tensor]:
    """Return tensors below one module prefix with that prefix removed."""

    dotted_prefix = prefix if prefix.endswith(".") else f"{prefix}."
    return OrderedDict(
        (name.removeprefix(dotted_prefix), tensor)
        for name, tensor in state.items()
        if name.startswith(dotted_prefix)
    )


def extract_trainable_module_states(
    state: Mapping[str, torch.Tensor],
) -> dict[str, Mapping[str, torch.Tensor]]:
    """Translate DeepSpeed model keys into compact inference module states.

    DeepSpeed serializes LoRA once through ``wan_dit`` and once through the
    lightweight ``wan_lora`` state view. Only the latter copy is retained so
    CPU memory is not held by duplicate rank-128 adapter tensors.
    """

    module_states: dict[str, Mapping[str, torch.Tensor]] = {}
    for module_name in (
        "plucker_encoder",
        "camera_encoder",
        "hand_encoder",
        "reference_hand_encoder",
    ):
        extracted = state_below_prefix(state, module_name)
        if extracted:
            module_states[module_name] = extracted

    lora_state = state_below_prefix(state, "wan_lora")
    if not lora_state:
        # Compatibility fallback for checkpoints without the lightweight view.
        lora_state = OrderedDict(
            (name.removeprefix("wan_dit."), tensor)
            for name, tensor in state.items()
            if name.startswith("wan_dit.")
            and (".lora_A." in name or ".lora_B." in name)
        )
    if lora_state:
        module_states["wan_lora"] = lora_state
    obj_state = state_below_prefix(state, "obj_encoder")
    if not obj_state:
        # Compatibility fallback mirrors the compact object view without
        # confusing its lora_down/lora_up tensors with PEFT hand adapters.
        obj_state = OrderedDict(
            (name.removeprefix("wan_dit."), tensor)
            for name, tensor in state.items()
            if name.startswith("wan_dit.")
            and (".lora_down." in name or ".lora_up." in name)
        )
    if obj_state:
        module_states["obj_encoder"] = obj_state
    return module_states


def validate_module_states(
    module_states: dict[str, Mapping[str, torch.Tensor]],
) -> tuple[bool, bool, bool, bool, bool]:
    """Validate branch completeness and return enabled architecture flags."""

    has_plucker = "plucker_encoder" in module_states
    has_camera = "camera_encoder" in module_states
    if has_plucker != has_camera:
        raise KeyError(
            "DeepSpeed camera checkpoint requires both plucker_encoder and "
            "camera_encoder states."
        )
    camera_enabled = has_plucker and has_camera
    hand_enabled = "hand_encoder" in module_states
    reference_hand_enabled = "reference_hand_encoder" in module_states
    hand_lora_enabled = "wan_lora" in module_states
    obj_enabled = "obj_encoder" in module_states
    if (reference_hand_enabled or hand_lora_enabled) and not hand_enabled:
        raise KeyError(
            "Reference-hand and Wan LoRA states require hand_encoder state."
        )
    if not camera_enabled and not hand_enabled and not obj_enabled:
        raise ValueError("Checkpoint contains no camera, hand, or object controls.")
    return (
        camera_enabled,
        hand_enabled,
        reference_hand_enabled,
        hand_lora_enabled,
        obj_enabled,
    )


def load_zero2_checkpoint_spec(path: Path) -> base_infer.CheckpointSpec:
    """Load one ZeRO-2 tag as the checkpoint specification used by inference."""

    checkpoint_path, model_state_path = resolve_model_state_path(path)
    payload = torch.load(model_state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            f"DeepSpeed model checkpoint must be a mapping, got {type(payload).__name__}."
        )
    config = payload.get("config")
    model_state = payload.get("module")
    if not isinstance(config, dict):
        raise KeyError("DeepSpeed checkpoint is missing its resolved config mapping.")
    if not isinstance(model_state, Mapping):
        raise KeyError("DeepSpeed checkpoint is missing its rank-zero module state.")

    module_states = extract_trainable_module_states(model_state)
    obj_rope = base_infer.resolve_checkpoint_obj_rope(payload, config)
    (
        camera_enabled,
        hand_enabled,
        reference_hand_enabled,
        hand_lora_enabled,
        obj_enabled,
    ) = validate_module_states(module_states)

    # Keep only lightweight metadata in the spec. The ignored duplicate LoRA
    # tensors and optimizer-related fields can be reclaimed before Wan loads.
    metadata: dict[str, Any] = {
        key: payload.get(key)
        for key in (
            "format_version",
            "global_step",
            "obj_rope",
            "global_steps",
            "best_loss",
            "epoch",
            "ds_version",
        )
    }
    metadata["deepspeed_model_state"] = str(model_state_path)
    return base_infer.CheckpointSpec(
        path=checkpoint_path,
        payload=metadata,
        config=deepcopy(config),
        module_states=module_states,
        camera_enabled=camera_enabled,
        hand_enabled=hand_enabled,
        reference_hand_enabled=reference_hand_enabled,
        hand_lora_enabled=hand_lora_enabled,
        obj_enabled=obj_enabled,
        obj_rope=obj_rope,
    )


def configure_base_inference() -> None:
    """Install ZeRO-2 defaults and loader into the shared inference workflow."""

    prepare_optional_deepspeed_import()
    base_infer.DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH
    base_infer.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    base_infer.load_checkpoint_spec = load_zero2_checkpoint_spec


def parse_args_for_documentation() -> argparse.Namespace:
    """Expose the shared parser signature to documentation and static tools.

    Runtime parsing remains inside ``scripts.infer.main`` after this adapter
    installs its checkpoint and output defaults.
    """

    configure_base_inference()
    return base_infer.parse_args()


def main() -> None:
    """Run adaptive inference using a DeepSpeed ZeRO-2 directory checkpoint."""

    configure_base_inference()
    base_infer.main()


if __name__ == "__main__":
    main()
