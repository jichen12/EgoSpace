"""EgoSpace generation: first frame + unified 3D action video (+ mask, cameras) + 3 object views + prompt.

Thin wrapper over the training-code inference path (src/scr/inference/infer.py), which loads the
Wan2.1-I2V-14B backbone plus the camera (Uni3C-style) and object encoders from the checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .io import load_camera


def generate(first_frame: str | Path, control_video: str | Path, control_mask: str | Path,
             camera: str | Path, object_images: list[str | Path], prompt: str, output: str | Path, *,
             checkpoint: str | Path, wan_dir: str | Path, negative_prompt: str = "",
             seed: int = 23, sampling_steps: int = 40, guide_scale: float = 5.0, shift: float = 3.0,
             device: str = "auto", fps: int = 16) -> Path:
    if len(object_images) != 3:
        raise ValueError("exactly three object views are required")
    # infer.py resolves relative paths against src/, so make everything absolute
    first_frame, control_video, control_mask, checkpoint = (Path(p).resolve() for p in
                                                            (first_frame, control_video, control_mask, checkpoint))
    object_images = [Path(p).resolve() for p in object_images]
    output = Path(output).resolve()
    os.environ["EGOSPACE_WAN_DIR"] = str(Path(wan_dir).expanduser().resolve())
    from scr.inference import infer

    spec = infer.load_checkpoint_spec(Path(checkpoint).resolve())
    if not spec.camera_enabled or not spec.obj_enabled or spec.hand_enabled or spec.reference_hand_enabled:
        raise ValueError("expected the EgoSpace camera + object checkpoint (train_camera_obj)")
    cam = load_camera(camera)
    extrinsic = cam.get("model_world_to_camera", cam["world_to_camera"]).astype(np.float32)
    infer.infer(
        reference_img=Path(first_frame),
        hand_video=None,  # hands are already in the control video
        view_render={"video": Path(control_video), "mask": Path(control_mask),
                     "intrinsic": cam["K"].astype(np.float32), "extrinsic": extrinsic},
        three_obj_imgs=[Path(p) for p in object_images],
        ckpt=Path(checkpoint).resolve(),
        caption=prompt,
        negative_prompt=negative_prompt,
        output_dir=output.parent,
        output_name=output.name,
        device=device,
        frame_num=81,
        sampling_steps=sampling_steps,
        shift=shift,
        guide_scale=guide_scale,
        seed=seed,
        fps=fps,
    )
    return output
