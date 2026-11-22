"""SAM 3 first-frame hand masks (only needed when no --hand-mask is given).

Requires the SAM 3 package (https://github.com/facebookresearch/sam3) and its checkpoint.
Text prompts 'left hand' / 'right hand' / 'hand' at confidence 0.35; an instance is kept only if
the projected MANO vertices of frame 0 fall on it, then the union is dilated by 7 px.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sam3_hand_mask(rgb: np.ndarray, projected: list[np.ndarray], checkpoint: str | Path,
                   device: str, confidence: float = 0.35) -> np.ndarray:
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device=device, checkpoint_path=str(checkpoint), load_from_HF=False).to(device).eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence)
    candidates = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        state = processor.set_image(Image.fromarray(rgb))
        for prompt in ("left hand", "right hand", "hand"):
            processor.reset_all_prompts(state)
            masks = processor.set_text_prompt(state=state, prompt=prompt).get("masks")
            if masks is not None and len(masks):
                candidates += list(masks.detach().cpu().numpy().astype(bool).reshape(len(masks), *masks.shape[-2:]))
    del model, processor
    h, w = rgb.shape[:2]
    selected = np.zeros((h, w), bool)
    for mask in candidates:
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not 80 <= mask.sum() <= 0.2 * h * w:
            continue
        grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
        for xy in projected:
            xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)]
            if len(xy) and grown[xy[:, 1], xy[:, 0]].sum() >= 2 and grown[xy[:, 1], xy[:, 0]].mean() >= 0.2:
                selected |= mask
                break
    selected = cv2.dilate(selected.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, 8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 100
    return keep[labels]
