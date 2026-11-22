# EgoSpace: code

Preprocessing (hand MANO + camera trajectory + initial frame → unified 3D action video) and
inference (EgoSpace world model). The project page lives in `page/`.

```
src/
  egospace/          new code: preprocessing pipeline and inference wrapper
    io.py            camera.npz / mano.npz formats, video I/O
    mano.py          MANO forward (smplx)
    scene_render.py  Depth Pro depth, first-frame point cloud
    render.py        joint renderer: point cloud + MANO meshes in one z-buffer (training hand shading, grayscale)
    hand_seg.py      SAM 3 first-frame hand mask (optional)
    preprocess.py    full preprocessing
    inference.py     generation call
  scr/               model code from the training repository (encoders, trainer, Depth Pro,
                     inference in scr/inference/infer.py)
  wan/               Wan2.1 package (Apache-2.0, src/wan/LICENSE.txt)
examples/
  preprocess.py      CLI: first frame + camera + MANO → control.mp4 + control_mask.mp4
  inference.py       CLI: first frame + control video/mask + camera + 3 object views + prompt → video
data/                the two Counterfactual Action Following clips of the project page (data/README.md)
checkpoints/         weights (not in git, see below)
```

## Setup

```bash
pip install -r requirements.txt
```

Weights go in `checkpoints/` (the CLIs default to these paths):

| path | source |
|---|---|
| `checkpoints/egospace/final.pt` | EgoSpace checkpoint (`train_camera_obj`; `resolved_config.yaml` next to it) |
| `checkpoints/Wan2.1-I2V-14B-720P/` | [Wan-AI/Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |
| `checkpoints/depth_pro.pt` | [apple/DepthPro](https://huggingface.co/apple/DepthPro) |
| `checkpoints/mano/MANO_{LEFT,RIGHT}.pkl` | [MANO](https://mano.is.tue.mpg.de) (licence required) |
| `checkpoints/sam3.pt` | [facebook/sam3](https://huggingface.co/facebook/sam3), only without `--hand-mask` |

## Preprocessing

Inputs: the initial RGB frame, the camera trajectory (`camera.npz`: `K`, `world_to_camera` [N,4,4],
OpenCV axes) and the hand MANO tracks in the same world frame (`mano.npz`: per hand `hand_pose`
[N,15 PCA or 45], `wrist` [N,6] = global orient + translation, `betas`). Formats in `src/egospace/io.py`.

```bash
python examples/preprocess.py \
    --first-frame data/withdraw_clip-001995/first_frame.png \
    --camera      data/withdraw_clip-001995/camera.npz \
    --mano        data/withdraw_clip-001995/mano.npz \
    --hand-mask   data/withdraw_clip-001995/hand_mask.png \
    --output-dir  outputs/withdraw_clip-001995
```

Steps: the first-frame hand pixels (`--hand-mask`, or SAM 3 instances that the frame-0 MANO hands
fall on) are removed; Depth Pro (with the known focal length; or `--depth`) lifts the rest into a
coloured point cloud; for each of the N cameras the point cloud and the MANO meshes are rendered
together with one shared z-buffer (hands occlude the scene and are occluded by it; hands use the
training shading, in grayscale). Outputs: `control.mp4`, `control_mask.mp4` (white = generate:
disocclusions and visible hands; black = known scene; uncovered pixels gray 127), plus
`depth.npy`, `hand_mask.png`.
Renders are capped at `--max-size` px (default 640; HOT3D training controls were 608×608).
About 1.5 min per 81-frame clip (Depth Pro on GPU, rendering on CPU).

Counterfactual actions: edit the MANO tracks (e.g. `data/*/mano_original.npz` → `mano.npz`) and rerun.

## Inference

```bash
python examples/inference.py \
    --first-frame   data/withdraw_clip-001995/first_frame.png \
    --control-video data/withdraw_clip-001995/control.mp4 \
    --control-mask  data/withdraw_clip-001995/control_mask.mp4 \
    --camera        data/withdraw_clip-001995/camera.npz \
    --object-images data/withdraw_clip-001995/object_views/view_{1,2,3}.png \
    --prompt        "$(cat data/withdraw_clip-001995/prompt.txt)" \
    --output        outputs/withdraw_clip-001995/egospace.mp4
```

Use `outputs/<clip>/control.mp4` / `control_mask.mp4` / `camera.npz` from preprocessing in the same
way. The object views are three renders/crops of the manipulated object on a gray background.
Output: 81 frames, 480×480, 16 fps. Defaults: seed 23, 40 steps, guidance 5, shift 3 (paper
settings). One GPU with ~65 GB, about 8 min per clip.
