# Example data: Counterfactual Action Following (project page)

Two HOT3D clips (Aria stream 214-1, frames 69–149, 81 frames) with the edited hand actions shown on
the project page:

| folder | edit | object |
|---|---|---|
| `withdraw_clip-001995/` | hands stop 5 cm before the flask (never touch it) → the flask must stay still | flask |
| `reroute_clip-002054/` | right hand grasps, then is redirected 15 cm → the bottle must follow | mustard bottle |

Each folder:

| file | content |
|---|---|
| `first_frame.png` | initial frame, 1408×1408 (undistorted, upright) |
| `camera.npz` | `K` [3,3] (1408 px), `world_to_camera` [81,4,4] image-aligned OpenCV cameras in the HOT3D world, `model_world_to_camera` [81,4,4] (see below) |
| `mano.npz` | edited hand tracks (input of `examples/preprocess.py`): `betas`, `{left,right}_hand_pose` [81,15] PCA, `{left,right}_wrist` [81,6], `left_shapedirs_fix=True`, `flat_hand_mean=False` (HOT3D MANO convention) |
| `mano_original.npz` | the recorded (unedited) hand tracks, same format |
| `hand_mask.png` | first-frame hand mask (white = hand) |
| `control.mp4`, `control_mask.mp4` | the unified 3D action video + mask used for the paper (608×608), input of `examples/inference.py` |
| `object_views/view_{1,2,3}.png` | three views of the manipulated object |
| `prompt.txt` | text prompt used for the paper (HOT3D auto-caption borrowed from the nearest captioned window, so it is terse) |
| `egospace_paper.mp4` | the EgoSpace video on the project page (seed 23, 40 steps, guidance 5, shift 3) |
| `real_video.mp4` | the recorded video of the window (original action, 480×480) |
| `source.json` | provenance and edit parameters |

`model_world_to_camera`: the HOT3D training data passed the camera encoder HOT3D's camera axes,
which are rotated by 90° about the optical axis relative to the upright images
(`model_world_to_camera` = Rz(-90°) · `world_to_camera` · const). `examples/inference.py` uses it when present so the
paper videos are reproduced; for your own data give only `world_to_camera`.

`examples/preprocess.py` on these folders regenerates the control video with Depth Pro + the
point-cloud renderer here instead of the original PyTorch3D renderer: 95–97 % of mask pixels agree
with `control.mp4`, known-scene PSNR 24–27 dB (`--max-size 608` matches the paper resolution).
