"""Re-render a finished SPIDER MJWP rollout in the light teaser style.

Only visuals change (floor, sky, hand colours); qpos comes unchanged from trajectory_mjwp.npz.
Usage: MUJOCO_GL=osmesa python render_sim_light.py <task_dir> <out_dir> <state_idx> [...]
"""
import os, re, sys
import numpy as np
import mujoco
from PIL import Image

task_dir, out_dir = sys.argv[1], sys.argv[2]
STRIDE=int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)
xml = open(os.path.join(task_dir, 'scene.xml')).read()
xml = xml.replace('meshdir="../../../assets/"', f'meshdir="{os.path.abspath(os.path.join(task_dir, "../../../assets"))}/"')
xml = re.sub(r'rgb1="0\.2 0\.3 0\.4" rgb2="0\.1 0\.2 0\.3" markrgb="0\.8 0\.8 0\.8"',
             'rgb1="0.96 0.97 0.99" rgb2="0.86 0.89 0.94" markrgb="1 1 1"', xml)
xml = xml.replace('rgb1="0.3 0.5 0.7" rgb2="0 0 0"', 'rgb1="1 1 1" rgb2="1 1 1"')
xml = xml.replace('reflectance="0.2"', 'reflectance="0.0"')
assets = os.path.abspath(os.path.join(task_dir, '../../../assets'))
xml = xml.replace('<compiler angle="radian"', f'<compiler angle="radian" texturedir="{assets}/"')
m = re.search(r'file="(objects/\d+)/visual.obj"', xml)
if m:
    pngs = sorted(f for f in os.listdir(os.path.join(assets, m.group(1))) if f.lower().endswith('.png'))
    if pngs:
        xml = xml.replace('<asset>', f'<asset>\n    <texture name="objtex" type="2d" file="{m.group(1)}/{pngs[0]}"/>\n    <material name="objmat" texture="objtex" specular="0.2"/>', 1)
        xml = re.sub(r'(<geom [^>]*mesh="right_visual")', r'\1 material="objmat"', xml)
        xml = re.sub(r'(<geom [^>]*)rgba="[^"]*"([^>]*mesh="right_visual")', r'\1\2', xml)
os.chdir(task_dir)
model = mujoco.MjModel.from_xml_string(xml)
BLUE = np.array([0.18, 0.47, 0.90, 1.0]); ORANGE = np.array([0.95, 0.55, 0.15, 1.0])
for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    if name.endswith('_visual') and not 'object' in name:
        base = BLUE if name.startswith('right_') else ORANGE if name.startswith('left_') else None
        if base is not None:
            shade = 0.85 + 0.15 * (g % 3) / 2
            model.geom_rgba[g] = np.r_[base[:3] * shade, 1.0]
    if name.startswith('left_object'):
        model.geom_rgba[g, 3] = 0.0  # empty placeholder object in bimanual scenes
model.vis.global_.offwidth = 2000; model.vis.global_.offheight = 2000
model.vis.headlight.ambient[:] = 0.45
model.vis.headlight.diffuse[:] = 0.55
model.vis.quality.shadowsize = 4096
data = mujoco.MjData(model)
traj = np.load(os.path.join(task_dir, '0', 'trajectory_mjwp.npz'))
qpos = traj['qpos'].reshape(-1, traj['qpos'].shape[-1])
print('states', qpos.shape, 'nq', model.nq)
W, H = int(os.environ.get('W', 900)), int(os.environ.get('H', 900))
renderer = mujoco.Renderer(model, H, W)
cam = mujoco.MjvCamera()
names=tuple(os.environ.get('LOOK','right_object,right_hand_link,left_hand_link').split(','))
bodies=[b for b in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or '').startswith(names)]
# fixed camera: look at the mean target over the first 40% of the rollout (before any failure drift)
P=[]
for k in range(0, max(1,int(len(qpos)*0.4)), STRIDE):
    data.qpos[:] = qpos[k]; mujoco.mj_forward(model, data)
    P.append(np.mean([data.xpos[b] for b in bodies],0))
cam.lookat[:] = np.mean(P,0)
cam.distance = float(os.environ.get('DIST', 0.75)); cam.azimuth = float(os.environ.get('AZ', 90)); cam.elevation = float(os.environ.get('EL', -35))
for i,k in enumerate(range(0, len(qpos), STRIDE)):
    data.qpos[:] = qpos[k]; mujoco.mj_forward(model, data)
    renderer.update_scene(data, cam)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    Image.fromarray(renderer.render()).save(os.path.join(out_dir, f'{i:04d}.png'))
print('done', i+1)
