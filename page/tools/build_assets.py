"""Collect and encode the project-page assets from local results (host, python3 + ffmpeg).

usage: python3 tools/build_assets.py [figs] [hero] [compare] [cf] [sim]   (no args = all)

Sources (all local, see README in this folder):
  figs     manuscript/fig/{teaser_v7,fig2_v2_pc,fig3_v2_4row,fig_cf_main_2row,sim_rgb_v5}.pdf -> webp
  hero     action video + EgoSpace rollout of clip-001973_69-149 (physca_cf roots/orig, rollouts/physca/orig)
  compare  the fig. 3 clips, 8-cell grid: first frame | real | EgoSpace | EgoHOI / Uni3C | EgoSim | GR | JCV
           (paths from manuscript/fig/fig_3_src/srcs2.py)
  cf       the counterfactual figure units (withdraw clip-002838_30-110, reroute clip-001881_69-149):
           edited action | JCV | EgoSim | EgoSpace, cyan commanded-hand outline + violet SAM3 object mask
  sim      SPIDER sequences rendered by tools/sim_render (jc-con) + ProxyPose RGB crops, 3x2 grid
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

PAGE = Path(__file__).resolve().parents[1]
IMG, VID = PAGE / 'assets/images', PAGE / 'assets/videos'
FIG = Path('/DATA2/jc/manuscript/fig')
P = Path('/DATA2/jc/physca_cf')
OUT = Path('/DATA2/jc/jc/outputs')
SIMF = Path(os.environ.get('SIM_FRAMES', PAGE / 'tools/sim_render/frames'))  # frames pulled out of jc-con (not committed)
FPS = 16


def host(p):
    """Resolve container symlinks (/home/jc -> /DATA2/jc)."""
    p = str(p)
    while os.path.islink(p):
        t = os.readlink(p)
        p = t.replace('/home/jc/', '/DATA2/jc/') if t.startswith('/') else os.path.join(os.path.dirname(p), t)
    return p


def read(path, size=None, crop=None, start=0, n=81):
    """Frames [start, start+n) as uint8 arrays; crop=(x,y,w,h) in source pixels, then resize to size x size."""
    path = host(path)
    w, h = map(int, subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                                    '-of', 'csv=p=0', path], capture_output=True, text=True, check=True).stdout.strip().split(',')[:2])
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vsync', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)[start:start + n]
    out = []
    for f in fr:
        im = Image.fromarray(f)
        if crop:
            x, y, cw, ch = crop
            im = im.crop((x, y, x + cw, y + ch))
        if size:
            im = im.resize((size, size), Image.LANCZOS)
        out.append(np.asarray(im))
    return out


def write(frames, path, crf=24, poster=40):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                          '-an', str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    Image.fromarray(frames[min(poster, len(frames) - 1)]).save(str(path).replace('/videos/', '/images/').replace('.mp4', '.webp'), quality=82)
    print('wrote', path, f'{os.path.getsize(path) / 1e6:.1f} MB')


def grid(cells, cols):
    """cells: list of frame lists (same size) -> list of tiled frames, white 4 px gutters."""
    n = min(len(c) for c in cells); s = cells[0][0].shape[0]; g = 4
    rows = (len(cells) + cols - 1) // cols
    out = []
    for t in range(n):
        canvas = np.full((rows * s + (rows - 1) * g, cols * s + (cols - 1) * g, 3), 255, np.uint8)
        for i, c in enumerate(cells):
            r, k = divmod(i, cols)
            canvas[r * (s + g):r * (s + g) + s, k * (s + g):k * (s + g) + s] = c[t]
        out.append(canvas)
    return out


# ---------------------------------------------------------------- figures
def figs():
    for name, w in [('teaser_v7', 2600), ('fig2_v2_pc', 2600), ('fig3_v2_4row', 2400), ('sim_rgb_v5', 1500)]:
        tmp = IMG / f'_{name}'
        subprocess.run(['pdftoppm', '-png', '-scale-to-x', str(w), '-scale-to-y', '-1', '-singlefile', str(FIG / f'{name}.pdf'), str(tmp)], check=True)
        im = Image.open(f'{tmp}.png').convert('RGB')
        im.save(IMG / f'{name}.webp', quality=90, method=6)
        os.remove(f'{tmp}.png')
        print('fig', name, im.size)


# ---------------------------------------------------------------- hero
def hero():
    u, clip, iv = 'clip-001973_69-149', 'clip-001973', '69-149'
    write(read(P / f'roots/orig/{u}/{clip}/hand_view_render/render_hand_{iv}.mp4', 480), VID / 'hero-action.mp4', 22, poster=0)
    write(read(P / f'rollouts/physca/orig/{u}/hot3d/{clip}/{u}_camera_obj_inference.mp4', 480), VID / 'hero-egospace.mp4', 22, poster=0)


# ---------------------------------------------------------------- baseline comparison
def compare():
    sys.path.insert(0, str(FIG / 'fig_3_src'))
    from srcs2 import src
    S = 360
    for key, c in [('hot3d-a', 'clip-003335_0-80'), ('hot3d-b', 'clip-002966_69-149'), ('h2o', 'subject3_ego__o1__5_0-80')]:
        gp, off = src(c, 'gt')
        gt = read(gp, S, start=off)
        V = {m: read(src(c, m)[0], S) for m in ['ours', 'egohoi', 'uni3c', 'egosim', 'gr', 'jcv']}
        first = [gt[0]] * len(gt)
        cells = [first, gt, V['ours'], V['egohoi'], V['uni3c'], V['egosim'], V['gr'], V['jcv']]
        write(grid(cells, 4), VID / f'compare-{key}.mp4', 25)


# ---------------------------------------------------------------- counterfactual
VIOLET, CYAN = np.array([150, 60, 255]), np.array([40, 215, 245])


def edge(m, w=5):
    im = Image.fromarray(m.astype(np.uint8) * 255)
    return (np.asarray(im.filter(ImageFilter.MaxFilter(w))) > 127) & ~(np.asarray(im.filter(ImageFilter.MinFilter(w))) > 127)


def cf():
    for cond, u in [('withdraw', 'clip-002838_30-110'), ('reroute', 'clip-001881_69-149')]:
        clip, iv = u.split('_')
        act = read(P / f'roots/{cond}/{u}/{clip}/hand_view_render/render_hand_{iv}.mp4', 480)
        scn = read(P / f'downloads/hot3d_prep_all/{clip}/view_render_old/render_{iv}.mp4', 480)
        hands = []
        for a, b in zip(act, scn):
            hm = Image.fromarray(((np.abs(a.astype(int) - b.astype(int)).sum(2) > 45) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
            hands.append(edge(np.asarray(hm.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.MaxFilter(7))) > 127, 7))
        cells = [[np.asarray(Image.fromarray(a).resize((400, 400), Image.LANCZOS)) for a in act]]
        for m in ['jcv', 'egosim', 'physca']:
            v = P / (f'rollouts/physca/{cond}/{u}/hot3d/{clip}/{u}_camera_obj_inference.mp4' if m == 'physca' else f'rollouts/{m}/{cond}/{u}/video.mp4')
            fr = read(v, 480)
            base = {'physca': 'recon2d', 'jcv': 'recon2d_jcv'}.get(m, 'recon2d_baselines')
            d = np.load(P / base / 'work' / f'{m}__{cond}__{u}' / 'masks.npz', allow_pickle=True)
            obj = np.unpackbits(d['obj'], axis=2)[..., :480].astype(bool)
            out = []
            for t, f in enumerate(fr):
                im = f.astype(float)
                if t < len(obj) and obj[t].any():
                    o = obj[t]
                    im[o] = 0.6 * im[o] + 0.4 * VIOLET
                    im[edge(o, 5)] = VIOLET
                if t < len(hands):
                    im[hands[t]] = CYAN
                out.append(np.asarray(Image.fromarray(im.astype(np.uint8)).resize((400, 400), Image.LANCZOS)))
            cells.append(out)
        write(grid(cells, 4), VID / f'cf-{cond}.mp4', 24)


# ---------------------------------------------------------------- simulation
def sim():
    S = 400
    for u in ['clip-002301_0-80', 'clip-002140_0-80']:
        clip, iv = u.split('_')
        rgb = [read(OUT / f'{d}/{clip}/{clip}_proxypose_{iv}.mp4', S, crop=(0, 0, 512, 512), n=80)
               for d in ['proxypose_hot3d_test', 'proxypose_hot3d_infer_ours', 'proxypose_hot3d_infer_camera_egohoihand_obj']]
        sims = []
        for m in ['gt_labels', 'infer_ours', 'egohoi']:
            fs = sorted((SIMF / u / m).glob('*.png'))[:80]
            sims.append([np.asarray(Image.open(f).convert('RGB').resize((S, S), Image.LANCZOS)) for f in fs])
        write(grid(rgb + sims, 3), VID / f'sim-{clip}.mp4', 24)


if __name__ == '__main__':
    IMG.mkdir(parents=True, exist_ok=True); VID.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ['figs', 'hero', 'compare', 'cf', 'sim']
    for t in todo:
        globals()[t]()
