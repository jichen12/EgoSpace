"""Collect and encode the project-page assets from local results (host, python3 + ffmpeg).

usage: python3 tools/build_assets.py [figs] [hero] [compare] [cf] [sim]   (no args = all)

Sources (all local, see README in this folder):
  figs     manuscript/fig/{teaser_v7,fig2_v2_pc,fig3_v2_4row,fig_cf_main_2row,sim_rgb_v5}.pdf -> webp
  hero     action video + EgoSpace rollout of clip-001973_69-149 (physca_cf roots/orig, rollouts/physca/orig)
  compare  the fig. 3 clips, 8-cell grid: first frame | real | EgoSpace | EgoHOI / Uni3C | EgoSim | GR | JCV
           (paths from manuscript/fig/fig_3_src/srcs2.py)
  cf       counterfactual clips (CF_CLIPS): four withdraw units + the reroute unit clip-002054_69-149 vs JCV;
           edited action | baseline | EgoSpace, cyan hand + violet SAM 3 object outline, labels burned in
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


ACCENT, TICK, CROSS = (23, 90, 158), (46, 157, 87), (214, 69, 61)
_LBL_FONT = '/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf'


def label(frame, text, verdict=None, ours=False, color=None, bottom=False):
    """Burn a page-style label into the top-left of a cell: dark chip (accent chip for ours) + green check / red cross."""
    from PIL import ImageDraw, ImageFont
    S = 3                                                  # draw at 3x, then downsample the label patch for smooth edges
    im = Image.fromarray(np.asarray(frame).astype(np.uint8))
    px = max(14, round(im.width * 0.042)) * S
    fnt = ImageFont.truetype(_LBL_FONT, px)
    tw = fnt.getlength(text); h = int(px * 1.55)
    w = int(tw + px * 0.7 + (h + px * 0.1 if verdict is not None else 0))
    patch = Image.new('RGBA', (w, h), (0, 0, 0, 0)); dr = ImageDraw.Draw(patch)
    fill = (tuple(color) + (235,)) if color is not None else ((ACCENT + (235,)) if ours else (0, 0, 0, 175))
    dr.rounded_rectangle([0, 0, w - 1, h - 1], int(px * 0.25), fill=fill)
    dr.text((px * 0.35, h / 2), text, font=fnt, fill=(255, 255, 255, 255), anchor='lm')
    if verdict is not None:
        cx, cy, rr = w - h / 2 - px * 0.12, h / 2, h * 0.36
        dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(TICK if verdict else CROSS) + (255,))
        lw = max(2, int(rr * 0.24))
        if verdict:
            dr.line([(cx - rr * .45, cy + rr * .02), (cx - rr * .12, cy + rr * .36), (cx + rr * .48, cy - rr * .34)], fill='white', width=lw, joint='curve')
        else:
            q = rr * .38
            dr.line([(cx - q, cy - q), (cx + q, cy + q)], fill='white', width=lw); dr.line([(cx - q, cy + q), (cx + q, cy - q)], fill='white', width=lw)
    patch = patch.resize((w // S, h // S), Image.LANCZOS)
    m = max(6, im.width // 60)
    im.paste(patch, (m, im.height - patch.height - m) if bottom else (m, m), patch)
    return np.asarray(im)


NAMES = {'egosim': 'EgoSim', 'jcv': 'JCV', 'uni3c': 'Uni3C', 'generated_reality': 'GR', 'egohoi': 'EgoHOI'}


def cf_clip(cond, u, bmodel, box, path, poster=40, S=480, obj_w=9, tag=None, base_hands=False):
    """edited action | baseline | EgoSpace for one counterfactual unit; cyan commanded-hand outline, violet SAM 3
    object outline, labels burned in (baseline ✗, EgoSpace ✓). box = crop in the 480 px frame (None = full frame).
    obj_w: object outline width; tag: extra red note at the bottom of the baseline cell;
    base_hands: outline the baseline's own generated hands (its SAM 3 hand mask) instead of the commanded hand."""
    box = box or (0, 0, 480, 480)

    def lum(x): return x @ np.array([0.299, 0.587, 0.114])

    def match_brightness(im, ref):                        # EgoSim renders much darker (as in the teaser)
        q = np.linspace(1, 99, 50); y, yr = lum(im), lum(ref)
        y2 = np.interp(y, np.percentile(y, q), np.percentile(yr, q))
        return (im * (y2 / np.maximum(y, 1))[..., None]).clip(0, 255)

    def up(m):
        return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).crop(box).resize((S, S), Image.BILINEAR)) > 127

    clip, iv = u.split('_')
    act = read(P / f'roots/{cond}/{u}/{clip}/hand_view_render/render_hand_{iv}.mp4', 480)
    scn = read(P / f'downloads/hot3d_prep_all/{clip}/view_render_old/render_{iv}.mp4', 480)
    ours = read(P / f'rollouts/physca/{cond}/{u}/hot3d/{clip}/{u}_camera_obj_inference.mp4', 480)
    bpath = (P / f'rollouts/egohoi/{cond}/{u}/hot3d/{clip}/{u}_camera_hand_obj_inference.mp4' if bmodel == 'egohoi'
             else P / f'rollouts/{bmodel}/{cond}/{u}/video.mp4')
    base = read(bpath, 480)
    hands = []
    for a_, b_ in zip(act, scn):
        hm = Image.fromarray(((np.abs(a_.astype(int) - b_.astype(int)).sum(2) > 45) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
        hands.append(edge(up(np.asarray(hm.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.MaxFilter(7))) > 127), 7))
    crop = lambda f: np.asarray(Image.fromarray(np.asarray(f).clip(0, 255).astype(np.uint8)).crop(box).resize((S, S), Image.LANCZOS))
    cells = [[label(crop(a_), 'Edited action') for a_ in act]]
    for m, fr in ((bmodel, base), ('physca', ours)):
        root = {'physca': 'recon2d', 'egohoi': 'recon2d', 'jcv': 'recon2d_jcv'}.get(m, 'recon2d_baselines')
        d = np.load(P / root / 'work' / f'{m}__{cond}__{u}' / 'masks.npz', allow_pickle=True)
        obj = np.unpackbits(d['obj'], axis=2)[..., :480].astype(bool)
        own = np.unpackbits(d['hands'], axis=2)[..., :480].astype(bool) if (base_hands and m != 'physca') else None
        out = []
        for t, f in enumerate(fr):
            im = f.astype(float)
            if m == 'egosim': im = match_brightness(im, ours[t].astype(float))
            im = crop(im).copy()
            if t < len(obj) and obj[t].any():
                im[edge(up(obj[t]), obj_w)] = VIOLET
            if own is not None:
                if t < len(own): im[edge(up(own[t]), 7)] = CYAN
            elif t < len(hands):
                im[hands[t]] = CYAN
            if m == 'physca':
                im = label(im, 'EgoSpace', True, ours=True)
            else:
                im = label(im, NAMES[m], False)
                if tag: im = label(im, tag, color=CROSS, bottom=True)
            out.append(im)
        cells.append(out)
    write(grid(cells, 3), path, crf=18, poster=poster)


# page clips: (file, cond, unit, baseline, crop box, poster frame, options). Withdraw picks: user review 2026-11-19
CF_CLIPS = [
    ('cf-withdraw-clip-002838', 'withdraw', 'clip-002838_30-110', 'jcv', None, 64, dict(obj_w=3, tag='picks up the wrong object')),
    ('cf-withdraw-clip-001995', 'withdraw', 'clip-001995_69-149', 'egohoi', None, 64, dict(base_hands=True)),
    ('cf-withdraw-clip-001918', 'withdraw', 'clip-001918_60-140', 'jcv', None, 64, {}),
    ('cf-withdraw-clip-001998', 'withdraw', 'clip-001998_0-80', 'egosim', None, 56, {}),
    ('cf-reroute', 'reroute', 'clip-002054_69-149', 'jcv', (110, 40, 470, 400), 68, {}),
]


def cf():
    """Page counterfactual clips (the reroute unit and clip-001998 are shared with the project video)."""
    for name, cond, u, b, box, poster, opt in CF_CLIPS:
        cf_clip(cond, u, b, box, VID / f'{name}.mp4', poster, **opt)


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
