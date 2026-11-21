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


def make_chip(text, verdict=None, ours=False, color=None, px=20):
    """RGBA label chip in the page style (dark / accent / given colour), optional green check or red cross."""
    from PIL import ImageDraw, ImageFont
    S = 3                                                  # draw at 3x, then downsample for smooth edges
    px = int(px * S)
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
    return patch.resize((w // S, h // S), Image.LANCZOS)


def label(frame, text, verdict=None, ours=False, color=None, bottom=False):
    """Burn a chip into the top-left (or bottom-left) corner of a cell."""
    im = Image.fromarray(np.asarray(frame).astype(np.uint8))
    patch = make_chip(text, verdict, ours, color, px=max(14, round(im.width * 0.042)))
    m = max(6, im.width // 60)
    im.paste(patch, (m, im.height - patch.height - m) if bottom else (m, m), patch)
    return np.asarray(im)


def inset(frame, small, note=None, frac=0.36, border=None):
    """Picture-in-picture: `small` (the edited action) in the top-right corner, labelled, with an optional note below."""
    im = Image.fromarray(np.asarray(frame).astype(np.uint8))
    W_ = im.width; s = int(W_ * frac); m = max(6, W_ // 60); bw = 3
    sm = Image.fromarray(np.asarray(small).astype(np.uint8)).resize((s, s), Image.LANCZOS)
    sm = label(sm, 'Edited action (input)')                 # chip scaled to the inset
    box = Image.new('RGB', (s + 2 * bw, s + 2 * bw), tuple(border) if border else (255, 255, 255))
    box.paste(Image.fromarray(np.asarray(sm)), (bw, bw))
    x0 = W_ - box.width - m
    im.paste(box, (x0, m))
    if note:
        ch = make_chip(note, color=ORANGE, px=max(12, round(W_ * 0.03)))
        im.paste(ch, (W_ - ch.width - m, m + box.height + 4), ch)
    return np.asarray(im)


NAMES = {'egosim': 'EgoSim', 'jcv': 'JCV', 'uni3c': 'Uni3C', 'generated_reality': 'GR', 'egohoi': 'EgoHOI'}


ORANGE = (232, 128, 40)


def cf_strip(width, height, parts, bg=(255, 255, 255), col_w=None):
    """One caption line: parts = list of (x_center or None, [(text, colour)], icon) centred in their column (None = whole width).
    icon: None | True (green check) | False (red cross), drawn before the text."""
    from PIL import ImageDraw, ImageFont
    S = 3
    im = Image.new('RGB', (width * S, height * S), bg); dr = ImageDraw.Draw(im)
    px = int(height * 0.44) * S
    fb = ImageFont.truetype(_LBL_FONT, px)
    fr = ImageFont.truetype(_LBL_FONT.replace('-Bold', '-Regular'), px)
    for xc, runs, icon in parts:
        avail = ((col_w if xc is not None and col_w else width) - 20) * S
        p_ = px
        while True:                                       # shrink this caption until it fits its column
            fb_ = ImageFont.truetype(_LBL_FONT, p_); fr_ = ImageFont.truetype(_LBL_FONT.replace('-Bold', '-Regular'), p_)
            fonts = [fb_ if i == 0 or col != (60, 64, 72) else fr_ for i, (t, col) in enumerate(runs)]
            tw = sum(f.getlength(t) for f, (t, _) in zip(fonts, runs))
            r = p_ * 0.48; iw = (2 * r + p_ * 0.3) if icon is not None else 0
            if tw + iw <= avail or p_ < px * 0.6: break
            p_ = int(p_ * 0.95)
        x = (xc * S if xc is not None else width * S / 2) - (tw + iw) / 2; y = height * S / 2
        if icon is not None:
            cx = x + r; col = TICK if icon else CROSS
            dr.ellipse([cx - r, y - r, cx + r, y + r], fill=col); lw = max(2, int(r * 0.24))
            if icon:
                dr.line([(cx - r * .45, y + r * .02), (cx - r * .12, y + r * .36), (cx + r * .48, y - r * .34)], fill='white', width=lw, joint='curve')
            else:
                q = r * .38
                dr.line([(cx - q, y - q), (cx + q, y + q)], fill='white', width=lw); dr.line([(cx - q, y + q), (cx + q, y - q)], fill='white', width=lw)
            x += iw
        px_used = p_
        for f, (t, col) in zip(fonts, runs):
            dr.text((x, y), t, font=f, fill=col, anchor='lm'); x += f.getlength(t)
    return np.asarray(im.resize((width, height), Image.LANCZOS))


CF_TAGS = {  # bottom captions: original (real), edit, baseline, EgoSpace
    'withdraw': ('original: the hand grasps the object', 'edit: the hand stops 5 cm short', 'object moves anyway', 'object stays still'),
    'reroute': ('original: the hand carries the object', 'edit: the hand is moved 15 cm away', 'object is left behind', 'object follows the hand')}


def cf_clip(cond, u, bmodel, box, path, poster=40, S=480, obj_w=9, tag=None, base_hands=False, base_name='Baseline',
            real=True, t_edit=None, t_resp=None, captions=None):
    """edited action | baseline | EgoSpace for one counterfactual unit; cyan commanded-hand outline, violet SAM 3
    object outline, labels burned in (baseline ✗, EgoSpace ✓). box = crop in the 480 px frame (None = full frame).
    obj_w: object outline width; tag: extra red note at the bottom of the baseline cell;
    base_hands: outline the baseline's own generated hands (its SAM 3 hand mask) instead of the commanded hand.
    Layout: real video (GT, original action) | EgoSpace | baseline; the edited action is a picture-in-picture in the
    top-right corner of both rollout cells; the only caption (what the edit does) sits under the EgoSpace column.
    real: prepend the real recorded video (= the original, unedited action) so the edit is readable;
    t_edit / t_resp: kept for reference (edit start / visible violation); captions are no longer timed."""
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
    t0 = int(iv.split('-')[0])
    tags = CF_TAGS[cond]
    tag_at = lambda im, t, i, col: label(im, tags[i], color=col, bottom=True) if (t_resp is None or t >= t_resp) else im
    small = [crop(a_) for a_ in act]
    edited = lambda t: t_edit is None or t >= t_edit
    pip = lambda im, t: inset(im, small[t])            # white border, no note (user: one caption only, under EgoSpace)
    cells = []
    if real:
        rv = read(P / f'downloads/hot3d_prep_all/{clip}/{clip}_rgb.mp4', 480, start=t0, n=len(act))
        cells.append([label(crop(f), 'Real video (GT)') for f in rv])
    for m, fr in (('physca', ours), (bmodel, base)):
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
                im = label(im, base_name or NAMES[m], False)
                if tag: im = label(im, tag, color=CROSS, bottom=True)
            out.append(pip(im, t) if t < len(small) else im)
        cells.append(out)
    frames = grid(cells, len(cells))
    # captions (whole clip): top = the edit and what it implies; bottom = per column, does the rollout follow the edit?
    Wd = frames[0].shape[1]; col_c = lambda i: i * (S + 4) + S / 2
    DARK, GREY = (35, 38, 44), (60, 64, 72)
    cap = captions
    top = cf_strip(Wd, int(S * 0.12) // 2 * 2, [(None, [('Edited action: ', ORANGE), (cap['edit'], DARK)], None)], bg=(246, 247, 249))
    bot = cf_strip(Wd, int(S * 0.11) // 2 * 2, [(col_c(0), [('original action: ', GREY), (cap['gt'], GREY)], None),
                                        (col_c(1), [('follows the edit: ', TICK), (cap['ours'], DARK)], True),
                                        (col_c(2), [('ignores the edit: ', CROSS), (cap['base'], DARK)], False)], col_w=S)
    write([np.vstack([top, f, bot]) for f in frames], path, crf=18, poster=poster)


# captions: top strip = the edit and what it implies; bottom = per column whether the rollout follows the edit
CAP_WITHDRAW = dict(edit='the hand now stops 5 cm before the flask instead of grasping it, so the flask should stay still',
                    gt='the hand grasps the flask', ours='the flask stays still', base='the flask still moves')
CAP_REROUTE = dict(edit='after grasping the bottle, the hand is moved 15 cm elsewhere, so the bottle should move with the hand',
                   gt='the hand carries the bottle', ours='the bottle moves along', base='the bottle stays behind')
# page clips: (file, cond, unit, baseline, crop box, poster frame, options). Withdraw picks: user review 2026-11-19;
# the baseline cell is labelled just 'Baseline' (user request), not the method name
CF_CLIPS = [
    # t_edit: edited action leaves the original (withdraw: t_ref - ease frames; reroute: t_start, specs/<unit>/spec.json)
    # t_resp: baseline violation becomes visible (results2d first_moved / first_detach)
    ('cf-withdraw-clip-001995', 'withdraw', 'clip-001995_69-149', 'egohoi', None, 66, dict(base_hands=True, t_edit=9, t_resp=44, captions=CAP_WITHDRAW)),
    ('cf-withdraw-clip-001918', 'withdraw', 'clip-001918_60-140', 'jcv', None, 66, dict(t_edit=3, t_resp=22, captions=CAP_WITHDRAW)),
    ('cf-reroute', 'reroute', 'clip-002054_69-149', 'jcv', (110, 40, 470, 400), 68, dict(t_edit=28, t_resp=43, captions=CAP_REROUTE)),
]


def cf():
    """Page counterfactual clips (the reroute unit and clip-001998 are shared with the project video)."""
    for name, cond, u, b, box, poster, opt in CF_CLIPS:
        cf_clip(cond, u, b, box, VID / f'{name}.mp4', poster, **opt)


# ---------------------------------------------------------------- simulation
def sim():
    """Dexterous-execution and project-video assets, rendered by tools/video (build_video.sh): EgoSpace rollout ->
    SPIDER simulation with per-hand contact-force maps and per-finger force shares; plus the 3D+t counterfactual shots
    and the project video itself. Copies the mp4s and writes a poster per clip."""
    import shutil, subprocess
    V = PAGE / 'video'
    for src, dst, t in [('egospace_dex_002301', 'sim-clip-002301', 5), ('egospace_dex_002140', 'sim-clip-002140', 5),
                        ('egospace_dex_o2', 'sim-h2o-book', 5), ('egospace_video', 'egospace-video', 2.5)]:
        shutil.copy(V / f'{src}.mp4', VID / f'{dst}.mp4')
        png = IMG / f'{dst}.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-i', str(VID / f'{dst}.mp4'), '-frames:v', '1',
                        '-vf', 'scale=1280:-2', str(png)], check=True)
        Image.open(png).save(IMG / f'{dst}.webp', quality=82); png.unlink()
    cfstack()


def cfstack():
    """One video per counterfactual experiment: the 3D+t shot of the edited action (tools/video cf3d) on top of the
    real / EgoSpace / baseline comparison (cf-*.mp4, played at 12 fps, then held on its last frame until the 3D shot ends)."""
    import subprocess
    V = PAGE / 'video'
    for src3d, cmp_, dst in [('egospace_cf3d_withdraw_001995', 'cf-withdraw-clip-001995', 'cfstack-withdraw-clip-001995'),
                             ('egospace_cf3d_reroute_002054', 'cf-reroute', 'cfstack-reroute')]:
        out = VID / f'{dst}.mp4'
        fc = ('[0:v]fps=30,scale=1448:-2:flags=lanczos,setsar=1[a];'
              '[1:v]setpts=PTS*16/12,fps=30,pad=1448:600:0:12:white,tpad=stop_mode=clone:stop_duration=20,setsar=1[b];'
              '[a][b]vstack=shortest=1,format=yuv420p[v]')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(V / f'{src3d}.mp4'), '-i', str(VID / f'{cmp_}.mp4'),
                        '-filter_complex', fc, '-map', '[v]', '-c:v', 'libx264', '-preset', 'slow', '-crf', '21',
                        '-movflags', '+faststart', '-an', str(out)], check=True)
        png = IMG / f'{dst}.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '5', '-i', str(out), '-frames:v', '1', '-vf', 'scale=1280:-2', str(png)], check=True)
        Image.open(png).save(IMG / f'{dst}.webp', quality=82); png.unlink()


if __name__ == '__main__':
    IMG.mkdir(parents=True, exist_ok=True); VID.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ['figs', 'hero', 'compare', 'cf', 'sim']
    for t in todo:
        globals()[t]()
