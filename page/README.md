# EgoSpace project page

Static page in the TouchScale / UMI style (`index.html`, `styles.css`, `app.js`). Preview locally:

    python3 -m http.server 8000   # then open http://localhost:8000

Text and numbers come from `manuscript/main.tex` (appendix excluded). Authors, paper/code links and the BibTeX entry are
placeholders until the paper is public. `--accent` in `styles.css` sets the theme colour (EgoSpace blue; TouchScale used #8c1515).

## Assets (`assets/`, ~9 MB)
Rebuilt with `python3 tools/build_assets.py [figs|hero|compare|cf|sim]` on the host (needs ffmpeg, pdftoppm, PIL):
- `images/{teaser_v7,fig2_v2_pc,fig3_v2_4row,sim_rgb_v5}.webp` — paper figures rasterised from `manuscript/fig/*.pdf`.
- `videos/hero-*.mp4` — action video and EgoSpace rollout of HOT3D clip-001973_69-149 (physca_cf roots/orig, rollouts/physca/orig).
- `videos/compare-*.mp4` — Fig. 3 clips (clip-003335_0-80, clip-002966_69-149, subject3_ego__o1__5_0-80), 4x2 grid:
  first frame | real | EgoSpace | EgoHOI / Uni3C | EgoSim | GR | JCV (paths from `manuscript/fig/fig_3_src/srcs2.py`).
- `videos/cf-{withdraw,reroute}.mp4` — counterfactual figure units (clip-002838_30-110, clip-001881_69-149):
  edited action | JCV | EgoSim | EgoSpace, cyan commanded hand + violet SAM 3 object mask (as in fig_cf_main).
- `videos/sim-clip-{002301,002140}.mp4` — Fig. 4 simulation units: ProxyPose RGB crops (real / EgoSpace / EgoHOI) over
  SPIDER rollouts (GT labels / EgoSpace / EgoHOI). The SPIDER frame sequences are rendered in `jc-con` with
  `tools/sim_render/run.sh` (copy `render_seq.py` to /tmp/egospace_render_seq.py), then pulled out to `tools/sim_render/frames/<clip>_0-80/<method>/` (or pass `SIM_FRAMES`).
- Posters (`images/*.webp` named like the videos) are frame 40 of each grid, frame 0 for the hero pair.
