# EgoSpace project page

Static page in the TouchScale / UMI style (`index.html`, `styles.css`, `app.js`). Preview locally:

    python3 -m http.server 8000   # then open http://localhost:8000

Text and numbers come from `manuscript/main.tex` (appendix excluded). Authors are a
placeholder until the paper is public (Resources and Citation sections were removed 2026-11-22). `--accent` in `styles.css` sets the theme colour (EgoSpace blue; TouchScale used #8c1515).

## Assets (`assets/`, ~9 MB)
Rebuilt with `python3 tools/build_assets.py [figs|hero|compare|cf|sim]` on the host (needs ffmpeg, pdftoppm, PIL):
- `images/{teaser_v7,fig2_v2_pc,fig3_v2_4row,sim_rgb_v5}.webp` — paper figures rasterised from `manuscript/fig/*.pdf`.
- `videos/hero-*.mp4` — action video and EgoSpace rollout of HOT3D clip-001973_69-149 (physca_cf roots/orig, rollouts/physca/orig).
- `videos/compare-*.mp4` — Fig. 3 clips (clip-003335_0-80, clip-002966_69-149, subject3_ego__o1__5_0-80), 4x2 grid:
  first frame | real | EgoSpace | EgoHOI / Uni3C | EgoSim | GR | JCV (paths from `manuscript/fig/fig_3_src/srcs2.py`).
- `videos/cf-withdraw-clip-001995.mp4`, `videos/cf-reroute.mp4` — counterfactual clips (`CF_CLIPS` in
  build_assets.py; the page shows only these two): real video (GT) | EgoSpace | baseline with the edited action as a top-right inset,
  480 px per cell; top strip = the edit and its implication, bottom strip = original action / follows the edit / ignores the edit
  (CAP_WITHDRAW, CAP_REROUTE), cyan hand + violet SAM 3 object outline, labels burned in. Baseline cell labelled 'Baseline'
  (001995: EgoHOI, outlined with EgoHOI's own SAM 3 hand mask; reroute clip-002054_69-149: JCV).
- `sim` step (copies from `page/video/`, rendered by `tools/video/build_video.sh`, see tools/video/README.md):
  `videos/cf3d-{withdraw-clip-001995,reroute}.mp4` — 3D+t point-cloud shot of each edited action, shown left of the clips;
  `videos/sim-{h2o-book,clip-002301,clip-002140}.mp4` — dexterous execution: EgoSpace rollout -> SPIDER simulation with
  live contact-force maps of the hands that touch the object and per-finger force shares (002301 / 002140: left hand only);
  `videos/egospace-video.mp4` — the project video (Video section).
- Posters (`images/*.webp` named like the videos) are frame 40 of each grid, frame 0 for the hero pair.
