'use strict';

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const dialogOpen = () => Boolean(document.querySelector('dialog[open]'));

// Opening pair: the action video and the rollout share a frame clock, so they start, loop and pause together.
const openingVideos = [...document.querySelectorAll('[data-playback-group="opening"]')];
let openingVisible = false;
function playOpening() {
  if (!openingVisible || document.hidden || reduceMotion.matches || dialogOpen()) return;
  openingVideos.forEach(video => video.play().catch(() => {}));
}
openingVideos.forEach(video => {
  video.addEventListener('seeked', () => openingVideos.forEach(other => {
    if (other !== video && Math.abs(other.currentTime - video.currentTime) > 0.08) other.currentTime = video.currentTime;
  }));
  video.addEventListener('play', () => openingVideos.forEach(other => { if (other.paused) other.play().catch(() => {}); }));
  video.addEventListener('pause', () => { if (openingVisible && !document.hidden) openingVideos.forEach(other => other.pause()); });
});
if (openingVideos.length) {
  new IntersectionObserver(entries => {
    openingVisible = entries[0].isIntersecting;
    if (openingVisible) playOpening();
    else openingVideos.forEach(video => video.pause());
  }, {threshold: .15}).observe(document.querySelector('.opening-pair'));
}

// Comparison grids autoplay muted and looping while at least half on screen,
// unless the viewer paused them there; they pause off screen.
const autoVideos = [...document.querySelectorAll('video.autoplay')];
const visible = new Set();
const pausedByViewer = new WeakSet();
function playVisible(videos) {
  if (document.hidden || reduceMotion.matches || dialogOpen()) return;
  videos.forEach(video => { if (!pausedByViewer.has(video)) video.play().catch(() => {}); });
}
const autoObserver = new IntersectionObserver(entries => entries.forEach(entry => {
  const video = entry.target;
  if (entry.isIntersecting) {
    visible.add(video);
    if (video.preload === 'none') video.preload = 'auto';
    playVisible([video]);
  } else {
    visible.delete(video);
    pausedByViewer.delete(video);
    video.pause();
  }
}), {threshold: .5});
autoVideos.forEach(video => {
  autoObserver.observe(video);
  video.addEventListener('play', () => pausedByViewer.delete(video));
  video.addEventListener('pause', () => {
    if (visible.has(video) && !document.hidden && !dialogOpen() && !video.dataset.swapping) pausedByViewer.add(video);
  });
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { autoVideos.forEach(v => v.pause()); openingVideos.forEach(v => v.pause()); }
  else { playVisible(visible); playOpening(); }
});

// Tabs that swap the clip shown in one grid video.
function bindTabs(selector, attr, video, srcFor) {
  const buttons = [...document.querySelectorAll(selector)];
  buttons.forEach(button => button.addEventListener('click', () => {
    buttons.forEach(item => { item.classList.toggle('active', item === button); item.setAttribute('aria-pressed', String(item === button)); });
    const key = button.dataset[attr];
    video.dataset.swapping = '1';
    video.pause();
    video.poster = `assets/images/${srcFor(key)}.webp`;
    video.src = `assets/videos/${srcFor(key)}.mp4`;
    video.load();
    delete video.dataset.swapping;
    pausedByViewer.delete(video);
    if (visible.has(video)) playVisible([video]);
  }));
}
bindTabs('[data-compare]', 'compare', document.querySelector('#compare-video'), key => `compare-${key}`);
bindTabs('[data-sim]', 'sim', document.querySelector('#sim-video'), key => `sim-${key}`);

// Enlarged paper figures.
const figureDialog = document.querySelector('#figure-dialog');
const figureImage = document.querySelector('#figure-image');
document.querySelectorAll('.figure-open').forEach(button => {
  button.addEventListener('click', () => {
    const source = button.querySelector('img');
    document.querySelector('#figure-title').textContent = button.dataset.figureTitle;
    figureImage.src = source.src;
    figureImage.alt = source.alt;
    document.querySelectorAll('video').forEach(video => video.pause());
    figureDialog.showModal();
    document.body.style.overflow = 'hidden';
    document.querySelector('.figure-viewport').scrollTo(0, 0);
  });
});
document.querySelector('#figure-close').addEventListener('click', () => figureDialog.close());
figureDialog.addEventListener('close', () => { document.body.style.overflow = ''; playVisible(visible); playOpening(); });
figureDialog.addEventListener('click', event => {
  const b = figureDialog.getBoundingClientRect();
  if (event.clientX < b.left || event.clientX > b.right || event.clientY < b.top || event.clientY > b.bottom) figureDialog.close();
});

// Visual versus physical ranking: PSNR (Table 1) against simulated Trans-AUC10 (Table 2).
(function rankChart() {
  const host = document.querySelector('#rank-chart');
  if (!host) return;
  const data = [
    ['Wan 2.1', 12.49, .0884], ['EgoHOI', 19.36, .1458], ['Uni3C', 17.64, .1453], ['EgoSim', 14.29, .1594],
    ['GR', 14.50, .1291], ['JCV', 14.88, .1299], ['EgoSpace', 20.96, .2108, 'ours'],
  ];
  const offsets = {'Wan 2.1': [8, 4], EgoHOI: [9, 16], Uni3C: [-8, -8, 'end'], EgoSim: [8, -8], GR: [-8, 4, 'end'], JCV: [8, 12], EgoSpace: [-10, -10, 'end']};
  const W = 440, H = 320, m = {l: 52, r: 16, t: 14, b: 44};
  const x = v => m.l + (v - 11) / (22 - 11) * (W - m.l - m.r);
  const y = v => H - m.b - (v - .06) / (.26 - .06) * (H - m.t - m.b);
  const ns = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text != null) node.textContent = text;
    return node;
  };
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, 'aria-hidden': 'true'});
  for (let v = 12; v <= 22; v += 2) {
    svg.append(el('line', {class: 'gridline', x1: x(v), x2: x(v), y1: m.t, y2: H - m.b}));
    svg.append(el('text', {class: 'tick-label', x: x(v), y: H - m.b + 16, 'text-anchor': 'middle'}, v));
  }
  for (let v = .08; v <= .261; v += .04) {
    svg.append(el('line', {class: 'gridline', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v)}));
    svg.append(el('text', {class: 'tick-label', x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end'}, v.toFixed(2)));
  }
  svg.append(el('line', {class: 'axis', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b}));
  svg.append(el('line', {class: 'axis', x1: m.l, x2: m.l, y1: m.t, y2: H - m.b}));
  svg.append(el('text', {class: 'axis-label', x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle'}, 'PSNR (dB) → visual fidelity'));
  svg.append(el('text', {class: 'axis-label', x: 14, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`}, 'Trans-AUC₁₀ → simulated execution'));
  svg.append(el('line', {class: 'ref', x1: m.l, x2: W - m.r, y1: y(.2388), y2: y(.2388)}));
  svg.append(el('text', {class: 'ref-label', x: m.l + 6, y: y(.2388) - 6}, 'Real video 0.239'));
  // Rank reversal: Uni3C beats EgoSim on PSNR but loses on executability.
  const u = data[2], e = data[3];
  svg.append(el('path', {class: 'pair', d: `M${x(u[1])},${y(u[2])} L${x(e[1])},${y(e[2])}`}));
  data.forEach(([name, psnr, auc, kind]) => {
    const hl = name === 'Uni3C' || name === 'EgoSim';
    const dot = el('circle', {class: `pt ${kind || ''} ${hl ? 'hl' : ''}`, cx: x(psnr), cy: y(auc), r: kind ? 6 : 4.5});
    dot.append(el('title', {}, `${name}: PSNR ${psnr.toFixed(2)} dB, Trans-AUC10 ${auc.toFixed(4)}`));
    svg.append(dot);
    const [dx, dy, anchor] = offsets[name];
    svg.append(el('text', {class: `pt-label ${kind || ''}`, x: x(psnr) + dx, y: y(auc) + dy, 'text-anchor': anchor || 'start'}, name));
  });
  host.append(svg);
})();
