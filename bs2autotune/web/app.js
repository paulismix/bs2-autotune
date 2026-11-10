/* BS2 Autotune - user interface.  Plain JavaScript, no dependencies. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const SVG = 'http://www.w3.org/2000/svg';

  /** Create an element; text is always set as text, never as markup. */
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children) if (c !== null && c !== undefined) node.append(c);
    return node;
  }
  function svg(tag, attrs, text) {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      // colours go through the style object: the page forbids inline style attributes
      if (k === 'stroke-var') node.style.stroke = `var(${v})`;
      else if (k === 'fill-var') node.style.fill = `var(${v})`;
      else node.setAttribute(k, v);
    }
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const fmtTime = (s) => (s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}` : `${s.toFixed(1)} s`);
  const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const noteName = (m) => `${NOTE_NAMES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`;

  let state = null;
  let devices = null;
  let job = null;
  const ui = { region: null, kind: 'stem', drag: null, lastJobStatus: {}, specCache: {}, refFile: null };

  // ------------------------------------------------------------------ server
  async function api(path, body, raw) {
    const opts = { method: body === undefined && !raw ? 'GET' : 'POST', headers: {} };
    if (raw) {
      opts.body = raw.body;
      opts.headers = raw.headers;
    } else if (body !== undefined) {
      opts.body = JSON.stringify(body);
      opts.headers['Content-Type'] = 'application/json';
    }
    let res;
    try {
      res = await fetch(path, opts);
    } catch (err) {
      throw new Error('The app is not answering. Is it still running in the terminal?');
    }
    let data = null;
    try { data = await res.json(); } catch (err) { /* not JSON */ }
    if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status}).`);
    return data;
  }

  function banner(message, info) {
    const b = $('banner');
    if (!message) { b.hidden = true; return; }
    b.textContent = message;
    b.classList.toggle('is-info', !!info);
    b.hidden = false;
  }
  async function act(fn) {
    try { banner(null); await fn(); } catch (err) { banner(err.message); }
  }

  async function refreshState() {
    state = await api('/api/state');
    if (state.job && (!job || state.job.id >= job.id)) job = state.job;
    render();
  }
  async function refreshDevices() {
    devices = await api('/api/devices');
    renderConnect();
    renderReference();
  }
  let stateTimer = null;
  function refreshSoon() {
    clearTimeout(stateTimer);
    stateTimer = setTimeout(() => refreshState().catch(() => {}), 120);
  }

  function connectEvents() {
    const source = new EventSource('/api/events');
    source.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); } catch (err) { return; }
      if (ev.type === 'job') {
        const prev = job;
        job = ev.job;
        renderJob();
        if (!prev || prev.id !== job.id || prev.status !== job.status) {
          if (job.status !== 'running') {
            if (job.status === 'error') banner(job.error);
            refreshSoon();
          }
        }
      } else if (ev.type === 'state') {
        refreshSoon();
      }
    };
    setInterval(() => { if (job && job.status === 'running') refreshState().catch(() => {}); }, 6000);
  }

  const running = () => !!(job && job.status === 'running');

  // -------------------------------------------------------------------- top
  function renderTop() {
    const s = state.settings;
    $('mode-chip').textContent = state.mode === 'demo' ? 'Demo: virtual synth'
      : `Bass Station II${s.midi_out ? ' · ' + s.midi_out : ''}`;
    $('version').textContent = `BS2 Autotune ${state.version}`;
  }

  // ---------------------------------------------------------------- connect
  function fillSelect(select, options, value, placeholder) {
    select.replaceChildren();
    if (placeholder) select.append(el('option', { value: '', text: placeholder }));
    let found = value === '' || value === undefined;
    for (const opt of options) {
      const o = el('option', { value: String(opt.value), text: opt.label });
      if (String(opt.value) === String(value)) { o.selected = true; found = true; }
      select.append(o);
    }
    if (!found) {
      const o = el('option', { value: String(value), text: `${value} (not connected)` });
      o.selected = true;
      select.append(o);
    }
  }

  function renderConnect() {
    if (!state) return;
    const s = state.settings;
    const demo = state.mode === 'demo';
    for (const b of document.querySelectorAll('#step-connect .seg button')) {
      b.setAttribute('aria-checked', String(b.dataset.mode === state.mode));
    }
    $('hardware-fields').hidden = demo;
    $('btn-refresh').hidden = demo;
    $('mode-hint').textContent = demo
      ? 'No hardware needed: the app matches sounds on a simulated synth whose response it does not know. Good for trying the workflow.'
      : '';
    if (devices && !demo) {
      fillSelect($('midi-out'), devices.midi_outputs.map((n) => ({ value: n, label: n })), s.midi_out, 'Choose …');
      fillSelect($('midi-in'), devices.midi_inputs.map((n) => ({ value: n, label: n })), s.midi_in, 'None');
      fillSelect($('midi-channel'), Array.from({ length: 16 }, (_, i) => ({ value: i + 1, label: String(i + 1) })), s.midi_channel);
      fillSelect($('audio-device'), devices.audio_inputs.map((d) => ({ value: d.name, label: `${d.name} (${d.channels} in)` })), s.audio_device, 'Choose …');
      const dev = devices.audio_inputs.find((d) => d.name === s.audio_device);
      const n = dev ? dev.channels : Math.max(2, s.input_channel);
      fillSelect($('input-channel'), Array.from({ length: n }, (_, i) => ({ value: i + 1, label: `Input ${i + 1}` })), s.input_channel);
      $('fw-features').checked = !!s.firmware_features;
      const problems = [devices.midi_error && `MIDI: ${devices.midi_error}`, devices.audio_error && `Audio: ${devices.audio_error}`].filter(Boolean);
      $('device-error').hidden = problems.length === 0;
      $('device-error').textContent = problems.join(' · ');
    }
    const ready = demo || (s.midi_out && s.audio_device);
    $('connect-status').textContent = demo ? 'Virtual synth' : (ready ? 'Devices chosen' : 'Choose the MIDI output and the audio input');

    const cal = state.calibration;
    $('cal-status').textContent = cal
      ? `Calibrated ${cal.date} · latency ${cal.latency_ms} ms · tuning ${cal.tune_cents >= 0 ? '+' : ''}${cal.tune_cents} cents`
      : (demo ? 'Not calibrated yet (done automatically before the first match)' : 'Not calibrated yet');
    $('btn-calibrate').textContent = cal ? 'Calibrate again' : 'Calibrate';
    const liveSteps = job && job.kind === 'calibration' && job.data.steps ? job.data.steps : null;
    const steps = liveSteps || (cal ? cal.steps : []);
    const list = $('cal-steps');
    list.replaceChildren();
    for (const st of steps) {
      list.append(el('li', { class: st.ok ? 'ok' : 'bad' },
        el('span', { class: 'mark', text: st.ok ? '✓' : '!' , 'aria-label': st.ok ? 'OK' : 'Attention' }),
        el('span', { class: 'name', text: st.name }),
        el('span', { class: 'msg', text: st.message })));
    }
    $('cal-charts').hidden = !cal;
    if (cal) {
      lineChart($('chart-cutoff'), {
        title: 'Filter cutoff', subtitle: 'frequency at each knob value, measured at C3',
        xLabel: 'Filter frequency (knob value)', yLog: true, yUnit: 'Hz',
        series: [
          { name: 'Classic', color: '--series-1', points: cal.cutoff },
          { name: 'Acid', color: '--series-2', points: cal.cutoff_acid },
        ],
      });
      lineChart($('chart-env'), {
        title: 'Envelope times', subtitle: 'seconds at each knob value',
        xLabel: 'Attack / decay / release (knob value)', yLog: true, yUnit: 's',
        series: [
          { name: 'Attack', color: '--series-1', points: cal.attack },
          { name: 'Decay', color: '--series-2', points: cal.decay },
          { name: 'Release', color: '--series-3', points: cal.release },
        ],
      });
    }
    $('step-connect').classList.toggle('is-ready', !!ready && !!cal);
  }

  // -------------------------------------------------------------- reference
  function renderReference() {
    if (!state) return;
    const ref = state.reference;
    $('ref-loaded').hidden = !ref;
    $('drop').classList.toggle('is-small', !!ref);
    for (const b of document.querySelectorAll('#ref-kind button')) {
      b.setAttribute('aria-checked', String(b.dataset.kind === ui.kind));
    }
    $('sep-field').hidden = ui.kind !== 'mix';
    if (!ref) {
      $('ref-status').textContent = 'No recording loaded';
      $('step-reference').classList.remove('is-ready');
      return;
    }
    $('ref-name').textContent = ref.filename;
    $('ref-meta').textContent = `${fmtTime(ref.duration)} · ${ref.sr} Hz · ${ref.channels === 1 ? 'mono' : ref.channels + ' channels'}`;
    const an = ref.analysis;
    const sel = ui.region;
    $('btn-analyse').textContent = sel ? 'Analyse the selection' : (an ? 'Analyse again' : 'Analyse');
    let preNote = '';
    if (ui.kind === 'mix' && devices && !devices.demucs) {
      preNote = 'Demucs is not installed, so the bass will be isolated with the built-in method, which only hears the low end of a mix. For complete songs install Demucs (pip install demucs) or choose a passage where the bass plays alone.';
    }
    const note = (an && an.note) || preNote;
    $('sep-note').hidden = !note;
    $('sep-note').textContent = note;
    $('analysis').hidden = !an;
    if (an) {
      const stats = $('analysis-stats');
      stats.replaceChildren();
      const add = (label, value) => stats.append(el('div', {}, el('strong', { text: value }), el('small', { text: label })));
      add('notes in the excerpt', String(an.phrase.length));
      add('excerpt', `${fmtTime(an.excerpt[0])} – ${fmtTime(an.excerpt[1])}`);
      add('length', fmtTime(an.excerpt[1] - an.excerpt[0]));
      add('tuning of the recording', `${an.tuning_cents >= 0 ? '+' : ''}${an.tuning_cents} cents`);
      add('bass', an.mode === 'stem' ? 'as recorded' : (an.method === 'demucs' ? 'isolated with Demucs' : 'isolated, low end only'));
      const audio = $('excerpt-audio');
      const src = `/media/reference/excerpt.wav?v=${an.version}`;
      if (audio.getAttribute('src') !== src) audio.setAttribute('src', src);
      const off = Math.abs(an.tuning_cents) >= 4;
      $('keep-tuning').parentElement.hidden = !off;
      $('keep-tuning-label').textContent = `Tune the patch to the recording (it is ${Math.abs(an.tuning_cents)} cents ${an.tuning_cents > 0 ? 'sharp' : 'flat'}); leave off for standard tuning`;
      $('ref-status').textContent = `${an.phrase.length} notes found`;
      drawRoll(an);
    } else {
      $('ref-status').textContent = 'Not analysed yet';
    }
    $('step-reference').classList.toggle('is-ready', !!an);
    drawWave();
  }

  function setupCanvas(canvas, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, Math.floor(canvas.clientWidth));
    const h = cssHeight;
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function drawWave() {
    const ref = state && state.reference;
    const canvas = $('wave');
    if (!ref || canvas.clientWidth === 0) return;
    const { ctx, w, h } = setupCanvas(canvas, 130);
    const an = ref.analysis;
    const x = (t) => (t / ref.duration) * w;
    const mid = h / 2 - 6;
    // analysed range and excerpt
    if (an) {
      ctx.fillStyle = css('--surface-2');
      ctx.fillRect(x(an.range[0]), 0, x(an.range[1]) - x(an.range[0]), h);
      ctx.fillStyle = css('--accent-soft');
      ctx.fillRect(x(an.excerpt[0]), 0, Math.max(2, x(an.excerpt[1]) - x(an.excerpt[0])), h);
    }
    // waveform
    ctx.fillStyle = css('--ink-2');
    const peaks = ref.peaks;
    const step = w / peaks.length;
    for (let i = 0; i < peaks.length; i++) {
      const lo = peaks[i][0] / 100, hi = peaks[i][1] / 100;
      const y0 = mid - hi * (mid - 4), y1 = mid - lo * (mid - 4);
      ctx.fillRect(i * step, y0, Math.max(1, step), Math.max(1, y1 - y0));
    }
    // detected notes as ticks along the bottom
    if (an) {
      ctx.fillStyle = css('--series-1');
      for (const n of an.notes) {
        ctx.fillRect(x(n[0]), h - 9, Math.max(1.5, x(n[1]) - x(n[0]) - 1), 4);
      }
      ctx.strokeStyle = css('--accent');
      ctx.lineWidth = 2;
      ctx.strokeRect(x(an.excerpt[0]) + 1, 1, Math.max(2, x(an.excerpt[1]) - x(an.excerpt[0])) - 2, h - 2);
    }
    // selection
    const sel = ui.drag || ui.region;
    if (sel) {
      const a = x(Math.min(sel[0], sel[1])), b = x(Math.max(sel[0], sel[1]));
      ctx.fillStyle = 'rgba(128,128,128,0.22)';
      ctx.fillRect(a, 0, b - a, h);
      ctx.fillStyle = css('--ink');
      ctx.fillRect(a, 0, 1.5, h);
      ctx.fillRect(b - 1.5, 0, 1.5, h);
      ctx.font = '12px system-ui, sans-serif';
      const label = `${fmtTime(Math.min(sel[0], sel[1]))} – ${fmtTime(Math.max(sel[0], sel[1]))}`;
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, clamp(a + 4, 2, w - tw - 2), 13);
    }
    $('wave-hint').textContent = ui.region
      ? `Selected ${fmtTime(ui.region[1] - ui.region[0])}. ${ui.region[1] - ui.region[0] <= 8 ? 'This part is used exactly as selected.' : 'The app will pick a suitable passage inside it.'} Double-click the waveform to clear the selection.`
      : 'Drag over the waveform to choose a part. A selection up to 8 seconds is used exactly as selected; from a longer one (or the whole file) the app picks a suitable passage itself.';
  }

  function bindWave() {
    const canvas = $('wave');
    const timeAt = (ev) => {
      const r = canvas.getBoundingClientRect();
      return clamp((ev.clientX - r.left) / r.width, 0, 1) * state.reference.duration;
    };
    canvas.addEventListener('pointerdown', (ev) => {
      if (!state.reference) return;
      canvas.setPointerCapture(ev.pointerId);
      const t = timeAt(ev);
      ui.drag = [t, t];
      drawWave();
    });
    canvas.addEventListener('pointermove', (ev) => {
      if (!ui.drag) return;
      ui.drag[1] = timeAt(ev);
      drawWave();
    });
    const finish = () => {
      if (!ui.drag) return;
      const a = Math.min(ui.drag[0], ui.drag[1]), b = Math.max(ui.drag[0], ui.drag[1]);
      ui.drag = null;
      if (b - a >= 0.5) ui.region = [a, b];
      renderReference();
    };
    canvas.addEventListener('pointerup', finish);
    canvas.addEventListener('pointercancel', finish);
    canvas.addEventListener('dblclick', () => { ui.region = null; renderReference(); });
  }

  let rollNotes = [];
  function drawRoll(an) {
    const canvas = $('roll');
    if (canvas.clientWidth === 0) return;
    const { ctx, w, h } = setupCanvas(canvas, 150);
    const notes = an.phrase;
    const dur = an.excerpt[1] - an.excerpt[0];
    const pitches = notes.map((n) => n[2]);
    const lo = Math.min(...pitches) - 1, hi = Math.max(...pitches) + 1;
    const left = 34, bottom = 18;
    const rowH = (h - bottom - 4) / (hi - lo + 1);
    const x = (t) => left + (t / dur) * (w - left - 6);
    const y = (p) => 4 + (hi - p) * rowH;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    for (let p = lo; p <= hi; p++) {
      ctx.fillStyle = css('--line');
      ctx.fillRect(left, Math.round(y(p) + rowH) - 0.5, w - left - 6, 1);
      if (rowH >= 11 || (p - lo) % 2 === 0) {
        ctx.fillStyle = css('--muted');
        ctx.fillText(noteName(p), 2, y(p) + rowH / 2);
      }
    }
    ctx.textBaseline = 'alphabetic';
    const ticks = dur <= 4 ? 0.5 : 1;
    for (let t = 0; t <= dur + 1e-6; t += ticks) {
      ctx.fillStyle = css('--muted');
      ctx.fillText(`${t.toFixed(ticks < 1 ? 1 : 0)} s`, x(t) - 8, h - 4);
    }
    rollNotes = [];
    ctx.fillStyle = css('--series-1');
    for (const n of notes) {
      const bx = x(n[0]), bw = Math.max(3, x(n[1]) - x(n[0]) - 2), by = y(n[2]) + 1.5, bh = Math.max(4, rowH - 3);
      const r = Math.min(4, bh / 2, bw / 2);
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, r);
      ctx.fill();
      rollNotes.push({ x: bx, y: by, w: bw, h: bh, n });
    }
  }
  function bindRoll() {
    const canvas = $('roll');
    canvas.addEventListener('pointermove', (ev) => {
      const r = canvas.getBoundingClientRect();
      const px = ev.clientX - r.left, py = ev.clientY - r.top;
      const hit = rollNotes.find((b) => px >= b.x - 3 && px <= b.x + b.w + 3 && py >= b.y - 4 && py <= b.y + b.h + 4);
      if (!hit) { hideTip(); return; }
      showTip(ev.clientX, ev.clientY, hit.n[3], [
        { value: `${hit.n[0].toFixed(2)} s`, label: 'start' },
        { value: `${Math.round((hit.n[1] - hit.n[0]) * 1000)} ms`, label: 'length' },
      ]);
    });
    canvas.addEventListener('pointerleave', hideTip);
  }

  // ---------------------------------------------------------------- tooltip
  function showTip(cx, cy, title, rows) {
    const tip = $('tooltip');
    tip.replaceChildren();
    if (title) tip.append(el('div', { class: 'tt-title', text: title }));
    for (const row of rows) {
      const line = el('div', { class: 'tt-row' });
      if (row.color) { const key = el('i'); key.style.color = `var(${row.color})`; line.append(key); }
      line.append(el('b', { text: row.value }), el('span', { text: row.label }));
      tip.append(line);
    }
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = `${clamp(cx + 14, 6, window.innerWidth - r.width - 6)}px`;
    tip.style.top = `${clamp(cy - r.height - 10, 6, window.innerHeight - r.height - 6)}px`;
  }
  function hideTip() { $('tooltip').hidden = true; }

  // ----------------------------------------------------------------- charts
  function niceNumber(v, unit) {
    if (unit === 'Hz') return v >= 1000 ? `${+(v / 1000).toPrecision(3)} kHz` : `${+v.toPrecision(3)} Hz`;
    if (unit === 's') return v >= 1 ? `${+v.toPrecision(3)} s` : `${+(v * 1000).toPrecision(3)} ms`;
    return String(+v.toPrecision(3));
  }
  function logTicks(lo, hi) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      const v = 10 ** e;
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
    return out;
  }
  function linTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const raw = span / n;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toPrecision(10));
    return out;
  }

  /** Line chart (SVG) with legend, crosshair tooltip and a table view. */
  function lineChart(fig, cfg) {
    // the viewBox is sized like the rendered chart, so text keeps its natural size
    const W = cfg.wide ? 1040 : 520, H = cfg.height || 210;
    const m = { l: 52, r: cfg.endLabels === false ? 14 : 58, t: 10, b: 34 };
    const all = cfg.series.flatMap((s) => s.points);
    fig.replaceChildren();
    const cap = el('figcaption', {}, el('span', { text: cfg.title }));
    if (cfg.subtitle) cap.append(el('small', { text: cfg.subtitle }));
    fig.append(cap);
    if (all.length === 0) return;
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]).filter((v) => !cfg.yLog || v > 0);
    const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (cfg.yLog) { y0 = 10 ** Math.floor(Math.log10(y0)); y1 = 10 ** Math.ceil(Math.log10(y1)); }
    else { y0 = cfg.yMin !== undefined ? cfg.yMin : Math.min(0, y0); y1 = y1 + (y1 - y0) * 0.08 || 1; }
    const fx = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const fy = (v) => (cfg.yLog
      ? H - m.b - ((Math.log10(Math.max(v, y0)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (H - m.t - m.b)
      : H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b));

    if (cfg.series.length > 1) {
      const legend = el('div', { class: 'legend' });
      for (const s of cfg.series) {
        const key = el('i', { class: s.kind === 'dots' ? 'dot' : '' });
        key.style.color = `var(${s.color})`;
        legend.append(el('span', {}, key, document.createTextNode(s.name)));
      }
      fig.append(legend);
    }
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${cfg.title}. ${cfg.series.map((s) => s.name).join(', ')}.` });
    const grid = svg('g', { class: 'grid' });
    const axis = svg('g', { class: 'axis' });
    const yt = cfg.yLog ? logTicks(y0, y1) : linTicks(y0, y1, 4);
    for (const v of yt) {
      grid.append(svg('line', { x1: m.l, x2: W - m.r, y1: fy(v), y2: fy(v) }));
      axis.append(svg('text', { x: m.l - 8, y: fy(v) + 4, 'text-anchor': 'end' }, niceNumber(v, cfg.yUnit)));
    }
    for (const v of linTicks(x0, x1, cfg.wide ? 10 : 5)) {
      axis.append(svg('text', { x: fx(v), y: H - m.b + 16, 'text-anchor': 'middle' }, String(v)));
    }
    root.append(grid, axis);
    root.append(svg('line', { class: 'baseline', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    if (cfg.xLabel) root.append(svg('text', { class: 'axis-title', x: (m.l + W - m.r) / 2, y: H - 3, 'text-anchor': 'middle' }, cfg.xLabel));
    const ends = [];
    for (const s of cfg.series) {
      const pts = s.points.filter((p) => !cfg.yLog || p[1] > 0);
      if (!pts.length) continue;
      if (s.kind === 'dots') {
        const g = svg('g', {});
        for (const p of pts) g.append(svg('circle', { cx: fx(p[0]), cy: fy(p[1]), r: 2.5, 'fill-var': s.color }));
        root.append(g);
      } else {
        const d = pts.map((p, i) => `${i ? 'L' : 'M'}${fx(p[0]).toFixed(1)},${fy(p[1]).toFixed(1)}`).join('');
        root.append(svg('path', { d, fill: 'none', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-var': s.color }));
        const last = pts[pts.length - 1];
        root.append(svg('circle', { cx: fx(last[0]), cy: fy(last[1]), r: 4, 'stroke-width': 2, 'fill-var': s.color, 'stroke-var': '--surface' }));
        ends.push({ y: fy(last[1]), text: cfg.endFormat ? cfg.endFormat(last[1]) : niceNumber(last[1], cfg.yUnit) });
      }
    }
    if (cfg.endLabels !== false) {
      // end labels only where they do not collide
      ends.sort((a, b) => a.y - b.y);
      let prev = -99;
      for (const e of ends) {
        if (e.y - prev >= 13) {
          root.append(svg('text', { class: 'end-label', x: W - m.r + 8, y: e.y + 4 }, e.text));
          prev = e.y;
        }
      }
    }
    // hover layer: crosshair snaps to the nearest x, one tooltip lists every series
    const cross = svg('line', { class: 'crosshair', y1: m.t, y2: H - m.b, visibility: 'hidden' });
    const hit = svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' });
    root.append(cross, hit);
    const uniqueX = [...new Set(xs)].sort((a, b) => a - b);
    hit.addEventListener('pointermove', (ev) => {
      const r = root.getBoundingClientRect();
      const vx = x0 + ((ev.clientX - r.left) / r.width * W - m.l) / (W - m.l - m.r) * (x1 - x0);
      const nearest = uniqueX.reduce((a, b) => (Math.abs(b - vx) < Math.abs(a - vx) ? b : a), uniqueX[0]);
      cross.setAttribute('x1', fx(nearest));
      cross.setAttribute('x2', fx(nearest));
      cross.setAttribute('visibility', 'visible');
      const rows = [];
      for (const s of cfg.series) {
        const p = s.points.reduce((a, b) => (Math.abs(b[0] - nearest) < Math.abs(a[0] - nearest) ? b : a), s.points[0]);
        if (p && (s.kind !== 'dots' || p[0] === nearest)) {
          rows.push({ color: s.color, value: cfg.tipFormat ? cfg.tipFormat(p[1]) : niceNumber(p[1], cfg.yUnit), label: s.name });
        }
      }
      showTip(ev.clientX, ev.clientY, `${cfg.xName || cfg.xLabel || 'x'}: ${nearest}`, rows);
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
    fig.append(root);

    // table view: every value without hovering
    const toggle = el('button', { type: 'button', class: 'table-toggle', text: 'Show as table' });
    cap.append(toggle);
    const holder = el('div', { hidden: true });
    fig.append(holder);
    toggle.addEventListener('click', () => {
      holder.hidden = !holder.hidden;
      toggle.textContent = holder.hidden ? 'Show as table' : 'Hide table';
      if (holder.hidden || holder.childElementCount) return;
      const table = el('table', { class: 'table' });
      table.append(el('thead', {}, el('tr', {}, el('th', { text: cfg.xName || cfg.xLabel || 'x' }),
        ...cfg.series.map((s) => el('th', { class: 'num', text: s.name })))));
      const body = el('tbody');
      const stepN = Math.max(1, Math.ceil(uniqueX.length / 24));
      uniqueX.forEach((vx, i) => {
        if (i % stepN && i !== uniqueX.length - 1) return;
        const tr = el('tr', {}, el('td', { text: String(vx) }));
        for (const s of cfg.series) {
          const p = s.points.find((q) => q[0] === vx);
          tr.append(el('td', { class: 'num', text: p ? (cfg.tipFormat ? cfg.tipFormat(p[1]) : niceNumber(p[1], cfg.yUnit)) : '' }));
        }
        body.append(tr);
      });
      table.append(body);
      holder.append(table);
    });
  }

  // ------------------------------------------------------------------ match
  function estimate(key) {
    const eff = state.efforts[key];
    const an = state.reference && state.reference.analysis;
    const dur = an ? an.excerpt[1] - an.excerpt[0] : 4;
    const model = { quick: 1, normal: 1.5, thorough: 2.5 }[key] || 1.5;
    const perPlay = state.mode === 'demo' ? 0.7 : dur + 0.6;
    return Math.max(1, Math.round(model + (eff.plays * perPlay) / 60 + (state.mode === 'demo' ? 0 : 0.3)));
  }

  function renderMatch() {
    if (!state) return;
    const seg = $('effort');
    seg.replaceChildren();
    for (const key of Object.keys(state.efforts)) {
      const b = el('button', { type: 'button', role: 'radio', 'aria-checked': String(state.settings.effort === key) },
        document.createTextNode(state.efforts[key].label), el('small', { text: `about ${estimate(key)} min` }));
      b.addEventListener('click', () => act(async () => { await api('/api/settings', { effort: key }); await refreshState(); }));
      seg.append(b);
    }
    const an = state.reference && state.reference.analysis;
    const needCal = state.mode !== 'demo' && !state.calibration;
    const matching = running() && job.kind === 'match';
    $('btn-match').disabled = !an || needCal || running();
    $('btn-match').hidden = matching;
    $('btn-stop').hidden = !running();
    $('match-hint').textContent = !an ? 'Load and analyse a reference first.'
      : (needCal ? 'Calibrate the instrument first (step 1).'
        : 'The synth plays the excerpt over and over while the app adjusts the patch. You will hear it; that is how it listens.');
    $('step-match').classList.toggle('is-ready', !!an && !needCal);
    renderJob();
  }

  function renderJob() {
    if (!state) return;
    const chip = $('job-chip');
    chip.hidden = !running();
    if (running()) chip.textContent = `${job.title}: ${job.phase || 'working'}`;
    $('btn-stop').hidden = !running();
    for (const id of ['btn-signal', 'btn-calibrate', 'btn-analyse', 'btn-send', 'btn-play', 'btn-refresh']) $(id).disabled = running();
    if ($('btn-match')) $('btn-match').disabled = running() || !(state.reference && state.reference.analysis) || (state.mode !== 'demo' && !state.calibration);
    $('btn-match').hidden = running() && job.kind === 'match';

    if (!job) return;
    if (job.kind === 'signal' && job.data.signal) {
      const s = job.data.signal;
      $('signal-readout').textContent = s.message;
    } else if (job.kind === 'signal') {
      $('signal-readout').textContent = job.status === 'error' ? job.error : job.message;
    }
    if (job.kind === 'calibration') {
      renderCalSteps();
      $('cal-status').textContent = job.status === 'running' ? `${job.message} … ${Math.round(job.fraction * 100)} %`
        : (job.status === 'error' ? job.error : $('cal-status').textContent);
    }
    if (job.kind === 'analysis') {
      $('ref-status').textContent = job.status === 'running' ? job.message : (job.status === 'error' ? 'Analysis failed' : $('ref-status').textContent);
    }
    if (job.kind === 'send' && job.status === 'done' && job.data.audition && state.mode === 'demo') {
      const a = $('audition-audio');
      const src = `/media/reference/audition.wav?v=${job.data.audition}`;
      if (a.getAttribute('src') !== src) { a.setAttribute('src', src); a.hidden = false; a.play().catch(() => {}); }
    }
    if (job.kind === 'match') {
      $('progress').hidden = false;
      const stage = job.status === 'done' ? 'done' : (job.data.stage || 'prepare');
      const order = ['prepare', 'model', 'refine', 'done'];
      for (const li of document.querySelectorAll('#stages li')) {
        const i = order.indexOf(li.dataset.stage), cur = order.indexOf(stage);
        li.classList.toggle('is-done', i < cur || job.status === 'done');
        li.classList.toggle('is-active', i === cur && job.status === 'running');
      }
      const pct = Math.round(job.fraction * 100);
      $('bar').firstElementChild.style.width = `${pct}%`;
      $('bar').setAttribute('aria-valuenow', String(pct));
      let msg = job.message;
      if (job.status === 'error') msg = job.error;
      if (job.status === 'cancelled') msg = 'Stopped. The best patch found so far (if any) is not kept; start again when ready.';
      if (job.status === 'running' && job.data.best !== undefined) msg = `${msg}  ·  best distance ${Number(job.data.best).toFixed(3)}`;
      $('progress-msg').textContent = msg;
      $('match-status').textContent = job.status === 'running' ? `${pct} % · ${fmtTime(job.elapsed)}`
        : (job.status === 'done' ? 'Finished' : (job.status === 'error' ? 'Failed' : 'Stopped'));
      $('log').textContent = job.log.join('\n');
      const hist = job.data.history || [];
      $('chart-history').hidden = hist.length < 2;
      if (hist.length >= 2) {
        lineChart($('chart-history'), {
          title: 'Distance to the reference', subtitle: 'lower is closer; every dot is one play of the excerpt on the synth',
          xLabel: 'Measurement', xName: 'Measurement', yMin: 0, height: 230, wide: window.innerWidth > 760,
          tipFormat: (v) => v.toFixed(3), endFormat: (v) => v.toFixed(3),
          series: [
            { name: 'Each measurement', color: '--muted', kind: 'dots', points: hist.map((h) => [h[0], h[1]]) },
            { name: 'Best so far', color: '--series-1', points: hist.map((h) => [h[0], h[2]]) },
          ],
        });
      }
    }
  }
  function renderCalSteps() {
    if (!job || job.kind !== 'calibration' || !job.data.steps) return;
    const list = $('cal-steps');
    list.replaceChildren();
    for (const st of job.data.steps) {
      list.append(el('li', { class: st.ok ? 'ok' : 'bad' },
        el('span', { class: 'mark', text: st.ok ? '✓' : '!' }),
        el('span', { class: 'name', text: st.name }),
        el('span', { class: 'msg', text: st.message })));
    }
  }

  // ----------------------------------------------------------------- result
  function rampLut() {
    const stops = ['--ramp-0', '--ramp-1', '--ramp-2', '--ramp-3', '--ramp-4'].map((n) => {
      const hex = css(n).replace('#', '');
      return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
    });
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const p = (i / 255) * (stops.length - 1);
      const a = Math.min(stops.length - 2, Math.floor(p)), f = p - a;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(stops[a][c] + (stops[a + 1][c] - stops[a][c]) * f);
    }
    return lut;
  }
  async function drawSpec(canvas, url) {
    const img = new Image();
    img.src = url;
    try { await img.decode(); } catch (err) { return; }
    const off = document.createElement('canvas');
    off.width = img.naturalWidth; off.height = img.naturalHeight;
    const octx = off.getContext('2d');
    octx.drawImage(img, 0, 0);
    const data = octx.getImageData(0, 0, off.width, off.height);
    const lut = rampLut();
    for (let i = 0; i < data.data.length; i += 4) {
      const g = data.data[i];
      data.data[i] = lut[g * 3]; data.data[i + 1] = lut[g * 3 + 1]; data.data[i + 2] = lut[g * 3 + 2]; data.data[i + 3] = 255;
    }
    canvas.width = off.width; canvas.height = off.height;
    canvas.getContext('2d').putImageData(data, 0, 0);
  }

  function renderSheet(holder, rows, onlyChanged) {
    holder.replaceChildren();
    const groups = new Map();
    for (const r of rows) {
      if (onlyChanged && !r.changed) continue;
      if (!groups.has(r.section_title)) groups.set(r.section_title, []);
      groups.get(r.section_title).push(r);
    }
    if (groups.size === 0) holder.append(el('p', { class: 'empty', text: 'Identical to an initialised patch.' }));
    for (const [title, list] of groups) {
      const sec = el('section', {}, el('h4', { text: title }));
      for (const r of list) {
        const val = el('span', { class: 'p-val', text: r.display });
        if (!r.discrete && String(r.raw) !== r.display) val.append(el('small', { text: `raw ${r.raw}` }));
        const row = el('div', { class: `p${r.changed ? '' : ' same'}`, title: `${r.address}${r.control === 'key' ? ' · on the synth: Function + key' : ''}` },
          el('span', { class: 'p-name', text: r.label }), val);
        if (!r.discrete) {
          const meter = el('div', { class: 'meter' });
          const pos = (r.raw - r.lo) / (r.hi - r.lo || 1);
          const fill = el('b');
          if (r.center !== null && r.center !== undefined) {
            const c = (r.center - r.lo) / (r.hi - r.lo || 1);
            fill.style.left = `${Math.min(pos, c) * 100}%`;
            fill.style.width = `${Math.max(Math.abs(pos - c) * 100, 0.8)}%`;
            const tick = el('u');
            tick.style.left = `${c * 100}%`;
            meter.append(tick);
          } else {
            fill.style.left = '0';
            fill.style.width = `${Math.max(pos * 100, 0.8)}%`;
          }
          meter.append(fill);
          row.append(meter);
        }
        sec.append(row);
      }
      holder.append(sec);
    }
  }

  const QUALITY = {
    'practically identical': 'Practically identical',
    excellent: 'Excellent match',
    'very close': 'Very close',
    close: 'Close',
    rough: 'Rough match',
    poor: 'Poor match',
  };

  function renderResult() {
    if (!state) return;
    const r = state.result;
    $('result').hidden = !r;
    $('result-empty').hidden = !!r;
    $('step-result').classList.toggle('is-ready', !!r);
    if (!r) { $('result-status').textContent = ''; return; }
    $('hero-value').textContent = r.loss.toFixed(3);
    $('hero-quality').textContent = QUALITY[r.quality] || r.quality;
    $('hero-detail').textContent = `${r.structure}. ${r.plays} measurements in ${fmtTime(r.seconds)}; started at ${r.start_loss.toFixed(3)}. ` +
      `Zero would be identical; two takes of the very same patch differ by about ${Math.max(r.noise * 2, 0.02).toFixed(2)} because the oscillators run free.`;
    $('result-status').textContent = `${r.date} · ${r.reference_file}`;
    const warns = [];
    if (r.clipped) warns.push('The input clipped during the measurements. Lower the input gain of the audio interface and match again for a cleaner result.');
    if (r.keep_tuning && r.fine_offset) warns.push(`Both oscillators are tuned ${r.fine_offset > 0 ? '+' : ''}${r.fine_offset} cents to follow the recording.`);
    $('result-warn').hidden = warns.length === 0;
    $('result-warn').textContent = warns.join(' ');
    const base = `/media/session/${r.id}`;
    for (const [id, file] of [['audio-ref', 'reference.wav'], ['audio-syn', 'synth.wav']]) {
      const src = `${base}/${file}`;
      if ($(id).getAttribute('src') !== src) $(id).setAttribute('src', src);
    }
    const specKey = `${r.id}|${css('--ramp-4')}`;
    if (ui.specCache.key !== specKey) {
      ui.specCache.key = specKey;
      drawSpec($('spec-ref'), `${base}/reference.png`);
      drawSpec($('spec-syn'), `${base}/synth.png`);
    }
    $('dl-txt').href = `${base}/patch.txt`;
    $('dl-json').href = `${base}/result.json`;
    $('dl-syx').hidden = !r.has_syx;
    $('dl-syx').href = `${base}/patch.syx`;
    $('sheet-structure').textContent = r.structure;
    renderSheet($('sheet'), r.rows, $('only-changed').checked);
    $('btn-play').textContent = state.mode === 'demo' ? 'Play the phrase' : 'Play the phrase on the synth';
    if (!$('save-name').value && document.activeElement !== $('save-name')) {
      $('save-name').placeholder = (r.reference_file || 'Patch').replace(/\.[^.]+$/, '').slice(0, 40);
    }
  }

  function renderLibrary() {
    const items = (state && state.library) || [];
    $('library-box').hidden = items.length === 0;
    const body = $('library').tBodies[0];
    body.replaceChildren();
    for (const it of items) {
      const actions = el('div', { class: 'row-actions' },
        el('button', { type: 'button', class: 'btn', text: 'Send', onclick: () => act(() => api('/api/patch/send', { library: it.id })) }),
        el('button', { type: 'button', class: 'btn', text: 'View', onclick: () => act(async () => {
          const data = await api(`/api/library/${it.id}`);
          const row = body.querySelector(`[data-detail="${it.id}"]`);
          if (row) { row.remove(); return; }
          const holder = el('div', { class: 'sheet' });
          renderSheet(holder, data.rows, true);
          const tr = el('tr', { 'data-detail': it.id }, el('td', { colspan: '5' }, holder));
          line.after(tr);
        }) }),
        el('button', { type: 'button', class: 'btn', text: 'Delete', onclick: () => act(async () => {
          if (!window.confirm(`Delete "${it.name}" from the library?`)) return;
          await api('/api/library/delete', { id: it.id });
          await refreshState();
        }) }));
      const line = el('tr', {},
        el('td', { text: it.name }), el('td', { text: it.structure }),
        el('td', { class: 'num', text: it.loss !== null && it.loss !== undefined ? Number(it.loss).toFixed(3) : '' }),
        el('td', { text: it.date }), el('td', {}, actions));
      body.append(line);
    }
  }

  function render() {
    renderTop();
    renderConnect();
    renderReference();
    renderMatch();
    renderResult();
    renderLibrary();
  }

  // ------------------------------------------------------------------ binds
  async function upload(file) {
    if (!file) return;
    $('ref-status').textContent = `Loading ${file.name} …`;
    ui.region = null;
    await api(`/api/reference?name=${encodeURIComponent(file.name)}`, undefined, {
      body: file, headers: { 'X-BS2-Upload': '1', 'Content-Type': 'application/octet-stream' },
    });
    await refreshState();
  }

  function bind() {
    for (const b of document.querySelectorAll('#step-connect .seg button')) {
      b.addEventListener('click', () => act(async () => {
        await api('/api/settings', { mode: b.dataset.mode });
        await refreshState();
        if (b.dataset.mode === 'hardware') await refreshDevices();
      }));
    }
    const setting = (id, key, numeric) => $(id).addEventListener('change', () => act(async () => {
      await api('/api/settings', { [key]: numeric ? Number($(id).value) : $(id).value });
      await refreshState();
    }));
    setting('midi-out', 'midi_out');
    setting('midi-in', 'midi_in');
    setting('midi-channel', 'midi_channel', true);
    setting('audio-device', 'audio_device');
    setting('input-channel', 'input_channel', true);
    $('fw-features').addEventListener('change', () => act(async () => {
      await api('/api/settings', { firmware_features: $('fw-features').checked });
      await refreshState();
    }));
    $('btn-refresh').addEventListener('click', () => act(refreshDevices));
    $('btn-signal').addEventListener('click', () => act(async () => { $('signal-readout').textContent = 'Playing a test note …'; await api('/api/signal', {}); }));
    $('btn-calibrate').addEventListener('click', () => act(() => api('/api/calibrate', {})));

    const drop = $('drop');
    drop.addEventListener('click', () => $('file').click());
    drop.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); $('file').click(); } });
    $('file').addEventListener('change', () => act(() => upload($('file').files[0])));
    for (const name of ['dragenter', 'dragover']) drop.addEventListener(name, (ev) => { ev.preventDefault(); drop.classList.add('is-over'); });
    for (const name of ['dragleave', 'drop']) drop.addEventListener(name, (ev) => { ev.preventDefault(); drop.classList.remove('is-over'); });
    drop.addEventListener('drop', (ev) => act(() => upload(ev.dataTransfer.files[0])));
    window.addEventListener('dragover', (ev) => ev.preventDefault());
    window.addEventListener('drop', (ev) => ev.preventDefault());

    for (const b of document.querySelectorAll('#ref-kind button')) {
      b.addEventListener('click', () => { ui.kind = b.dataset.kind; renderReference(); });
    }
    $('btn-analyse').addEventListener('click', () => act(() => api('/api/analyse', {
      mode: ui.kind, method: $('sep-method').value, sensitivity: Number($('sensitivity').value), region: ui.region,
    })));
    bindWave();
    bindRoll();

    $('btn-match').addEventListener('click', () => act(async () => {
      $('chart-history').hidden = true;
      await api('/api/match', { effort: state.settings.effort, keep_tuning: $('keep-tuning').checked && !$('keep-tuning').parentElement.hidden });
    }));
    $('btn-stop').addEventListener('click', () => act(() => api('/api/stop', {})));

    $('btn-send').addEventListener('click', () => act(() => api('/api/patch/send', {})));
    $('btn-play').addEventListener('click', () => act(() => api('/api/patch/send', { play: true })));
    $('only-changed').addEventListener('change', renderResult);
    $('btn-save').addEventListener('click', () => act(async () => {
      const name = $('save-name').value.trim() || $('save-name').placeholder || 'Patch';
      await api('/api/library/save', { name });
      $('save-name').value = '';
      await refreshState();
    }));

    let resizeTimer = null;
    window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (state) { renderReference(); } }, 150); });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { ui.specCache = {}; if (state) render(); });
  }

  async function init() {
    bind();
    try {
      await refreshState();
      if (state.reference && state.reference.analysis) ui.kind = state.reference.analysis.mode;
      await refreshDevices();
      connectEvents();
    } catch (err) {
      banner(err.message);
    }
  }
  init();
})();
