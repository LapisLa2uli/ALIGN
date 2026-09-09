let wavesurfer;
let regionsPlugin;
let osmdInstance = null;
let currentSample = null;
let sampleData = null;
let compareData = null;
let scoreLoadId = 0;
let selectedRegion = null;
let loopSelection = false;
let scrubbing = false;
let showGold = true;
let showPred = true;
let zoomFactor = 1;
let fitPxPerSec = 1;
let userZoomed = false;
let pendingLabels = [];
let scoreEvents = [];
let playheadRaf = null;

const TYPE_COLORS = {
  wrong_note: "rgba(255,60,60,0.45)",
  wrong_pitch: "rgba(255,60,60,0.45)",
  missed_note: "rgba(255,140,0,0.4)",
  extra_note: "rgba(255,200,0,0.4)",
  intonation_error: "rgba(180,80,255,0.4)",
  rhythm_error: "rgba(80,180,255,0.4)",
  repetition: "rgba(80,255,160,0.4)",
  stylistic_choice: "rgba(160,160,160,0.35)",
  bad_start: "rgba(210,110,40,0.45)",
  bad_timbre: "rgba(0,170,150,0.4)",
  squeak: "rgba(255,50,170,0.45)",
  sliding: "rgba(255,130,70,0.45)",
  click: "rgba(255,220,80,0.45)",
  misc: "rgba(140,140,160,0.45)",
};

const TYPE_LABELS = {
  wrong_note: "Wrong note",
  wrong_pitch: "Wrong note",
  intonation_error: "Intonation",
  missed_note: "Missed note",
  extra_note: "Extra note",
  rhythm_error: "Rhythm",
  repetition: "Repetition",
  stylistic_choice: "Stylistic",
  bad_start: "Bad start",
  bad_timbre: "Bad timbre",
  squeak: "Squeak",
  sliding: "Sliding",
  click: "Click",
  misc: "Misc",
};

function typeName(type) {
  return TYPE_LABELS[type] || type || "unlabeled";
}

function formatTime(sec) {
  if (!Number.isFinite(sec)) return "00:00.000";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
}

function fmtF1(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(3);
}

function getScrollWidth() {
  return document.getElementById("waveformScroll")?.clientWidth || 800;
}

function getPxPerSec() {
  return Math.max(fitPxPerSec, fitPxPerSec * zoomFactor);
}

function contentWidth() {
  const duration = wavesurfer?.getDuration() || 0;
  return Math.max(getScrollWidth(), duration * getPxPerSec());
}

function fitWaveform() {
  const duration = wavesurfer?.getDuration() || 0;
  if (!duration) return;
  fitPxPerSec = getScrollWidth() / duration;
  const px = getPxPerSec();
  const width = contentWidth();
  if (typeof wavesurfer.setOptions === "function") {
    wavesurfer.setOptions({ minPxPerSec: px, width });
  }
  const wrap = document.querySelector(".waveform-wrap");
  if (wrap) wrap.style.width = `${width}px`;
  const scrubber = document.getElementById("scrubber");
  if (scrubber) scrubber.style.width = `${width}px`;
  const melody = document.getElementById("melodyStrip");
  if (melody) melody.style.width = `${width}px`;
  layoutRegions();
  renderMelodyStrip();
  updatePlayhead();
}

function updatePlayhead() {
  if (playheadRaf) return;
  playheadRaf = requestAnimationFrame(() => {
    playheadRaf = null;
    const duration = wavesurfer?.getDuration();
    const playhead = document.getElementById("playhead");
    const progress = document.getElementById("scrubberProgress");
    if (!duration || !playhead) return;
    const t = wavesurfer.getCurrentTime();
    const x = Math.min(t * getPxPerSec(), Math.max(0, contentWidth() - 1));
    playhead.style.left = `${x}px`;
    if (progress) progress.style.width = `${x}px`;
    const clock = document.getElementById("timeDisplay");
    if (clock) clock.textContent = formatTime(t);
  });
}

function layoutRegions() {
  if (!regionsPlugin) return;
  const both = showGold && showPred;
  regionsPlugin.getRegions().forEach((region) => {
    const el = region.element;
    if (!el) return;
    const layer = region.data?.compare?.layer;
    const match = region.data?.compare?.match || "unmatched";
    const selected = region === selectedRegion;
    const visible =
      (layer === "gold" && showGold) || (layer === "pred" && showPred);
    el.style.display = visible ? "block" : "none";
    el.style.boxSizing = "border-box";
    el.style.overflow = "visible";
    if (both) {
      el.style.top = layer === "gold" ? "0" : "50%";
      el.style.height = "50%";
    } else {
      el.style.top = "0";
      el.style.height = "100%";
    }
    const color = TYPE_COLORS[region.data?.type] || "rgba(45,108,223,0.35)";
    el.style.backgroundColor = color;
    el.style.boxSizing = "border-box";
    el.style.overflow = "visible";
    if (selected) {
      el.style.border = "3px solid #fff";
      el.style.zIndex = "12";
    } else if (match === "full") {
      el.style.border = layer === "gold" ? "2px solid #9cf" : "2px solid #fc6";
      el.style.zIndex = "8";
    } else if (match === "type_mismatch") {
      el.style.border = "2px dashed #eee";
      el.style.opacity = "0.95";
      el.style.zIndex = "7";
    } else {
      el.style.border = "1px solid rgba(255,255,255,0.15)";
      el.style.opacity = "0.55";
      el.style.zIndex = "5";
    }
    const cap = el.querySelector("[part~='region-content']") || el.firstElementChild;
    if (cap) {
      const prefix = layer === "gold" ? "G" : "P";
      cap.textContent = `${prefix} ${typeName(region.data?.type)}`;
    }
  });
}

function addLabelRegion(label) {
  const start = Number(label.start_time);
  const end = Number(label.end_time);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  const color = TYPE_COLORS[label.type] || "rgba(45,108,223,0.35)";
  const layer = label.compare?.layer || "gold";
  const region = regionsPlugin.addRegion({
    start,
    end,
    color,
    drag: false,
    resize: false,
    content: `${layer === "gold" ? "G" : "P"} ${typeName(label.type)}`,
    data: label,
  });
  region.data = label;
  return region;
}

function allRegions() {
  return regionsPlugin ? regionsPlugin.getRegions() : [];
}

function findRegion(layer, index) {
  return allRegions().find(
    (r) => r.data?.compare?.layer === layer && r.data?.compare?.index === index,
  );
}

function selectRegion(region) {
  selectedRegion = region || null;
  layoutRegions();
  renderLists();
  renderSelection();
  renderMelodyStrip();
  if (region && wavesurfer) {
    const t = region.start;
    if (Number.isFinite(t)) wavesurfer.setTime(t);
  }
}

function matchTag(match) {
  if (match === "full") return "range+type";
  if (match === "type_mismatch") return "range, wrong type (0.5)";
  return "unmatched";
}

function renderSelection() {
  const info = document.getElementById("selectionInfo");
  const detail = document.getElementById("selectionDetail");
  if (!selectedRegion) {
    info.textContent = "Click a gold or model region, or a row in the lists below.";
    detail.innerHTML = "";
    return;
  }
  const d = selectedRegion.data || {};
  const c = d.compare || {};
  const layer = c.layer === "pred" ? "Model A" : "Gold";
  info.textContent =
    `${layer}: ${typeName(d.type)}  ${formatTime(selectedRegion.start)} – ${formatTime(selectedRegion.end)}`;
  const pitches = Array.isArray(d.pitches) ? d.pitches.join(" ") : "—";
  const partner = c.pair_index != null
    ? findRegion(c.layer === "gold" ? "pred" : "gold", c.pair_index)
    : null;
  const partnerType = partner ? typeName(partner.data?.type) : "—";
  detail.innerHTML = `
    <p><strong>${matchTag(c.match)}</strong> · credit ${c.credit ?? 0}</p>
    <p>Range score ${c.range_score ?? 0}</p>
    <p>Pitches: ${pitches}</p>
    <p>Paired with: ${partnerType}</p>
    ${d.comment ? `<p class="hint">${d.comment}</p>` : ""}
  `;
}

function renderLists() {
  const goldEl = document.getElementById("goldList");
  const predEl = document.getElementById("predList");
  const render = (labels, layer) => {
    if (!labels.length) return `<p class="hint">None</p>`;
    return labels.map((lab, i) => {
      const c = lab.compare || {};
      const active =
        selectedRegion
        && selectedRegion.data?.compare?.layer === layer
        && selectedRegion.data?.compare?.index === i;
      return `<button type="button" class="label-row match-${c.match || "unmatched"}${active ? " active" : ""}" data-layer="${layer}" data-index="${i}">
        <span class="swatch" style="background:${TYPE_COLORS[lab.type] || "#2d6cdf"}"></span>
        <span class="row-type">${typeName(lab.type)}</span>
        <span class="row-time">${formatTime(lab.start_time)}–${formatTime(lab.end_time)}</span>
        <span class="row-match">${matchTag(c.match)}</span>
      </button>`;
    }).join("");
  };
  goldEl.innerHTML = render(compareData?.gold || [], "gold");
  predEl.innerHTML = render(compareData?.pred || [], "pred");
  [...goldEl.querySelectorAll(".label-row"), ...predEl.querySelectorAll(".label-row")].forEach((btn) => {
    btn.onclick = () => {
      const region = findRegion(btn.dataset.layer, parseInt(btn.dataset.index, 10));
      if (region) selectRegion(region);
    };
  });
}

function eventStart(ev) {
  const v = ev.perf_start ?? ev.ref_start ?? ev.start;
  return Number(v) || 0;
}

function eventEnd(ev) {
  const v = ev.perf_end ?? ev.ref_end ?? ev.end;
  return Number(v) || eventStart(ev) + 0.1;
}

function renderMelodyStrip() {
  const container = document.getElementById("melodyStrip");
  if (!container) return;
  const width = contentWidth();
  container.style.width = `${width}px`;
  container.style.minWidth = `${width}px`;
  if (!scoreEvents.length) {
    container.innerHTML = `<p class="melody-empty">No aligned score notes for this sample.</p>`;
    return;
  }
  const goldIds = new Set();
  const predIds = new Set();
  const d = selectedRegion?.data;
  if (d?.compare?.layer === "gold") {
    (d.note_ids || d.core_note_ids || []).forEach((id) => goldIds.add(String(id)));
  } else if (d?.compare?.layer === "pred") {
    (d.note_ids || d.core_note_ids || []).forEach((id) => predIds.add(String(id)));
    const partner = d.compare?.pair_index != null
      ? (compareData?.gold || [])[d.compare.pair_index]
      : null;
    (partner?.note_ids || partner?.core_note_ids || []).forEach((id) => goldIds.add(String(id)));
  }
  const px = getPxPerSec();
  container.innerHTML = "";
  scoreEvents.forEach((ev) => {
    const hit = document.createElement("div");
    hit.className = "melody-hit";
    const id = String(ev.id || ev.note_id || "");
    if (goldIds.has(id)) hit.classList.add("core");
    if (predIds.has(id)) hit.classList.add("pred-core");
    const t0 = eventStart(ev);
    const t1 = eventEnd(ev);
    hit.style.left = `${t0 * px}px`;
    hit.style.width = `${Math.max(4, (t1 - t0) * px)}px`;
    hit.title = `${ev.is_rest ? "Rest" : (ev.pitch || id)}${ev.measure != null ? ` · m${ev.measure}` : ""}`;
    container.appendChild(hit);
  });
}

function setScoreStatus(msg) {
  const el = document.getElementById("osmdContainer");
  if (el && msg) el.innerHTML = `<p class="score-placeholder">${msg}</p>`;
}

function scorePageWidth() {
  const container = document.getElementById("osmdContainer");
  return Math.max(400, (container?.clientWidth || 800) - 24);
}

async function renderScoreXml(xml) {
  const container = document.getElementById("osmdContainer");
  container.innerHTML = "";
  osmdInstance = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
    autoResize: false,
    drawTitle: true,
  });
  await osmdInstance.load(xml);
  if (osmdInstance.EngravingRules) {
    osmdInstance.EngravingRules.PageWidth = scorePageWidth();
  }
  osmdInstance.render();
}

async function loadScore(url) {
  setScoreStatus("Loading score…");
  const res = await fetch(url);
  if (!res.ok) {
    setScoreStatus("Could not load score.");
    return;
  }
  await renderScoreXml(await res.text());
}

async function loadSample(sampleId) {
  const loadId = ++scoreLoadId;
  currentSample = sampleId;
  selectedRegion = null;
  scoreEvents = [];
  setScoreStatus("Loading…");

  const [metaRes, cmpRes] = await Promise.all([
    fetch(`/api/samples/${sampleId}`),
    fetch(`/api/compare/samples/${sampleId}`),
  ]);
  if (loadId !== scoreLoadId) return;
  if (!metaRes.ok) {
    setScoreStatus("Sample not found.");
    return;
  }
  sampleData = await metaRes.json();
  compareData = cmpRes.ok ? await cmpRes.json() : { gold: [], pred: [] };
  if (loadId !== scoreLoadId) return;

  const stats = document.getElementById("sampleStats");
  const m = compareData.metrics || {};
  stats.textContent =
    `gold ${compareData.gold?.length || 0} · model ${compareData.pred?.length || 0}`
    + ` · type-sens F1 ${fmtF1(m.hard_type_sensitive?.melody_f1)}`
    + ` · type-insens F1 ${fmtF1(m.hard_type_insensitive?.melody_f1)}`;

  const prep = sampleData.prep || {};
  const total = prep.total_measures;
  document.getElementById("measureTotal").textContent =
    total ? `${total} measures` : "";

  if (regionsPlugin) regionsPlugin.clearRegions();
  pendingLabels = [...(compareData.gold || []), ...(compareData.pred || [])];

  const duration = prep.performance_duration || 0;
  if (duration > 0 && typeof wavesurfer.setOptions === "function") {
    const fit = getScrollWidth() / duration;
    wavesurfer.setOptions({ minPxPerSec: fit, width: Math.max(getScrollWidth(), duration * fit) });
  }
  const audioUrl = `${sampleData.audio_url}?t=${sampleData.audio_mtime || 0}`;
  await wavesurfer.load(audioUrl);
  if (loadId !== scoreLoadId) return;

  const measures = prep.total_measures || 0;
  if (prep.score_segment) {
    const seg = prep.score_segment;
    const q = new URLSearchParams({
      start_measure: String(seg.start_measure),
      end_measure: String(seg.end_measure),
      start_beat: String(seg.start_beat || 1),
    });
    if (seg.end_beat != null) q.set("end_beat", String(seg.end_beat));
    await loadScore(`/api/samples/${sampleId}/score-preview?${q}`);
  } else if (measures > 40) {
    setScoreStatus(`Full score has ${measures} measures. Use View segment after opening the annotator, or View full.`);
  } else {
    await loadScore(sampleData.full_score_url || sampleData.score_url);
  }

  try {
    const evRes = await fetch(`/api/samples/${sampleId}/score-events`);
    if (evRes.ok) {
      const payload = await evRes.json();
      scoreEvents = payload.events || [];
    }
  } catch {
    scoreEvents = [];
  }
  renderMelodyStrip();
  renderLists();
  renderSelection();
}

function onWaveformReady() {
  userZoomed = false;
  zoomFactor = 1;
  document.getElementById("zoomSlider").value = "50";
  pendingLabels.forEach((label) => addLabelRegion(label));
  pendingLabels = [];
  fitWaveform();
  layoutRegions();
  renderLists();
}

function setupScrubber() {
  const scrubber = document.getElementById("scrubber");
  const playhead = document.getElementById("playhead");
  const seek = (clientX) => {
    const duration = wavesurfer.getDuration();
    if (!duration) return;
    const rect = scrubber.getBoundingClientRect();
    const scroll = document.getElementById("waveformScroll");
    const x = clientX - rect.left + (scroll?.scrollLeft || 0);
    wavesurfer.setTime(Math.max(0, Math.min(duration, x / getPxPerSec())));
    updatePlayhead();
  };
  scrubber.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    scrubbing = true;
    playhead.classList.add("dragging");
    scrubber.setPointerCapture(e.pointerId);
    seek(e.clientX);
  });
  scrubber.addEventListener("pointermove", (e) => {
    if (scrubbing) seek(e.clientX);
  });
  const stop = () => {
    scrubbing = false;
    playhead.classList.remove("dragging");
  };
  scrubber.addEventListener("pointerup", stop);
  scrubber.addEventListener("pointercancel", stop);
}

async function init() {
  regionsPlugin = WaveSurfer.Regions.create();
  wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#6af",
    progressColor: "#2d6cdf",
    cursorColor: "#fff",
    cursorWidth: 2,
    height: 160,
    minPxPerSec: 1,
    fillParent: false,
    hideScrollbar: true,
    autoScroll: false,
    autoCenter: false,
    dragToSeek: false,
    plugins: [regionsPlugin],
  });
  wavesurfer.on("timeupdate", updatePlayhead);
  wavesurfer.on("seeking", updatePlayhead);
  wavesurfer.on("ready", onWaveformReady);
  wavesurfer.on("finish", () => {
    if (loopSelection && selectedRegion) selectedRegion.play();
  });
  regionsPlugin.on("region-clicked", (region, e) => {
    e?.stopPropagation?.();
    selectRegion(region);
  });

  document.getElementById("playBtn").onclick = () => wavesurfer.playPause();
  document.getElementById("loopBtn").onclick = () => {
    loopSelection = !loopSelection;
    document.getElementById("loopBtn").textContent = loopSelection ? "Loop: ON" : "Loop selection";
  };
  document.getElementById("speedSelect").onchange = (e) => {
    wavesurfer.setPlaybackRate(parseFloat(e.target.value));
  };
  document.getElementById("toggleGoldBtn").onclick = () => {
    showGold = !showGold;
    document.getElementById("toggleGoldBtn").classList.toggle("active", showGold);
    layoutRegions();
  };
  document.getElementById("togglePredBtn").onclick = () => {
    showPred = !showPred;
    document.getElementById("togglePredBtn").classList.toggle("active", showPred);
    layoutRegions();
  };
  document.getElementById("zoomFitBtn").onclick = () => {
    userZoomed = false;
    zoomFactor = 1;
    document.getElementById("zoomSlider").value = "50";
    fitWaveform();
  };
  document.getElementById("zoomSlider").oninput = (e) => {
    userZoomed = true;
    const t = parseInt(e.target.value, 10) / 100;
    zoomFactor = 1 + t * 7;
    fitWaveform();
  };
  document.getElementById("viewFullScoreBtn").onclick = () => {
    if (sampleData?.full_score_url) loadScore(sampleData.full_score_url);
  };
  document.getElementById("viewSegmentScoreBtn").onclick = () => {
    const seg = sampleData?.prep?.score_segment;
    if (!seg || !currentSample) return;
    const q = new URLSearchParams({
      start_measure: String(seg.start_measure),
      end_measure: String(seg.end_measure),
      start_beat: String(seg.start_beat || 1),
    });
    if (seg.end_beat != null) q.set("end_beat", String(seg.end_beat));
    loadScore(`/api/samples/${currentSample}/score-preview?${q}`);
  };
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space") return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    e.preventDefault();
    wavesurfer.playPause();
  });
  window.addEventListener("resize", () => {
    if (!userZoomed) fitWaveform();
  });
  setupScrubber();

  const summary = await fetch("/api/compare/summary").then((r) => r.json());
  const meta = document.getElementById("evalMeta");
  const pills = document.getElementById("headlineStats");
  const crit = summary.criteria || {};
  meta.textContent = `${summary.n_samples || 0} labeled clips · checkpoint ${String(summary.checkpoint || "").split(/[/\\]/).pop()}`;
  pills.innerHTML = `
    <span class="pill">type-sens F1 ${fmtF1(crit.hard_type_sensitive?.mean_melody_f1)}</span>
    <span class="pill">type-insens F1 ${fmtF1(crit.hard_type_insensitive?.mean_melody_f1)}</span>
  `;
  const sel = document.getElementById("sampleSelect");
  const samples = summary.samples || [];
  sel.innerHTML = samples.map((s) =>
    `<option value="${s.id}">${s.id} · G${s.n_gold} P${s.n_pred} · F1 ${fmtF1(s.hard_type_sensitive_f1)}</option>`
  ).join("");
  sel.onchange = () => loadSample(sel.value);
  if (samples.length) await loadSample(samples[0].id);
}

init();
