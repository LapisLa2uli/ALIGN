let humanWave;
let agentWave;
let humanRegions;
let agentRegions;
let currentSample = null;
let currentHumanLabels = [];
let currentAgentLabels = [];
let fitPxPerSec = 1;
let zoomFactor = 1;
let loadToken = 0;
let syncRaf = null;
let pendingSyncTime = 0;

const TYPE_COLORS = {
  wrong_note: "rgba(255,60,60,0.48)",
  wrong_pitch: "rgba(255,60,60,0.48)",
  missed_note: "rgba(255,140,0,0.46)",
  extra_note: "rgba(255,200,0,0.46)",
  intonation_error: "rgba(180,80,255,0.46)",
  rhythm_error: "rgba(80,180,255,0.46)",
  repetition: "rgba(80,255,160,0.46)",
  stylistic_choice: "rgba(160,160,160,0.4)",
  bad_start: "rgba(210,110,40,0.48)",
  bad_timbre: "rgba(0,170,150,0.46)",
  squeak: "rgba(255,50,170,0.48)",
  sliding: "rgba(255,130,70,0.48)",
  click: "rgba(255,220,80,0.48)",
  misc: "rgba(140,140,160,0.48)",
};

const TYPE_LABELS = {
  wrong_note: "Wrong note",
  wrong_pitch: "Wrong note",
  missed_note: "Missed note",
  extra_note: "Extra note",
  intonation_error: "Intonation",
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
  return TYPE_LABELS[type] || type || "Unlabelled";
}

function formatTime(sec) {
  if (!Number.isFinite(Number(sec))) return "00:00.000";
  const value = Math.max(0, Number(sec));
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(3).padStart(6, "0")}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scrollWidth() {
  return document.getElementById("dualWaveformScroll")?.clientWidth || 900;
}

function duration() {
  return humanWave?.getDuration?.() || 0;
}

function currentPxPerSec() {
  return Math.max(fitPxPerSec, fitPxPerSec * zoomFactor);
}

function waveformWidth() {
  return Math.max(scrollWidth(), duration() * currentPxPerSec());
}

function styleRegion(region, source) {
  const element = region?.element;
  if (!element) return;
  element.style.border = source === "human"
    ? "2px solid rgba(153,204,255,0.95)"
    : "2px solid rgba(255,204,102,0.95)";
  element.style.boxSizing = "border-box";
  element.style.overflow = "visible";
}

function addRegions(plugin, labels, source) {
  plugin.clearRegions();
  labels.forEach((label, index) => {
    const start = Number(label.start_time);
    const end = Number(label.end_time);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
    const region = plugin.addRegion({
      start,
      end,
      color: TYPE_COLORS[label.type] || "rgba(45,108,223,0.42)",
      drag: false,
      resize: false,
      content: typeName(label.type),
      data: { ...label, compare_source: source, compare_index: index },
    });
    styleRegion(region, source);
  });
}

function selectLabel(label, source) {
  const start = Number(label?.start_time);
  if (Number.isFinite(start)) {
    humanWave.setTime(start);
    agentWave.setTime(start);
  }
  const sourceName = source === "human" ? "Your label" : "Agent";
  const info = document.getElementById("selectionInfo");
  info.textContent =
    `${sourceName}: ${typeName(label?.type)} · ` +
    `${formatTime(label?.start_time)}–${formatTime(label?.end_time)}` +
    (label?.comment ? ` · ${label.comment}` : "");
}

function renderLabelList(elementId, labels, source) {
  const root = document.getElementById(elementId);
  if (!labels.length) {
    root.innerHTML = `<p class="hint">No ${source === "human" ? "human" : "agent"} labels.</p>`;
    return;
  }
  root.innerHTML = labels.map((label, index) => `
    <button type="button" class="label-row" data-index="${index}">
      <span class="swatch" style="background:${TYPE_COLORS[label.type] || "#2d6cdf"}"></span>
      <span class="row-type">${escapeHtml(typeName(label.type))}</span>
      <span class="row-time">${formatTime(label.start_time)}–${formatTime(label.end_time)}</span>
      <span class="row-match">${escapeHtml(label.comment || label.source || source)}</span>
    </button>
  `).join("");
  root.querySelectorAll(".label-row").forEach((button) => {
    button.onclick = () => selectLabel(labels[Number(button.dataset.index)], source);
  });
}

function applyZoom() {
  const px = currentPxPerSec();
  const width = waveformWidth();
  [humanWave, agentWave].forEach((wave) => {
    wave.setOptions({ minPxPerSec: px, width });
  });
  const content = document.getElementById("dualWaveformContent");
  content.style.width = `${width}px`;
  document.querySelectorAll(".parallel-waveform").forEach((element) => {
    element.style.width = `${width}px`;
  });
}

function fitWaveforms() {
  const audioDuration = duration();
  if (!(audioDuration > 0)) return;
  fitPxPerSec = scrollWidth() / audioDuration;
  applyZoom();
}

function scheduleAgentPlayhead(time) {
  pendingSyncTime = time;
  if (syncRaf) return;
  syncRaf = requestAnimationFrame(() => {
    syncRaf = null;
    const agentTime = agentWave?.getCurrentTime?.() || 0;
    if (Math.abs(agentTime - pendingSyncTime) > 0.015) {
      agentWave.setTime(pendingSyncTime);
    }
  });
}

async function loadSample(sampleId) {
  const token = ++loadToken;
  currentSample = sampleId;
  humanWave.pause();
  humanRegions.clearRegions();
  agentRegions.clearRegions();
  document.getElementById("sampleStats").textContent = "Loading…";
  document.getElementById("selectionInfo").textContent = "Click a label to seek to it.";

  const humanQuery = new URLSearchParams({ label_source: "human" });
  const agentQuery = new URLSearchParams({ label_source: "agent" });
  const [humanResponse, agentResponse] = await Promise.all([
    fetch(`/api/samples/${sampleId}?${humanQuery}`),
    fetch(`/api/samples/${sampleId}?${agentQuery}`),
  ]);
  if (!humanResponse.ok || !agentResponse.ok) {
    throw new Error("Could not load label sets for this sample.");
  }
  const [humanData, agentData] = await Promise.all([
    humanResponse.json(),
    agentResponse.json(),
  ]);
  if (token !== loadToken) return;

  currentHumanLabels = humanData.labels || [];
  currentAgentLabels = agentData.labels || [];
  const audioResponse = await fetch(
    `${humanData.audio_url}?t=${humanData.audio_mtime || 0}`,
  );
  if (!audioResponse.ok) throw new Error("Could not load performance audio.");
  const audioBlob = await audioResponse.blob();
  if (token !== loadToken) return;

  await Promise.all([
    humanWave.loadBlob(audioBlob),
    agentWave.loadBlob(audioBlob),
  ]);
  if (token !== loadToken) return;

  agentWave.setMuted(true);
  zoomFactor = 1;
  document.getElementById("zoomSlider").value = "0";
  fitWaveforms();
  addRegions(humanRegions, currentHumanLabels, "human");
  addRegions(agentRegions, currentAgentLabels, "agent");
  renderLabelList("humanList", currentHumanLabels, "human");
  renderLabelList("agentList", currentAgentLabels, "agent");
  document.getElementById("humanCount").textContent =
    `${currentHumanLabels.length} label${currentHumanLabels.length === 1 ? "" : "s"}`;
  document.getElementById("agentCount").textContent =
    `${currentAgentLabels.length} label${currentAgentLabels.length === 1 ? "" : "s"}`;
  document.getElementById("sampleStats").textContent =
    `${sampleId} · your labels ${currentHumanLabels.length} · agent ${currentAgentLabels.length}`;
}

async function loadSampleList() {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error("Could not list samples.");
  const samples = await response.json();
  const select = document.getElementById("sampleSelect");
  select.innerHTML = samples.map((sample) => (
    `<option value="${escapeHtml(sample.id)}">${escapeHtml(sample.id)} ` +
    `(${sample.label_count || 0} yours, ${sample.agent_label_count || 0} agent)</option>`
  )).join("");
  select.onchange = () => {
    if (select.value) loadSample(select.value).catch(showError);
  };
  if (samples.length) {
    await loadSample(samples[0].id);
  } else {
    document.getElementById("sampleStats").textContent = "No samples found.";
  }
}

function showError(error) {
  document.getElementById("sampleStats").textContent =
    `Error: ${error?.message || String(error)}`;
}

function createWave(container, plugins, waveColor, progressColor) {
  return WaveSurfer.create({
    container,
    waveColor,
    progressColor,
    cursorColor: "#fff",
    cursorWidth: 2,
    height: 170,
    minPxPerSec: 1,
    fillParent: false,
    hideScrollbar: true,
    autoScroll: false,
    autoCenter: false,
    dragToSeek: true,
    plugins,
  });
}

async function init() {
  humanRegions = WaveSurfer.Regions.create();
  agentRegions = WaveSurfer.Regions.create();
  humanWave = createWave(
    "#humanWaveform",
    [humanRegions],
    "#6af",
    "#2d6cdf",
  );
  agentWave = createWave(
    "#agentWaveform",
    [agentRegions],
    "#f0b45a",
    "#c47b18",
  );

  humanWave.on("timeupdate", (time) => {
    document.getElementById("timeDisplay").textContent = formatTime(time);
    scheduleAgentPlayhead(time);
  });
  humanWave.on("seeking", (time) => scheduleAgentPlayhead(time));
  agentWave.on("interaction", (time) => {
    humanWave.setTime(time);
    scheduleAgentPlayhead(time);
  });
  humanWave.on("finish", () => agentWave.setTime(duration()));

  humanRegions.on("region-clicked", (region, event) => {
    event?.stopPropagation?.();
    selectLabel(region.data, "human");
  });
  agentRegions.on("region-clicked", (region, event) => {
    event?.stopPropagation?.();
    selectLabel(region.data, "agent");
  });

  document.getElementById("playBtn").onclick = () => humanWave.playPause();
  document.getElementById("speedSelect").onchange = (event) => {
    humanWave.setPlaybackRate(Number(event.target.value));
  };
  document.getElementById("zoomSlider").oninput = (event) => {
    const normalized = Number(event.target.value) / 100;
    zoomFactor = Math.pow(16, normalized);
    applyZoom();
  };
  document.getElementById("zoomFitBtn").onclick = () => {
    zoomFactor = 1;
    document.getElementById("zoomSlider").value = "0";
    fitWaveforms();
  };
  window.addEventListener("resize", () => {
    if (zoomFactor === 1) fitWaveforms();
  });
  window.addEventListener("keydown", (event) => {
    const target = event.target;
    if (
      event.code === "Space"
      && !["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target?.tagName)
    ) {
      event.preventDefault();
      humanWave.playPause();
    }
  });

  await loadSampleList();
}

init().catch(showError);
