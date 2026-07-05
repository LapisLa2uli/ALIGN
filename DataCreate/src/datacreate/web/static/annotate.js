let wavesurfer;
let regionsPlugin;
let taxonomy = [];
let currentSample = null;
let selectedRegion = null;
let loopSelection = false;
let scrubbing = false;

const TYPE_COLORS = {
  wrong_note: "rgba(255,60,60,0.45)",
  wrong_pitch: "rgba(255,60,60,0.45)", // legacy auto-detect alias
  missed_note: "rgba(255,140,0,0.4)",
  extra_note: "rgba(255,200,0,0.4)",
  intonation_error: "rgba(180,80,255,0.4)",
  rhythm_error: "rgba(80,180,255,0.4)",
  repetition: "rgba(80,255,160,0.4)",
  stylistic_choice: "rgba(160,160,160,0.35)",
};

const TYPE_LABELS = {
  wrong_note: "Wrong note (different pitch than score)",
  wrong_pitch: "Wrong note (legacy auto-detect)",
  intonation_error: "Intonation / tuning error",
  missed_note: "Missed note",
  extra_note: "Extra note",
  rhythm_error: "Rhythm / timing error",
  repetition: "Repetition",
  stylistic_choice: "Stylistic choice (not an error)",
};

async function init() {
  regionsPlugin = WaveSurfer.Regions.create();
  wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#6af",
    progressColor: "#2d6cdf",
    cursorColor: "#fff",
    cursorWidth: 2,
    height: 140,
    minPxPerSec: 50,
    dragToSeek: true,
    plugins: [regionsPlugin],
  });

  wavesurfer.on("timeupdate", updatePlayhead);
  wavesurfer.on("seeking", updatePlayhead);
  wavesurfer.on("ready", updatePlayhead);
  wavesurfer.on("finish", () => {
    if (loopSelection && selectedRegion) {
      selectedRegion.play();
    }
  });

  setupScrubber();

  regionsPlugin.enableDragSelection({ color: "rgba(45,108,223,0.25)" });
  regionsPlugin.on("region-clicked", (region, e) => {
    e.stopPropagation();
    selectRegion(region);
  });
  regionsPlugin.on("region-updated", (region) => selectRegion(region));

  document.getElementById("playBtn").onclick = () => wavesurfer.playPause();
  document.getElementById("loopBtn").onclick = () => {
    loopSelection = !loopSelection;
    document.getElementById("loopBtn").textContent = loopSelection ? "Loop: ON" : "Loop selection";
  };
  document.getElementById("speedSelect").onchange = (e) => {
    wavesurfer.setPlaybackRate(parseFloat(e.target.value));
  };
  document.getElementById("saveBtn").onclick = saveLabels;
  document.getElementById("applyLabelBtn").onclick = applyLabelToSelection;
  document.getElementById("confirmCandidateBtn").onclick = () => promoteCandidate("auto_confirmed");
  document.getElementById("rejectCandidateBtn").onclick = () => promoteCandidate("auto_rejected", "stylistic_choice");
  document.getElementById("deleteRegionBtn").onclick = deleteSelectedRegion;

  document.addEventListener("keydown", onKeyDown);

  await loadSampleList();
}

function setupScrubber() {
  const scrubber = document.getElementById("scrubber");
  const playhead = document.getElementById("playhead");

  const seekFromPointer = (clientX) => {
    const duration = wavesurfer.getDuration();
    if (!duration) return;
    const rect = scrubber.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    wavesurfer.setTime(ratio * duration);
    updatePlayhead();
  };

  scrubber.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    scrubbing = true;
    playhead.classList.add("dragging");
    scrubber.setPointerCapture(e.pointerId);
    seekFromPointer(e.clientX);
    e.preventDefault();
  });

  scrubber.addEventListener("pointermove", (e) => {
    if (!scrubbing) return;
    seekFromPointer(e.clientX);
  });

  const stopScrub = (e) => {
    if (!scrubbing) return;
    scrubbing = false;
    playhead.classList.remove("dragging");
    if (e && scrubber.hasPointerCapture(e.pointerId)) {
      scrubber.releasePointerCapture(e.pointerId);
    }
  };

  scrubber.addEventListener("pointerup", stopScrub);
  scrubber.addEventListener("pointercancel", stopScrub);
}

function updatePlayhead() {
  const duration = wavesurfer.getDuration();
  const scrubber = document.getElementById("scrubber");
  const progress = document.getElementById("scrubberProgress");
  const playhead = document.getElementById("playhead");
  if (!duration || !scrubber) return;

  const ratio = wavesurfer.getCurrentTime() / duration;
  const x = ratio * scrubber.clientWidth;
  playhead.style.left = `${x}px`;
  progress.style.width = `${ratio * 100}%`;
  document.getElementById("timeDisplay").textContent = formatTime(wavesurfer.getCurrentTime());
}

async function loadSampleList() {
  const res = await fetch("/api/samples");
  const samples = await res.json();
  const sel = document.getElementById("sampleSelect");
  sel.innerHTML = samples.map((s) => `<option value="${s}">${s}</option>`).join("");
  sel.onchange = () => loadSample(sel.value);
  if (samples.length) await loadSample(samples[0]);
}

async function loadSample(sampleId) {
  currentSample = sampleId;
  const res = await fetch(`/api/samples/${sampleId}`);
  const data = await res.json();
  taxonomy = data.taxonomy;
  populateTypeSelect();
  document.getElementById("annotatorId").value = data.annotator_id || "";

  regionsPlugin.clearRegions();
  await wavesurfer.load(data.audio_url);

  data.candidates.forEach((c) => addRegion(normalizeLabelType(c), true));
  data.labels.forEach((l) => addRegion(normalizeLabelType(l), false));

  updatePlayhead();
  await renderScore(data.score_url);
}

function normalizeLabelType(label) {
  if (label.type === "wrong_pitch") {
    return { ...label, type: "wrong_note", _legacyWrongPitch: true };
  }
  return label;
}

function populateTypeSelect() {
  const sel = document.getElementById("labelType");
  sel.innerHTML = taxonomy
    .map((t) => `<option value="${t}">${TYPE_LABELS[t] || t}</option>`)
    .join("");
}

function addRegion(label, isCandidate) {
  const color = TYPE_COLORS[label.type] || "rgba(45,108,223,0.35)";
  const display = TYPE_LABELS[label.type] || label.type;
  const region = regionsPlugin.addRegion({
    start: label.start_time,
    end: label.end_time,
    color,
    drag: true,
    resize: true,
    content: display.split(" (")[0],
    data: { ...label, isCandidate },
  });
  region.element.classList.add(isCandidate ? "region-candidate" : "region-label");
  return region;
}

function selectRegion(region) {
  selectedRegion = region;
  const d = region.data || {};
  const typeLabel = TYPE_LABELS[d.type] || d.type || "unlabeled";
  document.getElementById("selectionInfo").textContent =
    `${formatTime(region.start)} – ${formatTime(region.end)} (${typeLabel})`;
  if (d.type) document.getElementById("labelType").value = d.type;
  if (d.severity) document.getElementById("severity").value = d.severity;
  document.getElementById("comment").value = d.comment || "";
}

function applyLabelToSelection() {
  if (!selectedRegion) return;
  const type = document.getElementById("labelType").value;
  selectedRegion.data = {
    ...(selectedRegion.data || {}),
    id: (selectedRegion.data && selectedRegion.data.id) || `lbl_${Date.now()}`,
    source: (selectedRegion.data && selectedRegion.data.source) || "manual",
    start_time: selectedRegion.start,
    end_time: selectedRegion.end,
    type,
    severity: parseInt(document.getElementById("severity").value, 10),
    comment: document.getElementById("comment").value || null,
  };
  const display = (TYPE_LABELS[type] || type).split(" (")[0];
  selectedRegion.setOptions({
    content: display,
    color: TYPE_COLORS[type] || selectedRegion.color,
  });
  selectedRegion.element.classList.remove("region-candidate");
  selectedRegion.element.classList.add("region-label");
}

function promoteCandidate(source, overrideType) {
  if (!selectedRegion || !selectedRegion.data?.isCandidate) return;
  selectedRegion.data.source = source;
  if (overrideType) selectedRegion.data.type = overrideType;
  selectedRegion.data.isCandidate = false;
  applyLabelToSelection();
}

function deleteSelectedRegion() {
  if (selectedRegion) {
    selectedRegion.remove();
    selectedRegion = null;
  }
}

async function saveLabels() {
  const labels = [];
  const candidatesLeft = [];
  regionsPlugin.getRegions().forEach((r) => {
    const d = r.data || {
      id: `lbl_${r.start}`,
      source: "manual",
      start_time: r.start,
      end_time: r.end,
      type: document.getElementById("labelType").value,
    };
    d.start_time = r.start;
    d.end_time = r.end;
    if (d.isCandidate) candidatesLeft.push(d);
    else labels.push(d);
  });
  if (candidatesLeft.length) {
    alert(`${candidatesLeft.length} candidates still unreviewed. Confirm or reject them before saving.`);
    return;
  }
  const payload = {
    labels,
    self_reported: [],
    annotator_id: document.getElementById("annotatorId").value || null,
  };
  const res = await fetch(`/api/samples/${currentSample}/labels`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    alert(await res.text());
    return;
  }
  alert("Labels saved.");
}

async function renderScore(url) {
  const container = document.getElementById("osmdContainer");
  container.innerHTML = "";
  const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
    autoResize: true,
    drawTitle: true,
  });
  const xml = await (await fetch(url)).text();
  await osmd.load(xml);
  osmd.render();
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
}

function onKeyDown(e) {
  if (e.code === "Space") {
    e.preventDefault();
    wavesurfer.playPause();
  }
  const idx = parseInt(e.key, 10);
  if (idx >= 1 && idx <= taxonomy.length) {
    document.getElementById("labelType").value = taxonomy[idx - 1];
    applyLabelToSelection();
  }
  if (!selectedRegion) return;
  const step = 0.01;
  if (e.key === "ArrowLeft") selectedRegion.setOptions({ start: Math.max(0, selectedRegion.start - step) });
  if (e.key === "ArrowRight") selectedRegion.setOptions({ end: selectedRegion.end + step });
}

init();
