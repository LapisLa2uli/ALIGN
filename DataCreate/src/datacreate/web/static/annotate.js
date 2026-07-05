let wavesurfer;
let regionsPlugin;
let taxonomy = [];
let currentSample = null;
let selectedRegion = null;
let loopSelection = false;

const TYPE_COLORS = {
  wrong_pitch: "rgba(255,80,80,0.4)",
  missed_note: "rgba(255,140,0,0.4)",
  extra_note: "rgba(255,200,0,0.4)",
  intonation_error: "rgba(180,80,255,0.4)",
  rhythm_error: "rgba(80,180,255,0.4)",
  repetition: "rgba(80,255,160,0.4)",
  stylistic_choice: "rgba(160,160,160,0.35)",
};

async function init() {
  regionsPlugin = WaveSurfer.Regions.create();
  wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#6af",
    progressColor: "#2d6cdf",
    cursorColor: "#fff",
    height: 140,
    minPxPerSec: 50,
    plugins: [regionsPlugin],
  });

  wavesurfer.on("timeupdate", (t) => {
    document.getElementById("timeDisplay").textContent = formatTime(t);
  });
  wavesurfer.on("finish", () => {
    if (loopSelection && selectedRegion) {
      selectedRegion.play();
    }
  });

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

  data.candidates.forEach((c) => addRegion(c, true));
  data.labels.forEach((l) => addRegion(l, false));

  await renderScore(data.score_url);
}

function populateTypeSelect() {
  const sel = document.getElementById("labelType");
  sel.innerHTML = taxonomy.map((t) => `<option value="${t}">${t}</option>`).join("");
}

function addRegion(label, isCandidate) {
  const color = TYPE_COLORS[label.type] || "rgba(45,108,223,0.35)";
  const region = regionsPlugin.addRegion({
    start: label.start_time,
    end: label.end_time,
    color,
    drag: true,
    resize: true,
    content: label.type,
    data: { ...label, isCandidate },
  });
  region.element.classList.add(isCandidate ? "region-candidate" : "region-label");
  return region;
}

function selectRegion(region) {
  selectedRegion = region;
  const d = region.data || {};
  document.getElementById("selectionInfo").textContent =
    `${formatTime(region.start)} – ${formatTime(region.end)} (${d.type || "unlabeled"})`;
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
  selectedRegion.setOptions({
    content: type,
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
