let wavesurfer;
let regionsPlugin;
let osmdInstance = null;
let taxonomy = [];
let currentSample = null;
let sampleData = null;
let selectedRegion = null;
let trimRegion = null;
let loopSelection = false;
let scrubbing = false;
let pendingRegions = [];
let lastRegionClick = { id: null, time: 0 };
const REGION_DOUBLE_CLICK_MS = 400;
let repetitionLinkMode = null; // null | "pick" | "draw"
/** >0 while addRegion is called from code (labels, trim, link overlays) — skip auto-label. */
let programmaticRegionDepth = 0;

function withProgrammaticRegions(fn) {
  programmaticRegionDepth += 1;
  try {
    return fn();
  } finally {
    programmaticRegionDepth -= 1;
  }
}

const TYPE_COLORS = {
  wrong_note: "rgba(255,60,60,0.45)",
  wrong_pitch: "rgba(255,60,60,0.45)",
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

const LARGE_SCORE_MEASURES = 40;
const MAX_SEGMENT_PREVIEW_MEASURES = 64;

function audioUrlWithCacheBust(baseUrl, audioMtime) {
  const token = audioMtime || Date.now();
  const sep = baseUrl.includes("?") ? "&" : "?";
  return `${baseUrl}${sep}v=${encodeURIComponent(String(token))}`;
}

function defaultMeasureRange(prep) {
  const total = prep?.total_measures || 1;
  if (prep?.score_segment) {
    return {
      start: prep.score_segment.start_measure,
      end: prep.score_segment.end_measure,
      startBeat: prep.score_segment.start_beat || 1,
      endBeat: prep.score_segment.end_beat ?? null,
    };
  }
  if (total <= LARGE_SCORE_MEASURES) {
    return { start: 1, end: total, startBeat: 1, endBeat: null };
  }
  const perf = prep?.performance_duration;
  const refSec = prep?.reference_duration_seconds;
  if (perf && refSec && refSec > 0) {
    const estimatedEnd = Math.ceil(total * (perf / refSec) * 1.1);
    return {
      start: 1,
      end: Math.max(1, Math.min(total, estimatedEnd)),
      startBeat: 1,
      endBeat: null,
    };
  }
  return {
    start: 1,
    end: Math.min(total, LARGE_SCORE_MEASURES),
    startBeat: 1,
    endBeat: null,
  };
}

function parseBeatInput(id) {
  const raw = document.getElementById(id).value.trim();
  if (!raw) return null;
  const value = parseInt(raw, 10);
  return Number.isNaN(value) ? null : value;
}

function formatSegmentLabel(start, end, startBeat, endBeat) {
  const startPart = startBeat > 1 ? `m${start} beat ${startBeat}` : `m${start}`;
  const endPart = endBeat != null ? `m${end} beat ${endBeat}` : `m${end}`;
  return `${startPart}–${endPart}`;
}

function getDragSelectionColor() {
  if (repetitionLinkMode === "draw") {
    return "rgba(200, 200, 210, 0.35)";
  }
  const type = document.getElementById("labelType")?.value;
  return TYPE_COLORS[type] || "rgba(45,108,223,0.35)";
}

function refreshDragSelection() {
  if (!regionsPlugin?.enableDragSelection) return;
  if (typeof regionsPlugin.disableDragSelection === "function") {
    regionsPlugin.disableDragSelection();
  }
  regionsPlugin.enableDragSelection({ color: getDragSelectionColor() });
}

function generateLabelId() {
  return `lbl_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

function isLinkOverlay(region) {
  return region?.data?.role === "repetition-link";
}

function getRegionKind(region) {
  if (isTrimRegion(region)) return "trim";
  if (isLinkOverlay(region)) return "link";
  if (region.data?.isCandidate) return "candidate";
  return "label";
}

function syncRegionVisual(region, { selected = false } = {}) {
  const el = region.element;
  if (!el) return;

  const kind = getRegionKind(region);
  const parts = ["region"];
  if (kind === "trim") parts.push("region-trim");
  else if (kind === "link") parts.push("region-link");
  else if (kind === "candidate") parts.push("region-candidate");
  else parts.push("region-label");
  if (selected) parts.push("region-selected");
  el.setAttribute("part", parts.join(" "));

  if (kind === "trim") {
    el.style.border = "2px solid #3c3";
    el.style.backgroundColor = "rgba(60, 200, 60, 0.12)";
    el.style.boxShadow = selected ? "0 0 0 1px rgba(0, 0, 0, 0.45)" : "";
    el.style.zIndex = selected ? "10" : "1";
    el.style.boxSizing = "border-box";
    return;
  }

  if (kind === "link") {
    el.style.border = "2px dashed #c8c8d0";
    el.style.backgroundColor = "rgba(200, 200, 210, 0.35)";
    el.style.boxShadow = "";
    el.style.zIndex = "0";
    el.style.boxSizing = "border-box";
    el.style.pointerEvents = "none";
    return;
  }

  el.style.boxSizing = "border-box";
  if (selected) {
    el.style.border = "3px solid #fff";
    el.style.boxShadow = "0 0 0 1px rgba(0, 0, 0, 0.45)";
    el.style.zIndex = "10";
  } else if (kind === "candidate") {
    el.style.border = "2px dashed #f90";
    el.style.boxShadow = "";
    el.style.zIndex = "";
  } else {
    el.style.border = "none";
    el.style.boxShadow = "";
    el.style.zIndex = "";
  }
}

function findLinkOverlay(parentRegion) {
  const parentId = parentRegion?.data?.id;
  if (!parentId) return null;
  return regionsPlugin.getRegions().find(
    (region) => region.data?.role === "repetition-link" && region.data?.parentId === parentId,
  );
}

function syncRepetitionLinkOverlay(parentRegion) {
  const existing = findLinkOverlay(parentRegion);
  if (existing) existing.remove();

  const range = parentRegion?.data?.repeats_label_range;
  if (!range || parentRegion.data?.type !== "repetition") return;

  const createOverlay = () =>
    regionsPlugin.addRegion({
      start: range.start_time,
      end: range.end_time,
      color: "rgba(200, 200, 210, 0.35)",
      drag: false,
      resize: false,
      content: "original",
      id: `link-${parentRegion.data.id}`,
      data: { role: "repetition-link", parentId: parentRegion.data.id },
    });
  const overlay =
    programmaticRegionDepth > 0
      ? createOverlay()
      : withProgrammaticRegions(createOverlay);
  syncRegionVisual(overlay);
}

function setRepetitionLinkRange(parentRegion, start, end, existingRegion = null) {
  if (!parentRegion?.data || parentRegion.data.type !== "repetition") return;
  const lo = Math.min(start, end);
  const hi = Math.max(start, end);
  if (hi - lo < 1e-4) {
    existingRegion?.remove();
    return;
  }

  parentRegion.data.repeats_label_range = { start_time: lo, end_time: hi };
  repetitionLinkMode = null;
  refreshDragSelection();

  // Prefer converting the just-drawn region into the overlay so no label is created.
  if (existingRegion && !isTrimRegion(existingRegion)) {
    const oldOverlay = findLinkOverlay(parentRegion);
    if (oldOverlay && oldOverlay !== existingRegion) oldOverlay.remove();

    existingRegion.data = {
      role: "repetition-link",
      parentId: parentRegion.data.id,
    };
    existingRegion.setOptions({
      start: lo,
      end: hi,
      color: "rgba(200, 200, 210, 0.35)",
      drag: false,
      resize: false,
      content: "original",
    });
    if (typeof existingRegion.setId === "function") {
      existingRegion.setId(`link-${parentRegion.data.id}`);
    } else {
      existingRegion.id = `link-${parentRegion.data.id}`;
    }
    syncRegionVisual(existingRegion);
  } else {
    syncRepetitionLinkOverlay(parentRegion);
  }
  updateRepetitionPanel(parentRegion);
}

function updateRepetitionPanel(region) {
  const panel = document.getElementById("repetitionPanel");
  const info = document.getElementById("repetitionLinkInfo");
  if (!panel || !info) return;

  const isRepetition = region && !isTrimRegion(region) && !isLinkOverlay(region)
    && region.data?.type === "repetition";
  panel.classList.toggle("hidden", !isRepetition);
  if (!isRepetition) {
    repetitionLinkMode = null;
    return;
  }

  const range = region.data?.repeats_label_range;
  if (range) {
    info.textContent =
      `Linked original: ${formatTime(range.start_time)} – ${formatTime(range.end_time)}`;
  } else if (repetitionLinkMode === "pick") {
    info.textContent = "Double-click the original passage region on the waveform.";
  } else if (repetitionLinkMode === "draw") {
    info.textContent = "Drag on the waveform to mark the original passage.";
  } else {
    info.textContent = "Required: link the earlier range this repetition restates.";
  }
}

function regionDataToLabel(region) {
  const d = region.data || {};
  const label = {
    id: d.id || generateLabelId(),
    source: d.source || "manual",
    start_time: region.start,
    end_time: region.end,
    type: d.type || document.getElementById("labelType").value,
    severity: d.severity ?? null,
    comment: d.comment ?? null,
    deviation_cents: d.deviation_cents ?? null,
    deviation_ms: d.deviation_ms ?? null,
    measure_number: d.measure_number ?? null,
    note_id: d.note_id ?? null,
  };
  if (label.type === "repetition" && d.repeats_label_range) {
    label.repeats_label_range = {
      start_time: d.repeats_label_range.start_time,
      end_time: d.repeats_label_range.end_time,
    };
  }
  return label;
}

function setupRepetitionControls() {
  document.getElementById("linkRepetitionRegionBtn").onclick = () => {
    if (!selectedRegion || selectedRegion.data?.type !== "repetition") return;
    repetitionLinkMode = "pick";
    refreshDragSelection();
    updateRepetitionPanel(selectedRegion);
  };
  document.getElementById("drawRepetitionLinkBtn").onclick = () => {
    if (!selectedRegion || selectedRegion.data?.type !== "repetition") return;
    repetitionLinkMode = "draw";
    refreshDragSelection();
    updateRepetitionPanel(selectedRegion);
  };
  document.getElementById("clearRepetitionLinkBtn").onclick = () => {
    if (!selectedRegion || selectedRegion.data?.type !== "repetition") return;
    delete selectedRegion.data.repeats_label_range;
    repetitionLinkMode = null;
    refreshDragSelection();
    syncRepetitionLinkOverlay(selectedRegion);
    updateRepetitionPanel(selectedRegion);
  };
  document.getElementById("labelType").addEventListener("change", () => {
    refreshDragSelection();
    if (selectedRegion && !isTrimRegion(selectedRegion)) {
      if (selectedRegion.data?.type === "repetition"
        && document.getElementById("labelType").value !== "repetition") {
        delete selectedRegion.data.repeats_label_range;
        syncRepetitionLinkOverlay(selectedRegion);
      }
      updateRepetitionPanel(selectedRegion);
    }
  });
}

function clearRegionSelection() {
  regionsPlugin.getRegions().forEach((region) => {
    syncRegionVisual(region, { selected: false });
  });
}

function resetSelectionInfo() {
  document.getElementById("selectionInfo").textContent =
    "Drag on waveform to add a label. Double-click a region to edit it.";
}

function computeFitPxPerSec(duration) {
  const wrap = document.querySelector(".waveform-wrap");
  if (!wrap || !duration || duration <= 0) return 1;
  const width = wrap.clientWidth;
  if (width <= 0) return 1;
  return width / duration;
}

function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

let layoutFitTimer = null;

function fitWaveformToContainer() {
  const wrap = document.querySelector(".waveform-wrap");
  if (!wrap || !wavesurfer) return;
  const duration = wavesurfer.getDuration();
  if (!duration || duration <= 0) return;
  const width = wrap.clientWidth;
  if (width <= 0) return;
  const fitPx = width / duration;
  if (typeof wavesurfer.zoom === "function") {
    wavesurfer.zoom(fitPx);
  } else if (typeof wavesurfer.setOptions === "function") {
    wavesurfer.setOptions({ minPxPerSec: fitPx });
  }
}

function scheduleWaveformFit() {
  clearTimeout(layoutFitTimer);
  layoutFitTimer = setTimeout(() => {
    requestAnimationFrame(() => {
      fitWaveformToContainer();
      updatePlayhead();
    });
  }, 50);
}

function showScorePlaceholder(measures) {
  const container = document.getElementById("osmdContainer");
  container.innerHTML =
    `<p class="score-placeholder">Full score has ${measures} measures. ` +
    "Adjust the measure range above, then click <strong>View segment</strong>. " +
    "Use <strong>Apply &amp; regenerate</strong> to persist and regenerate reference audio.</p>";
  scheduleWaveformFit();
}

function setScoreStatus(message) {
  const container = document.getElementById("osmdContainer");
  if (message) {
    container.innerHTML = `<p class="score-placeholder">${message}</p>`;
  }
}

async function init() {
  regionsPlugin = WaveSurfer.Regions.create();
  wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#6af",
    progressColor: "#2d6cdf",
    cursorColor: "#fff",
    cursorWidth: 2,
    height: 140,
    minPxPerSec: 1,
    dragToSeek: false,
    plugins: [regionsPlugin],
  });

  wavesurfer.on("timeupdate", updatePlayhead);
  wavesurfer.on("seeking", updatePlayhead);
  wavesurfer.on("ready", onWaveformReady);
  wavesurfer.on("finish", () => {
    if (loopSelection && selectedRegion && !isTrimRegion(selectedRegion)) {
      selectedRegion.play();
    }
  });

  setupScrubber();
  setupPrepControls();
  setupBatchControls();
  setupRepetitionControls();
  window.addEventListener("resize", debounce(scheduleWaveformFit, 150));

  refreshDragSelection();
  regionsPlugin.on("region-created", (region) => {
    if (programmaticRegionDepth > 0) return;
    if (isTrimRegion(region) || isLinkOverlay(region)) return;

    // Drawing the original passage for a repetition: never create a label.
    if (
      repetitionLinkMode === "draw" &&
      selectedRegion?.data?.type === "repetition"
    ) {
      setRepetitionLinkRange(
        selectedRegion,
        region.start,
        region.end,
        region,
      );
      return;
    }

    // Loaded / already-typed regions must not be overwritten by the type dropdown.
    if (region.data?.type || region.data?.role) return;
    applyLabelToRegion(region, { comment: null });
  });
  regionsPlugin.on("region-clicked", (region, e) => {
    e.stopPropagation();
    if (isTrimRegion(region)) {
      updateTrimInfo();
      return;
    }
    if (isLinkOverlay(region)) return;

    const now = Date.now();
    const isDoubleClick =
      lastRegionClick.id === region.id &&
      now - lastRegionClick.time <= REGION_DOUBLE_CLICK_MS;

    if (
      isDoubleClick &&
      repetitionLinkMode === "pick" &&
      selectedRegion?.data?.type === "repetition" &&
      region !== selectedRegion
    ) {
      setRepetitionLinkRange(selectedRegion, region.start, region.end);
      lastRegionClick = { id: null, time: 0 };
      return;
    }

    if (isDoubleClick) {
      repetitionLinkMode = null;
      selectRegion(region);
      lastRegionClick = { id: null, time: 0 };
    } else {
      lastRegionClick = { id: region.id, time: now };
    }
  });
  regionsPlugin.on("region-updated", (region) => {
    if (isTrimRegion(region)) {
      updateTrimInfo();
      return;
    }
    if (selectedRegion === region) {
      if (region.data) {
        region.data.start_time = region.start;
        region.data.end_time = region.end;
      }
      updateSelectionInfo(region);
      syncRegionVisual(region, { selected: true });
    }
  });

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

function isTrimRegion(region) {
  return region?.data?.role === "trim" || region?.id === "trim-keep";
}

function setupBatchControls() {
  document.getElementById("runBatchBtn").onclick = runBatchRange;
  loadBatchInfo();
}

async function loadBatchInfo() {
  try {
    const res = await fetch("/api/batch/info");
    const info = await res.json();
    const el = document.getElementById("batchDefaults");
    if (info.available_audio_ids?.length) {
      const first = info.available_audio_ids[0];
      const last = info.available_audio_ids[info.available_audio_ids.length - 1];
      const firstNum = parseInt(first, 10);
      const lastNum = parseInt(last, 10);
      document.getElementById("batchFrom").value = firstNum;
      document.getElementById("batchTo").value = lastNum;
      el.textContent =
        `Score: ${info.score_path || "?"} | Audio: ${info.audio_dir || "?"} ` +
        `(${info.available_audio_ids.length} files: ${first}–${last})`;
    } else {
      el.textContent = "No raw audio files found. Set paths.raw_data_audio in config.";
    }
  } catch {
    document.getElementById("batchDefaults").textContent = "Could not load batch defaults.";
  }
}

async function runBatchRange() {
  const idFrom = parseInt(document.getElementById("batchFrom").value, 10);
  const idTo = parseInt(document.getElementById("batchTo").value, 10);
  const skipExisting = document.getElementById("batchSkipExisting").checked;
  if (!confirm(`Batch process IDs ${idFrom}–${idTo} (inclusive)?`)) return;

  const btn = document.getElementById("runBatchBtn");
  const status = document.getElementById("batchStatus");
  btn.disabled = true;
  status.textContent = "Processing…";
  try {
    const res = await fetch("/api/batch/range", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id_from: idFrom,
        id_to: idTo,
        skip_existing: skipExisting,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    status.textContent =
      `Done: ${data.succeeded} ok, ${data.skipped} skipped, ${data.failed} failed.`;
    await loadSampleList();
    if (data.results?.length) {
      const firstOk = data.results.find((r) => r.status === "ok" || r.status === "skipped");
      if (firstOk) {
        document.getElementById("sampleSelect").value = firstOk.sample_id;
        await loadSample(firstOk.sample_id);
      }
    }
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
}

function getMeasureRange() {
  const start = parseInt(document.getElementById("measureStart").value, 10);
  const end = parseInt(document.getElementById("measureEnd").value, 10);
  const startBeat = parseBeatInput("startBeat") ?? 1;
  const endBeat = parseBeatInput("endBeat");
  return { start, end, startBeat, endBeat };
}

function buildScoreSegmentQuery({ start, end, startBeat, endBeat }) {
  const params = new URLSearchParams({
    start_measure: String(start),
    end_measure: String(end),
    start_beat: String(startBeat ?? 1),
  });
  if (endBeat != null) {
    params.set("end_beat", String(endBeat));
  }
  return params.toString();
}

function setupPrepControls() {
  document.getElementById("applySegmentBtn").onclick = applyScoreSegment;
  document.getElementById("applyTrimBtn").onclick = applyPerformanceTrim;
  document.getElementById("viewFullScoreBtn").onclick = () => {
    if (!sampleData?.full_score_url) return;
    if ((sampleData.prep?.total_measures || 0) > LARGE_SCORE_MEASURES) {
      if (!confirm("Rendering the full score may be slow and can affect layout. Continue?")) return;
    }
    renderScore(sampleData.full_score_url, { mode: "full" });
  };
  document.getElementById("viewSegmentScoreBtn").onclick = () => viewScoreSegment();
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

let playheadRaf = null;

function updatePlayhead() {
  if (playheadRaf) return;
  playheadRaf = requestAnimationFrame(() => {
    playheadRaf = null;
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
  });
}

function onWaveformReady() {
  fitWaveformToContainer();
  withProgrammaticRegions(() => {
    pendingRegions.forEach(({ label, isCandidate }) => addRegion(label, isCandidate));
    pendingRegions = [];
    ensureTrimRegion();
  });
  updatePlayhead();
  scheduleWaveformFit();
}

function ensureTrimRegion() {
  const duration = wavesurfer.getDuration();
  if (!duration) return;

  if (trimRegion) {
    trimRegion.remove();
    trimRegion = null;
  }

  const createTrim = () =>
    regionsPlugin.addRegion({
      start: 0,
      end: duration,
      color: "rgba(60, 200, 60, 0.15)",
      drag: false,
      resize: true,
      content: "keep",
      id: "trim-keep",
      data: { role: "trim" },
    });

  trimRegion =
    programmaticRegionDepth > 0 ? createTrim() : withProgrammaticRegions(createTrim);
  if (trimRegion) {
    trimRegion.data = { ...(trimRegion.data || {}), role: "trim" };
  }
  syncRegionVisual(trimRegion);
  updateTrimInfo();
}

function updateTrimInfo() {
  if (!trimRegion) return;
  const el = document.getElementById("trimInfo");
  el.textContent =
    `Keep ${formatTime(trimRegion.start)} – ${formatTime(trimRegion.end)} ` +
    `(discards ${formatTime(trimRegion.start)} from start, ` +
    `${formatTime(Math.max(0, wavesurfer.getDuration() - trimRegion.end))} from end)`;
}

function updateMeasureControls(prep) {
  const total = prep?.total_measures || 1;
  const beatsPerMeasure = prep?.beats_per_measure || 4;
  const { start, end, startBeat, endBeat } = defaultMeasureRange(prep);
  const startInput = document.getElementById("measureStart");
  const endInput = document.getElementById("measureEnd");
  const startBeatInput = document.getElementById("startBeat");
  const endBeatInput = document.getElementById("endBeat");
  if (!startInput || !endInput || !startBeatInput || !endBeatInput) return;
  startInput.max = total;
  endInput.max = total;
  startBeatInput.max = beatsPerMeasure;
  endBeatInput.max = beatsPerMeasure;
  startInput.value = start;
  endInput.value = end;
  startBeatInput.value = startBeat;
  endBeatInput.value = endBeat ?? "";

  const perf = prep?.performance_duration;
  const rangeLabel = formatSegmentLabel(start, end, startBeat, endBeat);
  const hint = perf
    ? `(${total} measures in full score; performance ~${perf.toFixed(1)}s — suggested ${rangeLabel})`
    : `(${total} measures in full score; suggested ${rangeLabel})`;
  document.getElementById("measureTotal").textContent = hint;
}

async function loadSampleList() {
  const res = await fetch("/api/samples");
  const samples = await res.json();
  const sel = document.getElementById("sampleSelect");
  sel.innerHTML = samples
    .map((s) => {
      const tags = [];
      if (s.label_count) tags.push(`${s.label_count} labels`);
      else if (s.has_candidates) tags.push("needs review");
      const suffix = tags.length ? ` (${tags.join(", ")})` : "";
      return `<option value="${s.id}">${s.id}${suffix}</option>`;
    })
    .join("");
  sel.onchange = () => loadSample(sel.value);
  if (samples.length) {
    const current = currentSample && samples.some((s) => s.id === currentSample)
      ? currentSample
      : samples[0].id;
    sel.value = current;
    await loadSample(current);
  }
}

async function loadSample(sampleId) {
  currentSample = sampleId;
  const res = await fetch(`/api/samples/${sampleId}`);
  const data = await res.json();
  sampleData = data;
  taxonomy = data.taxonomy;
  populateTypeSelect();
  document.getElementById("annotatorId").value = data.annotator_id || "";
  updateMeasureControls(data.prep);

  trimRegion = null;
  selectedRegion = null;
  repetitionLinkMode = null;
  lastRegionClick = { id: null, time: 0 };
  resetSelectionInfo();
  regionsPlugin.clearRegions();
  pendingRegions = [
    ...data.candidates.map((c) => ({ label: normalizeLabelType(c), isCandidate: true })),
    ...data.labels.map((l) => ({ label: normalizeLabelType(l), isCandidate: false })),
  ];

  const duration = data.prep?.performance_duration || 0;
  if (typeof wavesurfer.setOptions === "function") {
    wavesurfer.setOptions({ minPxPerSec: computeFitPxPerSec(duration) });
  }

  const audioUrl = audioUrlWithCacheBust(data.audio_url, data.audio_mtime);
  await wavesurfer.load(audioUrl);
  scheduleWaveformFit();

  const measures = data.prep?.total_measures || 0;
  const hasSegment = !!data.prep?.score_segment;
  if (hasSegment) {
    const seg = data.prep.score_segment;
    await viewScoreSegment(
      seg.start_measure,
      seg.end_measure,
      seg.start_beat || 1,
      seg.end_beat ?? null,
    );
  } else if (measures > LARGE_SCORE_MEASURES) {
    showScorePlaceholder(measures);
  } else {
    await renderScore(data.full_score_url || data.score_url, { mode: "full" });
  }
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
  sel.onchange = () => refreshDragSelection();
  refreshDragSelection();
}

function addRegion(label, isCandidate) {
  const color = TYPE_COLORS[label.type] || "rgba(45,108,223,0.35)";
  const display = TYPE_LABELS[label.type] || label.type;
  const create = () =>
    regionsPlugin.addRegion({
      start: label.start_time,
      end: label.end_time,
      color,
      drag: true,
      resize: true,
      content: display.split(" (")[0],
      data: {
        ...label,
        isCandidate,
        repeats_label_range: label.repeats_label_range || null,
      },
    });
  const region =
    programmaticRegionDepth > 0 ? create() : withProgrammaticRegions(create);
  // Re-assert label data in case the plugin event path mutated it.
  region.data = {
    ...(region.data || {}),
    ...label,
    isCandidate,
    type: label.type,
    repeats_label_range: label.repeats_label_range || null,
  };
  region.setOptions({
    content: display.split(" (")[0],
    color,
  });
  syncRegionVisual(region, { selected: selectedRegion === region });
  if (label.type === "repetition" && label.repeats_label_range) {
    syncRepetitionLinkOverlay(region);
  }
  return region;
}

function updateSelectionInfo(region) {
  const d = region.data || {};
  const typeLabel = TYPE_LABELS[d.type] || d.type || "unlabeled";
  document.getElementById("selectionInfo").textContent =
    `${formatTime(region.start)} – ${formatTime(region.end)} (${typeLabel})`;
}

function selectRegion(region) {
  if (isTrimRegion(region)) return;
  clearRegionSelection();
  selectedRegion = region;
  syncRegionVisual(region, { selected: true });
  requestAnimationFrame(() => {
    if (selectedRegion === region) {
      syncRegionVisual(region, { selected: true });
    }
  });
  const d = region.data || {};
  updateSelectionInfo(region);
  if (d.type) document.getElementById("labelType").value = d.type;
  if (d.severity) document.getElementById("severity").value = d.severity;
  document.getElementById("comment").value = d.comment || "";
  updateRepetitionPanel(region);
}

function applyLabelToRegion(region, overrides = {}) {
  if (!region || isTrimRegion(region) || isLinkOverlay(region)) return;
  const type =
    overrides.type ||
    region.data?.type ||
    document.getElementById("labelType").value;
  const severity =
    overrides.severity ??
    region.data?.severity ??
    parseInt(document.getElementById("severity").value, 10);
  const comment = "comment" in overrides
    ? overrides.comment
    : (region.data?.comment ?? document.getElementById("comment").value || null);
  region.data = {
    ...(region.data || {}),
    id: (region.data && region.data.id) || generateLabelId(),
    source: (region.data && region.data.source) || "manual",
    start_time: region.start,
    end_time: region.end,
    type,
    severity,
    comment,
  };
  if (type !== "repetition") {
    delete region.data.repeats_label_range;
    syncRepetitionLinkOverlay(region);
  }
  const display = (TYPE_LABELS[type] || type).split(" (")[0];
  region.setOptions({
    content: display,
    color: TYPE_COLORS[type] || region.color,
  });
  if (selectedRegion === region) {
    updateSelectionInfo(region);
    updateRepetitionPanel(region);
  }
  syncRegionVisual(region, { selected: selectedRegion === region });
}

function applyLabelToSelection() {
  if (!selectedRegion || isTrimRegion(selectedRegion)) return;
  applyLabelToRegion(selectedRegion);
}

function promoteCandidate(source, overrideType) {
  if (!selectedRegion || !selectedRegion.data?.isCandidate) return;
  selectedRegion.data.source = source;
  if (overrideType) selectedRegion.data.type = overrideType;
  selectedRegion.data.isCandidate = false;
  applyLabelToSelection();
}

function deleteSelectedRegion() {
  if (selectedRegion && !isTrimRegion(selectedRegion)) {
    findLinkOverlay(selectedRegion)?.remove();
    selectedRegion.remove();
    selectedRegion = null;
    repetitionLinkMode = null;
    resetSelectionInfo();
    updateRepetitionPanel(null);
  }
}

async function applyScoreSegment() {
  const { start, end, startBeat, endBeat } = getMeasureRange();
  const rangeLabel = formatSegmentLabel(start, end, startBeat, endBeat);
  if (!confirm(
    `Extract ${rangeLabel} and regenerate reference audio + alignment? ` +
    "Existing auto-candidates will be replaced."
  )) return;

  const btn = document.getElementById("applySegmentBtn");
  btn.disabled = true;
  btn.textContent = "Processing…";
  try {
    const payload = {
      start_measure: start,
      end_measure: end,
      start_beat: startBeat,
    };
    if (endBeat != null) {
      payload.end_beat = endBeat;
    }
    const res = await fetch(`/api/samples/${currentSample}/score-segment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    await loadSample(currentSample);
    alert("Score segment applied. Reference audio and candidates updated.");
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Apply & regenerate";
  }
}

async function applyPerformanceTrim() {
  if (!trimRegion) return;
  if (!confirm(
    `Trim performance to ${formatTime(trimRegion.start)} – ${formatTime(trimRegion.end)} ` +
    "and re-run alignment? This may take up to a minute for long scores. Existing auto-candidates will be replaced."
  )) return;

  const btn = document.getElementById("applyTrimBtn");
  btn.disabled = true;
  btn.textContent = "Processing…";
  try {
    const res = await fetch(`/api/samples/${currentSample}/trim-performance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trim_start: trimRegion.start,
        trim_end: trimRegion.end,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const result = await res.json();
    await loadSample(currentSample);
    alert(
      `Performance trimmed to ${result.performance_trim?.trimmed_duration?.toFixed?.(1) ?? "?"}s. ` +
      "Waveform and alignment updated."
    );
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Apply trim & re-align";
  }
}

async function saveLabels() {
  const labels = [];
  const candidatesLeft = [];
  regionsPlugin.getRegions().forEach((r) => {
    if (isTrimRegion(r) || isLinkOverlay(r)) return;
    const label = regionDataToLabel(r);
    if (r.data?.isCandidate) candidatesLeft.push(label);
    else labels.push(label);
  });
  if (candidatesLeft.length) {
    alert(`${candidatesLeft.length} candidates still unreviewed. Confirm or reject them before saving.`);
    return;
  }
  const missingRepetitionLink = labels.filter(
    (label) => label.type === "repetition" && !label.repeats_label_range,
  );
  if (missingRepetitionLink.length) {
    alert(
      `${missingRepetitionLink.length} repetition label(s) need an original passage link. ` +
      "Select each repetition, then use Link to region or Draw original range.",
    );
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

async function viewScoreSegment(startMeasure, endMeasure, startBeat, endBeat) {
  const range =
    startMeasure != null && endMeasure != null
      ? {
          start: startMeasure,
          end: endMeasure,
          startBeat: startBeat ?? 1,
          endBeat: endBeat ?? null,
        }
      : getMeasureRange();
  const { start, end } = range;
  if (!currentSample || Number.isNaN(start) || Number.isNaN(end)) return;

  const span = end - start + 1;
  if (span > MAX_SEGMENT_PREVIEW_MEASURES) {
    alert(
      `Selected range spans ${span} measures (max ${MAX_SEGMENT_PREVIEW_MEASURES} for preview). ` +
      "Narrow the measure range to match your recording."
    );
    return;
  }

  const rangeLabel = formatSegmentLabel(
    start,
    end,
    range.startBeat,
    range.endBeat,
  );
  setScoreStatus(`Rendering ${rangeLabel}…`);
  const url =
    `/api/samples/${currentSample}/score-preview?${buildScoreSegmentQuery(range)}`;
  const res = await fetch(url);
  if (!res.ok) {
    alert(await res.text());
    setScoreStatus("Could not load score segment.");
    return;
  }
  const xml = await res.text();
  await renderScoreXml(xml, {
    mode: "segment",
    start,
    end,
    startBeat: range.startBeat,
    endBeat: range.endBeat,
    xmlBytes: xml.length,
  });
}

async function renderScore(url, meta = {}) {
  setScoreStatus("Loading score…");
  const res = await fetch(url);
  if (!res.ok) {
    alert(await res.text());
    setScoreStatus("Could not load score.");
    return;
  }
  const xml = await res.text();
  await renderScoreXml(xml, { ...meta, url, xmlBytes: xml.length });
}

async function renderScoreXml(xml, meta = {}) {
  const container = document.getElementById("osmdContainer");
  container.innerHTML = "";
  osmdInstance = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
    autoResize: false,
    drawTitle: true,
  });
  await osmdInstance.load(xml);
  const pageWidth = Math.max(400, container.clientWidth - 24);
  if (osmdInstance.EngravingRules) {
    osmdInstance.EngravingRules.PageWidth = pageWidth;
  }
  osmdInstance.render();
  if (meta.mode === "segment" && meta.start != null && meta.end != null) {
    container.dataset.segment = `${meta.start}-${meta.end}`;
  }
  scheduleWaveformFit();
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
    refreshDragSelection();
    if (selectedRegion && !isTrimRegion(selectedRegion)) {
      applyLabelToSelection();
    }
  }
  if (!selectedRegion || isTrimRegion(selectedRegion)) return;
  const step = 0.01;
  if (e.key === "ArrowLeft") selectedRegion.setOptions({ start: Math.max(0, selectedRegion.start - step) });
  if (e.key === "ArrowRight") selectedRegion.setOptions({ end: selectedRegion.end + step });
}

init();
