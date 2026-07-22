let wavesurfer;
let regionsPlugin;
let osmdInstance = null;
let taxonomy = [];
let currentSample = null;
let sampleData = null;
let selectedRegions = [];
/** Sole selection when exactly one region is selected; null when empty or multi. */
let selectedRegion = null;
let trimRegion = null;
let loopSelection = false;
let scrubbing = false;
let marqueeSelecting = false;
let pendingRegions = [];
let lastRegionClick = { id: null, time: 0 };
const REGION_DOUBLE_CLICK_MS = 400;
let repetitionLinkMode = null; // null | "pick" | "draw"
/** >0 while addRegion is called from code (labels, trim, link overlays) — skip auto-label. */
let programmaticRegionDepth = 0;
let viewMode = "normal"; // normal | alignment
let zoomFactor = 1;
let fitPxPerSec = 1;
let userZoomed = false;
let noteAlignmentData = null;
let labelsVisible = true;
let staffStripHeight = 0;
const SCRUBBER_HEIGHT = 28;
const EWMA_STRIP_HEIGHT = 72;
const EWMA_ALPHA = 0.3;
const ZOOM_STEP = 1.25;
const MIN_ZOOM_FACTOR = 1;
const MAX_ZOOM_FACTOR = 32;
let scoreLoadId = 0;
let cachedScoreXml = null;
let cachedScoreMeta = null;

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
    el.style.zIndex = selected ? "10" : "5";
  } else {
    el.style.border = "none";
    el.style.boxShadow = "";
    el.style.zIndex = "";
  }
  applyRegionLabelVisibility(region);
}

function applyRegionLabelVisibility(region) {
  if (!region?.element) return;
  if (isTrimRegion(region) || isLinkOverlay(region)) {
    region.element.style.visibility = "visible";
    return;
  }
  region.element.style.visibility = labelsVisible ? "visible" : "hidden";
}

function applyAllLabelsVisibility() {
  if (!regionsPlugin) return;
  regionsPlugin.getRegions().forEach((region) => applyRegionLabelVisibility(region));
}

function toggleLabelsVisibility() {
  labelsVisible = !labelsVisible;
  const btn = document.getElementById("toggleLabelsBtn");
  if (btn) {
    btn.classList.toggle("active", labelsVisible);
    btn.textContent = labelsVisible ? "Labels" : "Labels (off)";
  }
  applyAllLabelsVisibility();
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
  selectedRegions = [];
  selectedRegion = null;
  if (!regionsPlugin) return;
  regionsPlugin.getRegions().forEach((region) => {
    syncRegionVisual(region, { selected: false });
  });
}

function syncPrimarySelection() {
  selectedRegion = selectedRegions.length === 1 ? selectedRegions[0] : null;
}

function isRegionSelected(region) {
  return selectedRegions.includes(region);
}

function refreshSelectionVisuals() {
  if (!regionsPlugin) return;
  regionsPlugin.getRegions().forEach((region) => {
    syncRegionVisual(region, { selected: isRegionSelected(region) });
  });
}

function updateMultiSelectionInspector() {
  const confirmBtn = document.getElementById("confirmCandidateBtn");
  const rejectBtn = document.getElementById("rejectCandidateBtn");
  const n = selectedRegions.length;
  if (n === 0) {
    resetSelectionInfo();
    updateRepetitionPanel(null);
    if (confirmBtn) confirmBtn.disabled = false;
    if (rejectBtn) rejectBtn.disabled = false;
    return;
  }
  if (n > 1) {
    document.getElementById("selectionInfo").textContent =
      `${n} regions selected. Delete or assign type (1–8) applies to all.`;
    updateRepetitionPanel(null);
    if (confirmBtn) confirmBtn.disabled = true;
    if (rejectBtn) rejectBtn.disabled = true;
    return;
  }
  if (confirmBtn) confirmBtn.disabled = false;
  if (rejectBtn) rejectBtn.disabled = false;
  const region = selectedRegions[0];
  const d = region.data || {};
  updateSelectionInfo(region);
  if (d.type) document.getElementById("labelType").value = d.type;
  if (d.severity) document.getElementById("severity").value = d.severity;
  document.getElementById("comment").value = d.comment || "";
  updateRepetitionPanel(region);
}

function selectRegion(region) {
  if (isTrimRegion(region) || isLinkOverlay(region)) return;
  clearRegionSelection();
  selectedRegions = [region];
  syncPrimarySelection();
  syncRegionVisual(region, { selected: true });
  requestAnimationFrame(() => {
    if (isRegionSelected(region)) {
      syncRegionVisual(region, { selected: true });
    }
  });
  updateMultiSelectionInspector();
}

function toggleRegionInSelection(region) {
  if (isTrimRegion(region) || isLinkOverlay(region)) return;
  const idx = selectedRegions.indexOf(region);
  if (idx >= 0) {
    selectedRegions.splice(idx, 1);
    syncRegionVisual(region, { selected: false });
  } else {
    selectedRegions.push(region);
    syncRegionVisual(region, { selected: true });
  }
  syncPrimarySelection();
  updateMultiSelectionInspector();
}

function setSelectedRegions(regions) {
  const next = regions.filter((r) => r && !isTrimRegion(r) && !isLinkOverlay(r));
  selectedRegions = next;
  syncPrimarySelection();
  refreshSelectionVisuals();
  updateMultiSelectionInspector();
}

function resetSelectionInfo() {
  document.getElementById("selectionInfo").textContent =
    "Drag on waveform to add a label. Double-click a region to edit it. Ctrl+drag to multi-select.";
}

function getScrollContainerWidth() {
  const scroll = document.getElementById("waveformScroll");
  if (scroll && scroll.clientWidth > 0) return scroll.clientWidth;
  const wrap = document.querySelector(".waveform-wrap");
  return wrap?.clientWidth || 800;
}

function computeFitPxPerSec(duration) {
  if (!duration || duration <= 0) return 1;
  return getScrollContainerWidth() / duration;
}

function getCurrentPxPerSec() {
  return fitPxPerSec * zoomFactor;
}

/** WaveSurfer does not shrink below fit; clamp overlays to the same scale. */
function getEffectivePxPerSec() {
  return Math.max(fitPxPerSec, getCurrentPxPerSec());
}

function contentXFromClientX(clientX) {
  const scroll = document.getElementById("waveformScroll");
  if (!scroll) return 0;
  const rect = scroll.getBoundingClientRect();
  return clientX - rect.left + scroll.scrollLeft;
}

function applyWaveformZoom(pxPerSec, anchorClientX = null) {
  if (!wavesurfer) return;
  const scroll = document.getElementById("waveformScroll");
  const prevPx = getEffectivePxPerSec();
  const prevScrollLeft = scroll?.scrollLeft || 0;

  let anchorTime = 0;
  let viewOffset = 0;
  if (scroll && prevPx > 0) {
    if (anchorClientX != null) {
      const rect = scroll.getBoundingClientRect();
      viewOffset = Math.max(0, Math.min(scroll.clientWidth, anchorClientX - rect.left));
      anchorTime = (prevScrollLeft + viewOffset) / prevPx;
    } else {
      anchorTime = prevScrollLeft / prevPx;
      viewOffset = 0;
    }
  }

  const effective = Math.max(fitPxPerSec, pxPerSec);
  if (typeof wavesurfer.zoom === "function") {
    wavesurfer.zoom(effective);
  } else if (typeof wavesurfer.setOptions === "function") {
    wavesurfer.setOptions({ minPxPerSec: effective });
  }
  syncAlignmentStackWidth(effective);
  if (viewMode === "alignment" && noteAlignmentData) {
    renderAlignmentOverlays();
  }

  if (scroll) {
    const maxScroll = Math.max(0, scroll.scrollWidth - scroll.clientWidth);
    scroll.scrollLeft = Math.max(0, Math.min(anchorTime * effective - viewOffset, maxScroll));
  }
  updatePlayhead();
}

function syncAlignmentStackWidth(pxPerSec) {
  const stack = document.getElementById("alignmentStack");
  const duration = wavesurfer?.getDuration() || 0;
  if (!stack || !duration) return;
  const width = Math.max(getScrollContainerWidth(), duration * pxPerSec);
  stack.style.width = `${width}px`;
  stack.style.minWidth = `${width}px`;
  const scrubber = document.getElementById("scrubber");
  if (scrubber) scrubber.style.width = `${width}px`;
  const ewma = document.getElementById("ewmaStrip");
  if (ewma) ewma.style.width = `${width}px`;
  updateOverlayTop();
}

function updateOverlayTop() {
  const stack = document.getElementById("alignmentStack");
  // In alignment mode, boundaries span staff + scrubber + waveform + ewma.
  const top = viewMode === "alignment" ? 0 : SCRUBBER_HEIGHT;
  if (stack) stack.style.setProperty("--overlay-top", `${top}px`);
  const boundaries = document.getElementById("noteBoundaries");
  if (boundaries) boundaries.style.top = `${top}px`;
}

function zoomFactorToSlider(factor) {
  const logMin = Math.log(MIN_ZOOM_FACTOR);
  const logMax = Math.log(MAX_ZOOM_FACTOR);
  const logVal = Math.log(Math.max(MIN_ZOOM_FACTOR, factor));
  return Math.round(100 * (logVal - logMin) / (logMax - logMin));
}

function sliderToZoomFactor(sliderVal) {
  const t = sliderVal / 100;
  const logMin = Math.log(MIN_ZOOM_FACTOR);
  const logMax = Math.log(MAX_ZOOM_FACTOR);
  return Math.exp(logMin + t * (logMax - logMin));
}

function syncZoomSlider() {
  const slider = document.getElementById("zoomSlider");
  if (slider) slider.value = String(zoomFactorToSlider(zoomFactor));
}

function fitWaveformToContainer() {
  if (!wavesurfer) return;
  const duration = wavesurfer.getDuration();
  if (!duration || duration <= 0) return;
  fitPxPerSec = computeFitPxPerSec(duration);
  if (!userZoomed) {
    zoomFactor = 1;
  }
  applyWaveformZoom(getCurrentPxPerSec());
  syncZoomSlider();
}

function setZoomFactor(factor, anchorClientX = null) {
  zoomFactor = Math.max(MIN_ZOOM_FACTOR, Math.min(MAX_ZOOM_FACTOR, factor));
  userZoomed = true;
  applyWaveformZoom(getCurrentPxPerSec(), anchorClientX);
  syncZoomSlider();
}

function zoomIn(anchorClientX = null) {
  setZoomFactor(zoomFactor * ZOOM_STEP, anchorClientX);
}

function zoomOut(anchorClientX = null) {
  setZoomFactor(zoomFactor / ZOOM_STEP, anchorClientX);
}

function zoomFit() {
  userZoomed = false;
  fitWaveformToContainer();
}

function setupWaveformWheel() {
  const scroll = document.getElementById("waveformScroll");
  if (!scroll) return;
  scroll.addEventListener(
    "wheel",
    (e) => {
      if (!wavesurfer) return;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        if (e.deltaY < 0) zoomIn(e.clientX);
        else zoomOut(e.clientX);
        return;
      }
      const delta = e.shiftKey ? e.deltaY : e.deltaY || e.deltaX;
      if (delta !== 0) {
        e.preventDefault();
        scroll.scrollLeft += delta;
      }
    },
    { passive: false },
  );
}

function setupMarqueeSelect() {
  const scroll = document.getElementById("waveformScroll");
  const stack = document.getElementById("alignmentStack");
  if (!scroll || !stack) return;

  let startX = 0;
  let clientStartX = 0;
  let clientStartY = 0;
  let box = null;

  const stackXFromClient = (clientX) => contentXFromClientX(clientX);

  const finishMarquee = (e) => {
    if (!marqueeSelecting) return;
    marqueeSelecting = false;
    const endX = stackXFromClient(e.clientX);
    const x0 = Math.min(startX, endX);
    const x1 = Math.max(startX, endX);
    if (box) {
      box.remove();
      box = null;
    }
    refreshDragSelection();

    const moved = Math.hypot(e.clientX - clientStartX, e.clientY - clientStartY);
    if (moved < 6) return;

    const pxPerSec = getEffectivePxPerSec();
    const t0 = x0 / pxPerSec;
    const t1 = x1 / pxPerSec;
    const hits = regionsPlugin.getRegions().filter((r) => {
      if (isTrimRegion(r) || isLinkOverlay(r)) return false;
      return r.end > t0 && r.start < t1;
    });
    setSelectedRegions(hits);
  };

  scroll.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !(e.ctrlKey || e.metaKey)) return;
    if (e.target.closest("#scrubber")) return;
    if (isEditableKeyTarget(e.target)) return;
    e.preventDefault();
    e.stopPropagation();
    marqueeSelecting = true;
    startX = stackXFromClient(e.clientX);
    clientStartX = e.clientX;
    clientStartY = e.clientY;
    if (typeof regionsPlugin.disableDragSelection === "function") {
      regionsPlugin.disableDragSelection();
    }
    box = document.createElement("div");
    box.className = "marquee-select";
    box.style.left = `${startX}px`;
    box.style.width = "0px";
    stack.appendChild(box);
    scroll.setPointerCapture(e.pointerId);
  });

  scroll.addEventListener("pointermove", (e) => {
    if (!marqueeSelecting || !box) return;
    const x = stackXFromClient(e.clientX);
    const left = Math.min(startX, x);
    const width = Math.abs(x - startX);
    box.style.left = `${left}px`;
    box.style.width = `${width}px`;
  });

  scroll.addEventListener("pointerup", finishMarquee);
  scroll.addEventListener("pointercancel", (e) => {
    if (!marqueeSelecting) return;
    marqueeSelecting = false;
    if (box) {
      box.remove();
      box = null;
    }
    refreshDragSelection();
  });
}

function setupViewControls() {
  const slider = document.getElementById("zoomSlider");
  slider.addEventListener("input", () => {
    setZoomFactor(sliderToZoomFactor(parseInt(slider.value, 10)));
  });
  document.getElementById("zoomFitBtn").onclick = zoomFit;
  document.getElementById("viewNormalBtn").onclick = () => setViewMode("normal");
  document.getElementById("viewAlignmentBtn").onclick = () => setViewMode("alignment");
  document.getElementById("toggleLabelsBtn").onclick = toggleLabelsVisibility;
  document.getElementById("realignBtn").onclick = rerunAlignment;
  syncZoomSlider();
}

async function setViewMode(mode) {
  viewMode = mode;
  document.getElementById("viewNormalBtn").classList.toggle("active", mode === "normal");
  document.getElementById("viewAlignmentBtn").classList.toggle("active", mode === "alignment");
  document.getElementById("scorePanel").classList.toggle("collapsed", mode === "alignment");
  document.getElementById("staffStrip").classList.toggle("hidden", mode !== "alignment");
  document.getElementById("noteBoundaries").classList.toggle("hidden", mode !== "alignment");
  document.getElementById("ewmaStrip")?.classList.toggle("hidden", mode !== "alignment");
  document.getElementById("alignmentInfo").classList.toggle("hidden", mode !== "alignment");
  if (mode === "alignment") {
    await loadNoteAlignment();
    renderAlignmentOverlays();
  } else {
    clearAlignmentOverlays();
    // Re-render score now that the panel is visible (OSMD needs real width).
    if (cachedScoreXml) {
      requestAnimationFrame(() => {
        renderScoreXml(cachedScoreXml, cachedScoreMeta || {});
      });
    }
  }
  updateOverlayTop();
  scheduleWaveformFit();
}

async function loadNoteAlignment() {
  if (!currentSample) return;
  try {
    const res = await fetch(`/api/samples/${currentSample}/note-alignment`);
    if (!res.ok) throw new Error(await res.text());
    noteAlignmentData = await res.json();
    renderAlignmentInfo(noteAlignmentData);
  } catch (err) {
    noteAlignmentData = null;
    const info = document.getElementById("alignmentInfo");
    info.classList.remove("hidden");
    info.innerHTML = `<p class="summary">Could not load alignment: ${err.message}</p>`;
  }
}

function clearAlignmentOverlays() {
  document.getElementById("staffStrip").innerHTML = "";
  document.getElementById("noteBoundaries").innerHTML = "";
  const ewma = document.getElementById("ewmaStrip");
  if (ewma) ewma.innerHTML = "";
  staffStripHeight = 0;
  updateOverlayTop();
}

function renderAlignmentOverlays() {
  if (!noteAlignmentData?.events?.length) {
    clearAlignmentOverlays();
    return;
  }
  const pxPerSec = getEffectivePxPerSec();
  renderStaffStrip(noteAlignmentData.events, pxPerSec);
  renderNoteBoundaries(noteAlignmentData.events, pxPerSec);
  renderEwmaStrip(noteAlignmentData.events, pxPerSec);
}

function renderNoteBoundaries(events, pxPerSec) {
  const container = document.getElementById("noteBoundaries");
  container.innerHTML = "";
  const seen = new Set();
  events.forEach((ev) => {
    [
      { t: ev.perf_start, cls: "start" },
      { t: ev.perf_end, cls: "end" },
    ].forEach(({ t, cls }) => {
      const key = `${cls}:${t.toFixed(4)}`;
      if (seen.has(key)) return;
      seen.add(key);
      const line = document.createElement("div");
      line.className = `note-boundary-line ${cls}`;
      line.style.left = `${t * pxPerSec}px`;
      container.appendChild(line);
    });
  });
}

const DIATONIC_STEPS_FROM_E = {
  0: 2.5, 1: 2.5, 2: 3, 3: 3, 4: 0, 5: 0.5, 6: 0.5, 7: 1, 8: 1, 9: 1.5, 10: 1.5, 11: 2,
};

function midiToStaffSteps(midi) {
  const octave = Math.floor(midi / 12) - 1;
  const pc = midi % 12;
  const octaveBase = pc < 4 ? octave - 1 : octave;
  return (octaveBase - 4) * 3.5 + DIATONIC_STEPS_FROM_E[pc];
}

function midiToStaffY(midi, staffBottomY, lineGap) {
  const steps = midiToStaffSteps(midi);
  return staffBottomY - steps * lineGap;
}

function classifyDurationQl(ql) {
  const bases = [4, 2, 1, 0.5, 0.25, 0.125, 0.0625];
  if (!(ql > 0)) return { base: 1, dots: 0 };
  for (const b of bases) {
    if (Math.abs(ql - b) < 0.04) return { base: b, dots: 0 };
    if (Math.abs(ql - b * 1.5) < 0.04) return { base: b, dots: 1 };
    if (Math.abs(ql - b * 1.75) < 0.05) return { base: b, dots: 2 };
  }
  let best = 1;
  let bestDist = Infinity;
  for (const b of bases) {
    const d = Math.abs(ql - b);
    if (d < bestDist) {
      bestDist = d;
      best = b;
    }
  }
  return { base: best, dots: 0 };
}

function appendDurationDots(svgParts, cx, cy, dots) {
  for (let i = 0; i < dots; i += 1) {
    svgParts.push(
      `<circle cx="${cx + 10 + i * 5}" cy="${cy}" r="1.6" fill="#222"/>`,
    );
  }
}

function appendNoteGlyph(svgParts, cx, cy, ql, stemUp) {
  const { base, dots } = classifyDurationQl(ql);
  const open = base >= 2;
  const rx = base >= 4 ? 7 : 5;
  const ry = base >= 4 ? 5 : 4;
  svgParts.push(
    `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" ` +
      `fill="${open ? "none" : "#222"}" stroke="#222" stroke-width="1.5" ` +
      `transform="rotate(-20 ${cx} ${cy})"/>`,
  );
  if (base <= 2) {
    const stemH = 22;
    const sx = stemUp ? cx + rx - 1 : cx - rx + 1;
    const sy1 = cy;
    const sy2 = stemUp ? cy - stemH : cy + stemH;
    svgParts.push(
      `<line x1="${sx}" y1="${sy1}" x2="${sx}" y2="${sy2}" stroke="#222" stroke-width="1.4"/>`,
    );
    let flags = 0;
    if (base <= 0.5) flags = 1;
    if (base <= 0.25) flags = 2;
    if (base <= 0.125) flags = 3;
    for (let f = 0; f < flags; f += 1) {
      const fy = stemUp ? sy2 + f * 5 : sy2 - f * 5;
      const tipY = stemUp ? fy + 8 : fy - 8;
      const tipX = stemUp ? sx + 9 : sx - 9;
      svgParts.push(
        `<path d="M${sx} ${fy} Q${sx + (stemUp ? 6 : -6)} ${fy + (stemUp ? 3 : -3)} ${tipX} ${tipY}" ` +
          `fill="none" stroke="#222" stroke-width="1.4"/>`,
      );
    }
  }
  appendDurationDots(svgParts, cx + (base >= 4 ? 4 : 0), cy, dots);
}

function appendRestGlyph(svgParts, cx, staffBottomY, lineGap, ql) {
  const { base, dots } = classifyDurationQl(ql);
  const mid = staffBottomY - 2 * lineGap;
  if (base >= 4) {
    // whole rest: hang from 2nd line from top
    const y = staffBottomY - 3 * lineGap;
    svgParts.push(`<rect x="${cx - 6}" y="${y}" width="12" height="4" fill="#333"/>`);
  } else if (base >= 2) {
    const y = staffBottomY - 2 * lineGap - 4;
    svgParts.push(`<rect x="${cx - 6}" y="${y}" width="12" height="4" fill="#333"/>`);
  } else if (base >= 1) {
    // quarter rest (simplified zigzag)
    svgParts.push(
      `<path d="M${cx - 2} ${mid - 10} L${cx + 3} ${mid - 4} L${cx - 3} ${mid + 2} L${cx + 2} ${mid + 10}" ` +
        `fill="none" stroke="#333" stroke-width="1.8" stroke-linejoin="round"/>`,
    );
  } else {
    // eighth / shorter: flag rest
    const flags = base <= 0.125 ? 3 : base <= 0.25 ? 2 : 1;
    svgParts.push(
      `<line x1="${cx}" y1="${mid - 10}" x2="${cx}" y2="${mid + 8}" stroke="#333" stroke-width="1.5"/>`,
    );
    for (let f = 0; f < flags; f += 1) {
      const fy = mid - 10 + f * 6;
      svgParts.push(
        `<path d="M${cx} ${fy} q6 2 8 7" fill="none" stroke="#333" stroke-width="1.5"/>`,
        `<circle cx="${cx + 8}" cy="${fy + 8}" r="2" fill="#333"/>`,
      );
    }
  }
  appendDurationDots(svgParts, cx + 4, mid, dots);
}

function renderStaffStrip(events, pxPerSec) {
  const container = document.getElementById("staffStrip");
  const width = Math.max(
    getScrollContainerWidth(),
    (wavesurfer?.getDuration() || 0) * pxPerSec,
  );
  const lineGap = 9;
  const padding = 10;
  const ledgerHalfWidth = 14;

  let minStep = 0;
  let maxStep = 4;
  events.forEach((ev) => {
    if (ev.is_rest) return;
    const midi = pitchToMidi(ev.pitch);
    if (midi == null) return;
    const steps = midiToStaffSteps(midi);
    minStep = Math.min(minStep, Math.floor(steps));
    maxStep = Math.max(maxStep, Math.ceil(steps));
  });
  minStep = Math.min(minStep, -1);
  maxStep = Math.max(maxStep, 5);

  const staffBottomStep = 0;
  const staffTopStep = 4;
  const height = (maxStep - minStep) * lineGap + padding * 2;
  const staffBottomY = padding + (maxStep - staffBottomStep) * lineGap;
  const staffMidY = staffBottomY - 2 * lineGap;

  const svgParts = [
    `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`,
    `<rect width="100%" height="100%" fill="#f8f8f8"/>`,
  ];

  for (let s = 0; s <= 4; s += 1) {
    const y = staffBottomY - s * lineGap;
    svgParts.push(
      `<line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="#999" stroke-width="1"/>`,
    );
  }

  const appendLedgerLines = (noteSteps, noteCx) => {
    const x1 = noteCx - ledgerHalfWidth;
    const x2 = noteCx + ledgerHalfWidth;
    if (noteSteps < staffBottomStep) {
      for (let s = -1; s >= Math.ceil(noteSteps); s -= 1) {
        const ly = staffBottomY - s * lineGap;
        svgParts.push(
          `<line x1="${x1}" y1="${ly}" x2="${x2}" y2="${ly}" stroke="#999" stroke-width="1"/>`,
        );
      }
    } else if (noteSteps > staffTopStep) {
      for (let s = staffTopStep + 1; s <= Math.floor(noteSteps); s += 1) {
        const ly = staffBottomY - s * lineGap;
        svgParts.push(
          `<line x1="${x1}" y1="${ly}" x2="${x2}" y2="${ly}" stroke="#999" stroke-width="1"/>`,
        );
      }
    }
  };

  events.forEach((ev) => {
    const x = ev.perf_start * pxPerSec;
    const ql = Number(ev.duration_ql) || 1;
    if (ev.is_rest) {
      const cx = x + 8;
      appendRestGlyph(svgParts, cx, staffBottomY, lineGap, ql);
    } else {
      const midi = pitchToMidi(ev.pitch);
      const cy = midi != null
        ? midiToStaffY(midi, staffBottomY, lineGap)
        : staffMidY;
      const noteCx = x + 6;
      const noteSteps = midi != null ? midiToStaffSteps(midi) : null;
      if (noteSteps != null) {
        appendLedgerLines(noteSteps, noteCx);
      }
      const stemUp = cy >= staffMidY;
      appendNoteGlyph(svgParts, noteCx, cy, ql, stemUp);
    }
  });

  svgParts.push("</svg>");
  container.innerHTML = svgParts.join("");
  container.style.height = `${height}px`;
  staffStripHeight = height;
  updateOverlayTop();
}

function computeEwmaSeries(events, alpha = EWMA_ALPHA) {
  const byPart = new Map();
  events.forEach((ev) => {
    const part = ev.part ?? 0;
    if (!byPart.has(part)) byPart.set(part, []);
    byPart.get(part).push(ev);
  });
  const points = [];
  byPart.forEach((partEvents) => {
    let ewma = null;
    partEvents.forEach((ev) => {
      const refDur = ev.ref_end - ev.ref_start;
      const perfDur = ev.perf_end - ev.perf_start;
      if (!(refDur > 1e-4) || !(perfDur > 1e-4)) return;
      const ratio = perfDur / refDur;
      if (ewma == null) {
        ewma = ratio;
      } else {
        ewma = alpha * ratio + (1 - alpha) * ewma;
      }
      const t = (ev.perf_start + ev.perf_end) / 2;
      points.push({ t, ratio, ewma, start: ev.perf_start, end: ev.perf_end });
    });
  });
  points.sort((a, b) => a.t - b.t);
  return points;
}

function renderEwmaStrip(events, pxPerSec) {
  const container = document.getElementById("ewmaStrip");
  if (!container) return;
  const width = Math.max(
    getScrollContainerWidth(),
    (wavesurfer?.getDuration() || 0) * pxPerSec,
  );
  const height = EWMA_STRIP_HEIGHT;
  const padTop = 14;
  const padBot = 10;
  const plotH = height - padTop - padBot;
  const series = computeEwmaSeries(events);
  if (!series.length) {
    container.innerHTML =
      `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">` +
      `<rect width="100%" height="100%" fill="#141414"/>` +
      `<text x="8" y="20" fill="#888" font-size="11">EWMA tempo ratio (no events)</text></svg>`;
    container.style.width = `${width}px`;
    return;
  }

  let minV = Infinity;
  let maxV = -Infinity;
  series.forEach((p) => {
    minV = Math.min(minV, p.ewma, p.ratio);
    maxV = Math.max(maxV, p.ewma, p.ratio);
  });
  // Always include 1.0 (score tempo match) in range.
  minV = Math.min(minV, 0.7);
  maxV = Math.max(maxV, 1.3);
  if (maxV - minV < 0.05) {
    minV -= 0.1;
    maxV += 0.1;
  }

  const yAt = (v) => padTop + plotH * (1 - (v - minV) / (maxV - minV));
  const xAt = (t) => t * pxPerSec;

  const ewmaPath = series
    .map((p, i) => `${i === 0 ? "M" : "L"}${xAt(p.t).toFixed(1)} ${yAt(p.ewma).toFixed(1)}`)
    .join(" ");
  const ratioPath = series
    .map((p, i) => `${i === 0 ? "M" : "L"}${xAt(p.t).toFixed(1)} ${yAt(p.ratio).toFixed(1)}`)
    .join(" ");
  const yOne = yAt(1);

  const svgParts = [
    `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`,
    `<rect width="100%" height="100%" fill="#141414"/>`,
    `<text x="8" y="12" fill="#9ab" font-size="10">EWMA tempo ratio (perf/ref duration)</text>`,
    `<line x1="0" y1="${yOne}" x2="${width}" y2="${yOne}" stroke="#355" stroke-width="1" stroke-dasharray="4 3"/>`,
    `<text x="8" y="${yOne - 2}" fill="#466" font-size="9">1.0</text>`,
    `<path d="${ratioPath}" fill="none" stroke="#666" stroke-width="1" opacity="0.7"/>`,
    `<path d="${ewmaPath}" fill="none" stroke="#6af" stroke-width="2"/>`,
  ];
  series.forEach((p) => {
    svgParts.push(
      `<circle cx="${xAt(p.t)}" cy="${yAt(p.ewma)}" r="2" fill="#8cf"/>`,
    );
  });
  svgParts.push("</svg>");
  container.innerHTML = svgParts.join("");
  container.style.width = `${width}px`;
  container.style.height = `${height}px`;
}

function pitchToMidi(pitch) {
  if (!pitch || pitch === "rest") return null;
  // Accept C5, Bb3, B-3 (music21 flat), F#4
  const m = String(pitch).match(/^([A-Ga-g])([#b-]?)(\d+)$/);
  if (!m) return null;
  const names = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
  let pc = names[m[1].toUpperCase()];
  if (m[2] === "#") pc += 1;
  if (m[2] === "b" || m[2] === "-") pc -= 1;
  return (parseInt(m[3], 10) + 1) * 12 + pc;
}

function escapeXml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderAlignmentInfo(data) {
  const el = document.getElementById("alignmentInfo");
  const s = data.summary || {};
  const rows = (data.events || [])
    .map(
      (ev) =>
        `<tr>
          <td>${ev.measure ?? "—"}</td>
          <td>${ev.is_rest ? "rest" : escapeXml(ev.pitch || "?")}</td>
          <td>${formatTime(ev.perf_start)}</td>
          <td>${formatTime(ev.perf_end)}</td>
          <td>${(ev.perf_end - ev.perf_start).toFixed(3)}s</td>
          <td>${ev.residual_mean != null ? ev.residual_mean.toFixed(3) : "—"}</td>
        </tr>`,
    )
    .join("");
  el.innerHTML =
    `<div class="summary">` +
    `DTW path: ${s.warping_path_length ?? "?"} steps · ` +
    `mean residual ${s.mean_residual ?? "?"} · ` +
    `max ${s.max_residual ?? "?"} · ` +
    `${s.event_count ?? 0} score events · ` +
    `${s.candidate_count ?? 0} auto-candidates` +
    `</div>` +
    `<table><thead><tr>` +
    `<th>m</th><th>note</th><th>perf start</th><th>perf end</th><th>dur</th><th>residual</th>` +
    `</tr></thead><tbody>${rows}</tbody></table>`;
}

function updateCandidateHint() {
  const el = document.getElementById("candidateHint");
  if (!el || !sampleData) return;
  const count = sampleData.candidate_count ?? sampleData.candidates?.length ?? 0;
  if (count > 0) {
    el.textContent = `${count} auto-candidate(s) on waveform (dashed orange).`;
  } else if (sampleData.has_alignment) {
    el.textContent =
      "No auto-candidates (alignment clean under thresholds). Click Re-run alignment to refresh.";
  } else {
    el.textContent = "No alignment data yet — run batch or apply segment.";
  }
}

async function rerunAlignment() {
  if (!currentSample) return;
  if (!confirm("Re-run DTW alignment and regenerate auto-candidates?")) return;
  const btn = document.getElementById("realignBtn");
  btn.disabled = true;
  btn.textContent = "Aligning…";
  try {
    const res = await fetch(`/api/samples/${currentSample}/re-align`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const result = await res.json();
    await loadSample(currentSample);
    alert(`Alignment complete. ${result.candidate_count ?? 0} candidate(s) detected.`);
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Re-run alignment";
  }
}

function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

let layoutFitTimer = null;

function scheduleWaveformFit() {
  clearTimeout(layoutFitTimer);
  layoutFitTimer = setTimeout(() => {
    requestAnimationFrame(() => {
      if (!userZoomed) {
        fitWaveformToContainer();
      } else {
        syncAlignmentStackWidth(getEffectivePxPerSec());
        if (viewMode === "alignment" && noteAlignmentData) {
          renderAlignmentOverlays();
        }
      }
      updatePlayhead();
    });
  }, 50);
}

function showScorePlaceholder(measures) {
  cachedScoreXml = null;
  cachedScoreMeta = null;
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
  setupViewControls();
  setupWaveformWheel();
  setupMarqueeSelect();
  window.addEventListener("resize", debounce(() => {
    if (!userZoomed) fitWaveformToContainer();
    else syncAlignmentStackWidth(getEffectivePxPerSec());
  }, 150));

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

    if (e.ctrlKey || e.metaKey) {
      toggleRegionInSelection(region);
      lastRegionClick = { id: null, time: 0 };
      return;
    }

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
    if (isRegionSelected(region)) {
      if (region.data) {
        region.data.start_time = region.start;
        region.data.end_time = region.end;
      }
      if (selectedRegions.length === 1) {
        updateSelectionInfo(region);
      } else {
        updateMultiSelectionInspector();
      }
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
    const pxPerSec = getEffectivePxPerSec();
    const x = contentXFromClientX(clientX);
    const time = Math.max(0, Math.min(duration, x / pxPerSec));
    wavesurfer.setTime(time);
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
    // Keep the playhead visible with minimal scroll — do not jump to left-quarter.
    updatePlayhead({ ensureVisible: true, preferEnd: true });
  };

  scrubber.addEventListener("pointerup", stopScrub);
  scrubber.addEventListener("pointercancel", stopScrub);
}

function scrollPlayheadIntoView(playheadX, { preferEnd = false } = {}) {
  const scroll = document.getElementById("waveformScroll");
  if (!scroll) return;
  const viewW = scroll.clientWidth;
  const left = scroll.scrollLeft;
  const right = left + viewW;
  const margin = Math.min(48, viewW * 0.05);
  const maxScroll = Math.max(0, scroll.scrollWidth - viewW);

  let target = null;
  if (playheadX < left + margin) {
    // Off (or near) the left edge — bring to left quarter, or flush left near start.
    target = preferEnd
      ? playheadX - margin
      : playheadX - viewW * 0.25;
  } else if (playheadX > right - margin) {
    // Off (or near) the right edge — keep near the right so end-of-track scrubbing stays put.
    target = preferEnd
      ? playheadX - viewW + margin
      : playheadX - viewW * 0.25;
  }
  if (target == null) return;
  scroll.scrollLeft = Math.max(0, Math.min(target, maxScroll));
}

let playheadRaf = null;
let playheadPendingOpts = null;

function updatePlayhead(opts = {}) {
  playheadPendingOpts = {
    ensureVisible: !!(playheadPendingOpts?.ensureVisible || opts.ensureVisible),
    preferEnd: !!(playheadPendingOpts?.preferEnd || opts.preferEnd),
  };
  if (playheadRaf) return;
  playheadRaf = requestAnimationFrame(() => {
    playheadRaf = null;
    const pending = playheadPendingOpts || {};
    playheadPendingOpts = null;
    const ensureVisible = pending.ensureVisible === true;
    const preferEnd = pending.preferEnd === true;
    const duration = wavesurfer.getDuration();
    const scrubber = document.getElementById("scrubber");
    const progress = document.getElementById("scrubberProgress");
    const playhead = document.getElementById("playhead");
    if (!duration || !scrubber) return;

    const pxPerSec = getEffectivePxPerSec();
    const currentTime = wavesurfer.getCurrentTime();
    const contentWidth = Math.max(
      getScrollContainerWidth(),
      duration * pxPerSec,
    );
    // Keep the mark inside the scrubber when at/near the end.
    const x = Math.min(currentTime * pxPerSec, Math.max(0, contentWidth - 1));
    playhead.style.left = `${x}px`;
    progress.style.width = `${x}px`;
    document.getElementById("timeDisplay").textContent = formatTime(currentTime);

    if (!scrubbing) {
      const playing = typeof wavesurfer.isPlaying === "function" && wavesurfer.isPlaying();
      if (playing || ensureVisible) {
        scrollPlayheadIntoView(x, { preferEnd: preferEnd || !playing });
      }
    }
  });
}

function onWaveformReady() {
  userZoomed = false;
  fitWaveformToContainer();
  withProgrammaticRegions(() => {
    pendingRegions.forEach(({ label, isCandidate }) => addRegion(label, isCandidate));
    pendingRegions = [];
    ensureTrimRegion();
  });
  updatePlayhead();
  updateCandidateHint();
  applyAllLabelsVisibility();
  if (viewMode === "alignment") {
    loadNoteAlignment().then(() => renderAlignmentOverlays());
  }
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
  const loadId = ++scoreLoadId;
  currentSample = sampleId;
  setScoreStatus("Loading score…");
  cachedScoreXml = null;
  cachedScoreMeta = null;

  const res = await fetch(`/api/samples/${sampleId}`);
  const data = await res.json();
  if (loadId !== scoreLoadId) return;
  sampleData = data;
  taxonomy = data.taxonomy;
  populateTypeSelect();
  document.getElementById("annotatorId").value = data.annotator_id || "";
  updateMeasureControls(data.prep);

  trimRegion = null;
  selectedRegions = [];
  selectedRegion = null;
  repetitionLinkMode = null;
  lastRegionClick = { id: null, time: 0 };
  resetSelectionInfo();
  regionsPlugin.clearRegions();
  pendingRegions = [
    ...data.candidates.map((c) => ({ label: normalizeLabelType(c), isCandidate: true })),
    ...data.labels.map((l) => ({ label: normalizeLabelType(l), isCandidate: false })),
  ];

  noteAlignmentData = null;
  userZoomed = false;
  clearAlignmentOverlays();

  const duration = data.prep?.performance_duration || 0;
  if (typeof wavesurfer.setOptions === "function") {
    wavesurfer.setOptions({ minPxPerSec: computeFitPxPerSec(duration) });
  }

  const audioUrl = audioUrlWithCacheBust(data.audio_url, data.audio_mtime);
  await wavesurfer.load(audioUrl);
  if (loadId !== scoreLoadId) return;
  updateCandidateHint();

  const measures = data.prep?.total_measures || 0;
  const hasSegment = !!data.prep?.score_segment;
  if (hasSegment) {
    const seg = data.prep.score_segment;
    await viewScoreSegment(
      seg.start_measure,
      seg.end_measure,
      seg.start_beat || 1,
      seg.end_beat ?? null,
      loadId,
    );
  } else if (measures > LARGE_SCORE_MEASURES) {
    if (loadId === scoreLoadId) showScorePlaceholder(measures);
  } else {
    await renderScore(data.full_score_url || data.score_url, { mode: "full", loadId });
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
  syncRegionVisual(region, { selected: isRegionSelected(region) });
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
    : (region.data?.comment ?? document.getElementById("comment").value) || null;
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
  if (isRegionSelected(region)) {
    if (selectedRegions.length === 1) {
      updateSelectionInfo(region);
      updateRepetitionPanel(region);
    } else {
      updateMultiSelectionInspector();
    }
  }
  syncRegionVisual(region, { selected: isRegionSelected(region) });
}

function applyLabelToSelection() {
  const targets = selectedRegions.filter((r) => !isTrimRegion(r) && !isLinkOverlay(r));
  if (!targets.length) return;
  targets.forEach((region) => applyLabelToRegion(region));
}

function promoteCandidate(source, overrideType) {
  if (!selectedRegion || !selectedRegion.data?.isCandidate) return;
  selectedRegion.data.source = source;
  if (overrideType) selectedRegion.data.type = overrideType;
  selectedRegion.data.isCandidate = false;
  applyLabelToSelection();
}

function deleteSelectedRegion() {
  const targets = selectedRegions.filter((r) => !isTrimRegion(r) && !isLinkOverlay(r));
  if (!targets.length) return;
  targets.forEach((region) => {
    findLinkOverlay(region)?.remove();
    region.remove();
  });
  selectedRegions = [];
  selectedRegion = null;
  repetitionLinkMode = null;
  resetSelectionInfo();
  updateRepetitionPanel(null);
  const confirmBtn = document.getElementById("confirmCandidateBtn");
  const rejectBtn = document.getElementById("rejectCandidateBtn");
  if (confirmBtn) confirmBtn.disabled = false;
  if (rejectBtn) rejectBtn.disabled = false;
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

async function viewScoreSegment(startMeasure, endMeasure, startBeat, endBeat, loadId = null) {
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
  if (loadId != null && loadId !== scoreLoadId) return;
  if (!res.ok) {
    alert(await res.text());
    setScoreStatus("Could not load score segment.");
    return;
  }
  const xml = await res.text();
  if (loadId != null && loadId !== scoreLoadId) return;
  await renderScoreXml(xml, {
    mode: "segment",
    start,
    end,
    startBeat: range.startBeat,
    endBeat: range.endBeat,
    xmlBytes: xml.length,
    loadId,
  });
}

async function renderScore(url, meta = {}) {
  setScoreStatus("Loading score…");
  const res = await fetch(url);
  if (meta.loadId != null && meta.loadId !== scoreLoadId) return;
  if (!res.ok) {
    alert(await res.text());
    setScoreStatus("Could not load score.");
    return;
  }
  const xml = await res.text();
  if (meta.loadId != null && meta.loadId !== scoreLoadId) return;
  await renderScoreXml(xml, { ...meta, url, xmlBytes: xml.length });
}

async function renderScoreXml(xml, meta = {}) {
  if (meta.loadId != null && meta.loadId !== scoreLoadId) return;
  cachedScoreXml = xml;
  cachedScoreMeta = { ...meta };
  delete cachedScoreMeta.loadId;

  const container = document.getElementById("osmdContainer");
  const panel = document.getElementById("scorePanel");
  const timeline = document.getElementById("timelinePanel");
  container.innerHTML = "";
  osmdInstance = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
    autoResize: false,
    drawTitle: true,
  });
  await osmdInstance.load(xml);
  if (meta.loadId != null && meta.loadId !== scoreLoadId) return;

  // When score panel is collapsed (alignment mode), clientWidth is 0 — use fallbacks.
  let pageWidth = container.clientWidth - 24;
  if (pageWidth < 100) {
    pageWidth = (panel?.clientWidth || 0) - 24;
  }
  if (pageWidth < 100) {
    pageWidth = (timeline?.clientWidth || 0) - 48;
  }
  pageWidth = Math.max(400, pageWidth);

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

function isEditableKeyTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

function onKeyDown(e) {
  if (isEditableKeyTarget(e.target)) return;

  if (e.code === "Space") {
    e.preventDefault();
    wavesurfer.playPause();
    return;
  }

  if (e.key === "Delete" || e.key === "Backspace") {
    if (selectedRegions.length) {
      e.preventDefault();
      deleteSelectedRegion();
    }
    return;
  }

  const idx = parseInt(e.key, 10);
  if (idx >= 1 && idx <= taxonomy.length) {
    document.getElementById("labelType").value = taxonomy[idx - 1];
    refreshDragSelection();
    if (selectedRegions.length) {
      applyLabelToSelection();
    }
  }
  if (!selectedRegion || isTrimRegion(selectedRegion)) return;
  const step = 0.01;
  if (e.key === "ArrowLeft") selectedRegion.setOptions({ start: Math.max(0, selectedRegion.start - step) });
  if (e.key === "ArrowRight") selectedRegion.setOptions({ end: selectedRegion.end + step });
}

init();
