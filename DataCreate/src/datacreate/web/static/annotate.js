let wavesurfer;
let regionsPlugin;
let osmdInstance = null;
let taxonomy = [];
let currentSample = null;
let sampleData = null;
let labelSource = "human";
let selectedRegions = [];
/** Sole selection when exactly one region is selected; null when empty or multi. */
let selectedRegion = null;
let trimRegion = null;
let loopSelection = false;
let scrubbing = false;
let marqueeSelecting = false;
let lastWaveformPointerClientX = null;
let pendingRegions = [];
let lastRegionClick = { id: null, time: 0 };
const REGION_DOUBLE_CLICK_MS = 400;
let lastRegionDragAt = 0;
let ignorePickerHideUntil = 0;
let suppressMoveLock = false;
const regionPosByRef = new WeakMap();
let repetitionLinkMode = null; // null | "pick" | "draw"
/** >0 while addRegion is called from code (labels, trim, link overlays) — skip auto-label. */
let programmaticRegionDepth = 0;
let viewMode = "normal"; // normal | alignment
let zoomFactor = 1;
let fitPxPerSec = 1;
let userZoomed = false;
let noteAlignmentData = null;
let scoreEventData = null;
const MELODY_PAD_NOTES = 2;
let melodyDrag = null;
let labelsVisible = true;
const SCRUBBER_HEIGHT = 28;
const EWMA_STRIP_HEIGHT = 140;
const EWMA_ALPHA = 0.3;
const ZOOM_STEP = 1.25;
const MIN_ZOOM_FACTOR = 1;
const MAX_ZOOM_FACTOR = 32;
const MIN_NOTE_GAP_PX = 12;
const MAX_UNDO = 50;
let undoStack = [];
let idleSnapshot = null;
let undoSuspended = false;
let regionDragUndoArmed = false;
let ignoreRegionUpdateUndo = false;
let overlapPickerRegions = [];
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

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function snapshotRegions() {
  if (!regionsPlugin) return [];
  return regionsPlugin.getRegions()
    .filter((r) => !isTrimRegion(r) && !isLinkOverlay(r))
    .map((r) => ({
      start: r.start,
      end: r.end,
      isCandidate: !!r.data?.isCandidate,
      data: cloneJson({
        id: r.data?.id,
        source: r.data?.source,
        type: r.data?.type,
        severity: r.data?.severity ?? null,
        comment: r.data?.comment ?? null,
        deviation_cents: r.data?.deviation_cents ?? null,
        deviation_ms: r.data?.deviation_ms ?? null,
        measure_number: r.data?.measure_number ?? null,
        note_id: r.data?.note_id ?? null,
        repeats_label_range: r.data?.repeats_label_range || null,
        score_part: r.data?.score_part || null,
        pitches: r.data?.pitches || null,
        note_ids: r.data?.note_ids || null,
        core_note_ids: r.data?.core_note_ids || null,
      }),
    }));
}

function snapshotsEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function captureIdleSnapshot() {
  if (undoSuspended) return;
  idleSnapshot = snapshotRegions();
}

function pushUndoFromIdle() {
  if (undoSuspended || !idleSnapshot) return;
  const top = undoStack[undoStack.length - 1];
  if (top && snapshotsEqual(top, idleSnapshot)) return;
  undoStack.push(cloneJson(idleSnapshot));
  if (undoStack.length > MAX_UNDO) undoStack.shift();
}

function clearUndoHistory() {
  undoStack = [];
  idleSnapshot = null;
  regionDragUndoArmed = false;
  ignoreRegionUpdateUndo = false;
}

function restoreRegionSnapshot(snap) {
  if (!regionsPlugin) return;
  undoSuspended = true;
  try {
    withProgrammaticRegions(() => {
      regionsPlugin.getRegions().slice().forEach((region) => {
        if (isTrimRegion(region)) return;
        try {
          region.remove();
        } catch {
          /* WaveSurfer can throw when removing the last remaining region */
        }
      });
      snap.forEach((item) => {
        addRegion(
          {
            ...item.data,
            start_time: item.start,
            end_time: item.end,
            repeats_label_range: item.data?.repeats_label_range || null,
          },
          item.isCandidate,
        );
      });
    });
  } finally {
    undoSuspended = false;
  }
  selectedRegions = [];
  selectedRegion = null;
  repetitionLinkMode = null;
  resetSelectionInfo();
  updateRepetitionPanel(null);
  applyAllLabelsVisibility();
  hideOverlapPicker();
  refreshAllCaptions();
  refreshMelodyUi();
  captureIdleSnapshot();
}

function undoLastAction() {
  if (!undoStack.length) return;
  restoreRegionSnapshot(undoStack.pop());
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
  bad_start: "rgba(210,110,40,0.45)",
  bad_timbre: "rgba(0,170,150,0.4)",
  squeak: "rgba(255,50,170,0.45)",
  sliding: "rgba(255,130,70,0.45)",
  misc: "rgba(140,140,160,0.45)",
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
  bad_start: "Bad start (messy / failed attack)",
  bad_timbre: "Bad timbre (poor tone quality)",
  squeak: "Squeak",
  sliding: "Sliding (rolling over a note too fast)",
  misc: "Misc (other error)",
};

function typeDisplayName(type) {
  return (TYPE_LABELS[type] || type || "unlabeled").split(" (")[0];
}

function labelOverrides(value) {
  if (!value || typeof value !== "object") return {};
  if (typeof Event !== "undefined" && value instanceof Event) return {};
  return value;
}

function currentLabelType() {
  const raw = document.getElementById("labelType")?.value;
  if (raw && (!taxonomy.length || taxonomy.includes(raw))) return raw;
  return taxonomy[0] || "misc";
}

function getRepetitionRegionsSorted() {
  if (!regionsPlugin) return [];
  return regionsPlugin.getRegions()
    .filter((r) => !isTrimRegion(r) && !isLinkOverlay(r) && r.data?.type === "repetition")
    .sort((a, b) => {
      const dt = a.start - b.start;
      if (Math.abs(dt) > 1e-4) return dt;
      return String(a.data?.id || a.id || "").localeCompare(String(b.data?.id || b.id || ""));
    });
}

function getRepetitionNumberMap() {
  const map = new Map();
  getRepetitionRegionsSorted().forEach((r, i) => {
    const n = i + 1;
    if (r.data?.id) map.set(r.data.id, n);
    map.set(r, n);
  });
  return map;
}

function regionCaptionText(region, numbers = getRepetitionNumberMap()) {
  if (isLinkOverlay(region)) {
    const n = numbers.get(region.data?.parentId);
    return n != null ? `Original ${n}` : "Original";
  }
  const type = region.data?.type;
  const base = typeDisplayName(type);
  if (type === "repetition") {
    const n = numbers.get(region.data?.id) ?? numbers.get(region);
    return n != null ? `${base} ${n}` : base;
  }
  return base;
}

function captionFontSize(region) {
  const dur = Math.max(0, (region.end ?? 0) - (region.start ?? 0));
  const px = dur * (getEffectivePxPerSec() || 1);
  let size = 12;
  if (dur < 0.12) size = 7;
  else if (dur < 0.22) size = 8;
  else if (dur < 0.4) size = 9;
  else if (dur < 0.8) size = 10;
  else if (dur < 1.6) size = 11;
  if (px < 18) size = Math.min(size, 7);
  else if (px < 32) size = Math.min(size, 8);
  else if (px < 52) size = Math.min(size, 9);
  return size;
}

function getRegionContentEl(region) {
  if (region?.content instanceof HTMLElement) return region.content;
  return region?.element?.querySelector?.('[part~="region-content"]') || null;
}

function styleRegionCaptionEl(el, region, text) {
  if (!el) return;
  if (el.textContent !== text) el.textContent = text;
  el.setAttribute("part", "region-content");
  const size = captionFontSize(region);
  el.style.fontSize = `${size}px`;
  el.style.lineHeight = "1.15";
  el.style.padding = "0 3px";
  el.style.whiteSpace = "nowrap";
  el.style.overflow = "visible";
  el.style.display = "inline-block";
  el.style.width = "auto";
  el.style.minWidth = "max-content";
  el.style.maxWidth = "none";
  el.style.pointerEvents = "none";
  el.style.color = "#f4f4f4";
  el.style.textShadow = "0 1px 2px rgba(0,0,0,0.9)";
  el.style.fontWeight = "600";
  el.style.position = "relative";
  el.style.zIndex = "12";
}

function applyRegionCaption(region, numbers = getRepetitionNumberMap()) {
  if (!region || isTrimRegion(region)) return;
  const text = regionCaptionText(region, numbers);
  let el = getRegionContentEl(region);
  if (!el) {
    if (typeof region.setOptions === "function") {
      region.setOptions({ content: text });
    }
    el = getRegionContentEl(region);
  }
  styleRegionCaptionEl(el, region, text);
  rememberRegionPos(region);
}

let captionLayoutTimer = null;

function layoutCaptionRows() {
  if (!regionsPlugin) return;
  const items = [];
  regionsPlugin.getRegions().forEach((r) => {
    if (isTrimRegion(r)) return;
    if (!isLinkOverlay(r) && !isRegionVisible(r)) return;
    const el = getRegionContentEl(r);
    if (!el) return;
    items.push({ region: r, el });
  });
  items.sort((a, b) => {
    const dt = a.region.start - b.region.start;
    if (Math.abs(dt) > 1e-4) return dt;
    return String(a.region.data?.type || "").localeCompare(String(b.region.data?.type || ""));
  });
  items.forEach(({ el }) => {
    el.style.marginTop = "0px";
  });
  requestAnimationFrame(() => {
    const rows = [];
    const waveH = wavesurfer?.options?.height || 140;
    items.forEach(({ el }) => {
      if (!el.isConnected) return;
      const box = el.getBoundingClientRect();
      const left = box.left;
      const right = Math.max(box.right, left + 8);
      const h = Math.max(box.height || 0, 10);
      let row = 0;
      for (; row < rows.length; row += 1) {
        if (!rows[row].some((o) => left < o.right && right > o.left)) break;
      }
      if (!rows[row]) rows[row] = [];
      rows[row].push({ left, right });
      const maxRow = Math.max(0, Math.floor((waveH - h - 2) / h));
      el.style.marginTop = `${Math.min(row, maxRow) * h}px`;
    });
  });
}

function scheduleCaptionLayout() {
  clearTimeout(captionLayoutTimer);
  captionLayoutTimer = setTimeout(layoutCaptionRows, 30);
}

function refreshAllCaptions() {
  if (!regionsPlugin) return;
  const numbers = getRepetitionNumberMap();
  regionsPlugin.getRegions().forEach((region) => applyRegionCaption(region, numbers));
  scheduleCaptionLayout();
}

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
  const type = currentLabelType();
  return TYPE_COLORS[type] || "rgba(45,108,223,0.35)";
}

function refreshDragSelection() {
  if (!regionsPlugin) return;
  if (typeof regionsPlugin.disableDragSelection === "function") {
    regionsPlugin.disableDragSelection();
  }
  // Labels are drawn on the staff. Waveform drag is only for a repetition's original range.
  if (repetitionLinkMode === "draw" && regionsPlugin.enableDragSelection) {
    regionsPlugin.enableDragSelection({ color: getDragSelectionColor() });
  }
}

function generateLabelId() {
  return `lbl_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

function isLinkOverlay(region) {
  return region?.data?.role === "repetition-link"
    || String(region?.id || "").startsWith("link-");
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
  el.style.overflow = "visible";

  if (kind === "trim") {
    el.style.border = "2px solid #3c3";
    el.style.backgroundColor = "rgba(60, 200, 60, 0.12)";
    el.style.boxShadow = selected ? "0 0 0 1px rgba(0, 0, 0, 0.45)" : "";
    el.style.zIndex = selected ? "10" : "1";
    el.style.boxSizing = "border-box";
    el.style.pointerEvents = "none";
    el.querySelectorAll('[part*="region-handle"]').forEach((handle) => {
      handle.style.pointerEvents = "auto";
    });
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
    el.style.zIndex = selected ? "10" : "6";
  }
  applyRegionLabelVisibility(region);
  applyRegionMoveMode(region);
}

function isRegionVisible(region) {
  const kind = getRegionKind(region);
  if (kind === "trim" || kind === "link") return true;
  return labelsVisible;
}

function applyRegionMoveMode(region) {
  if (!region || isTrimRegion(region) || isLinkOverlay(region)) return;
  const canMove = isRegionSelected(region) && isRegionVisible(region);
  // WaveSurfer starts drag-create on pointerdown; don't lock until pointerup.
  if (!canMove && suppressMoveLock) return;
  try {
    if (typeof region.setOptions === "function") {
      region.setOptions({ drag: canMove, resize: canMove });
    } else {
      region.drag = canMove;
      region.resize = canMove;
    }
  } catch {
    /* plugin may ignore drag/resize after removal */
  }
  const el = region.element;
  if (!el) return;
  el.style.pointerEvents = canMove ? "auto" : "none";
  el.querySelectorAll('[part*="region-handle"]').forEach((handle) => {
    handle.style.pointerEvents = canMove ? "auto" : "none";
  });
}

function applyRegionLabelVisibility(region) {
  if (!region?.element) return;
  if (isTrimRegion(region) || isLinkOverlay(region)) {
    region.element.style.visibility = "visible";
    return;
  }
  const visible = isRegionVisible(region);
  region.element.style.visibility = visible ? "visible" : "hidden";
  applyRegionMoveMode(region);
}

function applyAllLabelsVisibility() {
  if (!regionsPlugin) return;
  regionsPlugin.getRegions().forEach((region) => applyRegionLabelVisibility(region));
  scheduleCaptionLayout();
}

function toggleLabelsVisibility() {
  labelsVisible = !labelsVisible;
  const btn = document.getElementById("toggleLabelsBtn");
  if (btn) {
    btn.classList.toggle("active", labelsVisible);
    btn.textContent = labelsVisible ? "Labels" : "Labels (off)";
  }
  applyAllLabelsVisibility();
  hideOverlapPicker();
}

function findLinkOverlay(parentRegion) {
  return findLinkOverlays(parentRegion)[0] || null;
}

function findLinkOverlays(parentRegion) {
  const parentId = parentRegion?.data?.id;
  if (!parentId || !regionsPlugin) return [];
  return regionsPlugin.getRegions().filter((region) => {
    if (!isLinkOverlay(region)) return false;
    return region.data?.parentId === parentId || region.id === `link-${parentId}`;
  });
}

function removeLinkedOriginals(parentRegion) {
  if (!regionsPlugin) return;
  findLinkOverlays(parentRegion).forEach((overlay) => {
    try {
      overlay.remove();
    } catch {
      /* overlay may already be gone */
    }
  });
}

function sweepOrphanLinkOverlays() {
  if (!regionsPlugin) return;
  const liveParentIds = new Set(
    regionsPlugin.getRegions()
      .filter((r) => r.data?.type === "repetition" && r.data?.id)
      .map((r) => r.data.id),
  );
  regionsPlugin.getRegions().slice().forEach((region) => {
    if (!isLinkOverlay(region)) return;
    const parentId = region.data?.parentId;
    if (parentId && !liveParentIds.has(parentId)) {
      try {
        region.remove();
      } catch {
        /* overlay may already be gone */
      }
    }
  });
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
      content: "Original",
      id: `link-${parentRegion.data.id}`,
      data: { role: "repetition-link", parentId: parentRegion.data.id },
    });
  const overlay =
    programmaticRegionDepth > 0
      ? createOverlay()
      : withProgrammaticRegions(createOverlay);
  syncRegionVisual(overlay);
  applyRegionCaption(overlay);
  scheduleCaptionLayout();
}

function setRepetitionLinkRange(parentRegion, start, end, existingRegion = null) {
  if (!parentRegion?.data || parentRegion.data.type !== "repetition") return;
  const lo = Math.min(start, end);
  const hi = Math.max(start, end);
  if (hi - lo < 1e-4) {
    existingRegion?.remove();
    return;
  }

  if (!undoSuspended) {
    pushUndoFromIdle();
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
      content: "Original",
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
  refreshAllCaptions();
  if (!undoSuspended) captureIdleSnapshot();
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
    info.textContent = "Optional: link the earlier range this repetition restates.";
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
  if (d.score_part) label.score_part = d.score_part;
  if (d.pitches?.length) label.pitches = d.pitches;
  if (d.note_ids?.length) label.note_ids = d.note_ids;
  if (d.core_note_ids?.length) label.core_note_ids = d.core_note_ids;
  if (d.extra_copies != null) label.extra_copies = d.extra_copies;
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
    pushUndoFromIdle();
    delete selectedRegion.data.repeats_label_range;
    repetitionLinkMode = null;
    refreshDragSelection();
    syncRepetitionLinkOverlay(selectedRegion);
    updateRepetitionPanel(selectedRegion);
    captureIdleSnapshot();
  };
  document.getElementById("labelType").addEventListener("change", () => {
    refreshDragSelection();
    if (selectedRegions.length) {
      applyLabelToSelection();
      return;
    }
    updateRepetitionPanel(selectedRegion);
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
  const n = selectedRegions.length;
  if (n === 0) {
    resetSelectionInfo();
    updateRepetitionPanel(null);
    refreshMelodyUi();
    return;
  }
  if (n > 1) {
    document.getElementById("selectionInfo").textContent =
      `${n} regions selected. Delete or assign type (1–8) applies to all.`;
    updateRepetitionPanel(null);
    refreshMelodyUi();
    return;
  }
  const region = selectedRegions[0];
  const d = region.data || {};
  updateSelectionInfo(region);
  if (d.type) document.getElementById("labelType").value = d.type;
  if (d.severity) document.getElementById("severity").value = d.severity;
  document.getElementById("comment").value = d.comment || "";
  updateRepetitionPanel(region);
  refreshMelodyUi();
}

function selectRegion(region) {
  if (isTrimRegion(region) || isLinkOverlay(region)) return;
  if (!isRegionVisible(region)) return;
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
  if (!isRegionVisible(region)) return;
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
  const next = regions.filter(
    (r) => r && !isTrimRegion(r) && !isLinkOverlay(r) && isRegionVisible(r),
  );
  selectedRegions = next;
  syncPrimarySelection();
  refreshSelectionVisuals();
  updateMultiSelectionInspector();
}

function selectAllRegions() {
  if (!regionsPlugin) return;
  const hits = regionsPlugin.getRegions().filter(
    (r) => !isTrimRegion(r) && !isLinkOverlay(r) && isRegionVisible(r),
  );
  setSelectedRegions(hits);
  hideOverlapPicker();
}

function resetSelectionInfo() {
  document.getElementById("selectionInfo").textContent =
    "Drag on the reference-score staff under the waveform to add a label. Double-click a region to select it before moving.";
}

function getScrollContainerWidth() {
  const scroll = document.getElementById("waveformScroll");
  if (scroll && scroll.clientWidth > 0) return scroll.clientWidth;
  const wrap = document.querySelector(".waveform-wrap");
  return wrap?.clientWidth || 800;
}

function typicalScoreEventGapSec(events = getScoreEvents()) {
  if (!events || events.length < 2) return null;
  const starts = events.map((ev) => eventPerfStart(ev)).sort((a, b) => a - b);
  const gaps = [];
  for (let i = 1; i < starts.length; i += 1) {
    const gap = starts[i] - starts[i - 1];
    if (gap > 0.008) gaps.push(gap);
  }
  if (!gaps.length) return null;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length * 0.1)] || gaps[0];
}

function computeFitPxPerSec(duration) {
  if (!duration || duration <= 0) return 1;
  const widthFit = getScrollContainerWidth() / duration;
  const gap = typicalScoreEventGapSec();
  if (!gap) return widthFit;
  const densityFit = MIN_NOTE_GAP_PX / gap;
  return Math.max(widthFit, Math.min(densityFit, widthFit * 6, 160));
}

function unpackStaffXs(events, pxPerSec, startOf, minGap = MIN_NOTE_GAP_PX) {
  const xs = events.map((ev) => startOf(ev) * pxPerSec);
  for (let i = 1; i < xs.length; i += 1) {
    if (xs[i] < xs[i - 1] + minGap) xs[i] = xs[i - 1] + minGap;
  }
  return xs;
}

function getCurrentPxPerSec() {
  return fitPxPerSec * zoomFactor;
}

/** WaveSurfer does not shrink below fit; clamp overlays to the same scale. */
function getEffectivePxPerSec() {
  return Math.max(fitPxPerSec, getCurrentPxPerSec());
}

/** Shared content width for staff, scrubber, waveform, and EWMA (px). */
function getAlignmentContentWidth(pxPerSec = getEffectivePxPerSec()) {
  const duration = wavesurfer?.getDuration() || 0;
  const fromScale = duration > 0 ? duration * pxPerSec : 0;
  let fromWs = 0;
  if (wavesurfer && typeof wavesurfer.getWidth === "function") {
    fromWs = wavesurfer.getWidth() || 0;
  }
  return Math.max(getScrollContainerWidth(), fromScale, fromWs);
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
  const clientX = Number.isFinite(anchorClientX)
    ? anchorClientX
    : lastWaveformPointerClientX;

  let anchorTime = 0;
  let viewOffset = 0;
  if (scroll && prevPx > 0) {
    if (Number.isFinite(clientX)) {
      const rect = scroll.getBoundingClientRect();
      viewOffset = Math.max(0, Math.min(scroll.clientWidth, clientX - rect.left));
      anchorTime = (prevScrollLeft + viewOffset) / prevPx;
    } else {
      const duration = wavesurfer.getDuration() || 0;
      const currentTime = typeof wavesurfer.getCurrentTime === "function"
        ? wavesurfer.getCurrentTime()
        : 0;
      const playheadTime = Math.max(0, Math.min(duration || currentTime, currentTime));
      const playheadViewX = playheadTime * prevPx - prevScrollLeft;
      const inView = playheadViewX >= 0 && playheadViewX <= scroll.clientWidth;
      anchorTime = playheadTime;
      viewOffset = inView ? playheadViewX : scroll.clientWidth / 2;
    }
  }

  const effective = Math.max(fitPxPerSec, pxPerSec);
  const duration = wavesurfer.getDuration() || 0;
  const contentWidth = Math.max(
    getScrollContainerWidth(),
    duration > 0 ? duration * effective : 0,
  );
  if (typeof wavesurfer.setOptions === "function") {
    wavesurfer.setOptions({ minPxPerSec: effective, width: contentWidth });
  } else if (typeof wavesurfer.zoom === "function") {
    wavesurfer.zoom(effective);
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
  refreshAllCaptions();
}

function syncAlignmentStackWidth(pxPerSec) {
  const stack = document.getElementById("alignmentStack");
  const duration = wavesurfer?.getDuration() || 0;
  if (!stack || !duration) return;
  const width = getAlignmentContentWidth(pxPerSec);
  stack.style.width = `${width}px`;
  stack.style.minWidth = `${width}px`;
  const scrubber = document.getElementById("scrubber");
  if (scrubber) scrubber.style.width = `${width}px`;
  const wrap = document.querySelector(".waveform-wrap");
  if (wrap) {
    wrap.style.width = `${width}px`;
    wrap.style.minWidth = `${width}px`;
  }
  const waveEl = document.getElementById("waveform");
  if (waveEl) {
    waveEl.style.width = `${width}px`;
    waveEl.style.minWidth = `${width}px`;
  }
  const ewma = document.getElementById("ewmaStrip");
  if (ewma) {
    ewma.style.width = `${width}px`;
    ewma.style.minWidth = `${width}px`;
  }
  ["transcriptionStrip", "scoreAlignOverlay", "melodyStrip", "scoreCompare"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.width = `${width}px`;
    el.style.minWidth = `${width}px`;
  });
  updateOverlayTop();
  renderScoreCompare();
}

function updateOverlayTop() {
  const stack = document.getElementById("alignmentStack");
  const top = SCRUBBER_HEIGHT;
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
  scroll.addEventListener("pointermove", (e) => {
    lastWaveformPointerClientX = e.clientX;
  });
  scroll.addEventListener(
    "wheel",
    (e) => {
      if (!wavesurfer) return;
      lastWaveformPointerClientX = e.clientX;
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
      if (!isRegionVisible(r)) return false;
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

function contentPointFromClient(clientX, clientY) {
  const scroll = document.getElementById("waveformScroll");
  if (!scroll) return { x: 0, y: 0 };
  const rect = scroll.getBoundingClientRect();
  return {
    x: clientX - rect.left + scroll.scrollLeft,
    y: clientY - rect.top + scroll.scrollTop,
  };
}

function setupRectSelect() {
  const scroll = document.getElementById("waveformScroll");
  const stack = document.getElementById("alignmentStack");
  if (!scroll || !stack) return;

  let startX = 0;
  let startY = 0;
  let clientStartX = 0;
  let clientStartY = 0;
  let box = null;

  const finishRect = (e) => {
    if (!rectSelecting) return;
    rectSelecting = false;
    const end = contentPointFromClient(e.clientX, e.clientY);
    const left = Math.min(startX, end.x);
    const right = Math.max(startX, end.x);
    const top = Math.min(startY, end.y);
    const bottom = Math.max(startY, end.y);
    if (box) {
      box.remove();
      box = null;
    }
    refreshDragSelection();

    const moved = Math.hypot(e.clientX - clientStartX, e.clientY - clientStartY);
    if (moved < 6) return;

    const scrollRect = scroll.getBoundingClientRect();
    const clientBox = {
      left: scrollRect.left + (left - scroll.scrollLeft),
      right: scrollRect.left + (right - scroll.scrollLeft),
      top: scrollRect.top + (top - scroll.scrollTop),
      bottom: scrollRect.top + (bottom - scroll.scrollTop),
    };
    const hits = regionsPlugin.getRegions().filter((r) => {
      if (isTrimRegion(r) || isLinkOverlay(r) || !isRegionVisible(r)) return false;
      const el = r.element;
      if (!el) {
        const px = getEffectivePxPerSec();
        return r.end * px > left && r.start * px < right;
      }
      const b = el.getBoundingClientRect();
      return (
        b.left < clientBox.right
        && b.right > clientBox.left
        && b.top < clientBox.bottom
        && b.bottom > clientBox.top
      );
    });
    setSelectedRegions(hits);
    hideOverlapPicker();
  };

  scroll.addEventListener("contextmenu", (e) => {
    if (e.shiftKey || rectSelecting) e.preventDefault();
  });

  scroll.addEventListener("pointerdown", (e) => {
    if (e.button !== 2 || !e.shiftKey) return;
    if (e.target.closest("#scrubber")) return;
    if (isTextEntryTarget(e.target)) return;
    e.preventDefault();
    e.stopPropagation();
    rectSelecting = true;
    const p = contentPointFromClient(e.clientX, e.clientY);
    startX = p.x;
    startY = p.y;
    clientStartX = e.clientX;
    clientStartY = e.clientY;
    if (typeof regionsPlugin.disableDragSelection === "function") {
      regionsPlugin.disableDragSelection();
    }
    box = document.createElement("div");
    box.className = "marquee-select rect";
    box.style.left = `${startX}px`;
    box.style.top = `${startY}px`;
    box.style.width = "0px";
    box.style.height = "0px";
    stack.appendChild(box);
    scroll.setPointerCapture(e.pointerId);
  });

  scroll.addEventListener("pointermove", (e) => {
    if (!rectSelecting || !box) return;
    const p = contentPointFromClient(e.clientX, e.clientY);
    box.style.left = `${Math.min(startX, p.x)}px`;
    box.style.top = `${Math.min(startY, p.y)}px`;
    box.style.width = `${Math.abs(p.x - startX)}px`;
    box.style.height = `${Math.abs(p.y - startY)}px`;
  });

  scroll.addEventListener("pointerup", finishRect);
  scroll.addEventListener("pointercancel", () => {
    if (!rectSelecting) return;
    rectSelecting = false;
    if (box) {
      box.remove();
      box = null;
    }
    refreshDragSelection();
  });
}

function hideOverlapPicker() {
  const el = document.getElementById("overlapPicker");
  if (el) {
    el.classList.add("hidden");
    el.innerHTML = "";
  }
  overlapPickerRegions = [];
}

function rememberRegionPos(region) {
  if (!region) return;
  regionPosByRef.set(region, { start: region.start, end: region.end });
}

function regionPosChanged(region) {
  const prev = regionPosByRef.get(region);
  rememberRegionPos(region);
  if (!prev) return false;
  return Math.abs(prev.start - region.start) > 0.012
    || Math.abs(prev.end - region.end) > 0.012;
}

function sortRegionsForPicker(regions) {
  return regions.slice().sort((a, b) => {
    const dt = a.start - b.start;
    if (Math.abs(dt) > 1e-6) return dt;
    return String(a.data?.type || "").localeCompare(String(b.data?.type || ""));
  });
}

function regionsAtTime(t) {
  if (!regionsPlugin) return [];
  return sortRegionsForPicker(regionsPlugin.getRegions().filter((r) => {
    if (isTrimRegion(r) || isLinkOverlay(r) || !isRegionVisible(r)) return false;
    return r.start <= t && t <= r.end;
  }));
}

function regionsOverlappingRegion(region) {
  if (!regionsPlugin || !region) return [];
  return sortRegionsForPicker(regionsPlugin.getRegions().filter((r) => {
    if (isTrimRegion(r) || isLinkOverlay(r) || !isRegionVisible(r)) return false;
    return r.end > region.start && r.start < region.end;
  }));
}

function regionsAtClientPoint(clientX, clientY) {
  const pxPerSec = getEffectivePxPerSec();
  if (!(pxPerSec > 0) || !Number.isFinite(clientX)) return [];
  return regionsAtTime(contentXFromClientX(clientX) / pxPerSec);
}

function clientPointForRegionEvent(region, e) {
  if (Number.isFinite(e?.clientX) && Number.isFinite(e?.clientY)) {
    return { x: e.clientX, y: e.clientY };
  }
  const box = region?.element?.getBoundingClientRect?.();
  if (box) {
    return { x: box.left + Math.min(48, Math.max(8, box.width / 2)), y: box.top + 10 };
  }
  return { x: 80, y: 80 };
}

function overlappingHitsForClick(region, e) {
  const point = clientPointForRegionEvent(region, e);
  const atPoint = regionsAtClientPoint(point.x, point.y);
  if (atPoint.length > 1) return { hits: atPoint, ...point };
  const overlapping = regionsOverlappingRegion(region);
  if (overlapping.length > 1) return { hits: overlapping, ...point };
  const atMid = regionsAtTime((region.start + region.end) / 2);
  if (atMid.length > 1) return { hits: atMid, ...point };
  return { hits: atPoint.length ? atPoint : [region], ...point };
}

function showOverlapPicker(regions, clientX, clientY) {
  const el = document.getElementById("overlapPicker");
  if (!el || regions.length < 2) {
    hideOverlapPicker();
    return;
  }
  overlapPickerRegions = regions;
  ignorePickerHideUntil = Date.now() + REGION_DOUBLE_CLICK_MS + 80;
  el.innerHTML = "";
  const title = document.createElement("span");
  title.className = "overlap-picker-title";
  title.textContent = "Select label";
  el.appendChild(title);
  regions.forEach((region) => {
    const btn = document.createElement("button");
    btn.type = "button";
    const d = region.data || {};
    const name = regionCaptionText(region);
    btn.textContent = name;
    btn.title = `${formatTime(region.start)} – ${formatTime(region.end)}`;
    btn.className = "overlap-picker-item";
    if (isRegionSelected(region)) btn.classList.add("active");
    btn.style.setProperty("--swatch", TYPE_COLORS[d.type] || "#2d6cdf");
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      if (ev.ctrlKey || ev.metaKey) toggleRegionInSelection(region);
      else selectRegion(region);
      el.querySelectorAll(".overlap-picker-item").forEach((item, i) => {
        item.classList.toggle("active", isRegionSelected(regions[i]));
      });
    });
    el.appendChild(btn);
  });
  el.classList.remove("hidden");
  el.style.left = "0px";
  el.style.top = "0px";
  const rect = el.getBoundingClientRect();
  const pad = 8;
  let left = clientX + 12;
  let top = clientY + 12;
  if (left + rect.width + pad > window.innerWidth) {
    left = Math.max(pad, window.innerWidth - rect.width - pad);
  }
  if (top + rect.height + pad > window.innerHeight) {
    top = Math.max(pad, clientY - rect.height - 12);
  }
  el.style.left = `${left}px`;
  el.style.top = `${top}px`;
}

function setupOverlapPicker() {
  document.addEventListener("pointerdown", (e) => {
    const el = document.getElementById("overlapPicker");
    if (!el || el.classList.contains("hidden")) return;
    if (el.contains(e.target)) return;
    if (e.target.closest?.("#waveformScroll")) return;
    hideOverlapPicker();
  });
  const scroll = document.getElementById("waveformScroll");
  scroll?.addEventListener("click", (e) => {
    if (Date.now() < ignorePickerHideUntil) return;
    if (e.target.closest?.("#overlapPicker")) return;
    hideOverlapPicker();
    if (e.detail === 2 || marqueeSelecting || rectSelecting) return;
    if (!selectedRegions.length) return;
    const hits = regionsAtClientPoint(e.clientX, e.clientY);
    if (hits.some((r) => isRegionSelected(r))) return;
    clearRegionSelection();
    updateMultiSelectionInspector();
  });
  scroll?.addEventListener("dblclick", (e) => {
    if (e.target.closest?.("#scrubber")) return;
    if (isTextEntryTarget(e.target)) return;
    const hits = regionsAtClientPoint(e.clientX, e.clientY);
    if (!hits.length) return;
    e.preventDefault();
    e.stopPropagation();
    if (hits.length > 1) {
      showOverlapPicker(hits, e.clientX, e.clientY);
      return;
    }
    selectRegion(hits[0]);
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
  syncZoomSlider();
}

async function setViewMode(mode) {
  viewMode = mode;
  document.getElementById("viewNormalBtn").classList.toggle("active", mode === "normal");
  document.getElementById("viewAlignmentBtn").classList.toggle("active", mode === "alignment");
  document.getElementById("scorePanel").classList.toggle("collapsed", mode === "alignment");
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
        requestAnimationFrame(() => {
          renderScoreXml(cachedScoreXml, cachedScoreMeta || {});
        });
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
    renderScoreCompare();
  } catch (err) {
    noteAlignmentData = null;
    const info = document.getElementById("alignmentInfo");
    if (viewMode === "alignment" && info) {
      info.classList.remove("hidden");
      info.innerHTML = `<p class="summary">Could not load alignment: ${err.message}</p>`;
    }
    renderScoreCompare();
  }
}

function clearAlignmentOverlays() {
  document.getElementById("noteBoundaries").innerHTML = "";
  const ewma = document.getElementById("ewmaStrip");
  if (ewma) ewma.innerHTML = "";
  updateOverlayTop();
}

function renderAlignmentOverlays() {
  if (!noteAlignmentData?.events?.length) {
    clearAlignmentOverlays();
    return;
  }
  const pxPerSec = getEffectivePxPerSec();
  renderNoteBoundaries(noteAlignmentData.events, pxPerSec);
  renderEwmaStrip(noteAlignmentData.events, pxPerSec);
}

function renderNoteBoundaries(events, pxPerSec) {
  const container = document.getElementById("noteBoundaries");
  container.innerHTML = "";
  const width = getAlignmentContentWidth(pxPerSec);
  container.style.width = `${width}px`;
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

function appendDurationDots(svgParts, cx, cy, dots, color = "#222") {
  for (let i = 0; i < dots; i += 1) {
    svgParts.push(
      `<circle cx="${cx + 10 + i * 5}" cy="${cy}" r="1.6" fill="${color}"/>`,
    );
  }
}

function appendNoteGlyph(svgParts, cx, cy, ql, stemUp, color = "#222") {
  const { base, dots } = classifyDurationQl(ql);
  const open = base >= 2;
  const rx = base >= 4 ? 7 : 5;
  const ry = base >= 4 ? 5 : 4;
  svgParts.push(
    `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" ` +
      `fill="${open ? "none" : color}" stroke="${color}" stroke-width="1.5" ` +
      `transform="rotate(-20 ${cx} ${cy})"/>`,
  );
  if (base <= 2) {
    const stemH = 22;
    const sx = stemUp ? cx + rx - 1 : cx - rx + 1;
    const sy1 = cy;
    const sy2 = stemUp ? cy - stemH : cy + stemH;
    svgParts.push(
      `<line x1="${sx}" y1="${sy1}" x2="${sx}" y2="${sy2}" stroke="${color}" stroke-width="1.4"/>`,
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
          `fill="none" stroke="${color}" stroke-width="1.4"/>`,
      );
    }
  }
  appendDurationDots(svgParts, cx + (base >= 4 ? 4 : 0), cy, dots, color);
}

function appendRestGlyph(svgParts, cx, staffBottomY, lineGap, ql, color = "#333") {
  const { base, dots } = classifyDurationQl(ql);
  const mid = staffBottomY - 2 * lineGap;
  if (base >= 4) {
    // whole rest: hang from 2nd line from top
    const y = staffBottomY - 3 * lineGap;
    svgParts.push(`<rect x="${cx - 6}" y="${y}" width="12" height="4" fill="${color}"/>`);
  } else if (base >= 2) {
    const y = staffBottomY - 2 * lineGap - 4;
    svgParts.push(`<rect x="${cx - 6}" y="${y}" width="12" height="4" fill="${color}"/>`);
  } else if (base >= 1) {
    // quarter rest (simplified zigzag)
    svgParts.push(
      `<path d="M${cx - 2} ${mid - 10} L${cx + 3} ${mid - 4} L${cx - 3} ${mid + 2} L${cx + 2} ${mid + 10}" ` +
        `fill="none" stroke="${color}" stroke-width="1.8" stroke-linejoin="round"/>`,
    );
  } else {
    // eighth / shorter: flag rest
    const flags = base <= 0.125 ? 3 : base <= 0.25 ? 2 : 1;
    svgParts.push(
      `<line x1="${cx}" y1="${mid - 10}" x2="${cx}" y2="${mid + 8}" stroke="${color}" stroke-width="1.5"/>`,
    );
    for (let f = 0; f < flags; f += 1) {
      const fy = mid - 10 + f * 6;
      svgParts.push(
        `<path d="M${cx} ${fy} q6 2 8 7" fill="none" stroke="${color}" stroke-width="1.5"/>`,
        `<circle cx="${cx + 8}" cy="${fy + 8}" r="2" fill="${color}"/>`,
      );
    }
  }
  appendDurationDots(svgParts, cx + 4, mid, dots, color);
}

function eventMidi(ev) {
  if (ev?.midi != null) return ev.midi;
  return pitchToMidi(ev?.pitch);
}

function soundingNoteId(ev) {
  if (ev?.note_id) return ev.note_id;
  if (ev?.sounding_index == null) return ev?.id;
  return `note_${String(ev.sounding_index).padStart(4, "0")}`;
}

function eventRefStart(ev) {
  return Number(ev?.ref_start ?? ev?.perf_start) || 0;
}

function eventRefEnd(ev) {
  if (ev?.ref_end != null) return Number(ev.ref_end);
  if (ev?.perf_end != null) return Number(ev.perf_end);
  return eventRefStart(ev) + 0.05;
}

function eventPerfStart(ev) {
  if (ev?.perf_start != null) return Number(ev.perf_start);
  return eventRefStart(ev);
}

function eventPerfEnd(ev) {
  if (ev?.perf_end != null) return Number(ev.perf_end);
  return eventRefEnd(ev);
}

function buildStaffSvg(events, pxPerSec, width, { fill = "#f8f8f8", startOf, xs } = {}) {
  const startAt = startOf || ((ev) => ev.perf_start || 0);
  const noteXs = xs || unpackStaffXs(events, pxPerSec, startAt);
  const lineGap = 9;
  const padding = 10;
  const ledgerHalfWidth = 14;
  const staffBottomStep = 0;
  const staffTopStep = 4;
  const KIND_INK = {
    match: "#222",
    sub: "#c45c12",
    extra: "#c0392b",
    miss: "#7a7a7a",
  };

  let minStep = 0;
  let maxStep = 4;
  events.forEach((ev) => {
    if (ev.is_rest) return;
    const midi = eventMidi(ev);
    if (midi == null) return;
    const steps = midiToStaffSteps(midi);
    minStep = Math.min(minStep, Math.floor(steps));
    maxStep = Math.max(maxStep, Math.ceil(steps));
  });
  minStep = Math.min(minStep, -1);
  maxStep = Math.max(maxStep, 5);

  const height = (maxStep - minStep) * lineGap + padding * 2;
  const staffBottomY = padding + (maxStep - staffBottomStep) * lineGap;
  const staffMidY = staffBottomY - 2 * lineGap;
  const lastX = noteXs.length ? noteXs[noteXs.length - 1] : 0;
  const drawWidth = Math.max(width, lastX + 16);
  const svgParts = [
    `<svg width="${drawWidth}" height="${height}" xmlns="http://www.w3.org/2000/svg">`,
    `<rect width="100%" height="100%" fill="${fill}"/>`,
  ];
  for (let s = 0; s <= 4; s += 1) {
    const y = staffBottomY - s * lineGap;
    svgParts.push(
      `<line x1="0" y1="${y}" x2="${drawWidth}" y2="${y}" stroke="#999" stroke-width="1"/>`,
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

  events.forEach((ev, i) => {
    const x = noteXs[i];
    const ql = Number(ev.duration_ql) || 1;
    const ink = KIND_INK[ev.alignKind] || (ev.is_rest ? "#333" : "#222");
    if (ev.is_rest) {
      appendRestGlyph(svgParts, x, staffBottomY, lineGap, ql, ink);
    } else {
      const midi = eventMidi(ev);
      const cy = midi != null ? midiToStaffY(midi, staffBottomY, lineGap) : staffMidY;
      const noteSteps = midi != null ? midiToStaffSteps(midi) : null;
      if (noteSteps != null) appendLedgerLines(noteSteps, x);
      appendNoteGlyph(svgParts, x, cy, ql, cy >= staffMidY, ink);
    }
  });
  svgParts.push("</svg>");
  return { html: svgParts.join(""), height, noteXs, width: drawWidth };
}

function getTranscribedNotes() {
  return noteAlignmentData?.transcribed_notes || [];
}

function midiToPitchName(midi) {
  if (midi == null || !Number.isFinite(Number(midi))) return null;
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const m = Math.round(Number(midi));
  const pc = ((m % 12) + 12) % 12;
  return `${names[pc]}${Math.floor(m / 12) - 1}`;
}

function referenceByScoreIndex(events = getScoreEvents()) {
  const byIndex = new Map();
  events.forEach((ev) => {
    if (ev.sounding_index != null) byIndex.set(ev.sounding_index, ev);
    else if (ev.score_index != null) byIndex.set(ev.score_index, ev);
  });
  return byIndex;
}

function mappedScoreIndices(transcribed = getTranscribedNotes()) {
  const mapped = new Set();
  transcribed.forEach((note) => {
    if (note.score_index != null) mapped.add(note.score_index);
  });
  return mapped;
}

function transcribedAlignKind(note, refByIndex) {
  if (note.score_index == null) return "extra";
  const ref = refByIndex.get(note.score_index);
  if (!ref) return "extra";
  const played = eventMidi(note);
  const written = eventMidi(ref);
  if (played != null && written != null && played !== written) return "sub";
  return "match";
}

function renderScoreCompare() {
  renderTranscriptionStrip();
  renderMelodyStrip();
  renderScoreAlignOverlay();
}

function renderTranscriptionStrip() {
  const container = document.getElementById("transcriptionStrip");
  if (!container) return;
  const transcribed = getTranscribedNotes();
  const pxPerSec = getEffectivePxPerSec();
  const width = getAlignmentContentWidth(pxPerSec);
  container.style.width = `${width}px`;
  container.style.minWidth = `${width}px`;
  if (!transcribed.length) {
    container.dataset.renderKey = "";
    container.innerHTML =
      `<p class="melody-empty">No audio transcription for this sample yet. ` +
      `Apply a score segment (or Re-align) so Basic Pitch and the joint decoder can transcribe the take.</p>`;
    return;
  }
  const refByIndex = referenceByScoreIndex();
  const events = transcribed.map((note) => ({
    ...note,
    alignKind: transcribedAlignKind(note, refByIndex),
  }));
  const lastEnd = events.length ? eventPerfEnd(events[events.length - 1]) : 0;
  const renderKey = `trans:${events.length}:${getScoreEvents().length}:${width}:${pxPerSec}:${events[0]?.id}:${lastEnd}`;
  if (container.dataset.renderKey === renderKey && container.querySelector("svg")) {
    return;
  }
  const noteXs = unpackStaffXs(events, pxPerSec, eventPerfStart);
  const staff = buildStaffSvg(events, pxPerSec, width, {
    fill: "#eef4fb",
    startOf: eventPerfStart,
    xs: noteXs,
  });
  container.innerHTML = staff.html;
  container.style.height = `${staff.height}px`;
  container.dataset.renderKey = renderKey;
  events.forEach((ev, i) => {
    const hit = document.createElement("button");
    hit.type = "button";
    hit.className = "trans-hit";
    hit.dataset.transIndex = String(ev.index ?? i);
    if (ev.score_index != null) hit.dataset.scoreIndex = String(ev.score_index);
    const x0 = noteXs[i];
    const nextX = i + 1 < noteXs.length ? noteXs[i + 1] : x0 + MIN_NOTE_GAP_PX + 6;
    hit.style.left = `${Math.max(0, x0 - 3)}px`;
    hit.style.width = `${Math.max(8, nextX - x0)}px`;
    const kind = ev.alignKind;
    const ref = ev.score_index != null ? refByIndex.get(ev.score_index) : null;
    const played = ev.pitch || midiToPitchName(ev.midi) || "?";
    const written = ref ? (ref.pitch || midiToPitchName(eventMidi(ref)) || "?") : "—";
    hit.title = kind === "extra"
      ? `Extra ${played}`
      : `${played} → ${written}${ref?.measure != null ? ` · m${ref.measure}` : ""}`;
    container.appendChild(hit);
  });
}

function renderScoreAlignOverlay() {
  const container = document.getElementById("scoreAlignOverlay");
  if (!container) return;
  const transcribed = getTranscribedNotes();
  const reference = getScoreEvents();
  const pxPerSec = getEffectivePxPerSec();
  const width = getAlignmentContentWidth(pxPerSec);
  const height = 44;
  container.style.width = `${width}px`;
  container.style.minWidth = `${width}px`;
  container.style.height = `${height}px`;
  if (!transcribed.length || !reference.length) {
    container.innerHTML = "";
    return;
  }
  const refByIndex = referenceByScoreIndex(reference);
  const transXs = unpackStaffXs(transcribed, pxPerSec, eventPerfStart);
  const refXs = unpackStaffXs(reference, pxPerSec, eventPerfStart);
  const refXByScore = new Map();
  reference.forEach((ev, i) => {
    if (ev.sounding_index != null) refXByScore.set(ev.sounding_index, refXs[i]);
    if (ev.score_index != null) refXByScore.set(ev.score_index, refXs[i]);
  });
  const parts = [
    `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`,
    `<rect width="100%" height="100%" fill="#1a1a1a"/>`,
  ];
  transcribed.forEach((note, i) => {
    if (note.score_index == null) return;
    const x1 = transXs[i];
    const x2 = refXByScore.get(note.score_index);
    if (x1 == null || x2 == null) return;
    const kind = transcribedAlignKind(note, refByIndex);
    const mid = height / 2;
    parts.push(
      `<path class="pair-link ${kind}" data-trans-index="${note.index ?? i}" ` +
        `data-score-index="${note.score_index}" ` +
        `d="M${x1.toFixed(1)} 0 C${x1.toFixed(1)} ${mid.toFixed(1)}, ${x2.toFixed(1)} ${mid.toFixed(1)}, ${x2.toFixed(1)} ${height}"/>`,
    );
  });
  parts.push("</svg>");
  container.innerHTML = parts.join("");
}

function highlightScorePair(transIndex, scoreIndex) {
  document.querySelectorAll(".trans-hit.hover").forEach((el) => el.classList.remove("hover"));
  document.querySelectorAll(".melody-hit.pair-hover").forEach((el) => el.classList.remove("pair-hover"));
  document.querySelectorAll(".score-align-overlay path.pair-link.active").forEach((el) => {
    el.classList.remove("active");
  });
  if (transIndex == null && scoreIndex == null) return;
  if (transIndex != null) {
    document.querySelectorAll(`.trans-hit[data-trans-index="${transIndex}"]`)
      .forEach((el) => el.classList.add("hover"));
  }
  if (scoreIndex != null) {
    document.querySelectorAll(`.melody-hit[data-score-index="${scoreIndex}"]`)
      .forEach((el) => el.classList.add("pair-hover"));
  }
  const sel = transIndex != null
    ? `.score-align-overlay path.pair-link[data-trans-index="${transIndex}"]`
    : `.score-align-overlay path.pair-link[data-score-index="${scoreIndex}"]`;
  document.querySelectorAll(sel).forEach((el) => el.classList.add("active"));
}

function setupScoreCompareHover() {
  const root = document.getElementById("scoreCompare");
  if (!root) return;
  root.addEventListener("pointerover", (e) => {
    const trans = e.target.closest?.(".trans-hit");
    if (trans) {
      const t = trans.dataset.transIndex === undefined ? null : parseInt(trans.dataset.transIndex, 10);
      const s = trans.dataset.scoreIndex === undefined ? null : parseInt(trans.dataset.scoreIndex, 10);
      highlightScorePair(Number.isNaN(t) ? null : t, Number.isNaN(s) ? null : s);
      return;
    }
    const hit = e.target.closest?.(".melody-hit");
    if (hit) {
      const s = hit.dataset.scoreIndex === undefined ? null : parseInt(hit.dataset.scoreIndex, 10);
      const t = hit.dataset.transIndex === undefined ? null : parseInt(hit.dataset.transIndex, 10);
      highlightScorePair(Number.isNaN(t) ? null : t, Number.isNaN(s) ? null : s);
      return;
    }
    const link = e.target.closest?.("path.pair-link");
    if (link) {
      const t = link.dataset.transIndex === undefined ? null : parseInt(link.dataset.transIndex, 10);
      const s = link.dataset.scoreIndex === undefined ? null : parseInt(link.dataset.scoreIndex, 10);
      highlightScorePair(Number.isNaN(t) ? null : t, Number.isNaN(s) ? null : s);
    }
  });
  root.addEventListener("pointerleave", () => highlightScorePair(null, null));
}

function getScoreEvents() {
  return scoreEventData?.events || [];
}

function melodyCoreRange(events, coreIds) {
  if (!events.length || !coreIds?.length) return null;
  const wanted = new Set(coreIds);
  const idxs = [];
  events.forEach((ev, i) => {
    if (wanted.has(ev.id)) idxs.push(i);
  });
  if (!idxs.length) return null;
  return { lo: Math.min(...idxs), hi: Math.max(...idxs) };
}

function expandMelodyPad(events, coreLo, coreHi, padNotes = MELODY_PAD_NOTES) {
  let lo = coreLo;
  let notes = 0;
  for (let i = coreLo - 1; i >= 0 && notes < padNotes; i -= 1) {
    lo = i;
    if (!events[i].is_rest) notes += 1;
  }
  let hi = coreHi;
  notes = 0;
  for (let i = coreHi + 1; i < events.length && notes < padNotes; i += 1) {
    hi = i;
    if (!events[i].is_rest) notes += 1;
  }
  return { lo, hi };
}

function melodyFieldsFromCore(events, coreIds) {
  const core = melodyCoreRange(events, coreIds);
  if (!core) return null;
  const pad = expandMelodyPad(events, core.lo, core.hi);
  const span = events.slice(pad.lo, pad.hi + 1);
  const sounding = span.filter((ev) => !ev.is_rest && ev.sounding_index != null);
  const coreSounding = events
    .slice(core.lo, core.hi + 1)
    .filter((ev) => !ev.is_rest && ev.sounding_index != null);
  const fields = {
    core_note_ids: events.slice(core.lo, core.hi + 1).map((ev) => ev.id),
    score_part: {
      start_note_index: sounding.length ? sounding[0].sounding_index : pad.lo,
      end_note_index: sounding.length
        ? sounding[sounding.length - 1].sounding_index
        : pad.hi,
      pad_notes: MELODY_PAD_NOTES,
      start_measure: span[0]?.measure ?? null,
      end_measure: span[span.length - 1]?.measure ?? null,
      core_start_note_index: coreSounding.length
        ? coreSounding[0].sounding_index
        : null,
      core_end_note_index: coreSounding.length
        ? coreSounding[coreSounding.length - 1].sounding_index
        : null,
    },
    pitches: sounding.map((ev) => eventMidi(ev)).filter((p) => p != null),
    note_ids: sounding.map((ev) => soundingNoteId(ev)),
    measure_number: events[core.lo]?.measure ?? null,
  };
  return fields;
}

function applyMelodyToRegion(region, coreIds, { undo = true } = {}) {
  if (!region || isTrimRegion(region) || isLinkOverlay(region)) return;
  const events = getScoreEvents();
  const fields = melodyFieldsFromCore(events, coreIds);
  if (undo) pushUndoFromIdle();
  if (!fields) {
    delete region.data.score_part;
    delete region.data.pitches;
    delete region.data.note_ids;
    delete region.data.core_note_ids;
  } else {
    region.data = { ...(region.data || {}), ...fields };
  }
  if (!undoSuspended) captureIdleSnapshot();
  const container = document.getElementById("melodyStrip");
  if (container?.querySelector("svg") && events.length) {
    paintMelodyHits(container, events);
    updateMelodyPanel();
  } else {
    refreshMelodyUi();
  }
}

function coreIdsFromStoredLabel(label, events = getScoreEvents()) {
  if (label?.core_note_ids?.length) return label.core_note_ids;
  const part = label?.score_part;
  if (!part || !events.length) return [];
  if (part.core_start_note_index != null && part.core_end_note_index != null) {
    return events
      .filter((ev) => (
        ev.sounding_index != null
        && ev.sounding_index >= part.core_start_note_index
        && ev.sounding_index <= part.core_end_note_index
      ))
      .map((ev) => ev.id);
  }
  if (part.start_note_index == null || part.end_note_index == null) return [];
  const pad = part.pad_notes ?? MELODY_PAD_NOTES;
  const sounding = events.filter((ev) => ev.sounding_index != null);
  const i0 = sounding.findIndex((ev) => ev.sounding_index === part.start_note_index);
  const i1 = sounding.findIndex((ev) => ev.sounding_index === part.end_note_index);
  if (i0 < 0 || i1 < 0) return [];
  const core0 = Math.min(i1, i0 + pad);
  const core1 = Math.max(i0, i1 - pad);
  if (core1 < core0) return sounding.slice(i0, i1 + 1).map((ev) => ev.id);
  return sounding.slice(core0, core1 + 1).map((ev) => ev.id);
}

function selectedMelodyCoreIds() {
  if (!selectedRegion) return [];
  const stored = selectedRegion.data?.core_note_ids;
  if (stored?.length) return stored;
  return coreIdsFromStoredLabel(selectedRegion.data);
}

function updateMelodyPanel(region = selectedRegion) {
  const info = document.getElementById("melodyInfo");
  const clearBtn = document.getElementById("clearMelodyBtn");
  if (!info) return;
  const editable = !!(region && !isTrimRegion(region) && !isLinkOverlay(region));
  if (clearBtn) clearBtn.disabled = !editable || !(region.data?.core_note_ids?.length);
  if (!editable) {
    info.textContent =
      "Drag on the reference-score staff under the waveform to mark the erred notes/rests. That creates a label; two notes of padding are added automatically on each side.";
    return;
  }
  const events = getScoreEvents();
  const core = melodyCoreRange(events, region.data?.core_note_ids || []);
  if (!core) {
    info.textContent =
      "Drag on the reference score to mark the erred notes/rests. Padding (2 notes each side) is added automatically.";
    return;
  }
  const pad = expandMelodyPad(events, core.lo, core.hi);
  const coreLabel = events
    .slice(core.lo, core.hi + 1)
    .map((ev) => (ev.is_rest ? "rest" : (ev.pitch || "?")))
    .join(" · ");
  const startM = events[pad.lo]?.measure;
  const endM = events[pad.hi]?.measure;
  const meas = startM != null && endM != null
    ? `m${startM}${endM !== startM ? `–${endM}` : ""}`
    : "";
  info.textContent =
    `Core: ${coreLabel}. Saved melody ${meas} with ${MELODY_PAD_NOTES} notes of pad on each side.`;
}

function refreshMelodyUi() {
  renderScoreCompare();
  updateMelodyPanel();
}

function paintMelodyHits(container, events) {
  const coreIds = new Set(selectedMelodyCoreIds());
  const core = melodyCoreRange(events, [...coreIds]);
  const pad = core ? expandMelodyPad(events, core.lo, core.hi) : null;
  const mapped = mappedScoreIndices();
  const transByScore = new Map();
  getTranscribedNotes().forEach((note) => {
    if (note.score_index != null) transByScore.set(note.score_index, note.index);
  });
  container.querySelectorAll(".melody-hit").forEach((hit) => {
    const i = parseInt(hit.dataset.index, 10);
    if (Number.isNaN(i) || !events[i]) return;
    const ev = events[i];
    const inCore = !!(core && i >= core.lo && i <= core.hi);
    const inPad = !!(pad && i >= pad.lo && i <= pad.hi && !inCore);
    const scoreIndex = ev.sounding_index ?? ev.score_index;
    const miss = !ev.is_rest && scoreIndex != null && mapped.size > 0 && !mapped.has(scoreIndex);
    hit.classList.toggle("core", inCore);
    hit.classList.toggle("pad", inPad);
    hit.classList.toggle("miss", miss && !inCore && !inPad);
    if (scoreIndex != null) hit.dataset.scoreIndex = String(scoreIndex);
    else delete hit.dataset.scoreIndex;
    if (transByScore.has(scoreIndex)) hit.dataset.transIndex = String(transByScore.get(scoreIndex));
    else delete hit.dataset.transIndex;
  });
}

function renderMelodyStrip() {
  const container = document.getElementById("melodyStrip");
  if (!container) return;
  const events = getScoreEvents();
  const pxPerSec = getEffectivePxPerSec();
  const width = getAlignmentContentWidth(pxPerSec);
  const canDrag = events.length > 0;
  container.style.width = `${width}px`;
  container.style.minWidth = `${width}px`;
  container.classList.toggle("armed", canDrag);
  container.classList.toggle("idle", !canDrag);
  if (!events.length) {
    container.dataset.renderKey = "";
    container.innerHTML =
      `<p class="melody-empty">No reference score snippet loaded for this sample.</p>`;
    return;
  }

  const lastPerf = events.length ? eventPerfEnd(events[events.length - 1]) : 0;
  const renderKey = `perf:${events.length}:${getTranscribedNotes().length}:${width}:${pxPerSec}:${events[0]?.id}:${lastPerf}`;
  if (container.dataset.renderKey === renderKey && container.querySelector("svg")) {
    paintMelodyHits(container, events);
    return;
  }

  const noteXs = unpackStaffXs(events, pxPerSec, eventPerfStart);
  const mapped = mappedScoreIndices();
  const staffEvents = events.map((ev) => {
    const scoreIndex = ev.sounding_index ?? ev.score_index;
    const miss = !ev.is_rest && scoreIndex != null && mapped.size > 0 && !mapped.has(scoreIndex);
    return miss ? { ...ev, alignKind: "miss" } : ev;
  });
  const staff = buildStaffSvg(staffEvents, pxPerSec, width, {
    fill: "#f4f1ea",
    startOf: eventPerfStart,
    xs: noteXs,
  });
  container.innerHTML = staff.html;
  container.style.height = `${staff.height}px`;
  container.dataset.renderKey = renderKey;
  const transByScore = new Map();
  getTranscribedNotes().forEach((note) => {
    if (note.score_index != null) transByScore.set(note.score_index, note.index);
  });
  events.forEach((ev, i) => {
    const hit = document.createElement("button");
    hit.type = "button";
    hit.className = "melody-hit";
    hit.dataset.index = String(i);
    const scoreIndex = ev.sounding_index ?? ev.score_index;
    if (scoreIndex != null) hit.dataset.scoreIndex = String(scoreIndex);
    if (transByScore.has(scoreIndex)) hit.dataset.transIndex = String(transByScore.get(scoreIndex));
    const x0 = noteXs[i];
    const nextX = i + 1 < noteXs.length ? noteXs[i + 1] : x0 + MIN_NOTE_GAP_PX + 6;
    hit.style.left = `${Math.max(0, x0 - 3)}px`;
    hit.style.width = `${Math.max(8, nextX - x0)}px`;
    hit.title = `${ev.is_rest ? "Rest" : (ev.pitch || "note")}${ev.measure != null ? ` · m${ev.measure}` : ""}`;
    container.appendChild(hit);
  });
  paintMelodyHits(container, events);
}

function melodyIndexFromTarget(target) {
  const hit = target?.closest?.(".melody-hit");
  if (!hit) return null;
  const idx = parseInt(hit.dataset.index, 10);
  return Number.isNaN(idx) ? null : idx;
}

function melodyIndexFromClientX(clientX) {
  const events = getScoreEvents();
  if (!events.length) return null;
  const px = getEffectivePxPerSec();
  if (!(px > 0)) return null;
  const xs = unpackStaffXs(events, px, eventPerfStart);
  const x = contentXFromClientX(clientX);
  let best = 0;
  let bestDist = Infinity;
  xs.forEach((cx, i) => {
    const next = i + 1 < xs.length ? xs[i + 1] : cx + 16;
    if (x >= cx && x < next) {
      best = i;
      bestDist = 0;
      return;
    }
    const dist = Math.abs(x - cx);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  });
  return best;
}

function timesFromEventRange(events, lo, hi) {
  const slice = events.slice(lo, hi + 1);
  let start = Infinity;
  let end = -Infinity;
  slice.forEach((ev) => {
    const t0 = eventPerfStart(ev);
    const t1 = eventPerfEnd(ev);
    start = Math.min(start, t0);
    end = Math.max(end, t1);
  });
  if (!Number.isFinite(start)) start = 0;
  if (!Number.isFinite(end) || end <= start) end = start + 0.05;
  const duration = wavesurfer?.getDuration?.() || end;
  return {
    start: Math.max(0, start),
    end: Math.min(duration, Math.max(end, start + 0.05)),
  };
}

function setRegionTimes(region, start, end) {
  if (!region) return;
  ignoreRegionUpdateUndo = true;
  if (typeof region.setOptions === "function") {
    region.setOptions({ start, end });
  }
  region.start = start;
  region.end = end;
  if (region.data) {
    region.data.start_time = start;
    region.data.end_time = end;
  }
}

function createMelodyLabelRegion(times) {
  const type = currentLabelType();
  const region = addRegion(
    {
      id: generateLabelId(),
      source: "manual",
      start_time: times.start,
      end_time: times.end,
      type,
      severity: parseInt(document.getElementById("severity")?.value, 10) || 3,
      comment: null,
    },
    false,
  );
  selectRegion(region);
  return region;
}

function commitMelodyCoreRange(lo, hi, { undo = false } = {}) {
  const events = getScoreEvents();
  if (!events.length) return;
  const a = Math.max(0, Math.min(lo, hi));
  const b = Math.min(events.length - 1, Math.max(lo, hi));
  const ids = events.slice(a, b + 1).map((ev) => ev.id);
  const times = timesFromEventRange(events, a, b);
  if (undo) pushUndoFromIdle();
  let region = melodyDrag?.region || null;
  if (!region) {
    region = createMelodyLabelRegion(times);
    if (melodyDrag) melodyDrag.region = region;
  }
  if (!region) return;
  setRegionTimes(region, times.start, times.end);
  applyMelodyToRegion(region, ids, { undo: false });
  if (isRegionSelected(region) && selectedRegions.length === 1) {
    updateSelectionInfo(region);
    applyRegionCaption(region);
  }
}

function setupMelodyStrip() {
  const container = document.getElementById("melodyStrip");
  const clearBtn = document.getElementById("clearMelodyBtn");
  if (clearBtn) {
    clearBtn.onclick = () => {
      if (!selectedRegion || isTrimRegion(selectedRegion) || isLinkOverlay(selectedRegion)) return;
      applyMelodyToRegion(selectedRegion, []);
    };
  }
  if (!container) return;
  container.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    if (!getScoreEvents().length) return;
    const idx = melodyIndexFromClientX(e.clientX) ?? melodyIndexFromTarget(e.target);
    if (idx == null) return;
    e.preventDefault();
    e.stopPropagation();
    melodyDrag = { start: idx, end: idx, region: null };
    container.setPointerCapture?.(e.pointerId);
    commitMelodyCoreRange(idx, idx, { undo: true });
  });
  container.addEventListener("pointermove", (e) => {
    if (!melodyDrag) return;
    const idx = melodyIndexFromClientX(e.clientX);
    if (idx == null || idx === melodyDrag.end) return;
    melodyDrag.end = idx;
    commitMelodyCoreRange(melodyDrag.start, melodyDrag.end, { undo: false });
  });
  const endDrag = () => {
    melodyDrag = null;
  };
  container.addEventListener("pointerup", endDrag);
  container.addEventListener("pointercancel", endDrag);
}

async function loadScoreEvents() {
  if (!currentSample) return;
  try {
    const res = await fetch(`/api/samples/${currentSample}/score-events`);
    if (!res.ok) throw new Error(await res.text());
    scoreEventData = await res.json();
  } catch {
    scoreEventData = null;
  }
  if (regionsPlugin && scoreEventData?.events?.length) {
    regionsPlugin.getRegions().forEach((region) => {
      if (isTrimRegion(region) || isLinkOverlay(region)) return;
      if (region.data?.core_note_ids?.length) return;
      const inferred = coreIdsFromStoredLabel(region.data);
      if (inferred.length) region.data.core_note_ids = inferred;
    });
  }
  if (!userZoomed) fitWaveformToContainer();
  else refreshMelodyUi();
}

function mergeConsecutiveRests(events, eps = 1e-3) {
  if (!events?.length) return [];
  const byPart = new Map();
  events.forEach((ev) => {
    const part = ev.part ?? 0;
    if (!byPart.has(part)) byPart.set(part, []);
    byPart.get(part).push(ev);
  });
  const merged = [];
  [...byPart.keys()].sort((a, b) => a - b).forEach((part) => {
    const partEvents = byPart.get(part);
    let i = 0;
    while (i < partEvents.length) {
      const cur = { ...partEvents[i] };
      if (cur.is_rest) {
        let j = i + 1;
        while (j < partEvents.length) {
          const nxt = partEvents[j];
          if (!nxt.is_rest) break;
          if (Math.abs(nxt.ref_start - cur.ref_end) > eps) break;
          cur.ref_end = nxt.ref_end;
          cur.perf_end = nxt.perf_end;
          if (cur.duration_ql != null && nxt.duration_ql != null) {
            cur.duration_ql = Number(cur.duration_ql) + Number(nxt.duration_ql);
          }
          j += 1;
        }
        merged.push(cur);
        i = j;
      } else {
        merged.push(cur);
        i += 1;
      }
    }
  });
  return merged;
}

function computeEwmaSeries(events, alpha = EWMA_ALPHA) {
  const byPart = new Map();
  mergeConsecutiveRests(events).forEach((ev) => {
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
  const width = getAlignmentContentWidth(pxPerSec);
  const height = EWMA_STRIP_HEIGHT;
  const padTop = 36;
  const padBot = 12;
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
  const legendX = Math.max(8, width - 210);

  const svgParts = [
    `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`,
    `<rect width="100%" height="100%" fill="#141414"/>`,
    `<text x="8" y="14" fill="#9ab" font-size="11">Tempo ratio (perf/ref duration)</text>`,
    `<line x1="0" y1="${yOne}" x2="${width}" y2="${yOne}" stroke="#355" stroke-width="1" stroke-dasharray="4 3"/>`,
    `<text x="8" y="${yOne - 3}" fill="#6a8" font-size="9">Match (1.0)</text>`,
    `<path d="${ratioPath}" fill="none" stroke="#888" stroke-width="1.2" opacity="0.85"/>`,
    `<path d="${ewmaPath}" fill="none" stroke="#6af" stroke-width="2.2"/>`,
    // Legend
    `<rect x="${legendX - 6}" y="2" width="208" height="28" rx="3" fill="#1a1a1a" stroke="#333"/>`,
    `<line x1="${legendX}" y1="10" x2="${legendX + 18}" y2="10" stroke="#888" stroke-width="1.5"/>`,
    `<text x="${legendX + 22}" y="13" fill="#bbb" font-size="10">Instantaneous ratio</text>`,
    `<line x1="${legendX}" y1="22" x2="${legendX + 18}" y2="22" stroke="#6af" stroke-width="2"/>`,
    `<text x="${legendX + 22}" y="25" fill="#bbb" font-size="10">EWMA</text>`,
    `<line x1="${legendX + 118}" y1="16" x2="${legendX + 136}" y2="16" stroke="#355" stroke-width="1" stroke-dasharray="3 2"/>`,
    `<text x="${legendX + 140}" y="19" fill="#bbb" font-size="10">1.0</text>`,
  ];
  series.forEach((p) => {
    svgParts.push(
      `<circle cx="${xAt(p.t)}" cy="${yAt(p.ewma)}" r="2.2" fill="#8cf"/>`,
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
  const orderWarn =
    s.score_order_ok === false
      ? `<div class="summary" style="color:#b45309">Score measures are out of written order. ` +
        `Use <strong>Apply &amp; regenerate</strong> on the measure range so the pickup ` +
        `and reference audio stay in score order, then alignment will rematch.</div>`
      : "";
  const engine = s.backend || s.engine || "alignment";
  const version = s.candidate_generation
    ? ` · ${escapeXml(String(s.candidate_generation))}`
    : "";
  el.innerHTML =
    `<div class="summary">` +
    `${escapeXml(String(engine))}${version} · ` +
    `${s.transcribed_note_count ?? "?"} transcribed · ` +
    `${s.mapped_note_count ?? "?"} mapped · ` +
    `${s.event_count ?? 0} aligned events` +
    `</div>` +
    orderWarn +
    `<table><thead><tr>` +
    `<th>m</th><th>note</th><th>perf start</th><th>perf end</th><th>dur</th><th>residual</th>` +
    `</tr></thead><tbody>${rows}</tbody></table>`;
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

async function pollRendererStatus() {
  const el = document.getElementById("rendererStatus");
  if (!el) return;
  const apply = (data) => {
    const state = data?.state || "idle";
    el.dataset.state = state;
    if (state === "ready") {
      const sec = data.loaded_sec != null ? ` in ${data.loaded_sec}s` : "";
      el.textContent = `Renderer: ready${sec}`;
      return true;
    }
    if (state === "error") {
      el.textContent = `Renderer: ${data.error || "unavailable"}`;
      return true;
    }
    if (state === "loading") {
      el.textContent = "Renderer: loading SoundFont…";
      return false;
    }
    el.textContent = "Renderer: starting…";
    return false;
  };
  for (let i = 0; i < 120; i += 1) {
    try {
      const res = await fetch("/api/renderer/status");
      const data = res.ok ? await res.json() : null;
      if (apply(data)) return;
    } catch {
      el.textContent = "Renderer: starting…";
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
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
    fillParent: false,
    hideScrollbar: true,
    // Outer #waveformScroll owns horizontal scroll so staff/EWMA stay locked.
    autoScroll: false,
    autoCenter: false,
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
  setupMelodyStrip();
  setupScoreCompareHover();
  setupViewControls();
  pollRendererStatus();
  setupWaveformWheel();
  setupMarqueeSelect();
  setupRectSelect();
  setupOverlapPicker();
  window.addEventListener("resize", debounce(() => {
    if (!userZoomed) fitWaveformToContainer();
    else syncAlignmentStackWidth(getEffectivePxPerSec());
  }, 150));
  document.addEventListener("pointerup", () => {
    if (regionDragUndoArmed) {
      regionDragUndoArmed = false;
      refreshAllCaptions();
      captureIdleSnapshot();
    }
    if (suppressMoveLock) {
      suppressMoveLock = false;
      refreshSelectionVisuals();
    }
    ignoreRegionUpdateUndo = false;
  });

  refreshDragSelection();
  regionsPlugin.on("region-created", (region) => {
    if (programmaticRegionDepth > 0) return;
    if (isTrimRegion(region) || isLinkOverlay(region)) return;

    // Drawing the original passage for a repetition: never create a label.
    if (
      repetitionLinkMode === "draw" &&
      selectedRegion?.data?.type === "repetition"
    ) {
      ignoreRegionUpdateUndo = true;
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
    pushUndoFromIdle();
    ignoreRegionUpdateUndo = true;
    suppressMoveLock = true;
    applyLabelToRegion(region, { comment: null });
    selectRegion(region);
    captureIdleSnapshot();
  });
  const onRegionPointerPick = (region, e, { doubleClick = false } = {}) => {
    e?.stopPropagation?.();
    if (isTrimRegion(region)) {
      updateTrimInfo();
      return;
    }
    if (isLinkOverlay(region)) return;
    if (rectSelecting || marqueeSelecting) return;
    if (e && typeof e.button === "number" && e.button !== 0) return;

    const recentRealDrag = Date.now() - lastRegionDragAt < 250;
    const { hits, x, y } = overlappingHitsForClick(region, e);

    if (e?.ctrlKey || e?.metaKey) {
      toggleRegionInSelection(region);
      lastRegionClick = { id: null, time: 0 };
      hideOverlapPicker();
      return;
    }

    if (hits.length > 1) {
      // A real drag-end click should not pop the menu, but a double-click must.
      if (recentRealDrag && !doubleClick) return;
      lastRegionClick = { id: null, time: 0 };
      showOverlapPicker(hits, x, y);
      return;
    }

    if (recentRealDrag) return;
    hideOverlapPicker();

    const now = Date.now();
    const isDoubleClick = doubleClick
      || (lastRegionClick.id === region.id
        && now - lastRegionClick.time <= REGION_DOUBLE_CLICK_MS);

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
  };

  regionsPlugin.on("region-clicked", (region, e) => {
    const doubleClick = !!(e && (e.detail === 2 || e.type === "dblclick"));
    onRegionPointerPick(region, e, { doubleClick });
  });
  if (typeof regionsPlugin.on === "function") {
    regionsPlugin.on("region-double-clicked", (region, e) => {
      onRegionPointerPick(region, e, { doubleClick: true });
    });
  }
  regionsPlugin.on("region-updated", (region) => {
    if (isTrimRegion(region)) {
      updateTrimInfo();
      return;
    }
    if (isLinkOverlay(region)) return;
    const moved = regionPosChanged(region);
    if (moved) lastRegionDragAt = Date.now();
    if (moved && !ignoreRegionUpdateUndo && !regionDragUndoArmed && !undoSuspended) {
      pushUndoFromIdle();
      regionDragUndoArmed = true;
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
    applyRegionCaption(region);
    scheduleCaptionLayout();
  });

  document.getElementById("playBtn").onclick = () => wavesurfer.playPause();
  document.getElementById("loopBtn").onclick = () => {
    loopSelection = !loopSelection;
    document.getElementById("loopBtn").textContent = loopSelection ? "Loop: ON" : "Loop selection";
  };
  document.getElementById("speedSelect").onchange = (e) => {
    wavesurfer.setPlaybackRate(parseFloat(e.target.value));
  };
  document.getElementById("labelSourceSelect").onchange = async (event) => {
    labelSource = event.target.value === "agent" ? "agent" : "human";
    if (currentSample) await loadSample(currentSample);
  };
  document.getElementById("saveBtn").onclick = saveLabels;
  document.getElementById("applyLabelBtn").onclick = () => applyLabelToSelection();
  document.getElementById("deleteRegionBtn").onclick = deleteSelectedRegion;

  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("keyup", onSpaceKeyUp, true);

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
  document.getElementById("realignBtn").onclick = reAlignSample;
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
    const contentWidth = getAlignmentContentWidth(pxPerSec);
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
  applyAllLabelsVisibility();
  refreshAllCaptions();
  captureIdleSnapshot();
  loadScoreEvents();
  loadNoteAlignment().then(() => {
    if (viewMode === "alignment") renderAlignmentOverlays();
  });
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
      if (s.label_count) tags.push(`${s.label_count} yours`);
      if (s.agent_label_count) tags.push(`${s.agent_label_count} agent`);
      const suffix = tags.length ? ` (${tags.join(", ")})` : "";
      return `<option value="${s.id}">${s.id}${suffix}</option>`;
    })
    .join("");
  sel.onchange = () => {
    if (sel.value && sel.value !== currentSample) {
      loadSample(sel.value);
    }
  };
  if (samples.length) {
    const current = currentSample && samples.some((s) => s.id === currentSample)
      ? currentSample
      : samples[0].id;
    // Avoid onchange firing a duplicate load when setting value programmatically.
    const prevHandler = sel.onchange;
    sel.onchange = null;
    sel.value = current;
    sel.onchange = prevHandler;
    await loadSample(current);
  }
}

async function loadSample(sampleId) {
  const loadId = ++scoreLoadId;
  currentSample = sampleId;
  setScoreStatus("Loading score…");
  cachedScoreXml = null;
  cachedScoreMeta = null;

  const sourceQuery = new URLSearchParams({ label_source: labelSource });
  const res = await fetch(`/api/samples/${sampleId}?${sourceQuery}`);
  const data = await res.json();
  if (loadId !== scoreLoadId) return;
  sampleData = data;
  labelSource = data.label_source === "agent" ? "agent" : "human";
  document.getElementById("labelSourceSelect").value = labelSource;
  document.getElementById("labelTrack").textContent =
    labelSource === "agent" ? "Agent labels" : "Your labels";
  document.getElementById("saveBtn").textContent =
    labelSource === "agent" ? "Save agent labels" : "Save your labels";
  taxonomy = data.taxonomy;
  populateTypeSelect();
  document.getElementById("annotatorId").value = data.annotator_id || "";
  updateMeasureControls(data.prep);

  trimRegion = null;
  selectedRegions = [];
  selectedRegion = null;
  repetitionLinkMode = null;
  lastRegionClick = { id: null, time: 0 };
  hideOverlapPicker();
  clearUndoHistory();
  resetSelectionInfo();
  regionsPlugin.clearRegions();
  pendingRegions = data.labels.map((l) => ({
    label: normalizeLabelType(l),
    isCandidate: false,
  }));

  noteAlignmentData = null;
  scoreEventData = null;
  userZoomed = false;
  clearAlignmentOverlays();
  const transStrip = document.getElementById("transcriptionStrip");
  if (transStrip) transStrip.dataset.renderKey = "";
  const melody = document.getElementById("melodyStrip");
  if (melody) melody.dataset.renderKey = "";
  renderScoreCompare();

  const duration = data.prep?.performance_duration || 0;
  if (typeof wavesurfer.setOptions === "function" && duration > 0) {
    const fit = computeFitPxPerSec(duration);
    wavesurfer.setOptions({
      minPxPerSec: fit,
      width: Math.max(getScrollContainerWidth(), duration * fit),
    });
  }

  const audioUrl = audioUrlWithCacheBust(data.audio_url, data.audio_mtime);
  await wavesurfer.load(audioUrl);
  if (loadId !== scoreLoadId) return;

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
  const display = typeDisplayName(label.type);
  const create = () =>
    regionsPlugin.addRegion({
      start: label.start_time,
      end: label.end_time,
      color,
      drag: false,
      resize: false,
      content: display,
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
  region.setOptions({ color });
  syncRegionVisual(region, { selected: isRegionSelected(region) });
  if (!region.data.core_note_ids?.length) {
    const inferred = coreIdsFromStoredLabel(region.data);
    if (inferred.length) region.data.core_note_ids = inferred;
  }
  if (label.type === "repetition" && label.repeats_label_range) {
    syncRepetitionLinkOverlay(region);
  }
  if (programmaticRegionDepth === 0) refreshAllCaptions();
  else applyRegionCaption(region);
  return region;
}

function updateSelectionInfo(region) {
  const typeLabel = regionCaptionText(region);
  document.getElementById("selectionInfo").textContent =
    `${formatTime(region.start)} – ${formatTime(region.end)} (${typeLabel})`;
}

function applyLabelToRegion(region, overrides = {}) {
  if (!region || isTrimRegion(region) || isLinkOverlay(region)) return;
  overrides = labelOverrides(overrides);
  const type = (overrides.type && (!taxonomy.length || taxonomy.includes(overrides.type)))
    ? overrides.type
    : currentLabelType();
  const severity =
    overrides.severity ?? parseInt(document.getElementById("severity").value, 10);
  const comment = "comment" in overrides
    ? overrides.comment
    : (document.getElementById("comment").value || null);
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
  const color = TYPE_COLORS[type] || region.color;
  region.setOptions({ color });
  if (region.element) {
    region.element.style.backgroundColor = color;
  }
  if (isRegionSelected(region)) {
    if (selectedRegions.length === 1) {
      updateSelectionInfo(region);
      updateRepetitionPanel(region);
    } else {
      updateMultiSelectionInspector();
    }
  }
  syncRegionVisual(region, { selected: isRegionSelected(region) });
  refreshAllCaptions();
}

function applyLabelToSelection(overrides = {}) {
  overrides = labelOverrides(overrides);
  const targets = selectedRegions.filter((r) => !isTrimRegion(r) && !isLinkOverlay(r));
  if (!targets.length) return;
  if (!overrides.skipUndo) pushUndoFromIdle();
  const type = (overrides.type && (!taxonomy.length || taxonomy.includes(overrides.type)))
    ? overrides.type
    : currentLabelType();
  const severity =
    overrides.severity ?? parseInt(document.getElementById("severity").value, 10);
  const comment = "comment" in overrides
    ? overrides.comment
    : (document.getElementById("comment").value || null);
  targets.forEach((region) => applyLabelToRegion(region, { type, severity, comment }));
  captureIdleSnapshot();
}

function safeRemoveRegion(region) {
  if (!region) return;
  removeLinkedOriginals(region);
  try {
    region.remove();
  } catch {
    /* WaveSurfer can throw when removing the last remaining region */
  }
}

function deleteSelectedRegion() {
  const targets = selectedRegions.filter((r) => !isTrimRegion(r) && !isLinkOverlay(r));
  if (!targets.length) return;
  const ids = new Set(targets.map((r) => r.id).filter(Boolean));
  const dataIds = new Set(targets.map((r) => r.data?.id).filter(Boolean));

  pushUndoFromIdle();
  hideOverlapPicker();

  // Disable drag-create so a leftover pointerup cannot recreate the last label.
  if (typeof regionsPlugin.disableDragSelection === "function") {
    regionsPlugin.disableDragSelection();
  }

  targets.forEach((region) => safeRemoveRegion(region));

  // Sweep leftovers (plugin sometimes keeps the last region in its list).
  if (regionsPlugin) {
    regionsPlugin.getRegions().slice().forEach((region) => {
      if (isTrimRegion(region) || isLinkOverlay(region)) return;
      if (ids.has(region.id) || dataIds.has(region.data?.id)) {
        safeRemoveRegion(region);
      }
    });
  }
  sweepOrphanLinkOverlays();

  selectedRegions = [];
  selectedRegion = null;
  repetitionLinkMode = null;
  resetSelectionInfo();
  updateRepetitionPanel(null);
  refreshDragSelection();
  refreshAllCaptions();
  captureIdleSnapshot();
}

async function applyScoreSegment() {
  const { start, end, startBeat, endBeat } = getMeasureRange();
  const rangeLabel = formatSegmentLabel(start, end, startBeat, endBeat);
  if (!confirm(
    `Extract ${rangeLabel} and regenerate reference audio?`
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
    alert("Score segment applied. Reference audio updated. Use Re-align to transcribe and map notes again.");
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Apply & regenerate";
  }
}

async function reAlignSample() {
  if (!currentSample) return;
  if (!confirm("Re-run Basic Pitch transcription and joint alignment on the current performance?")) return;
  const btn = document.getElementById("realignBtn");
  btn.disabled = true;
  btn.textContent = "Aligning…";
  try {
    const res = await fetch(`/api/samples/${currentSample}/re-align`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    await loadSample(currentSample);
    alert("Transcription and alignment updated.");
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Re-align";
  }
}

async function applyPerformanceTrim() {
  if (!trimRegion) return;
  if (!confirm(
    `Trim performance to ${formatTime(trimRegion.start)} – ${formatTime(trimRegion.end)}?`
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
      `Performance trimmed to ${result.performance_trim?.trimmed_duration?.toFixed?.(1) ?? "?"}s.`
    );
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Apply trim";
  }
}

async function saveLabels() {
  const labels = [];
  regionsPlugin.getRegions().forEach((r) => {
    if (isTrimRegion(r) || isLinkOverlay(r)) return;
    labels.push(regionDataToLabel(r));
  });
  const allowedTypes = new Set([...taxonomy, "wrong_pitch"]);
  const invalidType = labels.filter((label) => label.type && !allowedTypes.has(label.type));
  if (invalidType.length) {
    alert(
      `${invalidType.length} label(s) have an unknown type. ` +
      "Select each one and choose an error type in the inspector.",
    );
    return;
  }
  const payload = {
    labels,
    self_reported: [],
    annotator_id: document.getElementById("annotatorId").value || null,
  };
  const sourceQuery = new URLSearchParams({ label_source: labelSource });
  const res = await fetch(`/api/samples/${currentSample}/labels?${sourceQuery}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    alert(await res.text());
    return;
  }
  alert(labelSource === "agent" ? "Agent labels saved." : "Your labels saved.");
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
  if (loadId != null && loadId !== scoreLoadId) return;

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
  if (meta.loadId != null && meta.loadId !== scoreLoadId) return;
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

function _scorePageWidthPx() {
  const container = document.getElementById("osmdContainer");
  const panel = document.getElementById("scorePanel");
  const timeline = document.getElementById("timelinePanel");
  let pageWidth = (container?.clientWidth || 0) - 24;
  if (pageWidth < 100) pageWidth = (panel?.clientWidth || 0) - 24;
  if (pageWidth < 100) pageWidth = (timeline?.clientWidth || 0) - 48;
  return Math.max(400, pageWidth);
}

async function renderScoreXml(xml, meta = {}) {
  if (meta.loadId != null && meta.loadId !== scoreLoadId) return;

  cachedScoreXml = xml;
  cachedScoreMeta = { ...meta };
  delete cachedScoreMeta.loadId;

  const container = document.getElementById("osmdContainer");
  if (!container) return;

  try {
    if (meta.loadId != null && meta.loadId !== scoreLoadId) return;
    container.innerHTML = "";
    osmdInstance = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
      autoResize: false,
      drawTitle: true,
    });
    await osmdInstance.load(xml);
    if (meta.loadId != null && meta.loadId !== scoreLoadId) return;

    const paint = () => {
      if (meta.loadId != null && meta.loadId !== scoreLoadId) return;
      try {
        const pageWidth = _scorePageWidthPx();
        if (osmdInstance.EngravingRules) {
          // Historic project convention: pass CSS px as PageWidth (OSMD scales from it).
          osmdInstance.EngravingRules.PageWidth = pageWidth;
        }
        osmdInstance.render();
        if (meta.mode === "segment" && meta.start != null && meta.end != null) {
          container.dataset.segment = `${meta.start}-${meta.end}`;
        }
        scheduleWaveformFit();
      } catch (err) {
        setScoreStatus(`Score render failed: ${err.message || err}`);
      }
    };

    // Double-rAF so Normal-mode layout has non-zero width after uncollapse.
    requestAnimationFrame(() => {
      requestAnimationFrame(paint);
    });
  } catch (err) {
    setScoreStatus(`Could not display score: ${err.message || err}`);
  }
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

function isTextEntryTarget(el) {
  if (!el) return false;
  if (el.isContentEditable) return true;
  const tag = (el.tagName || "").toUpperCase();
  if (tag === "TEXTAREA") return true;
  if (tag !== "INPUT") return false;
  const type = (el.type || "text").toLowerCase();
  return [
    "text", "search", "email", "password", "url", "tel", "number",
    "date", "datetime-local", "month", "time", "week",
  ].includes(type);
}

function isSpaceKey(e) {
  return e.code === "Space" || e.key === " " || e.key === "Spacebar";
}

function onSpaceKeyUp(e) {
  if (isTextEntryTarget(e.target)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (isSpaceKey(e)) {
    e.preventDefault();
    e.stopPropagation();
  }
}

function onKeyDown(e) {
  const typing = isTextEntryTarget(e.target);

  if (e.key === "Escape") {
    hideOverlapPicker();
  }

  if ((e.ctrlKey || e.metaKey) && !e.altKey) {
    const key = e.key.toLowerCase();
    if (key === "z" && !e.shiftKey) {
      if (typing) return;
      e.preventDefault();
      e.stopPropagation();
      undoLastAction();
      return;
    }
    if (key === "a") {
      if (typing) return;
      e.preventDefault();
      e.stopPropagation();
      selectAllRegions();
      return;
    }
  }

  if (isSpaceKey(e)) {
    if (typing) return;
    e.preventDefault();
    e.stopPropagation();
    if (wavesurfer) wavesurfer.playPause();
    return;
  }

  if (typing || isEditableKeyTarget(e.target)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  if (e.key === "Delete" || e.key === "Backspace") {
    if (selectedRegions.length) {
      e.preventDefault();
      deleteSelectedRegion();
    }
    return;
  }

  if (e.key === "0") {
    const misc = taxonomy.includes("misc") ? "misc" : null;
    if (misc) {
      document.getElementById("labelType").value = misc;
      refreshDragSelection();
      if (selectedRegions.length) applyLabelToSelection({ type: misc });
    }
    return;
  }
  const idx = parseInt(e.key, 10);
  if (idx >= 1 && idx <= taxonomy.length) {
    document.getElementById("labelType").value = taxonomy[idx - 1];
    refreshDragSelection();
    if (selectedRegions.length) {
      applyLabelToSelection({ type: taxonomy[idx - 1] });
    }
  }
  if (!selectedRegion || isTrimRegion(selectedRegion)) return;
  const step = 0.01;
  if (e.key === "ArrowLeft") selectedRegion.setOptions({ start: Math.max(0, selectedRegion.start - step) });
  if (e.key === "ArrowRight") selectedRegion.setOptions({ end: selectedRegion.end + step });
}

init();
