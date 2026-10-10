"""Local, note-synchronized score films from an existing spoken-feedback run.

Notation is engraved by Verovio. Audio is never regenerated or quantized here:
the saved synthesis events and final mix timeline are the animation clock.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 854, 480, 24
COLORS = {"extra": "#FF805F", "wrong": "#FF77A4", "missing": "#F8C56A", "issue": "#EDB65A"}
SVG = "{http://www.w3.org/2000/svg}"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _local(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing local animation input: {name}")
    return path


def active_events(events, seconds):
    """Half-open intervals prevent two adjacent notes flashing simultaneously."""
    return [e for e in events if e["start"] <= seconds < e["end"]]


def _score_context(sample):
    from music21 import converter, meter, stream
    from datacreate.melody import parse_sounding_notes
    from datacreate.feedback_score import whole_score_locations
    selected = converter.parse(str(sample / "verified_score.musicxml"), forceSource=True)
    notes = parse_sounding_notes(selected)
    locations, bar_map = whole_score_locations(selected, notes, sample)
    full_path = sample / "full_score.musicxml"
    full = converter.parse(str(full_path), forceSource=True) if full_path.is_file() else selected
    measures = list(full.parts[0].getElementsByClass(stream.Measure))
    by_bar = ({i+1: m for i, m in enumerate(measures)} if full_path.is_file()
              else {bar_map[m.number]: m for m in measures})
    segment = _read(sample / "metadata.json").get("score_segment", {}) if (sample / "metadata.json").is_file() else {}
    first = by_bar[min(bar_map.values())]
    signature = first.timeSignature or first.getContextByClass(meter.TimeSignature)
    origin = (float(first.getOffsetInHierarchy(full)) + (segment.get("start_beat", 1)-1)
              * (float(signature.beatDuration.quarterLength) if signature else 1.)) if full_path.is_file() else 0.
    return full, by_bar, notes, locations, origin


def _reference_score(full, measure, bar, core_locations):
    from music21 import clef, key, meter, note, stream
    from datacreate.score_notes import is_decorative_element
    source = copy.deepcopy(measure)
    source.number = bar
    for kind in (clef.Clef, key.KeySignature, meter.TimeSignature):
        if not source.getElementsByClass(kind):
            context = measure.getContextByClass(kind)
            if context is not None:
                source.insert(0, copy.deepcopy(context))
    descriptors, ordinal = [], 0
    for index, n in enumerate(source.recurse().getElementsByClass(note.Note)):
        n.id = f"r-{bar}-{index}"
        decorative = is_decorative_element(n)
        continuation = n.tie is not None and n.tie.type in {"continue", "stop"}
        if not decorative and not continuation:
            ordinal += 1
        q0 = float(n.getOffsetInHierarchy(source)) + float(measure.getOffsetInHierarchy(full))
        descriptors.append({"id": n.id, "pitch": int(n.pitch.midi), "ql_start": q0,
                            "ql_end": q0+float(n.quarterLength), "bar": bar,
                            "marked": (bar, ordinal) in core_locations})
    part, score = stream.Part(), stream.Score()
    part.append(source)
    score.append(part)
    return score, descriptors


def _transcription_rows(spec, alignment):
    """Join played events to their original transcription, never to corrected pitches."""
    shift = spec.get("pitch_shift_semitones", 0)
    p0, p1, factor = spec["source_start_time"], spec["source_end_time"], spec["slowdown_factor"]
    candidates = sorted([(i, n) for i, n in enumerate(alignment["transcribed_notes"])
                         if not n.get("ignored") and n["start"] < p1 and n["end"] > p0],
                        key=lambda pair: (pair[1]["start"], pair[1]["pitch"]))
    if len(candidates) != len(spec["notes"]):
        raise ValueError("Saved music and transcription disagree. Regenerate feedback with the matching alignment.")
    result = []
    for (i, raw), played in zip(candidates, spec["notes"]):
        if (raw["pitch"]+shift != played["pitch"] or
                abs((max(p0, raw["start"])-p0)*factor-played["start"]) > 1e-5 or
                abs((min(p1, raw["end"])-p0)*factor-played["end"]) > 1e-5):
            raise ValueError("Saved synthesis pitch/timing no longer matches the alignment.")
        result.append({**raw, **played, "id": f"p-{i}", "transcription_index": i,
                       "written_pitch": raw["pitch"], "source_start": raw["start"], "source_end": raw["end"]})
    return result


def _performed_score(rows, missing, source_measure, bar, seconds_per_quarter, factor, display_start=0.):
    from music21 import clef, key, note, stream
    score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=bar)
    for kind in (clef.Clef, key.KeySignature):
        context = source_measure.getElementsByClass(kind).first() or source_measure.getContextByClass(kind)
        if context is not None:
            measure.insert(0, copy.deepcopy(context))
    if not measure.getElementsByClass(clef.Clef):
        measure.insert(0, clef.TrebleClef())
    # Readable note values are display-only. The cursor uses unrounded seconds.
    values = np.array([.125, .25, .375, .5, .75, 1, 1.5, 2, 3, 4])
    cursor = display_start
    for row in sorted(rows+missing, key=lambda n: (n["start"], not n.get("ghost", False))):
        n = note.Note(row["written_pitch"])
        if row.get("ghost"):
            n = n.getGrace()
            n.style.noteSize = "cue"
        else:
            ql = (row["end"]-row["start"]) / factor / seconds_per_quarter
            n.quarterLength = float(values[np.argmin(abs(values-ql))])
            # Include substantial actual gaps as rests in the displayed sequence.
            gap = (row["start"]-cursor) / factor / seconds_per_quarter
            if gap >= .125:
                measure.append(note.Rest(quarterLength=float(values[np.argmin(abs(values-gap))])))
            cursor = row["end"]
            n.stemDirection = "down" if n.pitch.midi >= 71 else "up"
        n.id = row["id"]
        if row.get("error"):
            n.style.color = COLORS[row["error"]]
        measure.append(n)
    if not rows:
        measure.append(note.Rest(quarterLength=4))
    measure.makeAccidentals(inPlace=True, overrideStatus=True, cautionaryNotImmediateRepeat=False)
    part.append(measure)
    score.append(part)
    return score


def _engrave(score, destination, descriptors, performance=False):
    import verovio
    import resvg_py
    score.write("musicxml", fp=destination.with_suffix(".musicxml"), makeNotation=False)
    if performance:
        # This is an unmetered transcription of an original-score bar, not new
        # bar numbering or a claim that quantized note lengths are exact.
        tree = ET.parse(destination.with_suffix(".musicxml"))
        for attrs in tree.findall(".//attributes"):
            for timing in attrs.findall("time"):
                attrs.remove(timing)
        for measure in tree.findall(".//measure"):
            measure.set("implicit", "yes")
        tree.write(destination.with_suffix(".musicxml"), encoding="utf-8", xml_declaration=True)
    # Verovio's default resource path is thread-local. Studio engraves in a
    # background worker, which may differ from the thread that imported it.
    from importlib.resources import files
    toolkit = verovio.toolkit(False)
    if not toolkit.setResourcePath(str(files("verovio") / "data")):
        raise ValueError("Verovio's bundled notation fonts could not be loaded.")
    toolkit.setOptions({"pageWidth": 2600, "pageHeight": 900, "scale": 40, "breaks": "none",
                        "adjustPageHeight": True, "header": "none", "footer": "none",
                        "svgBoundingBoxes": True, "xmlIdSeed": 1})
    if not toolkit.loadFile(str(destination.with_suffix(".musicxml"))) or toolkit.getPageCount() != 1:
        raise ValueError("The passage could not be engraved on one animation page.")
    svg = toolkit.renderToSVG(1)
    root = ET.fromstring(svg)
    definition = root.find(f"{SVG}svg")
    view = [float(x) for x in definition.attrib["viewBox"].split()]
    margin = root.find(f".//{SVG}g[@class='page-margin']")
    dx, dy = [float(v) for v in re.findall(r"[-\d.]+", margin.attrib["transform"])]
    rectangles = {r["id"]: root.find(f".//{SVG}g[@id='bbox-{r['id']}']/{SVG}rect") for r in descriptors}
    # Bounding boxes are layout metadata, not part of the printed score.
    for parent in root.iter():
        for child in list(parent):
            if "bounding-box" in child.attrib.get("class", "").split():
                parent.remove(child)
    # Light notation on black; keep the error colors and transparent background.
    root.set("fill", "#F8FAFC")
    root.set("color", "#F8FAFC")
    for element in root.iter():
        for attribute in ("color", "fill", "stroke"):
            if element.get(attribute, "").lower() in {"black", "#000", "#000000"}:
                element.set(attribute, "#F8FAFC")
    ET.register_namespace("", SVG[1:-1])
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    svg = ET.tostring(root, encoding="unicode")
    destination.with_suffix(".svg").write_text(svg, encoding="utf-8")
    picture = Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg_string=svg, width=1800))).convert("RGBA")
    scale = picture.width/view[2]
    crop = picture.getchannel("A").getbbox()
    if not crop:
        raise ValueError("Engraver produced an empty score.")
    factor = min(754/(crop[2]-crop[0]), 193/(crop[3]-crop[1]))
    picture = picture.crop(crop)
    picture = picture.resize((round(picture.width*factor), round(picture.height*factor)), Image.Resampling.LANCZOS)
    left, top = (WIDTH-picture.width)//2, 127+(193-picture.height)//2
    boxes = {}
    for row in descriptors:
        rect = rectangles[row["id"]]
        if rect is None:
            raise ValueError(f"Engraved note lost its identity: {row['id']}")
        x, y, w, h = (float(rect.attrib[k]) for k in ("x", "y", "width", "height"))
        boxes[row["id"]] = [left+((x+dx)*scale-crop[0])*factor,
                            top+((y+dy)*scale-crop[1])*factor, w*scale*factor, h*scale*factor]
    picture.save(destination.with_suffix(".png"))
    return {"image": picture, "position": (left, top), "boxes": boxes}


def build_scenes(sample: Path, feedback: Path, output: Path, *, detail_progress=None):
    from datacreate.feedback_score import label_note_indices, reference_note_times
    from datacreate.feedback_audio import checked_clip
    from datacreate.feedback_synthesis import _midi_clock
    plan, report = _read(feedback / "playback_plan.json"), _read(feedback / "report.json")
    covered = {p["label_index"] for p in plan["points"]}
    if covered != set(range(len(report["labels"]))) or len(covered) != len(plan["points"]):
        raise ValueError("Video requires a narrated comparison for every label. Regenerate feedback with --all-labels first.")
    alignment_path = sample / "note_alignment_v2.json"
    alignment = _read(alignment_path)
    alignment_hash = hashlib.sha256(alignment_path.read_bytes()).hexdigest()
    full, by_bar, selected, locations, origin = _score_context(sample)
    clock, _, _ = _midi_clock(sample / "reference_audio.mid")
    reference_notes = reference_note_times(sample / "verified_score.musicxml", sample / "reference_audio.mid")
    written_to_sounding = reference_notes[0]["pitch"]-selected[0].pitch
    assets = output / "score-assets"
    assets.mkdir()
    scenes = {}
    for position, point in enumerate(plan["points"]):
        if detail_progress:
            detail_progress({'substep': 'engrave', 'completed': position, 'total': len(plan['points']),
                             'message': f'Drawing reference and played notation for point {position+1} of {len(plan["points"])}'})
        index = point["label_index"]
        label = report["labels"][index]
        specs = {}
        for kind, key in (("reference", "reference_clip"), ("performance", "clip")):
            clip = point.get(key)
            if not clip or not clip.get("synthesized"):
                raise ValueError("Video needs synthesized reference and performance examples with saved note events.")
            checked_clip(feedback, clip)
            spec = _read(_local(feedback, clip["note_events_file"]))
            if spec["alignment_sha256"] != alignment_hash:
                raise ValueError("The alignment changed after feedback was generated. Regenerate feedback first.")
            specs[kind] = spec
        core = set(label_note_indices(label))
        if not core and "start_time" in label:
            core = {e["sounding_index"] for e in alignment.get("events", [])
                    if type(e.get("sounding_index")) is int and e.get("perf_start") is not None
                    and e.get("perf_end") is not None and e["perf_start"] < label["end_time"]
                    and e["perf_end"] > label["start_time"]}
        core_locations = {locations[i] for i in core}
        reference_pages, descriptors = [], []
        for bar in specs["reference"]["bars"]:
            score, rows = _reference_score(full, by_bar[bar], bar, core_locations)
            page = _engrave(score, assets / f"point-{index:03d}-reference-{bar}", rows)
            page.update(bar=bar, notes=rows, marked=[r["id"] for r in rows if r["marked"]])
            reference_pages.append(page)
            descriptors += rows
        ref = specs["reference"]
        reference_events = []
        for n in ref["notes"]:
            source_time = ref["source_start_time"]+n["start"]/ref["slowdown_factor"]
            candidates = [r for r in descriptors if clock(r["ql_start"]-origin)-1e-5 <= source_time
                          < clock(r["ql_end"]-origin)-1e-5]
            if not candidates:
                candidates = [min(descriptors, key=lambda r: abs(clock(r["ql_start"]-origin)-source_time))]
            reference_events.append({**n, "ids": [r["id"] for r in candidates], "bar": candidates[0]["bar"]})
        perf = specs["performance"]
        rows = _transcription_rows(perf, alignment)
        events = alignment.get("events", [])
        by_transcription = {}
        for e in events:
            if type(e.get("transcription_index")) is int and type(e.get("sounding_index")) is int:
                by_transcription.setdefault(e["transcription_index"], []).append(e["sounding_index"])
        for row in rows:
            row["written_pitch"] = row["pitch"]-written_to_sounding
            span = row.get("score_span")
            indices = list(range(*span)) if span else by_transcription.get(row["transcription_index"], [])
            if not indices:
                indices = [e["sounding_index"] for e in events if type(e.get("sounding_index")) is int
                           and e.get("perf_start") is not None and e.get("perf_end") is not None
                           and abs(e["perf_start"]-row["source_start"]) < 1e-5
                           and abs(e["perf_end"]-row["source_end"]) < 1e-5]
            row["score_indices"] = indices
            if indices:
                row["bar"] = locations[indices[0]][0]
            else:
                nearest = min((e for e in events if type(e.get("sounding_index")) is int
                               and e.get("perf_start") is not None),
                              key=lambda e: abs(e["perf_start"]-row["source_start"]), default=None)
                row["bar"] = locations[nearest["sounding_index"]][0] if nearest else perf["bars"][0]
            row["bar"] = min(max(row["bar"], perf["bars"][0]), perf["bars"][-1])
            overlap = (row["source_end"] > label.get("start_time", -math.inf)
                       and row["source_start"] < label.get("end_time", math.inf))
            related = bool(core.intersection(indices))
            row["error"] = None
            if label["type"] == "extra_note" and not indices and overlap:
                row["error"] = "extra"
            elif label["type"] in {"wrong_note", "intonation_error"} and (related or (not core and overlap)) and (
                    label["type"] == "intonation_error" or not indices
                    or any(row["written_pitch"] != selected[i].pitch for i in indices)):
                row["error"] = "wrong"
            elif label["type"] not in {"extra_note", "missed_note", "wrong_note", "intonation_error"} and (related or (not core and overlap)):
                row["error"] = "issue"
            row["ids"] = [row["id"]]
        missing = []
        if label["type"] == "missed_note":
            mapped = {i for row in rows for i in row["score_indices"]}
            for i in sorted(core-mapped):
                following = [r for r in rows if r["score_indices"] and min(r["score_indices"]) > i]
                when = min((r["start"] for r in following), default=perf["duration_seconds"])
                missing.append({"id": f"missing-{i}", "written_pitch": selected[i].pitch,
                                "start": when, "end": when, "bar": locations[i][0], "ghost": True, "error": "missing"})
        ratios = [(e["perf_end"]-e["perf_start"])/(selected[e["sounding_index"]].ql_end-selected[e["sounding_index"]].ql_start)
                  for e in events if type(e.get("sounding_index")) is int
                  and e.get("perf_start") is not None and e.get("perf_end") is not None
                  and e["perf_end"] > e["perf_start"]]
        quarter = float(np.median(ratios)) if ratios else .5
        performance_pages = []
        for bar in perf["bars"]:
            bar_rows = [r for r in rows if r["bar"] == bar]
            # Keep notation readable at 480p; page turns follow exact note events.
            chunks = [bar_rows[i:i+18] for i in range(0, len(bar_rows), 18)] or [[]]
            for chunk_index, chunk in enumerate(chunks):
                lower = chunk[0]["start"] if chunk_index else -math.inf
                upper = chunks[chunk_index+1][0]["start"] if chunk_index+1 < len(chunks) else math.inf
                ghosts = [r for r in missing if r["bar"] == bar and lower <= r["start"] < upper]
                score = _performed_score(chunk, ghosts, by_bar[bar], bar, quarter, perf["slowdown_factor"],
                                         chunk[0]["start"] if chunk_index else 0.)
                page = _engrave(score, assets / f"point-{index:03d}-performance-{bar}-{chunk_index}", chunk+ghosts, True)
                page.update(bar=bar, notes=chunk+ghosts, marked=[r["id"] for r in chunk+ghosts if r.get("error")])
                performance_pages.append(page)
        scenes[index] = {"point": point, "label": label, "reference": reference_pages,
                         "performance": performance_pages, "reference_events": reference_events,
                         "performance_events": rows}
    return scenes


@lru_cache(maxsize=16)
def _font(size, bold=False):
    for path in ([Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("segoeuib.ttf" if bold else "segoeui.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _wrap(text, font, width):
    lines, line = [], ""
    for word in text.split():
        candidate = (line+" "+word).strip()
        if font.getlength(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines+[line] if line else lines


def render_frame(seconds, segments, scenes, duration):
    segment = next((s for s in reversed(segments) if s["start_time"] <= seconds), segments[0])
    scene = scenes[segment["label_index"]]
    role = segment.get("role", segment["kind"])
    kind = "reference" if role in {"intro", "reference"} else "performance"
    pages = scene[kind]
    local = seconds-segment["start_time"]
    playing = segment["kind"] in {"reference", "performance"} and seconds < segment["end_time"]
    active = active_events(scene[kind+"_events"], local) if playing else []
    ids = {i for e in active for i in e["ids"]}
    page = next((p for p in pages if ids.intersection(p["boxes"])), None)
    if page is None:
        if role == "feedback":
            marked_pages = [p for p in pages if p["marked"]] or pages
            page = marked_pages[min(int(local/5), len(marked_pages)-1)]
        else:
            previous = [e for e in scene[kind+"_events"] if e["start"] <= local] if playing else []
            previous_ids = set(previous[-1]["ids"]) if previous else set()
            page = next((p for p in pages if previous_ids.intersection(p["boxes"])), pages[0])
    image = Image.new("RGBA", (WIDTH, HEIGHT), "#000000")
    draw = ImageDraw.Draw(image)
    ink, muted, blue = "#F8FAFC", "#ADB8C6", "#6AD8FF"
    draw.text((28, 16), "PERFORMANCE FEEDBACK", font=_font(12, True), fill=muted)
    draw.text((28, 41), f"Bar {page['bar']}", font=_font(30, True), fill=ink)
    heading = "Reference score" if kind == "reference" else "Your playing"
    if role == "feedback":
        heading = scene["label"]["type"].replace("_", " ").capitalize()+" · How to improve"
    draw.text((179, 49), heading, font=_font(22, True), fill=blue if kind == "reference" else ink)
    order = list(scenes).index(segment["label_index"])+1
    draw.text((WIDTH-94, 20), f"{order:02d} / {len(scenes):02d}", font=_font(14), fill=muted)
    draw.rounded_rectangle((28, 103, WIDTH-28, 335), radius=12, fill="#000000", outline="#252D38")
    overlay = Image.new("RGBA", image.size)
    painter = ImageDraw.Draw(overlay)
    if role in {"intro", "feedback"}:
        marked = [page["boxes"][i] for i in page["marked"] if i in page["boxes"]]
        if marked:
            x0 = min(b[0] for b in marked)-9
            x1 = max(b[0]+b[2] for b in marked)+9
            painter.rounded_rectangle((x0, 120, x1, 323), radius=7, fill=(248, 197, 106, 35), outline=(248, 197, 106, 220), width=2)
            label = "Focus here" if role == "intro" else scene["label"]["type"].replace("_", " ").capitalize()
            draw.text((max(43, min(x0, WIDTH-180)), 81), label, font=_font(14, True), fill="#F8C56A")
    for identity in ids.intersection(page["boxes"]):
        x, y, w, h = page["boxes"][identity]
        painter.rounded_rectangle((x-5, 120, x+w+5, 323), radius=5, fill=(106, 216, 255, 40))
        painter.line((x+w/2, 119, x+w/2, 323), fill=(106, 216, 255, 215), width=2)
        painter.ellipse((x-4, y-4, x+w+4, y+h+4), outline=(106, 216, 255, 255), width=2)
    image = Image.alpha_composite(image, overlay)
    image.alpha_composite(page["image"], page["position"])
    draw = ImageDraw.Draw(image)
    draw.text((39, 344), "Written instrument pitches" if kind == "reference" else "Transcribed rhythm (approx.)", font=_font(13), fill=muted)
    x = 280
    legend_font = _font(20, True)
    for category, title in (("extra", "Added"), ("wrong", "Wrong pitch"), ("missing", "Missing (ghost)")):
        draw.ellipse((x, 350, x+12, 362), fill=COLORS[category])
        draw.text((x+20, 342), title, font=legend_font, fill="#FFFFFF")
        x += 20 + legend_font.getlength(title) + 28
    if segment["kind"] == "speech":
        caption = scene["point"][role]
    else:
        caption = "Listen to the reference." if kind == "reference" else "Listen to the transcription of your playing."
    lines = _wrap(caption, _font(18), WIDTH-90)
    # Long feedback remains readable as timed caption cards, preserving sentences.
    groups = [lines[i:i+3] for i in range(0, len(lines), 3)]
    fraction = max(0., min(.999, local/max(.01, segment["end_time"]-segment["start_time"])))
    shown = groups[min(int(fraction*len(groups)), len(groups)-1)] if groups else []
    for i, line in enumerate(shown):
        draw.text((39, 382+i*23), line, font=_font(18), fill=ink)
    draw.rectangle((28, 470, WIDTH-28, 473), fill="#252D38")
    draw.rectangle((28, 470, 28+(WIDTH-56)*min(1., seconds/duration), 473), fill=blue)
    return image.convert("RGB"), {"label_index": segment["label_index"], "phase": role,
                                  "bar": page["bar"], "active_ids": sorted(ids)}


def render_video(sample: Path, feedback: Path, output: Path, *, detail_progress=None):
    if detail_progress:
        detail_progress({'substep': 'engrave', 'message': 'Loading score, note timings and audio timeline'})
    from datacreate.audio_utils import _find_ffmpeg
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise ValueError("Install imageio-ffmpeg or put ffmpeg on PATH for video export.")
    sample, feedback, output = Path(sample), Path(feedback), Path(output)
    if output.exists():
        raise ValueError("Video output directory already exists; choose a new directory.")
    audio = _local(feedback, "feedback.mp3")
    segments = _read(feedback / "timeline.json")["segments"]
    duration = sf.info(audio).duration
    output.mkdir(parents=True)
    scenes = build_scenes(sample, feedback, output, detail_progress=detail_progress)
    destination = output / "feedback.mp4"
    temp = output / "feedback.part.mp4"
    count = math.ceil(duration*FPS)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-i", str(audio),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temp)]
    snapshots = {}
    for s in segments:
        when = s["start_time"]+min(1., (s["end_time"]-s["start_time"])/2)
        snapshots[round(when*FPS)] = f"preview-{s['label_index']:03d}-{s.get('role', s['kind'])}.png"
    with (output / "ffmpeg.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for frame in range(count):
                if detail_progress and frame % FPS == 0:
                    detail_progress({'substep': 'frames', 'completed': frame, 'total': count,
                                     'message': f'Rendering frame {frame+1:,} of {count:,}'})
                picture, state = render_frame(frame/FPS, segments, scenes, duration)
                process.stdin.write(picture.tobytes())
                if frame in snapshots:
                    picture.save(output / snapshots[frame])
            process.stdin.close()
            if detail_progress:
                detail_progress({'substep': 'finalize', 'message': 'Finalizing the MP4 audio and video tracks'})
            if process.wait(timeout=60):
                raise ValueError("Video encoding failed; see ffmpeg.log.")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    temp.replace(destination)
    event_manifest = {str(i): {k: scene[k] for k in ("reference_events", "performance_events")} for i, scene in scenes.items()}
    (output / "animation-events.json").write_text(json.dumps(event_manifest, indent=2), encoding="utf-8")
    manifest = {"status": "complete", "width": WIDTH, "height": HEIGHT, "fps": FPS, "frames": count,
                "theme": "black_background_white_notation",
                "duration_seconds": count/FPS, "source_audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "labels": list(scenes), "audio_resynthesized": False, "cursor_clock": "saved_synthesis_note_events",
                "notation_timing": "reference original; performance display rounded to readable note values",
                "file": destination.name}
    (output / "video.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--feedback", required=True, type=Path, help="Completed synthesized feedback run")
    parser.add_argument("--output", required=True, type=Path, help="New video output directory")
    args = parser.parse_args(argv)
    try:
        print(render_video(args.sample, args.feedback, args.output))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Video failed: {error}\n")


if __name__ == "__main__":
    main()
