"""Reproducible solo-instrument transcription corpus using the synth pipeline.

Targets are exact sounding MIDI notes, with one canonical reference-note index
per attack. This is a transcription corpus, not a planted-error benchmark.
All timbres sharing a melody seed stay in the same split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import SynthConfig


@dataclass(frozen=True)
class SoloInstrument:
    name: str
    program: int
    low: int
    high: int
    written_to_sounding: int = 0


# GM programs are zero-based. Ranges below are sounding, intentionally broad.
INSTRUMENTS = tuple(SoloInstrument(*row) for row in [
    ('flute', 73, 60, 96), ('piccolo', 72, 74, 108, 12),
    ('oboe', 68, 58, 91), ('english_horn', 69, 52, 81, -7),
    ('bassoon', 70, 34, 75), ('clarinet', 71, 50, 94, -2),
    ('soprano_sax', 64, 56, 88, -2), ('alto_sax', 65, 49, 81, -9),
    ('tenor_sax', 66, 44, 76, -14), ('baritone_sax', 67, 36, 69, -21),
    ('trumpet', 56, 54, 86, -2), ('muted_trumpet', 59, 54, 86, -2),
    ('french_horn', 60, 34, 77, -7), ('trombone', 57, 40, 77),
    ('tuba', 58, 28, 65), ('violin', 40, 55, 100),
    ('viola', 41, 48, 88), ('cello', 42, 36, 81),
    ('contrabass', 43, 28, 67, -12), ('nylon_guitar', 24, 40, 88, -12),
    ('steel_guitar', 25, 40, 88, -12), ('electric_bass', 33, 28, 67, -12),
    ('recorder', 74, 60, 96), ('pan_flute', 75, 60, 96),
])


def split_for(index: int, count: int, seed: int) -> str:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    position = order.index(index)
    return 'train' if position < int(.8 * count) else 'val' if position < int(.9 * count) else 'test'


def create_notes(config: SynthConfig, instrument: SoloInstrument, seed: int):
    from music21 import instrument as m21instrument, note, pitch, tempo, stream, meter
    from .scoregen import generate_score
    rng = random.Random(seed)
    cfg = SynthConfig(generation=dict(config.generation))
    cfg.generation.update(pitch_min=pitch.Pitch(midi=instrument.low).nameWithOctave,
                          pitch_max=pitch.Pitch(midi=instrument.high).nameWithOctave,
                          pitch_center=rng.randint(instrument.low, instrument.high))
    style = rng.choices(['mixed', 'fast', 'repeated', 'single'], [5, 2, 2, 1])[0]
    if style == 'fast':
        cfg.generation.update(duration_units=[1], tempo_min=120, tempo_max=200,
                              measures_min=2, measures_max=3, rest_probability=.03)
    if style == 'single':
        cfg.generation.update(duration_units=[16], meters=[[4, 4]],
                              measures_min=1, measures_max=1, rest_probability=0)
    if config.dataset.get('mode') in {'chromatic_coverage', 'chromatic_validation'}:
        # Two adjacent indices render the same coverage pitch with different banks.
        index = (seed - int(config.dataset['seed'])) // 1009
        target = instrument.low + (index // 2) % (instrument.high - instrument.low + 1)
        score = stream.Score(); part = stream.Part(); measure = stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature('4/4'))
        measure.insert(0, tempo.MetronomeMark(number=rng.randint(90, 190)))
        keys = [target]*5 + [max(instrument.low, target-1), min(instrument.high, target+1), target]
        durations = [1, .25, .25, .25, .25, .5, .5, 1]
        if config.dataset.get('mode') == 'chromatic_validation':
            keys = [target, max(instrument.low, target-1), target, target,
                    min(instrument.high, target+1), target, target, target]
            durations = [.5, .5, .25, .25, .5, .25, .75, 1]
        for key, duration in zip(keys, durations):
            measure.append(note.Note(pitch.Pitch(midi=key), quarterLength=duration))
        part.append(measure); score.append(part); style = 'coverage'
    else:
        score = generate_score(rng, cfg)
    part = score.parts[0]
    for inst in list(part.recurse().getElementsByClass(m21instrument.Instrument)):
        inst.activeSite.remove(inst)
    inst = m21instrument.Instrument(instrumentName=instrument.name)
    inst.midiProgram = instrument.program
    inst.transposition = None  # This corpus explicitly exports concert-pitch XML.
    part.insert(0, inst)
    bpm = float(next(iter(score.recurse().getElementsByClass(tempo.MetronomeMark))).number)
    flat = score.flatten()
    notes = list(flat.getElementsByClass(note.Note))
    if not notes:
        raise ValueError('Empty generated melody')
    repeat_probability = .65 if style == 'repeated' else float(config.generation.get('repeat_pitch_probability', .28))
    prior = None
    rows = []
    lead = rng.uniform(.1, .4)
    for i, n in enumerate(notes):
        key = prior if prior is not None and rng.random() < repeat_probability else int(n.pitch.midi)
        n.pitch = pitch.Pitch(midi=key)
        start = lead + float(n.offset) * 60 / bpm
        duration = float(n.quarterLength) * 60 / bpm
        # Ten-millisecond separations are included explicitly, plus legato and staccato.
        gap = rng.choice([0., .01, .02, .04, duration * rng.uniform(.08, .25)])
        end = start + max(.035, duration - gap)
        rows.append({'pitch': key, 'start': start, 'end': end,
                     'velocity': rng.randint(55, 115), 'score_event_index': i})
        prior = key
    for a, b in zip(rows, rows[1:]):
        a['end'] = min(a['end'], b['start'])
    assert all(a['end'] <= b['start'] + 1e-9 for a, b in zip(rows, rows[1:]))
    return score, rows, style


def write_exact_midi(path: Path, notes: list[dict], program: int):
    import mido
    midi = mido.MidiFile(ticks_per_beat=1000)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=1_000_000))
    track.append(mido.Message('program_change', program=program))
    events = []
    for row in notes:
        events += [(round(row['start'] * 1000), 1, row), (round(row['end'] * 1000), 0, row)]
    last = 0
    for tick, on, row in sorted(events, key=lambda x: (x[0], x[1])):
        track.append(mido.Message('note_on' if on else 'note_off', note=row['pitch'],
                                 velocity=row['velocity'] if on else 0, time=tick-last))
        last = tick
    midi.save(path)


def probe_bank_coverage(banks, instruments, sample_rate=22050):
    """Detect missing SoundFont key regions before assigning audible targets."""
    import tinysoundfont
    coverage = {}
    for bank, path in banks:
        synth = tinysoundfont.Synth(samplerate=sample_rate, gain=-6)
        synth.sfload(path)
        coverage[bank] = {}
        for inst in instruments:
            silent = []
            for pitch in range(inst.low, inst.high + 1):
                synth.sounds_off()
                synth.program_change(0, inst.program)
                synth.noteon(0, pitch, 90)
                audio = np.frombuffer(synth.generate(sample_rate // 2), np.float32).reshape(-1, 2).mean(1)
                if float(np.sqrt(np.mean(audio * audio))) < 1e-5:
                    silent.append(pitch)
            coverage[bank][inst.name] = silent
        synth.sounds_off()
    return coverage


def choose_supported_bank(notes, instrument, requested, banks, coverage):
    pitches = {n['pitch'] for n in notes}
    choices = [requested] + [b for b in banks if b != requested]
    for bank, path in choices:
        if not pitches.intersection(coverage[bank][instrument]):
            return bank, path
    raise ValueError(f'No SoundFont covers all required pitches for {instrument}')


def silent_note_windows(audio, sample_rate, notes):
    silent = []
    for i, note in enumerate(notes):
        window = audio[int((note['start']+.025)*sample_rate):int(min(note['end'], note['start']+.22)*sample_rate)]
        if len(window) >= 128 and float(np.sqrt(np.mean(window**2))) < 1e-5:
            silent.append(i)
    return silent


def render_one(job):
    import soundfile as sf
    from scipy.signal import fftconvolve
    from datacreate.config import PipelineConfig
    from .render import render_midi_clarinet
    from .scoregen import write_musicxml
    config_path, root, count, seed, instrument_dict, index, bank, bank_path, banks, coverage = job
    config = SynthConfig.load(config_path)
    instrument = SoloInstrument(**instrument_dict)
    sample_id = f'{instrument.name}-{index:05d}'
    sample = Path(root) / sample_id
    sample.mkdir(parents=True, exist_ok=True)
    melody_seed = seed + index * 1009
    requested_bank = bank
    marker = sample/'metadata.json'
    saved = json.loads(marker.read_text()) if marker.exists() else {}
    if saved.get('bank_fallback_reason') == 'silent_note_window' and saved.get('requested_bank') == bank:
        bank, bank_path = next(b for b in banks if b[0] == saved['bank'])
    # Inspect existing targets without regenerating MusicXML on resume.
    score = None
    if (sample/'notes.json').exists():
        notes = json.loads((sample/'notes.json').read_text())
    else:
        score, notes, style = create_notes(config, instrument, melody_seed)
    bank, bank_path = choose_supported_bank(notes, instrument.name, (bank, bank_path), banks, coverage)
    expected = {'instrument': instrument.name, 'melody_seed': melody_seed, 'bank': bank,
                'split': {'chromatic_coverage':'train', 'chromatic_validation':'val'}.get(config.dataset.get('mode'), split_for(index, count, seed)),
                'schema_version': 'align-solo-corpus-v1'}
    if marker.exists():
        if any(saved.get(k) != v for k, v in expected.items() if k != 'bank'):
            raise ValueError(f'Conflicting existing sample {sample}')
        if saved.get('bank') == bank and (sample/'performance_audio.wav').exists() and (sample/'notes.json').exists():
            existing, rate = sf.read(sample/'performance_audio.wav', dtype='float32')
            if not silent_note_windows(existing, rate, notes):
                return saved
    if score is None:
        score, notes, style = create_notes(config, instrument, melody_seed)
    # Align target seconds to the exact 1ms MIDI scheduler grid.
    for row in notes:
        row['start'] = round(row['start'], 3)
        row['end'] = round(row['end'], 3)
    write_musicxml(score, sample/'verified_score.musicxml')
    write_exact_midi(sample/'performance_audio.mid', notes, instrument.program)
    fallback_reason = 'missing_key_region' if bank != requested_bank else None
    for render_bank, render_path in [(bank, bank_path)] + [b for b in banks if b[0] != bank]:
        if {n['pitch'] for n in notes}.intersection(coverage[render_bank][instrument.name]):
            continue
        dc = PipelineConfig(paths={'soundfont': render_path}, audio=config.audio, musescore=config.musescore)
        render_midi_clarinet(sample/'performance_audio.mid', sample/'performance_audio.wav', dc,
                            logging.getLogger('solo'), clarinet_program=instrument.program)
        audio, sr = sf.read(sample/'performance_audio.wav', dtype='float32')
        if not silent_note_windows(audio, sr, notes):
            if render_bank != bank:
                fallback_reason = 'silent_note_window'
            bank, bank_path = render_bank, render_path
            expected['bank'] = bank
            break
    else:
        raise ValueError(f'All available banks produced silent note targets: {sample_id}')
    rng = np.random.default_rng(melody_seed + instrument.program * 17)
    degraded = rng.random() < float(config.degrade.get('probability', 0))
    if degraded:
        impulse = np.zeros(int(sr * .18), np.float32)
        for delay, decay in [(0.017, .7), (.037, .5), (.071, .3), (.13, .15)]:
            impulse[int(delay * sr)] = decay
        mix = rng.uniform(*config.degrade.get('reverb_mix', [.02, .15]))
        audio += mix * fftconvolve(audio, impulse)[:len(audio)]
        noise = rng.normal(0, 1, len(audio)).astype(np.float32)
        audio += noise * float(np.sqrt(np.mean(audio**2))) * 10 ** (-rng.uniform(*config.degrade.get('snr_db', [25, 45])) / 20)
    raw_peak = float(np.max(np.abs(audio)))
    if not np.isfinite(audio).all() or raw_peak < 1e-5:
        raise ValueError(f'Invalid or silent render: {sample_id}')
    audio *= min(1., .92 / raw_peak)
    sf.write(sample/'performance_audio.wav', audio, sr, subtype='PCM_16')
    rms = float(np.sqrt(np.mean(audio**2)))
    row = {**expected, 'id': sample_id, 'path': str(sample.resolve()), 'style': style,
           'requested_bank': requested_bank, 'bank_fallback': bank != requested_bank,
           'bank_fallback_reason': fallback_reason,
           'program': instrument.program, 'sounding_range': [instrument.low, instrument.high],
           'pitch_space': 'sounding', 'score_pitch_space': 'concert',
           'score_written_to_sounding': 0, 'instrument_standard_transposition': instrument.written_to_sounding,
           'note_count': len(notes), 'duration': len(audio)/sr, 'degraded': bool(degraded),
           'peak': float(np.max(np.abs(audio))), 'rms_db': float(20*np.log10(max(rms,1e-10))),
           'clipping_fraction': float(np.mean(np.abs(audio) >= .999)), 'monophonic_gates': True}
    (sample/'notes.json').write_text(json.dumps(notes, indent=2)+'\n')
    marker.write_text(json.dumps(row, indent=2)+'\n')
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--count', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--instruments', nargs='+')
    parser.add_argument('--extra-bank', type=Path)
    args = parser.parse_args(argv)
    config = SynthConfig.load(args.config)
    root = args.output or config.output_root()
    count = args.count or int(config.dataset.get('clips_per_instrument', 500))
    seed = int(config.dataset.get('seed', 20261017))
    if count < 10:
        raise ValueError('At least 10 melody families are required for disjoint splits')
    instruments = [i for i in INSTRUMENTS if not args.instruments or i.name in args.instruments]
    if args.instruments and set(args.instruments) - {i.name for i in instruments}:
        raise ValueError('Unknown instrument')
    coverage_only = config.dataset.get('mode') in {'chromatic_coverage', 'chromatic_validation'}
    range_validation = config.dataset.get('mode') == 'chromatic_validation'
    if coverage_only and count < 2 * max(i.high-i.low+1 for i in instruments):
        raise ValueError('Coverage corpus needs every pitch for both banks')
    banks = [('msbasic', str(config.resolved_path('soundfont')))]
    if args.extra_bank:
        banks.append(('fluidr3', str(args.extra_bank.resolve())))
    root.mkdir(parents=True, exist_ok=True)
    provenance = {'schema_version': 'align-solo-manifest-v1', 'seed': seed, 'count': count,
                  'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  'instruments': [asdict(i) for i in instruments],
                  'banks': [{'name': k, 'path': p, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for k,p in banks],
                  'split_policy': 'validation only; six register probes per instrument and bank; independent seeds and rhythm' if range_validation else 'training only; chromatic supplement with independent seeds' if coverage_only else 'melody seed grouped across every instrument and bank; 80/10/10',
                  'purpose': 'solo transcription; no planted-error labels'}
    spec = root/'provenance.json'
    if spec.exists() and json.loads(spec.read_text()) != provenance:
        raise ValueError('Generation provenance changed; use a new output directory')
    spec.write_text(json.dumps(provenance, indent=2)+'\n')
    coverage_path = root/'bank_coverage.json'
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    previous = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    if previous.get('provenance_sha256') == signature:
        coverage = previous['silent_keys']
    else:
        coverage = probe_bank_coverage(banks, instruments, config.audio.get('sample_rate', 22050))
        coverage_path.write_text(json.dumps({'provenance_sha256': signature, 'silent_keys': coverage}, indent=2)+'\n')
    jobs = [(str(args.config.resolve()), str(root.resolve()), count, seed, asdict(inst), n, *banks[n % len(banks)], banks, coverage)
            for n in range(count) for inst in instruments
            if not range_validation or n//2 in {0, 1, 4, (inst.high-inst.low+1)//2, inst.high-inst.low-4, inst.high-inst.low}]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(render_one, jobs, chunksize=8):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(jobs):
                print(json.dumps({'generated': len(rows), 'total': len(jobs), 'hours': sum(r['duration'] for r in rows)/3600}), flush=True)
    (root/'manifest.json').write_text(json.dumps({'provenance': provenance, 'rows': rows}, indent=2)+'\n')
    print(json.dumps({'complete': True, 'root': str(root), 'clips': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
