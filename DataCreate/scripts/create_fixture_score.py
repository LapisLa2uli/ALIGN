"""Generate a minimal MusicXML fixture for pipeline testing."""

from pathlib import Path

from music21 import chord, metadata, note, stream, tempo


def build_demo_score() -> stream.Score:
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = "Demo Scale"
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=100))
    pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
    offset = 0.0
    for pitch in pitches:
        n = note.Note(pitch, quarterLength=0.5)
        n.offset = offset
        part.insert(offset, n)
        offset += 0.5
    part.insert(offset, chord.Chord(["C4", "E4", "G4"], quarterLength=1.0))
    score.insert(0, part)
    return score


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures"
    root.mkdir(parents=True, exist_ok=True)
    score = build_demo_score()
    out = root / "demo_scale.musicxml"
    score.write("musicxml", fp=str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
