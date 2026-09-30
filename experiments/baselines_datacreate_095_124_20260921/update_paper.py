"""Insert the 095-124 recorded-clip comparison into the local Overleaf manuscript."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAPER = ROOT / "paper" / "sections" / "current_results.tex"
MARKER = "%DATACREATE_095_124%"


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def row(name: str, block: dict) -> str:
    official = block["official_note_wise"]
    if official.get("status") != "available":
        return rf"{name} & --- & --- & --- & checkpoint unavailable \\"
    ci = official.get("bootstrap_95_ci") or {}
    lower = pct(ci.get("lower_95", 0.0))
    upper = pct(ci.get("upper_95", 0.0))
    return (
        f"{name} & {pct(official['precision'])} & {pct(official['recall'])} & "
        f"{pct(official['f1'])} & [{lower}, {upper}] "
        + r"\\"
    )


def type_row(name: str, block: dict) -> str:
    official = block["official_note_wise"]
    if official.get("status") != "available":
        return rf"{name} & --- & --- & --- & --- & --- \\"
    types = official["per_type"]
    cells = []
    for kind in (
        "wrong_note",
        "missed_note",
        "extra_note",
        "rhythm_error",
        "repetition",
    ):
        cells.append(pct(types[kind]["f1"]))
    return f"{name} & " + " & ".join(cells) + r" \\"


def section(report: dict) -> str:
    models = report["models"]
    subset = report["subset"]
    n_all = len(subset["official_note_wise_available"])
    n_unavail = len(subset["official_note_wise_unavailable"])
    n_manual = len(subset["manual_source_ids"])
    n_agent = len(subset["agent_source_ids"])
    unavailable = ", ".join(subset["official_note_wise_unavailable"]) or "none"
    rows = "\n".join(
        [
            row("Polytune", models["polytune"]["all_auditable"]),
            row("LadderSym (prompted)", models["laddersym"]["all_auditable"]),
            row("AudioEval", models["audioeval"]["all_auditable"]),
        ]
    )
    type_rows = "\n".join(
        [
            type_row("Polytune", models["polytune"]["all_auditable"]),
            type_row("LadderSym (prompted)", models["laddersym"]["all_auditable"]),
            type_row("AudioEval", models["audioeval"]["all_auditable"]),
        ]
    )
    return f"""
\\subsection{{Recorded Clarinet Clips 095--124}}
\\label{{sec:datacreate-095-124}}

The 358-recording comparison above uses synthetic Mozart performances whose gold
contains every correct, repeated, and erroneous event. Recorded DataCreate clips
do not provide that complete transcript. An earlier timestamp-span evaluation on
clips 001--040 therefore reported onset-tolerance F1 because many gold documents
failed canonical location audit, and those numbers cannot be compared with the
synthetic micro-F1. The later protocol used here keeps the official exclusive
note-wise identity metric, but only on error labels that have an auditable
score-event location. Correct notes are not in the gold and are not scored.

We evaluate clips 095--124. {n_all} documents have auditable gold locations;
{n_unavail} are officially unavailable ({unavailable}). Of the auditable
documents, {n_manual} {'has' if n_manual == 1 else 'have'} \\texttt{{source=manual}} labels and {n_agent} have
\\texttt{{source=agent}} labels copied from AudioEval-family proposals after the
corresponding human files were empty. Agent-copied labels are working gold, not
independent human annotation, and empty files before that copy are not treated
as clean negatives. AudioEval predictions were frozen before these labels were
read. The Polytune and LadderSym checkpoints used in Table~\\ref{{tab:baseline-results}}
were not present on the evaluation machine, so those two systems are not scored
here.

\\begin{{table}}[!htbp]
\\centering
\\small
\\caption{{Score-based weighted precision, recall, and micro-F1 on recorded
DataCreate clips 095--124. Values are percentages. Confidence intervals resample
recordings 1,000 times. Only auditable error labels are scored.}}
\\label{{tab:datacreate-095-124}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Method & Precision & Recall & F1 & 95\\% F1 interval \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[!htbp]
\\centering
\\small
\\caption{{Per-type official note-wise F1 on the same auditable 095--124 labels.
Supports remain those of the gold documents.}}
\\label{{tab:datacreate-095-124-types}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Method & Wrong & Missed & Extra & Rhythm & Repetition \\\\
\\midrule
{type_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main() -> None:
    report = json.loads((HERE / "report.json").read_text(encoding="utf-8"))
    text = PAPER.read_text(encoding="utf-8")
    body = section(report).strip() + "\n"
    if MARKER in text:
        prefix, rest = text.split(MARKER, 1)
        if "%END_DATACREATE_095_124%" in rest:
            _, suffix = rest.split("%END_DATACREATE_095_124%", 1)
        else:
            suffix = rest
        PAPER.write_text(
            prefix + MARKER + "\n" + body + "%END_DATACREATE_095_124%\n" + suffix.lstrip(),
            encoding="utf-8",
        )
    else:
        PAPER.write_text(text.rstrip() + "\n\n" + MARKER + "\n" + body + "%END_DATACREATE_095_124%\n", encoding="utf-8")
    print(PAPER)


if __name__ == "__main__":
    main()
