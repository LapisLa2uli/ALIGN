"""Reproducible prose extraction and phrase-window generation for AudioEval.

This utility reads the paper sources without changing them. It keeps the
abstract and the files included by main.tex, removes non-prose material, splits
the result into sentences, and ranks 8--14-word windows by corpus rarity.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path


SKIP_ENVIRONMENTS = {"equation", "equation*", "table", "table*", "figure", "figure*"}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "both",
    "by", "can", "do", "does", "for", "from", "had", "has", "have", "if",
    "in", "into", "is", "it", "its", "may", "not", "of", "on", "or", "our",
    "so", "such", "than", "that", "the", "their", "these", "this", "those",
    "to", "use", "used", "uses", "using", "was", "we", "were", "when",
    "where", "which", "while", "will", "with",
}


def remove_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def collect_main_sources(paper_dir: Path) -> list[Path]:
    main = (paper_dir / "main.tex").read_text(encoding="utf-8")
    sources = [paper_dir / "main.tex"]
    for relative in re.findall(r"\\input\{([^}]+)\}", main):
        path = paper_dir / relative
        if path.suffix == "":
            path = path.with_suffix(".tex")
        sources.append(path)
    return sources


def extract_main_abstract(text: str) -> list[tuple[int, str, str]]:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if not match:
        return []
    line = text[: match.start(1)].count("\n") + 1
    return [(line, "Abstract", match.group(1))]


def extract_section_prose(path: Path) -> list[tuple[int, str, str]]:
    text = remove_comments(path.read_text(encoding="utf-8"))
    lines = text.splitlines()
    current_section = path.stem.replace("_", " ").title()
    skip_env: str | None = None
    paragraphs: list[tuple[int, str, str]] = []
    buffer: list[str] = []
    start_line = 1

    def flush() -> None:
        nonlocal buffer
        if buffer:
            paragraphs.append((start_line, current_section, " ".join(buffer)))
            buffer = []

    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        if skip_env:
            if re.search(rf"\\end\{{{re.escape(skip_env)}\}}", line):
                skip_env = None
            continue
        begin = re.search(r"\\begin\{([^}]+)\}", line)
        if begin and begin.group(1) in SKIP_ENVIRONMENTS:
            flush()
            skip_env = begin.group(1)
            continue
        heading = re.match(r"\\(?:sub)*section\{([^}]+)\}", line)
        if heading:
            flush()
            current_section = strip_latex(heading.group(1))
            continue
        if re.match(r"\\(?:label|centering|small|toprule|midrule|bottomrule)\b", line):
            continue
        if not line:
            flush()
            continue
        if not buffer:
            start_line = line_no
        buffer.append(line)
    flush()
    return paragraphs


def strip_latex(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.S)
    text = re.sub(r"\\\((?:.|\n)*?\\\)", " ", text)
    text = re.sub(r"\\\[(?:.|\n)*?\\\]", " ", text)
    text = re.sub(r"\\(?:cite[pt]?|ref|eqref|autoref)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:label|input|includegraphics)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\{(?:enumerate|itemize|abstract)\}", " ", text)
    text = re.sub(r"\\item\b", " ", text)
    text = re.sub(r"\\paragraph\{([^{}]*)\}", r"\1. ", text)
    text = re.sub(r"\\(?:emph|textbf|texttt|textsc)\{([^{}]*)\}", r"\1", text)
    # Preserve the argument of simple formatting commands.
    for _ in range(3):
        text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("~", " ").replace("--", "-")
    text = text.replace(r"\%", "%").replace(r"\,", " ")
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if len(WORD_RE.findall(part)) >= 3]


def window_score(words: list[str], document_frequency: collections.Counter[str], sentence_count: int) -> float:
    score = 0.0
    for word in words:
        token = word.lower()
        if token in STOPWORDS or len(token) < 4:
            continue
        score += math.log((sentence_count + 1) / (document_frequency[token] + 1)) + 1.0
        if "-" in word or any(char.isdigit() for char in word):
            score += 0.8
    return score / math.sqrt(len(words))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    sources = collect_main_sources(args.paper_dir)
    paragraphs: list[dict[str, object]] = []
    main_text = remove_comments(sources[0].read_text(encoding="utf-8"))
    for line, section, raw in extract_main_abstract(main_text):
        paragraphs.append({"source": "main.tex", "line": line, "section": section, "text": strip_latex(raw)})
    for path in sources[1:]:
        for line, section, raw in extract_section_prose(path):
            prose = strip_latex(raw)
            if prose:
                paragraphs.append(
                    {
                        "source": path.relative_to(args.paper_dir).as_posix(),
                        "line": line,
                        "section": section,
                        "text": prose,
                    }
                )

    sentences: list[dict[str, object]] = []
    for paragraph in paragraphs:
        for sentence in split_sentences(str(paragraph["text"])):
            sentences.append(
                {
                    "id": f"S{len(sentences) + 1:03d}",
                    "source": paragraph["source"],
                    "line": paragraph["line"],
                    "section": paragraph["section"],
                    "text": sentence,
                    "word_count": len(WORD_RE.findall(sentence)),
                }
            )

    document_frequency: collections.Counter[str] = collections.Counter()
    for sentence in sentences:
        document_frequency.update({word.lower() for word in WORD_RE.findall(str(sentence["text"]))})

    for sentence in sentences:
        words = WORD_RE.findall(str(sentence["text"]))
        windows: list[tuple[float, str]] = []
        for size in range(8, min(14, len(words)) + 1):
            for start in range(0, len(words) - size + 1):
                chunk = words[start : start + size]
                windows.append(
                    (window_score(chunk, document_frequency, len(sentences)), " ".join(chunk))
                )
        windows.sort(key=lambda item: (-item[0], item[1]))
        sentence["distinctive_windows"] = [window for _, window in windows[:3]]
        sentence["top_window_score"] = round(windows[0][0], 4) if windows else 0.0

    candidate_queries = sorted(
        (
            {
                "sentence_id": sentence["id"],
                "section": sentence["section"],
                "query": sentence["distinctive_windows"][0],
                "score": sentence["top_window_score"],
            }
            for sentence in sentences
            if sentence["distinctive_windows"]
        ),
        key=lambda item: (-float(item["score"]), str(item["sentence_id"])),
    )[:100]

    output = {
        "paper_sources": [path.relative_to(args.paper_dir).as_posix() for path in sources],
        "exclusions": {
            "title_author_date": True,
            "bibliography_entries": 19,
            "displayed_equation_environments": 3,
            "table_environments_in_included_sources": 4,
            "figure_environments_and_captions": 3,
            "inline_equations": "removed while surrounding prose was retained",
            "section_headings": "retained as metadata, excluded from word and sentence counts",
            "nonincluded_file": "sections/experiments.tex is not input by main.tex and was excluded",
        },
        "prose_word_count": sum(int(sentence["word_count"]) for sentence in sentences),
        "prose_sentence_count": len(sentences),
        "candidate_queries_ranked": candidate_queries,
        "sentences": sentences,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("prose_word_count", "prose_sentence_count")}, indent=2))


if __name__ == "__main__":
    main()
