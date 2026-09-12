"""Learned symbolic note-list to written-score alignment.

The decoder is deliberately small: a learned MLP supplies local operation
costs and a constrained edit-distance DP supplies sequence structure.  A
second constrained DP resolves insertion runs that are repeated score blocks.
No audio features are required; pitches are expected to be written pitches.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from alignmodel.types import GraphNote, ScoreGraph

OP_CLASSES = ("match", "substitute", "extra", "deletion", "reject")
FEATURE_NAMES = (
    "signed_pitch_delta",
    "absolute_pitch_delta",
    "exact_pitch",
    "same_pitch_class",
    "octave_distance",
    "log_duration_ratio",
    "relative_onset_delta",
    "relative_end_delta",
    "observed_duration",
    "score_duration",
    "input_confidence",
    "observed_missing",
    "score_missing",
    "relative_order_delta",
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass(frozen=True)
class ObservedNote:
    """Normalized note produced by a symbolic/audio note frontend."""

    pitch: int
    start: float
    end: float
    confidence: float = 1.0
    source_index: int = -1
    source: Any = field(default=None, compare=False, repr=False)

    @property
    def duration(self) -> float:
        return max(0.001, self.end - self.start)


@dataclass
class AlignmentOperation:
    """One edit operation in performance order."""

    kind: str
    performance_index: int | None
    score_index: int | None
    confidence: float
    cost: float = 0.0
    is_copy: bool = False
    copy_id: int | None = None

    @property
    def observed_index(self) -> int | None:
        return self.performance_index


@dataclass
class CopyBlock:
    """A replayed performance span mapped to the original written notes."""

    copy_id: int
    performance_indices: list[int]
    score_indices: list[int]
    score_start: int
    score_end: int
    confidence: float


@dataclass
class AlignmentResult:
    operations: list[AlignmentOperation]
    n_performance_notes: int
    n_score_notes: int
    total_cost: float
    copy_blocks: list[CopyBlock] = field(default_factory=list)

    @property
    def mapping(self) -> list[int | None]:
        """Written score index for every input note; extras are ``None``."""
        out: list[int | None] = [None] * self.n_performance_notes
        for op in self.operations:
            if op.performance_index is None:
                continue
            if op.kind in {"match", "substitute"}:
                out[op.performance_index] = op.score_index
        return out

    @property
    def confidences(self) -> list[float]:
        out = [0.0] * self.n_performance_notes
        for op in self.operations:
            if op.performance_index is not None:
                out[op.performance_index] = float(op.confidence)
        return out

    @property
    def unattached(self) -> list[int]:
        return [
            int(op.performance_index)
            for op in self.operations
            if op.performance_index is not None and op.kind == "unattached"
        ]

    @property
    def extras(self) -> list[int]:
        return [
            int(op.performance_index)
            for op in self.operations
            if op.performance_index is not None and op.kind == "extra"
        ]

    @property
    def deletions(self) -> list[int]:
        return [
            int(op.score_index)
            for op in self.operations
            if op.score_index is not None and op.kind == "deletion" and not op.is_copy
        ]


@dataclass
class NoteAlignConfig:
    """Decode settings. Pitch evidence intentionally outweighs timing."""

    timing_weight: float = 0.12
    learned_weight: float = 0.65
    extra_cost: float = 0.92
    deletion_cost: float = 0.92
    substitution_cost: float = 0.62
    min_copy_notes: int = 2
    max_copy_notes: int = 64
    copy_length_slack: int = 2
    copy_max_cost: float = 0.58
    copy_min_exact_fraction: float = 0.50
    unattached_threshold: float = 0.18
    substitute_threshold: float = 0.30
    max_pair_pitch_distance: int = 24
    temperature: float = 1.0


class LearnedNoteScorer(nn.Module):
    """Tiny residual MLP; well below 20k parameters at the default width."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 48):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, len(OP_CLASSES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _value(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
    else:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def normalize_notes(notes: Iterable[Any]) -> list[ObservedNote]:
    """Accept dicts/dataclasses/objects with pitch, start, end, confidence.

    ``midi`` and common onset/offset aliases are accepted for interoperability.
    Returned notes are stably sorted by onset while retaining ``source_index``.
    """

    out: list[ObservedNote] = []
    for source_index, raw in enumerate(notes):
        pitch = _value(raw, ("pitch", "midi", "written_pitch"))
        start = _value(raw, ("start", "onset", "start_time"))
        end = _value(raw, ("end", "offset", "end_time"))
        confidence = _value(raw, ("confidence", "probability", "score"), 1.0)
        if pitch is None or start is None:
            raise ValueError("Each note must provide written pitch and start")
        start_f = float(start)
        if end is None:
            duration = float(_value(raw, ("duration",), 0.05))
            end = start_f + duration
        end_f = max(float(end), start_f + 0.001)
        out.append(
            ObservedNote(
                pitch=int(round(float(pitch))),
                start=start_f,
                end=end_f,
                confidence=float(np.clip(float(confidence), 0.0, 1.0)),
                source_index=source_index,
                source=raw,
            )
        )
    out.sort(key=lambda n: (n.start, n.pitch, n.source_index))
    return out


def note_features(
    observed: ObservedNote | None,
    score_note: GraphNote | Any | None,
    *,
    observed_span: tuple[float, float] = (0.0, 1.0),
    score_span: tuple[float, float] = (0.0, 1.0),
    observed_order: float = 0.0,
    score_order: float = 0.0,
) -> np.ndarray:
    """Features for one local edit. Relative timing is secondary evidence."""

    obs_missing = observed is None
    score_missing = score_note is None
    opitch = int(observed.pitch) if observed is not None else 60
    spitch = int(_value(score_note, ("pitch", "midi"), 60)) if score_note is not None else 60
    delta = opitch - spitch if not obs_missing and not score_missing else 0
    odur = observed.duration if observed is not None else 0.0
    sstart = float(_value(score_note, ("start",), 0.0)) if score_note is not None else 0.0
    send = float(_value(score_note, ("end",), sstart + 0.001)) if score_note is not None else 0.0
    sdur = max(0.001, send - sstart) if score_note is not None else 0.0
    o0, o1 = observed_span
    s0, s1 = score_span
    olen = max(o1 - o0, 0.001)
    slen = max(s1 - s0, 0.001)
    orel = (observed.start - o0) / olen if observed is not None else 0.0
    oerel = (observed.end - o0) / olen if observed is not None else 0.0
    srel = (sstart - s0) / slen if score_note is not None else 0.0
    serel = (send - s0) / slen if score_note is not None else 0.0
    log_ratio = math.log((odur + 0.01) / (sdur + 0.01)) if odur and sdur else 0.0
    return np.asarray(
        [
            np.clip(delta / 12.0, -2.0, 2.0),
            np.clip(abs(delta) / 12.0, 0.0, 2.0),
            float(not obs_missing and not score_missing and delta == 0),
            float(not obs_missing and not score_missing and delta % 12 == 0),
            np.clip(abs(delta) // 12 / 2.0, 0.0, 2.0),
            np.clip(log_ratio, -2.0, 2.0),
            np.clip(orel - srel, -1.5, 1.5),
            np.clip(oerel - serel, -1.5, 1.5),
            np.clip(odur / olen * 16.0, 0.0, 2.0),
            np.clip(sdur / slen * 16.0, 0.0, 2.0),
            observed.confidence if observed is not None else 1.0,
            float(obs_missing),
            float(score_missing),
            np.clip(observed_order - score_order, -1.5, 1.5),
        ],
        dtype=np.float32,
    )


def _score_span(score: ScoreGraph) -> tuple[float, float]:
    if not score.notes:
        return (0.0, 1.0)
    return (
        float(min(n.start for n in score.notes)),
        float(max(n.end for n in score.notes)),
    )


def _observed_span(notes: Sequence[ObservedNote]) -> tuple[float, float]:
    if not notes:
        return (0.0, 1.0)
    return (float(notes[0].start), float(max(n.end for n in notes)))


class NoteAligner:
    """Constrained edit decoder with optional learned local operation costs."""

    def __init__(
        self,
        scorer: LearnedNoteScorer | None = None,
        config: NoteAlignConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        calibration: Mapping[str, Any] | None = None,
    ):
        self.config = config or NoteAlignConfig()
        self.device = torch.device(device)
        self.scorer = scorer.to(self.device).eval() if scorer is not None else None
        self.calibration = dict(calibration or {})
        if "temperature" in self.calibration:
            self.config.temperature = max(0.05, float(self.calibration["temperature"]))
        if "unattached_threshold" in self.calibration:
            self.config.unattached_threshold = float(
                self.calibration["unattached_threshold"]
            )

    @classmethod
    def from_checkpoint(
        cls, path: Path | str, *, device: str | torch.device = "cpu"
    ) -> "NoteAligner":
        blob = torch.load(Path(path), map_location=device, weights_only=False)
        model_cfg = blob.get("model_config") or {}
        scorer = LearnedNoteScorer(
            feature_dim=int(model_cfg.get("feature_dim", FEATURE_DIM)),
            hidden_dim=int(model_cfg.get("hidden_dim", 48)),
        )
        scorer.load_state_dict(blob["model"])
        cfg_fields = NoteAlignConfig.__dataclass_fields__
        raw_cfg = blob.get("align_config") or {}
        cfg = NoteAlignConfig(**{k: v for k, v in raw_cfg.items() if k in cfg_fields})
        return cls(
            scorer,
            cfg,
            device=device,
            calibration=blob.get("calibration") or {},
        )

    def align(self, notes: Iterable[Any], score: ScoreGraph) -> AlignmentResult:
        observed = normalize_notes(notes)
        if not observed:
            operations = [
                AlignmentOperation("deletion", None, n.index, 1.0, self.config.deletion_cost)
                for n in score.notes
            ]
            return AlignmentResult(operations, 0, len(score.notes), sum(o.cost for o in operations))
        if not score.notes:
            operations = [
                AlignmentOperation(
                    "unattached" if n.confidence < self.config.unattached_threshold else "extra",
                    i,
                    None,
                    n.confidence,
                    self.config.extra_cost,
                )
                for i, n in enumerate(observed)
            ]
            return AlignmentResult(operations, len(observed), 0, sum(o.cost for o in operations))

        costs, learned_probs = self._cost_tables(observed, score)
        operations, total_cost = self._decode(observed, score, costs, learned_probs)
        copy_blocks = self._resolve_copy_blocks(observed, score, operations)
        self._apply_uncertainty(observed, operations)
        return AlignmentResult(
            operations=operations,
            n_performance_notes=len(observed),
            n_score_notes=len(score.notes),
            total_cost=float(total_cost),
            copy_blocks=copy_blocks,
        )

    def _cost_tables(
        self, observed: Sequence[ObservedNote], score: ScoreGraph
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        n, m = len(observed), len(score.notes)
        pair_h = np.empty((n, m), dtype=np.float32)
        pair_features: list[np.ndarray] = []
        ospan, sspan = _observed_span(observed), _score_span(score)
        for i, obs in enumerate(observed):
            for j, snote in enumerate(score.notes):
                delta = abs(obs.pitch - snote.pitch)
                pitch_cost = 0.015 if delta == 0 else self.config.substitution_cost + min(delta, 12) / 20.0
                ot = (obs.start - ospan[0]) / max(ospan[1] - ospan[0], 0.001)
                st = (snote.start - sspan[0]) / max(sspan[1] - sspan[0], 0.001)
                timing = abs(ot - st)
                pair_h[i, j] = pitch_cost + self.config.timing_weight * timing
                pair_features.append(
                    note_features(
                        obs,
                        snote,
                        observed_span=ospan,
                        score_span=sspan,
                        observed_order=i / max(n - 1, 1),
                        score_order=j / max(m - 1, 1),
                    )
                )
        extra_h = np.asarray(
            [self.config.extra_cost + 0.08 * (1.0 - x.confidence) for x in observed],
            dtype=np.float32,
        )
        delete_h = np.full(m, self.config.deletion_cost, dtype=np.float32)
        learned_probs: np.ndarray | None = None
        if self.scorer is not None:
            feature_rows = pair_features
            feature_rows += [
                note_features(
                    obs,
                    None,
                    observed_span=ospan,
                    score_span=sspan,
                    observed_order=i / max(n - 1, 1),
                )
                for i, obs in enumerate(observed)
            ]
            feature_rows += [
                note_features(
                    None,
                    snote,
                    observed_span=ospan,
                    score_span=sspan,
                    score_order=j / max(m - 1, 1),
                )
                for j, snote in enumerate(score.notes)
            ]
            with torch.no_grad():
                tensor = torch.from_numpy(np.stack(feature_rows)).to(self.device)
                logits = self.scorer(tensor) / max(self.config.temperature, 0.05)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            learned_probs = probs[: n * m].reshape(n, m, len(OP_CLASSES))
            observed_pitch = np.asarray([note.pitch for note in observed])
            score_pitch = np.asarray([note.pitch for note in score.notes])
            pair_kind = (observed_pitch[:, None] != score_pitch[None, :]).astype(
                np.int64
            )
            learned_pair = -np.log(
                np.take_along_axis(learned_probs, pair_kind[..., None], axis=-1)[..., 0]
                + 1e-6
            )
            start = n * m
            extra_probs = probs[start : start + n, OP_CLASSES.index("extra")]
            delete_probs = probs[start + n :, OP_CLASSES.index("deletion")]
            w = float(np.clip(self.config.learned_weight, 0.0, 1.0))
            pair_h = (1.0 - w) * pair_h + w * learned_pair
            extra_h = (1.0 - w) * extra_h + w * -np.log(extra_probs + 1e-6)
            delete_h = (1.0 - w) * delete_h + w * -np.log(delete_probs + 1e-6)
        return {"pair": pair_h, "extra": extra_h, "delete": delete_h}, learned_probs

    def _decode(
        self,
        observed: Sequence[ObservedNote],
        score: ScoreGraph,
        costs: Mapping[str, np.ndarray],
        learned_probs: np.ndarray | None,
    ) -> tuple[list[AlignmentOperation], float]:
        """Global monotone DP. Replays are resolved from its insertion runs."""

        n, m = len(observed), len(score.notes)
        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 pair, 2 extra, 3 deletion
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            dp[i, 0] = dp[i - 1, 0] + float(costs["extra"][i - 1])
            back[i, 0] = 2
        for j in range(1, m + 1):
            dp[0, j] = dp[0, j - 1] + float(costs["delete"][j - 1])
            back[0, j] = 3
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                pair_cost = float(costs["pair"][i - 1, j - 1])
                if abs(observed[i - 1].pitch - score.notes[j - 1].pitch) > self.config.max_pair_pitch_distance:
                    pair_cost += 2.0
                candidates = (
                    (dp[i - 1, j - 1] + pair_cost, 1),
                    (dp[i - 1, j] + float(costs["extra"][i - 1]), 2),
                    (dp[i, j - 1] + float(costs["delete"][j - 1]), 3),
                )
                best, code = min(candidates, key=lambda item: (item[0], item[1]))
                dp[i, j], back[i, j] = best, code

        operations: list[AlignmentOperation] = []
        i, j = n, m
        while i or j:
            code = int(back[i, j])
            if i and j and code == 1:
                oi, sj = i - 1, j - 1
                exact = observed[oi].pitch == score.notes[sj].pitch
                kind = "match" if exact else "substitute"
                confidence = self._pair_confidence(observed[oi], score.notes[sj], learned_probs, oi, sj)
                operations.append(
                    AlignmentOperation(
                        kind,
                        oi,
                        score.notes[sj].index,
                        confidence,
                        float(costs["pair"][oi, sj]),
                    )
                )
                i -= 1
                j -= 1
            elif i and (not j or code == 2):
                oi = i - 1
                operations.append(
                    AlignmentOperation(
                        "extra",
                        oi,
                        None,
                        observed[oi].confidence,
                        float(costs["extra"][oi]),
                    )
                )
                i -= 1
            else:
                sj = j - 1
                operations.append(
                    AlignmentOperation(
                        "deletion",
                        None,
                        score.notes[sj].index,
                        1.0,
                        float(costs["delete"][sj]),
                    )
                )
                j -= 1
        operations.reverse()
        return operations, float(dp[n, m])

    def _pair_confidence(
        self,
        obs: ObservedNote,
        snote: GraphNote,
        learned_probs: np.ndarray | None,
        oi: int,
        sj: int,
    ) -> float:
        delta = abs(obs.pitch - snote.pitch)
        heuristic = (0.97 if delta == 0 else max(0.34, 0.76 - 0.055 * delta))
        if learned_probs is None:
            probability = heuristic
        else:
            cls = 0 if delta == 0 else 1
            probability = float(learned_probs[oi, sj, cls])
            probability = 0.35 * heuristic + 0.65 * probability
        return float(np.clip(probability * obs.confidence, 0.0, 1.0))

    def _resolve_copy_blocks(
        self,
        observed: Sequence[ObservedNote],
        score: ScoreGraph,
        operations: list[AlignmentOperation],
    ) -> list[CopyBlock]:
        runs: list[list[AlignmentOperation]] = []
        current: list[AlignmentOperation] = []
        for op in operations:
            if op.kind == "extra" and op.performance_index is not None:
                current.append(op)
            elif op.kind == "deletion" and current:
                continue
            else:
                if current:
                    runs.append(current)
                    current = []
        if current:
            runs.append(current)

        blocks: list[CopyBlock] = []
        for run in runs:
            if not (self.config.min_copy_notes <= len(run) <= self.config.max_copy_notes):
                continue
            obs_indices = [int(op.performance_index) for op in run if op.performance_index is not None]
            copy_notes = [observed[i] for i in obs_indices]
            best: tuple[float, float, int, list[tuple[int | None, int | None, str]]] | None = None
            lo_width = max(self.config.min_copy_notes, len(run) - self.config.copy_length_slack)
            hi_width = min(
                len(score.notes), len(run) + self.config.copy_length_slack
            )
            for width in range(lo_width, hi_width + 1):
                for start in range(0, len(score.notes) - width + 1):
                    local = self._local_copy_dp(copy_notes, score.notes[start : start + width])
                    norm_cost, exact_fraction, path = local
                    candidate = (norm_cost, -exact_fraction, start, path)
                    if best is None or candidate[:3] < best[:3]:
                        best = candidate
            if best is None:
                continue
            norm_cost, neg_exact, start, path = best
            exact_fraction = -neg_exact
            if (
                norm_cost > self.config.copy_max_cost
                or exact_fraction < self.config.copy_min_exact_fraction
            ):
                continue
            block_id = len(blocks)
            mapped_score: list[int] = []
            mapped_obs: list[int] = []
            path_by_obs = {pi: (sj, kind) for pi, sj, kind in path if pi is not None and sj is not None}
            for local_i, op in enumerate(run):
                hit = path_by_obs.get(local_i)
                if hit is None:
                    continue
                local_j, kind = hit
                score_index = score.notes[start + int(local_j)].index
                op.kind = kind
                op.score_index = score_index
                op.is_copy = True
                op.copy_id = block_id
                delta = abs(copy_notes[local_i].pitch - score.notes[start + int(local_j)].pitch)
                base = 0.96 if delta == 0 else max(0.36, 0.72 - 0.05 * delta)
                op.confidence = float(
                    np.clip(base * copy_notes[local_i].confidence * (1.0 - 0.35 * norm_cost), 0.0, 1.0)
                )
                mapped_score.append(score_index)
                mapped_obs.append(int(op.performance_index))
            if len(mapped_obs) < self.config.min_copy_notes:
                for op in run:
                    if op.copy_id == block_id:
                        op.kind, op.score_index, op.is_copy, op.copy_id = "extra", None, False, None
                continue
            blocks.append(
                CopyBlock(
                    copy_id=block_id,
                    performance_indices=mapped_obs,
                    score_indices=mapped_score,
                    score_start=min(mapped_score),
                    score_end=max(mapped_score),
                    confidence=float(np.mean([operations_confidence(run, i) for i in mapped_obs])),
                )
            )
        return blocks

    @staticmethod
    def _local_copy_dp(
        observed: Sequence[ObservedNote], score_notes: Sequence[GraphNote]
    ) -> tuple[float, float, list[tuple[int | None, int | None, str]]]:
        n, m = len(observed), len(score_notes)
        dp = np.zeros((n + 1, m + 1), dtype=np.float32)
        bt = np.zeros((n + 1, m + 1), dtype=np.int8)
        dp[:, 0] = np.arange(n + 1, dtype=np.float32) * 0.82
        dp[0, :] = np.arange(m + 1, dtype=np.float32) * 0.82
        bt[1:, 0] = 2
        bt[0, 1:] = 3
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                delta = abs(observed[i - 1].pitch - score_notes[j - 1].pitch)
                pair = 0.0 if delta == 0 else 0.58 + min(delta, 12) / 24.0
                candidates = (
                    (float(dp[i - 1, j - 1]) + pair, 1),
                    (float(dp[i - 1, j]) + 0.82, 2),
                    (float(dp[i, j - 1]) + 0.82, 3),
                )
                dp[i, j], bt[i, j] = min(candidates, key=lambda x: (x[0], x[1]))
        path: list[tuple[int | None, int | None, str]] = []
        exact = 0
        paired = 0
        i, j = n, m
        while i or j:
            code = int(bt[i, j])
            if i and j and code == 1:
                delta = observed[i - 1].pitch - score_notes[j - 1].pitch
                kind = "match" if delta == 0 else "substitute"
                path.append((i - 1, j - 1, kind))
                paired += 1
                exact += int(delta == 0)
                i -= 1
                j -= 1
            elif i and (not j or code == 2):
                path.append((i - 1, None, "extra"))
                i -= 1
            else:
                path.append((None, j - 1, "deletion"))
                j -= 1
        path.reverse()
        return (
            float(dp[n, m]) / max(n, m, 1),
            exact / max(paired, 1),
            path,
        )

    def _apply_uncertainty(
        self, observed: Sequence[ObservedNote], operations: list[AlignmentOperation]
    ) -> None:
        for op in operations:
            if op.performance_index is None:
                continue
            obs = observed[op.performance_index]
            threshold = (
                self.config.substitute_threshold
                if op.kind == "substitute"
                else self.config.unattached_threshold
            )
            if op.kind == "extra" and obs.confidence < self.config.unattached_threshold:
                op.kind = "unattached"
                op.confidence = obs.confidence
            elif op.kind in {"match", "substitute"} and op.confidence < threshold:
                op.kind = "unattached"
                op.score_index = None
                op.is_copy = False
                op.copy_id = None


def operations_confidence(
    operations: Sequence[AlignmentOperation], performance_index: int
) -> float:
    for op in operations:
        if op.performance_index == performance_index:
            return float(op.confidence)
    return 0.0


def align_notes(
    notes: Iterable[Any],
    score: ScoreGraph,
    *,
    checkpoint: Path | str | None = None,
    scorer: LearnedNoteScorer | None = None,
    config: NoteAlignConfig | None = None,
    device: str | torch.device = "cpu",
) -> AlignmentResult:
    """Convenience API for one note list and an existing :class:`ScoreGraph`."""

    aligner = (
        NoteAligner.from_checkpoint(checkpoint, device=device)
        if checkpoint is not None
        else NoteAligner(scorer, config, device=device)
    )
    return aligner.align(notes, score)


def alignment_metrics(
    prediction: AlignmentResult | Sequence[int | None],
    target_score_indices: Sequence[int | None],
    *,
    target_is_copy: Sequence[bool] | None = None,
) -> dict[str, float | int]:
    """Mapping metrics over all played notes and over replayed notes only."""

    pred_map = prediction.mapping if isinstance(prediction, AlignmentResult) else list(prediction)
    truth = [None if x is None or int(x) < 0 else int(x) for x in target_score_indices]
    if len(pred_map) != len(truth):
        raise ValueError(f"prediction/target length mismatch: {len(pred_map)} != {len(truth)}")
    pred_copy = [False] * len(pred_map)
    if isinstance(prediction, AlignmentResult):
        for op in prediction.operations:
            if op.performance_index is not None and op.is_copy:
                pred_copy[op.performance_index] = True
    copy_truth = list(target_is_copy or [False] * len(truth))
    if len(copy_truth) != len(truth):
        raise ValueError("target_is_copy length must match targets")

    def mapping_prf(indices: Sequence[int]) -> tuple[float, float, float, float, int]:
        correct = sum(pred_map[i] == truth[i] and truth[i] is not None for i in indices)
        predicted = sum(pred_map[i] is not None for i in indices)
        positives = sum(truth[i] is not None for i in indices)
        precision = correct / max(predicted, 1)
        recall = correct / max(positives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        accuracy = sum(pred_map[i] == truth[i] for i in indices) / max(len(indices), 1)
        return accuracy, precision, recall, f1, correct

    overall = mapping_prf(list(range(len(truth))))
    copy_indices = [i for i, flag in enumerate(copy_truth) if flag]
    copy_values = mapping_prf(copy_indices)
    copy_tp = sum(pred_copy[i] and copy_truth[i] and pred_map[i] == truth[i] for i in range(len(truth)))
    copy_pred = sum(pred_copy)
    copy_true = sum(copy_truth)
    copy_precision = copy_tp / max(copy_pred, 1)
    copy_recall = copy_tp / max(copy_true, 1)
    copy_f1 = 2 * copy_precision * copy_recall / max(copy_precision + copy_recall, 1e-12)
    return {
        "accuracy": overall[0],
        "precision": overall[1],
        "recall": overall[2],
        "f1": overall[3],
        "n_correct": overall[4],
        "n_notes": len(truth),
        "copy_accuracy": copy_values[0] if copy_indices else 0.0,
        "copy_mapping_precision": copy_values[1] if copy_indices else 0.0,
        "copy_mapping_recall": copy_values[2] if copy_indices else 0.0,
        "copy_precision": copy_precision,
        "copy_recall": copy_recall,
        "copy_f1": copy_f1,
        "n_copy_notes": copy_true,
    }


evaluate_alignment = alignment_metrics
NoteListAligner = NoteAligner


def checkpoint_payload(
    scorer: LearnedNoteScorer,
    config: NoteAlignConfig,
    *,
    epoch: int,
    metrics: Mapping[str, Any],
    calibration: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stable checkpoint schema shared by training and inference."""

    return {
        "model": scorer.state_dict(),
        "model_config": {
            "feature_dim": scorer.feature_dim,
            "hidden_dim": scorer.hidden_dim,
        },
        "align_config": asdict(config),
        "feature_names": FEATURE_NAMES,
        "operation_classes": OP_CLASSES,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "calibration": dict(calibration),
        "history": list(history),
        "format_version": 1,
    }
