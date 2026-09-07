"""V6 linear-chain CRF on MelodyFirst 7-way types. Constrained Viterbi (max error run 16)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from alignmodel.bakeoff.even_common import copies_and_coverage, decode_runs_no_tile, notes_from_tensors
from alignmodel.config import FRAME_HOP_SEC, MELODY_NOTE_CLASSES, ModelConfig
from alignmodel.melody import load_bundle_notes
from alignmodel.melody_model import MelodyFirst, class_index
from alignmodel.melody_train import MelodyBundleDataset, MelodyTrainConfig

VARIANT = "v6"
RUN_ID = "melody-b-v6-crf"
MAX_ERROR_RUN = 16
MATCH_I = class_index("match")
NEG = -1e4


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def log_partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
        bsz, _steps, _n_class = emissions.shape
        alpha = self.start + emissions[:, 0]
        for t in range(1, _steps):
            nxt = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )
            nxt = torch.logsumexp(nxt, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def sequence_score(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        bsz, _steps, _n_class = emissions.shape
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, _steps):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score = score + (trans + emit) * mask[:, t].float()
        lengths = mask.long().sum(1).clamp_min(1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        return score + self.end[last_tags]

    def nll(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        return (self.log_partition(emissions, mask) - self.sequence_score(emissions, tags, mask)).mean()

    def _expanded_maps(self, device: torch.device, max_run: int) -> tuple[Tensor, Tensor, Tensor]:
        n_class = self.num_tags
        err_ids = [c for c in range(n_class) if c != MATCH_I]
        n_state = 1 + len(err_ids) * max_run
        cls_of = torch.zeros(n_state, dtype=torch.long, device=device)
        trans_exp = torch.full((n_state, n_state), NEG, device=device)
        cls_of[0] = MATCH_I
        trans_exp[0, 0] = self.transitions[MATCH_I, MATCH_I]
        for e, src in enumerate(err_ids):
            s1 = 1 + e * max_run
            trans_exp[0, s1] = self.transitions[MATCH_I, src]
            for k in range(max_run):
                s = s1 + k
                cls_of[s] = src
                trans_exp[s, 0] = self.transitions[src, MATCH_I]
                if k + 1 < max_run:
                    nxt = k + 1
                    for e2, dst in enumerate(err_ids):
                        trans_exp[s, 1 + e2 * max_run + nxt] = self.transitions[src, dst]
        return trans_exp, cls_of, torch.tensor(n_state, device=device)

    @torch.no_grad()
    def decode(self, emissions: Tensor, mask: Tensor, max_error_run: int = MAX_ERROR_RUN) -> list[list[int]]:
        results: list[list[int]] = []
        trans_exp, cls_of, n_state_t = self._expanded_maps(emissions.device, max_error_run)
        n_state = int(n_state_t.item())
        start_exp = torch.full((n_state,), NEG, device=emissions.device)
        start_exp[0] = self.start[MATCH_I]
        err_ids = [c for c in range(self.num_tags) if c != MATCH_I]
        for e, src in enumerate(err_ids):
            start_exp[1 + e * max_error_run] = self.start[src]
        end_exp = self.end[cls_of]
        for b in range(emissions.size(0)):
            n = int(mask[b].sum().item())
            emit = emissions[b, :n]
            emit_exp = emit[:, cls_of]
            dp = emit_exp[0] + start_exp
            back = []
            for t in range(1, n):
                scores = dp.unsqueeze(1) + trans_exp + emit_exp[t].unsqueeze(0)
                best, arg = scores.max(dim=0)
                back.append(arg)
                dp = best
            dp = dp + end_exp
            last = int(dp.argmax().item())
            states = [last]
            for arg in reversed(back):
                last = int(arg[last].item())
                states.append(last)
            states.reverse()
            results.append([int(cls_of[s].item()) for s in states])
        return results


class MelodyCRF(MelodyFirst):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__(cfg)
        self.crf = LinearChainCRF(len(MELODY_NOTE_CLASSES))

    def forward(
        self,
        mel: Tensor,
        mel_mask: Tensor,
        pitch: Tensor,
        onset: Tensor,
        duration: Tensor,
        note_mask: Tensor,
        hop_sec: float,
    ) -> dict[str, Tensor]:
        out = super().forward(mel, mel_mask, pitch, onset, duration, note_mask, hop_sec)
        out["crf"] = self.crf
        return out


def build_model(cfg: ModelConfig) -> nn.Module:
    return MelodyCRF(cfg)


def compute_loss(
    outputs: dict[str, Tensor],
    batch: dict,
    cfg: MelodyTrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    crf: LinearChainCRF = outputs["crf"]
    type_logits = outputs["type_logits"]
    note_mask = batch["note_mask"]
    y = batch["score_y"]
    type_loss = crf.nll(type_logits, y, note_mask)
    extra, copies_loss, coverage = copies_and_coverage(
        type_logits,
        outputs["copies_logits"],
        note_mask,
        batch["copies_y"],
        cfg.copies_loss_weight,
        cfg.coverage_loss_weight,
    )
    with torch.no_grad():
        pred = type_logits.argmax(-1)
        err_mask = note_mask & (y != MATCH_I)
        err_loss = 1.0 - ((pred == y) & err_mask).float().sum() / err_mask.sum().clamp_min(1)
    total = type_loss + extra
    return total, {
        "loss": float(total.detach()),
        "type": float(type_loss.detach()),
        "err": float(err_loss.detach()),
        "copies": float(copies_loss.detach()),
        "coverage": float(coverage.detach()),
    }


def _types_from_ids(ids: list[int]) -> list[str]:
    return [MELODY_NOTE_CLASSES[int(i)] for i in ids]


def decode_outputs(outputs: dict[str, Tensor], batch: dict, b: int) -> tuple[list[str], list[dict]]:
    crf: LinearChainCRF = outputs["crf"]
    n = int(batch["n_notes"][b])
    mask = batch["note_mask"][b : b + 1, :n]
    ids = crf.decode(outputs["type_logits"][b : b + 1, :n], mask, MAX_ERROR_RUN)[0]
    types = _types_from_ids(ids)
    copies = int(outputs["copies_logits"][b].argmax(-1).cpu())
    notes = notes_from_tensors(batch["pitch"][b], batch["onset"][b], batch["duration"][b], n)
    labels = decode_runs_no_tile(types, notes, copies)
    return types, labels


@torch.no_grad()
def infer_sample(model: MelodyCRF, sample_dir: Path, device: torch.device) -> dict:
    sample_dir = Path(sample_dir)
    ds = MelodyBundleDataset([sample_dir], model.cfg)
    item = ds[0]
    out = model(
        item["mel"].unsqueeze(0).to(device),
        item["mel_mask"].unsqueeze(0).to(device),
        item["pitch"].unsqueeze(0).to(device),
        item["onset"].unsqueeze(0).to(device),
        item["duration"].unsqueeze(0).to(device),
        item["note_mask"].unsqueeze(0).to(device),
        FRAME_HOP_SEC,
    )
    n = int(item["n_notes"])
    mask = item["note_mask"][:n].unsqueeze(0).to(device)
    ids = model.crf.decode(out["type_logits"][:, :n], mask, MAX_ERROR_RUN)[0]
    types = _types_from_ids(ids)
    extra_copies = int(out["copies_logits"][0].argmax(-1).cpu())
    notes = load_bundle_notes(sample_dir)[:n]
    labels = decode_runs_no_tile(types, notes, extra_copies)
    return {
        "sample_id": sample_dir.name,
        "labels": labels,
        "extra_copies": extra_copies,
        "note_types": types,
    }


def pred_type_ids(outputs: dict[str, Tensor], n: int, b: int = 0) -> Tensor:
    crf: LinearChainCRF = outputs["crf"]
    mask = torch.ones(1, n, dtype=torch.bool, device=outputs["type_logits"].device)
    ids = crf.decode(outputs["type_logits"][b : b + 1, :n], mask, MAX_ERROR_RUN)[0]
    return torch.tensor(ids, device=outputs["type_logits"].device, dtype=torch.long)
