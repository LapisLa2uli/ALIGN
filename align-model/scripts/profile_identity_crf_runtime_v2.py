"""Profile and prove fast-v2 CRF equivalence on representative frozen rows."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import train_identity_crf_v1 as training
from alignmodel.joint.identity_crf_fast_v2 import (
    SCHEMA_VERSION,
    fast_decode_identity_crf,
    fast_identity_crf_nll,
)
from alignmodel.joint.identity_crf_v1 import (
    OrnamentIdentityCRF,
    decode_identity_crf,
    identity_crf_nll,
)
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _gradients(model: OrnamentIdentityCRF) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--max-negative-hypotheses", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    if sha256_file(args.release_manifest) != args.expected_release_sha256:
        raise ValueError("Frozen release hash mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
    if not coverage["all_gold_paths_covered"]:
        raise ValueError("Coverage gate failed")
    targets = training._targets(release)
    examples = training._build_examples(
        release,
        "train",
        targets,
        max_rows=args.rows,
        max_negative_hypotheses=args.max_negative_hypotheses,
    )
    torch.manual_seed(args.seed)
    reference = OrnamentIdentityCRF(hidden=args.hidden)
    fast = copy.deepcopy(reference)
    # Compile Numba before measured work.
    warm_loss, _ = fast_identity_crf_nll(
        fast, examples[0]["lattice"], normalize=True
    )
    warm_loss.backward()
    fast.zero_grad(set_to_none=True)
    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track="identity-crf-fast-v2-profile",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(examples), "locked_test": False},
    ):
        started = time.perf_counter()
        reference_loss = (
            sum(
                identity_crf_nll(
                    reference, example["lattice"], normalize=True
                )[0]
                for example in examples
            )
            / len(examples)
        )
        reference_loss.backward()
        reference_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fast_loss = (
            sum(
                fast_identity_crf_nll(
                    fast, example["lattice"], normalize=True
                )[0]
                for example in examples
            )
            / len(examples)
        )
        fast_loss.backward()
        fast_seconds = time.perf_counter() - started
    loss_delta = abs(float(reference_loss) - float(fast_loss))
    reference_grad = _gradients(reference)
    fast_grad = _gradients(fast)
    gradient_deltas = {
        name: float((reference_grad[name] - fast_grad[name]).abs().max())
        for name in reference_grad
    }
    viterbi = []
    for example in examples:
        reference_events, reference_deletions, reference_diagnostics = (
            decode_identity_crf(reference, example["lattice"])
        )
        fast_events, fast_deletions, fast_diagnostics = (
            fast_decode_identity_crf(fast, example["lattice"])
        )
        viterbi.append(
            {
                "sample": example["sample"],
                "events_equal": fast_events == reference_events,
                "deletions_equal": fast_deletions == reference_deletions,
                "actions_equal": fast_diagnostics["actions"]
                == reference_diagnostics["actions"],
                "copy_state_equal": (
                    fast_diagnostics["copies"],
                    fast_diagnostics["source_span"],
                )
                == (
                    reference_diagnostics["copies"],
                    reference_diagnostics["source_span"],
                ),
            }
        )
    passed = (
        loss_delta <= 5e-5
        and max(gradient_deltas.values(), default=0.0) <= 1e-4
        and all(
            all(
                row[name]
                for name in (
                    "events_equal",
                    "deletions_equal",
                    "actions_equal",
                    "copy_state_equal",
                )
            )
            for row in viterbi
        )
    )
    report = {
        "schema_version": f"{SCHEMA_VERSION}-profile",
        "release_manifest_sha256": sha256_file(args.release_manifest),
        "coverage_report_sha256": sha256_file(args.coverage_report),
        "rows": len(examples),
        "reference": {
            "seconds": reference_seconds,
            "rows_per_second": len(examples) / reference_seconds,
        },
        "fast_v2": {
            "seconds": fast_seconds,
            "rows_per_second": len(examples) / fast_seconds,
        },
        "speedup": reference_seconds / max(fast_seconds, 1e-9),
        "loss_delta": loss_delta,
        "maximum_parameter_gradient_delta": max(
            gradient_deltas.values(), default=0.0
        ),
        "gradient_deltas": gradient_deltas,
        "viterbi": viterbi,
        "passed": passed,
        "open_validation_read": False,
        "lockbox_targets_read": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    import argparse

    raise SystemExit(main())
