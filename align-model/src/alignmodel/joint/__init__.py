"""Canonical score-event indexing and strict joint evaluation."""

from .end_to_end import (
    EndToEndTrainConfig,
    load_end_to_end_model,
    train_end_to_end_model,
)
from .error_heads import (
    ErrorHeadsConfig,
    FrozenUpstreamErrorHeads,
    filter_prediction_for_schema,
    infer_frozen_error_pipeline,
    load_error_heads,
)
from .index import JointEvent, ScoreEvent, ScoreEventIndex
from .lattice import (
    JointCandidate,
    JointEdgeScorer,
    JointOperation,
    LatticeConfig,
    LatticePath,
    SparseJointLattice,
)
from .metrics import (
    DEFAULT_TOLERANCES_SEC,
    JointMetricSample,
    evaluate_joint_dataset,
    evaluate_joint_events,
    pair_exact_pitch_onset,
)
from .oracle import OracleHarnessConfig, run_oracle_harness

__all__ = [
    "DEFAULT_TOLERANCES_SEC",
    "EndToEndTrainConfig",
    "ErrorHeadsConfig",
    "FrozenUpstreamErrorHeads",
    "JointEvent",
    "JointCandidate",
    "JointEdgeScorer",
    "JointMetricSample",
    "JointOperation",
    "LatticeConfig",
    "LatticePath",
    "OracleHarnessConfig",
    "ScoreEvent",
    "ScoreEventIndex",
    "SparseJointLattice",
    "evaluate_joint_dataset",
    "evaluate_joint_events",
    "filter_prediction_for_schema",
    "load_end_to_end_model",
    "infer_frozen_error_pipeline",
    "load_error_heads",
    "pair_exact_pitch_onset",
    "run_oracle_harness",
    "train_end_to_end_model",
]
