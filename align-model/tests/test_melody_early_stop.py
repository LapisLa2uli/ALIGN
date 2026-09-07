from pathlib import Path

from alignmodel.melody_train import (
    MelodyTrainConfig,
    dump_train_history,
    step_ema_is_plateau,
    step_ema_update,
)


def test_step_ema_update_and_plateau():
    ema = step_ema_update(None, 1.0)
    assert ema == 1.0
    ema = step_ema_update(ema, 1.0, alpha=0.05)
    assert abs(ema - 1.0) < 1e-9
    assert step_ema_is_plateau(1.0005, 1.0, rel_tol=0.002)
    assert not step_ema_is_plateau(1.01, 1.0, rel_tol=0.002)


def test_es_defaults_on_and_history_fields(tmp_path: Path):
    cfg = MelodyTrainConfig()
    assert cfg.es_min_steps == 80
    assert cfg.es_plateau_steps == 150
    assert cfg.es_ema_alpha == 0.05
    assert cfg.es_rel_tol == 0.002
    assert cfg.es_f1_delta == 0.005
    assert cfg.es_patience_epochs == 2
    path = tmp_path / "history.json"
    dump_train_history(path, [{"epoch": 2, "step": 99}], "val_f1_patience", 2, 99)
    blob = path.read_text(encoding="utf-8")
    assert "stopped_reason" in blob
    assert "val_f1_patience" in blob
    assert '"epoch": 2' in blob
    assert '"step": 99' in blob
