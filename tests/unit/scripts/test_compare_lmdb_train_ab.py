"""Tests for the LMDB training A/B evidence producer."""

from __future__ import annotations

from pathlib import Path

from scripts.validation.compare_lmdb_train_ab import (
    build_result,
    compare_train_metrics,
    parse_lcurve,
    parse_wall_times,
)


def _lcurve(path: Path, *, energy: float, rmse: float) -> None:
    path.write_text(
        "# step mae_e_val mae_e_trn mae_f_val mae_f_trn mae_v_val mae_v_trn "
        "rmse_val rmse_trn lr\n"
        f"1 11.0 14.0 4.8 6.4 4.0 4.8 18.0 21.0 1.0e-04\n"
        f"100 10.0 11.0 5.0 7.0 4.0 4.0 18.0 20.0 8.8e-05\n"
        f"200 {energy} 10.9 5.1 7.0 4.0 4.0 {rmse} 19.0 1.0e-06\n",
        encoding="utf-8",
    )


def _log(path: Path, *, wall100: float, wall200: float) -> None:
    path.write_text(
        f"Batch 100: total wall time = {wall100:.2f} s\n"
        f"Batch 200: total wall time = {wall200:.2f} s\n",
        encoding="utf-8",
    )


def test_parse_lcurve_and_wall_times(tmp_path: Path) -> None:
    curve = tmp_path / "lcurve.out"
    log = tmp_path / "train.log"
    _lcurve(curve, energy=9.92, rmse=17.7)
    _log(log, wall100=36.52, wall200=36.25)

    assert parse_lcurve(curve)[200]["mae_e_val"] == 9.92
    assert parse_wall_times(log) == {100: 36.52, 200: 36.25}


def test_compare_train_metrics_accepts_bounded_stochastic_gap() -> None:
    mixed = parse_lcurve_from_values(9.92, 19.7, 11.0, 7.54, 5.17)
    lmdb = parse_lcurve_from_values(9.98, 19.5, 10.9, 7.23, 5.15)

    comparison = compare_train_metrics(
        mixed,
        lmdb,
        {100: 36.52, 200: 36.25},
        {100: 34.89, 200: 36.95},
        loss_threshold=0.05,
        wall_threshold=0.10,
    )

    assert comparison["passed"] is True
    assert comparison["loss_at_step_200"]["mae_e_val"]["relative_diff"] < 0.01


def test_build_result_requires_both_checkpoints(tmp_path: Path) -> None:
    mixed_data = tmp_path / "mixed"
    lmdb_data = tmp_path / "set.lmdb"
    mixed_data.mkdir()
    lmdb_data.mkdir()
    (mixed_data / "type.raw").write_text("0\n", encoding="utf-8")
    (lmdb_data / "data.mdb").write_bytes(b"lmdb")
    stat = tmp_path / "stat.hdf5"
    stat.write_bytes(b"stats")
    mixed_config = tmp_path / "mixed.json"
    lmdb_config = tmp_path / "lmdb.json"
    mixed_config.write_text("{}", encoding="utf-8")
    lmdb_config.write_text("{}", encoding="utf-8")
    mixed_log = tmp_path / "mixed.log"
    lmdb_log = tmp_path / "lmdb.log"
    _log(mixed_log, wall100=36.5, wall200=36.2)
    _log(lmdb_log, wall100=36.0, wall200=36.9)
    mixed_curve = tmp_path / "mixed.lcurve.out"
    lmdb_curve = tmp_path / "lmdb.lcurve.out"
    _lcurve(mixed_curve, energy=9.92, rmse=17.7)
    _lcurve(lmdb_curve, energy=9.98, rmse=17.8)
    mixed_models = tmp_path / "mixed-models"
    lmdb_models = tmp_path / "lmdb-models"
    mixed_models.mkdir()
    lmdb_models.mkdir()
    (mixed_models / "model.ckpt-200.pt").write_bytes(b"mixed")
    (lmdb_models / "model.ckpt-200.pt").write_bytes(b"lmdb")

    result = build_result(
        mixed_config=mixed_config,
        lmdb_config=lmdb_config,
        mixed_data=mixed_data,
        lmdb_data=lmdb_data,
        stat_file=stat,
        mixed_log=mixed_log,
        lmdb_log=lmdb_log,
        mixed_lcurve=mixed_curve,
        lmdb_lcurve=lmdb_curve,
        mixed_models=mixed_models,
        lmdb_models=lmdb_models,
        job_id="1",
    )

    assert result["status"] == "finished"
    assert result["metric_equivalence_claimed"] is True
    assert result["sha256"]["mixed_checkpoint_200"] is not None
    assert result["sha256"]["lmdb_checkpoint_200"] is not None


def test_execution_only_keeps_stochastic_loss_informational(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    config = tmp_path / "input.json"
    config.write_text("{}", encoding="utf-8")
    stat = tmp_path / "stat.hdf5"
    stat.write_bytes(b"stats")
    mixed_log = tmp_path / "mixed.log"
    lmdb_log = tmp_path / "lmdb.log"
    _log(mixed_log, wall100=36.0, wall200=36.0)
    _log(lmdb_log, wall100=36.0, wall200=36.0)
    mixed_curve = tmp_path / "mixed.lcurve.out"
    lmdb_curve = tmp_path / "lmdb.lcurve.out"
    _lcurve(mixed_curve, energy=9.0, rmse=17.0)
    _lcurve(lmdb_curve, energy=12.0, rmse=20.0)
    mixed_models = tmp_path / "mixed-models"
    lmdb_models = tmp_path / "lmdb-models"
    mixed_models.mkdir()
    lmdb_models.mkdir()
    (mixed_models / "model.ckpt-200.pt").write_bytes(b"mixed")
    (lmdb_models / "model.ckpt-200.pt").write_bytes(b"lmdb")

    result = build_result(
        mixed_config=config,
        lmdb_config=config,
        mixed_data=data,
        lmdb_data=data,
        stat_file=stat,
        mixed_log=mixed_log,
        lmdb_log=lmdb_log,
        mixed_lcurve=mixed_curve,
        lmdb_lcurve=lmdb_curve,
        mixed_models=mixed_models,
        lmdb_models=lmdb_models,
        execution_only=True,
    )

    assert result["status"] == "finished"
    assert result["metric_equivalence_claimed"] is False
    assert result["metrics"]["comparison"]["passed"] is False


def parse_lcurve_from_values(
    energy: float, rmse: float, mae_e_trn: float, mae_f_trn: float, mae_f_val: float
):
    return {
        200: {
            "mae_e_val": energy,
            "mae_e_trn": mae_e_trn,
            "mae_f_val": mae_f_val,
            "mae_f_trn": mae_f_trn,
            "mae_v_val": 4.0,
            "mae_v_trn": 4.0,
            "rmse_val": 17.7,
            "rmse_trn": rmse,
            "lr": 1e-6,
        }
    }
