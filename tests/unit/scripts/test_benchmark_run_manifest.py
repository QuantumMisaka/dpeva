import json

from scripts.benchmark_run_manifest import percentile, run_benchmark


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 95) == 4.0


def test_benchmark_has_machine_readable_summary() -> None:
    result = run_benchmark(repetitions=2, warmups=0)
    assert result["repetitions"] == 2
    assert result["percentile_method"].startswith("nearest-rank")
    assert len(result["overhead_ms"]) == 2
    json.dumps(result)
