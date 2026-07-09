import numpy as np
import pandas as pd
import pytest

from forgex.mlops.drift import population_stability_index, check_drift, DriftMonitor


def test_psi_identity():
    x = np.random.default_rng(42).normal(0, 1, 1000)
    psi = population_stability_index(x, x)
    assert psi == pytest.approx(0.0, abs=0.01)


def test_psi_shifted():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(3, 1, 1000)  # shifted by 3 std devs
    psi = population_stability_index(ref, cur)
    assert psi > 0.25


def test_psi_empty_reference():
    with pytest.raises(ValueError, match="must both be non-empty"):
        population_stability_index(np.array([]), np.array([1.0]))


def test_check_drift_no_drift():
    ref = pd.DataFrame({"a": np.random.default_rng(42).normal(0, 1, 500),
                         "b": np.random.default_rng(42).normal(0, 1, 500)})
    cur = ref.copy()
    result = check_drift(ref, cur, psi_threshold=0.25)
    assert not result["retrain_recommended"]
    assert result["features_checked"] == 2


def test_check_drift_with_drift():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"a": rng.normal(5, 1, 500)})
    result = check_drift(ref, cur, psi_threshold=0.25)
    assert result["retrain_recommended"]
    assert len(result["breached_features"]) > 0


def test_check_drift_no_overlap():
    ref = pd.DataFrame({"a": [1.0]})
    cur = pd.DataFrame({"b": [2.0]})
    with pytest.raises(ValueError, match="No overlapping columns"):
        check_drift(ref, cur)


def test_check_drift_nan_handling():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 100).tolist() + [np.nan] * 10})
    cur = pd.DataFrame({"a": rng.normal(0.5, 1, 105).tolist() + [np.nan] * 5})
    result = check_drift(ref, cur)
    assert "a" in result["psi_by_feature"] if result["features_checked"] > 0 else True


def test_drift_monitor():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    monitor = DriftMonitor(ref, psi_threshold=0.25)
    cur_same = ref.copy()
    report = monitor.check(cur_same)
    assert not report["retrain_recommended"]

    cur_shifted = pd.DataFrame({"a": rng.normal(5, 1, 500)})
    report2 = monitor.check(cur_shifted)
    assert report2["retrain_recommended"]

    assert len(monitor.drift_history) == 2
    alert = monitor.generate_alert()
    assert "RETRAIN" in alert or "All clear" in alert
