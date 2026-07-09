import numpy as np
import pytest

from forgex.simulation.utils import stable_sigmoid, compute_renewal_probability, assign_hidden_segments


def test_stable_sigmoid_basic():
    logits = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    probs = stable_sigmoid(logits)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert probs[0] < 0.5
    assert probs[-1] > 0.5


def test_stable_sigmoid_extreme_values():
    logits = np.array([-100.0, 100.0])
    probs = stable_sigmoid(logits)
    assert probs[0] == pytest.approx(0.0, abs=1e-10)
    assert probs[1] == pytest.approx(1.0, abs=1e-10)


def test_stable_sigmoid_non_finite_raises():
    with pytest.raises(ValueError, match="Non-finite logits"):
        stable_sigmoid(np.array([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError, match="Non-finite logits"):
        stable_sigmoid(np.array([1.0, np.inf, 3.0]))


def test_compute_renewal_probability_shape_mismatch():
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_renewal_probability(
            np.array([0.5, 0.6]),
            np.array([0.3]),
            np.array([0.1]),
            np.array([0.05]),
            np.array([12]),
            rng,
        )


def test_compute_renewal_probability_missing_weights():
    rng = np.random.default_rng(42)
    with pytest.raises(KeyError, match="Missing DGP weights"):
        compute_renewal_probability(
            np.array([0.5]),
            np.array([0.3]),
            np.array([0.1]),
            np.array([0.05]),
            np.array([12]),
            rng,
            weights={"w1": 1.0},
        )


def test_compute_renewal_probability_basic():
    rng = np.random.default_rng(42)
    prob = compute_renewal_probability(
        np.array([0.8]),
        np.array([0.1]),
        np.array([0.02]),
        np.array([0.05]),
        np.array([24]),
        rng,
    )
    assert 0.0 <= prob[0] <= 1.0


def test_assign_hidden_segments():
    rng = np.random.default_rng(42)
    sizes = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    segments = assign_hidden_segments(sizes, rng)
    assert len(segments) == 6
    assert all(s in {"maintenance_sensitive", "price_sensitive"} for s in segments)


def test_assign_hidden_segments_invalid_rate():
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="base rate must be in"):
        assign_hidden_segments(np.array([1.0, 2.0]), rng, maintenance_sensitive_base_rate=1.5)
