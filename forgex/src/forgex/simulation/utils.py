from __future__ import annotations

import numpy as np


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Split-form sigmoid. Prevents overflow for large |logit|, which is a
    real failure mode once tenure_months and financial_stress terms compound
    across a multi-year simulation, not a theoretical edge case."""
    if not np.all(np.isfinite(logits)):
        bad = np.where(~np.isfinite(logits))[0]
        raise ValueError(
            f"Non-finite logits at indices {bad[:5].tolist()}"
            f"{'...' if len(bad) > 5 else ''}. Check upstream feature scaling."
        )
    out = np.empty_like(logits, dtype=np.float64)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_neg = np.exp(logits[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out


def compute_renewal_probability(
    cumulative_satisfaction: np.ndarray,
    financial_stress_at_decision: np.ndarray,
    rent_increase_pct: np.ndarray,
    market_gap_pct: np.ndarray,
    tenure_months: np.ndarray,
    rng: np.random.Generator,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """P(renew) = sigmoid(w1*satisfaction - w2*stress - w3*(rent_gap) +
    w4*tenure + shock). Mismatched array lengths are the #1 source of a
    silently-wrong synthetic cohort, so shape-check before any math runs."""
    if weights is None:
        weights = {"w1": 2.0, "w2": 2.5, "w3": 1.5, "w4": 0.04, "bias": -1.5, "shock_scale": 1.0}

    arrays = {
        "cumulative_satisfaction": cumulative_satisfaction,
        "financial_stress_at_decision": financial_stress_at_decision,
        "rent_increase_pct": rent_increase_pct,
        "market_gap_pct": market_gap_pct,
        "tenure_months": tenure_months,
    }
    lengths = {k: len(v) for k, v in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Length mismatch in renewal inputs: {lengths}")

    required_weights = {"w1", "w2", "w3", "w4", "shock_scale"}
    if missing := required_weights - weights.keys():
        raise KeyError(f"Missing DGP weights: {missing}")

    bias = weights.get("bias", 0.0)
    shock = rng.normal(0, weights["shock_scale"], size=lengths["tenure_months"])
    logits = (
        bias
        + weights["w1"] * cumulative_satisfaction
        - weights["w2"] * financial_stress_at_decision
        - weights["w3"] * (rent_increase_pct - market_gap_pct)
        + weights["w4"] * tenure_months
        + shock
    )
    logits = np.clip(logits, -30, 30)
    return stable_sigmoid(logits)


def assign_hidden_segments(
    household_size: np.ndarray,
    rng: np.random.Generator,
    maintenance_sensitive_base_rate: float = 0.5,
    household_size_effect: float = 0.15,
) -> np.ndarray:
    """Assigns each tenant a latent behavioral segment correlated with
    household_size but NEVER written to any feature or training column.
    This is the ground truth Phase 8's uplift model must recover blind.
    Leaking this into features turns your headline demo claim into a
    party trick. Grep your feature pipeline for this function's output
    variable name before every training run."""
    if not (0 < maintenance_sensitive_base_rate < 1):
        raise ValueError("base rate must be in (0, 1)")
    logit_base = np.log(maintenance_sensitive_base_rate / (1 - maintenance_sensitive_base_rate))
    logits = logit_base + household_size_effect * (household_size - household_size.mean())
    probs = stable_sigmoid(logits)
    draws = rng.random(len(household_size))
    return np.where(draws < probs, "maintenance_sensitive", "price_sensitive")
