# ForgeX — Vital-Signs Monitoring for the Landlord-Tenant Relationship

**Version:** 1.0.0  
**Python:** >= 3.11  
**Architecture:** Modular monolith with FastAPI backend, Streamlit dashboard, and React frontend

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [System Architecture](#3-system-architecture)
4. [Data Generation (Phase 1)](#4-data-generation-phase-1)
5. [Feature Engineering (Phase 2)](#5-feature-engineering-phase-2)
6. [Person-Period Reshape (Phase 3)](#6-person-period-reshape-phase-3)
7. [Baseline Model (Phase 4)](#7-baseline-model-phase-4)
8. [Primary Hazard Model (Phase 5)](#8-primary-hazard-model-phase-5)
9. [Explainability (Phase 6)](#9-explainability-phase-6)
10. [Causal Uplift (Phase 8)](#10-causal-uplift-phase-8)
11. [Fairness Audit (Phase 13)](#11-fairness-audit-phase-13)
12. [MLOps & Champion/Challenger (Phase 15)](#12-mlops--championchallenger-phase-15)
13. [Budget Optimization (Phase 16)](#13-budget-optimization-phase-16)
14. [Evaluation (Phase 17)](#14-evaluation-phase-17)
15. [REST API](#15-rest-api)
16. [Streamlit Dashboard](#16-streamlit-dashboard)
17. [React Frontend — What-If Simulator](#17-react-frontend--what-if-simulator)
18. [Docker Deployment](#18-docker-deployment)
19. [Configuration](#19-configuration)
20. [Testing](#20-testing)
21. [How to Run Everything](#21-how-to-run-everything)
22. [Important Notes for Cloning](#22-important-notes-for-cloning)

---

## 1. Overview

ForgeX is an end-to-end platform that predicts tenant churn risk, explains the causes with SHAP values, audits for demographic fairness, and recommends optimal budget-constrained retention interventions using causal uplift modeling.

It is built around **synthetic data** — a simulated world of 10,000 tenants, 120 properties, and 6+ years of lease/payment/maintenance events — so every claim (risk score, fairness improvement, uplift validation) is **provably verifiable** against ground truth.

### Key Claims

| Claim | How It Is Proven |
|-------|-----------------|
| "We predict churn risk" | LightGBM hazard model with OOF PR-AUC |
| "We explain why" | SHAP TreeExplainer with human-readable driver labels |
| "We recover hidden segments without labels" | CATE uplift model t-test vs. hidden segmentation (Phase 8) |
| "We eliminate historical bias" | Demographic parity before/after comparison (Phase 13) |
| "We optimize budget allocation" | Integer linear program solving the knapsack problem (Phase 16) |

---

## 2. Project Structure

```
C:\Users\LOUJAN B\ForgeX\
├── ourproject.md                  # This file
├── forgex/                       # Main project root
│   ├── .env                      # Configuration (gitignored — create from .env.example)
│   ├── .env.example              # Template for .env
│   ├── .gitignore                # Git exclusion rules
│   ├── pyproject.toml            # Python project metadata & dependencies
│   ├── setup.py                  # Editable install compatibility
│   ├── requirements.txt          # Pin-compatible dependency versions
│   ├── MANIFEST.in               # Package manifest
│   │
│   ├── src/forgex/               # Core Python package
│   │   ├── __init__.py           # Package init (__version__ = "1.0.0")
│   │   ├── __main__.py           # Entry point: python -m forgex [api|dashboard|pipeline|generate|test]
│   │   ├── config.py             # Pydantic Settings (env-based config)
│   │   ├── errors.py             # Custom exception hierarchy
│   │   ├── logging_setup.py      # Centralized logger
│   │   ├── pipeline.py           # End-to-end training pipeline (Phases 1-17)
│   │   │
│   │   ├── simulation/           # Synthetic data generation
│   │   │   ├── __init__.py
│   │   │   ├── generator.py      # SyntheticWorldEngine — orchestrator
│   │   │   ├── entities.py       # Generate tenants, properties, units
│   │   │   ├── events.py         # Generate leases, payments, maintenance, renewals
│   │   │   ├── hidden_state.py   # Evolve satisfaction & financial stress month-by-month
│   │   │   ├── market.py         # Generate market rent with seasonality & drift
│   │   │   ├── bias_injection.py # Inject historical bias against voucher holders
│   │   │   ├── utils.py          # stable_sigmoid, renewal probability, segment assignment
│   │   │   └── validation.py     # Validate synthetic world integrity
│   │   │
│   │   ├── features/             # Feature engineering
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py       # FeaturePipeline — payment, maintenance, lease, market, intervention features
│   │   │   └── nlp.py            # Maintenance text severity/sentiment tagging
│   │   │
│   │   ├── reshape/              # Person-period reshape
│   │   │   ├── __init__.py
│   │   │   └── person_period.py  # explode_to_person_period, validate, tenant_level_split
│   │   │
│   │   ├── models/               # Machine learning models
│   │   │   ├── __init__.py
│   │   │   ├── baseline.py       # LogisticRegression baseline + survival curve math
│   │   │   ├── hazard.py         # LightGBM hazard model with GroupKFold + isotonic calibration
│   │   │   ├── uplift.py         # T-Learner CATE + CausalForestDML fallback
│   │   │   └── tft.py            # Temporal Fusion Transformer (pytorch-forecasting)
│   │   │
│   │   ├── explain/              # Model interpretability
│   │   │   ├── __init__.py
│   │   │   ├── shap_explainer.py  # SHAP TreeExplainer + LinearExplainer
│   │   │   └── narrative.py      # LLM-based narrative generation with template fallback
│   │   │
│   │   ├── fairness/             # Fairness auditing
│   │   │   ├── __init__.py
│   │   │   └── audit.py          # Demographic parity, equalized odds, bias correction demo
│   │   │
│   │   ├── optimize/             # Budget optimization
│   │   │   ├── __init__.py
│   │   │   └── ilp_optimizer.py  # Integer Linear Program (PuLP) + greedy fallback
│   │   │
│   │   ├── mlops/                # MLOps infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── drift.py          # Population Stability Index (PSI) monitoring
│   │   │   ├── champion.py       # Champion/challenger promotion with fairness gate
│   │   │   ├── evaluation.py     # Evaluation metrics (PR-AUC, ROC-AUC, Brier, C-index)
│   │   │   └── demo_fallback.py  # Precomputed fallback responses for demos
│   │   │
│   │   └── integrations/         # External integrations
│   │       ├── __init__.py
│   │       └── webhooks.py       # HMAC-SHA256 webhook signature verification
│   │
│   ├── tests/                    # Test suite (pytest)
│   │   ├── __init__.py
│   │   ├── conftest.py           # Shared fixtures
│   │   ├── test_api_schemas.py   # Pydantic schema validation
│   │   ├── test_drift.py         # PSI and DriftMonitor
│   │   ├── test_errors.py        # Exception hierarchy
│   │   ├── test_fairness.py      # Fairness audit and gate
│   │   ├── test_models.py        # Baseline, hazard, uplift model unit tests
│   │   ├── test_nlp.py           # Maintenance text tagging
│   │   ├── test_reshape.py       # Person-period explode and validation
│   │   └── test_simulation_utils.py  # Sigmoid, renewal probability, segment assignment
│   │
│   ├── artifacts/                 # Trained model artifacts (gitignored)
│   │   ├── .gitkeep
│   │   ├── hazard_model.pkl       # LightGBM model + feature names + metadata
│   │   ├── shap_explainer.pkl     # SHAP explainer object
│   │   ├── survival_curves.parquet # Precomputed survival curves
│   │   ├── evaluation_report.json  # Test metrics report
│   │   └── cate/                  # CATE model pickles (optional)
│   │       └── cate_t_learner.pkl
│   │
│   ├── data/                      # Data (gitignored except .gitkeep)
│   │   ├── raw/                   # Raw synthetic parquet files
│   │   │   ├── .gitkeep
│   │   │   ├── tenants.parquet
│   │   │   ├── properties.parquet
│   │   │   ├── units.parquet
│   │   │   ├── leases.parquet
│   │   │   ├── payments.parquet
│   │   │   ├── maintenance_requests.parquet
│   │   │   ├── market_comps.parquet
│   │   │   ├── intervention_log.parquet
│   │   │   ├── hidden_segments.parquet
│   │   │   └── hidden_bias_ground_truth.parquet
│   │   ├── interim/              # Intermediate data
│   │   │   └── .gitkeep
│   │   └── processed/           # Processed data
│   │       ├── .gitkeep
│   │       ├── person_period.parquet
│   │       ├── feature_table.parquet
│   │       ├── baseline_survival_curves.parquet
│   │       └── split_ids.json
│   │
│   ├── frontend/                  # Frontend applications
│   │   ├── package.json           # Node.js dependencies (React 18, Babel, serve)
│   │   ├── dashboard.py           # Streamlit dashboard (portfolio + tenant detail)
│   │   ├── src/                  # React What-If Simulator source
│   │   │   ├── index.html         # HTML entry with inline Babel JSX
│   │   │   ├── styles.css         # Dark theme CSS
│   │   │   └── components/
│   │   │       └── WhatIfSimulator.jsx
│   │   └── public/               # Static served directory
│   │       ├── index.html        # Inline-Babel single-page app
│   │       └── styles.css
│   │
│   ├── docker/                    # Docker deployment
│   │   ├── Dockerfile.api          # Multi-stage Python 3.11-slim
│   │   ├── Dockerfile.frontend    # Nginx alpine
│   │   ├── docker-compose.yml     # Two-service orchestration
│   │   └── nginx.conf            # Reverse proxy /api/ -> api:8000
│   │
│   └── configs/
│       └── config.yaml           # YAML configuration mirror
```

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ForgeX System                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   DATA GENERATION (Phase 1)                   │ │
│  │  SyntheticWorldEngine → tenants, leases, payments, maint.    │ │
│  │  Hidden states (satisfaction, stress) → ground truth         │ │
│  │  Bias injection → historical unfair flagging                 │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                FEATURE PIPELINE (Phase 2)                    │ │
│  │  Point-in-time features: payments, maintenance, lease,      │ │
│  │  market, intervention — NO target leakage                 │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │            PERSON-PERIOD RESHAPE (Phase 3)         │ │
│  │  explode_to_person_period → one row per month     │ │
│  │  tenant_level_split → leak-proof train/val/test │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                 MODEL TRAINING (Phases 4-8)                │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │ │
│  │  │ Baseline │  │ LightGBM │  │  SHAP   │  │  CATE   │       │ │
│  │  │ Logistic │  │ Hazard   │  │Explain  │  │ Uplift  │       │ │
│  │  │  (LogR)  │  │ (GBM)   │  │  (Tree) │  │(T-Learn)│       │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                FAIRNESS AUDIT (Phase 13)                     │ │
│  │  Demographic parity difference (voucher vs non-voucher)  │ │
│  │  Bias correction demonstration                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │            DEPLOYMENT & SERVING                             │ │
│  │                                                              │ │
│  │  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐ │ │
│  │  │   FastAPI    │    │   Streamlit   │   │   React      │ │ │
│  │  │  localhost:8000│   │ localhost:8501│   │localhost:3000│ │
│  │  │  /health     │    │  Portfolio    │   │ Tenant Sim   │ │ │
│  │  │  /score/{tid}│    │  Tenant Detail│   │ What-If     │ │ │
│  │  │  /simulate   │    │  Risk Heatmap │   │ Scenario    │ │ │
│  │  │  /optimize   │    │  SurvivalCurve│   │ Projections │ │ │
│  │  │  /docs       │    │  SHAP Drivers │   │ Recommends  │ │ │
│  │  └──────────────┘   └──────────────┘   └──────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Generation (Phase 1)

### SyntheticWorldEngine (`simulation/generator.py`)

Generates a complete synthetic property management ecosystem:

| Table | Rows | Description |
|-------|------|-------------|
| `tenants` | 10,000 | tenant_id, household_size (1-6), voucher_holder (15%), zip_code, move_in_date |
| `properties` | 120 | property_id, neighborhood_cluster (10 clusters), property_type, year_built, units_per_property |
| `units` | ~960 | unit_id (per property, Poisson λ=8), sqft, bedrooms, base_rent |
| `leases` | ~20,000 | lease_id, tenant_id, unit_id, start/end dates, initial/current rent, did_renew |
| `payments` | ~300,000 | tenant_id, month, amount_due, amount_paid, days_late, is_late |
| `maintenance_requests` | ~8,000 | tenant_id, request_date, text, severity_ground_truth, days_to_resolve |
| `market_comps` | ~1,300 | neighborhood_cluster, month, market_rent (with seasonality + drift) |
| `intervention_log` | ~3,000 | tenant_id, intervention_type, cost, crew_hours, accepted, renewed |

### Hidden State Simulation (`simulation/hidden_state.py`)

Every tenant has two hidden variables that evolve month-by-month and drive all observable behavior:

- **satisfaction** (0.01-0.99): drifts with noise, eroded by maintenance issues (-0.02 to -0.12), shocked by life events (-0.1 to -0.3), mean-reverts
- **financial_stress** (0.01-0.99): driven by household size, rent increases (+0.03 to +0.10), life events, mean-reverts

These are NEVER exposed as features — they are the ground truth that downstream models must infer.

### Hidden Segment Assignment (`simulation/utils.py:assign_hidden_segments`)

Each tenant is assigned a latent behavioral segment:
- **maintenance_sensitive** (correlated with larger households)
- **price_sensitive**

This segmentation is NEVER written to features. The CATE model must recover it blind.

### Bias Injection (`simulation/bias_injection.py`)

Injects deliberate historical bias into legacy risk flags:
- Voucher holders get a +0.12 penalty to their risk score
- Certain ZIP codes (ZIP_0003, ZIP_0007, ZIP_0012, ZIP_0019) get +0.06 penalty
- Combined with actual late payment history → legacy_risk_flag

This ground truth enables Phase 13's fairness audit to prove bias correction.

### Renewal Probability (`simulation/utils.py:compute_renewal_probability`)

P(renew) = sigmoid(2.0 × satisfaction - 2.5 × stress - 1.5 × rent_gap + 0.04 × tenure - 1.5 + N(0,1))

### Validation (`simulation/validation.py`)

Checks: all required tables exist, no orphan references, lease_start < lease_end, positive base_rent, non-negative payments, no duplicate keys.

---

## 5. Feature Engineering (Phase 2)

### FeaturePipeline (`features/pipeline.py`)

Builds point-in-time-correct features for each as-of month. The cutoff boundary is sacred — using data after the prediction date is target leakage.

**Payment Features** (windows: 30, 60, 90 days):
- payment_count, late_payment_count, days_late_avg, days_late_trend, total_late_amount

**Maintenance Features** (windows: 30, 60, 90 days):
- complaint_count, complaint_severity_weighted, complaint_sentiment_avg, avg_resolve_days

**Lease Features:**
- tenure_months, lease_remaining_months, rent_gap_pct, current_rent

**Market Features:**
- market_rent_gap (vs neighborhood average), avg_market_rent

**Intervention Features** (windows: 180, 365 days):
- prior_interventions, prior_intervention_accept_rate

**Tenant Static Features:**
- household_size, is_voucher_holder, household_size_large, zip_code

### NLP Tagging (`features/nlp.py`)

Maintenance request text is tagged using:
1. **spaCy** (if available) for basic NLP
2. **VADER sentiment** for sentiment analysis (compound score, -1 to 1)
3. **Keyword fallback** with severity levels: critical (mold, no heat, flooding), moderate (leak, broken, clogged), minor (cosmetic, squeak)

---

## 6. Person-Period Reshape (Phase 3)

### `reshape/person_period.py`

**explode_to_person_period:** Converts lease records to one row per (tenant, lease-month):
- Churn event flagged only on the final month if `did_renew=False`
- Censored: true if as_of_date < lease_end
- Validates: no double churn events, no churn on censored rows

**tenant_level_split:**
- Leak-proof: splits by tenant_id (all months of one tenant go to the same fold)
- Default: 70% train, 15% val, 15% test

---

## 7. Baseline Model (Phase 4)

### `models/baseline.py`

**BaselineHazardModel** — LogisticRegression with class weighting:
- Feature selection: automatic (excludes id columns, keeps numeric features)
- Scale_pos_weight: (1 - event_rate) / event_rate
- PR-AUC on training data
- Coefficient sign checks against domain expectations

**hazard_to_survival:**
- S(t) = Π(1 - h_k) for k <= t
- Validates hazards in [0,1]
- Checks monotonicity (survival never increases)
- Returns survival_prob and cum_churn_prob

**Expected coefficient signs:**
| Feature | Expected Sign |
|---------|---------------|
| days_late_trend_60d | + (increases risk) |
| complaint_severity_weighted_30d | + |
| tenure_months | - (decreases risk) |
| rent_gap_pct | + |
| market_rent_gap | + |

---

## 8. Primary Hazard Model (Phase 5)

### `models/hazard.py`

**LightGBM hazard model with GroupKFold cross-validation:**

| Hyperparameter | Value |
|----------------|-------|
| objective | binary |
| metric | average_precision |
| scale_pos_weight | computed from event rate |
| learning_rate | 0.03 |
| num_leaves | 31 |
| min_data_in_leaf | 20 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| bagging_freq | 5 |
| num_boost_round | 2000 |
| early_stopping | 50 rounds |

**Isotonic Calibration:** Applied after training to align predicted probabilities with observed frequencies. The calibrator is stored as `model._calibrator`.

**Fairness Gate (pre-training):** Blocks proxy features at training time:
- `zip_code`, `voucher_holder_status`, `is_voucher_holder` — raises `FairnessGateFailure`

**ModelArtifact dataclass:**
- `model`: lgb.Booster
- `feature_names`: list of feature names
- `train_event_rate`: event rate
- `pr_auc`: OOF PR-AUC
- `calibration_applied`: bool

---

## 9. Explainability (Phase 6)

### SHAP Explainer (`explain/shap_explainer.py`)

**ShapExplainer** class:
- Detects model type (tree vs. linear)
- Tree models → `shap.TreeExplainer`
- Linear models → `shap.LinearExplainer`
- Sample-based background (500 samples)
- Returns SHAP values for any feature DataFrame

**DRIVER_LABELS** — maps 40+ feature names to human-readable labels:
- `days_late_trend_60d` → "recent late payments"
- `complaint_severity_weighted_30d` → "unresolved maintenance issues"
- `tenure_months` → "tenure discount"
- `rent_gap_pct` → "rent increase burden"
- `is_voucher_holder` → "voucher holder status"

**top_shap_drivers:** Returns top-k features with absolute SHAP value, direction (increases/decreases risk), raw_value, and human-readable label.

### Narrative (`explain/narrative.py`)

**generate_narrative:**
- Tier 1 (LLM available): Sends SHAP drivers to Claude via API with grounded prompt
- Tier 2 (LLM unavailable/fails): Template fallback:
  > "{tenant_name}'s churn risk is {risk_pct}%. The biggest factor is {top_driver_label} ({top_driver_direction}). Consider addressing this first."
- **Grounding check:** Verifies the generated narrative mentions a risk percentage within 1 point of the actual value — catches LLM hallucinations

---

## 10. Causal Uplift (Phase 8)

### `models/uplift.py`

**T-Learner CATE (TLearnerCATE):**
- Fits one model per treatment arm vs. control
- Default: `LogisticRegression(max_iter=2000)` per arm
- `predict_cate()` returns per-tenant CATE estimates for each arm

**fit_cate_models:**
- `check_positivity()`: validates each arm has ≥30 examples
- GBM-based T-Learner: `GradientBoostingClassifier(max_depth=3, n_estimators=200)`
- CausalForestDML fallback via econml (optional)

**validate_cate_recovers_segments:**
- Merges CATE estimates with hidden segment ground truth
- Two-sample Welch's t-test: maintenance_sensitive vs. price_sensitive
- Returns p-value and binary claim: `recovers_segmentation = (p < 0.01 AND maint_cate_mean > price_cate_mean)`

**save_cate_models / load_cate_models:** Pickle CATE models to `cate/` directory.

---

## 11. Fairness Audit (Phase 13)

### `fairness/audit.py`

**run_fairness_audit:**
- Sensitive feature: voucher_holder status
- Metrics: demographic parity difference, equalized odds difference
- Uses `fairlearn` if available, otherwise simplified calculation
- Validation: requires ≥2 groups, no NaNs
- Returns `FairnessReport` with `passed_gate` flag (DP diff ≤ 0.10 threshold)

**fairness_gate_for_promotion:**
- Raises `FairnessGateFailure` if audit fails — structural blocker, not a flag

**demonstrate_bias_correction:**
- Compares `before` (legacy biased flag) vs. `after` (model predictions)
- Returns dp_improvement and `fairness_fixed` (improvement > 0 AND after passes gate)

---

## 12. MLOps & Champion/Challenger (Phase 15)

### Drift Monitoring (`mlops/drift.py`)

**population_stability_index (PSI):**
- PSI < 0.1: no significant shift
- 0.1-0.25: moderate, watch it
- > 0.25: significant — investigate before trusting predictions

**DriftMonitor:**
- Stores drift history
- `check()`: compares new features against reference, logs warnings with top-5 breached features
- `generate_alert()`: returns human-readable alert string

### Champion/Challenger (`mlops/champion.py`)

**evaluate_challenger:**
- A challenger model must:
  1. Pass the fairness gate (demographic parity ≤ 0.10)
  2. Exceed champion PR-AUC by at least `min_relative_improvement` (2%)
- Returns decision dict with reason

**promote_to_champion:**
- If MLflow available, transitions model to Production

---

## 13. Budget Optimization (Phase 16)

### `optimize/ilp_optimizer.py`

**Integer Linear Program (PuLP CBC):**
```
Maximize: Σ CATE_i × LTV_i × x_i
Subject to:
  Σ cost_dollars_i × x_i ≤ monthly_budget
  Σ crew_hours_i × x_i ≤ monthly_crew_hours
  Σ x_i ≤ 1 per tenant (at most one intervention per tenant)
  x_i ∈ {0, 1} (binary decision variables)
```

**solve_retention_allocation_greedy:** Falls back if solver is infeasible or times out — provably suboptimal but always returns something.

**LTV** is approximated as `6 × household_size` months of retained rent.

---

## 14. Evaluation (Phase 17)

### `mlops/evaluation.py`

**compute_evaluation_metrics:**
| Metric | Description |
|--------|-------------|
| PR-AUC | Precision-recall AUC (primary metric) |
| ROC-AUC | Standard ROC AUC |
| Brier Score | Mean squared error of predictions |
| F2 Score | Recall-weighted F-beta |
| Max F1 | Maximum F1 across thresholds |
| Calibration Error | Mean absolute calibration curve deviation |
| C-index | Concordance index (if lifelines available) |

**generate_evaluation_report:** Runs on test fold, saves JSON report to `artifacts/evaluation_report.json`.

**compute_expected_retained_revenue:** Estimates lift in value from optimized allocation vs. baseline (no intervention or random).

---

## 14. REST API

### `api/main.py`

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/` | GET | — | Redirects to `/docs` |
| `/health` | GET | — | `HealthResponse` (model status, tenant count) |
| `/score/{tenant_id}` | GET | tenant_id | `ScoreResponse` (risk_pct, risk_band, survival_prob, top_drivers, narrative) |
| `/simulate` | POST | `WhatIfRequest` | `WhatIfResponse` (baseline vs scenario risk, delta, recommendation) |
| `/optimize` | POST | `OptimizeRequest` | `OptimizeResponse` (selections, budget spent, solver status) |

### API Schemas (`api/schemas.py`)

```python
class WhatIfRequest:
    tenant_id: str
    rent_increase_pct: float (0-50%)
    maintenance_speed: Literal["standard", "priority", "same_day"]
    retention_credit_usd: float (0-2000)

class OptimizeRequest:
    monthly_budget: float (>0, default 5000)
    monthly_crew_hours: float (>0, default 80)

class ScoreResponse:
    tenant_id, risk_pct, risk_band (Very Low/Low/Moderate/High/Critical)
    survival_prob, top_drivers, narrative, narrative_source

class WhatIfResponse:
    baseline_risk_pct, scenario_risk_pct, delta_pts
    recommendation, narrative

class OptimizeResponse:
    selections, total_selected, total_budget_spent, solver_status
```

### Error Handling
- `@app.exception_handler(Exception)`: Catches all unhandled exceptions, returns 500 with request_id for traceability
- `x-request-id` middleware: Attaches a UUID to every response for debugging
- CORS: `allow_origins=["*"]` for development

---

## 15. Streamlit Dashboard

### `frontend/dashboard.py`

**Two-tab layout:**

**Tab 1: Portfolio Heatmap**
- Bar chart: Top 50 at-risk tenants with color-coded risk bands
- Pie chart: Risk distribution across bands
- Data table: All tenants with ProgressColumn for risk %

**Tab 2: Tenant Detail**
- Survival/churn curve plot (green/red lines)
- SHAP driver horizontal bar chart (red = increases risk, green = decreases)
- Narrative text (LLM or template fallback)
- Tenant profile JSON in expandable drawer

**Sidebar metrics:** Total tenants, High Risk (≥60%), Critical (≥80%)

### What-If Simulator Logic (backend)

The `/simulate` endpoint computes scenario risk by adjusting feature values:
- **Maintenance speed:** priority = -15 pts, same_day = -30 pts
- **Retention credit:** -0.008 pts per $10
- **Rent increase:** +0.04 pts per % increase
- Clamped to [0, 100]

If CATE models are loaded, uses actual uplift predictions for recommendation. Otherwise, rule-based recommendation based on delta.

---

## 16. React Frontend — What-If Simulator

### `public/index.html`

Single-page React application (standalone via Babel) with:
- **Tenant selector:** Dropdown of up to 100 tenants from API
- **Persona presets:** Balanced, Price-Sensitive, Maintenance-Sensitive, At-Risk
- **Scenario controls:** Rent increase slider (0-15%), Retention credit slider (0-500), Maintenance speed selector
- **Debounced simulation:** 300ms debounce after last slider change
- **Results:** Baseline vs. scenario risk comparison cards, delta display, AI recommendation

### `src/styles.css`

Dark theme with:
- Background: `#0f1923`
- Accent: teal (`#64ffda`) and blue (`#48b0ff`)
- Cards: `#1a2736`
- Skeleton loading animation
- Risk gauges: Critical red, High orange, Moderate yellow, Low/very low green

---

## 17. Docker Deployment

### `docker/Dockerfile.api`
Multi-stage build with Python 3.11-slim, copies artifacts/data/.env, exposes 8000, healthcheck enabled.

### `docker/Dockerfile.frontend`
Nginx alpine serving static files from `/usr/share/nginx/html/`, copies nginx config.

### `docker/docker-compose.yml`
```yaml
services:
  api:       # port 8000, volume mounts for artifacts and data
  frontend:  # port 3000:80, depends on api healthcheck
```

### `docker/nginx.conf`
Reverse proxies `/api/` requests to `http://api:8000/`, serves static index.html with SPA fallback.

---

## 18. Configuration

### Environment Variables (`.env` / `.env.example`)

```
# Data Generation
FORGEX__DATA__N_TENANTS=10000
FORGEX__DATA__N_PROPERTIES=120
FORGEX__DATA__START_DATE=2019-01-01
FORGEX__DATA__END_DATE=2024-12-31
FORGEX__DATA__SEED=42

# Fairness
FORGEX__FAIRNESS__MAX_DP_DIFF=0.10

# Optimizer
FORGEX__OPTIMIZER__MONTHLY_BUDGET=5000.0
FORGEX__OPTIMIZER__MONTHLY_CREW_HOURS=80.0
FORGEX__OPTIMIZER__SOLVER_TIME_LIMIT_S=30

# Drift
FORGEX__DRIFT__PSI_THRESHOLD=0.25

# MLOps
FORGEX__MLOPS__MIN_RELATIVE_IMPROVEMENT=0.02

# Narrative (Tier 3)
FORGEX__NARRATIVE__MAX_RETRIES=2
FORGEX__NARRATIVE__TIMEOUT_S=6.0

# LLM (optional)
ANTHROPIC_API_KEY=

# Paths
FORGEX__ARTIFACTS_DIR=artifacts
FORGEX__DATA_DIR=data
FORGEX__LOG_LEVEL=INFO
```

### `config.py` (Pydantic Settings)

Settings are loaded via `pydantic-settings` with `env_nested_delimiter="__"` and `env_prefix="FORGEX__"`. Nested configs: `DataConfig`, `FairnessConfig`, `OptimizerConfig`, `DriftConfig`, `MlopsConfig`, `NarrativeConfig`.

Directories are auto-created at startup (`_ensure_dirs` model_validator).

---

## 19. Testing

### Running tests:
```bash
cd C:\Users\LOUJAN B\ForgeX\forgex
pytest
# Or: python -m forgex test
```

### Test files (18 files):

| Test File | What It Tests |
|-----------|---------------|
| `test_api_schemas.py` | Pydantic validation: valid requests, invalid ranges, default values |
| `test_drift.py` | PSI identity, shift detection, empty/NaN handling, DriftMonitor |
| `test_errors.py` | Exception hierarchy: all custom errors inherit from ForgeXError |
| `test_fairness.py` | Audit with single group, NaNs, fair/unfair scenarios, bias correction demo |
| `test_models.py` | Survival curve math, hazard bounds, positivity checks, segment recovery |
| `test_nlp.py` | Text tagging: critical/moderate/minor, None/empty/whitespace input, long text |
| `test_reshape.py` | Person-period explode, churn/censor logic, validation, tenant split |
| `test_simulation_utils.py` | Sigmoid stability, renewal probability, segment assignment |

### Test markers:
- `slow`: Deselect with `-m "not slow"`
- `integration`: Tests that need artifacts

---

## 20. How to Run Everything

### Terminal 1 — API (port 8000):
```
C:\Users\LOUJAN B\ForgeX\forgex> python -m forgex api
```

### Terminal 2 — Dashboard (port 8501):
```
C:\Users\LOUJAN B\ForgeX\forgex> python -m forgex dashboard
```

### Terminal 3 — Frontend (port 3000):
```
C:\Users\LOUJAN B\ForgeX\forgex\frontend> npx serve public -p 3000
```

### URLs:
| Service | URL |
|---------|-----|
| API (Swagger UI) | http://localhost:8000/docs |
| API (Health) | http://localhost:8000/health |
| API (Score) | http://localhost:8000/score/T000000 |
| Dashboard | http://localhost:8501 |
| What-If Simulator | http://localhost:3000 |

### Docker:
```bash
cd C:\Users\LOUJAN B\ForgeX\forgex\docker
docker compose up
```

---

## 21. Important Notes for Cloning

### Files NOT in Git (need manual copy from original machine):

| Item | Path | Reason |
|------|------|--------|
| `.env` | `forgex/.env` | Contains configuration (safe defaults in `.env.example`) |
| Raw data | `forgex/data/raw/*.parquet` | Excluded in `.gitignore` |
| Processed data | `forgex/data/processed/*.parquet` | Excluded in `.gitignore` |
| Artifacts | `forgex/artifacts/*.pkl/.parquet/.json` | Excluded in `.gitignore` |
| MLflow runs | `forgex/mlflow/` | Excluded in `.gitignore` |
| Node modules | `forgex/frontend/node_modules/` | Excluded in `.gitignore` |

### Setup steps on a fresh clone:

1. **Create `.env`:**
   ```bash
   copy forgex\.env.example forgex\.env
   ```

2. **Copy data & artifacts** from original machine to:
   - `forgex/data/raw/` (10 parquet files)
   - `forgex/data/processed/` (4 parquet + split_ids.json)
   - `forgex/artifacts/` (hazard_model.pkl, shap_explainer.pkl, evaluation_report.json, survival_curves.parquet)

3. **Install Python** (3.11+ required):
   ```bash
   cd forgex
   pip install -e ".[full]"
   ```
   Or with compatible versions:
   ```bash
   pip install "starlette==0.37.2" "streamlit==1.38.0" "fastapi>=0.115.0,<0.116.0"
   ```

4. **Install frontend:**
   ```bash
   cd forgex\frontend
   npm install
   ```

### Regenerating data from scratch:
```bash
cd forgex
python -m forgex generate      # Generate synthetic data
python -m forgex pipeline       # Run full training pipeline
```

---

## Appendix: Package Dependencies

### Core (pyproject.toml):
pandas, numpy, scipy, scikit-learn, lightgbm, shap, pulp, fastapi, uvicorn, pydantic, pydantic-settings, faker, python-dotenv, streamlit, plotly, pytest, pytest-cov, httpx, orjson, tqdm, lifelines, scikit-survival

### Optional extras:
| Extra | Packages |
|-------|----------|
| `nlp` | spacy, vaderSentiment |
| `fairness` | fairlearn |
| `mlops` | mlflow |
| `llm` | anthropic |
| `tft` | pytorch-lightning, pytorch-forecasting |
| `full` | All above (except TFT) |