"""不可事後調整的前瞻實驗設定。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "prospective_validation"
DATA_PATH = ROOT / "data" / "processed" / "lotto649.csv"
LEDGER_PATH = STATE_DIR / "ledger.jsonl"
LEDGER_HEAD_PATH = STATE_DIR / "ledger_head.json"
FROZEN_CONFIG_PATH = STATE_DIR / "frozen_config.json"
FROZEN_MANIFEST_PATH = STATE_DIR / "frozen_manifest.json"

EXPERIMENT_ID = "TW649-PROSPECTIVE-20260729-V1"
EXPERIMENT_VERSION = "1.0.0"
MODEL_VERSION = "ProspectiveBatchLogistic-v1"
TIMEZONE = "Asia/Taipei"
RANDOM_SEED = 20260729
UNIFORM_PROBABILITY = 6.0 / 49.0

FROZEN_CONFIG = {
    "experiment_id": EXPERIMENT_ID,
    "experiment_version": EXPERIMENT_VERSION,
    "model_version": MODEL_VERSION,
    "model": {
        "kind": "batch_expanding_logistic_regression",
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 500,
        "random_state": RANDOM_SEED,
        "numeric_scaler": "StandardScaler-fit-on-training-only",
        "number_encoder": "OneHotEncoder-fit-on-training-only",
        "features": [
            "long_z", "recent20_z", "recent50_z", "recent100_z",
            "ewma_z", "gap_z", "transition_z", "in_last_draw",
        ],
        "number_one_hot": True,
        "minimum_history": 30,
        "ewma_alpha": 0.06,
        "transition_alpha": 30.0,
        "long_frequency_prior_strength": 12.0,
        "shrink_strength": 0.10,
        "probability_projection": "capped-simplex-sum-6",
        "tie_break": "probability-descending-number-ascending",
    },
    "baselines": ["ExactUniform", "Recent50Hot", "PreviousDraw"],
    "primary_comparison": "ProspectiveBatchLogistic-v1 vs ExactUniform",
    "stages": [{"stage": 1, "draws": [1, 100]}, {"stage": 2, "draws": [101, 200]}],
    "descriptive_report_interval": 20,
    "confirmatory_analysis_points": [100, 200],
    "primary_metrics": ["brier_difference", "log_loss_difference"],
    "secondary_metrics": ["hits_top6", "hits_top10", "hits_top12"],
    "success": {
        "mean_brier_difference_lt": 0.0,
        "mean_log_loss_difference_lt": 0.0,
        "holm_adjusted_one_sided_p_lt": 0.05,
        "consistent_improvement_20_draw_blocks_at_least": 0.60,
        "no_leakage_or_late_prediction": True,
        "independent_replication_same_direction": True,
    },
    "conditional_monte_carlo": {
        "simulations": 1_000_000,
        "seed": RANDOM_SEED,
        "plus_one_p_value": True,
        "confidence_level": 0.95,
        "fair_draw": "sample-6-without-replacement-from-1-to-49",
    },
    "missing_draw_policy": "Only remotely confirmed pre-draw predictions are valid; missing predictions remain missing and are never backfilled.",
    "official_correction_policy": "Append a correction event; never overwrite an earlier event.",
    "bug_policy": "Close the version, create a new version, and restart prospective counting from zero.",
    "stopping_rule": "Accumulate 100 valid draws in stage 1 and 100 new valid draws in stage 2 regardless of stage-1 outcome.",
}

SOURCE_FILES = ["next_draw_predictor_audited.py", *[f"prospective/{name}" for name in (
    "__init__.py", "canonical.py", "config.py", "gitops.py", "metrics.py",
    "model.py", "official.py", "workflow.py", "freeze_experiment.py",
    "predict_next.py", "ingest_result.py", "verify_ledger.py", "status.py",
    "report.py", "export_review_bundle.py",
)]]
