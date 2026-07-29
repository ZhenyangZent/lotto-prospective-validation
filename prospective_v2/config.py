"""V2 證據設計設定；模型方法與 V1 完全相同。"""
from __future__ import annotations

from pathlib import Path

from prospective.config import FROZEN_CONFIG as V1_FROZEN_CONFIG

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "prospective_validation_v2"
DATA_PATH = ROOT / "data" / "processed" / "lotto649.csv"
LEDGER_PATH = STATE_DIR / "ledger.jsonl"
LEDGER_HEAD_PATH = STATE_DIR / "ledger_head.json"
FROZEN_CONFIG_PATH = STATE_DIR / "frozen_config.json"
FROZEN_MANIFEST_PATH = STATE_DIR / "frozen_manifest.json"

EXPERIMENT_ID = "TW649-PROSPECTIVE-20260729-V2"
EXPERIMENT_VERSION = "2.0.0"
MODEL_VERSION = "ProspectiveBatchLogistic-v1"
TIMEZONE = "Asia/Taipei"
TAG_NAME = "lotto-prospective-v2-preregistered"
REMOTE_NAME = "origin"

FROZEN_CONFIG = {
    **V1_FROZEN_CONFIG,
    "experiment_id": EXPERIMENT_ID,
    "experiment_version": EXPERIMENT_VERSION,
    "model_version": MODEL_VERSION,
    "evidence_protocol": {
        "version": "remote-time-anchor-v2",
        "prediction_event_remote_status": "PENDING_REMOTE_ANCHOR",
        "prediction_commit_field": "parent_commit",
        "remote_verification": "git-ls-remote",
        "confirmation_requires_remote_oid_match": True,
        "confirmation_requires_pre_draw_official_status": True,
        "status_resolved_from_ledger_anchor_only": True,
        "cross_platform_probability_atol": 1e-12,
    },
}

SOURCE_FILES = [
    "next_draw_predictor_audited.py", "prospective/model.py", "prospective/canonical.py",
    *[f"prospective_v2/{name}" for name in (
        "__init__.py", "config.py", "official.py", "remote.py", "workflow.py",
        "freeze_experiment.py", "predict_next.py", "verify_ledger.py", "status.py",
        "export_review_bundle.py",
    )],
]
