"""Derive V2 completion status from linked anchor evidence, never booleans."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prospective_v2.remote import resolve_remote_anchor

from .result_remote_anchor import resolve_result_anchor


def status_summary(
    records: list[dict[str, Any]], *, root: str | Path, revalidate_remote: bool = True,
    expected_repository: str | None = None, expected_branch: str | None = None,
    ledger_relative_path: str = "prospective_validation_v2/ledger.jsonl",
    head_relative_path: str = "prospective_validation_v2/ledger_head.json",
) -> dict[str, Any]:
    predictions = [record for record in records if record.get("event_type") == "prediction"]
    results = [record for record in records if record.get("event_type") in {"result", "correction"}]
    prediction_anchors = {
        prediction["prediction_id"]: resolve_remote_anchor(
            prediction["prediction_id"], records, root=root, revalidate_remote=revalidate_remote,
        )
        for prediction in predictions
    }
    valid_result_anchors = {
        result["record_hash"]: resolve_result_anchor(
            result, records, root=root, revalidate_remote=revalidate_remote,
            expected_repository=expected_repository, expected_branch=expected_branch,
            ledger_relative_path=ledger_relative_path, head_relative_path=head_relative_path,
        )
        for result in results
    }
    effective: dict[str, dict[str, Any]] = {}
    for result in results:
        if valid_result_anchors[result["record_hash"]] is not None:
            effective[str(result["target_draw_id"])] = result
    completed = [
        result for result in effective.values()
        if prediction_anchors.get(result["prediction_id"]) is not None
    ]
    completed_ids = {str(result["target_draw_id"]) for result in completed}
    pending_predictions = [
        str(prediction["target_draw_id"]) for prediction in predictions
        if str(prediction["target_draw_id"]) not in completed_ids
        and not any(str(result["target_draw_id"]) == str(prediction["target_draw_id"]) for result in results)
    ]
    unanchored_results = [
        str(result["target_draw_id"]) for result in results
        if valid_result_anchors[result["record_hash"]] is None
    ]
    count = len(completed)
    mean = lambda field: None if not completed else sum(float(item[field]) for item in completed) / count
    next_point = 100 if count < 100 else 200 if count < 200 else None
    return {
        "prediction_events": len(predictions),
        "valid_prediction_anchors": sum(anchor is not None for anchor in prediction_anchors.values()),
        "result_events": len(results),
        "valid_result_anchors": sum(anchor is not None for anchor in valid_result_anchors.values()),
        "valid_completed_draws": count,
        "pending_prediction_draw_ids": pending_predictions,
        "unanchored_result_draw_ids": unanchored_results,
        "mean_hits_top6": mean("hits_top6"),
        "mean_brier_difference": mean("brier_difference"),
        "mean_log_loss_difference": mean("log_loss_difference"),
        "next_formal_analysis_point": next_point,
        "formal_interim_conclusion": "PROHIBITED_BEFORE_100_VALID_COMPLETED_DRAWS",
    }
