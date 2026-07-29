"""Validate and append official result/correction events to a V2 ledger."""
from __future__ import annotations

import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from prospective.canonical import append_record, read_ledger
from prospective.metrics import score_prediction
from prospective.official import validate_official_source
from prospective.official import normalize_result
from prospective_v2.config import EXPERIMENT_ID, EXPERIMENT_VERSION, TIMEZONE

from .verify import EvidenceError, require_prediction_anchor, sha256_bytes, unique_prediction


def now_iso() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")


def validate_result_payload(result: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "ANNOUNCED":
        raise EvidenceError("official result has not been announced")
    source = str(result.get("source", ""))
    validate_official_source(source)
    if str(result.get("draw_id")) != str(prediction.get("target_draw_id")):
        raise EvidenceError("official result draw id does not match prediction")
    if str(result.get("draw_date")) != str(prediction.get("target_draw_date")):
        raise EvidenceError("official result date does not match prediction")
    raw_numbers = result.get("numbers", [])
    if not isinstance(raw_numbers, list) or any(type(value) is not int for value in raw_numbers):
        raise EvidenceError("ordinary numbers must be JSON integers")
    numbers = list(raw_numbers)
    if len(numbers) != 6 or any(value < 1 or value > 49 for value in numbers):
        raise EvidenceError("ordinary numbers must contain six values in 1..49")
    if len(set(numbers)) != 6:
        raise EvidenceError("ordinary numbers must be unique")
    special_raw = result.get("special_number", 0)
    if type(special_raw) is not int:
        raise EvidenceError("special number must be a JSON integer")
    special = special_raw
    if special < 1 or special > 49:
        raise EvidenceError("special number must be in 1..49")
    if special in numbers:
        raise EvidenceError("special number must differ from ordinary numbers")
    return {**result, "numbers": sorted(numbers), "special_number": special, "source": source}


def validate_raw_response(raw_response: bytes, result: dict[str, Any]) -> None:
    """Bind the archived HTTP body to every normalized draw field used by the event."""
    try:
        payload = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("official raw response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or payload.get("rtCode") != 0:
        raise EvidenceError("official raw response has a non-success rtCode")
    rows = payload.get("content", {}).get("lotto649Res", [])
    exact = [row for row in rows if str(row.get("period")) == str(result["draw_id"])]
    if len(exact) != 1:
        raise EvidenceError("official raw response does not contain exactly one requested draw")
    normalized = normalize_result(exact[0])
    expected = {
        "draw_id": str(result["draw_id"]), "draw_date": str(result["draw_date"]),
        "numbers": sorted(int(value) for value in result["numbers"]),
        "special_number": int(result["special_number"]),
    }
    actual = {
        "draw_id": str(normalized.get("draw_id")), "draw_date": str(normalized.get("draw_date")),
        "numbers": sorted(int(value) for value in normalized.get("numbers", [])),
        "special_number": int(normalized.get("special_number", 0)),
    }
    if actual != expected:
        raise EvidenceError("normalized result does not match archived official raw response")


def _prior_results(records: list[dict[str, Any]], draw_id: str) -> list[dict[str, Any]]:
    return [
        record for record in records
        if record.get("event_type") in {"result", "correction"}
        and str(record.get("target_draw_id")) == str(draw_id)
    ]


def build_result_event(
    records: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    raw_response_path: str | Path,
    raw_response: bytes,
    root: str | Path,
    revalidate_remote: bool = True,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    draw_id = str(result.get("draw_id", ""))
    prediction = unique_prediction(records, draw_id)
    anchor = require_prediction_anchor(
        prediction, records, root=root, revalidate_remote=revalidate_remote,
    )
    checked = validate_result_payload(result, prediction)
    validate_raw_response(raw_response, checked)
    prior = _prior_results(records, draw_id)
    if prior and sorted(prior[-1]["actual_numbers"]) == checked["numbers"] and int(prior[-1]["special_number"]) == checked["special_number"]:
        raise EvidenceError("identical official result already exists")
    metrics = score_prediction(prediction["probabilities_1_to_49"], checked["numbers"])
    event_type = "correction" if prior else "result"
    raw_relative = _repository_relative_path(root, raw_response_path)
    return {
        "event_type": event_type,
        "event_id": str(uuid.uuid4()),
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "target_draw_id": draw_id,
        "target_draw_date": checked["draw_date"],
        "prediction_id": prediction["prediction_id"],
        "prediction_record_hash": prediction["record_hash"],
        "prediction_anchor_record_hash": anchor["record_hash"],
        "actual_numbers": checked["numbers"],
        "special_number": checked["special_number"],
        "official_result_status": "CORRECTED" if prior else "OFFICIAL",
        "result_source": checked["source"],
        "result_retrieved_at": retrieved_at or now_iso(),
        "official_raw_response_path": raw_relative,
        "official_raw_response_sha256": sha256_bytes(raw_response),
        **metrics,
        "brier_difference": metrics["brier"] - metrics["uniform_brier"],
        "log_loss_difference": metrics["log_loss"] - metrics["uniform_log_loss"],
    }


def ingest_result(
    ledger_path: str | Path,
    head_path: str | Path,
    result: dict[str, Any],
    *,
    raw_response_path: str | Path,
    raw_response: bytes,
    result_json_path: str | Path,
    root: str | Path,
    revalidate_remote: bool = True,
) -> dict[str, Any]:
    ledger = Path(ledger_path); head = Path(head_path)
    raw_path = Path(raw_response_path)
    json_path = Path(result_json_path)
    lock_path = ledger.with_name(ledger.name + ".prospective-ops.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise EvidenceError("another V2 ledger operation holds the exclusive lock") from exc
    raw_tmp: Path | None = None; json_tmp: Path | None = None
    raw_published = False; json_published = False
    ledger_before: bytes | None = None; head_before: bytes | None = None
    appended_bytes: bytes | None = None
    appended_head: bytes | None = None
    try:
        # Everything that can become stale belongs inside the exclusive section:
        # ledger read, duplicate/correction classification, artifact existence,
        # rollback snapshot, append, and artifact publication.
        records = read_ledger(ledger)
        event = build_result_event(
            records, result, raw_response_path=raw_response_path,
            raw_response=raw_response, root=root,
            revalidate_remote=revalidate_remote,
        )
        if raw_path.exists() or json_path.exists():
            raise FileExistsError("result artifacts are append-only and must not be overwritten")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_before = ledger.read_bytes(); head_before = head.read_bytes()
        token = uuid.uuid4().hex
        raw_tmp = raw_path.with_name(raw_path.name + f".{token}.tmp")
        json_tmp = json_path.with_name(json_path.name + f".{token}.tmp")
        raw_tmp.write_bytes(raw_response)
        record = append_record(ledger, head, event)
        appended_bytes = ledger.read_bytes(); appended_head = head.read_bytes()
        json_tmp.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        raw_tmp.replace(raw_path)
        raw_published = True
        json_tmp.replace(json_path)
        json_published = True
        return record
    except Exception:
        concurrent = appended_bytes is not None and not (ledger.read_bytes() == appended_bytes and head.read_bytes() == appended_head)
        if not concurrent and ledger_before is not None and head_before is not None:
            ledger.write_bytes(ledger_before); head.write_bytes(head_before)
        cleanup = [raw_tmp, json_tmp]
        if raw_published: cleanup.append(raw_path)
        if json_published: cleanup.append(json_path)
        for artifact in cleanup:
            if artifact is not None and artifact.exists():
                artifact.unlink()
        if concurrent:
            raise EvidenceError("concurrent ledger change detected; refusing destructive rollback")
        raise
    finally:
        os.close(lock_fd)
        if lock_path.exists(): lock_path.unlink()


def _repository_relative_path(root: str | Path, value: str | Path) -> str:
    base = Path(root).resolve(); supplied = Path(value)
    candidate = supplied.resolve() if supplied.is_absolute() else (base / supplied).resolve()
    try:
        relative = candidate.relative_to(base)
    except ValueError as exc:
        raise EvidenceError("official raw response path must stay inside the repository") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise EvidenceError("official raw response path must be a canonical repository-relative path")
    return relative.as_posix()
