"""Shared verification helpers for V2 result operations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from prospective.canonical import read_ledger, sha256_file, verify_ledger
from prospective_v2.remote import resolve_remote_anchor


class EvidenceError(RuntimeError):
    """Raised when an append-only evidence prerequisite is not satisfied."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def unique_prediction(records: list[dict[str, Any]], draw_id: str) -> dict[str, Any]:
    matches = [
        record for record in records
        if record.get("event_type") == "prediction"
        and str(record.get("target_draw_id")) == str(draw_id)
    ]
    if len(matches) != 1:
        raise EvidenceError(f"expected exactly one prediction for draw {draw_id}; found {len(matches)}")
    return matches[0]


def require_prediction_anchor(
    prediction: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    root: str | Path,
    revalidate_remote: bool = True,
) -> dict[str, Any]:
    anchor = resolve_remote_anchor(
        str(prediction["prediction_id"]), records, root=root,
        revalidate_remote=revalidate_remote,
    )
    if anchor is None:
        raise EvidenceError("prediction has no valid remotely revalidated anchor")
    return anchor


def repository_fingerprint(root: str | Path, paths: list[str | Path]) -> dict[str, str | None]:
    base = Path(root)
    result: dict[str, str | None] = {}
    for value in paths:
        relative = Path(value)
        path = relative if relative.is_absolute() else base / relative
        result[relative.as_posix()] = sha256_file(path) if path.is_file() else None
    return result


def verify_state(ledger: str | Path, head: str | Path) -> dict[str, Any]:
    state = verify_ledger(ledger, head)
    state["records"] = len(read_ledger(ledger))
    return state
