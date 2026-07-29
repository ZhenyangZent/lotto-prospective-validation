"""Canonical JSON、SHA-256 與 append-only hash-chain 帳本。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

ZERO_HASH = "0" * 64


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_record(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes({key: value for key, value in record.items() if key != "record_hash"}))


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"ledger 第 {line_number} 行為空白")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ledger 第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"ledger 第 {line_number} 行不是 JSON object")
        if canonical_bytes(value).decode("utf-8") != line:
            raise ValueError(f"ledger 第 {line_number} 行不是 canonical JSON")
        records.append(value)
    return records


def verify_ledger(path: str | Path, head_path: str | Path | None = None) -> dict[str, Any]:
    records = read_ledger(path)
    previous = ZERO_HASH
    seen_events: set[str] = set()
    for index, record in enumerate(records):
        if record.get("sequence") != index + 1:
            raise ValueError(f"ledger sequence 在第 {index + 1} 筆中斷")
        if record.get("previous_record_hash") != previous:
            raise ValueError(f"ledger previous_record_hash 在第 {index + 1} 筆中斷")
        expected = hash_record(record)
        if record.get("record_hash") != expected:
            raise ValueError(f"ledger record_hash 在第 {index + 1} 筆不符")
        event_id = str(record.get("event_id", ""))
        if not event_id or event_id in seen_events:
            raise ValueError(f"ledger event_id 在第 {index + 1} 筆缺失或重複")
        seen_events.add(event_id)
        previous = expected
    if head_path is not None:
        anchor_file = Path(head_path)
        if not anchor_file.exists():
            raise ValueError("ledger head anchor 不存在")
        anchor = json.loads(anchor_file.read_text(encoding="utf-8"))
        expected_anchor = {"record_count": len(records), "record_hash": previous}
        if anchor != expected_anchor:
            raise ValueError("ledger 與 head anchor 不一致，可能遭尾端刪除或替換")
    return {"valid": True, "record_count": len(records), "record_hash": previous}


def initialize_ledger(path: str | Path, head_path: str | Path) -> None:
    ledger = Path(path); head = Path(head_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if not ledger.exists():
        ledger.write_bytes(b"")
    if not head.exists():
        head.write_text(json.dumps({"record_count": 0, "record_hash": ZERO_HASH}, sort_keys=True), encoding="utf-8")
    verify_ledger(ledger, head)


def append_record(path: str | Path, head_path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    status = verify_ledger(path, head_path)
    complete = dict(record)
    complete["sequence"] = status["record_count"] + 1
    complete["previous_record_hash"] = status["record_hash"]
    complete.pop("record_hash", None)
    complete["record_hash"] = hash_record(complete)
    encoded = canonical_bytes(complete) + b"\n"
    ledger = Path(path)
    with ledger.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    anchor = {"record_count": complete["sequence"], "record_hash": complete["record_hash"]}
    temporary = Path(head_path).with_suffix(".tmp")
    temporary.write_text(json.dumps(anchor, sort_keys=True), encoding="utf-8")
    temporary.replace(head_path)
    verify_ledger(path, head_path)
    return complete


def aggregate_file_hash(root: str | Path, relative_paths: Iterable[str]) -> tuple[str, dict[str, str]]:
    base = Path(root)
    files = {str(relative).replace("\\", "/"): sha256_file(base / relative) for relative in sorted(relative_paths)}
    return sha256_bytes(canonical_bytes(files)), files
