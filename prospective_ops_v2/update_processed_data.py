"""Append one remotely anchored official draw without rewriting historical CSV bytes."""
from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path
from typing import Any, Callable

from prospective.canonical import sha256_file

from .result_remote_anchor import resolve_result_anchor
from .verify import EvidenceError


REQUIRED_COLUMNS = ["draw_id", "draw_date", *(f"number_{index}" for index in range(1, 7)), "special_number"]


def _decode_csv(raw: bytes) -> tuple[str, str]:
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    return raw.decode(encoding), encoding


def append_anchored_result(
    csv_path: str | Path,
    result: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    root: str | Path,
    revalidate_remote: bool = True,
    validator: Callable[[str | Path], Any] | None = None,
    expected_repository: str | None = None,
    expected_branch: str | None = None,
    ledger_relative_path: str = "prospective_validation_v2/ledger.jsonl",
    head_relative_path: str = "prospective_validation_v2/ledger_head.json",
) -> dict[str, Any]:
    if resolve_result_anchor(
        result, records, root=root, revalidate_remote=revalidate_remote,
        expected_repository=expected_repository, expected_branch=expected_branch,
        ledger_relative_path=ledger_relative_path, head_relative_path=head_relative_path,
    ) is None:
        raise EvidenceError("CSV update requires a valid result remote anchor")
    path = Path(csv_path)
    before = path.read_bytes()
    text, encoding = _decode_csv(before)
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows and not text.strip():
        raise EvidenceError("processed CSV must already contain its frozen schema")
    fieldnames = list(csv.DictReader(io.StringIO(text)).fieldnames or [])
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise EvidenceError(f"processed CSV missing required columns: {missing}")
    _validate_rows(rows)
    draw_id = str(result["target_draw_id"])
    matches = [row for row in rows if str(row["draw_id"]) == draw_id]
    expected_numbers = [str(value) for value in result["actual_numbers"]]
    if matches:
        row = matches[0]
        same = (
            row["draw_date"] == str(result["target_draw_date"])
            and [row[f"number_{index}"] for index in range(1, 7)] == expected_numbers
            and row["special_number"] == str(result["special_number"])
        )
        if same:
            digest = sha256_file(path)
            return {"added": 0, "idempotent": True, "before_sha256": digest, "after_sha256": digest}
        raise EvidenceError("processed CSV contains conflicting values for draw id")

    target_key = (date.fromisoformat(str(result["target_draw_date"])), int(draw_id))
    if rows:
        last = rows[-1]
        last_key = (date.fromisoformat(last["draw_date"]), int(last["draw_id"]))
        if target_key <= last_key:
            raise EvidenceError("processed CSV permits append of the newest draw only")

    row = {name: "" for name in fieldnames}
    row.update({"draw_id": draw_id, "draw_date": str(result["target_draw_date"]),
                "special_number": str(result["special_number"])})
    row.update({f"number_{index}": str(number) for index, number in enumerate(result["actual_numbers"], 1)})
    if "source" in row:
        row["source"] = str(result["result_source"])
    if "source_version" in row:
        row["source_version"] = "prospective-v2-official-result"
    newline = "\r\n" if b"\r\n" in before else "\n"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator=newline)
    writer.writerow(row)
    separator = b"" if before.endswith((b"\n", b"\r")) else newline.encode("ascii")
    addition = output.getvalue().encode("utf-8")
    path.write_bytes(before + separator + addition)
    try:
        new_text, _ = _decode_csv(path.read_bytes())
        _validate_rows(list(csv.DictReader(io.StringIO(new_text))))
        if validator is not None:
            validator(path)
    except Exception:
        path.write_bytes(before)
        raise
    return {
        "added": 1,
        "idempotent": False,
        "before_sha256": __import__("hashlib").sha256(before).hexdigest(),
        "after_sha256": sha256_file(path),
        "historical_prefix_preserved": path.read_bytes().startswith(before),
    }


def _validate_rows(rows: list[dict[str, str]]) -> None:
    seen: set[str] = set(); previous: tuple[date, int] | None = None
    for row in rows:
        draw_id = str(row["draw_id"])
        if draw_id in seen:
            raise EvidenceError("processed CSV contains duplicate draw ids")
        seen.add(draw_id)
        try:
            key = (date.fromisoformat(row["draw_date"]), int(draw_id))
            numbers = [int(row[f"number_{index}"]) for index in range(1, 7)]
            special = int(row["special_number"])
        except (ValueError, TypeError) as exc:
            raise EvidenceError("processed CSV contains malformed draw data") from exc
        if previous is not None and key <= previous:
            raise EvidenceError("processed CSV is not strictly chronological")
        if len(set(numbers)) != 6 or any(number < 1 or number > 49 for number in numbers):
            raise EvidenceError("processed CSV ordinary numbers are invalid")
        if special < 1 or special > 49 or special in numbers:
            raise EvidenceError("processed CSV special number is invalid")
        previous = key
