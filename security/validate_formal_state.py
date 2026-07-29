"""Read-only CI validation of frozen V2 evidence and the formal processed CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_SOURCE_HASH = "36c21ae342e53277e0538e58cab9fd4bca7a0a03bf5719a9d0acca58c23c364a"
EXPECTED_CONFIG_HASH = "e15410fc52406150d89c16792b317cdfc09d39d7c0d5ec411d6712c91273a00d"
EXPECTED_CSV_HASH = "b07017b0077fc458c8046aaf0370620277adcc46110928d477884e57768a972e"
EXPECTED_DRAW_ID = "115000075"


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob(root: Path, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"], cwd=root, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tracked blob is missing: {relative_path}")
    return result.stdout


def validate(root: Path, csv_path: Path) -> dict[str, object]:
    manifest = json.loads(git_blob(root, "prospective_validation_v2/frozen_manifest.json"))
    source_files = {
        path: sha256(git_blob(root, path))
        for path in sorted(manifest["source_files"])
    }
    canonical = json.dumps(
        source_files, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    source_hash = sha256(canonical)
    config_blob = git_blob(root, "prospective_validation_v2/frozen_config.json")
    # The immutable manifest was created on Windows and bound this generated JSON
    # to CRLF bytes. Git stores the same text as LF, so reconstruct that original
    # frozen byte representation explicitly on both runner platforms.
    if b"\r\n" in config_blob:
        raise RuntimeError("tracked frozen config unexpectedly contains CRLF bytes")
    config_hash = sha256(config_blob.replace(b"\n", b"\r\n"))
    if json.loads(config_blob) != json.loads((root / "prospective_validation_v2/frozen_config.json").read_text(encoding="utf-8")):
        raise RuntimeError("frozen config worktree semantics differ from the tracked blob")
    if source_hash != EXPECTED_SOURCE_HASH or source_hash != manifest["source_code_sha256"]:
        raise RuntimeError(f"frozen source hash changed: {source_hash}")
    if config_hash != EXPECTED_CONFIG_HASH or config_hash != manifest["config_sha256"]:
        raise RuntimeError(f"frozen config hash changed: {config_hash}")

    ledger_bytes = git_blob(root, "prospective_validation_v2/ledger.jsonl")
    records = [json.loads(line) for line in ledger_bytes.decode("utf-8").splitlines()]
    if len(records) != 2:
        raise RuntimeError(f"formal ledger record count changed: {len(records)}")
    if any(record.get("event_type") in {"result", "correction", "result_remote_anchor"} for record in records):
        raise RuntimeError("formal ledger contains a result event")
    predictions = [record for record in records if record.get("event_type") == "prediction"]
    if len(predictions) != 1 or str(predictions[0].get("target_draw_id")) != EXPECTED_DRAW_ID:
        raise RuntimeError("formal ledger contains an unexpected or next-draw prediction")

    if not csv_path.is_file():
        raise RuntimeError(f"formal processed CSV is missing: {csv_path}")
    csv_hash = sha256(csv_path.read_bytes())
    if csv_hash != EXPECTED_CSV_HASH:
        raise RuntimeError(f"formal processed CSV hash changed: {csv_hash}")
    return {
        "frozen_source_sha256": source_hash,
        "frozen_config_sha256": config_hash,
        "formal_csv_sha256": csv_hash,
        "ledger_records": len(records),
        "result_events": 0,
        "next_prediction_events": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate(root, args.csv.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
