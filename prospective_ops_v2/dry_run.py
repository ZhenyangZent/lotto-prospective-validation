"""End-to-end simulation using a temporary repository and a real bare remote."""
from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
import hashlib
from pathlib import Path
from typing import Any

from prospective.canonical import append_record, initialize_ledger, read_ledger
from prospective_v2.config import EXPERIMENT_ID, EXPERIMENT_VERSION

from .result_events import ingest_result
from .result_remote_anchor import build_result_remote_anchor
from .update_processed_data import append_anchored_result
from .complete_cycle import CycleServices, execute_complete_cycle


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True).stdout.strip()


def _commit_push(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)
    oid = _git(repo, "rev-parse", "HEAD")
    _git(repo, "push", "origin", "HEAD:refs/heads/main")
    remote_oid = _git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]
    if remote_oid != oid:
        raise RuntimeError("dry-run remote OID mismatch")
    return oid


def _prediction(draw_id: str, draw_date: str, probabilities: list[float]) -> dict[str, Any]:
    order = sorted(range(1, 50), key=lambda number: (-probabilities[number - 1], number))
    return {
        "event_type": "prediction", "event_id": str(uuid.uuid4()),
        "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
        "prediction_id": f"{EXPERIMENT_ID}-{draw_id}", "target_draw_id": draw_id,
        "target_draw_date": draw_date, "parent_commit": "0" * 40,
        "probabilities_1_to_49": probabilities, "top6": order[:6],
        "top10": order[:10], "top12": order[:12], "late_prediction": False,
    }


def _prediction_anchor(prediction: dict[str, Any], commit: str) -> dict[str, Any]:
    return {
        "event_type": "remote_anchor", "event_id": str(uuid.uuid4()),
        "experiment_id": EXPERIMENT_ID, "prediction_id": prediction["prediction_id"],
        "prediction_record_hash": prediction["record_hash"], "prediction_commit": commit,
        "remote_ref_oid": commit, "remote_name": "origin", "remote_branch": "main",
        "verified_before_draw": True, "official_draw_status_at_verification": "NOT_ANNOUNCED",
    }


def _snapshot_live(root: Path, *, public_ref: str, operational_csv: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return completed.stdout.strip() if completed.returncode == 0 else None
    relative_paths = [
        "prospective_validation_v2/ledger.jsonl", "prospective_validation_v2/ledger_head.json",
        "prospective_validation_v2/predictions/prediction-115000075.json",
        "prospective_validation_v2/predictions/prediction-115000075-probabilities.csv",
        "prospective_validation_v2/predictions/remote-anchor-115000075.json",
        "prospective_validation_v2/frozen_config.json", "prospective_validation_v2/frozen_manifest.json",
        "data/processed/lotto649.csv",
    ]
    files = {}
    for relative in relative_paths:
        path = root / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    branch = command("branch", "--show-current")
    remote_line = command("ls-remote", "origin", f"refs/heads/{public_ref}")
    remote_rows = [line.split() for line in (remote_line or "").splitlines() if line.strip()]
    remote_oid = remote_rows[0][0] if len(remote_rows) == 1 and len(remote_rows[0]) == 2 and remote_rows[0][1] == f"refs/heads/{public_ref}" else None
    return {
        "head": command("rev-parse", "HEAD"), "branch": branch,
        "main_ref": command("rev-parse", "refs/heads/main"),
        "master_ref": command("rev-parse", "refs/heads/master"),
        "public_default_ref": public_ref, "public_remote_default_oid": remote_oid,
        "operational_csv_path": str(operational_csv.resolve()),
        "operational_csv_sha256": hashlib.sha256(operational_csv.read_bytes()).hexdigest() if operational_csv.is_file() else None,
        "files": files,
    }


def _official_raw(draw_id: str, draw_date: str, numbers: list[int], special: int) -> bytes:
    value = {"rtCode": 0, "content": {"lotto649Res": [{
        "period": draw_id, "lotteryDate": draw_date,
        "drawNumberSize": [*numbers, special], "sellAmount": 1, "totalAmount": 1,
    }]}}
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _failure_recovery_matrix() -> dict[str, Any]:
    """Fault every durable mutation boundary and prove the next run resumes once."""
    from .resume import (
        STATE_ORDER, CycleObservation, CycleState, ResumableCycleServices,
        execute_resumable_cycle,
    )

    reports: list[dict[str, Any]] = []
    for failed_state in STATE_ORDER[:-1]:
        durable = {"index": 0, "failed": False}
        counts = {"result": 0, "result_anchor": 0, "csv": 0,
                  "prediction": 0, "prediction_anchor": 0}

        def inspect(_draw_id: str) -> CycleObservation:
            return CycleObservation(STATE_ORDER[durable["index"]])

        def make_action(state: CycleState):
            def action(_draw_id: str, _observation: CycleObservation) -> str:
                created = {
                    CycleState.NO_RESULT: "result",
                    CycleState.RESULT_COMMIT_REMOTE: "result_anchor",
                    CycleState.RESULT_ANCHOR_REMOTE: "csv",
                    CycleState.CSV_COMMIT_REMOTE: "prediction",
                    CycleState.NEXT_PREDICTION_REMOTE: "prediction_anchor",
                }.get(state)
                if created:
                    counts[created] += 1
                durable["index"] += 1
                if state is failed_state and not durable["failed"]:
                    durable["failed"] = True
                    raise RuntimeError(f"injected after {state.value}")
                return state.value
            return action

        services = ResumableCycleServices(
            inspect=inspect,
            actions={state: make_action(state) for state in STATE_ORDER[:-1]},
        )
        injected = False
        try:
            execute_resumable_cycle("DRY-RUN", services)
        except RuntimeError as exc:
            injected = str(exc) == f"injected after {failed_state.value}"
        resumed = execute_resumable_cycle("DRY-RUN", services)
        unique = all(value == 1 for value in counts.values())
        reports.append({
            "failure_after": failed_state.value,
            "failure_injected": injected,
            "resumed_state": resumed["state"],
            "unique_artifacts": unique,
        })
    return {
        "passed": all(
            row["failure_injected"]
            and row["resumed_state"] == CycleState.CYCLE_COMPLETE.value
            and row["unique_artifacts"]
            for row in reports
        ),
        "injected_boundaries": len(reports),
        "reports": reports,
    }


def run_dry_run(*, draw_id: str = "115000075", live_root: str | Path | None = None,
                formal_source_root: str | Path | None = None,
                operational_csv_path: str | Path | None = None,
                public_ref: str | None = None) -> dict[str, Any]:
    from prospective_v2.config import ROOT
    live = Path(live_root) if live_root is not None else ROOT
    formal_source = Path(formal_source_root) if formal_source_root is not None else ROOT
    operational_csv = Path(operational_csv_path) if operational_csv_path is not None else ROOT.parent / "lotto_analysis" / "data" / "processed" / "lotto649.csv"
    ref = public_ref or ("master" if live == ROOT else (_git(live, "branch", "--show-current") or "main"))
    before_live = _snapshot_live(live, public_ref=ref, operational_csv=operational_csv)
    if before_live["public_remote_default_oid"] is None:
        raise RuntimeError(f"public remote refs/heads/{ref} did not resolve to exactly one OID")
    formal_ledger = formal_source / "prospective_validation_v2" / "ledger.jsonl"
    formal_head = formal_source / "prospective_validation_v2" / "ledger_head.json"
    if not formal_ledger.is_file() or not formal_head.is_file() or not operational_csv.is_file():
        raise FileNotFoundError("dry-run requires the complete formal ledger/head and explicit operational CSV source")
    copied_ledger_bytes = formal_ledger.read_bytes(); copied_head_bytes = formal_head.read_bytes()
    copied_ledger_sha = hashlib.sha256(copied_ledger_bytes).hexdigest()
    copied_head_sha = hashlib.sha256(copied_head_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="prospective-ops-v2-") as temporary:
        base = Path(temporary); repo = base / "repo"; remote = base / "remote.git"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.email", "dry-run@example.invalid")
        _git(repo, "config", "user.name", "V2 Dry Run")
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        _git(repo, "remote", "add", "origin", str(remote))
        ledger = repo / "prospective_validation_v2" / "ledger.jsonl"
        head = repo / "prospective_validation_v2" / "ledger_head.json"
        ledger.parent.mkdir(parents=True); ledger.write_bytes(copied_ledger_bytes); head.write_bytes(copied_head_bytes)
        csv_path = repo / "data" / "processed" / "lotto649.csv"
        csv_path.parent.mkdir(parents=True); csv_path.write_bytes(operational_csv.read_bytes())
        initial = _commit_push(repo, "dry-run: initialize")

        probabilities = [6 / 49] * 49
        copied_records = read_ledger(ledger)
        predictions = [record for record in copied_records if record.get("event_type") == "prediction" and str(record.get("target_draw_id")) == draw_id]
        if len(predictions) != 1:
            raise RuntimeError("formal ledger copy does not contain exactly one requested prediction")
        prediction = predictions[0]
        anchors = [record for record in copied_records if record.get("event_type") == "remote_anchor" and record.get("prediction_id") == prediction["prediction_id"]]
        if len(anchors) != 1:
            raise RuntimeError("formal ledger copy does not contain exactly one prediction anchor")
        prediction_anchor = anchors[0]
        prediction_commit = str(prediction_anchor["prediction_commit"])
        prediction_anchor_commit = "COPIED_FORMAL_PREDICTION_ANCHOR"

        result_numbers = [8, 9, 10, 11, 12, 13]; special = 14
        raw = _official_raw(draw_id, "2026-07-31", result_numbers, special)
        state: dict[str, Any] = {}
        next_id = str(int(draw_id) + 1)

        def fetch_result(requested: str) -> dict[str, Any]:
            if requested != draw_id: raise RuntimeError("dry-run draw id was ignored")
            return {"status": "ANNOUNCED", "source": "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Lotto649Result",
                    "draw_id": draw_id, "draw_date": "2026-07-31", "numbers": result_numbers, "special_number": special}
        def ingest(official: dict[str, Any]) -> dict[str, Any]:
            value = ingest_result(ledger, head, official, raw_response_path=repo / "official-result.raw",
                                  raw_response=raw, result_json_path=repo / f"result-{draw_id}.json", root=repo,
                                  revalidate_remote=False)
            state["result"] = value; return value
        def push_result(result: dict[str, Any]) -> str:
            value = _commit_push(repo, f"prospective-v2: official result for draw {draw_id}"); state["result_commit"] = value; return value
        def anchor_result(commit: str) -> dict[str, Any]:
            value = append_record(ledger, head, build_result_remote_anchor(
                state["result"], commit, root=repo, remote="origin", branch="main", repository=str(remote),
                ledger_relative_path="prospective_validation_v2/ledger.jsonl",
                head_relative_path="prospective_validation_v2/ledger_head.json"))
            state["result_anchor"] = value; state["result_anchor_commit"] = _commit_push(repo, "dry-run: result anchor"); return value
        def update_csv(_: Any) -> dict[str, Any]:
            value = append_anchored_result(csv_path, state["result"], read_ledger(ledger), root=repo,
                                           expected_repository=str(remote), expected_branch="main",
                                           ledger_relative_path="prospective_validation_v2/ledger.jsonl",
                                           head_relative_path="prospective_validation_v2/ledger_head.json")
            state["csv_update"] = value; return value
        def push_csv(_: Any) -> str:
            value = _commit_push(repo, f"data: append official draw {draw_id}"); state["csv_commit"] = value; return value
        def create_next(next_draw: dict[str, str]) -> dict[str, Any]:
            value = append_record(ledger, head, _prediction(next_draw["draw_id"], next_draw["draw_date"], probabilities))
            state["next_prediction"] = value; state["next_prediction_commit"] = _commit_push(repo, "dry-run: next prediction"); return value
        def anchor_next(value: dict[str, Any]) -> dict[str, Any]:
            anchor = append_record(ledger, head, _prediction_anchor(value, state["next_prediction_commit"]))
            state["next_anchor"] = anchor; state["next_anchor_commit"] = _commit_push(repo, "dry-run: next prediction anchor"); return anchor
        def clean() -> None:
            if git_status := _git(repo, "status", "--porcelain"):
                raise RuntimeError(f"dry-run worktree is dirty: {git_status}")
        services = CycleServices(
            fetch_result, ingest, push_result, anchor_result, update_csv, push_csv,
            lambda: {"draw_id": next_id, "draw_date": "2026-08-04"},
            lambda value: None if value["draw_id"] == next_id else (_ for _ in ()).throw(RuntimeError("bad next draw")),
            clean, create_next, anchor_next, lambda: {"ledger_records": len(read_ledger(ledger))},
            preflight=clean, postflight=clean,
        )
        cycle = execute_complete_cycle(draw_id, services)
        final_remote = _git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]
        after_live = _snapshot_live(live, public_ref=ref, operational_csv=operational_csv)
        result = state["result"]; result_anchor = state["result_anchor"]
        return {
            "passed": final_remote == state["next_anchor_commit"] and before_live == after_live,
            "sequence": ["prediction", "prediction_anchor", "result", "result_anchor", "csv", "next_prediction", "next_prediction_anchor"],
            "commits": {"initial": initial, "prediction": prediction_commit,
                        "prediction_anchor": prediction_anchor_commit, "result": state["result_commit"],
                        "result_anchor": state["result_anchor_commit"], "csv": state["csv_commit"],
                        "next_prediction": state["next_prediction_commit"], "next_prediction_anchor": state["next_anchor_commit"]},
            "remote_main": final_remote, "ledger_records": len(read_ledger(ledger)),
            "csv_update": state["csv_update"], "result_record_hash": result["record_hash"],
            "result_anchor_record_hash": result_anchor["record_hash"],
            "prediction_anchor_record_hash": prediction_anchor["record_hash"],
            "next_prediction_anchor_record_hash": state["next_anchor"]["record_hash"],
            "requested_draw_id": draw_id, "cycle": cycle,
            "formal_state_unchanged": before_live == after_live,
            "formal_state_before": before_live, "formal_state_after": after_live,
            "copied_formal_ledger_sha256": copied_ledger_sha,
            "copied_formal_head_sha256": copied_head_sha,
            "temp_initial_ledger_sha256": hashlib.sha256(copied_ledger_bytes).hexdigest(),
            "temp_initial_head_sha256": hashlib.sha256(copied_head_bytes).hexdigest(),
            "failure_recovery": _failure_recovery_matrix(),
        }


if __name__ == "__main__":
    print(json.dumps(run_dry_run(), ensure_ascii=False, indent=2))
