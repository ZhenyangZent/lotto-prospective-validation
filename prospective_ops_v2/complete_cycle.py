"""Fail-closed orchestration of an official V2 result-to-next-prediction cycle."""
from __future__ import annotations

import argparse
import json
import tempfile
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CycleServices:
    fetch_result: Callable[[str], Any]
    ingest_result: Callable[[Any], Any]
    push_result: Callable[[Any], Any]
    anchor_result: Callable[[Any], Any]
    update_csv: Callable[[Any], Any]
    push_csv: Callable[[Any], Any]
    fetch_next_draw: Callable[[], Any]
    require_next_unannounced: Callable[[Any], Any]
    require_clean: Callable[[], Any]
    create_next_prediction: Callable[[Any], Any]
    anchor_next_prediction: Callable[[Any], Any]
    status: Callable[[], Any]
    preflight: Callable[[], Any] = lambda: None
    postflight: Callable[[], Any] = lambda: None


def execute_complete_cycle(draw_id: str, services: CycleServices) -> dict[str, Any]:
    """Run strictly in order; an exception prevents every subsequent operation."""
    services.preflight()
    official = services.fetch_result(str(draw_id))
    result = services.ingest_result(official)
    result_commit = services.push_result(result)
    result_anchor = services.anchor_result(result_commit)
    csv_update = services.update_csv(result_anchor)
    csv_commit = services.push_csv(csv_update)
    next_draw = services.fetch_next_draw()
    services.require_next_unannounced(next_draw)
    services.require_clean()
    prediction = services.create_next_prediction(next_draw)
    prediction_anchor = services.anchor_next_prediction(prediction)
    final_status = services.status()
    services.postflight()
    return {
        "draw_id": str(draw_id), "result": result, "result_commit": result_commit,
        "result_anchor": result_anchor, "csv_update": csv_update, "csv_commit": csv_commit,
        "next_draw": next_draw, "next_prediction": prediction,
        "next_prediction_anchor": prediction_anchor, "status": final_status,
    }


def commit_ignored_data_path(root: str | Path, data_path: str | Path, message: str) -> str:
    """Force-add exactly one reviewed ignored CSV, verify its staged bytes, and commit."""
    from prospective_v2.remote import git
    base = Path(root); path = Path(data_path)
    relative = path.relative_to(base).as_posix()
    git(base, "-c", "core.autocrlf=false", "add", "-f", "--", relative)
    staged = [line for line in git(base, "diff", "--cached", "--name-only").stdout.splitlines() if line]
    if staged != [relative]:
        raise RuntimeError(f"refusing commit with unexpected staged paths: {staged}")
    staged_bytes = subprocess.run(["git", "show", f":{relative}"], cwd=base, capture_output=True, check=True).stdout
    if hashlib.sha256(staged_bytes).hexdigest() != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("staged CSV bytes differ from the reviewed worktree file")
    git(base, "commit", "-m", message)
    return git(base, "rev-parse", "HEAD").stdout.strip()


def require_processed_data_current(data_path: str | Path, records: list[dict[str, Any]], loader: Callable[[str | Path], Any]) -> None:
    from prospective.canonical import sha256_file
    path = Path(data_path)
    if not path.is_file():
        raise RuntimeError("processed CSV is absent; refusing to begin a partial live cycle")
    predictions = [record for record in records if record.get("event_type") == "prediction"]
    if not predictions:
        raise RuntimeError("processed CSV cannot be bound to a pending prediction")
    latest = predictions[-1]; data = loader(path)
    if sha256_file(path) != latest.get("data_sha256"):
        raise RuntimeError("processed CSV hash does not match the latest prediction input hash")
    if str(data.iloc[-1]["draw_id"]) != str(latest.get("data_end_draw_id")):
        raise RuntimeError("processed CSV tail does not match the latest prediction data end")


def build_live_services() -> CycleServices:  # pragma: no cover - superseded live-network adapter
    """Build the reviewed, fail-closed public-repository workflow.

    Merely constructing this object does not write anything. Every mutable step is
    preceded by frozen/ledger/clean/remote gates, and every push verifies its OID.
    """
    from next_draw_predictor_audited import load_lotto_data
    from prospective.canonical import append_record, read_ledger, sha256_file
    from prospective.gitops import assert_clean
    from prospective.official import fetch_next_draw
    from prospective_v2.config import (DATA_PATH, LEDGER_HEAD_PATH, LEDGER_PATH,
                                       REMOTE_NAME, ROOT, STATE_DIR)
    from prospective_v2.official import fetch_official_precheck
    from prospective_v2.remote import (branch_name, remote_repository,
                                       require_remote_oid)
    from prospective_v2.workflow import (commit_paths, create_prediction,
                                         push_head, verify_frozen_integrity,
                                         verify_v2_ledger, write_json)
    from .result_events import ingest_result
    from .result_remote_anchor import build_result_remote_anchor
    from .status import status_summary
    from .update_processed_data import append_anchored_result

    expected_repository = "https://github.com/ZhenyangZent/lotto-prospective-validation"
    branch = branch_name(ROOT)
    if not branch:
        raise RuntimeError("detached HEAD is prohibited for live V2 operations")
    state: dict[str, Any] = {"branch": branch, "repository": remote_repository(ROOT)}
    temporary = Path(tempfile.mkdtemp(prefix="v2-official-fetch-"))

    def gate(*, clean: bool) -> None:
        verify_frozen_integrity(); verify_v2_ledger(revalidate_remote=True)
        if branch_name(ROOT) != state["branch"]:
            raise RuntimeError("V2 operations branch changed during cycle")
        if remote_repository(ROOT).removesuffix(".git") != expected_repository:
            raise RuntimeError("unexpected public evidence repository")
        if clean:
            assert_clean(ROOT)

    def require_current_data() -> None:
        require_processed_data_current(DATA_PATH, read_ledger(LEDGER_PATH), load_lotto_data)

    def preflight() -> None:
        gate(clean=True); require_current_data()

    def fetch_result(draw_id: str) -> dict[str, Any]:
        gate(clean=True)
        precheck = fetch_official_precheck(draw_id, output_dir=temporary)
        parsed = precheck["metadata"].get("parsed_result")
        if precheck["metadata"]["target_draw_status"] != "ANNOUNCED" or not parsed:
            raise RuntimeError("official result is not announced")
        state["precheck"] = precheck
        # request_url is the final response URL after redirects; result validation
        # rejects any redirect that leaves the official HTTPS host allow-list.
        return {**parsed, "status": "ANNOUNCED", "source": precheck["metadata"]["request_url"]}

    def ingest(official: dict[str, Any]) -> dict[str, Any]:
        precheck = state["precheck"]; draw_id = str(official["draw_id"])
        raw = Path(precheck["raw_path"]).read_bytes()
        raw_path = STATE_DIR / "official_responses" / f"result-{draw_id}-{precheck['metadata']['http_retrieved_at'].replace(':', '')}.raw"
        event_path = STATE_DIR / "predictions" / f"result-{draw_id}-{precheck['metadata']['http_retrieved_at'].replace(':', '')}.json"
        result = ingest_result(LEDGER_PATH, LEDGER_HEAD_PATH, official,
                               raw_response_path=raw_path, raw_response=raw,
                               result_json_path=event_path, root=ROOT, revalidate_remote=True)
        state.update({"raw_path": raw_path, "event_path": event_path, "result": result})
        return result

    def push_result(result: dict[str, Any]) -> str:
        commit = commit_paths([LEDGER_PATH, LEDGER_HEAD_PATH, state["raw_path"], state["event_path"]],
                              f"prospective-v2: official result for draw {result['target_draw_id']}")
        if not push_head():
            raise RuntimeError("result push failed")
        require_remote_oid(ROOT, REMOTE_NAME, f"refs/heads/{branch}", commit)
        state["result_commit"] = commit
        return commit

    def anchor_result(result_commit: str) -> dict[str, Any]:
        result = state["result"]
        event = build_result_remote_anchor(result, result_commit, root=ROOT, remote=REMOTE_NAME,
                                           branch=branch, repository=state["repository"])
        anchor = append_record(LEDGER_PATH, LEDGER_HEAD_PATH, event)
        path = STATE_DIR / "predictions" / f"result-remote-anchor-{result['target_draw_id']}-{anchor['sequence']:04d}.json"
        write_json(path, anchor)
        commit = commit_paths([LEDGER_PATH, LEDGER_HEAD_PATH, path],
                              f"prospective-v2: anchor result for draw {result['target_draw_id']}")
        if not push_head():
            raise RuntimeError("result anchor push failed")
        require_remote_oid(ROOT, REMOTE_NAME, f"refs/heads/{branch}", commit)
        state.update({"result_anchor": anchor, "result_anchor_commit": commit})
        return anchor

    def update_csv(_: Any) -> dict[str, Any]:
        records = read_ledger(LEDGER_PATH)
        update = append_anchored_result(DATA_PATH, state["result"], records, root=ROOT,
                                        validator=load_lotto_data, expected_repository=expected_repository,
                                        expected_branch=branch)
        state["csv_update"] = update
        return update

    def push_csv(update: dict[str, Any]) -> str:
        if update["added"] == 0:
            return "NO_NEW_CSV_COMMIT"
        commit = commit_ignored_data_path(ROOT, DATA_PATH, f"data: append official draw {state['result']['target_draw_id']}")
        if not push_head():
            raise RuntimeError("CSV push failed")
        require_remote_oid(ROOT, REMOTE_NAME, f"refs/heads/{branch}", commit)
        return commit

    def next_draw() -> dict[str, str]:
        value, _ = fetch_next_draw(); return value

    def require_unannounced(value: dict[str, str]) -> None:
        check = fetch_official_precheck(value["draw_id"], output_dir=temporary)
        if check["metadata"]["target_draw_status"] != "NOT_ANNOUNCED":
            raise RuntimeError("next draw is already announced")

    def clean() -> None:
        assert_clean(ROOT)

    def create(value: dict[str, str]) -> dict[str, Any]:
        created = create_prediction(value["draw_id"], value["draw_date"], auto_git=True)
        if not created.get("remote_confirmed") or created.get("remote_anchor") is None:
            raise RuntimeError("next prediction was not fully remotely anchored")
        return created

    def anchor_prediction(created: dict[str, Any]) -> dict[str, Any]:
        require_remote_oid(ROOT, REMOTE_NAME, f"refs/heads/{branch}", created["anchor_commit"])
        return created["remote_anchor"]

    return CycleServices(
        fetch_result, ingest, push_result, anchor_result, update_csv, push_csv,
        next_draw, require_unannounced, clean, create, anchor_prediction,
        lambda: status_summary(read_ledger(LEDGER_PATH), root=ROOT, revalidate_remote=True,
                               expected_repository=expected_repository, expected_branch=branch),
        preflight=preflight, postflight=lambda: (gate(clean=True), require_current_data()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-id", required=True)
    parser.add_argument("--dry-run", action="store_true", help="run the isolated temp-repository simulation")
    parser.add_argument(
        "--recover-stale-lock", action="store_true",
        help="remove the operation lock only when its local owner process is provably absent",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        from .dry_run import run_dry_run
        result = run_dry_run(draw_id=args.draw_id)
    else:
        from prospective_v2.config import LEDGER_PATH
        from .live_resume import LiveCycleController
        from .locking import exclusive_operation_lock, recover_stale_lock
        from .resume import execute_resumable_cycle

        lock_path = LEDGER_PATH.with_name("complete_cycle.operation.lock")
        if args.recover_stale_lock:
            result = recover_stale_lock(lock_path)
        else:
            with exclusive_operation_lock(lock_path, purpose=f"complete-cycle:{args.draw_id}"):
                controller = LiveCycleController()
                result = execute_resumable_cycle(args.draw_id, controller.services())
                result["status"] = controller.final_status()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
