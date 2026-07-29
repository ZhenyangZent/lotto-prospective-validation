"""Live repository adapter for the retry-safe V2 cycle state machine."""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from prospective.canonical import append_record, canonical_bytes, read_ledger
from prospective.gitops import assert_clean
from prospective.official import fetch_next_draw
from prospective_v2.config import (
    DATA_PATH, LEDGER_HEAD_PATH, LEDGER_PATH, REMOTE_NAME, ROOT, STATE_DIR,
)
from prospective_v2.official import fetch_official_precheck
from prospective_v2.remote import branch_name, ls_remote_oid, remote_repository
from prospective_v2.workflow import (
    build_remote_anchor_event, create_prediction, verify_frozen_integrity,
    verify_v2_ledger, write_json,
)

from .complete_cycle import commit_ignored_data_path, require_processed_data_current
from .result_events import ingest_result
from .result_remote_anchor import build_result_remote_anchor
from .resume import (
    CycleObservation, CycleState, ResumableCycleServices, commit_is_remote,
    find_commit_with_exact_bytes, find_commit_with_exact_file, git_output,
    require_fast_forward_push_topology,
)
from .status import status_summary
from .update_processed_data import append_anchored_result
from .verify import EvidenceError


EXPECTED_REPOSITORY = "https://github.com/ZhenyangZent/lotto-prospective-validation"
LEDGER_RELATIVE = "prospective_validation_v2/ledger.jsonl"
HEAD_RELATIVE = "prospective_validation_v2/ledger_head.json"


def _single(values: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    if len(values) > 1:
        raise EvidenceError(f"multiple {label} candidates exist; refusing ambiguous resume")
    return values[0] if values else None


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


class LiveCycleController:  # pragma: no cover - requires real GitHub and official lottery endpoints
    def __init__(self) -> None:
        self.branch = branch_name(ROOT)
        if not self.branch:
            raise RuntimeError("detached HEAD is prohibited for live V2 operations")
        self.repository = remote_repository(ROOT)
        if self.repository.removesuffix(".git") != EXPECTED_REPOSITORY:
            raise RuntimeError("unexpected public evidence repository")
        self.temporary = Path(tempfile.mkdtemp(prefix="v2-official-fetch-"))

    def _remote_oid(self) -> str:
        ref = f"refs/heads/{self.branch}"
        oid = ls_remote_oid(ROOT, REMOTE_NAME, ref)
        if not oid:
            raise RuntimeError(f"remote {ref} did not resolve to exactly one OID")
        fetched = subprocess.run(
            ["git", "fetch", "--quiet", REMOTE_NAME, ref], cwd=ROOT, capture_output=True,
        )
        if fetched.returncode != 0:
            raise RuntimeError("remote branch could not be fetched")
        return oid

    def _head(self) -> str:
        return git_output(ROOT, "rev-parse", "HEAD")

    def _safe_push(self, expected_head: str) -> str:
        if self._head() != expected_head:
            raise RuntimeError("local HEAD changed before push")
        remote_oid = self._remote_oid()
        require_fast_forward_push_topology(
            ROOT, remote=REMOTE_NAME, branch=self.branch,
            local_oid=expected_head, remote_oid=remote_oid,
        )
        if remote_oid != expected_head:
            pushed = subprocess.run(
                ["git", "push", REMOTE_NAME, f"HEAD:refs/heads/{self.branch}"],
                cwd=ROOT, capture_output=True,
            )
            if pushed.returncode != 0:
                raise RuntimeError("fast-forward push failed")
        confirmed = self._remote_oid()
        if confirmed != expected_head:
            raise RuntimeError(f"remote OID mismatch after push: expected={expected_head}, actual={confirmed}")
        return confirmed

    def _commit_paths(self, paths: list[Path], message: str) -> str:
        relative = [_relative(path) for path in paths]
        subprocess.run(["git", "add", "--", *relative], cwd=ROOT, check=True, capture_output=True)
        staged = git_output(ROOT, "diff", "--cached", "--name-only").splitlines()
        if sorted(staged) != sorted(relative):
            raise RuntimeError(f"refusing commit with unexpected staged paths: {staged}")
        subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True, capture_output=True)
        return self._head()

    def _commit_for_record(self, record: dict[str, Any]) -> str | None:
        return find_commit_with_exact_bytes(
            ROOT, LEDGER_RELATIVE, canonical_bytes(record) + b"\n",
        )

    def _record_path(self, prefix: str, record: dict[str, Any]) -> Path:
        matches: list[Path] = []
        for path in (STATE_DIR / "predictions").glob(prefix):
            try:
                if json.loads(path.read_text(encoding="utf-8")) == record:
                    matches.append(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        if len(matches) != 1:
            raise EvidenceError(f"expected exactly one artifact matching {prefix}")
        return matches[0]

    def _csv_matches(self, result: dict[str, Any]) -> bool:
        if not DATA_PATH.is_file():
            raise EvidenceError("processed CSV is absent")
        with DATA_PATH.open(encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if str(row.get("draw_id")) == str(result["target_draw_id"])]
        if not rows:
            return False
        if len(rows) != 1:
            raise EvidenceError("processed CSV contains duplicate target draw rows")
        row = rows[0]
        same = (
            row.get("draw_date") == str(result["target_draw_date"])
            and [row.get(f"number_{index}") for index in range(1, 7)]
            == [str(value) for value in result["actual_numbers"]]
            and row.get("special_number") == str(result["special_number"])
        )
        if not same:
            raise EvidenceError("processed CSV contains conflicting target draw values")
        return True

    def inspect(self, draw_id: str) -> CycleObservation:
        verify_v2_ledger(revalidate_remote=False)
        records = read_ledger(LEDGER_PATH)
        remote_oid = self._remote_oid()
        results = [
            record for record in records
            if record.get("event_type") in {"result", "correction"}
            and str(record.get("target_draw_id")) == str(draw_id)
        ]
        result = _single(results, "result/correction")
        base: dict[str, Any] = {
            "head": self._head(), "remote_oid": remote_oid, "branch": self.branch,
        }
        if result is None:
            return CycleObservation(CycleState.NO_RESULT, base)
        base["result"] = result
        result_commit = self._commit_for_record(result)
        base["result_commit"] = result_commit
        if result_commit is None:
            return CycleObservation(CycleState.RESULT_APPENDED_LOCAL, base)
        if not commit_is_remote(ROOT, result_commit, remote_oid):
            return CycleObservation(CycleState.RESULT_COMMIT_LOCAL_ONLY, base)
        anchors = [
            record for record in records
            if record.get("event_type") == "result_remote_anchor"
            and record.get("result_record_hash") == result.get("record_hash")
        ]
        anchor = _single(anchors, "result anchor")
        if anchor is None:
            return CycleObservation(CycleState.RESULT_COMMIT_REMOTE, base)
        base["result_anchor"] = anchor
        anchor_commit = self._commit_for_record(anchor)
        base["result_anchor_commit"] = anchor_commit
        if anchor_commit is None or not commit_is_remote(ROOT, anchor_commit, remote_oid):
            return CycleObservation(CycleState.RESULT_ANCHORED_LOCAL, base)
        if not self._csv_matches(result):
            return CycleObservation(CycleState.RESULT_ANCHOR_REMOTE, base)
        csv_commit = find_commit_with_exact_file(ROOT, _relative(DATA_PATH), DATA_PATH.read_bytes())
        base["csv_commit"] = csv_commit
        if csv_commit is None:
            return CycleObservation(CycleState.CSV_UPDATED_LOCAL, base)
        if not commit_is_remote(ROOT, csv_commit, remote_oid):
            return CycleObservation(CycleState.CSV_COMMIT_LOCAL_ONLY, base)
        next_predictions = [
            record for record in records
            if record.get("event_type") == "prediction"
            and str(record.get("data_end_draw_id")) == str(draw_id)
            and str(record.get("target_draw_id")) != str(draw_id)
        ]
        prediction = _single(next_predictions, "next prediction")
        if prediction is None:
            return CycleObservation(CycleState.CSV_COMMIT_REMOTE, base)
        base["next_prediction"] = prediction
        prediction_commit = self._commit_for_record(prediction)
        base["prediction_commit"] = prediction_commit
        if prediction_commit is None or not commit_is_remote(ROOT, prediction_commit, remote_oid):
            return CycleObservation(CycleState.NEXT_PREDICTION_LOCAL, base)
        prediction_anchors = [
            record for record in records
            if record.get("event_type") == "remote_anchor"
            and record.get("prediction_id") == prediction.get("prediction_id")
        ]
        prediction_anchor = _single(prediction_anchors, "next prediction anchor")
        if prediction_anchor is None:
            return CycleObservation(CycleState.NEXT_PREDICTION_REMOTE, base)
        base["next_prediction_anchor"] = prediction_anchor
        prediction_anchor_commit = self._commit_for_record(prediction_anchor)
        base["prediction_anchor_commit"] = prediction_anchor_commit
        if prediction_anchor_commit is None or not commit_is_remote(ROOT, prediction_anchor_commit, remote_oid):
            return CycleObservation(CycleState.NEXT_PREDICTION_ANCHORED, base)
        return CycleObservation(CycleState.CYCLE_COMPLETE, base)

    def preflight(self, draw_id: str) -> None:
        verify_frozen_integrity()
        verify_v2_ledger(revalidate_remote=True)
        if branch_name(ROOT) != self.branch:
            raise RuntimeError("V2 operations branch changed")
        observation = self.inspect(draw_id)
        expected_head = {
            CycleState.RESULT_COMMIT_LOCAL_ONLY: observation.details.get("result_commit"),
            CycleState.CSV_COMMIT_LOCAL_ONLY: observation.details.get("csv_commit"),
            CycleState.NEXT_PREDICTION_LOCAL: observation.details.get("prediction_commit"),
            CycleState.NEXT_PREDICTION_ANCHORED: observation.details.get("prediction_anchor_commit"),
        }.get(observation.state)
        if observation.state is CycleState.RESULT_ANCHORED_LOCAL:
            expected_head = observation.details.get("result_anchor_commit")
        expected_head = expected_head or observation.details["remote_oid"]
        if observation.details["head"] != expected_head:
            raise RuntimeError(
                "local HEAD contains an unknown commit for the observed resume state; stopping"
            )
        if observation.state is CycleState.NO_RESULT:
            from next_draw_predictor_audited import load_lotto_data
            assert_clean(ROOT)
            require_processed_data_current(DATA_PATH, read_ledger(LEDGER_PATH), load_lotto_data)

    def postflight(self, draw_id: str) -> None:
        if self.inspect(draw_id).state is not CycleState.CYCLE_COMPLETE:
            raise RuntimeError("cycle postflight did not reach CYCLE_COMPLETE")
        assert_clean(ROOT)

    def append_result(self, draw_id: str, _: CycleObservation) -> dict[str, Any]:
        precheck = fetch_official_precheck(draw_id, output_dir=self.temporary)
        parsed = precheck["metadata"].get("parsed_result")
        if precheck["metadata"]["target_draw_status"] != "ANNOUNCED" or not parsed:
            raise RuntimeError("official result is not announced")
        official = {**parsed, "status": "ANNOUNCED", "source": precheck["metadata"]["request_url"]}
        stamp = precheck["metadata"]["http_retrieved_at"].replace(":", "")
        raw_path = STATE_DIR / "official_responses" / f"result-{draw_id}-{stamp}.raw"
        event_path = STATE_DIR / "predictions" / f"result-{draw_id}-{stamp}.json"
        return ingest_result(
            LEDGER_PATH, LEDGER_HEAD_PATH, official,
            raw_response_path=raw_path, raw_response=Path(precheck["raw_path"]).read_bytes(),
            result_json_path=event_path, root=ROOT, revalidate_remote=True,
        )

    def commit_result(self, draw_id: str, observation: CycleObservation) -> str:
        result = dict(observation.details["result"])
        raw_path = ROOT / str(result["official_raw_response_path"])
        event_path = self._record_path(f"result-{draw_id}-*.json", result)
        return self._commit_paths(
            [LEDGER_PATH, LEDGER_HEAD_PATH, raw_path, event_path],
            f"prospective-v2: official result for draw {draw_id}",
        )

    def push_result(self, _draw_id: str, observation: CycleObservation) -> str:
        return self._safe_push(str(observation.details["result_commit"]))

    def append_result_anchor(self, draw_id: str, observation: CycleObservation) -> dict[str, Any]:
        result = dict(observation.details["result"])
        result_commit = str(observation.details["result_commit"])
        event = build_result_remote_anchor(
            result, result_commit, root=ROOT, remote=REMOTE_NAME, branch=self.branch,
            repository=self.repository,
        )
        anchor = append_record(LEDGER_PATH, LEDGER_HEAD_PATH, event)
        path = STATE_DIR / "predictions" / f"result-remote-anchor-{draw_id}-{anchor['sequence']:04d}.json"
        write_json(path, anchor)
        return anchor

    def finish_result_anchor(self, draw_id: str, observation: CycleObservation) -> str:
        anchor = dict(observation.details["result_anchor"])
        commit = observation.details.get("result_anchor_commit")
        if commit is None:
            path = self._record_path(f"result-remote-anchor-{draw_id}-*.json", anchor)
            commit = self._commit_paths(
                [LEDGER_PATH, LEDGER_HEAD_PATH, path],
                f"prospective-v2: anchor result for draw {draw_id}",
            )
        return self._safe_push(str(commit))

    def append_csv(self, _draw_id: str, observation: CycleObservation) -> dict[str, Any]:
        from next_draw_predictor_audited import load_lotto_data
        return append_anchored_result(
            DATA_PATH, dict(observation.details["result"]), read_ledger(LEDGER_PATH),
            root=ROOT, validator=load_lotto_data, expected_repository=EXPECTED_REPOSITORY,
            expected_branch=self.branch,
        )

    def commit_csv(self, draw_id: str, _observation: CycleObservation) -> str:
        return commit_ignored_data_path(ROOT, DATA_PATH, f"data: append official draw {draw_id}")

    def push_csv(self, _draw_id: str, observation: CycleObservation) -> str:
        return self._safe_push(str(observation.details["csv_commit"]))

    def create_next(self, _draw_id: str, observation: CycleObservation) -> dict[str, Any]:
        if self._head() != observation.details["remote_oid"]:
            raise RuntimeError("next prediction creation requires local HEAD equal to remote OID")
        assert_clean(ROOT)
        next_draw, _ = fetch_next_draw()
        check = fetch_official_precheck(next_draw["draw_id"], output_dir=self.temporary)
        if check["metadata"]["target_draw_status"] != "NOT_ANNOUNCED":
            raise RuntimeError("next draw is already announced")
        return create_prediction(next_draw["draw_id"], next_draw["draw_date"], auto_git=True)

    def finish_prediction(self, _draw_id: str, observation: CycleObservation) -> str:
        prediction = dict(observation.details["next_prediction"])
        commit = observation.details.get("prediction_commit")
        if commit is None:
            draw_id = str(prediction["target_draw_id"])
            prediction_path = STATE_DIR / "predictions" / f"prediction-{draw_id}.json"
            probability_path = STATE_DIR / "predictions" / f"prediction-{draw_id}-probabilities.csv"
            support: list[Path] = []
            for field in ("official_precheck_metadata", "official_precheck_raw"):
                matches = list(STATE_DIR.rglob(str(prediction[field])))
                if len(matches) != 1:
                    raise EvidenceError(f"expected one prediction support artifact for {field}")
                support.append(matches[0])
            commit = self._commit_paths(
                [LEDGER_PATH, LEDGER_HEAD_PATH, prediction_path, probability_path, *support],
                f"prospective-v2: prediction for draw {draw_id}",
            )
        return self._safe_push(str(commit))

    def append_prediction_anchor(self, _draw_id: str, observation: CycleObservation) -> dict[str, Any]:
        prediction = dict(observation.details["next_prediction"])
        commit = str(observation.details["prediction_commit"])
        check = fetch_official_precheck(prediction["target_draw_id"], output_dir=self.temporary)
        if check["metadata"]["target_draw_status"] != "NOT_ANNOUNCED":
            raise RuntimeError("official result was announced before prediction anchor")
        event = build_remote_anchor_event(
            prediction, commit, remote=REMOTE_NAME, branch=self.branch,
            remote_oid=self._remote_oid(), repository=self.repository,
            verification_precheck=check,
        )
        anchor = append_record(LEDGER_PATH, LEDGER_HEAD_PATH, event)
        path = STATE_DIR / "predictions" / f"remote-anchor-{prediction['target_draw_id']}.json"
        write_json(path, anchor)
        return anchor

    def finish_prediction_anchor(self, _draw_id: str, observation: CycleObservation) -> str:
        prediction = dict(observation.details["next_prediction"])
        anchor = dict(observation.details["next_prediction_anchor"])
        commit = observation.details.get("prediction_anchor_commit")
        if commit is None:
            path = self._record_path(f"remote-anchor-{prediction['target_draw_id']}.json", anchor)
            support: list[Path] = []
            for field in ("verification_precheck_metadata", "verification_precheck_raw"):
                matches = list(STATE_DIR.rglob(str(anchor[field])))
                if len(matches) != 1:
                    raise EvidenceError(f"expected one anchor support artifact for {field}")
                support.append(matches[0])
            commit = self._commit_paths(
                [LEDGER_PATH, LEDGER_HEAD_PATH, path, *support],
                f"prospective-v2: anchor prediction for draw {prediction['target_draw_id']}",
            )
        return self._safe_push(str(commit))

    def services(self) -> ResumableCycleServices:
        return ResumableCycleServices(
            inspect=self.inspect,
            actions={
                CycleState.NO_RESULT: self.append_result,
                CycleState.RESULT_APPENDED_LOCAL: self.commit_result,
                CycleState.RESULT_COMMIT_LOCAL_ONLY: self.push_result,
                CycleState.RESULT_COMMIT_REMOTE: self.append_result_anchor,
                CycleState.RESULT_ANCHORED_LOCAL: self.finish_result_anchor,
                CycleState.RESULT_ANCHOR_REMOTE: self.append_csv,
                CycleState.CSV_UPDATED_LOCAL: self.commit_csv,
                CycleState.CSV_COMMIT_LOCAL_ONLY: self.push_csv,
                CycleState.CSV_COMMIT_REMOTE: self.create_next,
                CycleState.NEXT_PREDICTION_LOCAL: self.finish_prediction,
                CycleState.NEXT_PREDICTION_REMOTE: self.append_prediction_anchor,
                CycleState.NEXT_PREDICTION_ANCHORED: self.finish_prediction_anchor,
            },
            preflight=self.preflight,
            postflight=self.postflight,
        )

    def final_status(self) -> dict[str, Any]:
        return status_summary(
            read_ledger(LEDGER_PATH), root=ROOT, revalidate_remote=True,
            expected_repository=EXPECTED_REPOSITORY, expected_branch=self.branch,
        )
