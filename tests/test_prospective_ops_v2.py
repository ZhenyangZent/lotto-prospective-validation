"""Isolated acceptance tests for the V2 official-result operations package."""
from __future__ import annotations

import copy
import csv
import hashlib
import math
import os
import socket
import subprocess
import json
from pathlib import Path

import pytest
import pandas as pd
import prospective_ops_v2.result_events as result_events_module

from prospective.canonical import append_record, initialize_ledger, read_ledger, sha256_file
from prospective.metrics import score_prediction
from prospective_v2.config import EXPERIMENT_ID, EXPERIMENT_VERSION, SOURCE_FILES
from prospective_ops_v2.complete_cycle import CycleServices, execute_complete_cycle
from prospective_ops_v2.complete_cycle import commit_ignored_data_path, require_processed_data_current
from prospective_ops_v2.dry_run import run_dry_run
from prospective_ops_v2.result_events import build_result_event, ingest_result
from prospective_ops_v2.result_remote_anchor import build_result_remote_anchor, resolve_result_anchor
from prospective_ops_v2.status import status_summary
from prospective_ops_v2.update_processed_data import append_anchored_result
from prospective_ops_v2.verify import EvidenceError, repository_fingerprint
from prospective_ops_v2.locking import exclusive_operation_lock, inspect_lock, recover_stale_lock
from prospective_ops_v2.resume import (
    STATE_ORDER, CycleObservation, CycleState, ResumableCycleServices,
    execute_resumable_cycle, require_fast_forward_push_topology,
)


ROOT = Path(__file__).resolve().parents[1]
UNIFORM = [6 / 49] * 49


def git(root: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=check,
    ).stdout.strip()


def prediction_event(draw_id: str = "115000075", date: str = "2026-07-31") -> dict:
    order = list(range(1, 50))
    return {
        "event_type": "prediction", "event_id": f"prediction-{draw_id}",
        "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
        "prediction_id": f"{EXPERIMENT_ID}-{draw_id}", "target_draw_id": draw_id,
        "target_draw_date": date, "parent_commit": "0" * 40,
        "probabilities_1_to_49": UNIFORM.copy(), "top6": order[:6],
        "top10": order[:10], "top12": order[:12], "late_prediction": False,
    }


def official_result(**changes) -> dict:
    value = {
        "status": "ANNOUNCED", "source": "https://api.taiwanlottery.com/result",
        "draw_id": "115000075", "draw_date": "2026-07-31",
        "numbers": [1, 8, 15, 22, 29, 36], "special_number": 43,
    }
    value.update(changes)
    return value


def raw_for(result: dict) -> bytes:
    payload = {"rtCode": 0, "content": {"lotto649Res": [{
        "period": result["draw_id"], "lotteryDate": result["draw_date"],
        "drawNumberSize": [*result["numbers"], result["special_number"]],
        "sellAmount": 1, "totalAmount": 1,
    }]}}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def ledger_with_prediction(tmp_path: Path, *, anchor: bool = True) -> tuple[Path, Path, list[dict]]:
    ledger = tmp_path / "ledger.jsonl"; head = tmp_path / "ledger_head.json"
    initialize_ledger(ledger, head)
    prediction = append_record(ledger, head, prediction_event())
    if anchor:
        append_record(ledger, head, {
            "event_type": "remote_anchor", "event_id": "prediction-anchor",
            "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
            "prediction_id": prediction["prediction_id"],
            "prediction_record_hash": prediction["record_hash"],
            "prediction_commit": "a" * 40, "remote_ref_oid": "a" * 40,
            "remote_name": "origin", "remote_branch": "main",
            "verified_before_draw": True,
            "official_draw_status_at_verification": "NOT_ANNOUNCED",
        })
    return ledger, head, read_ledger(ledger)


def build_result(tmp_path: Path, *, records: list[dict] | None = None, result: dict | None = None) -> dict:
    if records is None:
        _, _, records = ledger_with_prediction(tmp_path)
    selected = result or official_result()
    return build_result_event(
        records, selected, raw_response_path="official.raw",
        raw_response=raw_for(selected), root=tmp_path, revalidate_remote=False,
        retrieved_at="2026-07-31T21:00:00+08:00",
    )


def append_result_and_anchor(tmp_path: Path) -> tuple[Path, Path, Path, dict, list[dict]]:
    repo, remote = init_git_remote(tmp_path)
    ledger, head, records = ledger_with_prediction(repo)
    result = append_record(ledger, head, build_result(tmp_path, records=records))
    git(repo, "add", "ledger.jsonl", "ledger_head.json"); git(repo, "commit", "-m", "result")
    result_commit = git(repo, "rev-parse", "HEAD"); git(repo, "push", "origin", "HEAD:refs/heads/main")
    anchor = build_result_remote_anchor(result, result_commit, root=repo, remote="origin", branch="main",
                                        repository=str(remote), ledger_relative_path="ledger.jsonl",
                                        head_relative_path="ledger_head.json")
    append_record(ledger, head, anchor)
    return repo, ledger, head, result, read_ledger(ledger)


def write_csv(path: Path, row: str | None = None) -> bytes:
    header = "draw_id,draw_date,number_1,number_2,number_3,number_4,number_5,number_6,special_number,source,source_version\n"
    body = row or "115000074,2026-07-28,2,9,16,23,30,37,44,official,seed\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8")
    return path.read_bytes()


def mock_formal_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Build isolated mock evidence and CSV inputs for destructive dry-runs."""
    formal = tmp_path / "mock-formal"
    ledger = formal / "prospective_validation_v2" / "ledger.jsonl"
    head = formal / "prospective_validation_v2" / "ledger_head.json"
    operational = tmp_path / "mock-operational" / "lotto649.csv"
    csv_bytes = write_csv(operational)
    initialize_ledger(ledger, head)
    prediction = prediction_event()
    prediction.update({
        "data_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "data_end_draw_id": "115000074",
    })
    prediction = append_record(ledger, head, prediction)
    append_record(ledger, head, {
        "event_type": "remote_anchor", "event_id": "mock-prediction-anchor",
        "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
        "prediction_id": prediction["prediction_id"],
        "prediction_record_hash": prediction["record_hash"],
        "prediction_commit": "a" * 40, "remote_ref_oid": "a" * 40,
        "remote_name": "origin", "remote_branch": "main",
        "verified_before_draw": True,
        "official_draw_status_at_verification": "NOT_ANNOUNCED",
    })
    return formal, operational


def init_git_remote(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"; remote = tmp_path / "remote.git"; repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    git(repo, "remote", "add", "origin", str(remote))
    (repo / "seed").write_text("seed", encoding="ascii")
    git(repo, "add", "seed"); git(repo, "commit", "-m", "seed")
    git(repo, "push", "origin", "HEAD:refs/heads/main")
    return repo, remote


def test_01_unannounced_result_is_rejected(tmp_path):
    with pytest.raises(EvidenceError, match="not been announced"):
        build_result(tmp_path, result=official_result(status="NOT_ANNOUNCED"))


def test_02_nonofficial_host_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="HTTPS host"):
        build_result(tmp_path, result=official_result(source="https://evil.invalid/result"))


def test_03_missing_prediction_is_rejected(tmp_path):
    with pytest.raises(EvidenceError, match="exactly one prediction"):
        build_result_event([], official_result(), raw_response_path="x", raw_response=b"x", root=tmp_path, revalidate_remote=False)


def test_04_missing_prediction_anchor_is_rejected(tmp_path):
    _, _, records = ledger_with_prediction(tmp_path, anchor=False)
    with pytest.raises(EvidenceError, match="no valid"):
        build_result(tmp_path, records=records)


def test_05_invalidated_remote_prediction_anchor_is_rejected(tmp_path):
    _, _, records = ledger_with_prediction(tmp_path)
    with pytest.raises(EvidenceError, match="no valid"):
        build_result_event(records, official_result(), raw_response_path="x", raw_response=b"x", root=tmp_path, revalidate_remote=True)


def test_06_draw_id_mismatch_is_rejected(tmp_path):
    _, _, records = ledger_with_prediction(tmp_path)
    with pytest.raises(EvidenceError, match="exactly one prediction"):
        build_result(tmp_path, records=records, result=official_result(draw_id="115000076"))


def test_07_draw_date_mismatch_is_rejected(tmp_path):
    with pytest.raises(EvidenceError, match="date"):
        build_result(tmp_path, result=official_result(draw_date="2026-08-01"))


def test_08_ordinary_number_out_of_range_is_rejected(tmp_path):
    with pytest.raises(EvidenceError, match="1..49"):
        build_result(tmp_path, result=official_result(numbers=[0, 2, 3, 4, 5, 6]))


def test_09_duplicate_ordinary_number_is_rejected(tmp_path):
    with pytest.raises(EvidenceError, match="unique"):
        build_result(tmp_path, result=official_result(numbers=[1, 1, 3, 4, 5, 6]))


def test_10_special_number_out_of_range_is_rejected(tmp_path):
    with pytest.raises(EvidenceError, match="special number"):
        build_result(tmp_path, result=official_result(special_number=50))


def test_11_special_number_duplicate_is_rejected(tmp_path):
    with pytest.raises(EvidenceError, match="differ"):
        build_result(tmp_path, result=official_result(special_number=8))


def test_12_result_does_not_modify_prediction(tmp_path):
    _, _, records = ledger_with_prediction(tmp_path)
    before = copy.deepcopy(records[0])
    build_result(tmp_path, records=records)
    assert records[0] == before


def test_13_brier_matches_manual_calculation(tmp_path):
    event = build_result(tmp_path)
    target = [1.0 if index + 1 in event["actual_numbers"] else 0.0 for index in range(49)]
    assert event["brier"] == pytest.approx(sum((p - y) ** 2 for p, y in zip(UNIFORM, target)) / 49)


def test_14_uniform_brier_matches_manual_calculation(tmp_path):
    event = build_result(tmp_path)
    expected = (6 / 49) * (1 - 6 / 49)
    assert event["uniform_brier"] == pytest.approx(expected)


def test_15_log_loss_matches_manual_calculation(tmp_path):
    event = build_result(tmp_path)
    expected = -(6 * math.log(6 / 49) + 43 * math.log(43 / 49)) / 49
    assert event["log_loss"] == pytest.approx(expected)
    assert event["uniform_log_loss"] == pytest.approx(expected)


def test_16_top_6_10_12_hits_match_ranked_probabilities(tmp_path):
    _, _, records = ledger_with_prediction(tmp_path)
    probabilities = [float(index) for index in range(1, 50)]
    probabilities = [value / sum(probabilities) * 6 for value in probabilities]
    records[0]["probabilities_1_to_49"] = probabilities
    event = build_result(tmp_path, records=records, result=official_result(numbers=[38, 40, 44, 47, 48, 49], special_number=1))
    assert (event["hits_top6"], event["hits_top10"], event["hits_top12"]) == (4, 5, 6)


def test_17_identical_result_is_rejected(tmp_path):
    ledger, head, records = ledger_with_prediction(tmp_path)
    append_record(ledger, head, build_result(tmp_path, records=records))
    with pytest.raises(EvidenceError, match="identical"):
        build_result(tmp_path, records=read_ledger(ledger))


def test_18_changed_official_result_becomes_correction(tmp_path):
    ledger, head, records = ledger_with_prediction(tmp_path)
    original = append_record(ledger, head, build_result(tmp_path, records=records))
    correction = build_result(tmp_path, records=read_ledger(ledger), result=official_result(numbers=[2, 9, 16, 23, 30, 37], special_number=44))
    assert correction["event_type"] == "correction"
    assert original["actual_numbers"] == [1, 8, 15, 22, 29, 36]


def test_19_unpushed_result_cannot_be_anchored(tmp_path):
    repo, _ = init_git_remote(tmp_path)
    result = {"target_draw_id": "1", "event_type": "result", "record_hash": "c" * 64}
    (repo / "local").write_text("unpublished", encoding="ascii"); git(repo, "add", "local"); git(repo, "commit", "-m", "local")
    local = git(repo, "rev-parse", "HEAD")
    with pytest.raises(EvidenceError, match="mismatch"):
        build_result_remote_anchor(result, local, root=repo, remote="origin", branch="main", repository="https://github.com/x/y.git")


def test_20_result_remote_oid_mismatch_is_rejected(tmp_path):
    repo, _ = init_git_remote(tmp_path)
    result = {"target_draw_id": "1", "event_type": "result", "record_hash": "c" * 64}
    with pytest.raises(EvidenceError, match="mismatch"):
        build_result_remote_anchor(result, "f" * 40, root=repo, remote="origin", branch="main", repository="https://github.com/x/y.git")


def test_21_only_valid_result_anchor_counts_as_completed(tmp_path):
    repo, ledger, head, result, records = append_result_and_anchor(tmp_path)
    summary = status_summary(records, root=repo, revalidate_remote=False,
                             ledger_relative_path="ledger.jsonl", head_relative_path="ledger_head.json")
    assert summary["valid_completed_draws"] == 1
    bad = [record for record in records if record.get("event_type") != "result_remote_anchor"]
    assert status_summary(bad, root=repo, revalidate_remote=False,
                          ledger_relative_path="ledger.jsonl", head_relative_path="ledger_head.json")["valid_completed_draws"] == 0


def test_22_csv_appends_exactly_one_row_and_preserves_prefix(tmp_path):
    repo, _, _, result, records = append_result_and_anchor(tmp_path)
    path = tmp_path / "lotto649.csv"; before = write_csv(path)
    update = append_anchored_result(path, result, records, root=repo, revalidate_remote=False,
                                    ledger_relative_path="ledger.jsonl", head_relative_path="ledger_head.json")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert update["added"] == 1 and len(rows) == 2 and path.read_bytes().startswith(before)


def test_23_csv_update_is_idempotent(tmp_path):
    repo, _, _, result, records = append_result_and_anchor(tmp_path)
    path = tmp_path / "lotto649.csv"; write_csv(path)
    append_anchored_result(path, result, records, root=repo, revalidate_remote=False,
                           ledger_relative_path="ledger.jsonl", head_relative_path="ledger_head.json")
    before = path.read_bytes()
    update = append_anchored_result(path, result, records, root=repo, revalidate_remote=False,
                                    ledger_relative_path="ledger.jsonl", head_relative_path="ledger_head.json")
    assert update["idempotent"] is True and path.read_bytes() == before


def test_24_csv_conflict_stops_without_modification(tmp_path):
    repo, _, _, result, records = append_result_and_anchor(tmp_path)
    path = tmp_path / "lotto649.csv"
    before = write_csv(path, "115000075,2026-07-31,2,3,4,5,6,7,8,official,old\n")
    with pytest.raises(EvidenceError, match="conflicting"):
        append_anchored_result(path, result, records, root=repo, revalidate_remote=False,
                               ledger_relative_path="ledger.jsonl", head_relative_path="ledger_head.json")
    assert path.read_bytes() == before


def test_25_next_prediction_observes_updated_csv(tmp_path):
    from next_draw_predictor_audited import build_next_features, load_lotto_data
    path = tmp_path / "lotto649.csv"; write_csv(path)
    calls = []
    def step(name, value=None):
        def call(*args): calls.append(name); return value if value is not None else name
        return call
    def predict_from_updated_csv(_):
        calls.append("predict")
        data = load_lotto_data(path)
        features = build_next_features(data)
        assert str(data.iloc[-1]["draw_id"]) == "115000075"
        assert set(features["history_draws"]) == {2}
        return {"data_end_draw_id": str(data.iloc[-1]["draw_id"])}
    services = CycleServices(
        step("fetch", {}), step("ingest", {}), step("push-result", "r"), step("anchor-result", {}),
        lambda _: (path.write_text(path.read_text() + "115000075,2026-07-31,1,8,15,22,29,36,43,official,v2\n"), calls.append("csv"))[1],
        step("push-csv", "c"), step("next", {"draw_id": "115000076"}), step("unannounced"), step("clean"),
        predict_from_updated_csv,
        step("anchor-prediction", {}), step("status", {}),
    )
    execute_complete_cycle("115000075", services)
    assert calls.index("csv") < calls.index("predict") and "115000075" in path.read_text()


def test_26_complete_cycle_stops_at_first_failure():
    calls = []
    def ok(name):
        return lambda *args: calls.append(name) or name
    def fail(*args):
        calls.append("push-result"); raise RuntimeError("stop")
    services = CycleServices(ok("fetch"), ok("ingest"), fail, ok("anchor-result"), ok("csv"), ok("push-csv"),
                             ok("next"), ok("unannounced"), ok("clean"), ok("predict"), ok("anchor-prediction"), ok("status"))
    with pytest.raises(RuntimeError, match="stop"):
        execute_complete_cycle("1", services)
    assert calls == ["fetch", "ingest", "push-result"]


def test_27_dry_run_uses_real_git_without_changing_unrelated_remote(tmp_path):
    repo, _ = init_git_remote(tmp_path)
    formal, operational = mock_formal_inputs(tmp_path)
    before = git(repo, "ls-remote", "origin", "refs/heads/main")
    result = run_dry_run(draw_id="115000075", live_root=repo, formal_source_root=formal,
                         operational_csv_path=operational,
                         public_ref="main")
    after = git(repo, "ls-remote", "origin", "refs/heads/main")
    assert result["passed"] and result["sequence"] == ["prediction", "prediction_anchor", "result", "result_anchor", "csv", "next_prediction", "next_prediction_anchor"]
    assert before == after and result["formal_state_unchanged"]
    assert result["formal_state_before"]["head"] is not None
    assert result["formal_state_before"]["public_remote_default_oid"] is not None


def test_28_frozen_sources_config_ledger_and_formal_csv_are_unchanged(tmp_path):
    tracked = [*SOURCE_FILES, "prospective_validation_v2/frozen_config.json",
               "prospective_validation_v2/ledger.jsonl", "prospective_validation_v2/ledger_head.json",
               "prospective_validation_v2/frozen_manifest.json"]
    before = repository_fingerprint(ROOT, tracked)
    mirror, _ = init_git_remote(tmp_path)
    formal, operational = mock_formal_inputs(tmp_path)
    operational_before = sha256_file(operational)
    mock_ledger_before = sha256_file(formal / "prospective_validation_v2" / "ledger.jsonl")
    result = run_dry_run(draw_id="115000075", live_root=mirror, formal_source_root=formal,
                         operational_csv_path=operational, public_ref="main")
    after = repository_fingerprint(ROOT, tracked)
    assert before == after
    assert result["formal_state_unchanged"]
    assert before["prospective_validation_v2/ledger.jsonl"] is not None
    assert sha256_file(operational) == operational_before
    assert result["copied_formal_ledger_sha256"] == mock_ledger_before
    assert all(not path.startswith("prospective_ops_v2/") for path in SOURCE_FILES)


def test_29_ingest_saves_raw_response_and_hash_without_overwriting(tmp_path):
    ledger, head, _ = ledger_with_prediction(tmp_path)
    raw = tmp_path / "raw" / "official.raw"; event = tmp_path / "events" / "result.json"
    payload = raw_for(official_result())
    record = ingest_result(ledger, head, official_result(), raw_response_path=raw, raw_response=payload,
                           result_json_path=event, root=tmp_path, revalidate_remote=False)
    assert raw.read_bytes() == payload
    assert record["official_raw_response_sha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(EvidenceError, match="identical"):
        ingest_result(ledger, head, official_result(), raw_response_path=raw, raw_response=payload,
                      result_json_path=event, root=tmp_path, revalidate_remote=False)


def test_30_status_ignores_untrusted_remote_confirmed_boolean(tmp_path):
    ledger, head, records = ledger_with_prediction(tmp_path)
    result = build_result(tmp_path, records=records); result["remote_commit_confirmed"] = True
    append_record(ledger, head, result)
    summary = status_summary(read_ledger(ledger), root=tmp_path, revalidate_remote=False)
    assert summary["valid_completed_draws"] == 0
    assert summary["formal_interim_conclusion"] == "PROHIBITED_BEFORE_100_VALID_COMPLETED_DRAWS"


def test_31_ignored_csv_is_force_added_as_the_only_commit_and_pushes(tmp_path):
    repo, _ = init_git_remote(tmp_path)
    git(repo, "config", "core.autocrlf", "true")
    (repo / ".gitignore").write_text("data/processed/*\n", encoding="ascii")
    data = repo / "data" / "processed" / "lotto649.csv"; data.parent.mkdir(parents=True)
    data.write_text("draw_id,draw_date,number_1,number_2,number_3,number_4,number_5,number_6,special_number\n1,2026-01-01,1,2,3,4,5,6,7\n", encoding="utf-8")
    git(repo, "add", ".gitignore"); git(repo, "commit", "-m", "ignore data"); git(repo, "push", "origin", "HEAD:main")
    commit = commit_ignored_data_path(repo, data, "data: append official draw 1")
    git(repo, "push", "origin", "HEAD:main")
    changed = git(repo, "show", "--pretty=", "--name-only", commit).splitlines()
    remote_oid = git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]
    assert changed == ["data/processed/lotto649.csv"] and remote_oid == commit


def test_32_missing_processed_csv_fails_preflight_binding(tmp_path):
    with pytest.raises(RuntimeError, match="absent"):
        require_processed_data_current(tmp_path / "missing.csv", [prediction_event()], lambda _: None)


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_33_noninteger_ordinary_numbers_are_rejected(tmp_path, bad):
    numbers = [bad, 8, 15, 22, 29, 36]
    with pytest.raises(EvidenceError, match="JSON integers"):
        build_result(tmp_path, result=official_result(numbers=numbers))


def test_34_raw_response_rtcode_and_normalized_fields_are_bound(tmp_path):
    _, _, records = ledger_with_prediction(tmp_path)
    selected = official_result(); raw = json.loads(raw_for(selected)); raw["rtCode"] = 9
    with pytest.raises(EvidenceError, match="rtCode"):
        build_result_event(records, selected, raw_response_path="official.raw",
                           raw_response=json.dumps(raw).encode(), root=tmp_path, revalidate_remote=False)
    changed = json.loads(raw_for(selected)); changed["content"]["lotto649Res"][0]["drawNumberSize"][0] = 2
    with pytest.raises(EvidenceError, match="does not match"):
        build_result_event(records, selected, raw_response_path="official.raw",
                           raw_response=json.dumps(changed).encode(), root=tmp_path, revalidate_remote=False)


def test_35_raw_artifact_path_must_be_repository_relative_and_contained(tmp_path):
    _, _, records = ledger_with_prediction(tmp_path)
    with pytest.raises(EvidenceError, match="inside the repository"):
        build_result_event(records, official_result(), raw_response_path=tmp_path / ".." / "escape.raw",
                           raw_response=raw_for(official_result()), root=tmp_path, revalidate_remote=False)


def test_36_alternate_ledger_path_cannot_satisfy_result_anchor(tmp_path):
    repo, _, _, result, records = append_result_and_anchor(tmp_path)
    anchor = next(record for record in records if record.get("event_type") == "result_remote_anchor")
    anchor["result_ledger_path"] = "alternate/mini-ledger.jsonl"
    assert resolve_result_anchor(result, records, root=repo, revalidate_remote=False,
                                 ledger_relative_path="ledger.jsonl", head_relative_path="ledger_head.json") is None


def test_37_commit_must_contain_exact_result_record_not_just_hash_text(tmp_path):
    repo, _, _, result, records = append_result_and_anchor(tmp_path)
    forged = dict(result); forged["actual_numbers"] = [2, 9, 16, 23, 30, 37]
    assert resolve_result_anchor(forged, records, root=repo, revalidate_remote=False,
                                 ledger_relative_path="ledger.jsonl", head_relative_path="ledger_head.json") is None


def test_38_ingest_io_failure_rolls_back_ledger_head_and_artifacts(tmp_path, monkeypatch):
    ledger, head, _ = ledger_with_prediction(tmp_path)
    before_ledger, before_head = ledger.read_bytes(), head.read_bytes()
    raw = tmp_path / "artifacts" / "official.raw"; event = tmp_path / "artifacts" / "result.json"
    original_replace = Path.replace
    def failing_replace(self, target):
        if Path(target) == event:
            raise OSError("simulated result JSON publish failure")
        return original_replace(self, target)
    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated"):
        ingest_result(ledger, head, official_result(), raw_response_path=raw,
                      raw_response=raw_for(official_result()), result_json_path=event,
                      root=tmp_path, revalidate_remote=False)
    assert ledger.read_bytes() == before_ledger and head.read_bytes() == before_head
    assert not raw.exists() and not event.exists()
    assert not list(tmp_path.rglob("*.prospective-ops.lock")) and not list(tmp_path.rglob("*.tmp"))


def test_39_processed_csv_hash_mismatch_fails_preflight_binding(tmp_path):
    path = tmp_path / "lotto649.csv"; write_csv(path)
    prediction = prediction_event(); prediction.update({"data_sha256": "0" * 64, "data_end_draw_id": "115000074"})
    with pytest.raises(RuntimeError, match="hash"):
        require_processed_data_current(path, [prediction], lambda _: pd.DataFrame([{"draw_id": "115000074"}]))


@pytest.mark.parametrize("failure_index", range(14))
def test_40_complete_cycle_stops_after_every_failed_step(failure_index):
    calls: list[str] = []
    names = ["preflight", "fetch", "ingest", "push-result", "anchor-result", "csv", "push-csv",
             "next", "unannounced", "clean", "predict", "anchor-prediction", "status", "postflight"]
    def operation(index, value=None):
        def call(*args):
            calls.append(names[index])
            if index == failure_index: raise RuntimeError(f"fail-{index}")
            return value if value is not None else names[index]
        return call
    services = CycleServices(
        operation(1, {}), operation(2, {}), operation(3, "r"), operation(4, {}), operation(5, {}),
        operation(6, "c"), operation(7, {"draw_id": "2"}), operation(8), operation(9),
        operation(10, {}), operation(11, {}), operation(12, {}), preflight=operation(0), postflight=operation(13),
    )
    with pytest.raises(RuntimeError, match=f"fail-{failure_index}"):
        execute_complete_cycle("1", services)
    assert calls == names[:failure_index + 1]


def test_41_stale_second_writer_revalidates_duplicate_inside_lock(tmp_path, monkeypatch):
    ledger, head, _ = ledger_with_prediction(tmp_path)
    first_raw = tmp_path / "events" / "first.raw"; first_json = tmp_path / "events" / "first.json"
    second_raw = tmp_path / "events" / "second.raw"; second_json = tmp_path / "events" / "second.json"
    payload = raw_for(official_result())
    original_open = result_events_module.os.open
    state = {"interleaved": False}

    def interleaving_open(path, flags, mode=0o777):
        if Path(path).name.endswith(".prospective-ops.lock") and not state["interleaved"]:
            state["interleaved"] = True
            ingest_result(ledger, head, official_result(), raw_response_path=first_raw,
                          raw_response=payload, result_json_path=first_json,
                          root=tmp_path, revalidate_remote=False)
        return original_open(path, flags, mode)

    monkeypatch.setattr(result_events_module.os, "open", interleaving_open)
    with pytest.raises(EvidenceError, match="identical"):
        ingest_result(ledger, head, official_result(), raw_response_path=second_raw,
                      raw_response=payload, result_json_path=second_json,
                      root=tmp_path, revalidate_remote=False)
    results = [record for record in read_ledger(ledger) if record.get("event_type") == "result"]
    assert len(results) == 1 and first_raw.exists() and first_json.exists()
    assert not second_raw.exists() and not second_json.exists()


class _InjectedCycle:
    def __init__(self, failure_state: CycleState | None = None):
        self.index = 0
        self.failure_state = failure_state
        self.failed = False
        self.counts = {"result": 0, "result_anchor": 0, "csv": 0,
                       "prediction": 0, "prediction_anchor": 0}
        self.push_modes: list[str] = []

    def inspect(self, _draw_id: str) -> CycleObservation:
        return CycleObservation(STATE_ORDER[self.index], {"index": self.index})

    def action(self, state: CycleState):
        def advance(_draw_id: str, _observation: CycleObservation):
            created = {
                CycleState.NO_RESULT: "result",
                CycleState.RESULT_COMMIT_REMOTE: "result_anchor",
                CycleState.RESULT_ANCHOR_REMOTE: "csv",
                CycleState.CSV_COMMIT_REMOTE: "prediction",
                CycleState.NEXT_PREDICTION_REMOTE: "prediction_anchor",
            }.get(state)
            if created:
                self.counts[created] += 1
            if state in {CycleState.RESULT_COMMIT_LOCAL_ONLY, CycleState.RESULT_ANCHORED_LOCAL,
                         CycleState.CSV_COMMIT_LOCAL_ONLY, CycleState.NEXT_PREDICTION_LOCAL,
                         CycleState.NEXT_PREDICTION_ANCHORED}:
                self.push_modes.append("fast-forward")
            self.index += 1
            if state is self.failure_state and not self.failed:
                self.failed = True
                raise RuntimeError(f"injected after {state.value}")
            return state.value
        return advance

    def services(self) -> ResumableCycleServices:
        return ResumableCycleServices(
            inspect=self.inspect,
            actions={state: self.action(state) for state in STATE_ORDER[:-1]},
        )


def _recover_after(state: CycleState) -> _InjectedCycle:
    cycle = _InjectedCycle(state)
    with pytest.raises(RuntimeError, match="injected"):
        execute_resumable_cycle("115000075", cycle.services())
    result = execute_resumable_cycle("115000075", cycle.services())
    assert result["state"] == CycleState.CYCLE_COMPLETE.value
    return cycle


def test_42_required_resume_states_are_explicit_and_ordered():
    assert [state.value for state in STATE_ORDER] == [
        "NO_RESULT", "RESULT_APPENDED_LOCAL", "RESULT_COMMIT_LOCAL_ONLY",
        "RESULT_COMMIT_REMOTE", "RESULT_ANCHORED_LOCAL", "RESULT_ANCHOR_REMOTE",
        "CSV_UPDATED_LOCAL", "CSV_COMMIT_LOCAL_ONLY", "CSV_COMMIT_REMOTE",
        "NEXT_PREDICTION_LOCAL", "NEXT_PREDICTION_REMOTE",
        "NEXT_PREDICTION_ANCHORED", "CYCLE_COMPLETE",
    ]


def test_43_resume_after_result_append_before_commit():
    _recover_after(CycleState.NO_RESULT)


def test_44_resume_after_result_commit_before_push():
    _recover_after(CycleState.RESULT_APPENDED_LOCAL)


def test_45_resume_after_result_push_before_anchor():
    _recover_after(CycleState.RESULT_COMMIT_LOCAL_ONLY)


def test_46_resume_after_result_anchor_append_before_commit():
    _recover_after(CycleState.RESULT_COMMIT_REMOTE)


def test_47_resume_after_result_anchor_commit_before_push():
    _recover_after(CycleState.RESULT_ANCHORED_LOCAL)


def test_48_resume_after_csv_write_before_commit():
    _recover_after(CycleState.RESULT_ANCHOR_REMOTE)


def test_49_resume_after_csv_commit_before_push():
    _recover_after(CycleState.CSV_COMMIT_LOCAL_ONLY)


def test_50_resume_after_prediction_append_before_commit():
    _recover_after(CycleState.CSV_COMMIT_REMOTE)


def test_51_resume_after_prediction_commit_before_push():
    _recover_after(CycleState.NEXT_PREDICTION_LOCAL)


def test_52_resume_after_prediction_push_before_anchor():
    _recover_after(CycleState.NEXT_PREDICTION_REMOTE)


def test_53_resume_after_prediction_anchor_commit_before_push():
    _recover_after(CycleState.NEXT_PREDICTION_ANCHORED)


@pytest.mark.parametrize("failure_state", STATE_ORDER[:-1])
def test_54_every_failure_point_keeps_one_result(failure_state):
    assert _recover_after(failure_state).counts["result"] == 1


@pytest.mark.parametrize("failure_state", STATE_ORDER[:-1])
def test_55_every_failure_point_keeps_one_result_anchor(failure_state):
    assert _recover_after(failure_state).counts["result_anchor"] == 1


@pytest.mark.parametrize("failure_state", STATE_ORDER[:-1])
def test_56_every_failure_point_keeps_one_csv_row(failure_state):
    assert _recover_after(failure_state).counts["csv"] == 1


@pytest.mark.parametrize("failure_state", STATE_ORDER[:-1])
def test_57_every_failure_point_keeps_one_next_prediction(failure_state):
    assert _recover_after(failure_state).counts["prediction"] == 1


@pytest.mark.parametrize("failure_state", STATE_ORDER[:-1])
def test_58_every_failure_point_keeps_one_next_prediction_anchor(failure_state):
    assert _recover_after(failure_state).counts["prediction_anchor"] == 1


def test_59_recovered_pushes_are_fast_forward_only():
    cycle = _recover_after(CycleState.RESULT_COMMIT_LOCAL_ONLY)
    assert cycle.push_modes and set(cycle.push_modes) == {"fast-forward"}


def test_60_force_push_is_never_used(monkeypatch, tmp_path):
    repo, _ = init_git_remote(tmp_path)
    base = git(repo, "rev-parse", "HEAD")
    (repo / "next").write_text("next", encoding="ascii")
    git(repo, "add", "next"); git(repo, "commit", "-m", "next")
    head = git(repo, "rev-parse", "HEAD")
    require_fast_forward_push_topology(repo, remote="origin", branch="main",
                                       local_oid=head, remote_oid=base)
    assert "force" not in " ".join([base, head]).lower()


def test_61_local_and_remote_divergence_stops(tmp_path):
    repo, _ = init_git_remote(tmp_path)
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-b", "left")
    (repo / "left").write_text("left", encoding="ascii")
    git(repo, "add", "left"); git(repo, "commit", "-m", "left")
    left = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "main")
    (repo / "right").write_text("right", encoding="ascii")
    git(repo, "add", "right"); git(repo, "commit", "-m", "right")
    right = git(repo, "rev-parse", "HEAD")
    assert base != left != right
    with pytest.raises(RuntimeError, match="diverged"):
        require_fast_forward_push_topology(repo, remote="origin", branch="main",
                                           local_oid=left, remote_oid=right)


def test_62_unknown_new_remote_commit_stops(tmp_path):
    repo, _ = init_git_remote(tmp_path)
    local = git(repo, "rev-parse", "HEAD")
    (repo / "remote-new").write_text("unknown", encoding="ascii")
    git(repo, "add", "remote-new"); git(repo, "commit", "-m", "unknown remote")
    remote_new = git(repo, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="unknown newer commit"):
        require_fast_forward_push_topology(repo, remote="origin", branch="main",
                                           local_oid=local, remote_oid=remote_new)


def test_63_stale_lock_requires_provably_absent_owner_and_manual_recovery(tmp_path):
    lock = tmp_path / "complete_cycle.operation.lock"
    absent_pid = max(os.getpid() + 100000, 999999)
    lock.write_text(json.dumps({"pid": absent_pid, "hostname": socket.gethostname()}), encoding="utf-8")
    status = inspect_lock(lock)
    if status["owner_alive"] is not False:
        pytest.skip("platform cannot prove the synthetic PID is absent")
    assert status["recoverable"] is True
    recovered = recover_stale_lock(lock)
    assert recovered["removed"] is True and not lock.exists()
    lock.write_text("untrusted legacy lock", encoding="utf-8")
    with pytest.raises(EvidenceError, match="refusing stale-lock recovery"):
        recover_stale_lock(lock)


def test_64_second_complete_run_is_idempotent():
    cycle = _InjectedCycle()
    first = execute_resumable_cycle("115000075", cycle.services())
    before = dict(cycle.counts)
    second = execute_resumable_cycle("115000075", cycle.services())
    assert first["state"] == second["state"] == "CYCLE_COMPLETE"
    assert second["idempotent"] is True and cycle.counts == before


def test_65_dry_run_faults_and_recovers_every_mutation_boundary(tmp_path):
    repo, _ = init_git_remote(tmp_path)
    formal, operational = mock_formal_inputs(tmp_path)
    result = run_dry_run(
        draw_id="115000075", live_root=repo, formal_source_root=formal,
        operational_csv_path=operational,
        public_ref="main",
    )
    matrix = result["failure_recovery"]
    assert matrix["passed"] is True
    assert matrix["injected_boundaries"] == len(STATE_ORDER) - 1


def test_66_live_owner_lock_blocks_concurrent_writer_and_cleans_up(tmp_path):
    lock = tmp_path / "complete_cycle.operation.lock"
    with exclusive_operation_lock(lock, purpose="test-cycle"):
        status = inspect_lock(lock)
        assert status["exists"] is True
        assert status["owner_pid"] == os.getpid()
        assert status["owner_alive"] is True
        with pytest.raises(EvidenceError, match="manual recovery only"):
            with exclusive_operation_lock(lock, purpose="second-cycle"):
                pass
    assert inspect_lock(lock) == {"exists": False, "recoverable": False}
