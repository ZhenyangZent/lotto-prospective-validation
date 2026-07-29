"""第三階段前瞻鎖模驗證的最低必要防護測試。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prospective import workflow
from prospective.canonical import (ZERO_HASH, aggregate_file_hash, append_record, canonical_bytes,
                                   initialize_ledger, sha256_file, verify_ledger)
from prospective.gitops import assert_clean
from prospective.metrics import conditional_monte_carlo, score_prediction
from prospective.model import assert_pre_target_data, fit_predict
from prospective.official import validate_official_source


@pytest.fixture
def draws() -> pd.DataFrame:
    """審查 ZIP 中可獨立執行，不依賴專案外部 conftest。"""
    rows = []
    for index in range(80):
        numbers = sorted({((index * 7 + offset * 8) % 49) + 1 for offset in range(6)})
        while len(numbers) < 6:
            candidate = (numbers[-1] % 49) + 1
            if candidate not in numbers: numbers.append(candidate)
        rows.append({"draw_id": str(100000000 + index),
                     "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index * 3),
                     **{f"number_{i + 1}": value for i, value in enumerate(sorted(numbers))}})
    return pd.DataFrame(rows)


def _event(event_id: str) -> dict:
    return {"event_id": event_id, "event_type": "test", "payload": event_id}


@pytest.fixture
def isolated_workflow(tmp_path, monkeypatch, draws):
    state = tmp_path / "prospective_validation"; predictions = state / "predictions"; predictions.mkdir(parents=True)
    data_path = tmp_path / "lotto649.csv"; draws.to_csv(data_path, index=False)
    ledger = state / "ledger.jsonl"; head = state / "ledger_head.json"; initialize_ledger(ledger, head)
    monkeypatch.setattr(workflow, "STATE_DIR", state); monkeypatch.setattr(workflow, "DATA_PATH", data_path)
    monkeypatch.setattr(workflow, "LEDGER_PATH", ledger); monkeypatch.setattr(workflow, "LEDGER_HEAD_PATH", head)
    monkeypatch.setattr(workflow, "verify_frozen_integrity", lambda: {"source_code_sha256": "c" * 64, "config_sha256": "f" * 64})
    monkeypatch.setattr(workflow, "fit_predict", lambda data: {
        "probabilities": np.full(49, 6 / 49), "top6": [1, 2, 3, 4, 5, 6],
        "top10": list(range(1, 11)), "top12": list(range(1, 13)),
    })
    monkeypatch.setattr(workflow, "save_raw_response", lambda payload, stem: _save_json(state / f"{stem}.json", payload))
    return {"state": state, "data": data_path, "ledger": ledger, "head": head,
            "draw_id": "999999999", "draw_date": "2021-01-01"}


def _save_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8"); return path


def _official_payload(draw_id: str | None = None, draw_date: str = "2021-01-01") -> dict:
    rows = [] if draw_id is None else [{"period": int(draw_id), "lotteryDate": draw_date,
                                        "drawNumberSize": [1, 2, 3, 4, 5, 6, 7]}]
    return {"rtCode": 0, "content": {"lotto649Res": rows}}


def test_01_unannounced_draw_can_be_predicted(isolated_workflow):
    ctx = isolated_workflow
    result = workflow.predict_next(ctx["draw_id"], ctx["draw_date"], enforce_git=False,
                                   official_fetcher=lambda *_: _official_payload())
    assert result["target_draw_id"] == ctx["draw_id"] and Path(result["prediction_json"]).exists()


def test_02_announced_draw_is_rejected(isolated_workflow):
    ctx = isolated_workflow
    with pytest.raises(ValueError, match="已公布"):
        workflow.predict_next(ctx["draw_id"], ctx["draw_date"], enforce_git=False,
                              official_fetcher=lambda *_: _official_payload(ctx["draw_id"]))


def test_03_duplicate_draw_id_is_rejected(isolated_workflow):
    ctx = isolated_workflow; fetch = lambda *_: _official_payload()
    workflow.predict_next(ctx["draw_id"], ctx["draw_date"], enforce_git=False, official_fetcher=fetch)
    with pytest.raises(ValueError, match="已有預測"):
        workflow.predict_next(ctx["draw_id"], ctx["draw_date"], enforce_git=False, official_fetcher=fetch)


def test_04_result_append_does_not_modify_prediction(isolated_workflow):
    ctx = isolated_workflow
    workflow.predict_next(ctx["draw_id"], ctx["draw_date"], enforce_git=False, official_fetcher=lambda *_: _official_payload())
    before = json.loads(ctx["ledger"].read_text(encoding="utf-8").splitlines()[0])
    workflow.ingest_result(ctx["draw_id"], enforce_git=False,
                           official_fetcher=lambda *_: _official_payload(ctx["draw_id"]))
    after = json.loads(ctx["ledger"].read_text(encoding="utf-8").splitlines()[0])
    assert before == after and len(ctx["ledger"].read_text(encoding="utf-8").splitlines()) == 2


def test_05_single_character_ledger_tamper_fails(tmp_path):
    ledger=tmp_path/"ledger.jsonl"; head=tmp_path/"head.json"; initialize_ledger(ledger,head); append_record(ledger,head,_event("a"))
    ledger.write_text(ledger.read_text(encoding="utf-8").replace('"payload":"a"','"payload":"b"'),encoding="utf-8")
    with pytest.raises(ValueError): verify_ledger(ledger,head)


def test_06_deleting_middle_record_fails(tmp_path):
    ledger=tmp_path/"ledger.jsonl"; head=tmp_path/"head.json"; initialize_ledger(ledger,head)
    for value in "abc": append_record(ledger,head,_event(value))
    lines=ledger.read_text(encoding="utf-8").splitlines(); ledger.write_text("\n".join([lines[0],lines[2]])+"\n",encoding="utf-8")
    with pytest.raises(ValueError): verify_ledger(ledger,head)


def test_07_reordering_records_fails(tmp_path):
    ledger=tmp_path/"ledger.jsonl"; head=tmp_path/"head.json"; initialize_ledger(ledger,head)
    append_record(ledger,head,_event("a")); append_record(ledger,head,_event("b"))
    lines=ledger.read_text(encoding="utf-8").splitlines(); ledger.write_text("\n".join(reversed(lines))+"\n",encoding="utf-8")
    with pytest.raises(ValueError): verify_ledger(ledger,head)


def test_08_previous_record_hash_chain_is_correct(tmp_path):
    ledger=tmp_path/"ledger.jsonl"; head=tmp_path/"head.json"; initialize_ledger(ledger,head)
    first=append_record(ledger,head,_event("a")); second=append_record(ledger,head,_event("b"))
    assert first["previous_record_hash"] == ZERO_HASH and second["previous_record_hash"] == first["record_hash"]


def _integrity_fixture(tmp_path, monkeypatch):
    code=tmp_path/"code.py"; code.write_text("x=1\n",encoding="utf-8")
    config=tmp_path/"config.json"; config.write_text(json.dumps(workflow.FROZEN_CONFIG,sort_keys=True),encoding="utf-8")
    code_hash, files=aggregate_file_hash(tmp_path,["code.py"])
    manifest=tmp_path/"manifest.json"; manifest.write_text(json.dumps({"source_code_sha256":code_hash,"source_files":files,"config_sha256":sha256_file(config)}),encoding="utf-8")
    monkeypatch.setattr(workflow,"ROOT",tmp_path); monkeypatch.setattr(workflow,"SOURCE_FILES",["code.py"])
    monkeypatch.setattr(workflow,"FROZEN_CONFIG_PATH",config); monkeypatch.setattr(workflow,"FROZEN_MANIFEST_PATH",manifest)
    return code,config


def test_09_source_hash_change_rejects_prediction_integrity(tmp_path, monkeypatch):
    code,_=_integrity_fixture(tmp_path,monkeypatch); code.write_text("x=2\n",encoding="utf-8")
    with pytest.raises(RuntimeError,match="程式 SHA-256"): workflow.verify_frozen_integrity()


def test_10_config_change_rejects_prediction_integrity(tmp_path, monkeypatch):
    _,config=_integrity_fixture(tmp_path,monkeypatch); config.write_text("{}",encoding="utf-8")
    with pytest.raises(RuntimeError,match="設定 SHA-256"): workflow.verify_frozen_integrity()


def test_11_dirty_git_worktree_is_rejected(tmp_path):
    subprocess.run(["git","init"],cwd=tmp_path,check=True,capture_output=True)
    subprocess.run(["git","config","user.email","test@example.invalid"],cwd=tmp_path,check=True)
    subprocess.run(["git","config","user.name","Test"],cwd=tmp_path,check=True)
    item=tmp_path/"tracked.txt"; item.write_text("clean",encoding="utf-8")
    subprocess.run(["git","add","tracked.txt"],cwd=tmp_path,check=True); subprocess.run(["git","commit","-m","base"],cwd=tmp_path,check=True,capture_output=True)
    item.write_text("dirty",encoding="utf-8")
    with pytest.raises(RuntimeError,match="不乾淨"): assert_clean(tmp_path)


def test_12_model_rejects_target_or_future_data(draws):
    target_date=pd.Timestamp(draws.iloc[-1]["draw_date"]).date().isoformat()
    with pytest.raises(ValueError,match="目標日期"): assert_pre_target_data(draws,"999",target_date)


def test_13_probability_sum_is_six(draws):
    result=fit_predict(draws); assert sum(result["probabilities"]) == pytest.approx(6,abs=1e-10)


def test_14_all_probabilities_are_bounded(draws):
    values=fit_predict(draws)["probabilities"]; assert np.all((values>=0)&(values<=1))


def test_15_same_data_and_version_are_deterministic(draws):
    first=fit_predict(draws); second=fit_predict(draws)
    np.testing.assert_array_equal(first["probabilities"],second["probabilities"]); assert first["top6"]==second["top6"]


def test_16_nonofficial_result_source_is_rejected():
    with pytest.raises(ValueError,match="不是允許"): validate_official_source("https://example.com/fake")


def test_17_brier_matches_hand_calculation():
    p=np.full(49,6/49); result=score_prediction(p,[1,2,3,4,5,6])
    target=np.r_[np.ones(6),np.zeros(43)]; assert result["brier"]==pytest.approx(np.mean((p-target)**2))


def test_18_log_loss_matches_hand_calculation():
    p=np.full(49,6/49); result=score_prediction(p,[1,2,3,4,5,6])
    expected=-(6*np.log(6/49)+43*np.log(43/49))/49; assert result["log_loss"]==pytest.approx(expected)


def test_19_top6_hits_match_hand_calculation():
    p=np.linspace(1,49,49); result=score_prediction(p,[44,45,46,47,48,49]); assert result["hits_top6"]==6


def test_20_conditional_monte_carlo_is_reproducible():
    p=np.tile(np.linspace(.05,.2,49),(3,1)); p=p/p.sum(axis=1)[:,None]*6
    actual=np.zeros((3,49),dtype=np.int8); actual[:,:6]=1
    assert conditional_monte_carlo(p,actual,simulations=100,seed=7,batch_size=20)==conditional_monte_carlo(p,actual,simulations=100,seed=7,batch_size=20)


def test_21_review_bundle_manifest_detects_missing_file(tmp_path):
    archive=tmp_path/"bundle.zip"; manifest=[{"relative_path":"missing.txt","bytes":1,"sha256":hashlib.sha256(b"x").hexdigest(),"description":"x"}]
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("file_manifest.json",json.dumps(manifest)); z.writestr("file_manifest.csv","x"); z.writestr("bundle_validation.txt","x")
    result=workflow.verify_bundle(archive); assert not result["validation_passed"] and result["manifest_file_count"]==1


def test_22_zip_per_file_sha256_validation_passes(tmp_path):
    archive=tmp_path/"bundle.zip"; content=b"verified"; manifest=[{"relative_path":"payload.txt","bytes":len(content),"sha256":hashlib.sha256(content).hexdigest(),"description":"payload"}]
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("payload.txt",content); z.writestr("file_manifest.json",json.dumps(manifest)); z.writestr("file_manifest.csv","x"); z.writestr("bundle_validation.txt","x")
    assert workflow.verify_bundle(archive)["validation_passed"]
