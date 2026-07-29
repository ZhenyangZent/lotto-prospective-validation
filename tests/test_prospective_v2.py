"""V2 remote time anchor 的必要驗證。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prospective.canonical import append_record, initialize_ledger, read_ledger, sha256_file
from prospective.model import fit_predict as v1_fit_predict
from prospective_v2 import remote, workflow
from prospective_v2.config import EXPERIMENT_ID, LEDGER_PATH as V2_LEDGER_PATH
from prospective_v2.export_review_bundle import verify_zip


def _prediction(record_hash="p" * 64):
    return {"event_type":"prediction","event_id":"pred","experiment_id":EXPERIMENT_ID,
            "prediction_id":"v2-p1","target_draw_id":"1","target_draw_date":"2099-01-01",
            "parent_commit":"a"*40,"record_hash":record_hash,"late_prediction":False}


def _anchor(prediction, *, before=True, oid="b"*40, prediction_commit=None):
    commit=prediction_commit or oid
    return {"event_type":"remote_anchor","event_id":"anchor","experiment_id":EXPERIMENT_ID,
            "prediction_id":prediction["prediction_id"],"prediction_record_hash":prediction["record_hash"],
            "prediction_commit":commit,"remote_ref_oid":oid,"remote_name":"origin","remote_branch":"master",
            "verified_before_draw":before,"official_draw_status_at_verification":"NOT_ANNOUNCED"}


def _precheck(tmp_path, status="NOT_ANNOUNCED"):
    metadata=tmp_path/"precheck.json"; raw=tmp_path/"precheck.raw"; raw.write_bytes(b"raw")
    return {"metadata":{"target_draw_status":status,"raw_response_sha256":"0"*64},
            "metadata_path":metadata,"raw_path":raw}


def test_01_origin_present_but_push_failure_is_not_confirmed(monkeypatch):
    monkeypatch.setattr(workflow,"git",lambda *args,**kwargs: SimpleNamespace(returncode=1,stdout="",stderr="failed"))
    assert workflow.push_head("origin") is False


def test_02_composable_commit_url_alone_is_not_confirmation():
    prediction=_prediction(); prediction["remote_commit_url"]="https://github.com/x/y/commit/abc"
    assert remote.resolve_remote_anchor(prediction["prediction_id"],[prediction],revalidate_remote=False) is None


def test_03_ls_remote_oid_mismatch_rejects_anchor(monkeypatch):
    monkeypatch.setattr(remote,"ls_remote_oid",lambda *args,**kwargs:"c"*40)
    with pytest.raises(RuntimeError,match="OID 不符"): remote.require_remote_oid(".","origin","refs/heads/master","b"*40)


def test_04_ls_remote_oid_match_accepts_anchor(monkeypatch):
    monkeypatch.setattr(remote,"ls_remote_oid",lambda *args,**kwargs:"b"*40)
    assert remote.require_remote_oid(".","origin","refs/heads/master","b"*40)=="b"*40


def test_05_prediction_records_parent_not_fictional_self_commit(tmp_path):
    payload={"prediction_id":"p","parent_commit":"a"*40}; event=workflow.build_prediction_event(payload,_precheck(tmp_path))
    assert event["parent_commit"]=="a"*40 and "prediction_commit" not in event


def test_06_remote_anchor_references_prediction_commit(tmp_path):
    prediction=_prediction(); commit="b"*40
    anchor=workflow.build_remote_anchor_event(prediction,commit,remote="origin",branch="master",remote_oid=commit,
                                               repository="https://github.com/x/y.git",verification_precheck=_precheck(tmp_path))
    assert anchor["prediction_commit"]==anchor["remote_ref_oid"]==commit


def test_07_wrong_prediction_hash_makes_ledger_validation_fail(tmp_path,monkeypatch):
    ledger=tmp_path/"ledger.jsonl"; head=tmp_path/"head.json"; initialize_ledger(ledger,head)
    prediction=append_record(ledger,head,{key:value for key,value in _prediction().items() if key!="record_hash"})
    anchor=_anchor(prediction); anchor["prediction_record_hash"]="wrong"; append_record(ledger,head,anchor)
    monkeypatch.setattr(workflow,"LEDGER_PATH",ledger); monkeypatch.setattr(workflow,"LEDGER_HEAD_PATH",head)
    with pytest.raises(ValueError,match="錯誤 prediction hash"): workflow.verify_v2_ledger()


def test_08_anchor_after_draw_does_not_count():
    prediction=_prediction(); anchor=_anchor(prediction,before=False)
    assert remote.resolve_remote_anchor("v2-p1",[prediction,anchor],revalidate_remote=False) is None


def test_09_prediction_without_anchor_does_not_count():
    prediction=_prediction(); assert remote.resolve_remote_anchor("v2-p1",[prediction],revalidate_remote=False) is None


def test_10_deleted_remote_branch_loses_remote_evidence(monkeypatch):
    prediction=_prediction(); anchor=_anchor(prediction)
    monkeypatch.setattr(remote,"ls_remote_oid",lambda *args,**kwargs:None)
    assert remote.resolve_remote_anchor("v2-p1",[prediction,anchor],root=".",revalidate_remote=True) is None


def test_11_v1_ledger_is_not_modified_by_v2_test(tmp_path):
    root=Path(__file__).resolve().parents[1]; v1=root/"prospective_validation"/"ledger.jsonl"; before=sha256_file(v1)
    ledger=tmp_path/"ledger.jsonl"; head=tmp_path/"head.json"; initialize_ledger(ledger,head); append_record(ledger,head,{"event_id":"v2","experiment_id":EXPERIMENT_ID})
    assert sha256_file(v1)==before


def test_12_v2_uses_a_new_ledger():
    root=Path(__file__).resolve().parents[1]
    assert V2_LEDGER_PATH != root/"prospective_validation"/"ledger.jsonl"


def test_13_v2_uses_the_exact_v1_model_callable():
    assert workflow.fit_predict is v1_fit_predict


def test_14_cross_platform_tolerance_is_one_e_minus_twelve():
    values=np.linspace(.1,.14,49); values=values/values.sum()*6
    order=np.arange(1,50)[np.lexsort((np.arange(1,50),-values))]
    expected={"probabilities_1_to_49":values.tolist(),"top6":order[:6].tolist(),"top10":order[:10].tolist(),"top12":order[:12].tolist()}
    actual={"probabilities":values.copy(),"top6":expected["top6"],"top10":expected["top10"],"top12":expected["top12"]}
    actual["probabilities"][0]+=5e-13; actual["probabilities"][1]-=5e-13; workflow.compare_reproduction(expected,actual,atol=1e-12)
    actual["probabilities"][0]+=2e-12; actual["probabilities"][1]-=2e-12
    with pytest.raises(AssertionError): workflow.compare_reproduction(expected,actual,atol=1e-12)


def _valid_zip(tmp_path):
    import csv,hashlib,zipfile
    path=tmp_path/"v2.zip"; content=b"payload"; manifest=[{"relative_path":"payload.txt","bytes":len(content),"sha256":hashlib.sha256(content).hexdigest(),"description":"x"}]
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("payload.txt",content); archive.writestr("file_manifest.json",json.dumps(manifest))
        archive.writestr("file_manifest.csv","x"); archive.writestr("zip_validation.json",json.dumps({"manifest_payload_count":1,"zip_entry_count":4}))
    return path


def test_15_bundle_entry_count_matches_report(tmp_path):
    result=verify_zip(_valid_zip(tmp_path)); assert result["zip_entry_count"]==result["manifest_payload_count"]+3


def test_16_zip_manifest_entries_and_metadata_counts_are_correct(tmp_path):
    result=verify_zip(_valid_zip(tmp_path)); assert result=={"validation_passed":True,"errors":[],"manifest_payload_count":1,"zip_entry_count":4}


def test_17_git_bundle_verify_passes(tmp_path):
    repo=tmp_path/"repo"; repo.mkdir(); subprocess.run(["git","init"],cwd=repo,check=True,capture_output=True)
    subprocess.run(["git","config","user.email","test@example.invalid"],cwd=repo,check=True); subprocess.run(["git","config","user.name","Test"],cwd=repo,check=True)
    (repo/"x").write_text("x"); subprocess.run(["git","add","x"],cwd=repo,check=True); subprocess.run(["git","commit","-m","x"],cwd=repo,check=True,capture_output=True)
    bundle=tmp_path/"history.bundle"; subprocess.run(["git","bundle","create",str(bundle),"--all"],cwd=repo,check=True)
    assert subprocess.run(["git","bundle","verify",str(bundle)],cwd=repo,capture_output=True).returncode==0


def test_18_remote_commit_can_be_fetched(tmp_path):
    source=tmp_path/"source"; bare=tmp_path/"remote.git"; clone=tmp_path/"clone"; source.mkdir()
    subprocess.run(["git","init"],cwd=source,check=True,capture_output=True); subprocess.run(["git","config","user.email","test@example.invalid"],cwd=source,check=True); subprocess.run(["git","config","user.name","Test"],cwd=source,check=True)
    (source/"x").write_text("x"); subprocess.run(["git","add","x"],cwd=source,check=True); subprocess.run(["git","commit","-m","x"],cwd=source,check=True,capture_output=True)
    subprocess.run(["git","init","--bare",str(bare)],check=True,capture_output=True); subprocess.run(["git","remote","add","origin",str(bare)],cwd=source,check=True)
    subprocess.run(["git","push","origin","HEAD"],cwd=source,check=True,capture_output=True); expected=subprocess.run(["git","rev-parse","HEAD"],cwd=source,text=True,capture_output=True,check=True).stdout.strip()
    subprocess.run(["git","clone",str(bare),str(clone)],check=True,capture_output=True); assert subprocess.run(["git","cat-file","-e",expected],cwd=clone).returncode==0
