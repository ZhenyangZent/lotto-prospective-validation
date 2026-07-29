"""V2 prediction → remote OID verification → remote_anchor 工作流程。"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from next_draw_predictor_audited import load_lotto_data
from prospective.canonical import (aggregate_file_hash, append_record, canonical_bytes,
                                   initialize_ledger, read_ledger, sha256_file, verify_ledger)
from prospective.gitops import assert_clean
from prospective.model import assert_pre_target_data, fit_predict
from prospective.official import fetch_next_draw
from .config import (DATA_PATH, EXPERIMENT_ID, EXPERIMENT_VERSION, FROZEN_CONFIG,
                     FROZEN_CONFIG_PATH, FROZEN_MANIFEST_PATH, LEDGER_HEAD_PATH,
                     LEDGER_PATH, MODEL_VERSION, REMOTE_NAME, ROOT, SOURCE_FILES,
                     STATE_DIR, TAG_NAME, TIMEZONE)
from .official import fetch_official_precheck
from .remote import (branch_name, commit_url, git, ls_remote_oid, remote_repository,
                     require_remote_oid, resolve_remote_anchor)


def now_iso() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def current_commit(root: str | Path = ROOT) -> str:
    return git(root, "rev-parse", "HEAD").stdout.strip()


def commit_paths(paths: list[Path], message: str) -> str:
    relative = [str(path.relative_to(ROOT)) for path in paths]
    git(ROOT, "add", "--", *relative); git(ROOT, "commit", "-m", message)
    return current_commit()


def push_head(remote: str = REMOTE_NAME) -> bool:
    return git(ROOT, "push", remote, "HEAD", check=False).returncode == 0


def preregistration(first_draw: dict[str, str]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
        "registered_at": now_iso(), "first_prospective_draw": first_draw,
        "research_question": "與V1相同固定模型是否在未來資料的完整49號機率品質優於ExactUniform？",
        "model_and_parameters": FROZEN_CONFIG["model"],
        "model_change_from_v1": "NONE",
        "v2_change_scope": "Evidence preservation, actual remote OID verification, and cross-platform reproduction tolerance only.",
        "evidence_protocol": FROZEN_CONFIG["evidence_protocol"],
        "official_source": "https://www.taiwanlottery.com/lotto/result/lotto649/",
        "planned_sample_size": 200, "stages": FROZEN_CONFIG["stages"],
        "primary_metrics": FROZEN_CONFIG["primary_metrics"],
        "secondary_metrics": FROZEN_CONFIG["secondary_metrics"],
        "success_criteria": FROZEN_CONFIG["success"],
        "monte_carlo": FROZEN_CONFIG["conditional_monte_carlo"],
        "valid_prediction_definition": [
            "prediction event created before official result", "prediction commit pushed before draw",
            "git ls-remote OID matched prediction commit", "valid remote_anchor references prediction hash and commit",
            "official status remained NOT_ANNOUNCED at verification", "no late_prediction",
            "code/config/data hashes match V2 freeze",
        ],
        "interim_rule": "Every 20 draws descriptive only; no formal decision before 100 valid anchored draws.",
        "missing_draw_policy": "Unanchored or late predictions remain invalid forever and are never backfilled.",
        "bug_policy": "Close the version and restart from zero; never rewrite this ledger.",
    }


def preregistration_md(reg: dict[str, Any]) -> str:
    return f"""# 台灣大樂透前瞻驗證 V2 預註冊

- Experiment：`{EXPERIMENT_ID}`
- 模型：`{MODEL_VERSION}`（與 V1 完全相同）
- 第一目標期：{reg['first_prospective_draw']['draw_id']}（{reg['first_prospective_draw']['draw_date']}）
- 凍結時間：{reg['registered_at']}

V2 不重新選模型或參數。特徵、C=0.01、L2/lbfgs、seed=20260729、EWMA alpha=0.06、transition alpha=30、長期先驗12、收縮0.10、capped-simplex與評估/成功標準均沿用 V1。

唯一設計修正是兩階段 Git 證據：prediction 事件只記 parent commit 與 `PENDING_REMOTE_ANCHOR`；prediction commit push 後，必須以 `git ls-remote` 得到與 prediction commit 完全相同的 branch OID，並在官方結果仍未公布時追加 remote_anchor。status 只解析合法 anchor，不信任 prediction/result 物件內的 boolean。

跨平台重現要求 Top-6/10/12與完整排名一致、機率總和誤差不超過1e-12、逐號機率 `atol<=1e-12`。相同正式環境另保留嚴格重現。

未錨定、遠端 branch 消失、OID無法證實、開獎後驗證或 late prediction 均不計正式樣本，且不得事後補算。
"""


def freeze_experiment(first_draw: dict[str, str] | None = None) -> dict[str, Any]:
    if FROZEN_MANIFEST_PATH.exists():
        raise FileExistsError("V2 已凍結，不得覆寫")
    STATE_DIR.mkdir(parents=True, exist_ok=True); (STATE_DIR / "predictions").mkdir(exist_ok=True)
    if first_draw is None: first_draw, _ = fetch_next_draw()
    reg = preregistration(first_draw)
    write_json(FROZEN_CONFIG_PATH, FROZEN_CONFIG); write_json(STATE_DIR / "preregistration.json", reg)
    (STATE_DIR / "preregistration.md").write_text(preregistration_md(reg), encoding="utf-8")
    initialize_ledger(LEDGER_PATH, LEDGER_HEAD_PATH)
    code_hash, files = aggregate_file_hash(ROOT, SOURCE_FILES)
    data = load_lotto_data(DATA_PATH)
    manifest = {"experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
                "frozen_at": reg["registered_at"], "first_prospective_draw": first_draw,
                "source_code_sha256": code_hash, "source_files": files,
                "config_sha256": sha256_file(FROZEN_CONFIG_PATH), "data_sha256_at_freeze": sha256_file(DATA_PATH),
                "data_end_draw_id": str(data.iloc[-1]["draw_id"]),
                "data_end_date": pd.Timestamp(data.iloc[-1]["draw_date"]).date().isoformat()}
    write_json(FROZEN_MANIFEST_PATH, manifest)
    return manifest


def verify_frozen_integrity() -> dict[str, Any]:
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    code_hash, files = aggregate_file_hash(ROOT, SOURCE_FILES)
    if code_hash != manifest["source_code_sha256"] or files != manifest["source_files"]:
        raise RuntimeError("V2 程式 SHA-256 與凍結版本不符")
    if sha256_file(FROZEN_CONFIG_PATH) != manifest["config_sha256"]:
        raise RuntimeError("V2 config SHA-256 與凍結版本不符")
    if json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8")) != FROZEN_CONFIG:
        raise RuntimeError("V2 frozen config 內容不符")
    return manifest


def verify_v2_ledger(revalidate_remote: bool = False) -> dict[str, Any]:
    base = verify_ledger(LEDGER_PATH, LEDGER_HEAD_PATH); records = read_ledger(LEDGER_PATH)
    predictions: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("V2 ledger 混入其他 experiment")
        if record.get("event_type") == "prediction":
            prediction_id = str(record.get("prediction_id"))
            if prediction_id in predictions: raise ValueError("V2 prediction_id 重複")
            if "prediction_commit" in record or not record.get("parent_commit"):
                raise ValueError("prediction 事件必須只保存 parent_commit")
            predictions[prediction_id] = record
        elif record.get("event_type") == "remote_anchor":
            prediction = predictions.get(str(record.get("prediction_id")))
            if not prediction or record.get("prediction_record_hash") != prediction.get("record_hash"):
                raise ValueError("remote_anchor 引用錯誤 prediction hash")
            if record.get("prediction_commit") != record.get("remote_ref_oid"):
                raise ValueError("remote_anchor 的 prediction commit 與 verified OID 不一致")
    if revalidate_remote:
        for prediction_id in predictions:
            resolve_remote_anchor(prediction_id, records, revalidate_remote=True)
    return base


def prediction_payload(draw_id: str, draw_date: str, prediction: dict[str, Any], data: pd.DataFrame,
                       manifest: dict[str, Any], parent_commit: str, created_at: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
        "model_version": MODEL_VERSION, "prediction_id": f"{EXPERIMENT_ID}-{draw_id}",
        "target_draw_id": str(draw_id), "target_draw_date": draw_date,
        "prediction_created_at": created_at, "timezone": TIMEZONE,
        "data_end_draw_id": str(data.iloc[-1]["draw_id"]),
        "data_end_date": pd.Timestamp(data.iloc[-1]["draw_date"]).date().isoformat(),
        "data_sha256": sha256_file(DATA_PATH), "source_code_sha256": manifest["source_code_sha256"],
        "config_sha256": manifest["config_sha256"], "parent_commit": parent_commit,
        "top6": prediction["top6"], "top10": prediction["top10"], "top12": prediction["top12"],
        "probabilities_1_to_49": [float(value) for value in prediction["probabilities"]],
        "probability_sum": float(np.sum(prediction["probabilities"])), "random_seed": 20260729,
    }


def build_prediction_event(payload: dict[str, Any], precheck: dict[str, Any]) -> dict[str, Any]:
    return {"event_type": "prediction", "event_id": str(uuid.uuid4()), **payload,
            "prediction_payload_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
            "remote_status": "PENDING_REMOTE_ANCHOR", "remote_commit_confirmed": False,
            "official_precheck_metadata": Path(precheck["metadata_path"]).name,
            "official_precheck_raw": Path(precheck["raw_path"]).name,
            "official_precheck_raw_sha256": precheck["metadata"]["raw_response_sha256"],
            "official_draw_status_at_prediction": precheck["metadata"]["target_draw_status"],
            "late_prediction": False}


def build_remote_anchor_event(prediction: dict[str, Any], prediction_commit: str, *,
                              remote: str, branch: str, remote_oid: str,
                              repository: str, verification_precheck: dict[str, Any],
                              verified_at: str | None = None) -> dict[str, Any]:
    if remote_oid != prediction_commit:
        raise RuntimeError("不得為 OID 不一致的 prediction 建立 anchor")
    status = verification_precheck["metadata"]["target_draw_status"]
    return {"event_type": "remote_anchor", "event_id": str(uuid.uuid4()),
            "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
            "target_draw_id": prediction["target_draw_id"], "target_draw_date": prediction["target_draw_date"],
            "prediction_id": prediction["prediction_id"], "prediction_record_hash": prediction["record_hash"],
            "prediction_commit": prediction_commit, "remote_name": remote,
            "remote_repository": repository, "remote_branch": branch, "remote_ref_oid": remote_oid,
            "remote_commit_url": commit_url(repository, prediction_commit),
            "remote_verified_at": verified_at or now_iso(), "verification_method": "git-ls-remote",
            "verified_before_draw": status == "NOT_ANNOUNCED", "official_draw_status_at_verification": status,
            "verification_precheck_metadata": Path(verification_precheck["metadata_path"]).name,
            "verification_precheck_raw": Path(verification_precheck["raw_path"]).name,
            "verification_precheck_raw_sha256": verification_precheck["metadata"]["raw_response_sha256"]}


def create_prediction(draw_id: str, draw_date: str, *, requester: Any = None,
                      auto_git: bool = True) -> dict[str, Any]:
    manifest = verify_frozen_integrity()
    if auto_git: assert_clean(ROOT)
    verify_v2_ledger(); records = read_ledger(LEDGER_PATH)
    if any(r.get("event_type") == "prediction" and str(r.get("target_draw_id")) == str(draw_id) for r in records):
        raise ValueError("V2 已存在相同期別 prediction")
    kwargs = {} if requester is None else {"requester": requester}
    precheck = fetch_official_precheck(draw_id, **kwargs)
    if precheck["metadata"]["target_draw_status"] != "NOT_ANNOUNCED":
        raise ValueError("官方已公布目標期，不得建立 V2 prediction")
    data = load_lotto_data(DATA_PATH); assert_pre_target_data(data, str(draw_id), draw_date)
    model_output = fit_predict(data); parent = current_commit() if auto_git else "TEST-PARENT"
    payload = prediction_payload(str(draw_id), draw_date, model_output, data, manifest, parent, now_iso())
    prediction = append_record(LEDGER_PATH, LEDGER_HEAD_PATH, build_prediction_event(payload, precheck))
    prediction_path = STATE_DIR / "predictions" / f"prediction-{draw_id}.json"; write_json(prediction_path, prediction)
    probability_path = STATE_DIR / "predictions" / f"prediction-{draw_id}-probabilities.csv"
    with probability_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["number", "probability", "rank", "is_top6"]); writer.writeheader()
        order = sorted(range(1, 50), key=lambda number: (-prediction["probabilities_1_to_49"][number - 1], number))
        ranks = {number: rank for rank, number in enumerate(order, 1)}
        for number, probability in enumerate(prediction["probabilities_1_to_49"], 1):
            writer.writerow({"number": number, "probability": format(probability, ".17g"),
                             "rank": ranks[number], "is_top6": number in prediction["top6"]})
    if not auto_git:
        return {"prediction": prediction, "prediction_commit": None, "remote_anchor": None}
    prediction_commit = commit_paths([LEDGER_PATH, LEDGER_HEAD_PATH, prediction_path, probability_path,
                                      precheck["metadata_path"], precheck["raw_path"]],
                                     f"prospective-v2: prediction for draw {draw_id}")
    if not push_head():
        return {"prediction": prediction, "prediction_commit": prediction_commit, "remote_anchor": None,
                "remote_confirmed": False}
    branch = branch_name(); ref = f"refs/heads/{branch}"
    remote_oid = require_remote_oid(ROOT, REMOTE_NAME, ref, prediction_commit)
    verification = fetch_official_precheck(draw_id, **kwargs)
    if verification["metadata"]["target_draw_status"] != "NOT_ANNOUNCED":
        return {"prediction": prediction, "prediction_commit": prediction_commit, "remote_anchor": None,
                "remote_confirmed": False, "reason": "OFFICIAL_RESULT_ANNOUNCED_BEFORE_ANCHOR"}
    repository = remote_repository(); anchor_event = build_remote_anchor_event(
        prediction, prediction_commit, remote=REMOTE_NAME, branch=branch, remote_oid=remote_oid,
        repository=repository, verification_precheck=verification)
    anchor = append_record(LEDGER_PATH, LEDGER_HEAD_PATH, anchor_event)
    anchor_path = STATE_DIR / "predictions" / f"remote-anchor-{draw_id}.json"; write_json(anchor_path, anchor)
    anchor_commit = commit_paths([LEDGER_PATH, LEDGER_HEAD_PATH, anchor_path,
                                  verification["metadata_path"], verification["raw_path"]],
                                 f"prospective-v2: anchor prediction for draw {draw_id}")
    anchor_pushed = push_head(); anchor_remote_oid = None
    if anchor_pushed: anchor_remote_oid = require_remote_oid(ROOT, REMOTE_NAME, ref, anchor_commit)
    return {"prediction": prediction, "prediction_commit": prediction_commit, "remote_anchor": anchor,
            "anchor_commit": anchor_commit, "remote_confirmed": anchor_pushed,
            "prediction_remote_ref_oid": remote_oid, "anchor_remote_ref_oid": anchor_remote_oid,
            "precheck": precheck, "verification_precheck": verification}


def status_summary(*, revalidate_remote: bool = True) -> dict[str, Any]:
    verify_v2_ledger(); records = read_ledger(LEDGER_PATH)
    predictions = [r for r in records if r.get("event_type") == "prediction"]
    anchors = {r["prediction_id"]: resolve_remote_anchor(r["prediction_id"], records, root=ROOT,
                                                          revalidate_remote=revalidate_remote) for r in predictions}
    valid = [r for r in predictions if anchors[r["prediction_id"]] is not None and not r.get("late_prediction")]
    return {"experiment_id": EXPERIMENT_ID, "prediction_events": len(predictions),
            "valid_remotely_anchored_predictions": len(valid),
            "unanchored_prediction_ids": [r["prediction_id"] for r in predictions if anchors[r["prediction_id"]] is None],
            "ledger": verify_ledger(LEDGER_PATH, LEDGER_HEAD_PATH),
            "formal_interim_conclusion": "PROHIBITED_BEFORE_100_VALID_ANCHORED_DRAWS"}


def compare_reproduction(expected: dict[str, Any], actual: dict[str, Any], atol: float = 1e-12) -> None:
    for key in ("top6", "top10", "top12"):
        if list(expected[key]) != list(actual[key]): raise AssertionError(f"{key} 不一致")
    expected_p = np.asarray(expected["probabilities_1_to_49"], dtype=float)
    actual_p = np.asarray(actual["probabilities"], dtype=float)
    np.testing.assert_allclose(actual_p, expected_p, rtol=0, atol=atol)
    if not np.isclose(actual_p.sum(), 6.0, rtol=0, atol=1e-12): raise AssertionError("機率總和不等於6")
    expected_rank = np.lexsort((np.arange(1, 50), -expected_p)).tolist()
    actual_rank = np.lexsort((np.arange(1, 50), -actual_p)).tolist()
    if expected_rank != actual_rank: raise AssertionError("完整機率排名不一致")
