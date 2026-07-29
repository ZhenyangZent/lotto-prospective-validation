"""前瞻鎖模工作流程：凍結、預測、結果追加、狀態、報告與審查封裝。"""
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from next_draw_predictor_audited import NUMBER_COLUMNS, load_lotto_data
from .canonical import (aggregate_file_hash, append_record, canonical_bytes, initialize_ledger,
                        read_ledger, sha256_bytes, sha256_file, verify_ledger)
from .config import (DATA_PATH, EXPERIMENT_ID, EXPERIMENT_VERSION, FROZEN_CONFIG,
                     FROZEN_CONFIG_PATH, FROZEN_MANIFEST_PATH, LEDGER_HEAD_PATH,
                     LEDGER_PATH, MODEL_VERSION, RANDOM_SEED, ROOT, SOURCE_FILES,
                     STATE_DIR, TIMEZONE, UNIFORM_PROBABILITY)
from .gitops import assert_clean, commit_paths, current_commit, push, remote_commit_url, remote_url, run_git
from .metrics import conditional_monte_carlo, exact_top6_sum_distribution, holm_adjust, score_prediction
from .model import assert_pre_target_data, baselines, fit_predict
from .official import OFFICIAL_PAGE, fetch_draw, fetch_next_draw, save_raw_response, validate_official_source

TAG_NAME = "lotto-prospective-v1-preregistered"
REMOTE_UNCONFIRMED = "REMOTE_COMMIT_NOT_CONFIRMED"


def now_iso() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def preregistration(first_draw: dict[str, str]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "registered_at": now_iso(),
        "research_question": "固定模型的完整49號邊際機率，能否在未來資料上優於精確均勻基準？",
        "model_and_parameters": FROZEN_CONFIG["model"],
        "data_source": {"organization": "台灣彩券", "official_page": OFFICIAL_PAGE},
        "first_prospective_draw": first_draw,
        "planned_sample_size": 200,
        "primary_metrics": FROZEN_CONFIG["primary_metrics"],
        "secondary_metrics": FROZEN_CONFIG["secondary_metrics"],
        "null_hypothesis": "在每期公平無放回抽取6/49下，鎖定機率向量的Brier及Log Loss不優於ExactUniform。",
        "success_criteria": FROZEN_CONFIG["success"],
        "monte_carlo": FROZEN_CONFIG["conditional_monte_carlo"],
        "interim_analysis": "每20期僅描述；第100期前不得作正式顯著性或有效/無效結論。",
        "missing_draw_policy": FROZEN_CONFIG["missing_draw_policy"],
        "official_correction_policy": FROZEN_CONFIG["official_correction_policy"],
        "bug_policy": FROZEN_CONFIG["bug_policy"],
        "stopping_rule": FROZEN_CONFIG["stopping_rule"],
        "replication_rule": "第101至200期獨立分析；不得與第一階段合併後擇優報告。",
        "immutable_fields": ["features", "model kind", "hyperparameters", "shrinkage", "Top-k rule",
                             "metrics", "success criteria", "Monte Carlo method", "stage boundaries"],
    }


def preregistration_markdown(registration: dict[str, Any]) -> str:
    return f"""# 第三階段前瞻性鎖模預註冊

- Experiment ID：`{EXPERIMENT_ID}`
- 版本：`{EXPERIMENT_VERSION}`
- 正式模型：`{MODEL_VERSION}`
- 預註冊時間：{registration['registered_at']}
- 第一個前瞻目標：{registration['first_prospective_draw']['draw_id']}（{registration['first_prospective_draw']['draw_date']}）

## 研究問題

固定模型的完整 49 號邊際機率，能否在未來資料上同時以 Brier Score 與 binary Log Loss 優於每號 6/49 的 ExactUniform？

## 鎖定方法

Batch expanding Logistic Regression；C=0.01、L2、lbfgs、max_iter=500、seed=20260729。數值特徵為 long_z、recent20_z、recent50_z、recent100_z、ewma_z、gap_z、transition_z、in_last_draw，另加號碼 one-hot。EWMA alpha=0.06、transition alpha=30、長期先驗強度=12、收縮=0.10、capped-simplex 總和=6。排序同分時小號優先。

## 資料與時序

唯一結果來源為台灣彩券官方 API/網站。預測只能在目標期開獎前，以目標期以前所有已完成資料重新 fit。遺漏預測不補寫；遠端 commit 未確認者不計正式樣本。官方修正以新事件追加。

## 指標、零假設與成功標準

主要指標為每期模型減均勻基準的 Brier 與 Log Loss；Top-6/10/12 命中為次要指標。第100期使用保留的100組機率向量，在公平6/49下以固定 seed 做至少1,000,000次條件式 Monte Carlo、plus-one p-value及95% Monte Carlo區間，兩個單尾 p 值作 Holm 修正。平均兩種差值皆須小於0、兩個修正 p 均小於0.05、至少60%的20期區塊同方向，且無洩漏或事後預測。Top-6另以 Hypergeometric 卷積核對。

## 階段、限制與停止規則

第1至100個有效預測為第一階段，第101至200個為獨立複驗。每20期可作描述性進度，但第100期前不得作正式結論；第二階段不得與第一階段合併擇優。無論第一階段結果均繼續第二階段。Bug 結束當前版本，開新版本並從零計數，不得修改本版本後續跑。

## 不得事後修改

特徵、模型、超參數、收縮、Top-k排序、指標、成功門檻、Monte Carlo方法、階段界線、缺失/修正/Bug規則皆已鎖定。
"""


def freeze_experiment(first_draw: dict[str, str] | None = None) -> dict[str, Any]:
    if FROZEN_MANIFEST_PATH.exists():
        raise FileExistsError("實驗已凍結；不得覆寫 frozen_manifest.json")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "predictions").mkdir(exist_ok=True)
    (STATE_DIR / "official_responses").mkdir(exist_ok=True)
    if first_draw is None:
        first_draw, raw = fetch_next_draw(); save_raw_response(raw, "next-draw-at-freeze")
    registration = preregistration(first_draw)
    write_json(FROZEN_CONFIG_PATH, FROZEN_CONFIG)
    write_json(STATE_DIR / "preregistration.json", registration)
    (STATE_DIR / "preregistration.md").write_text(preregistration_markdown(registration), encoding="utf-8")
    initialize_ledger(LEDGER_PATH, LEDGER_HEAD_PATH)
    code_hash, code_files = aggregate_file_hash(ROOT, SOURCE_FILES)
    manifest = {
        "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
        "frozen_at": registration["registered_at"], "first_prospective_draw": first_draw,
        "source_code_sha256": code_hash, "source_files": code_files,
        "config_sha256": sha256_file(FROZEN_CONFIG_PATH),
        "data_sha256_at_freeze": sha256_file(DATA_PATH),
        "data_end_draw_id": str(load_lotto_data(DATA_PATH).iloc[-1]["draw_id"]),
    }
    write_json(FROZEN_MANIFEST_PATH, manifest)
    return manifest


def verify_frozen_integrity() -> dict[str, Any]:
    if not FROZEN_MANIFEST_PATH.exists() or not FROZEN_CONFIG_PATH.exists():
        raise FileNotFoundError("尚未執行 freeze_experiment")
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    code_hash, code_files = aggregate_file_hash(ROOT, SOURCE_FILES)
    if code_hash != manifest["source_code_sha256"] or code_files != manifest["source_files"]:
        raise RuntimeError("程式 SHA-256 與預註冊版本不符")
    if sha256_file(FROZEN_CONFIG_PATH) != manifest["config_sha256"]:
        raise RuntimeError("設定 SHA-256 與預註冊版本不符")
    if json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8")) != FROZEN_CONFIG:
        raise RuntimeError("frozen_config 內容與程式內鎖定設定不符")
    return manifest


def _prediction_base(draw_id: str, draw_date: str, prediction: dict[str, Any], data: pd.DataFrame,
                     manifest: dict[str, Any], git_commit: str, remote_marker: str) -> dict[str, Any]:
    probabilities = [float(value) for value in prediction["probabilities"]]
    return {
        "event_type": "prediction", "event_id": str(uuid.uuid4()),
        "experiment_id": EXPERIMENT_ID, "experiment_version": EXPERIMENT_VERSION,
        "model_version": MODEL_VERSION, "prediction_id": f"{EXPERIMENT_ID}-{draw_id}",
        "target_draw_id": str(draw_id), "target_draw_date": draw_date,
        "prediction_created_at": now_iso(), "timezone": TIMEZONE,
        "data_end_draw_id": str(data.iloc[-1]["draw_id"]),
        "data_end_date": pd.Timestamp(data.iloc[-1]["draw_date"]).date().isoformat(),
        "data_sha256": sha256_file(DATA_PATH), "source_code_sha256": manifest["source_code_sha256"],
        "config_sha256": manifest["config_sha256"], "git_commit": git_commit,
        "git_remote_commit_url": remote_marker, "random_seed": RANDOM_SEED,
        "top6": prediction["top6"], "top10": prediction["top10"], "top12": prediction["top12"],
        "probabilities_1_to_49": probabilities, "probability_sum": float(sum(probabilities)),
        "official_result_status": "PENDING", "actual_numbers": None, "result_source": None,
        "result_retrieved_at": None, "hits_top6": None, "hits_top10": None, "hits_top12": None,
        "brier": None, "uniform_brier": None, "log_loss": None, "uniform_log_loss": None,
        "remote_commit_confirmed": remote_marker != REMOTE_UNCONFIRMED,
        "late_prediction": False,
    }


def predict_next(draw_id: str, draw_date: str, *, enforce_git: bool = True,
                 official_fetcher: Any = None, auto_commit: bool = True) -> dict[str, Any]:
    manifest = verify_frozen_integrity()
    if enforce_git: assert_clean(ROOT)
    verify_ledger(LEDGER_PATH, LEDGER_HEAD_PATH)
    records = read_ledger(LEDGER_PATH)
    if any(r.get("event_type") == "prediction" and str(r.get("target_draw_id")) == str(draw_id) for r in records):
        raise ValueError("同一 draw_id 已有預測，不得重複建立")
    if official_fetcher is None:
        result, raw = fetch_draw(draw_id)
    else:
        result, raw = fetch_draw(draw_id, official_fetcher)
    raw_path = save_raw_response(raw, f"precheck-{draw_id}")
    if result is not None:
        raise ValueError("官方已公布目標期結果，拒絕建立預測")
    data = load_lotto_data(DATA_PATH)
    assert_pre_target_data(data, str(draw_id), draw_date)
    prediction = fit_predict(data)
    precommit = current_commit(ROOT) if enforce_git else "TEST-NO-GIT"
    remote_marker = remote_commit_url(ROOT, precommit) if enforce_git else REMOTE_UNCONFIRMED
    record = append_record(LEDGER_PATH, LEDGER_HEAD_PATH,
                           _prediction_base(str(draw_id), draw_date, prediction, data, manifest, precommit, remote_marker))
    prediction_path = STATE_DIR / "predictions" / f"prediction-{draw_id}.json"
    csv_path = STATE_DIR / "predictions" / f"prediction-{draw_id}-probabilities.csv"
    write_json(prediction_path, record)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["number", "probability", "rank", "is_top6"]); writer.writeheader()
        ranks = {number: rank for rank, number in enumerate(record["top12"], 1)}
        for number, probability in enumerate(record["probabilities_1_to_49"], 1):
            writer.writerow({"number": number, "probability": format(probability, ".17g"),
                             "rank": ranks.get(number, ""), "is_top6": number in record["top6"]})
    digest_path = prediction_path.with_suffix(".json.sha256")
    digest_path.write_text(f"{sha256_file(prediction_path)}  {prediction_path.name}\n", encoding="ascii")
    commit = precommit; pushed = False
    if enforce_git and auto_commit:
        paths = [str(path.relative_to(ROOT)) for path in (LEDGER_PATH, LEDGER_HEAD_PATH, prediction_path, csv_path, digest_path, raw_path)]
        commit = commit_paths(ROOT, paths, f"prospective: freeze prediction for draw {draw_id}")
        pushed = push(ROOT)
    return {**record, "prediction_json": str(prediction_path), "probabilities_csv": str(csv_path),
            "prediction_git_commit": commit, "remote_commit_confirmed": pushed,
            "git_remote_commit_url": remote_commit_url(ROOT, commit) if pushed else REMOTE_UNCONFIRMED}


def _prediction_for(draw_id: str) -> dict[str, Any]:
    matches = [record for record in read_ledger(LEDGER_PATH)
               if record.get("event_type") == "prediction" and str(record.get("target_draw_id")) == str(draw_id)]
    if len(matches) != 1: raise ValueError("找不到唯一原始預測")
    return matches[0]


def ingest_result(draw_id: str, *, enforce_git: bool = True, official_fetcher: Any = None,
                  auto_commit: bool = True) -> dict[str, Any]:
    verify_frozen_integrity()
    if enforce_git: assert_clean(ROOT)
    verify_ledger(LEDGER_PATH, LEDGER_HEAD_PATH)
    prediction = _prediction_for(draw_id)
    validate_official_source(OFFICIAL_PAGE)
    result, raw = fetch_draw(draw_id) if official_fetcher is None else fetch_draw(draw_id, official_fetcher)
    if result is None: raise ValueError("官方尚未公布該期結果")
    if result["draw_date"] != prediction["target_draw_date"]: raise ValueError("官方期別日期與預測不符")
    raw_path = save_raw_response(raw, f"result-{draw_id}")
    prior_results = [r for r in read_ledger(LEDGER_PATH) if r.get("event_type") in {"result", "correction"}
                     and str(r.get("target_draw_id")) == str(draw_id)]
    if prior_results and prior_results[-1].get("actual_numbers") == result["numbers"]:
        raise ValueError("相同官方結果已寫入；不得建立重複事件")
    metrics = score_prediction(prediction["probabilities_1_to_49"], result["numbers"])
    event = {key: value for key, value in prediction.items() if key not in {"record_hash", "previous_record_hash", "sequence", "event_id", "event_type"}}
    event.update({"event_type": "correction" if prior_results else "result", "event_id": str(uuid.uuid4()),
                  "official_result_status": "CORRECTED" if prior_results else "OFFICIAL",
                  "actual_numbers": result["numbers"], "result_source": OFFICIAL_PAGE,
                  "result_retrieved_at": now_iso(), **metrics})
    record = append_record(LEDGER_PATH, LEDGER_HEAD_PATH, event)
    event_path = STATE_DIR / "predictions" / f"result-{draw_id}-{record['sequence']:04d}.json"; write_json(event_path, record)
    commit = current_commit(ROOT); pushed = False
    if enforce_git and auto_commit:
        commit = commit_paths(ROOT, [str(p.relative_to(ROOT)) for p in (LEDGER_PATH, LEDGER_HEAD_PATH, event_path, raw_path)],
                              f"prospective: append official result for draw {draw_id}")
        pushed = push(ROOT)
    return {**record, "result_git_commit": commit, "remote_commit_confirmed": pushed}


def status_summary() -> dict[str, Any]:
    integrity = verify_ledger(LEDGER_PATH, LEDGER_HEAD_PATH)
    records = read_ledger(LEDGER_PATH); predictions = [r for r in records if r.get("event_type") == "prediction"]
    latest_results: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("event_type") in {"result", "correction"}: latest_results[str(record["target_draw_id"])] = record
    valid = [r for r in latest_results.values() if r.get("remote_commit_confirmed") and not r.get("late_prediction")]
    pending = [r["target_draw_id"] for r in predictions if str(r["target_draw_id"]) not in latest_results]
    completed = len(valid)
    return {
        "completed_prospective_draws": completed, "pending_draws": pending, "ledger_valid": integrity["valid"],
        "latest_commit": current_commit(ROOT),
        "mean_hits_top6": None if not valid else float(np.mean([r["hits_top6"] for r in valid])),
        "mean_brier_difference": None if not valid else float(np.mean([r["brier"] - r["uniform_brier"] for r in valid])),
        "mean_log_loss_difference": None if not valid else float(np.mean([r["log_loss"] - r["uniform_log_loss"] for r in valid])),
        "next_formal_analysis_point": 100 if completed < 100 else 200 if completed < 200 else None,
        "formal_interim_conclusion": "PROHIBITED_BEFORE_PREREGISTERED_ANALYSIS_POINT",
    }


def report_results(simulations: int = 1_000_000) -> dict[str, Any]:
    summary = status_summary(); records = read_ledger(LEDGER_PATH)
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("event_type") in {"result", "correction"} and record.get("remote_commit_confirmed"):
            latest[str(record["target_draw_id"])] = record
    ordered = sorted(latest.values(), key=lambda r: r["sequence"])
    output: dict[str, Any] = {"status": summary, "generated_at": now_iso(), "analysis": None,
                              "notice": "未達預註冊分析點；僅提供描述性進度，不作正式顯著性結論。"}
    if len(ordered) in {100, 200}:
        stage = ordered[:100] if len(ordered) == 100 else ordered[100:200]
        probabilities = np.asarray([r["probabilities_1_to_49"] for r in stage])
        actual = np.zeros((len(stage), 49), dtype=np.int8)
        for row, record in zip(actual, stage): row[np.asarray(record["actual_numbers"]) - 1] = 1
        mc = conditional_monte_carlo(probabilities, actual, simulations=simulations)
        adjusted = holm_adjust({key: mc["p_values_plus_one"][key] for key in ("brier", "log_loss")})
        top_distribution = exact_top6_sum_distribution(len(stage)); observed_hits = sum(r["hits_top6"] for r in stage)
        blocks = [stage[index:index + 20] for index in range(0, len(stage), 20)]
        block_direction = [np.mean([r["brier"] - r["uniform_brier"] for r in block]) < 0 and
                           np.mean([r["log_loss"] - r["uniform_log_loss"] for r in block]) < 0 for block in blocks]
        output.update({"notice": "在預註冊分析點執行固定確認性分析。",
                       "analysis": {"stage": 1 if len(ordered) == 100 else 2, "monte_carlo": mc,
                                    "holm_adjusted_p": adjusted, "top6_exact_upper_tail_p": float(top_distribution[observed_hits:].sum()),
                                    "improving_20_draw_block_fraction": float(np.mean(block_direction))}})
    write_json(STATE_DIR / "prospective_report.json", output)
    (STATE_DIR / "prospective_report.md").write_text("# 前瞻驗證報告\n\n" + output["notice"] + "\n\n```json\n" + json.dumps(output, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    return output


def verify_bundle(zip_path: str | Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read("file_manifest.json"))
        errors = []
        for item in manifest:
            try: content = archive.read(item["relative_path"])
            except KeyError: errors.append(f"missing:{item['relative_path']}"); continue
            if len(content) != item["bytes"]: errors.append(f"size:{item['relative_path']}")
            if sha256_bytes(content) != item["sha256"]: errors.append(f"sha256:{item['relative_path']}")
        names = set(archive.namelist())
        expected = {item["relative_path"] for item in manifest} | {"file_manifest.json", "file_manifest.csv", "bundle_validation.txt"}
        extra = sorted(names - expected)
        if extra: errors.append("extra:" + ",".join(extra))
    return {"validation_passed": not errors, "errors": errors, "manifest_file_count": len(manifest)}


def export_review_bundle() -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = STATE_DIR / "prospective_setup_review.zip"
    with tempfile.TemporaryDirectory(prefix="prospective-review-") as temporary:
        stage = Path(temporary)
        for relative in SOURCE_FILES:
            destination = stage / relative; destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / relative, destination)
        for test in sorted((ROOT / "tests").glob("test_prospective*.py")):
            destination = stage / "tests" / test.name; destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(test, destination)
        requested = ["preregistration.md", "preregistration.json", "frozen_config.json", "frozen_manifest.json",
                     "ledger.jsonl", "ledger_head.json", "pytest_output.txt", "coverage_output.txt", "commands_executed.txt"]
        for name in requested:
            source = STATE_DIR / name
            if source.exists(): shutil.copy2(source, stage / name)
            else: (stage / name).write_text("NOT_AVAILABLE\n", encoding="utf-8")
        for prediction in sorted((STATE_DIR / "predictions").glob("*")):
            if prediction.is_file(): shutil.copy2(prediction, stage / prediction.name)
        shutil.copy2(DATA_PATH, stage / "official_data_snapshot.csv")
        write_json(stage / "official_data_source.json", {"source": OFFICIAL_PAGE, "retrieved_at": now_iso(),
                                                          "sha256": sha256_file(DATA_PATH)})
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], text=True, capture_output=True)
        (stage / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")
        (stage / "environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\n", encoding="utf-8")
        for name, args in {"git_status.txt": ["status", "--short"], "git_log.txt": ["log", "-10", "--oneline"],
                           "git_tag.txt": ["tag", "-n"]}.items():
            result = run_git(ROOT, *args, check=False); (stage / name).write_text(result.stdout + result.stderr, encoding="utf-8")
        (stage / "README_PROSPECTIVE.md").write_text("# Prospective validation review bundle\n\n前瞻性實驗預測；尚未證明提高中獎機率。\n", encoding="utf-8")
        (stage / "reproduce.ps1").write_text(
            "$ErrorActionPreference='Stop'\npython -m venv .venv\n.\\.venv\\Scripts\\python -m pip install -r requirements-lock.txt\n"
            ".\\.venv\\Scripts\\python -m pytest -q\n.\\.venv\\Scripts\\python -m prospective.verify_ledger\n"
            ".\\.venv\\Scripts\\python -m prospective.export_review_bundle --verify-only prospective_setup_review.zip\n", encoding="utf-8")
        excluded = {"file_manifest.json", "file_manifest.csv", "bundle_validation.txt"}
        files = [path for path in sorted(stage.rglob("*")) if path.is_file() and path.name not in excluded]
        descriptions = {"ledger.jsonl": "canonical append-only hash-chain ledger", "official_data_snapshot.csv": "official data snapshot"}
        manifest = [{"relative_path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size,
                     "sha256": sha256_file(path), "description": descriptions.get(path.name, "review bundle artifact")} for path in files]
        write_json(stage / "file_manifest.json", manifest)
        with (stage / "file_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256", "description"]); writer.writeheader(); writer.writerows(manifest)
        (stage / "bundle_validation.txt").write_text("VALIDATION_PENDING_ZIP_REOPEN\n", encoding="utf-8")
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(stage).as_posix())
    validation = verify_bundle(bundle_path)
    # Rebuild once so the validation artifact inside the ZIP records the actual result.
    if validation["validation_passed"]:
        with zipfile.ZipFile(bundle_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("bundle_validation.txt", json.dumps(validation, sort_keys=True) + "\n")
    validation = verify_bundle(bundle_path)
    return {"path": str(bundle_path.resolve()), "bytes": bundle_path.stat().st_size,
            "sha256": sha256_file(bundle_path), **validation}
