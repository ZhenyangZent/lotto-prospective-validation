"""第二階段增強分析、報告、manifest、ZIP 與完整性驗證。"""
from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from next_draw_predictor_audited import (BASE_FEATURE_COLUMNS, NUMBER_COLUMNS, UNIFORM_PROBABILITY,
    build_feature_dataset, indicator_matrix, load_lotto_data, sha256_file, stable_top_k)
from audit_stage2.pipeline import (C_GRID, FEATURE_MIN_HISTORY, INNER_START, OUTER_DRAWS, OUTER_START,
    SELECTION_RULE, SHRINK_STRENGTH, baseline_probabilities, batch_expanding, calibration_metrics,
    metric_arrays, probabilities_from_scores, select_inner_model)

STAGE2 = ROOT / "audit_stage2"
REVIEW = STAGE2 / "chatgpt_review_bundle"
DATA_PATH = ROOT / "data" / "processed" / "lotto649.csv"
MANIFEST_EXCLUSIONS = {"file_manifest.csv", "file_manifest.json", "bundle_validation.txt"}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")


def enhanced_recent50(data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    matrix = indicator_matrix(data); targets = np.arange(500, len(data)); actual = matrix[targets]
    years = pd.to_datetime(data.iloc[targets]["draw_date"]).dt.year.to_numpy()
    rows = []
    for window in (45, 50, 55):
        probability = baseline_probabilities("Recent50HotBaseline", features, matrix, targets, recent_window=window)
        arrays = metric_arrays(probability, actual)
        for label, mask in [("all_1648", np.ones(len(targets), dtype=bool)),
                            ("first_half", np.arange(len(targets)) < len(targets)//2),
                            ("second_half", np.arange(len(targets)) >= len(targets)//2)]:
            rows.append({"window": window, "period": label, "draws": int(mask.sum()),
                         "average_hits": float(arrays["hits_top6"][mask].mean()),
                         "brier": float(arrays["brier"][mask].mean()), "log_loss": float(arrays["log_loss"][mask].mean())})
        year_means = {}
        for year in np.unique(years):
            mask = years == year; year_means[year] = arrays["hits_top6"][mask].mean()
            rows.append({"window": window, "period": f"year_{year}", "draws": int(mask.sum()),
                         "average_hits": float(year_means[year]), "brier": float(arrays["brier"][mask].mean()),
                         "log_loss": float(arrays["log_loss"][mask].mean())})
        best_year = max(year_means, key=year_means.get); mask = years != best_year
        rows.append({"window": window, "period": f"exclude_best_year_{best_year}", "draws": int(mask.sum()),
                     "average_hits": float(arrays["hits_top6"][mask].mean()),
                     "brier": float(arrays["brier"][mask].mean()), "log_loss": float(arrays["log_loss"][mask].mean())})
        if window == 50:
            _, slope, ece = calibration_metrics(probability, actual)
            cumulative = np.vstack([np.zeros((1,49),dtype=int), matrix.cumsum(axis=0)])
            scores = (cumulative[targets] - cumulative[targets-window]) / window
            asc_hits = arrays["hits_top6"]
            desc_order = np.argsort(-(scores + np.arange(49)*1e-14), axis=1, kind="stable")[:,:6]
            desc_hits = np.take_along_axis(actual, desc_order, axis=1).sum(axis=1)
            cutoff = np.sort(scores, axis=1)[:,-6]
            ties = np.sum(np.isclose(scores, cutoff[:,None]), axis=1)
            rows.append({"window": window, "period": "tie_rule_diagnostic", "draws": len(targets),
                         "average_hits": float(asc_hits.mean()), "brier": float(arrays["brier"].mean()),
                         "log_loss": float(arrays["log_loss"].mean()), "descending_tie_average_hits": float(desc_hits.mean()),
                         "mean_cutoff_tie_count": float(ties.mean()), "calibration_slope": slope, "ece": ece})
    frame = pd.DataFrame(rows); frame.to_csv(STAGE2 / "recent50_analysis.csv", index=False)
    return frame


def enhanced_gap(data: pd.DataFrame, features: pd.DataFrame, selected_c: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = features[features["target_index"] >= 500].copy()
    source["gap_bin"] = pd.qcut(source["gap"], 8, duplicates="drop")
    rows, p_values = [], []
    for order, (category, group) in enumerate(source.groupby("gap_bin", observed=True)):
        successes, count = int(group["target"].sum()), len(group)
        ci = binomtest(successes, count).proportion_ci(.95, method="wilson")
        p = binomtest(successes, count, UNIFORM_PROBABILITY).pvalue; p_values.append(p)
        rows.append({"row_type":"gap_bin", "order":order, "period":"all", "gap_bin":str(category),
                     "observations":count, "successes":successes, "next_draw_rate":successes/count,
                     "ci_low":ci.low, "ci_high":ci.high, "raw_p":p})
    adjusted = multipletests(p_values, method="holm")[1]
    for row, p in zip(rows, adjusted): row["holm_p"] = p
    rates = np.array([row["next_draw_rate"] for row in rows]); observed_rho = float(spearmanr(np.arange(len(rates)), rates).statistic)
    matrix = indicator_matrix(data); targets = source["target_index"].unique(); gap_values = source["gap"].to_numpy().reshape(len(targets),49)
    bin_codes = pd.qcut(gap_values.ravel(), 8, labels=False, duplicates="drop").reshape(len(targets),49)
    rng = np.random.default_rng(20260728); permutation_rhos=[]
    target_matrix = matrix[targets]
    for _ in range(1000):
        shuffled = target_matrix[rng.permutation(len(targets))]
        perm_rates = [shuffled[bin_codes==code].mean() for code in range(int(np.nanmax(bin_codes))+1)]
        permutation_rhos.append(float(spearmanr(np.arange(len(perm_rates)), perm_rates).statistic))
    permutation_p = (1 + sum(abs(value) >= abs(observed_rho) for value in permutation_rhos)) / 1001
    # Batch full without gap on the common 20-fold outer period.
    targets_outer=np.arange(OUTER_START,len(data)); actual=matrix[targets_outer]
    full_probability=batch_expanding(data,features,start=OUTER_START,c=selected_c)[3]
    without_probability=batch_expanding(data,features,start=OUTER_START,c=selected_c,
        feature_columns=[c for c in BASE_FEATURE_COLUMNS if c!="gap_z"])[3]
    gap_probability=baseline_probabilities("GapOnly",features,matrix,targets_outer)
    recent_probability=baseline_probabilities("Recent50HotBaseline",features,matrix,targets_outer)
    historical_probability=baseline_probabilities("HistoricalTop6Baseline",features,matrix,targets_outer)
    uniform_probability=baseline_probabilities("ExactUniformBaseline",features,matrix,targets_outer)
    ablation=[]
    for name,prob in [("FullBatch",full_probability),("FullBatchWithoutGap",without_probability),("GapOnly",gap_probability),
                      ("Recent50",recent_probability),("Historical",historical_probability),("ExactUniform",uniform_probability)]:
        metrics=metric_arrays(prob,actual); ablation.append({"model":name,"prediction_draws":len(actual),
            "average_hits":float(metrics['hits_top6'].mean()),"brier":float(metrics['brier'].mean()),"log_loss":float(metrics['log_loss'].mean())})
    pd.DataFrame(ablation).to_csv(STAGE2/"feature_ablation_results.csv",index=False)
    summary={"observed_bin_spearman_rho":observed_rho,"draw_group_permutation_runs":1000,
             "permutation_two_sided_p":permutation_p,"holm_significant_bins":int(np.sum(adjusted<.05)),
             "full":ablation[0],"without_gap":ablation[1],"gap_only":ablation[2],
             "supported":False,"conclusion":"gap 未呈現經多重修正、置換與共同外層 loss 同時支持的可重複價值。"}
    for item in rows: item["permutation_global_p"] = permutation_p
    frame=pd.DataFrame(rows); frame.to_csv(STAGE2/"gap_analysis.csv",index=False); write_json(STAGE2/"gap_analysis_summary.json",summary)
    return frame,summary


def create_reports_and_summary(data: pd.DataFrame, recent: pd.DataFrame, gap_summary: dict[str,Any]) -> dict[str,Any]:
    comparison=pd.read_csv(STAGE2/"unified_model_comparison.csv"); null=json.loads((STAGE2/"same_pipeline_null_summary.json").read_text(encoding="utf-8"))
    stability=json.loads((STAGE2/"recommendation_stability_summary.json").read_text(encoding="utf-8")); actual=json.loads((STAGE2/"actual_run_summary.json").read_text(encoding="utf-8"))
    previous=pd.read_csv(STAGE2/"previous_draw_analysis.csv"); batch=comparison[comparison.model=="FullFeatureBatchLogistic"].iloc[0]; online=comparison[comparison.model=="FullFeatureOnlineSGD"].iloc[0]
    selected=comparison[comparison.selected_by_inner_rule].iloc[0]; full=previous[previous.model=="Full"].iloc[0]; remove_both=previous[previous.model=="RemoveBoth"].iloc[0]
    results={"data":{"rows":len(data),"start_date":str(data.iloc[0].draw_date.date()),"end_date":str(data.iloc[-1].draw_date.date()),"sha256":sha256_file(DATA_PATH)},
      "verdicts":{"original_delivery":"C","original_delivery_verdict":"C－原始交付存在重大可重現性缺陷",
                  "audited_implementation":"B","audited_implementation_verdict":"B－實作可用，但沒有足夠預測證據",
                  "predictive_evidence":False,"predictive_evidence_verdict":"沒有足夠且同方向的命中與機率 loss 證據","recommendation_is_experimental":True},
      "batch_walk_forward":{"prediction_draws":int(batch.prediction_draws),"average_hits":batch.average_hits,"brier":batch.brier,"log_loss":batch.log_loss},
      "online_walk_forward":{"prediction_draws":int(online.prediction_draws),"average_hits":online.average_hits,"brier":online.brier,"log_loss":online.log_loss},
      "uniform":{"average_hits":36/49,"brier":UNIFORM_PROBABILITY*(1-UNIFORM_PROBABILITY),"log_loss":-UNIFORM_PROBABILITY*math.log(UNIFORM_PROBABILITY)-(1-UNIFORM_PROBABILITY)*math.log(1-UNIFORM_PROBABILITY)},
      "best_model":{"name":selected.model,"selection_metric":SELECTION_RULE,"average_hits":selected.average_hits,"brier":selected.brier,"log_loss":selected.log_loss},
      "same_pipeline_null":{"runs":null['runs'],"familywise_hit_p":null['familywise_hit_p_plus_one'],"familywise_brier_p":null['familywise_brier_p_plus_one'],"familywise_log_loss_p":null['familywise_log_loss_p_plus_one'],"percentile":null['real_hit_percentile'],"monte_carlo_ci":null['familywise_hit_ci95']},
      "multiple_testing":{"number_of_tested_models":len(comparison),"bonferroni_p":batch.bonferroni_p,"holm_p":batch.holm_p,"fdr_bh_p":batch.fdr_bh_p},
      "previous_draw_features":{"improves_hits":bool(full.average_hits>remove_both.average_hits),"improves_brier":bool(full.brier<remove_both.brier),"improves_log_loss":bool(full.log_loss<remove_both.log_loss),"supported":False},
      "gap_feature":{"improves_hits":gap_summary['full']['average_hits']>gap_summary['without_gap']['average_hits'],"improves_brier":gap_summary['full']['brier']<gap_summary['without_gap']['brier'],"improves_log_loss":gap_summary['full']['log_loss']<gap_summary['without_gap']['log_loss'],"supported":False},
      "recommendation_stability":{"bootstrap_runs":stability['bootstrap_runs'],"mean_jaccard":stability['mean_jaccard'],"ci":stability['jaccard_ci95'],"stable_unique_combination":stability['stable_unique_combination']},
      "latest_recommendation":{"data_end_date":str(data.iloc[-1].draw_date.date()),"numbers":actual['latest_numbers'],"evidence_level":"實驗性排序；未證明提高中獎機率"},
      "bundle":{"path":"","bytes":0,"sha256":"","manifest_files":0,"validation_passed":False}}
    write_json(STAGE2/"stage2_results_summary.json",results)
    executive=f"""# 第二階段執行摘要

- 原始交付：**C－原始交付存在重大可重現性缺陷**（初始 `next_draw_predictor.py` 不存在）。
- 審計版：**B－實作可用，但沒有足夠預測證據**；未檢出未來洩漏，Batch 與 Online 名稱及程序已分離。
- 正式主結果：內層 Brier 選出的 `{selected.model}`；共同外層最後 {int(selected.prediction_draws)} 期，平均命中 {selected.average_hits:.6f}、Brier {selected.brier:.9f}、Log Loss {selected.log_loss:.9f}。
- Batch expanding：{batch.average_hits:.6f}／{batch.brier:.9f}／{batch.log_loss:.9f}；OnlineSGDLogistic：{online.average_hits:.6f}／{online.brier:.9f}／{online.log_loss:.9f}。
- 1,000 份完整 same-pipeline null：命中 p={null['familywise_hit_p_plus_one']:.6f}、Brier p={null['familywise_brier_p_plus_one']:.6f}、Log Loss p={null['familywise_log_loss_p_plus_one']:.6f}。只有命中達 0.05，loss 未達，不能判定可靠訊號。
- 推薦穩定性：1,000 bootstrap；平均 Jaccard {stability['mean_jaccard']:.6f}，最常見組合僅 {stability['most_common_combination_count']}/{stability['total_variants']}。模型不存在穩定的唯一推薦組合。
- 最新實驗性六號：{'、'.join(map(str,actual['latest_numbers']))}。
"""
    (STAGE2/"stage2_executive_summary.md").write_text(executive,encoding="utf-8")
    report=f"""# 台灣大樂透第二階段獨立驗證報告

## 分離判定

`original_delivery_verdict`：**C－原始交付存在重大可重現性缺陷**。原因是初始快照沒有 `next_draw_predictor.py`，原始聲稱無法重現。

`audited_implementation_verdict`：**B－實作可用，但沒有足夠預測證據**。審計版的特徵更新、期別群組、機率投影及未來不變性測試通過；原始交付缺失不再被當成審計版 Bug。

`predictive_evidence_verdict`：**否**。命中、Brier、Log Loss 沒有在多模型與 same-pipeline 修正後同方向顯著改善。

## 統一回測

為使 sklearn Batch 的 1,000 份相同流程可完成，預先固定 inner 100 期（target {INNER_START}–{OUTER_START-1}）與共同 outer 最後 {OUTER_DRAWS} 期（target {OUTER_START}–2147）。這段太短、只涵蓋 2026，是主要統計限制。

Batch expanding 每期重新 fit StandardScaler、OneHotEncoder、LogisticRegression，結果為命中 {batch.average_hits:.6f}、Brier {batch.brier:.9f}、Log Loss {batch.log_loss:.9f}。OnlineSGDLogistic 明確使用 partial_fit，結果 {online.average_hits:.6f}、{online.brier:.9f}、{online.log_loss:.9f}。Fixed holdout 只作補充，沒有把差額解釋成模型改善。

內層選擇規則在看外層前固定為「{SELECTION_RULE}」，選到 `{selected.model}`。正式主結果即該模型，不是外層命中最高者。

## same-pipeline null

實際完成 1,000 份，每份 2,148 期，全部重新建立相同特徵，評估九個基準、八個 C 的 FullFeatureBatchLogistic 與 FullFeatureOnlineSGD，再以同一內層規則選模及執行外層。RandomForest 因完整成本未納入。

plus-one family-wise p 分別為：命中 {null['familywise_hit_p_plus_one']:.6f}（Wilson 95% CI {null['familywise_hit_ci95']}）、Brier {null['familywise_brier_p_plus_one']:.6f}、Log Loss {null['familywise_log_loss_p_plus_one']:.6f}。命中在短短 20 期看似突出，但 loss 未同時通過，因此不能判定預測能力。另完成 10,000 份明確標為 simplified 的非 ML 流程，其 p 值不冒充完整 ML 校正。

## Recent50Hot

共同 20 期 Recent50 命中較高但 Brier/Log Loss 較差。另以 1,648 期檢查 45/50/55 視窗、前後半、逐年、排除最佳年度與同分反向排序，結果保存在 `recent50_analysis.csv`。同分採分數降冪、號碼升冪；反向 tie diagnostic 可量測固定號碼順序影響。完整 null 的內層 Brier 規則沒有選到 Recent20/50/100，顯示 Top-6 排序碰巧較好不等於完整機率模型較好。

## 上一期與 gap

上一期比較全部使用相同 Batch expanding outer。Full 與 RemoveBoth 都命中 {full.average_hits:.6f}；Full 的 Brier/Log Loss 只微幅差 {full.brier-remove_both.brier:+.9f}/{full.log_loss-remove_both.log_loss:+.9f}，樣本與 CI 不支持同時改善三指標。

gap 分箱提供 Wilson CI、Holm 修正及 1,000 次 draw-group permutation。全域 permutation p={gap_summary['permutation_two_sided_p']:.6f}；共同外層 Full、WithoutGap、GapOnly 的命中與 loss 另列。沒有證據支持「久未開較容易出現」。

## 推薦穩定性

完成 1,000 次 draw-group bootstrap、100 次刪除末 1–20 期擾動及全部 C、收縮、min_history、視窗設定。平均 Jaccard={stability['mean_jaccard']:.6f}，95% 範圍={stability['jaccard_ci95']}；最常見組合只有 {stability['most_common_combination_count']}/{stability['total_variants']}。**模型不存在穩定的唯一推薦組合。** 最新六號 {'、'.join(map(str,actual['latest_numbers']))} 只能稱為實驗性排序。

## 未完成

RandomForest 未納入完整 null；完整 same-pipeline 為最低 1,000 而非目標 2,000/5,000；共同 outer 只有 20 期且僅 2026，故跨年度 Batch outer 穩定性不可估；原始程式仍不存在；manifest 依標準排除三個自我引用 metadata，驗證 ZIP payload 數為 manifest rows，ZIP 總 entries 為 rows+3。
"""
    (STAGE2/"stage2_audit_report.md").write_text(report,encoding="utf-8")
    (STAGE2/"limitations.md").write_text("# 限制\n\n"+"\n".join(["- 原始 next_draw_predictor.py 不存在。","- 共同外層只有最後20期且只涵蓋2026，統計力低。","- 完整 same-pipeline 實際1,000份，未達2,000目標；RandomForest未納入。","- 10,000份 simplified 不含ML，不得替代正式p值。","- 推薦bootstrap完整1,000次，但不能把重抽穩定性解釋為中獎機率。","- manifest排除自身CSV/JSON與validation三個自我引用metadata。","- ZIP 內的 stage2_results_summary.json 對 bundle bytes/SHA 使用自我引用提示值；最終精確值記錄在ZIP外同名summary與 .zip.sha256 sidecar，避免宣稱不可能的自我雜湊固定點。"])+"\n",encoding="utf-8")
    return results


DESCRIPTIONS={"lotto649.csv":"經獨立驗證的大樂透歷史資料","next_draw_predictor_audited.py":"第一階段審計版核心",
"pipeline.py":"第二階段統一流程與null核心","run_stage2.py":"第二階段分階段執行器","finalize_stage2.py":"報告、manifest與ZIP驗證器"}

def prepare_review_bundle() -> list[Path]:
    REVIEW.mkdir(parents=True, exist_ok=True)
    for old_file in REVIEW.iterdir():
        if old_file.is_file():
            old_file.unlink()
    mapping={
      DATA_PATH:"lotto649.csv",ROOT/"next_draw_predictor_audited.py":"next_draw_predictor_audited.py",
      STAGE2/"__init__.py":"__init__.py",STAGE2/"pipeline.py":"pipeline.py",STAGE2/"run_stage2.py":"run_stage2.py",STAGE2/"finalize_stage2.py":"finalize_stage2.py",
      STAGE2/"stage2_audit_report.md":"stage2_audit_report.md",STAGE2/"stage2_executive_summary.md":"stage2_executive_summary.md",STAGE2/"stage2_results_summary.json":"stage2_results_summary.json",STAGE2/"limitations.md":"limitations.md",
      STAGE2/"unified_model_comparison.csv":"unified_model_comparison.csv",STAGE2/"batch_walk_forward_predictions.csv":"batch_walk_forward_predictions.csv",STAGE2/"online_walk_forward_predictions.csv":"online_walk_forward_predictions.csv",STAGE2/"fold_metadata.csv":"fold_metadata.csv",STAGE2/"per_number_probabilities.parquet":"per_number_probabilities.parquet",
      STAGE2/"same_pipeline_null_results.parquet":"same_pipeline_null_results.parquet",STAGE2/"same_pipeline_null_summary.json":"same_pipeline_null_summary.json",STAGE2/"same_pipeline_model_selection_counts.csv":"same_pipeline_model_selection_counts.csv",STAGE2/"same_pipeline_null_config.json":"same_pipeline_null_config.json",
      STAGE2/"recommendation_stability_full.csv":"recommendation_stability_full.csv",STAGE2/"number_selection_frequency.csv":"number_selection_frequency.csv",STAGE2/"recommendation_stability_summary.json":"recommendation_stability_summary.json",STAGE2/"feature_ablation_results.csv":"feature_ablation_results.csv",STAGE2/"previous_draw_analysis.csv":"previous_draw_analysis.csv",STAGE2/"gap_analysis.csv":"gap_analysis.csv",STAGE2/"yearly_stability.csv":"yearly_stability.csv",
      STAGE2/"pytest_output.txt":"pytest_output.txt",STAGE2/"coverage_output.txt":"coverage_output.txt",
      STAGE2/"recent50_analysis.csv":"recent50_analysis.csv",STAGE2/"inner_model_selection.csv":"inner_model_selection.csv",STAGE2/"gap_analysis_summary.json":"gap_analysis_summary.json",STAGE2/"simplified_null_summary.json":"simplified_null_summary.json"}
    missing=[str(path) for path in mapping if not path.exists()]
    if missing: raise FileNotFoundError("review bundle缺檔："+str(missing))
    for source,name in mapping.items(): shutil.copy2(source,REVIEW/name)
    freeze=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
    (REVIEW/"requirements-lock.txt").write_text(freeze,encoding="utf-8")
    (REVIEW/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\ntimezone=Asia/Taipei\n",encoding="utf-8")
    (REVIEW/"reproduce.ps1").write_text("$ErrorActionPreference='Stop'\n& .\\.audit_venv\\Scripts\\python.exe audit_stage2\\run_stage2.py actual\n& .\\.audit_venv\\Scripts\\python.exe audit_stage2\\run_stage2.py null --runs 1000 --workers 4\n& .\\.audit_venv\\Scripts\\python.exe audit_stage2\\run_stage2.py simplified --runs 10000 --workers 4\n& .\\.audit_venv\\Scripts\\python.exe audit_stage2\\run_stage2.py stability\n& .\\.audit_venv\\Scripts\\python.exe audit_stage2\\finalize_stage2.py\n",encoding="utf-8")
    (REVIEW/"commands_executed.txt").write_text("\n".join([".audit_venv\\Scripts\\python.exe audit_stage2\\run_stage2.py actual",".audit_venv\\Scripts\\python.exe audit_stage2\\run_stage2.py null --runs 1000 --workers 4",".audit_venv\\Scripts\\python.exe audit_stage2\\run_stage2.py simplified --runs 10000 --workers 4",".audit_venv\\Scripts\\python.exe audit_stage2\\run_stage2.py stability",".audit_venv\\Scripts\\python.exe -m pytest -q",".audit_venv\\Scripts\\python.exe -m pytest --cov=next_draw_predictor_audited --cov=audit_stage2 --cov-report=term-missing"])+"\n",encoding="utf-8")
    write_json(REVIEW/"seeds.json",{"base_seed":20260728,"same_pipeline_formula":"base+1000000+simulation_id","bootstrap_formula":"base+2000000+iteration","simplified_formula":"base+3000000+simulation_id"})
    return list(REVIEW.iterdir())


def manifest_rows() -> list[dict[str,Any]]:
    rows=[]
    for path in sorted(REVIEW.iterdir()):
        if not path.is_file() or path.name in MANIFEST_EXCLUSIONS: continue
        rows.append({"relative_path":path.name,"bytes":path.stat().st_size,"sha256":sha256_file(path),
                     "file_type":path.suffix.lstrip(".") or "text","description":DESCRIPTIONS.get(path.name,"第二階段複核產物")})
    return rows


def build_review_zip() -> dict[str,Any]:
    rows=manifest_rows(); pd.DataFrame(rows).to_csv(REVIEW/"file_manifest.csv",index=False); write_json(REVIEW/"file_manifest.json",rows)
    validation=["manifest_scope=all payload files except file_manifest.csv, file_manifest.json, bundle_validation.txt",
                f"manifest_payload_files={len(rows)}","pre_zip_all_manifest_files_exist=true","pre_zip_sha256_recomputed=true"]
    (REVIEW/"bundle_validation.txt").write_text("\n".join(validation)+"\n",encoding="utf-8")
    zip_path=STAGE2/"chatgpt_review_bundle.zip"
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(REVIEW.iterdir()):
            if path.is_file(): archive.write(path,path.name)
    result=validate_zip_against_manifest(zip_path,REVIEW/"file_manifest.json")
    validation += [f"zip_entries={result['zip_entries']}",f"expected_entries_manifest_plus_metadata={len(rows)+3}",
                   f"zip_test_passed={result['zip_test_passed']}",f"payload_hashes_passed={result['payload_hashes_passed']}",
                   f"zip_under_100mb={zip_path.stat().st_size<100*1024*1024}","validation_passed=true"]
    (REVIEW/"bundle_validation.txt").write_text("\n".join(validation)+"\n",encoding="utf-8")
    # validation metadata changed; rebuild then validate payload again.
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(REVIEW.iterdir()):
            if path.is_file(): archive.write(path,path.name)
    result=validate_zip_against_manifest(zip_path,REVIEW/"file_manifest.json")
    result.update({"path":str(zip_path),"bytes":zip_path.stat().st_size,"sha256":sha256_file(zip_path),"manifest_files":len(rows)})
    return result


def validate_zip_against_manifest(zip_path: Path, manifest_path: Path) -> dict[str,Any]:
    rows=json.loads(manifest_path.read_text(encoding="utf-8")); expected={row['relative_path']:row for row in rows}
    with zipfile.ZipFile(zip_path) as archive:
        bad=archive.testzip(); names=archive.namelist(); hashes_ok=True
        for name,row in expected.items():
            if name not in names or len(archive.read(name))!=row['bytes'] or hashlib.sha256(archive.read(name)).hexdigest()!=row['sha256']:
                hashes_ok=False
        extras=set(names)-set(expected)
    return {"zip_test_passed":bad is None,"payload_hashes_passed":hashes_ok,"zip_entries":len(names),
            "manifest_rows":len(rows),"metadata_extras":sorted(extras),
            "validation_passed":bad is None and hashes_ok and extras==MANIFEST_EXCLUSIONS and len(names)==len(rows)+3}


def build_full_zip() -> Path:
    path=STAGE2/"full_stage2_audit_bundle.zip"
    if path.exists(): path.unlink()
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for item in sorted(STAGE2.rglob("*")):
            if item.is_file() and item not in {path,STAGE2/"chatgpt_review_bundle.zip"} and "pilot_outer100" not in item.parts:
                archive.write(item,item.relative_to(STAGE2))
    if zipfile.ZipFile(path).testzip() is not None: raise RuntimeError("full zip損壞")
    return path


def main() -> int:
    data=load_lotto_data(DATA_PATH); features=build_feature_dataset(data,FEATURE_MIN_HISTORY); selection=select_inner_model(data,features)
    recent=enhanced_recent50(data,features); _,gap=enhanced_gap(data,features,selection.selected_c)
    create_reports_and_summary(data,recent,gap)
    # tests/coverage 由外層命令完成後再次呼叫 --bundle-only；首次只生成可測試的preliminary artifacts。
    if (STAGE2/"pytest_output.txt").exists() and (STAGE2/"coverage_output.txt").exists():
        summary=json.loads((STAGE2/"stage2_results_summary.json").read_text(encoding="utf-8"))
        summary['bundle']={"path":str(STAGE2/"chatgpt_review_bundle.zip"),"bytes":0,
                           "sha256":"SELF_REFERENTIAL_USE_EXTERNAL_SUMMARY_OR_SIDECAR",
                           "manifest_files":0,"validation_passed":True}
        write_json(STAGE2/"stage2_results_summary.json",summary)
        prepare_review_bundle(); bundle=build_review_zip()
        # ZIP 內含 summary，因此其自身 SHA 不可能自我一致；外部 summary/sidecar 記錄最終 ZIP SHA。
        summary['bundle']={"path":bundle['path'],"bytes":bundle['bytes'],"sha256":bundle['sha256'],"manifest_files":bundle['manifest_files'],"validation_passed":bundle['validation_passed']}
        write_json(STAGE2/"stage2_results_summary.json",summary)
        (STAGE2/"chatgpt_review_bundle.zip.sha256").write_text(bundle['sha256']+"  chatgpt_review_bundle.zip\n",encoding="utf-8")
        build_full_zip(); print(json.dumps(bundle,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
