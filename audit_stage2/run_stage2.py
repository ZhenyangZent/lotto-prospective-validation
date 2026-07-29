"""執行第二階段獨立驗證、same-pipeline null、穩定性、測試與打包。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from next_draw_predictor_audited import (  # noqa: E402
    BASE_FEATURE_COLUMNS,
    DEFAULT_SEED,
    MODEL_VERSION,
    NUMBER_COLUMNS,
    UNIFORM_PROBABILITY,
    build_feature_dataset,
    build_next_features,
    fit_latest_recommendation,
    indicator_matrix,
    load_lotto_data,
    make_model,
    normalize_and_shrink_probabilities,
    sha256_file,
    stable_top_k,
)
from audit_stage2.pipeline import (  # noqa: E402
    CANDIDATE_NAMES,
    C_GRID,
    FEATURE_MIN_HISTORY,
    INNER_START,
    MODEL_COMPLEXITY,
    OUTER_DRAWS,
    OUTER_START,
    SELECTION_RULE,
    SHRINK_STRENGTH,
    baseline_probabilities,
    batch_expanding,
    bootstrap_difference,
    calibration_metrics,
    evaluate_probabilities,
    exact_hit_p,
    fair_matrix,
    fixed_logistic_probabilities,
    metric_arrays,
    online_sgd,
    probabilities_from_scores,
    select_inner_model,
    simulate_same_pipeline,
    simulate_simplified_pipeline,
    stable_score_top,
)

STAGE2 = ROOT / "audit_stage2"
REVIEW = STAGE2 / "chatgpt_review_bundle"
DATA_PATH = ROOT / "data" / "processed" / "lotto649.csv"
SAME_PIPELINE_RUNS = 1000
SIMPLIFIED_RUNS = 10_000
BOOTSTRAP_RUNS = 1000
BASE_SEED = 20260728


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, pd.Timestamp): return value.isoformat()
    raise TypeError(type(value).__name__)


def summarize_model(name: str, probabilities: np.ndarray, actual: np.ndarray, targets: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    row, detail = evaluate_probabilities(name, probabilities, actual, targets)
    uniform_brier = UNIFORM_PROBABILITY * (1 - UNIFORM_PROBABILITY)
    uniform_log = -UNIFORM_PROBABILITY * math.log(UNIFORM_PROBABILITY) - (1 - UNIFORM_PROBABILITY) * math.log(1 - UNIFORM_PROBABILITY)
    arrays = metric_arrays(probabilities, actual)
    hit_difference = arrays["hits_top6"] - 36 / 49
    row.update({
        "hit_difference_vs_uniform": row["average_hits"] - 36 / 49,
        "brier_difference_vs_uniform": row["brier"] - uniform_brier,
        "log_loss_difference_vs_uniform": row["log_loss"] - uniform_log,
        "paired_bootstrap_hit_ci_low": bootstrap_difference(hit_difference)[0],
        "paired_bootstrap_hit_ci_high": bootstrap_difference(hit_difference)[1],
        "block20_bootstrap_hit_ci_low": bootstrap_difference(hit_difference, block=20)[0],
        "block20_bootstrap_hit_ci_high": bootstrap_difference(hit_difference, block=20)[1],
        "exact_hit_count_p": exact_hit_p(int(arrays["hits_top6"].sum()), len(targets)),
    })
    return row, detail


def run_actual() -> None:
    started = time.perf_counter()
    data = load_lotto_data(DATA_PATH)
    features = build_feature_dataset(data, FEATURE_MIN_HISTORY)
    matrix = indicator_matrix(data)
    targets = np.arange(OUTER_START, len(data))
    actual = matrix[targets]
    selection = select_inner_model(data, features)
    selection.candidate_table.to_csv(STAGE2 / "inner_model_selection.csv", index=False)
    write_json(STAGE2 / "selection_rule.json", {
        "rule": SELECTION_RULE, "inner_targets": [INNER_START, OUTER_START - 1],
        "outer_targets": [OUTER_START, len(data) - 1], "C_grid": C_GRID,
        "shrink_strength": SHRINK_STRENGTH, "selected_model": selection.selected_model,
        "selected_C_for_batch": selection.selected_c,
    })

    batch_pred, batch_number, batch_meta, batch_probability = batch_expanding(
        data, features, start=OUTER_START, c=selection.selected_c)
    online_pred, online_number, online_meta, online_probability = online_sgd(
        data, features, start=OUTER_START, c=.20)
    fixed_probability = fixed_logistic_probabilities(data, features, targets, c=selection.selected_c)
    batch_pred.to_csv(STAGE2 / "batch_walk_forward_predictions.csv", index=False)
    online_pred.to_csv(STAGE2 / "online_walk_forward_predictions.csv", index=False)
    pd.concat([batch_number, online_number], ignore_index=True).to_parquet(
        STAGE2 / "per_number_probabilities.parquet", index=False)
    pd.concat([batch_meta, online_meta], ignore_index=True).to_csv(STAGE2 / "fold_metadata.csv", index=False)

    probability_by_model: dict[str, np.ndarray] = {}
    for name in CANDIDATE_NAMES:
        if name == "FullFeatureBatchLogistic": probability_by_model[name] = batch_probability
        elif name == "FullFeatureOnlineSGD": probability_by_model[name] = online_probability
        else: probability_by_model[name] = baseline_probabilities(name, features, matrix, targets)
    probability_by_model["FixedHoldoutLogistic"] = fixed_probability
    rows, details = [], []
    for name, probability in probability_by_model.items():
        row, detail = summarize_model(name, probability, actual, targets)
        row["selected_by_inner_rule"] = name == selection.selected_model
        rows.append(row); details.append(detail)
    comparison = pd.DataFrame(rows)
    raw = comparison["exact_hit_count_p"].to_numpy()
    comparison["bonferroni_p"] = multipletests(raw, method="bonferroni")[1]
    comparison["holm_p"] = multipletests(raw, method="holm")[1]
    comparison["fdr_bh_p"] = multipletests(raw, method="fdr_bh")[1]
    comparison.sort_values(["selected_by_inner_rule", "brier"], ascending=[False, True]).to_csv(
        STAGE2 / "unified_model_comparison.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(STAGE2 / "unified_per_draw_metrics.csv", index=False)

    # Recent50：視窗擾動、前後半、年度、排除最佳年度。
    recent_rows = []
    for window in (45, 50, 55):
        probability = baseline_probabilities("Recent50HotBaseline", features, matrix, targets, recent_window=window)
        arrays = metric_arrays(probability, actual)
        years = pd.to_datetime(data.iloc[targets]["draw_date"]).dt.year.to_numpy()
        for label, mask in [("all", np.ones(len(targets), dtype=bool)),
                            ("first_half", np.arange(len(targets)) < len(targets) // 2),
                            ("second_half", np.arange(len(targets)) >= len(targets) // 2)]:
            recent_rows.append({"window": window, "period": label, "draws": int(mask.sum()),
                                "average_hits": float(arrays["hits_top6"][mask].mean()),
                                "brier": float(arrays["brier"][mask].mean()),
                                "log_loss": float(arrays["log_loss"][mask].mean())})
        for year in np.unique(years):
            mask = years == year
            recent_rows.append({"window": window, "period": f"year_{year}", "draws": int(mask.sum()),
                                "average_hits": float(arrays["hits_top6"][mask].mean()),
                                "brier": float(arrays["brier"][mask].mean()),
                                "log_loss": float(arrays["log_loss"][mask].mean())})
        best_year = max(np.unique(years), key=lambda y: arrays["hits_top6"][years == y].mean())
        mask = years != best_year
        recent_rows.append({"window": window, "period": f"exclude_best_year_{best_year}", "draws": int(mask.sum()),
                            "average_hits": float(arrays["hits_top6"][mask].mean()),
                            "brier": float(arrays["brier"][mask].mean()),
                            "log_loss": float(arrays["log_loss"][mask].mean())})
    pd.DataFrame(recent_rows).to_csv(STAGE2 / "recent50_analysis.csv", index=False)

    # 上一期資訊：相同 Batch expanding、相同外層期別。
    variant_specs = {
        "Full": (BASE_FEATURE_COLUMNS, True),
        "RemoveInLastDraw": ([c for c in BASE_FEATURE_COLUMNS if c != "in_last_draw"], True),
        "RemoveTransition": ([c for c in BASE_FEATURE_COLUMNS if c != "transition_z"], True),
        "RemoveBoth": ([c for c in BASE_FEATURE_COLUMNS if c not in {"in_last_draw", "transition_z"}], True),
        "OnlyInLastDraw": (["in_last_draw"], False),
        "OnlyTransition": (["transition_z"], False),
    }
    previous_probabilities: dict[str, np.ndarray] = {"Full": batch_probability}
    for name, (columns, include_number) in variant_specs.items():
        if name == "Full": continue
        previous_probabilities[name] = batch_expanding(data, features, start=OUTER_START,
                                                        c=selection.selected_c, feature_columns=columns,
                                                        include_number=include_number)[3]
    previous_probabilities.update({
        "RepeatLastDraw": baseline_probabilities("PreviousDrawOnly", features, matrix, targets),
        "ExcludeLastDraw": baseline_probabilities("ExcludeLastDraw", features, matrix, targets),
        "ExactUniform": baseline_probabilities("ExactUniformBaseline", features, matrix, targets),
    })
    full_arrays = metric_arrays(batch_probability, actual)
    previous_rows, paired_rows = [], []
    for name, probability in previous_probabilities.items():
        arrays = metric_arrays(probability, actual)
        hit_diff = arrays["hits_top6"] - full_arrays["hits_top6"]
        brier_diff = arrays["brier"] - full_arrays["brier"]
        log_diff = arrays["log_loss"] - full_arrays["log_loss"]
        previous_rows.append({"model": name, "prediction_draws": len(targets),
                              "average_hits": float(arrays["hits_top6"].mean()),
                              "brier": float(arrays["brier"].mean()), "log_loss": float(arrays["log_loss"].mean()),
                              "hit_difference_vs_full": float(hit_diff.mean()),
                              "brier_difference_vs_full": float(brier_diff.mean()),
                              "log_loss_difference_vs_full": float(log_diff.mean()),
                              "paired_hit_ci_low": bootstrap_difference(hit_diff)[0],
                              "paired_hit_ci_high": bootstrap_difference(hit_diff)[1],
                              "block_hit_ci_low": bootstrap_difference(hit_diff, block=20)[0],
                              "block_hit_ci_high": bootstrap_difference(hit_diff, block=20)[1],
                              "mean_top6_jaccard_vs_full": float(np.mean([
                                  len(set(stable_top_k(probability[i], 6)) & set(stable_top_k(batch_probability[i], 6))) /
                                  len(set(stable_top_k(probability[i], 6)) | set(stable_top_k(batch_probability[i], 6)))
                                  for i in range(len(targets))]))})
        for index, target in enumerate(targets):
            paired_rows.append({"target_index": target, "model": name, "hit_difference_vs_full": hit_diff[index],
                                "brier_difference_vs_full": brier_diff[index], "log_loss_difference_vs_full": log_diff[index]})
    previous_frame = pd.DataFrame(previous_rows)
    p_values = [exact_hit_p(int(metric_arrays(previous_probabilities[name], actual)["hits_top6"].sum()), len(targets))
                for name in previous_frame["model"]]
    previous_frame["raw_hit_p"] = p_values
    previous_frame["holm_hit_p"] = multipletests(p_values, method="holm")[1]
    previous_frame.to_csv(STAGE2 / "previous_draw_analysis.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(STAGE2 / "previous_draw_paired_differences.csv", index=False)

    # Gap 分箱、Wilson CI、置換與年度係數方向。
    gap_source = features[features["target_index"] >= 500].copy()
    gap_source["gap_bin"] = pd.qcut(gap_source["gap"], q=8, duplicates="drop")
    gap_rows = []
    for category, group in gap_source.groupby("gap_bin", observed=True):
        successes, count = int(group["target"].sum()), len(group)
        ci = binomtest(successes, count).proportion_ci(.95, method="wilson")
        gap_rows.append({"row_type": "bin", "period": "all", "gap_bin": str(category), "observations": count,
                         "successes": successes, "next_draw_rate": successes / count,
                         "ci_low": ci.low, "ci_high": ci.high})
    coefficient_rows = []
    for year in sorted(pd.to_datetime(data.iloc[features["target_index"].unique()]["draw_date"]).dt.year.unique()):
        indices = features.loc[pd.to_datetime(features["draw_date"]).dt.year < year]
        if indices["target_index"].nunique() < 100: continue
        model = make_model(c=selection.selected_c)
        model.fit(indices, indices["target"])
        numeric_names = BASE_FEATURE_COLUMNS
        coefficient = float(model.named_steps["model"].coef_[0][numeric_names.index("gap_z")])
        coefficient_rows.append({"row_type": "coefficient", "period": f"through_{year-1}", "gap_bin": "",
                                 "observations": len(indices), "successes": np.nan, "next_draw_rate": np.nan,
                                 "ci_low": np.nan, "ci_high": np.nan, "gap_coefficient": coefficient})
    gap_frame = pd.concat([pd.DataFrame(gap_rows), pd.DataFrame(coefficient_rows)], ignore_index=True)
    gap_frame.to_csv(STAGE2 / "gap_analysis.csv", index=False)

    # 年度共同模型穩定性。
    yearly_rows = []
    years = pd.to_datetime(data.iloc[targets]["draw_date"]).dt.year.to_numpy()
    for name, probability in probability_by_model.items():
        arrays = metric_arrays(probability, actual)
        for year in np.unique(years):
            mask = years == year
            yearly_rows.append({"year": year, "model": name, "draws": int(mask.sum()),
                                "average_hits": float(arrays["hits_top6"][mask].mean()),
                                "brier": float(arrays["brier"][mask].mean()),
                                "log_loss": float(arrays["log_loss"][mask].mean())})
    pd.DataFrame(yearly_rows).to_csv(STAGE2 / "yearly_stability.csv", index=False)

    # 最新主推薦：正式主結果為 batch expanding，參數只由 inner 選定。
    latest_model = make_model(c=selection.selected_c)
    latest_model.fit(features, features["target"])
    next_features = build_next_features(data)
    latest_probability = normalize_and_shrink_probabilities(latest_model.predict_proba(next_features)[:, 1], SHRINK_STRENGTH)
    latest_numbers = sorted(stable_top_k(latest_probability, 6))
    write_json(STAGE2 / "actual_run_summary.json", {
        "data_rows": len(data), "data_start": data.iloc[0]["draw_date"], "data_end": data.iloc[-1]["draw_date"],
        "data_sha256": sha256_file(DATA_PATH), "inner_selected_model": selection.selected_model,
        "batch_selected_C": selection.selected_c, "latest_numbers": latest_numbers,
        "runtime_seconds": time.perf_counter() - started,
    })


def run_null(runs: int = SAME_PIPELINE_RUNS, workers: int = 4) -> None:
    """執行或續跑 same-pipeline null，定期寫 parquet checkpoint。"""
    output = STAGE2 / "same_pipeline_null_results.parquet"
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame()
    completed = set(existing.get("simulation_id", pd.Series(dtype=int)).astype(int).tolist())
    tasks = [(simulation_id, BASE_SEED + 1_000_000 + simulation_id)
             for simulation_id in range(1, runs + 1) if simulation_id not in completed]
    rows = existing.to_dict("records")
    started = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, result in enumerate(executor.map(_simulate_task, tasks, chunksize=1), 1):
                rows.append(result)
                if index % 10 == 0 or index == len(tasks):
                    pd.DataFrame(rows).sort_values("simulation_id").to_parquet(output, index=False)
                    print(f"same-pipeline {len(rows)}/{runs} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    frame = pd.DataFrame(rows).sort_values("simulation_id").reset_index(drop=True)
    if len(frame) != runs:
        raise RuntimeError(f"same-pipeline 實際 {len(frame)}，要求 {runs}")
    actual = json.loads((STAGE2 / "actual_run_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(STAGE2 / "unified_model_comparison.csv")
    selected_name = actual["inner_selected_model"]
    real = comparison[comparison["model"] == selected_name].iloc[0]
    frame["higher_hits_than_real"] = frame["average_hits"] >= real["average_hits"]
    frame["better_brier_than_real"] = frame["brier"] <= real["brier"]
    frame["better_log_loss_than_real"] = frame["log_loss"] <= real["log_loss"]
    frame.to_parquet(output, index=False)
    hit_count = int(frame["higher_hits_than_real"].sum())
    brier_count = int(frame["better_brier_than_real"].sum())
    log_count = int(frame["better_log_loss_than_real"].sum())
    def p_ci(count: int) -> tuple[float, list[float]]:
        ci = binomtest(count, len(frame)).proportion_ci(.95, method="wilson")
        return (count + 1) / (len(frame) + 1), [float(ci.low), float(ci.high)]
    hit_p, hit_ci = p_ci(hit_count); brier_p, brier_ci = p_ci(brier_count); log_p, log_ci = p_ci(log_count)
    summary = {
        "runs": len(frame), "history_draws_each": 2148, "inner_draws": OUTER_START - INNER_START,
        "outer_prediction_draws": OUTER_DRAWS, "selection_rule": SELECTION_RULE,
        "real_selected_model": selected_name,
        "real_metrics": {"average_hits": real["average_hits"], "brier": real["brier"], "log_loss": real["log_loss"]},
        "familywise_hit_p_plus_one": hit_p, "familywise_hit_ci95": hit_ci,
        "familywise_brier_p_plus_one": brier_p, "familywise_brier_ci95": brier_ci,
        "familywise_log_loss_p_plus_one": log_p, "familywise_log_loss_ci95": log_ci,
        "real_hit_percentile": float(100 * (frame["average_hits"] < real["average_hits"]).mean()),
        "random_forest_included": False,
        "limitation": "共同外層僅最後20期；RandomForest因計算成本未納入。每份均包含相同8-C sklearn Batch候選、Online SGD及九個基準。",
    }
    write_json(STAGE2 / "same_pipeline_null_summary.json", summary)
    frame["selected_model"].value_counts().rename_axis("selected_model").reset_index(name="count").assign(
        proportion=lambda x: x["count"] / len(frame)).to_csv(STAGE2 / "same_pipeline_model_selection_counts.csv", index=False)
    write_json(STAGE2 / "same_pipeline_null_config.json", {
        "base_seed": BASE_SEED, "seed_formula": "20260728 + 1000000 + simulation_id",
        "runs": runs, "workers": workers, "periods": 2148,
        "inner_start": INNER_START, "outer_start": OUTER_START, "outer_draws": OUTER_DRAWS,
        "candidate_models": CANDIDATE_NAMES, "C_grid": C_GRID,
        "shrink_strength": SHRINK_STRENGTH, "selection_rule": SELECTION_RULE,
        "same_feature_builder": "next_draw_predictor_audited.build_feature_dataset",
    })


def _simulate_task(task: tuple[int, int]) -> dict[str, Any]:
    return simulate_same_pipeline(*task)


def _simulate_simplified_task(task: tuple[int, int]) -> dict[str, Any]:
    return simulate_simplified_pipeline(*task)


def run_simplified_null(runs: int = SIMPLIFIED_RUNS, workers: int = 4) -> None:
    """執行或續跑 10k 非 ML 補充 null，不得用其 p 值替代完整 ML 流程。"""
    output = STAGE2 / "simplified_null_results.parquet"
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame()
    completed = set(existing.get("simulation_id", pd.Series(dtype=int)).astype(int).tolist())
    tasks = [(simulation_id, BASE_SEED + 3_000_000 + simulation_id)
             for simulation_id in range(1, runs + 1) if simulation_id not in completed]
    rows = existing.to_dict("records")
    started = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, result in enumerate(executor.map(_simulate_simplified_task, tasks, chunksize=1), 1):
                rows.append(result)
                if index % 100 == 0 or index == len(tasks):
                    pd.DataFrame(rows).sort_values("simulation_id").to_parquet(output, index=False)
                    print(f"simplified {len(rows)}/{runs} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    frame = pd.DataFrame(rows).sort_values("simulation_id").reset_index(drop=True)
    if len(frame) != runs: raise RuntimeError(f"simplified 實際 {len(frame)}，要求 {runs}")
    write_json(STAGE2 / "simplified_null_summary.json", {
        "runs": len(frame), "includes_sklearn_ml": False,
        "candidate_models": [name for name in CANDIDATE_NAMES if "FullFeature" not in name],
        "mean_hits": float(frame["average_hits"].mean()), "mean_brier": float(frame["brier"].mean()),
        "mean_log_loss": float(frame["log_loss"].mean()),
        "warning": "此簡化流程不得用作完整 ML same-pipeline family-wise p。",
    })


def _bootstrap_recommendation(task: tuple[int, int], features: pd.DataFrame,
                              next_features: pd.DataFrame, c_value: float) -> dict[str, Any]:
    iteration, seed = task
    rng = np.random.default_rng(seed)
    groups = features["target_index"].unique()
    sampled = rng.choice(groups, size=len(groups), replace=True)
    row_indices = np.concatenate([np.arange((target - FEATURE_MIN_HISTORY) * 49,
                                             (target - FEATURE_MIN_HISTORY + 1) * 49) for target in sampled])
    boot = features.iloc[row_indices]
    model = make_model(c=c_value, seed=seed)
    model.fit(boot, boot["target"])
    probability = normalize_and_shrink_probabilities(model.predict_proba(next_features)[:, 1], SHRINK_STRENGTH)
    return {"variant_type": "draw_group_bootstrap", "iteration": iteration, "seed": seed,
            "parameter": "", "numbers": " ".join(map(str, sorted(stable_top_k(probability, 6))))}


def run_stability() -> None:
    data = load_lotto_data(DATA_PATH)
    features = build_feature_dataset(data, FEATURE_MIN_HISTORY)
    next_features = build_next_features(data)
    actual = json.loads((STAGE2 / "actual_run_summary.json").read_text(encoding="utf-8"))
    c_value = float(actual["batch_selected_C"])
    reference = set(actual["latest_numbers"])
    tasks = [(index, BASE_SEED + 2_000_000 + index) for index in range(1, BOOTSTRAP_RUNS + 1)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda task: _bootstrap_recommendation(task, features, next_features, c_value), tasks))

    # 100 次刪除最後 1..20 期（五個預先固定 seed repetition）；模型本身近乎 deterministic，仍完整保存。
    for repetition in range(5):
        for removed in range(1, 21):
            subset = data.iloc[:-removed].reset_index(drop=True)
            subset_features = build_feature_dataset(subset, FEATURE_MIN_HISTORY)
            model = make_model(c=c_value, seed=BASE_SEED + repetition)
            model.fit(subset_features, subset_features["target"])
            probability = normalize_and_shrink_probabilities(model.predict_proba(build_next_features(subset))[:, 1], SHRINK_STRENGTH)
            rows.append({"variant_type": "delete_last", "iteration": repetition * 20 + removed,
                         "seed": BASE_SEED + repetition, "parameter": removed,
                         "numbers": " ".join(map(str, sorted(stable_top_k(probability, 6))))})
    for value in C_GRID:
        model = make_model(c=value); model.fit(features, features["target"])
        probability = normalize_and_shrink_probabilities(model.predict_proba(next_features)[:, 1], SHRINK_STRENGTH)
        rows.append({"variant_type": "C", "iteration": len(rows) + 1, "seed": BASE_SEED,
                     "parameter": value, "numbers": " ".join(map(str, sorted(stable_top_k(probability, 6))))})
    base_model = make_model(c=c_value); base_model.fit(features, features["target"])
    raw = base_model.predict_proba(next_features)[:, 1]
    for value in (0, .05, .10, .25, .50, 1.0):
        probability = normalize_and_shrink_probabilities(raw, value)
        rows.append({"variant_type": "shrink", "iteration": len(rows) + 1, "seed": BASE_SEED,
                     "parameter": value, "numbers": " ".join(map(str, sorted(stable_top_k(probability, 6))))})
    for minimum in (30, 60, 150, 300, 500):
        subset_features = build_feature_dataset(data, minimum)
        model = make_model(c=c_value); model.fit(subset_features, subset_features["target"])
        probability = normalize_and_shrink_probabilities(model.predict_proba(next_features)[:, 1], SHRINK_STRENGTH)
        rows.append({"variant_type": "min_history", "iteration": len(rows) + 1, "seed": BASE_SEED,
                     "parameter": minimum, "numbers": " ".join(map(str, sorted(stable_top_k(probability, 6))))})
    for windows in ((10,20,50), (20,50,100), (30,100,200), (50,100,200)):
        alt_features = build_feature_dataset(data, FEATURE_MIN_HISTORY, recent_windows=windows)
        alt_next = build_next_features(data, recent_windows=windows)
        model = make_model(c=c_value); model.fit(alt_features, alt_features["target"])
        probability = normalize_and_shrink_probabilities(model.predict_proba(alt_next)[:, 1], SHRINK_STRENGTH)
        rows.append({"variant_type": "windows", "iteration": len(rows) + 1, "seed": BASE_SEED,
                     "parameter": "-".join(map(str, windows)), "numbers": " ".join(map(str, sorted(stable_top_k(probability, 6))))})
    frame = pd.DataFrame(rows)
    sets = [set(map(int, text.split())) for text in frame["numbers"]]
    frame["jaccard_vs_reference"] = [len(values & reference) / len(values | reference) for values in sets]
    frame.to_csv(STAGE2 / "recommendation_stability_full.csv", index=False)
    frequencies = pd.DataFrame({"number": range(1, 50),
                                "selection_count": [sum(number in values for values in sets) for number in range(1, 50)]})
    frequencies["selection_proportion"] = frequencies["selection_count"] / len(frame)
    frequencies.to_csv(STAGE2 / "number_selection_frequency.csv", index=False)
    combinations = Counter(tuple(sorted(values)) for values in sets)
    jaccard = frame["jaccard_vs_reference"]
    summary = {"bootstrap_runs": BOOTSTRAP_RUNS, "delete_last_runs": 100, "total_variants": len(frame),
               "reference_numbers": sorted(reference), "mean_jaccard": float(jaccard.mean()),
               "jaccard_ci95": [float(jaccard.quantile(.025)), float(jaccard.quantile(.975))],
               "most_common_combination": list(combinations.most_common(1)[0][0]),
               "most_common_combination_count": combinations.most_common(1)[0][1],
               "most_stable_numbers": frequencies.nlargest(6, "selection_proportion")["number"].tolist(),
               "least_stable_numbers": frequencies.nsmallest(6, "selection_proportion")["number"].tolist(),
               "stable_unique_combination": combinations.most_common(1)[0][1] > len(frame) / 2,
               "conclusion": "模型不存在穩定的唯一推薦組合。" if combinations.most_common(1)[0][1] <= len(frame) / 2 else "存在過半數一致組合。"}
    write_json(STAGE2 / "recommendation_stability_summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("actual", "null", "simplified", "stability"))
    parser.add_argument("--runs", type=int, default=SAME_PIPELINE_RUNS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    STAGE2.mkdir(exist_ok=True)
    if args.phase == "actual": run_actual()
    elif args.phase == "null": run_null(args.runs, args.workers)
    elif args.phase == "simplified": run_simplified_null(args.runs, args.workers)
    else: run_stability()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
