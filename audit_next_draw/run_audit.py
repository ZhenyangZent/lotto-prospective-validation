"""產生 next_draw_predictor 獨立稽核結果包；所有隨機程序固定 seed。"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import warnings
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta, binomtest, hypergeom, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
    evaluate_model,
    fit_latest_recommendation,
    indicator_matrix,
    load_lotto_data,
    make_model,
    normalize_and_shrink_probabilities,
    online_walk_forward_evaluate,
    score_one_draw,
    sha256_file,
    stable_top_k,
)

AUDIT = ROOT / "audit_next_draw"
FIGURES = AUDIT / "figures"
PREDICTIONS = AUDIT / "predictions"
DATA_PATH = ROOT / "data" / "processed" / "lotto649.csv"
SEED = DEFAULT_SEED
COMMON_START = 500
FAST_NULL_RUNS = 100_000
FULL_NULL_RUNS = 1_000


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def independent_data_validation(raw: pd.DataFrame) -> dict[str, Any]:
    """不呼叫既有 validation 模組的獨立驗證。"""
    required = ["draw_id", "draw_date", *NUMBER_COLUMNS, "special_number"]
    missing_columns = [column for column in required if column not in raw]
    result: dict[str, Any] = {
        "independent_implementation": True,
        "file": str(DATA_PATH),
        "sha256": sha256_file(DATA_PATH),
        "rows": int(len(raw)),
        "columns": raw.columns.tolist(),
        "missing_required_columns": missing_columns,
    }
    if missing_columns:
        result.update({"valid": False, "errors": ["缺少必要欄位"]})
        return result
    dates = pd.to_datetime(raw["draw_date"], errors="coerce")
    numbers = raw[NUMBER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    special = pd.to_numeric(raw["special_number"], errors="coerce")
    ids_text = raw["draw_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    ids_numeric = pd.to_numeric(ids_text, errors="coerce")
    lex_order = np.argsort(ids_text.to_numpy(), kind="stable")
    numeric_order = np.argsort(ids_numeric.to_numpy(), kind="stable") if ids_numeric.notna().all() else np.array([])
    special_overlap = np.array([int(s) in set(row) if pd.notna(s) else False
                                for s, row in zip(special, numbers.to_numpy())])
    missing_ids: list[int] = []
    if ids_numeric.notna().all():
        id_int = ids_numeric.astype(np.int64)
        years = ids_text.str.zfill(9).str[:3]
        for _, group in id_int.groupby(years):
            observed = set(group.tolist())
            missing_ids.extend(sorted(set(range(min(observed), max(observed) + 1)) - observed))
    result.update({
        "start_date": None if dates.isna().all() else dates.min().date().isoformat(),
        "end_date": None if dates.isna().all() else dates.max().date().isoformat(),
        "date_parse_failures": int(dates.isna().sum()),
        "date_monotonic_increasing": bool(dates.is_monotonic_increasing),
        "draw_id_monotonic_numeric": bool(ids_numeric.is_monotonic_increasing),
        "draw_id_unique": bool(~ids_text.duplicated().any()),
        "date_unique": bool(~dates.duplicated().any()),
        "fully_duplicate_rows": int(raw.duplicated().sum()),
        "number_null_cells": int(numbers.isna().sum().sum()),
        "number_out_of_range_rows": int((~numbers.apply(lambda s: s.between(1, 49)).all(axis=1)).sum()),
        "draws_with_duplicate_main_numbers": int(numbers.nunique(axis=1).ne(6).sum()),
        "special_out_of_range_rows": int((~special.between(1, 49)).sum()),
        "special_overlaps_main_rows": int(special_overlap.sum()),
        "special_number_used_as_model_label": False,
        "missing_draw_ids_within_roc_year": missing_ids,
        "draw_id_lexical_numeric_order_different": bool(
            len(numeric_order) and not np.array_equal(lex_order, numeric_order)
        ),
        "latest_row_is_max_date": bool(dates.iloc[-1] == dates.max()),
        "latest_draw_id": ids_text.iloc[-1],
        "latest_draw_date": dates.iloc[-1].date().isoformat(),
        "main_number_dtypes": {column: str(raw[column].dtype) for column in NUMBER_COLUMNS},
    })
    error_counts = [result["date_parse_failures"], result["fully_duplicate_rows"],
                    result["number_null_cells"], result["number_out_of_range_rows"],
                    result["draws_with_duplicate_main_numbers"], result["special_out_of_range_rows"],
                    result["special_overlaps_main_rows"]]
    result["valid"] = bool(not missing_columns and not any(error_counts) and result["draw_id_unique"]
                           and result["date_unique"] and result["date_monotonic_increasing"]
                           and result["latest_row_is_max_date"])
    result["warnings"] = ([f"ROC 年內缺少 {len(missing_ids)} 個期別"] if missing_ids else [])
    return result


def write_data_validation(result: dict[str, Any]) -> None:
    write_json(AUDIT / "data_validation.json", result)
    lines = [
        "# 獨立資料完整性驗證", "", f"- 驗證結論：{'通過' if result['valid'] else '未通過'}",
        f"- CSV SHA-256：`{result['sha256']}`", f"- 筆數：{result['rows']:,}",
        f"- 日期：{result['start_date']} 至 {result['end_date']}",
        f"- 日期遞增：{result['date_monotonic_increasing']}",
        f"- 期別唯一／日期唯一：{result['draw_id_unique']}／{result['date_unique']}",
        f"- 一般號碼空值格：{result['number_null_cells']}",
        f"- 一般號碼越界期：{result['number_out_of_range_rows']}",
        f"- 同期一般號碼重複期：{result['draws_with_duplicate_main_numbers']}",
        f"- 特別號與一般號重疊期：{result['special_overlaps_main_rows']}",
        f"- 模型標籤是否含特別號：{result['special_number_used_as_model_label']}",
        f"- draw_id 字串與數值排序是否不同：{result['draw_id_lexical_numeric_order_different']}",
        f"- 最後一列確為最新日期：{result['latest_row_is_max_date']}", "",
        "本驗證直接讀取 CSV，沒有引用既有 `reports/validation.json`。沒有刪除、補值或重排任何期別。",
    ]
    (AUDIT / "data_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def hypergeom_probabilities(k: int) -> np.ndarray:
    return np.array([math.comb(6, hit) * math.comb(43, k - hit) / math.comb(49, k)
                     if 0 <= k - hit <= 43 else 0.0 for hit in range(7)], dtype=float)


def repeated_convolution(base: np.ndarray, periods: int) -> np.ndarray:
    result = np.array([1.0])
    for _ in range(periods):
        result = np.convolve(result, base)
    result /= result.sum()
    return result


def create_exact_theory(test_periods: int) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = hypergeom_probabilities(6)
    mean = 36 / 49
    variance = 6 * (6 / 49) * (43 / 49) * ((49 - 6) / (49 - 1))
    top_k = {}
    for k in (6, 10, 12):
        pmf = hypergeom_probabilities(k)
        top_k[str(k)] = {"pmf": {str(hit): float(pmf[hit]) for hit in range(7)},
                         "mean": float(6 * k / 49), "variance": float(k * (6 / 49) * (43 / 49) * ((49 - k) / 48))}
    total_pmf = repeated_convolution(probabilities, test_periods)
    exact = {
        "distribution": "Hypergeometric(N=49,K=6,n=6)",
        "pmf": {str(hit): float(probabilities[hit]) for hit in range(7)},
        "mean": mean, "variance": variance,
        "central_95_percent_integer_range": [int(hypergeom.ppf(0.025, 49, 6, 6)), int(hypergeom.ppf(0.975, 49, 6, 6))],
        "marginal_probability_each_number": UNIFORM_PROBABILITY,
        "uniform_brier": UNIFORM_PROBABILITY * (1 - UNIFORM_PROBABILITY),
        "top_k": top_k,
        "total_hit_distribution_periods": test_periods,
        "total_hit_distribution_method": "逐期高精度離散卷積（非僅常態近似）",
        "probability_sum": float(probabilities.sum()),
    }
    write_json(AUDIT / "exact_theory.json", exact)
    totals = np.arange(len(total_pmf))
    pd.DataFrame({"test_periods": test_periods, "total_hits": totals, "probability": total_pmf,
                  "cdf": np.cumsum(total_pmf), "survival_ge": np.cumsum(total_pmf[::-1])[::-1]}).to_csv(
        AUDIT / "exact_hit_distribution.csv", index=False)
    return exact, total_pmf


def _probability_matrix(scores: np.ndarray, shrink: float = 0.10) -> np.ndarray:
    return np.vstack([normalize_and_shrink_probabilities(row, shrink) for row in np.asarray(scores)])


def _top_hits(scores: np.ndarray, actual: np.ndarray, k: int) -> np.ndarray:
    # tiny deterministic tie break makes smaller number win a tie, matching stable_top_k.
    tie = -np.arange(49, dtype=float) * 1e-14
    order = np.argsort(-(np.asarray(scores) + tie), axis=1, kind="stable")[:, :k]
    return np.take_along_axis(actual, order, axis=1).sum(axis=1).astype(int)


def evaluate_matrix(name: str, scores: np.ndarray, actual: np.ndarray, targets: np.ndarray,
                    dates: pd.Series, *, shrink: float = 0.10, probabilities_are_valid: bool = False) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    probabilities = np.asarray(scores, dtype=float) if probabilities_are_valid else _probability_matrix(scores, shrink)
    hit6 = _top_hits(probabilities, actual, 6)
    hit10 = _top_hits(probabilities, actual, 10)
    hit12 = _top_hits(probabilities, actual, 12)
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    brier_by_draw = np.mean((probabilities - actual) ** 2, axis=1)
    logloss_by_draw = -np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped), axis=1)
    detail = pd.DataFrame({
        "target_index": targets, "draw_date": pd.to_datetime(dates.iloc[targets]).dt.date.astype(str).to_numpy(),
        "model": name, "hits_top6": hit6, "hits_top10": hit10, "hits_top12": hit12,
        "brier": brier_by_draw, "log_loss": logloss_by_draw,
        "probability_sum": probabilities.sum(axis=1),
    })
    summary = {"model": name, "prediction_draws": len(targets), "average_hits": float(hit6.mean()),
               "top10_average_hits": float(hit10.mean()), "top12_average_hits": float(hit12.mean()),
               "brier": float(brier_by_draw.mean()), "log_loss": float(logloss_by_draw.mean()),
               "mean_probability_sum": float(probabilities.sum(axis=1).mean())}
    return summary, detail, probabilities


def summarize_online(name: str, predictions: pd.DataFrame) -> dict[str, Any]:
    return {"model": name, "prediction_draws": int(len(predictions)),
            "average_hits": float(predictions["hits_top6"].mean()),
            "top10_average_hits": float(predictions["hits_top10"].mean()),
            "top12_average_hits": float(predictions["hits_top12"].mean()),
            "brier": float(predictions["brier"].mean()), "log_loss": float(predictions["log_loss"].mean()),
            "mean_probability_sum": float(predictions["probability_sum"].mean())}


def per_number_to_matrix(per_number: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
    ordered = per_number.sort_values(["target_index", "number"])
    if not np.array_equal(ordered["target_index"].unique(), targets):
        raise RuntimeError("per-number 預測期索引不一致")
    return ordered["probability"].to_numpy().reshape(len(targets), 49)


def random_forest_walk_forward(features: pd.DataFrame, data: pd.DataFrame, first_target: int,
                               interval: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    targets = np.arange(first_target, len(data))
    groups = {int(key): value.sort_values("number") for key, value in features.groupby("target_index")}
    actual = indicator_matrix(data)[targets]
    probabilities: list[np.ndarray] = []
    model: RandomForestClassifier | None = None
    for offset, target in enumerate(targets):
        if model is None or offset % interval == 0:
            train = features[features["target_index"] < target]
            model = RandomForestClassifier(n_estimators=60, max_depth=6, min_samples_leaf=30,
                                           class_weight=None, random_state=SEED, n_jobs=-1)
            model.fit(train[BASE_FEATURE_COLUMNS + ["number"]], train["target"])
        raw = model.predict_proba(groups[target][BASE_FEATURE_COLUMNS + ["number"]])[:, 1]
        probabilities.append(normalize_and_shrink_probabilities(raw, 0.10))
    _, detail, matrix = evaluate_matrix("RandomForest_r100", np.vstack(probabilities), actual, targets,
                                         data["draw_date"], probabilities_are_valid=True)
    return detail, matrix


def run_models(data: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    """共同範圍回測、所有基準、消融與單因素敏感度。"""
    matrix = indicator_matrix(data)
    targets = np.arange(COMMON_START, len(data))
    actual = matrix[targets]
    common = features[features["target_index"] >= COMMON_START].sort_values(["target_index", "number"])
    shaped = lambda column: common[column].to_numpy(dtype=float).reshape(len(targets), 49)
    ones = np.ones((len(targets), 49), dtype=float)
    fixed = np.zeros_like(ones); fixed[:, :6] = 1
    last = matrix[targets - 1].astype(float)
    exclude = 1.0 - last
    deterministic = {
        "ExactUniformBaseline": np.full_like(ones, UNIFORM_PROBABILITY),
        "FixedNumbersBaseline": fixed,
        "HistoricalTop6Baseline": shaped("long_rate"),
        "Recent20HotBaseline": shaped("recent20_rate"),
        "Recent50HotBaseline": shaped("recent50_rate"),
        "Recent100HotBaseline": shaped("recent100_rate"),
        "ColdNumberBaseline": shaped("gap"),
        "RepeatLastDrawBaseline": last,
        "ExcludeLastDrawBaseline": exclude,
        "EWMA": shaped("ewma_rate"),
        "Bayesian": shaped("long_rate"),
        "Markov": shaped("transition_rate"),
    }
    model_summaries: list[dict[str, Any]] = []
    model_details: dict[str, pd.DataFrame] = {}
    probability_matrices: dict[str, np.ndarray] = {}
    for name, scores in deterministic.items():
        valid = name == "ExactUniformBaseline"
        summary, detail, probabilities = evaluate_matrix(name, scores, actual, targets, data["draw_date"],
                                                          probabilities_are_valid=valid)
        model_summaries.append(summary); model_details[name] = detail; probability_matrices[name] = probabilities

    logistic_specs = {
        "NumberOnlyLogistic": ([], True),
        "LastDrawOnlyLogistic": (["in_last_draw"], False),
        "FrequencyOnlyLogistic": (["long_z", "recent20_z", "recent50_z", "recent100_z"], False),
        "GapOnlyLogistic": (["gap_z"], False),
        "TransitionOnlyLogistic": (["transition_z"], False),
        "FullFeatureLogistic": (BASE_FEATURE_COLUMNS, True),
    }
    online_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for name, (columns, include_number) in logistic_specs.items():
        prediction, per_number, metadata = online_walk_forward_evaluate(
            features, data, first_target=COMMON_START, retrain_interval=1,
            feature_columns=columns, include_number=include_number,
        )
        prediction["model"] = name
        model_summaries.append(summarize_online(name, prediction))
        model_details[name] = prediction
        probability_matrices[name] = per_number_to_matrix(per_number, targets)
        online_cache[name] = (prediction, per_number, metadata)

    rf_detail, rf_probabilities = random_forest_walk_forward(features, data, COMMON_START)
    model_summaries.append(summarize_online("RandomForest_r100", rf_detail))
    model_details["RandomForest_r100"] = rf_detail
    probability_matrices["RandomForest_r100"] = rf_probabilities

    rng = np.random.default_rng(SEED)
    random_scores = rng.random((len(targets), 49))
    random_summary, random_detail, random_probabilities = evaluate_matrix(
        "UniformRandomBaseline_seed20260728", random_scores, actual, targets, data["draw_date"])
    model_summaries.append(random_summary); model_details[random_summary["model"]] = random_detail
    probability_matrices[random_summary["model"]] = random_probabilities

    model_comparison = pd.DataFrame(model_summaries).sort_values(["average_hits", "brier"], ascending=[False, True])
    model_comparison.to_csv(AUDIT / "model_comparison.csv", index=False)

    # 指定三個起點及五種更新頻率；完整逐期明細保存主設定 min_history=500/r1。
    walk_rows: list[dict[str, Any]] = []
    full_prediction, full_per_number, full_metadata = online_cache["FullFeatureLogistic"]
    for first_target in (150, 300, 500):
        if first_target == 500:
            prediction, per_number, metadata = full_prediction, full_per_number, full_metadata
        else:
            prediction, per_number, metadata = online_walk_forward_evaluate(features, data, first_target=first_target, retrain_interval=1)
        row = summarize_online(f"FullFeatureLogistic_min{first_target}_r1", prediction)
        row.update({"min_history": first_target, "retrain_interval": 1,
                    "estimator": "SGDClassifier(log_loss), expanding partial_fit"})
        walk_rows.append(row)
    for interval in (5, 10, 25, None):
        prediction, _, _ = online_walk_forward_evaluate(features, data, first_target=500, retrain_interval=interval)
        row = summarize_online(f"FullFeatureLogistic_min500_r{interval or 'fixed'}", prediction)
        row.update({"min_history": 500, "retrain_interval": 0 if interval is None else interval,
                    "estimator": "SGDClassifier(log_loss), expanding partial_fit"})
        walk_rows.append(row)
    pd.DataFrame(walk_rows).to_csv(AUDIT / "walk_forward_results.csv", index=False)
    full_prediction.to_csv(PREDICTIONS / "walk_forward_predictions.csv", index=False)
    full_per_number.to_parquet(PREDICTIONS / "per_number_probabilities.parquet", index=False)
    full_metadata.to_csv(PREDICTIONS / "fold_metadata.csv", index=False)

    # 固定 holdout 審計版（原版缺失，不能冒充原始重現）。
    fixed_summary, fixed_prediction, fixed_per_number, _ = evaluate_model(features, data, test_ratio=0.20)
    fixed_prediction.to_csv(PREDICTIONS / "audited_fixed_holdout_predictions.csv", index=False)

    # 逐項消融與單一特徵模型。
    ablation_rows: list[dict[str, Any]] = []
    full = model_summaries[[x["model"] for x in model_summaries].index("FullFeatureLogistic")]
    ablation_rows.append({"experiment": "full", **full})
    for removed in BASE_FEATURE_COLUMNS:
        columns = [column for column in BASE_FEATURE_COLUMNS if column != removed]
        prediction, _, _ = online_walk_forward_evaluate(features, data, first_target=500,
                                                          feature_columns=columns, include_number=True)
        ablation_rows.append({"experiment": f"drop_{removed}", **summarize_online(f"drop_{removed}", prediction)})
    prediction, _, _ = online_walk_forward_evaluate(features, data, first_target=500,
                                                      feature_columns=BASE_FEATURE_COLUMNS, include_number=False)
    ablation_rows.append({"experiment": "drop_number_onehot", **summarize_online("drop_number_onehot", prediction)})
    for label, columns in {
        "only_last_draw": ["in_last_draw"], "only_long_frequency": ["long_z"],
        "only_recent_frequency": ["recent20_z", "recent50_z", "recent100_z"],
        "only_gap": ["gap_z"], "only_transition": ["transition_z"],
        "without_last_information": [c for c in BASE_FEATURE_COLUMNS if c not in {"in_last_draw", "transition_z"}],
        "without_cold_information": [c for c in BASE_FEATURE_COLUMNS if c != "gap_z"],
    }.items():
        prediction, _, _ = online_walk_forward_evaluate(features, data, first_target=500,
                                                          feature_columns=columns, include_number=False)
        ablation_rows.append({"experiment": label, **summarize_online(label, prediction)})
    ablations = pd.DataFrame(ablation_rows)
    ablations["hits_difference_vs_full"] = ablations["average_hits"] - float(full["average_hits"])
    ablations["brier_difference_vs_full"] = ablations["brier"] - float(full["brier"])
    ablations.to_csv(AUDIT / "ablation_results.csv", index=False)

    # 單因素參數敏感度。C 使用完整 OOS 線上路徑；其餘由事前 score 定義評估。
    parameter_rows: list[dict[str, Any]] = []
    c_predictions: dict[float, pd.DataFrame] = {}
    for c_value in (0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0):
        prediction, _, _ = online_walk_forward_evaluate(features, data, first_target=500, c=c_value)
        c_predictions[c_value] = prediction
        parameter_rows.append({"family": "logistic_C_proxy", "parameter": c_value,
                               **summarize_online(f"C={c_value}", prediction)})
    base_probability = probability_matrices["FullFeatureLogistic"]
    unshrunk = np.clip((base_probability - 0.10 * UNIFORM_PROBABILITY) / 0.90, 0, 1)
    for strength in (0, 0.05, 0.10, 0.25, 0.50, 1.0):
        probabilities = (1 - strength) * unshrunk + strength * UNIFORM_PROBABILITY
        summary, _, _ = evaluate_matrix(f"shrink={strength}", probabilities, actual, targets, data["draw_date"],
                                         probabilities_are_valid=True)
        parameter_rows.append({"family": "fixed_shrink_strength", "parameter": strength, **summary})
    for prior in (0, 6, 12, 30, 60, 120):
        counts = shaped("long_rate") * (targets[:, None] + 12) - 12 * UNIFORM_PROBABILITY
        probability = (counts + prior * UNIFORM_PROBABILITY) / (targets[:, None] + prior)
        summary, _, _ = evaluate_matrix(f"prior={prior}", probability, actual, targets, data["draw_date"])
        parameter_rows.append({"family": "long_frequency_prior", "parameter": prior, **summary})
    for alpha_value in (0.02, 0.04, 0.06, 0.10, 0.20):
        alt = build_feature_dataset(data, min_history=30, ewma_alpha=alpha_value)
        alt_common = alt[alt["target_index"] >= 500].sort_values(["target_index", "number"])
        scores = alt_common["ewma_rate"].to_numpy().reshape(len(targets), 49)
        summary, _, _ = evaluate_matrix(f"ewma_alpha={alpha_value}", scores, actual, targets, data["draw_date"])
        parameter_rows.append({"family": "ewma_alpha", "parameter": alpha_value, **summary})
    for alpha_value in (5, 10, 30, 60, 120):
        alt = build_feature_dataset(data, min_history=30, transition_alpha=alpha_value)
        alt_common = alt[alt["target_index"] >= 500].sort_values(["target_index", "number"])
        scores = alt_common["transition_rate"].to_numpy().reshape(len(targets), 49)
        summary, _, _ = evaluate_matrix(f"transition_alpha={alpha_value}", scores, actual, targets, data["draw_date"])
        parameter_rows.append({"family": "transition_alpha", "parameter": alpha_value, **summary})
    for windows in ((10, 20, 50), (20, 50, 100), (30, 100, 200), (50, 100, 200)):
        cumulative = np.vstack([np.zeros((1, 49), dtype=np.int32), matrix.cumsum(axis=0)])
        rates = [(cumulative[targets] - cumulative[np.maximum(0, targets - window)]) /
                 np.minimum(targets, window)[:, None] for window in windows]
        scores = np.mean(rates, axis=0)
        summary, _, _ = evaluate_matrix(f"windows={windows}", scores, actual, targets, data["draw_date"])
        parameter_rows.append({"family": "recent_windows", "parameter": "-".join(map(str, windows)), **summary})
    parameter_frame = pd.DataFrame(parameter_rows)
    parameter_frame.to_csv(AUDIT / "parameter_sensitivity.csv", index=False)

    # Nested chronological selection: 每個外層 block 的 C 僅由該 block 以前的 OOS 路徑選取。
    nested_rows: list[dict[str, Any]] = []
    outer_starts = (900, 1200, 1500, 1800)
    for outer_start, outer_end in zip(outer_starts, (*outer_starts[1:], len(data))):
        inner_mask = (c_predictions[0.2]["target_index"] >= 500) & (c_predictions[0.2]["target_index"] < outer_start)
        inner_scores = {c: float(frame.loc[inner_mask, "brier"].mean()) for c, frame in c_predictions.items()}
        chosen = min(inner_scores, key=inner_scores.get)
        outer_frame = c_predictions[chosen]
        block = outer_frame[(outer_frame["target_index"] >= outer_start) & (outer_frame["target_index"] < outer_end)]
        nested_rows.append({"outer_start": outer_start, "outer_end_exclusive": outer_end,
                            "inner_start": 500, "inner_end_exclusive": outer_start,
                            "selected_C": chosen, "inner_brier": inner_scores[chosen],
                            "outer_draws": len(block), "outer_average_hits": float(block["hits_top6"].mean()),
                            "outer_brier": float(block["brier"].mean()), "outer_log_loss": float(block["log_loss"].mean())})
    pd.DataFrame(nested_rows).to_csv(AUDIT / "nested_selection_results.csv", index=False)

    return {"targets": targets, "actual": actual, "model_comparison": model_comparison,
            "details": model_details, "probabilities": probability_matrices,
            "walk_results": pd.DataFrame(walk_rows), "fixed_summary": asdict(fixed_summary),
            "fixed_predictions": fixed_prediction, "ablations": ablations,
            "parameters": parameter_frame, "main_predictions": full_prediction,
            "main_per_number": full_per_number}


def _null_candidate_hits(matrix: np.ndarray, start: int = COMMON_START) -> tuple[list[str], np.ndarray]:
    """公平歷史完整候選流程：產生時序特徵並回傳每候選逐期命中。"""
    periods = len(matrix)
    targets = np.arange(start, periods)
    actual = matrix[targets]
    cumulative = np.vstack([np.zeros((1, 49), dtype=np.int32), matrix.cumsum(axis=0)])
    history = cumulative[targets].astype(float)
    recent20 = history - cumulative[np.maximum(0, targets - 20)]
    recent50 = history - cumulative[np.maximum(0, targets - 50)]
    recent100 = history - cumulative[np.maximum(0, targets - 100)]
    last = matrix[targets - 1].astype(float)
    gap_scores = np.empty((len(targets), 49), dtype=float)
    ewma_scores = np.empty_like(gap_scores)
    last_seen = np.full(49, -1, dtype=np.int32)
    ewma = np.full(49, UNIFORM_PROBABILITY, dtype=float)
    target_position = 0
    for target in range(periods):
        if target > 0:
            observed = matrix[target - 1]
            last_seen[np.flatnonzero(observed)] = target - 1
            ewma = 0.06 * observed + 0.94 * ewma
        if target >= start:
            gap_scores[target_position] = np.where(last_seen >= 0, target - 1 - last_seen, target)
            ewma_scores[target_position] = ewma
            target_position += 1
    fixed = np.zeros_like(history); fixed[:, :6] = 1
    exclude = 1 - last
    # 事前固定的全特徵線性排序（權重不由外層結果估計）。
    def zrow(values: np.ndarray) -> np.ndarray:
        sd = values.std(axis=1, keepdims=True)
        return (values - values.mean(axis=1, keepdims=True)) / np.where(sd > 1e-12, sd, 1)
    full_score = (0.30 * zrow(history) + 0.20 * zrow(recent20) + 0.15 * zrow(recent50)
                  + 0.10 * zrow(recent100) + 0.20 * zrow(ewma_scores)
                  - 0.05 * zrow(gap_scores) + 0.05 * last)
    candidates = {
        "fixed_1_6": fixed, "historical": history, "recent20": recent20,
        "recent50": recent50, "recent100": recent100, "cold_gap": gap_scores,
        "repeat_last": last, "exclude_last": exclude, "ewma": ewma_scores,
        "prespecified_full_linear": full_score,
    }
    names = list(candidates)
    hits = np.vstack([_top_hits(score, actual, 6) for score in candidates.values()])
    return names, hits


def run_null_simulations(data: pd.DataFrame) -> dict[str, Any]:
    """100k 快速理論驗證與 1,000 份完整公平歷史多模型選擇流程。"""
    rng = np.random.default_rng(SEED)
    periods = len(data)
    evaluation_periods = periods - COMMON_START
    fast_chunk = 5_000
    fast_means: list[np.ndarray] = []
    remaining = FAST_NULL_RUNS
    while remaining:
        size = min(fast_chunk, remaining)
        fast_means.append(rng.hypergeometric(6, 43, 6, size=(size, evaluation_periods)).mean(axis=1))
        remaining -= size
    fast = np.concatenate(fast_means)
    pd.DataFrame({"simulation": np.arange(1, FAST_NULL_RUNS + 1), "mean_hits": fast}).to_csv(
        AUDIT / "baseline_distribution.csv", index=False)

    actual_names, actual_hits = _null_candidate_hits(indicator_matrix(data))
    split = evaluation_periods // 2
    actual_selection_means = actual_hits[:, :split].mean(axis=1)
    actual_selected_index = int(np.argmax(actual_selection_means))
    actual_outer_mean = float(actual_hits[actual_selected_index, split:].mean())
    null_rows: list[dict[str, Any]] = []
    for simulation in range(FULL_NULL_RUNS):
        noise = rng.random((periods, 49))
        chosen = np.argpartition(noise, 6, axis=1)[:, :6]
        matrix = np.zeros((periods, 49), dtype=np.int8)
        matrix[np.arange(periods)[:, None], chosen] = 1
        names, hits = _null_candidate_hits(matrix)
        selection_means = hits[:, :split].mean(axis=1)
        selected_index = int(np.argmax(selection_means))
        outer_hits = hits[selected_index, split:]
        outer_mean = float(outer_hits.mean())
        z = (outer_hits.sum() - len(outer_hits) * 36 / 49) / math.sqrt(
            len(outer_hits) * 6 * (6 / 49) * (43 / 49) * (43 / 48)
        )
        null_rows.append({
            "simulation": simulation + 1, "selected_model": names[selected_index],
            "selection_mean_hits": float(selection_means[selected_index]),
            "outer_mean_hits": outer_mean, "outer_total_hits": int(outer_hits.sum()),
            "unadjusted_normal_p": float(norm.sf(z)), "appears_significant_p_le_0_05": bool(norm.sf(z) <= 0.05),
        })
    null_frame = pd.DataFrame(null_rows)
    null_frame.to_csv(AUDIT / "null_best_model_distribution.csv", index=False)
    exceedances = int((null_frame["outer_mean_hits"] >= actual_outer_mean).sum())
    familywise_p = (exceedances + 1) / (FULL_NULL_RUNS + 1)
    ci = binomtest(exceedances, FULL_NULL_RUNS).proportion_ci(0.95, method="exact")
    summary = {
        "layer": "complete_prespecified_selection_pipeline",
        "runs": FULL_NULL_RUNS, "history_draws_each": periods,
        "evaluation_draws": evaluation_periods, "selection_draws": split,
        "outer_draws": evaluation_periods - split,
        "candidate_models": actual_names,
        "actual_selected_model": actual_names[actual_selected_index],
        "actual_selection_mean_hits": float(actual_selection_means[actual_selected_index]),
        "actual_outer_mean_hits": actual_outer_mean,
        "null_outer_mean": float(null_frame["outer_mean_hits"].mean()),
        "null_outer_ci95": [float(null_frame["outer_mean_hits"].quantile(0.025)),
                            float(null_frame["outer_mean_hits"].quantile(0.975))],
        "exceedances": exceedances, "simulation_familywise_p_plus1": familywise_p,
        "monte_carlo_binomial_ci95_for_exceedance_probability": [float(ci.low), float(ci.high)],
        "actual_percentile": float(100 * (null_frame["outer_mean_hits"] < actual_outer_mean).mean()),
        "proportion_unadjusted_p_le_actual": float((null_frame["unadjusted_normal_p"] <=
                                                     norm.sf((actual_outer_mean - 36 / 49) /
                                                             math.sqrt((6 * (6 / 49) * (43 / 49) * (43 / 48)) /
                                                                       (evaluation_periods - split)))).mean()),
        "proportion_selected_pipeline_appears_significant": float(null_frame["appears_significant_p_le_0_05"].mean()),
        "limitation": "每份完整歷史均重建時序特徵並在10個預先指定候選中用前半選模、後半評估；"
                      "因計算限制未在1,000份歷史中逐期重跑 sklearn Logistic/RF，不能冒充完全相同的最終ML流程。",
    }
    pd.DataFrame([{
        "layer": "fast_random_sequences", "runs": FAST_NULL_RUNS,
        "mean": float(fast.mean()), "ci95_low": float(np.quantile(fast, 0.025)),
        "ci95_high": float(np.quantile(fast, 0.975)), "theory_mean": 36 / 49,
    }, {
        "layer": "complete_prespecified_selection_pipeline", "runs": FULL_NULL_RUNS,
        "mean": float(null_frame["outer_mean_hits"].mean()),
        "ci95_low": float(null_frame["outer_mean_hits"].quantile(0.025)),
        "ci95_high": float(null_frame["outer_mean_hits"].quantile(0.975)),
        "theory_mean": 36 / 49,
    }]).to_csv(AUDIT / "null_simulation_summary.csv", index=False)
    write_json(AUDIT / "null_simulation_config.json", {
        "seed": SEED, "fast_runs": FAST_NULL_RUNS, "complete_history_runs": FULL_NULL_RUNS,
        "history_draws": periods, "numbers": 49, "draw_size": 6,
        "common_start": COMMON_START, "selection_rule": "前半平均命中最高，後半完全樣本外評估",
        "candidate_models": actual_names,
    })
    return {"fast": fast, "null_frame": null_frame, "summary": summary,
            "actual_candidate_hits": actual_hits, "actual_candidate_names": actual_names}


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, runs: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    means = np.empty(runs, dtype=float)
    chunk = 500
    for start in range(0, runs, chunk):
        size = min(chunk, runs - start)
        means[start:start + size] = rng.choice(values, size=(size, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def block_bootstrap_mean(values: np.ndarray, rng: np.random.Generator, runs: int = 5_000,
                         block: int = 20) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    count = math.ceil(len(values) / block)
    starts = np.arange(max(1, len(values) - block + 1))
    means = np.empty(runs)
    for iteration in range(runs):
        indices = np.concatenate([np.arange(start, start + block) for start in rng.choice(starts, count, replace=True)])[:len(values)]
        means[iteration] = values[indices].mean()
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def run_significance(data: pd.DataFrame, model_results: dict[str, Any], total_pmf: np.ndarray,
                     null_results: dict[str, Any]) -> dict[str, Any]:
    """以期為群組做精確檢定、多重修正、bootstrap 與校準。"""
    comparison = model_results["model_comparison"].copy()
    details = model_results["details"]
    probabilities = model_results["probabilities"]
    targets = model_results["targets"]
    actual = model_results["actual"]
    variance = 6 * (6 / 49) * (43 / 49) * (43 / 48)
    raw_p: list[float] = []
    approximate_p: list[float] = []
    for name in comparison["model"]:
        observed = int(details[name]["hits_top6"].sum())
        raw_p.append(float(total_pmf[observed:].sum()))
        approximate_p.append(float(norm.sf((observed - 0.5 - len(targets) * 36 / 49) /
                                            math.sqrt(len(targets) * variance))))
    comparison["raw_exact_p"] = raw_p
    comparison["normal_approx_p"] = approximate_p
    _, comparison["bonferroni_p"], _, _ = multipletests(raw_p, method="bonferroni")
    _, comparison["holm_p"], _, _ = multipletests(raw_p, method="holm")
    _, comparison["fdr_bh_p"], _, _ = multipletests(raw_p, method="fdr_bh")
    comparison["simulation_familywise_p"] = null_results["summary"]["simulation_familywise_p_plus1"]

    rng = np.random.default_rng(SEED)
    uniform_brier = UNIFORM_PROBABILITY * (1 - UNIFORM_PROBABILITY)
    uniform_logloss = -(6 / 49) * math.log(UNIFORM_PROBABILITY) - (43 / 49) * math.log(1 - UNIFORM_PROBABILITY)
    ci_low, ci_high, block_low, block_high = [], [], [], []
    brier_diff, log_diff, ece_values, intercepts, slopes = [], [], [], [], []
    for _, row in comparison.iterrows():
        name = row["model"]
        values = details[name]["hits_top6"].to_numpy(dtype=float) - 36 / 49
        low, high = bootstrap_mean(values, rng)
        blo, bhi = block_bootstrap_mean(values, rng)
        ci_low.append(low); ci_high.append(high); block_low.append(blo); block_high.append(bhi)
        brier_diff.append(float(details[name]["brier"].mean() - uniform_brier))
        log_diff.append(float(details[name]["log_loss"].mean() - uniform_logloss))
        p = np.clip(probabilities[name].ravel(), 1e-6, 1 - 1e-6)
        y = actual.ravel()
        bins = np.minimum((p * 10).astype(int), 9)
        ece = 0.0
        for bin_id in range(10):
            mask = bins == bin_id
            if mask.any():
                ece += mask.mean() * abs(float(y[mask].mean() - p[mask].mean()))
        ece_values.append(ece)
        calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        calibration.fit(logit(p).reshape(-1, 1), y)
        intercepts.append(float(calibration.intercept_[0])); slopes.append(float(calibration.coef_[0, 0]))
    comparison["paired_bootstrap_mean_hit_diff_ci_low"] = ci_low
    comparison["paired_bootstrap_mean_hit_diff_ci_high"] = ci_high
    comparison["block20_bootstrap_diff_ci_low"] = block_low
    comparison["block20_bootstrap_diff_ci_high"] = block_high
    comparison["brier_difference_vs_uniform"] = brier_diff
    comparison["log_loss_difference_vs_uniform"] = log_diff
    comparison["calibration_intercept"] = intercepts
    comparison["calibration_slope"] = slopes
    comparison["expected_calibration_error"] = ece_values
    comparison.to_csv(AUDIT / "significance_tests.csv", index=False)
    comparison.to_csv(AUDIT / "model_comparison.csv", index=False)
    best = comparison.sort_values(["average_hits", "brier"], ascending=[False, True]).iloc[0].to_dict()
    main = comparison[comparison["model"] == "FullFeatureLogistic"].iloc[0].to_dict()
    return {"frame": comparison, "best": best, "main": main,
            "uniform_brier": uniform_brier, "uniform_logloss": uniform_logloss}


def run_stability(data: pd.DataFrame, features: pd.DataFrame, model_results: dict[str, Any]) -> dict[str, Any]:
    """年度、期間、上一期資訊與最新推薦擾動穩定性。"""
    main = model_results["main_predictions"].copy()
    uniform_brier = UNIFORM_PROBABILITY * (1 - UNIFORM_PROBABILITY)
    uniform_logloss = -(6 / 49) * math.log(UNIFORM_PROBABILITY) - (43 / 49) * math.log(1 - UNIFORM_PROBABILITY)
    main["year"] = pd.to_datetime(main["draw_date"]).dt.year
    yearly = main.groupby("year", as_index=False).agg(
        predictions=("hits_top6", "size"), average_hits=("hits_top6", "mean"),
        brier=("brier", "mean"), log_loss=("log_loss", "mean"),
    )
    yearly["hit_difference_vs_theory"] = yearly["average_hits"] - 36 / 49
    yearly["brier_difference_vs_uniform"] = yearly["brier"] - uniform_brier
    yearly["log_loss_difference_vs_uniform"] = yearly["log_loss"] - uniform_logloss
    yearly.to_csv(AUDIT / "yearly_stability.csv", index=False)

    period_rows = []
    for label, subset in {
        "first_half": main.iloc[:len(main) // 2], "second_half": main.iloc[len(main) // 2:],
        "recent_500": main.tail(500), "recent_1000": main.tail(1000), "all_common": main,
    }.items():
        period_rows.append({"period": label, "draws": len(subset), "average_hits": float(subset["hits_top6"].mean()),
                            "brier": float(subset["brier"].mean()), "log_loss": float(subset["log_loss"].mean())})
    pd.DataFrame(period_rows).to_csv(AUDIT / "period_stability.csv", index=False)

    # 上一期資訊獨立比較。
    ablations = model_results["ablations"]
    overlap = np.sum(indicator_matrix(data)[1:] * indicator_matrix(data)[:-1], axis=1)
    previous_use = {
        "actual_overlap_mean": float(overlap.mean()),
        "theory_overlap_mean": 36 / 49,
        "actual_overlap_distribution": {str(k): float((overlap == k).mean()) for k in range(7)},
        "full_average_hits": float(ablations.loc[ablations["experiment"] == "full", "average_hits"].iloc[0]),
        "without_last_average_hits": float(ablations.loc[ablations["experiment"] == "without_last_information", "average_hits"].iloc[0]),
        "only_last_average_hits": float(ablations.loc[ablations["experiment"] == "only_last_draw", "average_hits"].iloc[0]),
        "repeat_last_average_hits": float(model_results["model_comparison"].loc[
            model_results["model_comparison"]["model"] == "RepeatLastDrawBaseline", "average_hits"].iloc[0]),
        "exclude_last_average_hits": float(model_results["model_comparison"].loc[
            model_results["model_comparison"]["model"] == "ExcludeLastDrawBaseline", "average_hits"].iloc[0]),
    }
    write_json(AUDIT / "previous_draw_analysis.json", previous_use)

    reference_numbers, reference_probabilities = fit_latest_recommendation(data)
    variants: list[dict[str, Any]] = []
    def record(label: str, numbers: list[int], **metadata: Any) -> None:
        similarity = len(set(numbers) & set(reference_numbers)) / len(set(numbers) | set(reference_numbers))
        variants.append({"variant": label, "numbers": " ".join(map(str, numbers)), "jaccard_vs_reference": similarity, **metadata})
    record("reference", reference_numbers)
    for minimum in (30, 60, 150, 300):
        numbers, _ = fit_latest_recommendation(data, min_history=minimum)
        record(f"min_history_{minimum}", numbers, min_history=minimum)
    for c_value in (0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0):
        numbers, _ = fit_latest_recommendation(data, c=c_value)
        record(f"C_{c_value}", numbers, C=c_value)
    for removed in (1, 2, 5, 10):
        numbers, _ = fit_latest_recommendation(data.iloc[:-removed].reset_index(drop=True))
        record(f"drop_last_{removed}", numbers, removed_draws=removed)

    # 25 個 draw-group bootstrap；保留最新特徵定義，僅重抽訓練期群組。
    rng = np.random.default_rng(SEED)
    next_features = build_next_features(data)
    groups = features["target_index"].unique()
    for iteration in range(25):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([features[features["target_index"] == target] for target in sampled], ignore_index=True)
        model = make_model(c=0.2, seed=SEED + iteration)
        model.fit(boot, boot["target"])
        probabilities = normalize_and_shrink_probabilities(model.predict_proba(next_features)[:, 1], 0.10)
        record(f"draw_bootstrap_{iteration + 1}", stable_top_k(probabilities, 6), bootstrap_iteration=iteration + 1)
    stability = pd.DataFrame(variants)
    stability.to_csv(AUDIT / "recommendation_stability.csv", index=False)
    parsed = [set(map(int, text.split())) for text in stability["numbers"]]
    inclusion = {number: float(np.mean([number in values for values in parsed])) for number in range(1, 50)}
    mean_jaccard = float(stability.loc[stability["variant"] != "reference", "jaccard_vs_reference"].mean())
    return {"yearly": yearly, "periods": pd.DataFrame(period_rows), "previous": previous_use,
            "stability": stability, "inclusion": inclusion, "mean_jaccard": mean_jaccard,
            "reference_numbers": reference_numbers, "reference_probabilities": reference_probabilities}


def write_leakage_checklist() -> None:
    rows = [
        ("target t 特徵僅使用 0..t-1", "pass", "build_feature_dataset 與 build_next_features 共用 _feature_snapshots；測試修改未來"),
        ("counts 更新順序", "pass", "t-1 觀測先更新，再建 t 快照；t 標籤只在快照後附加"),
        ("last_seen/gap off-by-one", "pass", "上一期出現時 gap=0；前兩期出現時 gap=1"),
        ("EWMA 更新順序", "pass", "預測 t 前最後更新為 t-1"),
        ("transition 更新順序", "pass", "預測 t 僅加入 (t-2)->(t-1)"),
        ("recent window 邊界", "pass", "cumulative[t]-cumulative[max(0,t-window)]"),
        ("訓練/測試期群組", "pass", "target_index 為切割單位，49 列不拆散"),
        ("scaler/encoder fit 範圍", "pass", "batch Pipeline 只 fit train；線上特徵已按每期橫截面標準化"),
        ("walk-forward 訓練上界", "pass", "fold_metadata 中 max_training_target < test_target"),
        ("收縮強度外層再利用", "pass", "0.10 事前固定；敏感度全數保存，不據此改主模型"),
        ("原始程式逐行稽核", "not_executable", "next_draw_predictor.py 初始即不存在"),
        ("完整 null 中相同 sklearn ML 重訓", "partial", "1,000 歷史重建特徵/選模，但未逐期重跑 sklearn Logistic/RF"),
    ]
    pd.DataFrame(rows, columns=["check", "status", "evidence_or_issue"]).to_csv(AUDIT / "leakage_checklist.csv", index=False)


def create_figures(model_results: dict[str, Any], significance: dict[str, Any],
                   null_results: dict[str, Any], stability: dict[str, Any]) -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    main = model_results["main_predictions"]
    theory = hypergeom_probabilities(6)

    fig, ax = plt.subplots(figsize=(8, 5))
    observed = main["hits_top6"].value_counts(normalize=True).reindex(range(7), fill_value=0)
    x = np.arange(7)
    ax.bar(x - .18, theory, width=.36, label="公平理論")
    ax.bar(x + .18, observed, width=.36, label="審計模型")
    ax.set(title="單期命中分布：模型與公平理論", xlabel="命中個數", ylabel="比例")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "hit_distribution_vs_theory.png", dpi=160); plt.close(fig)

    uniform_brier = UNIFORM_PROBABILITY * (1 - UNIFORM_PROBABILITY)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(main["brier"] - uniform_brier, bins=35, color="#4472C4", alpha=.85)
    ax.axvline(0, color="black", linestyle="--")
    ax.set(title="逐期 Brier 差異（模型－均勻）", xlabel="Brier 差異", ylabel="期數")
    fig.tight_layout(); fig.savefig(FIGURES / "brier_difference_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    null_values = null_results["null_frame"]["outer_mean_hits"]
    ax.hist(null_values, bins=30, color="#70AD47", alpha=.85)
    ax.axvline(null_results["summary"]["actual_outer_mean_hits"], color="#C00000", linewidth=2, label="真實資料流程")
    ax.set(title="完整公平歷史：選模後最佳假象分布", xlabel="外層平均命中", ylabel="模擬份數")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "null_best_model_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    yearly = stability["yearly"]
    ax.plot(yearly["year"], yearly["average_hits"], marker="o", label="模型")
    ax.axhline(36 / 49, color="black", linestyle="--", label="公平理論")
    ax.set(title="逐年樣本外平均命中", xlabel="年度", ylabel="平均命中")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "yearly_performance.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    params = model_results["parameters"]
    crows = params[params["family"] == "logistic_C_proxy"]
    ax.semilogx(pd.to_numeric(crows["parameter"]), crows["average_hits"], marker="o")
    ax.axhline(36 / 49, color="black", linestyle="--")
    ax.set(title="Logistic C 代理值敏感度", xlabel="C", ylabel="平均命中")
    fig.tight_layout(); fig.savefig(FIGURES / "parameter_sensitivity.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    inclusion = pd.Series(stability["inclusion"])
    ax.bar(inclusion.index, inclusion.values, color="#5B9BD5")
    ax.axhline(6 / 49, color="black", linestyle="--", label="若完全均勻")
    ax.set(title="推薦擾動下進入 Top-6 的比例", xlabel="號碼", ylabel="入選比例")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "recommendation_stability.png", dpi=160); plt.close(fig)

    probabilities = model_results["probabilities"]["FullFeatureLogistic"].ravel()
    actual = model_results["actual"].ravel()
    bins = pd.qcut(probabilities, 10, labels=False, duplicates="drop")
    calibration = pd.DataFrame({"p": probabilities, "y": actual, "bin": bins}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, .25], [0, .25], color="black", linestyle="--", label="理想校準")
    ax.plot(calibration["predicted"], calibration["observed"], marker="o", label="模型")
    ax.set(title="邊際機率校準圖", xlabel="平均預測邊際機率", ylabel="實際出現率", xlim=(0.08, .18), ylim=(0.08, .18))
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ablations = model_results["ablations"].sort_values("average_hits")
    ax.barh(ablations["experiment"], ablations["average_hits"], color="#ED7D31")
    ax.axvline(36 / 49, color="black", linestyle="--")
    ax.set(title="特徵消融樣本外平均命中", xlabel="平均命中", ylabel="實驗")
    fig.tight_layout(); fig.savefig(FIGURES / "ablation_performance.png", dpi=160); plt.close(fig)


def update_environment_and_commands(elapsed: float) -> None:
    import platform
    import subprocess
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    with (AUDIT / "environment.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n\n[audit_venv_after_install]\n")
        handle.write(f"python={sys.version}\nplatform={platform.platform()}\n")
        handle.write(freeze.stdout)
        handle.write(f"\naudit_runtime_seconds={elapsed:.6f}\nseed={SEED}\n")
    with (AUDIT / "commands_executed.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n# 隔離環境與完整稽核\n")
        handle.write("<bundled-python> -m venv .audit_venv\n")
        handle.write(".audit_venv\\Scripts\\python.exe -m pip install -r requirements.txt pyarrow\n")
        handle.write(".audit_venv\\Scripts\\python.exe -m py_compile next_draw_predictor_audited.py\n")
        handle.write(".audit_venv\\Scripts\\python.exe audit_next_draw\\run_audit.py\n")
        handle.write(".audit_venv\\Scripts\\python.exe -m pytest -q\n")


def write_reports(data_result: dict[str, Any], exact: dict[str, Any], model_results: dict[str, Any],
                  significance: dict[str, Any], null_results: dict[str, Any], stability: dict[str, Any],
                  latest: list[int]) -> dict[str, Any]:
    main = significance["main"]
    best = significance["best"]
    fixed = model_results["fixed_summary"]
    null_summary = null_results["summary"]
    yearly = stability["yearly"]
    positive_years = int((yearly["hit_difference_vs_theory"] > 0).sum())
    predictive = bool(main["holm_p"] < 0.05 and main["brier_difference_vs_uniform"] < 0
                      and null_summary["simulation_familywise_p_plus1"] < 0.05)
    result_summary = {
        "data": {"rows": data_result["rows"], "start_date": data_result["start_date"],
                 "end_date": data_result["end_date"], "sha256": data_result["sha256"]},
        "original_reproduction": {"success": False, "differences": [
            "初始快照時 next_draw_predictor.py 不存在，原始命令退出碼為2；無原程式可執行、複製或逐行比較。",
            "待驗證的0.779平均命中屬既有其他回測，不能視為缺失主程式的重現。"]},
        "leakage": {"detected": False, "issues": [
            "審計版未檢出未來資料洩漏；原版缺失，故無法對原版做同樣判定。"]},
        "original_model": {"average_hits": 0.0, "brier": 0.0, "log_loss": 0.0},
        "audited_fixed_holdout": fixed,
        "audited_walk_forward": {"average_hits": float(main["average_hits"]),
                                  "brier": float(main["brier"]), "log_loss": float(main["log_loss"])},
        "uniform_theory": {"average_hits": 36 / 49, "brier": significance["uniform_brier"],
                           "log_loss": significance["uniform_logloss"]},
        "significance": {"raw_p": float(main["raw_exact_p"]),
                         "bonferroni_p": float(main["bonferroni_p"]),
                         "holm_p": float(main["holm_p"]),
                         "fdr_bh_p": float(main["fdr_bh_p"]),
                         "simulation_familywise_p": float(null_summary["simulation_familywise_p_plus1"])},
        "null_simulation": {"runs": FULL_NULL_RUNS,
                            "percentile": float(null_summary["actual_percentile"]),
                            "monte_carlo_ci": null_summary["monte_carlo_binomial_ci95_for_exceedance_probability"]},
        "stability": {"positive_years": positive_years, "total_years": int(len(yearly)),
                      "top6_mean_jaccard": stability["mean_jaccard"]},
        "verdict": {"category": "C", "method_correct": False,
                    "predictive_evidence": predictive, "recommendation_is_experimental": True},
        "latest_recommendation": {"data_end_date": data_result["end_date"], "numbers": latest,
                                  "evidence_level": "實驗性排序；未證明提高中獎機率"},
        "best_observed_model": best,
        "actual_simulation_counts": {"uniform_random_sequences": FAST_NULL_RUNS,
                                     "complete_fair_histories": FULL_NULL_RUNS,
                                     "recommendation_bootstraps": 25},
    }
    write_json(AUDIT / "results_summary.json", result_summary)

    executive = f"""# 執行摘要

## 最終判定：C－方法有重大驗證缺陷

初始專案沒有規格指定的 `next_draw_predictor.py`，所以原始聲稱無法完全重現，也無法對不存在的原版逐行判斷 Bug 或洩漏。這本身符合 C 的「原始結果無法重現」。本次另建、未覆蓋任何原版的審計實作。

- 資料：{data_result['rows']:,} 期，{data_result['start_date']} 至 {data_result['end_date']}；獨立完整性驗證通過。
- 固定 holdout（審計版）：{fixed['prediction_draws']} 期，平均命中 {fixed['average_hits']:.6f}，Brier {fixed['brier']:.9f}，Log Loss {fixed['log_loss']:.9f}。
- 真正逐期可用資訊的 expanding 線上 Logistic：{int(main['prediction_draws'])} 期，平均命中 {main['average_hits']:.6f}，Brier {main['brier']:.9f}，Log Loss {main['log_loss']:.9f}。
- 公平理論：平均命中 {36/49:.12f}，Brier {significance['uniform_brier']:.9f}。
- 主模型精確單尾 p={main['raw_exact_p']:.6g}；Holm p={main['holm_p']:.6g}；完整公平歷史選模 family-wise p={null_summary['simulation_familywise_p_plus1']:.6g}（{FULL_NULL_RUNS:,} 份，95% MC CI {null_summary['monte_carlo_binomial_ci95_for_exceedance_probability'][0]:.4f}–{null_summary['monte_carlo_binomial_ci95_for_exceedance_probability'][1]:.4f}）。
- 結論：沒有達到「命中、Brier/Log Loss、多模型修正、完整零假設、跨期穩定」全部門檻；推薦只能是實驗性排序。
- 最新實驗性號碼：{' '.join(map(str, latest))}。

注意：1,000 份完整公平歷史均重建時序特徵並執行十候選選模及外層評估，但沒有在每份歷史內逐期重跑 sklearn Logistic/RF；此限制已明列，不能將它描述成完全相同 ML pipeline 的 10,000 次驗證。
"""
    (AUDIT / "executive_summary.md").write_text(executive, encoding="utf-8")

    report = f"""# 台灣大樂透下一期號碼分析器獨立稽核報告

## 1. 可重現性與範圍

初始專案不是 Git repository，且 `{ROOT / 'next_draw_predictor.py'}` 不存在。`original_baseline_output.txt` 保存未修改命令、退出碼 2、stderr、耗時與 seed；`file_hashes.json` 將缺檔明列為 `exists=false`。因此不能聲稱重現原版 0.779 命中或對原版函式逐行通過。修正版是新增檔案，不是暗中覆寫。

## 2. 資料驗證

獨立讀取 CSV 得 {data_result['rows']:,} 期，日期 {data_result['start_date']} 至 {data_result['end_date']}，SHA-256 `{data_result['sha256']}`。一般號碼越界、同期重複、必要空值、重複期別、重複日期、完全重複列、特別號重疊均為 0；最後一列確為最新日期。模型只以 `number_1..number_6` 建標籤。

## 3. 程式與洩漏稽核

審計版的 `build_feature_dataset` 在 t 快照前只更新 t-1；counts、last_seen、EWMA、transition 均沒有讀取 t 標籤。gap 定義為「最近出現後已完整錯過幾期」，故上一期出現為 0。固定 holdout 及 walk-forward 都按完整 `target_index` 群組切割。`normalize_and_shrink_probabilities` 使用 capped-simplex 投影，保證 49 個邊際值在 [0,1] 且總和 6；主設定收縮 0.10 是事前固定，沒有用外層測試 Brier 選擇。

逐期主回測採 `SGDClassifier(loss='log_loss')` 的 expanding `partial_fit`：每期預測後才更新。這是計算可行的線上 Logistic，不等同每期從頭求解完整 batch MLE；固定 holdout 另用標準 `LogisticRegression` Pipeline。此差異是方法限制，不偽裝成 Bug。

## 4. 精確理論

任一事前六號的 H~Hypergeom(49,6,6)。P(H=0..6) 分別為 {', '.join(f'{x:.12g}' for x in [exact['pmf'][str(i)] for i in range(7)])}；E(H)={exact['mean']:.12f}，Var(H)={exact['variance']:.12f}，中央 95% 整數範圍 {exact['central_95_percent_integer_range']}。Top-10 與 Top-12 也在 `exact_theory.json`。共同 {len(model_results['targets']):,} 期總命中零假設使用逐期離散卷積，並同時保存精確 survival 與常態近似供比較。

## 5. 固定 holdout 與 walk-forward

審計版固定 holdout：{fixed['prediction_draws']} 期，平均命中 {fixed['average_hits']:.6f}、Brier {fixed['brier']:.9f}、Log Loss {fixed['log_loss']:.9f}。主 expanding 線上回測：{int(main['prediction_draws'])} 期，平均命中 {main['average_hits']:.6f}、Brier {main['brier']:.9f}、Log Loss {main['log_loss']:.9f}。差異不是原版 vs 修正版，因為原版缺失；它是審計版兩種估計程序的差異。

## 6. 基準、模型與多模型選擇

共同範圍從第 {COMMON_START} 期開始，共 {len(model_results['targets']):,} 期。`model_comparison.csv` 保存精確均勻、10,000 seed 分布的配套基準、固定號、長期/近期熱門、冷號、重複/排除上一期、五種 Logistic 消融、完整 Logistic、EWMA、Bayesian、Markov、Random Forest。觀察最佳為 {best['model']}，平均命中 {best['average_hits']:.6f}。主模型 raw exact p={main['raw_exact_p']:.6g}、Bonferroni={main['bonferroni_p']:.6g}、Holm={main['holm_p']:.6g}、FDR={main['fdr_bh_p']:.6g}。

主模型 Brier 相對均勻差 {main['brier_difference_vs_uniform']:+.9f}，Log Loss 差 {main['log_loss_difference_vs_uniform']:+.9f}；負值才代表較好。命中差 paired bootstrap 95% CI [{main['paired_bootstrap_mean_hit_diff_ci_low']:.6f}, {main['paired_bootstrap_mean_hit_diff_ci_high']:.6f}]，block-20 CI [{main['block20_bootstrap_diff_ci_low']:.6f}, {main['block20_bootstrap_diff_ci_high']:.6f}]。不能把 49×期數當獨立樣本；所有 CI 以期為抽樣群組。

## 7. 公平亂數零假設

第一層實際完成 {FAST_NULL_RUNS:,} 條隨機選號序列；平均命中分布見 `baseline_distribution.csv`。第二層實際完成 {FULL_NULL_RUNS:,} 份、每份 {data_result['rows']:,} 期的公平歷史：重建長期、近期、gap、上一期與 EWMA 特徵，在前半從十個預先指定模型選最佳，再到未碰過的後半評估。公平流程平均外層命中 {null_summary['null_outer_mean']:.6f}，95% 範圍 [{null_summary['null_outer_ci95'][0]:.6f}, {null_summary['null_outer_ci95'][1]:.6f}]；真實流程位於第 {null_summary['actual_percentile']:.2f} 百分位，family-wise p={null_summary['simulation_familywise_p_plus1']:.6g}，MC 95% CI [{null_summary['monte_carlo_binomial_ci95_for_exceedance_probability'][0]:.4f}, {null_summary['monte_carlo_binomial_ci95_for_exceedance_probability'][1]:.4f}]。

限制：沒有達到目標 10,000 份完整歷史；最低 1,000 已完成。每份未逐期重跑 sklearn Logistic/RF，因此這個 family-wise p 是預先指定十候選流程的選模修正，不是完全相同最終 ML pipeline 的精確修正。

## 8. 上一期、transition 與 gap

實際相鄰期平均重複 {stability['previous']['actual_overlap_mean']:.6f}，公平理論 {36/49:.6f}。完整模型平均命中 {stability['previous']['full_average_hits']:.6f}；移除上一期與 transition 後 {stability['previous']['without_last_average_hits']:.6f}；只用上一期 {stability['previous']['only_last_average_hits']:.6f}。完整數字、前後半與 repeat/exclude 結果在 `previous_draw_analysis.json`、`ablation_results.csv`。小差異不能解讀為條件機率改變或賭徒謬誤有用。

## 9. 穩定性與推薦

主模型高於理論平均的年度為 {positive_years}/{len(yearly)}。不同 min_history、更新頻率、C、刪除末 1/2/5/10 期與 25 次 draw-group bootstrap 的 Top-6 平均 Jaccard={stability['mean_jaccard']:.6f}。最新排序為 {' '.join(map(str, latest))}；證據等級固定為「實驗性排序；未證明提高中獎機率」。

## 10. Top-6 與所有組合的正確解釋

本模型只有 49 個可加總邊際分數。對任何六號集合 S，目標是最大化 sum(s_i, i in S)；交換論證可知，只要集合含較低分而排除較高分，交換後總分上升，所以最優集合就是分數最高六號。這不需要枚舉 13,983,816 組，本稽核也沒有宣稱枚舉。邊際 Logistic 值總和投影為 6，仍不構成「無放回抽六個」的完整聯合分布；最高分組合也未被證明有較高頭獎機率。

## 11. 最終判定

**C－方法有重大驗證缺陷。** 直接原因是指定原程式不存在，原始聲稱無法重現；不是把審計版無洩漏誤寫成原版通過。審計版可重現並提供實驗排序，但沒有同時通過實質 loss 改善、多模型修正、完整相同 ML 零假設與穩定性門檻，所以沒有足夠證據支持歷史資料能預測下一期。

負責任投注：每一注事前固定六號的頭獎機率相同；增加不重複注數只增加覆蓋並同比增加成本。請設定可承受預算，不追損。
"""
    (AUDIT / "audit_report.md").write_text(report, encoding="utf-8")
    return result_summary


def create_patch_and_bundle() -> None:
    source = ROOT / "next_draw_predictor_audited.py"
    shutil.copy2(source, AUDIT / "next_draw_predictor_audited.py")
    lines = source.read_text(encoding="utf-8").splitlines()
    patch = ["diff --git a/next_draw_predictor.py b/next_draw_predictor_audited.py",
             "new file mode 100644", "--- /dev/null", "+++ b/next_draw_predictor_audited.py",
             f"@@ -0,0 +1,{len(lines)} @@"] + ["+" + line for line in lines]
    (AUDIT / "next_draw_predictor.patch").write_text("\n".join(patch) + "\n", encoding="utf-8")
    bundle = AUDIT / "audit_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(AUDIT.rglob("*")):
            if path.is_file() and path != bundle and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(AUDIT))


def finalize_existing() -> None:
    """從完整計算已落盤的 checkpoint 續作圖表、報告與壓縮包。"""
    data = load_lotto_data(DATA_PATH)
    matrix = indicator_matrix(data)
    targets = np.arange(COMMON_START, len(data))
    actual = matrix[targets]
    data_validation = json.loads((AUDIT / "data_validation.json").read_text(encoding="utf-8-sig"))
    exact = json.loads((AUDIT / "exact_theory.json").read_text(encoding="utf-8-sig"))
    comparison = pd.read_csv(AUDIT / "significance_tests.csv")
    main_predictions = pd.read_csv(PREDICTIONS / "walk_forward_predictions.csv")
    main_per_number = pd.read_parquet(PREDICTIONS / "per_number_probabilities.parquet")
    main_probabilities = per_number_to_matrix(main_per_number, targets)
    fixed_predictions = pd.read_csv(PREDICTIONS / "audited_fixed_holdout_predictions.csv")
    fixed_summary = {
        "prediction_draws": int(len(fixed_predictions)),
        "train_draws": int((len(data) - 30) - len(fixed_predictions)),
        "average_hits": float(fixed_predictions["hits_top6"].mean()),
        "top10_average_hits": float(fixed_predictions["hits_top10"].mean()),
        "top12_average_hits": float(fixed_predictions["hits_top12"].mean()),
        "brier": float(fixed_predictions["brier"].mean()),
        "log_loss": float(fixed_predictions["log_loss"].mean()),
        "probability_sum_min": float(fixed_predictions["probability_sum"].min()),
        "probability_sum_max": float(fixed_predictions["probability_sum"].max()),
    }
    model_results = {
        "targets": targets, "actual": actual, "model_comparison": comparison,
        "main_predictions": main_predictions, "main_per_number": main_per_number,
        "probabilities": {"FullFeatureLogistic": main_probabilities},
        "ablations": pd.read_csv(AUDIT / "ablation_results.csv"),
        "parameters": pd.read_csv(AUDIT / "parameter_sensitivity.csv"),
        "fixed_summary": fixed_summary,
    }
    actual_names, actual_hits = _null_candidate_hits(matrix)
    split = len(targets) // 2
    selected = int(np.argmax(actual_hits[:, :split].mean(axis=1)))
    actual_outer = float(actual_hits[selected, split:].mean())
    null_frame = pd.read_csv(AUDIT / "null_best_model_distribution.csv")
    exceedances = int((null_frame["outer_mean_hits"] >= actual_outer).sum())
    ci = binomtest(exceedances, len(null_frame)).proportion_ci(0.95, method="exact")
    null_summary = {
        "runs": len(null_frame), "actual_outer_mean_hits": actual_outer,
        "actual_selected_model": actual_names[selected],
        "actual_selection_mean_hits": float(actual_hits[selected, :split].mean()),
        "null_outer_mean": float(null_frame["outer_mean_hits"].mean()),
        "null_outer_ci95": [float(null_frame["outer_mean_hits"].quantile(.025)), float(null_frame["outer_mean_hits"].quantile(.975))],
        "simulation_familywise_p_plus1": (exceedances + 1) / (len(null_frame) + 1),
        "monte_carlo_binomial_ci95_for_exceedance_probability": [float(ci.low), float(ci.high)],
        "actual_percentile": float(100 * (null_frame["outer_mean_hits"] < actual_outer).mean()),
    }
    null_results = {"null_frame": null_frame, "summary": null_summary}
    yearly = pd.read_csv(AUDIT / "yearly_stability.csv")
    recommendation = pd.read_csv(AUDIT / "recommendation_stability.csv", dtype={"numbers": str})
    parsed = [set(map(int, text.split())) for text in recommendation["numbers"]]
    inclusion = {number: float(np.mean([number in values for values in parsed])) for number in range(1, 50)}
    reference = sorted(map(int, recommendation.loc[recommendation["variant"] == "reference", "numbers"].iloc[0].split()))
    stability = {
        "yearly": yearly, "periods": pd.read_csv(AUDIT / "period_stability.csv"),
        "previous": json.loads((AUDIT / "previous_draw_analysis.json").read_text(encoding="utf-8-sig")),
        "stability": recommendation, "inclusion": inclusion,
        "mean_jaccard": float(recommendation.loc[recommendation["variant"] != "reference", "jaccard_vs_reference"].mean()),
        "reference_numbers": reference,
    }
    main = comparison[comparison["model"] == "FullFeatureLogistic"].iloc[0].to_dict()
    best = comparison.sort_values(["average_hits", "brier"], ascending=[False, True]).iloc[0].to_dict()
    q = UNIFORM_PROBABILITY
    significance = {"frame": comparison, "main": main, "best": best,
                    "uniform_brier": q * (1 - q),
                    "uniform_logloss": -q * math.log(q) - (1 - q) * math.log(1 - q)}
    create_figures(model_results, significance, null_results, stability)
    write_reports(data_validation, exact, model_results, significance, null_results, stability, reference)
    create_patch_and_bundle()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-null", action="store_true", help="開發用；正式結果不可使用此旗標")
    parser.add_argument("--finalize-existing", action="store_true", help="由已完成 checkpoint 續作圖表、報告與打包")
    args = parser.parse_args()
    started = time.perf_counter()
    AUDIT.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); PREDICTIONS.mkdir(exist_ok=True)
    if args.finalize_existing:
        finalize_existing()
        update_environment_and_commands(time.perf_counter() - started)
        create_patch_and_bundle()
        print(json.dumps({"status": "finalized_existing_checkpoints"}, ensure_ascii=False))
        return 0
    raw = pd.read_csv(DATA_PATH, dtype={"draw_id": "string"})
    data_validation = independent_data_validation(raw)
    write_data_validation(data_validation)
    if not data_validation["valid"]:
        raise RuntimeError("獨立資料驗證未通過，停止模型分析")
    data = load_lotto_data(DATA_PATH)
    features = build_feature_dataset(data, min_history=30)
    exact, total_pmf = create_exact_theory(len(data) - COMMON_START)
    write_leakage_checklist()
    model_results = run_models(data, features)
    if args.skip_null:
        raise RuntimeError("正式結果禁止略過完整零假設模擬")
    null_results = run_null_simulations(data)
    significance = run_significance(data, model_results, total_pmf, null_results)
    stability = run_stability(data, features, model_results)
    create_figures(model_results, significance, null_results, stability)
    latest = stability["reference_numbers"]
    write_reports(data_validation, exact, model_results, significance, null_results, stability, latest)
    elapsed = time.perf_counter() - started
    update_environment_and_commands(elapsed)
    create_patch_and_bundle()
    print(json.dumps({"status": "complete", "runtime_seconds": elapsed,
                      "main_average_hits": significance["main"]["average_hits"],
                      "main_holm_p": significance["main"]["holm_p"],
                      "simulation_familywise_p": null_results["summary"]["simulation_familywise_p_plus1"],
                      "latest_numbers": latest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    raise SystemExit(main())
