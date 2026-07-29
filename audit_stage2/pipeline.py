"""第二階段統一回測與 same-pipeline null 的共用實作。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from next_draw_predictor_audited import (
    BASE_FEATURE_COLUMNS,
    DEFAULT_SEED,
    NUMBER_COLUMNS,
    UNIFORM_PROBABILITY,
    build_feature_dataset,
    indicator_matrix,
    make_model,
    normalize_and_shrink_probabilities,
    online_walk_forward_evaluate,
    stable_top_k,
    walk_forward_evaluate,
)

FEATURE_MIN_HISTORY = 30
INNER_START = 2028
OUTER_START = 2128
OUTER_DRAWS = 20
C_GRID = (0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0)
SHRINK_STRENGTH = 0.10
SELECTION_RULE = "inner Brier ascending; then inner Log Loss ascending; then fixed simplicity rank"
MODEL_COMPLEXITY = {
    "ExactUniformBaseline": 0,
    "FixedNumbersBaseline": 1,
    "HistoricalTop6Baseline": 2,
    "Recent20HotBaseline": 3,
    "Recent50HotBaseline": 3,
    "Recent100HotBaseline": 3,
    "GapOnly": 4,
    "PreviousDrawOnly": 4,
    "TransitionOnly": 4,
    "FullFeatureBatchLogistic": 5,
    "FullFeatureOnlineSGD": 6,
}
CANDIDATE_NAMES = tuple(MODEL_COMPLEXITY)


@dataclass(frozen=True)
class SelectionResult:
    selected_model: str
    selected_c: float
    best_inner_brier: float
    best_inner_log_loss: float
    candidate_table: pd.DataFrame


def synthetic_dataframe(matrix: np.ndarray, dates: Sequence[Any] | None = None) -> pd.DataFrame:
    """由 0/1 開獎矩陣建立與真實資料同 schema 的最小 DataFrame。"""
    matrix = np.asarray(matrix, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[1] != 49 or not np.all(matrix.sum(axis=1) == 6):
        raise ValueError("matrix 必須為每列恰有六個 1 的 (T,49) 矩陣")
    chosen = np.sort(np.where(matrix == 1)[1].reshape(len(matrix), 6) + 1, axis=1)
    if dates is None:
        parsed_dates = pd.date_range("2000-01-01", periods=len(matrix), freq="D")
    else:
        parsed_dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        if len(parsed_dates) != len(matrix):
            raise ValueError("dates 長度必須等於期數")
    data = pd.DataFrame({"draw_id": [f"{index + 1:09d}" for index in range(len(matrix))],
                         "draw_date": parsed_dates})
    for index, column in enumerate(NUMBER_COLUMNS):
        data[column] = chosen[:, index]
    return data


def build_feature_dataset_fast(data: pd.DataFrame, min_history: int = FEATURE_MIN_HISTORY) -> pd.DataFrame:
    """與 audited builder 同定義的 NumPy 向量化版本，供大量 null 歷史重建。"""
    matrix = indicator_matrix(data)
    periods = len(matrix)
    if min_history < 1 or min_history >= periods:
        raise ValueError("min_history 必須至少為1且小於期數")
    targets = np.arange(min_history, periods)
    n_targets = len(targets)
    cumulative = np.vstack([np.zeros((1, 49), dtype=np.int32), matrix.cumsum(axis=0)])
    counts = cumulative[targets].astype(float)
    long_rate = (counts + 12 * UNIFORM_PROBABILITY) / (targets[:, None] + 12)
    rates: dict[int, np.ndarray] = {}
    for window in (20, 50, 100):
        begin = np.maximum(0, targets - window)
        rates[window] = (cumulative[targets] - cumulative[begin]) / np.maximum(1, targets - begin)[:, None]
    ewma_rate = np.empty((n_targets, 49), dtype=float)
    gap = np.empty((n_targets, 49), dtype=float)
    transition_rate = np.empty((n_targets, 49), dtype=float)
    ewma = np.full(49, UNIFORM_PROBABILITY)
    last_seen = np.full(49, -1, dtype=np.int32)
    success = np.zeros((2, 49), dtype=float)
    total = np.zeros((2, 49), dtype=float)
    output_index = 0
    for target in range(periods):
        if target > 0:
            observed = matrix[target - 1]
            present = np.flatnonzero(observed)
            last_seen[present] = target - 1
            ewma = .06 * observed + .94 * ewma
            if target >= 2:
                previous = matrix[target - 2]
                for state in (0, 1):
                    mask = previous == state
                    total[state, mask] += 1
                    success[state, mask] += observed[mask]
        if target >= min_history:
            previous_state = matrix[target - 1].astype(int)
            ewma_rate[output_index] = ewma
            gap[output_index] = np.where(last_seen >= 0, target - 1 - last_seen, target)
            transition_rate[output_index] = (
                success[previous_state, np.arange(49)] + 30 * UNIFORM_PROBABILITY
            ) / (total[previous_state, np.arange(49)] + 30)
            output_index += 1
    def zrow(values: np.ndarray) -> np.ndarray:
        deviation = values.std(axis=1, keepdims=True)
        return (values - values.mean(axis=1, keepdims=True)) / np.where(deviation > 1e-15, deviation, 1)
    repeat = lambda values: np.asarray(values).reshape(-1)
    frame = pd.DataFrame({
        "number": np.tile(np.arange(1, 50), n_targets),
        "long_z": repeat(zrow(long_rate)), "recent20_z": repeat(zrow(rates[20])),
        "recent50_z": repeat(zrow(rates[50])), "recent100_z": repeat(zrow(rates[100])),
        "ewma_z": repeat(zrow(ewma_rate)), "gap_z": repeat(zrow(gap)),
        "transition_z": repeat(zrow(transition_rate)),
        "in_last_draw": repeat(matrix[targets - 1]).astype(int),
        "long_rate": repeat(long_rate), "recent20_rate": repeat(rates[20]),
        "recent50_rate": repeat(rates[50]), "recent100_rate": repeat(rates[100]),
        "ewma_rate": repeat(ewma_rate), "gap": repeat(gap),
        "transition_rate": repeat(transition_rate),
        "history_draws": np.repeat(targets, 49), "target_index": np.repeat(targets, 49),
        "draw_id": np.repeat(data.iloc[targets]["draw_id"].astype(str).to_numpy(), 49),
        "draw_date": np.repeat(pd.to_datetime(data.iloc[targets]["draw_date"]).to_numpy(), 49),
        "target": repeat(matrix[targets]).astype(int),
    })
    return frame


def stable_score_top(scores: np.ndarray, k: int) -> np.ndarray:
    """逐列依分數降冪、號碼升冪選 k 個索引。"""
    scores = np.asarray(scores, dtype=float)
    tie = -np.arange(49) * 1e-14
    return np.argsort(-(scores + tie), axis=1, kind="stable")[:, :k]


def probabilities_from_scores(scores: np.ndarray, shrink: float = SHRINK_STRENGTH) -> np.ndarray:
    """逐期套用與真實流程相同的 capped-simplex 投影與固定收縮。"""
    return np.vstack([normalize_and_shrink_probabilities(row, shrink) for row in np.asarray(scores)])


def baseline_probabilities(
    name: str,
    features: pd.DataFrame,
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    shrink: float = SHRINK_STRENGTH,
    recent_window: int | None = None,
) -> np.ndarray:
    """僅使用 target 以前資訊建立指定基準的 49 號邊際預測。"""
    targets = np.asarray(targets, dtype=int)
    frame = features[features["target_index"].isin(targets)].sort_values(["target_index", "number"])
    if len(frame) != len(targets) * 49:
        raise ValueError("features 未涵蓋全部 target")
    shaped = lambda column: frame[column].to_numpy(dtype=float).reshape(len(targets), 49)
    if name == "ExactUniformBaseline":
        return np.full((len(targets), 49), UNIFORM_PROBABILITY)
    if name == "FixedNumbersBaseline":
        scores = np.zeros((len(targets), 49)); scores[:, :6] = 1
    elif name == "HistoricalTop6Baseline":
        scores = shaped("long_rate")
    elif name in {"Recent20HotBaseline", "Recent50HotBaseline", "Recent100HotBaseline"}:
        window = recent_window or int(name.removeprefix("Recent").removesuffix("HotBaseline"))
        cumulative = np.vstack([np.zeros((1, 49), dtype=np.int32), matrix.cumsum(axis=0)])
        scores = (cumulative[targets] - cumulative[np.maximum(0, targets - window)]) / np.minimum(targets, window)[:, None]
    elif name == "GapOnly":
        scores = shaped("gap")
    elif name == "PreviousDrawOnly":
        scores = matrix[targets - 1].astype(float)
    elif name == "TransitionOnly":
        scores = shaped("transition_rate")
    elif name == "ExcludeLastDraw":
        scores = 1.0 - matrix[targets - 1]
    else:
        raise KeyError(name)
    return probabilities_from_scores(scores, shrink)


def calibration_metrics(probabilities: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    """回傳 calibration intercept、slope、ECE；常數機率 slope 定義為 0。"""
    p = np.clip(np.asarray(probabilities, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    y = np.asarray(actual, dtype=int).ravel()
    transformed = logit(p)
    if np.std(transformed) < 1e-12:
        intercept, slope = float(logit(np.clip(y.mean(), 1e-6, 1 - 1e-6))), 0.0
    else:
        model = LogisticRegression(C=1e6, max_iter=500, solver="lbfgs")
        model.fit(transformed.reshape(-1, 1), y)
        intercept, slope = float(model.intercept_[0]), float(model.coef_[0, 0])
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    return intercept, slope, ece


def metric_arrays(probabilities: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """所有 loss 先按一期 49 號群組平均。"""
    probabilities = np.asarray(probabilities, dtype=float)
    actual = np.asarray(actual, dtype=np.int8)
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return {
        "hits_top6": np.take_along_axis(actual, stable_score_top(probabilities, 6), axis=1).sum(axis=1),
        "hits_top10": np.take_along_axis(actual, stable_score_top(probabilities, 10), axis=1).sum(axis=1),
        "hits_top12": np.take_along_axis(actual, stable_score_top(probabilities, 12), axis=1).sum(axis=1),
        "brier": np.mean((probabilities - actual) ** 2, axis=1),
        "log_loss": -np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped), axis=1),
    }


def evaluate_probabilities(name: str, probabilities: np.ndarray, actual: np.ndarray,
                           targets: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = metric_arrays(probabilities, actual)
    intercept, slope, ece = calibration_metrics(probabilities, actual)
    row = {"model": name, "prediction_draws": int(len(targets)),
           "average_hits": float(arrays["hits_top6"].mean()),
           "top10_average_hits": float(arrays["hits_top10"].mean()),
           "top12_average_hits": float(arrays["hits_top12"].mean()),
           "brier": float(arrays["brier"].mean()), "log_loss": float(arrays["log_loss"].mean()),
           "calibration_intercept": intercept, "calibration_slope": slope, "ece": ece,
           "probability_sum_min": float(probabilities.sum(axis=1).min()),
           "probability_sum_max": float(probabilities.sum(axis=1).max())}
    detail = pd.DataFrame({"target_index": targets, "model": name, **arrays})
    return row, detail


def prediction_frames_to_probabilities(per_number: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
    ordered = per_number.sort_values(["target_index", "number"])
    if not np.array_equal(ordered["target_index"].unique(), targets):
        raise ValueError("逐號預測期與要求期間不一致")
    return ordered["probability"].to_numpy(dtype=float).reshape(len(targets), 49)


def batch_expanding(
    data: pd.DataFrame,
    features: pd.DataFrame,
    *,
    start: int = OUTER_START,
    c: float = 0.20,
    feature_columns: Sequence[str] = BASE_FEATURE_COLUMNS,
    include_number: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    predictions, per_number, metadata = walk_forward_evaluate(
        features, data, first_target=start, retrain_interval=1, c=c,
        shrink_strength=SHRINK_STRENGTH, feature_columns=feature_columns,
        include_number=include_number,
    )
    predictions["model"] = "FullFeatureBatchLogistic"
    per_number["model"] = "FullFeatureBatchLogistic"
    metadata["model"] = "FullFeatureBatchLogistic"
    targets = np.arange(start, len(data))
    return predictions, per_number, metadata, prediction_frames_to_probabilities(per_number, targets)


def online_sgd(
    data: pd.DataFrame,
    features: pd.DataFrame,
    *,
    start: int = OUTER_START,
    c: float = 0.20,
    feature_columns: Sequence[str] = BASE_FEATURE_COLUMNS,
    include_number: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    predictions, per_number, metadata = online_walk_forward_evaluate(
        features, data, first_target=start, retrain_interval=1, c=c,
        shrink_strength=SHRINK_STRENGTH, feature_columns=feature_columns,
        include_number=include_number,
    )
    predictions["model"] = "OnlineSGDLogistic"
    per_number["model"] = "OnlineSGDLogistic"
    metadata["model"] = "OnlineSGDLogistic"
    targets = np.arange(start, len(data))
    return predictions, per_number, metadata, prediction_frames_to_probabilities(per_number, targets)


def fixed_logistic_probabilities(data: pd.DataFrame, features: pd.DataFrame, targets: np.ndarray,
                                 *, c: float = .20, feature_columns: Sequence[str] = BASE_FEATURE_COLUMNS,
                                 include_number: bool = True) -> np.ndarray:
    """在 targets 第一個期別以前 fit 一次，整段不更新。"""
    start = int(targets[0])
    train = features[features["target_index"] < start]
    model = make_model(c=c, feature_columns=feature_columns, include_number=include_number)
    model.fit(train, train["target"])
    test = features[features["target_index"].isin(targets)].sort_values(["target_index", "number"])
    raw = model.predict_proba(test)[:, 1].reshape(len(targets), 49)
    return probabilities_from_scores(raw, SHRINK_STRENGTH)


def select_inner_model(data: pd.DataFrame, features: pd.DataFrame) -> SelectionResult:
    """只使用 OUTER_START 以前資料與 inner validation 選模型及 Batch C。"""
    matrix = indicator_matrix(data)
    targets = np.arange(INNER_START, OUTER_START)
    actual = matrix[targets]
    rows: list[dict[str, Any]] = []
    probabilities_by_name: dict[str, np.ndarray] = {}
    for name in CANDIDATE_NAMES:
        if name in {"FullFeatureBatchLogistic", "FullFeatureOnlineSGD"}:
            continue
        probabilities_by_name[name] = baseline_probabilities(name, features, matrix, targets)
    best_c = C_GRID[0]
    batch_candidates: list[tuple[float, float, float, np.ndarray]] = []
    for c_value in C_GRID:
        probability = fixed_logistic_probabilities(data, features, targets, c=c_value)
        arrays = metric_arrays(probability, actual)
        batch_candidates.append((float(arrays["brier"].mean()), float(arrays["log_loss"].mean()), c_value, probability))
    _, _, best_c, probabilities_by_name["FullFeatureBatchLogistic"] = min(batch_candidates, key=lambda item: item[:3])
    _, online_per_number, _, online_probability = online_sgd(data.iloc[:OUTER_START].reset_index(drop=True),
                                                              features[features["target_index"] < OUTER_START],
                                                              start=INNER_START, c=.20)
    probabilities_by_name["FullFeatureOnlineSGD"] = online_probability
    for name in CANDIDATE_NAMES:
        arrays = metric_arrays(probabilities_by_name[name], actual)
        rows.append({"model": name, "inner_brier": float(arrays["brier"].mean()),
                     "inner_log_loss": float(arrays["log_loss"].mean()),
                     "inner_average_hits": float(arrays["hits_top6"].mean()),
                     "complexity_rank": MODEL_COMPLEXITY[name],
                     "selected_C": best_c if name == "FullFeatureBatchLogistic" else np.nan})
    table = pd.DataFrame(rows).sort_values(["inner_brier", "inner_log_loss", "complexity_rank", "model"])
    selected = table.iloc[0]
    return SelectionResult(str(selected["model"]), float(best_c), float(selected["inner_brier"]),
                           float(selected["inner_log_loss"]), table.reset_index(drop=True))


def selected_outer_probabilities(selection: SelectionResult, data: pd.DataFrame,
                                 features: pd.DataFrame) -> np.ndarray:
    matrix = indicator_matrix(data)
    targets = np.arange(OUTER_START, len(data))
    if selection.selected_model == "FullFeatureBatchLogistic":
        return batch_expanding(data, features, start=OUTER_START, c=selection.selected_c)[3]
    if selection.selected_model == "FullFeatureOnlineSGD":
        return online_sgd(data, features, start=OUTER_START)[3]
    return baseline_probabilities(selection.selected_model, features, matrix, targets)


def fair_matrix(seed: int, periods: int = 2148) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = np.argpartition(rng.random((periods, 49)), 6, axis=1)[:, :6]
    matrix = np.zeros((periods, 49), dtype=np.int8)
    matrix[np.arange(periods)[:, None], chosen] = 1
    return matrix


def simulate_same_pipeline(simulation_id: int, seed: int) -> dict[str, Any]:
    """一份完整 same-pipeline 公平歷史；可由 ProcessPoolExecutor 呼叫。"""
    matrix = fair_matrix(seed)
    data = synthetic_dataframe(matrix)
    features = build_feature_dataset_fast(data, min_history=FEATURE_MIN_HISTORY)
    selection = select_inner_model(data, features)
    probabilities = selected_outer_probabilities(selection, data, features)
    actual = matrix[OUTER_START:]
    arrays = metric_arrays(probabilities, actual)
    _, slope, _ = calibration_metrics(probabilities, actual)
    return {"simulation_id": simulation_id, "seed": seed, "selected_model": selection.selected_model,
            "selected_C": selection.selected_c, "average_hits": float(arrays["hits_top6"].mean()),
            "total_hits": int(arrays["hits_top6"].sum()), "brier": float(arrays["brier"].mean()),
            "log_loss": float(arrays["log_loss"].mean()), "calibration_slope": slope,
            "best_inner_validation_score": selection.best_inner_brier,
            "best_inner_log_loss": selection.best_inner_log_loss,
            "outer_prediction_draws": len(actual)}


def simulate_simplified_pipeline(simulation_id: int, seed: int) -> dict[str, Any]:
    """不含 sklearn ML 的 10k 補充流程；名稱與完整 same-pipeline 嚴格分開。"""
    matrix = fair_matrix(seed)
    data = synthetic_dataframe(matrix)
    features = build_feature_dataset_fast(data, FEATURE_MIN_HISTORY)
    candidates = [name for name in CANDIDATE_NAMES
                  if name not in {"FullFeatureBatchLogistic", "FullFeatureOnlineSGD"}]
    inner_targets = np.arange(INNER_START, OUTER_START)
    inner_actual = matrix[inner_targets]
    inner_rows = []
    for name in candidates:
        probability = baseline_probabilities(name, features, matrix, inner_targets)
        arrays = metric_arrays(probability, inner_actual)
        inner_rows.append((float(arrays["brier"].mean()), float(arrays["log_loss"].mean()),
                           MODEL_COMPLEXITY[name], name))
    best_brier, best_log, _, selected = min(inner_rows)
    outer_targets = np.arange(OUTER_START, len(matrix))
    probability = baseline_probabilities(selected, features, matrix, outer_targets)
    arrays = metric_arrays(probability, matrix[outer_targets])
    return {"simulation_id": simulation_id, "seed": seed, "selected_model": selected,
            "average_hits": float(arrays["hits_top6"].mean()), "total_hits": int(arrays["hits_top6"].sum()),
            "brier": float(arrays["brier"].mean()), "log_loss": float(arrays["log_loss"].mean()),
            "best_inner_validation_score": best_brier, "best_inner_log_loss": best_log,
            "outer_prediction_draws": len(outer_targets)}


def exact_hit_p(total_hits: int, periods: int) -> float:
    """自適應事前 Top-6 在公平獨立開獎下仍為 Hypergeom 的逐期卷積。"""
    base = np.array([math.comb(6, h) * math.comb(43, 6 - h) / math.comb(49, 6) for h in range(7)])
    distribution = np.array([1.0])
    for _ in range(periods):
        distribution = np.convolve(distribution, base)
    return float(distribution[total_hits:].sum())


def bootstrap_difference(values: np.ndarray, seed: int = DEFAULT_SEED, runs: int = 5000,
                         block: int | None = None) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.empty(runs)
    if block is None:
        for index in range(runs):
            means[index] = rng.choice(values, len(values), replace=True).mean()
    else:
        starts = np.arange(max(1, len(values) - block + 1))
        count = math.ceil(len(values) / block)
        for index in range(runs):
            selected = np.concatenate([np.arange(start, start + block)
                                       for start in rng.choice(starts, count, replace=True)])[:len(values)]
            means[index] = values[selected].mean()
    return tuple(np.quantile(means, [.025, .975]).tolist())
