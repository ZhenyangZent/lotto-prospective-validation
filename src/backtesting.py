"""無洩漏 walk-forward 回測與基準比較。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.metrics import log_loss

from .data_cleaner import NUMBER_COLUMNS
from .strategies import Strategy, UniformRandomStrategy


@dataclass
class BacktestRecord:
    target_index: int
    draw_id: str
    draw_date: str
    strategy: str
    hits_top6: int
    hits_top10: int
    brier: float
    log_loss: float


def walk_forward_splits(length: int, min_train: int) -> Iterable[tuple[np.ndarray, int]]:
    """產生擴張視窗索引，保證訓練索引全部早於測試索引。"""
    for target in range(min_train, length):
        yield np.arange(target), target


def _evaluate(probabilities: np.ndarray, actual_numbers: np.ndarray) -> tuple[int, int, float, float]:
    actual = np.zeros(49, dtype=int); actual[actual_numbers - 1] = 1
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    top6 = set((np.argsort(probs)[-6:] + 1).tolist())
    top10 = set((np.argsort(probs)[-10:] + 1).tolist())
    actual_set = set(actual_numbers.tolist())
    brier = float(np.mean((probs - actual) ** 2))
    loss = float(log_loss(actual, probs, labels=[0, 1]))
    return len(top6 & actual_set), len(top10 & actual_set), brier, loss


def walk_forward_backtest(
    data: pd.DataFrame,
    strategies: list[Strategy],
    min_train: int = 300,
    max_predictions: int | None = None,
    ml_refit_interval: int = 20,
) -> pd.DataFrame:
    """逐期擴張視窗回測。ML 可按固定間隔重訓，區間內沿用過去模型輸出。"""
    start = min_train
    if max_predictions is not None:
        start = max(start, len(data) - max_predictions)
    records: list[BacktestRecord] = []
    cached: dict[str, np.ndarray] = {}
    for _, target in walk_forward_splits(len(data), start):
        history = data.iloc[:target]
        row = data.iloc[target]
        actual = row[NUMBER_COLUMNS].to_numpy(dtype=int)
        for strategy in strategies:
            is_ml = strategy.name in {"LogisticRegression", "RandomForest"}
            if (not is_ml) or strategy.name not in cached or (target - start) % ml_refit_interval == 0:
                cached[strategy.name] = strategy.predict(history).probabilities
            hits6, hits10, brier, loss = _evaluate(cached[strategy.name], actual)
            records.append(BacktestRecord(target, str(row["draw_id"]), str(pd.Timestamp(row["draw_date"]).date()),
                                          strategy.name, hits6, hits10, brier, loss))
    return pd.DataFrame([asdict(record) for record in records])


def bootstrap_mean_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    """平均值百分位 bootstrap 95% CI。"""
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    if not len(sample):
        return float("nan"), float("nan")
    means = np.mean(rng.choice(sample, size=(iterations, len(sample)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def summarize_backtest(results: pd.DataFrame, bootstrap_iterations: int = 2000, seed: int = 0) -> pd.DataFrame:
    """彙整命中分布、Brier、log loss、相對均勻基準與顯著性。"""
    if results.empty:
        return pd.DataFrame()
    baseline = results[results["strategy"] == "UniformRandom"].set_index("target_index")["hits_top6"]
    rows = []
    for name, group in results.groupby("strategy"):
        hits = group.set_index("target_index")["hits_top6"]
        aligned = hits.index.intersection(baseline.index)
        difference = hits.loc[aligned].to_numpy() - baseline.loc[aligned].to_numpy()
        low, high = bootstrap_mean_ci(hits.to_numpy(), bootstrap_iterations, seed)
        rows.append({
            "strategy": name, "predictions": len(group), "mean_hits": float(hits.mean()),
            **{f"hit_{i}_rate": float((hits == i).mean()) for i in range(7)},
            "top10_mean_hits": float(group["hits_top10"].mean()),
            "brier_score": float(group["brier"].mean()), "log_loss": float(group["log_loss"].mean()),
            "mean_hits_ci95_low": low, "mean_hits_ci95_high": high,
            "difference_vs_uniform": float(difference.mean()) if len(difference) else 0.0,
            "relative_improvement": float(difference.mean() / baseline.loc[aligned].mean()) if len(difference) and baseline.loc[aligned].mean() else 0.0,
            "paired_ttest_p": float(ttest_1samp(difference, 0).pvalue) if len(difference) > 1 and np.std(difference) else 1.0,
        })
    result = pd.DataFrame(rows)
    comparisons = max(1, int((result["strategy"] != "UniformRandom").sum()))
    result["model_selection_p_bonferroni"] = np.minimum(result["paired_ttest_p"] * comparisons, 1.0)
    return result.sort_values(["mean_hits", "brier_score"], ascending=[False, True]).reset_index(drop=True)


def random_baseline_distribution(n_predictions: int, repetitions: int = 1000, seed: int = 0) -> dict[str, float]:
    """重複均勻策略建立平均命中數基準分布（交集為超幾何）。"""
    rng = np.random.default_rng(seed)
    hits = rng.hypergeometric(6, 43, 6, size=(repetitions, n_predictions)).mean(axis=1)
    return {"repetitions": repetitions, "mean": float(hits.mean()),
            "ci95_low": float(np.quantile(hits, 0.025)), "ci95_high": float(np.quantile(hits, 0.975))}


def yearly_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """不同年度的策略穩定性。"""
    data = results.copy(); data["year"] = pd.to_datetime(data["draw_date"]).dt.year
    return data.groupby(["year", "strategy"], as_index=False).agg(
        predictions=("hits_top6", "size"), mean_hits=("hits_top6", "mean"),
        brier_score=("brier", "mean"), log_loss=("log_loss", "mean"),
    )
