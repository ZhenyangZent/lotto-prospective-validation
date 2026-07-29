"""逐期損失、條件式 Monte Carlo 與確認性判定。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import beta, hypergeom

from .config import RANDOM_SEED, UNIFORM_PROBABILITY


def score_prediction(probabilities: list[float] | np.ndarray, actual_numbers: list[int]) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=float)
    actual = sorted(int(value) for value in actual_numbers)
    if probs.shape != (49,) or not np.isfinite(probs).all():
        raise ValueError("必須提供 49 個有限機率")
    if len(actual) != 6 or len(set(actual)) != 6 or not all(1 <= value <= 49 for value in actual):
        raise ValueError("實際號碼必須為六個不重複的 1..49 整數")
    target = np.zeros(49, dtype=float); target[np.asarray(actual) - 1] = 1
    uniform = np.full(49, UNIFORM_PROBABILITY)
    clipped = np.clip(probs, 1e-12, 1 - 1e-12)
    uniform_clipped = np.clip(uniform, 1e-12, 1 - 1e-12)
    numbers = np.arange(1, 50)
    order = numbers[np.lexsort((numbers, -probs))]
    binary_log = lambda p: -float(np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))
    return {
        "hits_top6": int(len(set(actual).intersection(order[:6]))),
        "hits_top10": int(len(set(actual).intersection(order[:10]))),
        "hits_top12": int(len(set(actual).intersection(order[:12]))),
        "brier": float(np.mean((probs - target) ** 2)),
        "uniform_brier": float(np.mean((uniform - target) ** 2)),
        "log_loss": binary_log(clipped),
        "uniform_log_loss": binary_log(uniform_clipped),
    }


def monte_carlo_interval(extreme: int, simulations: int, confidence: float = 0.95) -> list[float]:
    alpha = 1 - confidence
    lower = 0.0 if extreme == 0 else float(beta.ppf(alpha / 2, extreme, simulations - extreme + 1))
    upper = 1.0 if extreme == simulations else float(beta.ppf(1 - alpha / 2, extreme + 1, simulations - extreme))
    return [lower, upper]


def conditional_monte_carlo(
    probabilities: np.ndarray,
    observed_actual: np.ndarray,
    simulations: int = 1_000_000,
    seed: int = RANDOM_SEED,
    batch_size: int = 10_000,
) -> dict[str, Any]:
    """保留預測向量，僅在公平零假設下重抽每期六號。"""
    probs = np.asarray(probabilities, dtype=float)
    observed = np.asarray(observed_actual, dtype=np.int8)
    if probs.ndim != 2 or probs.shape[1] != 49 or observed.shape != probs.shape:
        raise ValueError("probabilities 與 observed_actual 必須為相同的 (T,49)")
    if not np.all(observed.sum(axis=1) == 6):
        raise ValueError("每期 observed_actual 必須恰有六個 1")
    observed_metrics = _sequence_metrics(probs, observed)
    rng = np.random.default_rng(seed)
    extremes = {"brier": 0, "log_loss": 0, "hits_top6": 0}
    completed = 0
    while completed < simulations:
        count = min(batch_size, simulations - completed)
        random_scores = rng.random((count, len(probs), 49))
        selected = np.argpartition(random_scores, 6, axis=2)[:, :, :6]
        target = np.zeros_like(random_scores, dtype=np.int8)
        np.put_along_axis(target, selected, 1, axis=2)
        metrics = _sequence_metrics_batch(probs, target)
        extremes["brier"] += int(np.count_nonzero(metrics["brier_difference"] <= observed_metrics["brier_difference"]))
        extremes["log_loss"] += int(np.count_nonzero(metrics["log_loss_difference"] <= observed_metrics["log_loss_difference"]))
        extremes["hits_top6"] += int(np.count_nonzero(metrics["hits_top6"] >= observed_metrics["hits_top6"]))
        completed += count
    return {
        "simulations": simulations, "seed": seed,
        "observed": observed_metrics,
        "p_values_plus_one": {key: (value + 1) / (simulations + 1) for key, value in extremes.items()},
        "p_value_95_intervals": {key: monte_carlo_interval(value, simulations) for key, value in extremes.items()},
    }


def _sequence_metrics(probabilities: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    result = _sequence_metrics_batch(probabilities, actual[None, :, :])
    return {key: values[0].item() for key, values in result.items()}


def _sequence_metrics_batch(probabilities: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    uniform = UNIFORM_PROBABILITY
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)[None, :, :]
    brier = np.mean((p - actual) ** 2, axis=(1, 2))
    uniform_brier = np.mean((uniform - actual) ** 2, axis=(1, 2))
    log = -np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p), axis=(1, 2))
    uniform_log = -np.mean(actual * math.log(uniform) + (1 - actual) * math.log(1 - uniform), axis=(1, 2))
    top6 = np.argsort(-probabilities, axis=1, kind="stable")[:, :6]
    expanded = np.broadcast_to(top6[None, :, :], (len(actual), *top6.shape))
    hits = np.take_along_axis(actual, expanded, axis=2).sum(axis=(1, 2))
    return {"brier_difference": brier - uniform_brier, "log_loss_difference": log - uniform_log, "hits_top6": hits}


def exact_top6_sum_distribution(draws: int) -> np.ndarray:
    single = hypergeom.pmf(np.arange(7), 49, 6, 6)
    distribution = np.array([1.0])
    for _ in range(draws):
        distribution = np.convolve(distribution, single)
    return distribution


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0; total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted
