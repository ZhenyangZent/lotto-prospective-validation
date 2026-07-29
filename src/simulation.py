"""公平 6/49 完整歷史序列 Monte Carlo 模擬。"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .data_cleaner import NUMBER_COLUMNS
from .feature_engineering import indicator_matrix


def _maximum_gap(matrix: np.ndarray) -> int:
    maximum = 0
    for column in range(49):
        positions = np.flatnonzero(matrix[:, column])
        gaps = np.diff(np.r_[-1, positions, len(matrix)]) - 1
        maximum = max(maximum, int(gaps.max(initial=0)))
    return maximum


def _hottest_pair(numbers: np.ndarray) -> int:
    ordered = np.sort(numbers, axis=1)
    pair_ids = np.concatenate([
        ordered[:, a] * 49 + ordered[:, b] for a, b in combinations(range(6), 2)
    ])
    return int(np.bincount(pair_ids, minlength=49 * 49).max(initial=0))


def _sample_batch(rng: np.random.Generator, size: int, n_draws: int) -> np.ndarray:
    """向量化逐欄拒絕抽樣；比為每列 49 個亂數取前六名省約八倍亂數。"""
    samples = np.empty((size, n_draws, 6), dtype=np.int16)
    for column in range(6):
        candidate = rng.integers(0, 49, size=(size, n_draws), dtype=np.int16)
        if column:
            duplicated = np.any(samples[:, :, :column] == candidate[:, :, None], axis=2)
            while duplicated.any():
                candidate[duplicated] = rng.integers(0, 49, size=int(duplicated.sum()), dtype=np.int16)
                duplicated = np.any(samples[:, :, :column] == candidate[:, :, None], axis=2)
        samples[:, :, column] = candidate
    return samples


def sequence_metrics(numbers_zero_based: np.ndarray) -> dict[str, float]:
    """從完整序列計算全域極端統計量。"""
    n = len(numbers_zero_based)
    matrix = np.zeros((n, 49), dtype=np.int8)
    matrix[np.arange(n)[:, None], numbers_zero_based] = 1
    expected = n * 6 / 49
    odd_counts = ((numbers_zero_based + 1) % 2).sum(axis=1)
    sums = (numbers_zero_based + 1).sum(axis=1)
    overlaps = (matrix[1:] * matrix[:-1]).sum(axis=1) if n > 1 else np.array([0])
    return {
        "max_abs_number_deviation": float(np.max(np.abs(matrix.sum(axis=0) - expected))),
        "max_gap": float(_maximum_gap(matrix)),
        "hottest_pair_count": float(_hottest_pair(numbers_zero_based)),
        "odd_count_extreme": float(np.max(np.abs(odd_counts - 3))),
        "mean_sum": float(sums.mean()),
        "sum_std": float(sums.std()),
        "overlap_2plus_rate": float(np.mean(overlaps >= 2)),
        "max_overlap": float(overlaps.max(initial=0)),
    }


def observed_metrics(data: pd.DataFrame) -> dict[str, float]:
    numbers = data[NUMBER_COLUMNS].to_numpy(dtype=int) - 1
    return sequence_metrics(numbers)


def simulate_histories(
    n_draws: int,
    iterations: int = 10000,
    seed: int = 0,
    batch_size: int = 25,
) -> pd.DataFrame:
    """模擬 iterations 次、每次 n_draws 期的不放回公平抽樣完整序列。"""
    if iterations < 1 or n_draws < 1:
        raise ValueError("iterations 與 n_draws 必須為正整數")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    remaining = iterations
    while remaining:
        size = min(batch_size, remaining)
        samples = _sample_batch(rng, size, n_draws)
        for sequence in samples:
            rows.append(sequence_metrics(sequence))
        remaining -= size
    result = pd.DataFrame(rows)
    result.insert(0, "simulation", np.arange(1, len(result) + 1))
    return result


def empirical_comparison(observed: dict[str, float], simulations: pd.DataFrame) -> pd.DataFrame:
    """比較真實值和公平模擬；極端量使用右尾，中心量使用雙尾 empirical p-value。"""
    rows = []
    for metric, actual in observed.items():
        values = simulations[metric].to_numpy()
        if metric in {"mean_sum", "sum_std"}:
            center = np.median(values)
            exceed = np.sum(np.abs(values - center) >= abs(actual - center))
        else:
            exceed = np.sum(values >= actual)
        rows.append({
            "metric": metric, "observed": actual, "simulation_mean": float(values.mean()),
            "simulation_sd": float(values.std(ddof=1)),
            "simulation_q025": float(np.quantile(values, 0.025)),
            "simulation_q975": float(np.quantile(values, 0.975)),
            "empirical_p_value": float((exceed + 1) / (len(values) + 1)),
        })
    return pd.DataFrame(rows)


def global_anomaly_pvalue(comparison: pd.DataFrame) -> float:
    """以最小逐項 p 值的 Bonferroni 上界近似全域 look-elsewhere 機率。"""
    minimum = float(comparison["empirical_p_value"].min())
    return min(1.0, minimum * len(comparison))
