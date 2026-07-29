"""描述性統計、冷熱號與共現分析。"""
from __future__ import annotations

from itertools import combinations
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import binom

from .data_cleaner import NUMBER_COLUMNS
from .feature_engineering import draw_features, gap_statistics, indicator_matrix
from .probability import inclusion_probability


def number_frequency(data: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """號碼頻率、z 分數、二項近似信賴區間與遺漏統計。"""
    sample = data if window is None else data.tail(window)
    matrix = indicator_matrix(sample)
    full = indicator_matrix(data)
    n = len(sample)
    p = 6 / 49
    counts = matrix.sum(axis=0)
    expected = n * p
    std = sqrt(n * p * (1 - p)) if n else float("nan")
    rows = []
    for idx, count in enumerate(counts):
        current, maximum, average, last = gap_statistics(full[:, idx])
        low, high = binom.interval(0.95, n, p) if n else (0, 0)
        rows.append({
            "number": idx + 1, "count": int(count), "proportion": count / n if n else np.nan,
            "expected_count": expected, "difference": count - expected,
            "z_score": (count - expected) / std if std else np.nan,
            "ci95_low_count": int(low), "ci95_high_count": int(high),
            "last_draw_index": last, "current_gap": current, "max_gap": maximum,
            "mean_gap": average,
        })
    return pd.DataFrame(rows)


def frequency_by_period(data: pd.DataFrame, windows: list[int]) -> dict[str, pd.DataFrame]:
    """全部、近期視窗、年度與來源版本的頻率表。"""
    tables: dict[str, pd.DataFrame] = {"all": number_frequency(data)}
    for window in windows:
        tables[f"last_{window}"] = number_frequency(data, window)
    for year, group in data.groupby(pd.to_datetime(data["draw_date"]).dt.year):
        tables[f"year_{year}"] = number_frequency(group)
    for version, group in data.groupby("source_version", dropna=False):
        tables[f"period_{version}"] = number_frequency(group)
    return tables


def cooccurrence_tables(data: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """兩號與三號共現次數、z 分數及多重比較修正。"""
    n = len(data)
    pair_counts = {pair: 0 for pair in combinations(range(1, 50), 2)}
    triple_counts = {triple: 0 for triple in combinations(range(1, 50), 3)}
    for values in data[NUMBER_COLUMNS].to_numpy(dtype=int):
        values = sorted(values)
        for pair in combinations(values, 2):
            pair_counts[pair] += 1
        for triple in combinations(values, 3):
            triple_counts[triple] += 1

    def build(counts: dict[tuple[int, ...], int], order: int) -> pd.DataFrame:
        probability = inclusion_probability(order)
        expected = n * probability
        scale = sqrt(n * probability * (1 - probability))
        result = pd.DataFrame([
            {"combination": "-".join(map(str, key)), "count": value,
             "expected": expected, "z_score": (value - expected) / scale}
            for key, value in counts.items()
        ])
        observed = result["count"].to_numpy()
        # 兩尾離散二項尾機率；三號組合期望很低，不使用常態近似 p 值。
        result["p_value"] = np.minimum(
            1.0, 2 * np.minimum(binom.cdf(observed, n, probability), binom.sf(observed - 1, n, probability))
        )
        result["p_bonferroni"] = np.minimum(result["p_value"] * len(result), 1.0)
        order_index = np.argsort(result["p_value"].to_numpy())
        ranked = result["p_value"].to_numpy()[order_index]
        adjusted = np.minimum.accumulate((ranked * len(result) / np.arange(1, len(result) + 1))[::-1])[::-1]
        fdr = np.empty(len(result)); fdr[order_index] = np.minimum(adjusted, 1.0)
        result["p_fdr_bh"] = fdr
        result["significant_raw"] = result["p_value"] < alpha
        result["significant_bonferroni"] = result["p_bonferroni"] < alpha
        result["significant_fdr"] = result["p_fdr_bh"] < alpha
        return result.sort_values(["count", "combination"], ascending=[False, True]).reset_index(drop=True)

    return build(pair_counts, 2), build(triple_counts, 3)


def descriptive_summary(data: pd.DataFrame, windows: list[int], alpha: float = 0.05) -> dict[str, object]:
    """產生所有主要描述統計物件。"""
    frequencies = frequency_by_period(data, windows)
    patterns = draw_features(data)
    pairs, triples = cooccurrence_tables(data, alpha)
    return {"frequencies": frequencies, "patterns": patterns, "pairs": pairs, "triples": triples}
