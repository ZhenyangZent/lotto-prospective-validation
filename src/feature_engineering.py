"""無未來資訊洩漏的號碼層級與期別層級特徵。"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .data_cleaner import NUMBER_COLUMNS

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}


def indicator_matrix(data: pd.DataFrame) -> np.ndarray:
    """將每期六號轉成 shape=(期數, 49) 的 0/1 矩陣。"""
    matrix = np.zeros((len(data), 49), dtype=np.int8)
    if len(data):
        numbers = data[NUMBER_COLUMNS].to_numpy(dtype=int) - 1
        matrix[np.arange(len(data))[:, None], numbers] = 1
    return matrix


def gap_statistics(sequence: np.ndarray) -> tuple[int, int, float, int]:
    """回傳目前遺漏、歷史最大遺漏、平均出現間隔、最後出現索引。"""
    positions = np.flatnonzero(sequence)
    if not len(positions):
        return len(sequence), len(sequence), float("nan"), -1
    gaps = np.diff(np.r_[-1, positions]) - 1
    current = len(sequence) - 1 - positions[-1]
    return int(current), int(max(current, gaps.max(initial=0))), float(np.diff(positions).mean()) if len(positions) > 1 else float("nan"), int(positions[-1])


def build_number_features(data: pd.DataFrame, as_of: int | None = None) -> pd.DataFrame:
    """只使用 as_of 之前資料，為 1–49 各號碼建立預測特徵。"""
    end = len(data) if as_of is None else int(as_of)
    history = data.iloc[:end]
    matrix = indicator_matrix(history)
    windows = (5, 10, 20, 50, 100, 200)
    frequency = {
        window: matrix[-min(window, len(matrix)):].mean(axis=0) if len(matrix) else np.zeros(49)
        for window in windows
    }
    if len(matrix):
        alpha = 2 / 21
        weights = (1 - alpha) ** np.arange(len(matrix) - 1, -1, -1)
        ewma_values = (6 / 49) * (1 - alpha) ** len(matrix) + alpha * (weights @ matrix)
    else:
        ewma_values = np.full(49, 6 / 49)
    rows: list[dict[str, float | int]] = []
    for number in range(1, 50):
        seq = matrix[:, number - 1]
        current_gap, max_gap, mean_gap, _ = gap_statistics(seq)
        row: dict[str, float | int] = {"number": number, "draws_seen": end}
        for window in windows:
            row[f"freq_{window}"] = float(frequency[window][number - 1])
        row["ewma"] = float(ewma_values[number - 1])
        recent = float(seq[-20:].mean()) if len(seq) else 0.0
        long = float(seq[-200:].mean()) if len(seq) else 0.0
        row.update({
            "current_gap": current_gap,
            "max_gap": max_gap,
            "mean_gap": 0.0 if np.isnan(mean_gap) else mean_gap,
            "recent_long_diff": recent - long,
            "previous_1": int(seq[-1]) if len(seq) else 0,
            "previous_2": int(seq[-2]) if len(seq) > 1 else 0,
        })
        row["frequency_slope"] = (row["freq_20"] - row["freq_100"]) / 80
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(history):
        date = pd.to_datetime(history.iloc[-1]["draw_date"])
        result["year"] = date.year
        result["month"] = date.month
        result["weekday"] = date.weekday()
    return result


def draw_features(data: pd.DataFrame) -> pd.DataFrame:
    """計算每期組合型態特徵。"""
    output: list[dict[str, float | int]] = []
    previous: set[int] | None = None
    for numbers in data[NUMBER_COLUMNS].to_numpy(dtype=int):
        values = np.sort(numbers)
        tails = values % 10
        gaps = np.diff(values)
        decades = np.bincount(np.minimum(values // 10, 4), minlength=5)
        row = {
            "odd_count": int(np.sum(values % 2)),
            "high_count": int(np.sum(values >= 25)),
            "sum": int(values.sum()), "min": int(values.min()), "max": int(values.max()),
            "span": int(values.max() - values.min()), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "adjacent_pairs": int(np.sum(gaps == 1)),
            "same_tail_pairs": int(sum(1 for a, b in combinations(tails, 2) if a == b)),
            "duplicate_tail_count": int(len(tails) - len(set(tails))),
            "overlap_previous": 0 if previous is None else len(set(values) & previous),
            "prime_count": int(sum(int(x) in PRIMES for x in values)),
            **{f"decade_{i}": int(decades[i]) for i in range(5)},
        }
        for index, gap in enumerate(gaps, 1):
            row[f"gap_{index}"] = int(gap)
        output.append(row)
        previous = set(values)
    return pd.DataFrame(output, index=data.index)


def supervised_number_dataset(data: pd.DataFrame, start: int = 30) -> tuple[pd.DataFrame, np.ndarray]:
    """建立時間順序監督式資料；第 t 期特徵只看到 0..t-1 期，標籤為 t 期。"""
    labels = indicator_matrix(data)
    frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    cumulative = np.vstack([np.zeros((1, 49), dtype=int), labels.cumsum(axis=0)])
    last_seen = np.full(49, -1, dtype=int)
    max_past_gap = np.zeros(49, dtype=int)
    interval_sum = np.zeros(49, dtype=float)
    interval_count = np.zeros(49, dtype=int)
    ewma = np.full(49, 6 / 49, dtype=float)
    alpha = 2 / 21
    for target_index in range(1, len(data)):
        position = target_index - 1
        present = np.flatnonzero(labels[position])
        old = last_seen[present]
        max_past_gap[present] = np.maximum(max_past_gap[present], position - old - 1)
        previously_seen = old >= 0
        interval_sum[present[previously_seen]] += position - old[previously_seen]
        interval_count[present[previously_seen]] += 1
        last_seen[present] = position
        ewma = alpha * labels[position] + (1 - alpha) * ewma
        if target_index < start:
            continue
        counts = cumulative[target_index]
        features = pd.DataFrame({"number": np.arange(1, 50), "draws_seen": target_index})
        for window in (5, 10, 20, 50, 100, 200):
            begin = max(0, target_index - window)
            features[f"freq_{window}"] = (counts - cumulative[begin]) / (target_index - begin)
        current_gap = np.where(last_seen >= 0, target_index - 1 - last_seen, target_index)
        features["ewma"] = ewma
        features["current_gap"] = current_gap
        features["max_gap"] = np.maximum(max_past_gap, current_gap)
        features["mean_gap"] = np.divide(interval_sum, interval_count, out=np.zeros(49), where=interval_count > 0)
        features["recent_long_diff"] = features["freq_20"] - features["freq_200"]
        features["previous_1"] = labels[target_index - 1]
        features["previous_2"] = labels[target_index - 2] if target_index > 1 else 0
        features["frequency_slope"] = (features["freq_20"] - features["freq_100"]) / 80
        date = pd.to_datetime(data.iloc[target_index - 1]["draw_date"])
        features["year"], features["month"], features["weekday"] = date.year, date.month, date.weekday()
        features["target_index"] = target_index
        frames.append(features)
        targets.append(labels[target_index])
    if not frames:
        return pd.DataFrame(), np.array([], dtype=int)
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)
