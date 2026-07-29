"""大樂透精確組合機率。"""
from __future__ import annotations

from collections import OrderedDict
from math import comb


def total_combinations(n: int = 49, k: int = 6) -> int:
    """所有一般獎號組合數。"""
    return comb(n, k)


def jackpot_probability(n: int = 49, k: int = 6) -> float:
    """單注頭獎機率。"""
    return 1 / total_combinations(n, k)


def match_probability(main_matches: int, special: bool = False, n: int = 49, k: int = 6) -> float:
    """恰中指定一般號碼數，並可指定是否中特別號的單注機率。"""
    if not 0 <= main_matches <= k:
        return 0.0
    remaining_non_winning = n - k - 1
    special_choices = 1 if special else 0
    other_choices = k - main_matches - special_choices
    if other_choices < 0 or other_choices > remaining_non_winning:
        return 0.0
    numerator = comb(k, main_matches) * comb(remaining_non_winning, other_choices)
    return numerator / comb(n, k)


def prize_probabilities() -> OrderedDict[str, float]:
    """依現行 6/49 加特別號條件回傳各獎項精確機率。"""
    return OrderedDict([
        ("頭獎（6）", match_probability(6, False)),
        ("貳獎（5+特別號）", match_probability(5, True)),
        ("參獎（5）", match_probability(5, False)),
        ("肆獎（4+特別號）", match_probability(4, True)),
        ("伍獎（4）", match_probability(4, False)),
        ("陸獎（3+特別號）", match_probability(3, True)),
        ("柒獎（2+特別號）", match_probability(2, True)),
        ("普獎（3）", match_probability(3, False)),
    ])


def inclusion_probability(r: int, n: int = 49, k: int = 6) -> float:
    """指定 r 個相異號碼同時包含於一般獎號的機率。"""
    return comb(n - r, k - r) / comb(n, k) if 0 <= r <= k else 0.0


def overlap_distribution(n: int = 49, k: int = 6) -> dict[int, float]:
    """相鄰兩期重複 j 個一般號碼的超幾何分布。"""
    denominator = comb(n, k)
    return {j: comb(k, j) * comb(n - k, k - j) / denominator for j in range(k + 1)}


def odd_count_distribution(n: int = 49, k: int = 6) -> dict[int, float]:
    """一般獎號中奇數個數的精確分布。"""
    odds, evens = (n + 1) // 2, n // 2
    denominator = comb(n, k)
    return {j: comb(odds, j) * comb(evens, k - j) / denominator for j in range(k + 1)}


def high_count_distribution(threshold: int = 25, n: int = 49, k: int = 6) -> dict[int, float]:
    """大於等於 threshold 的號碼個數精確分布。"""
    highs, lows = n - threshold + 1, threshold - 1
    denominator = comb(n, k)
    return {j: comb(highs, j) * comb(lows, k - j) / denominator for j in range(k + 1)}


def consecutive_probability(n: int = 49, k: int = 6) -> float:
    """至少包含一組相鄰連號的精確機率。"""
    without_adjacent = comb(n - k + 1, k)
    return 1 - without_adjacent / comb(n, k)


def theoretical_summary() -> dict[str, object]:
    """供 CLI 與報告使用的完整理論摘要。"""
    overlap = overlap_distribution()
    return {
        "total_combinations": total_combinations(),
        "jackpot_probability": jackpot_probability(),
        "single_number_probability": inclusion_probability(1),
        "specific_pair_probability": inclusion_probability(2),
        "specific_triple_probability": inclusion_probability(3),
        "consecutive_probability": consecutive_probability(),
        "overlap_distribution": overlap,
        "overlap_3_or_more_probability": sum(v for k, v in overlap.items() if k >= 3),
        "odd_count_distribution": odd_count_distribution(),
        "high_count_distribution": high_count_distribution(),
        "prize_probabilities": prize_probabilities(),
    }
