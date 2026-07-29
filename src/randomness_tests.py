"""隨機性、獨立性、穩定性與多重比較檢定。"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare
from sklearn.metrics import mutual_info_score
from statsmodels.stats.diagnostic import acorr_ljungbox

from .feature_engineering import indicator_matrix


def _normal_two_sided(z: float) -> float:
    return 1 - erf(abs(z) / sqrt(2))


def runs_test(binary: np.ndarray) -> tuple[float, float]:
    """Wald–Wolfowitz 二元 runs test。"""
    values = np.asarray(binary, dtype=int)
    n1, n0 = int(values.sum()), int(len(values) - values.sum())
    if n1 == 0 or n0 == 0 or len(values) < 2:
        return float("nan"), float("nan")
    runs = 1 + int(np.sum(values[1:] != values[:-1]))
    mean = 1 + 2 * n1 * n0 / (n1 + n0)
    variance = 2 * n1 * n0 * (2 * n1 * n0 - n1 - n0) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
    z = (runs - mean) / sqrt(variance)
    return z, _normal_two_sided(z)


def adjust_pvalues(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni 與 Benjamini–Hochberg 修正。"""
    p = np.asarray(values, dtype=float)
    bonf = np.minimum(p * len(p), 1.0)
    order = np.argsort(p); ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    fdr = np.empty_like(adjusted); fdr[order] = np.minimum(adjusted, 1.0)
    return bonf, fdr


def geometric_gap_test(gaps: np.ndarray, success_probability: float = 6 / 49) -> float:
    """固定理論 p 的離散幾何適合度檢定，將稀疏尾端合併至最後一格。"""
    values = np.asarray(gaps, dtype=int)
    if len(values) < 20:
        return float("nan")
    q = 1 - success_probability
    # 每個獨立格理論期望至少約 5；最後一格收納其餘右尾。
    cutoff = 1
    while cutoff < 100 and len(values) * success_probability * q ** cutoff >= 5:
        cutoff += 1
    observed = np.array([np.sum(values == gap) for gap in range(cutoff)] + [np.sum(values >= cutoff)])
    expected = np.array(
        [len(values) * success_probability * q ** gap for gap in range(cutoff)]
        + [len(values) * q ** cutoff]
    )
    return float(chisquare(observed, expected).pvalue)


def number_tests(data: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """逐號自相關、runs、Ljung–Box 與遺漏幾何分布檢定。"""
    matrix = indicator_matrix(data)
    rows = []
    for idx in range(49):
        seq = matrix[:, idx].astype(float)
        autocorr = float(pd.Series(seq).autocorr(lag=1))
        run_z, run_p = runs_test(seq)
        lb = acorr_ljungbox(seq, lags=[min(lags, max(1, len(seq) // 5))], return_df=True)
        positions = np.flatnonzero(seq)
        gaps = np.diff(positions) - 1
        geometric_p = geometric_gap_test(gaps)
        rows.append({"number": idx + 1, "autocorr_lag1": autocorr, "runs_z": run_z,
                     "runs_p": run_p, "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
                     "gap_geometric_p": geometric_p})
    result = pd.DataFrame(rows)
    for column in ("runs_p", "ljung_box_p", "gap_geometric_p"):
        valid = result[column].fillna(1.0).to_numpy()
        result[f"{column}_bonferroni"], result[f"{column}_fdr"] = adjust_pvalues(valid)
    return result


def cross_dependence(data: pd.DataFrame) -> pd.DataFrame:
    """不同號碼跨相鄰期的 lag-1 相關與互資訊。

    同一期六號是不放回抽樣，必然負相關；以同一期獨立性為虛無假設會
    產生大量假異常。因此比較 t 期的 a 與 t+1 期的 b。
    """
    matrix = indicator_matrix(data)
    rows = []
    for a in range(49):
        for b in range(a + 1, 49):
            first, second = matrix[:-1, a], matrix[1:, b]
            table = pd.crosstab(first, second).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            _, p_value, _, _ = chi2_contingency(table, correction=False)
            rows.append({"number_a": a + 1, "number_b": b + 1,
                         "correlation": float(np.corrcoef(first, second)[0, 1]),
                         "mutual_information": float(mutual_info_score(first, second)),
                         "p_value": float(p_value)})
    result = pd.DataFrame(rows)
    result["p_bonferroni"], result["p_fdr_bh"] = adjust_pvalues(result["p_value"].to_numpy())
    return result.sort_values("p_value").reset_index(drop=True)


def randomness_summary(data: pd.DataFrame, alpha: float = 0.05) -> dict[str, object]:
    """整合均勻性、逐號、跨號與前後半期結構檢定。"""
    matrix = indicator_matrix(data)
    counts = matrix.sum(axis=0)
    chi_stat, chi_p = chisquare(counts, np.full(49, len(data) * 6 / 49))
    per_number = number_tests(data)
    cross = cross_dependence(data)
    midpoint = len(data) // 2
    halves = np.vstack([matrix[:midpoint].sum(axis=0), matrix[midpoint:].sum(axis=0)])
    structural_stat, structural_p, _, _ = chi2_contingency(halves)
    pcols = ["runs_p", "ljung_box_p", "gap_geometric_p"]
    raw = int(sum((per_number[c] < alpha).sum() for c in pcols) + (cross["p_value"] < alpha).sum())
    bonf = int(sum((per_number[f"{c}_bonferroni"] < alpha).sum() for c in pcols) + (cross["p_bonferroni"] < alpha).sum())
    fdr = int(sum((per_number[f"{c}_fdr"] < alpha).sum() for c in pcols) + (cross["p_fdr_bh"] < alpha).sum())
    significant_numbers = sorted(set(
        per_number.loc[
            np.logical_or.reduce([per_number[f"{column}_fdr"] < alpha for column in pcols]), "number"
        ].astype(int)
    ))
    stability: list[dict[str, object]] = []
    if significant_numbers and midpoint >= 50:
        first_half = number_tests(data.iloc[:midpoint]).set_index("number")
        second_half = number_tests(data.iloc[midpoint:]).set_index("number")
        for number in significant_numbers:
            stability.append({
                "number": number,
                "first_half": {column: float(first_half.loc[number, column]) for column in pcols},
                "second_half": {column: float(second_half.loc[number, column]) for column in pcols},
                "stable_same_test_raw": any(
                    first_half.loc[number, column] < alpha and second_half.loc[number, column] < alpha for column in pcols
                ),
                "stable_same_test_bonferroni": any(
                    first_half.loc[number, f"{column}_bonferroni"] < alpha
                    and second_half.loc[number, f"{column}_bonferroni"] < alpha for column in pcols
                ),
            })
    return {
        "chi_square_uniformity": {"statistic": float(chi_stat), "p_value": float(chi_p)},
        "half_period_structure": {"statistic": float(structural_stat), "p_value": float(structural_p)},
        "multiple_testing": {"raw_significant": raw, "bonferroni_significant": bonf, "fdr_significant": fdr},
        "stability_checks": stability,
        "number_tests": per_number,
        "cross_dependence": cross,
    }
