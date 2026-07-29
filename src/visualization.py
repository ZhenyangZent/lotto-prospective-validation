"""繁體中文統計圖表。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT
from .feature_engineering import indicator_matrix


def configure_chinese_font() -> str:
    """依序選擇常見繁中字型；找不到時由 matplotlib fallback。"""
    candidates = ["Microsoft JhengHei", "Noto Sans TC", "PingFang TC", "Arial Unicode MS", "DejaVu Sans"]
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def _save(fig: plt.Figure, directory: Path, name: str) -> Path:
    path = directory / f"{name}.png"
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def create_analysis_figures(
    data: pd.DataFrame, frequency: pd.DataFrame, patterns: pd.DataFrame,
    pairs: pd.DataFrame, output_dir: str | Path,
) -> list[Path]:
    """建立歷史頻率、型態、共現與滾動頻率圖。"""
    configure_chinese_font()
    directory = Path(output_dir); directory = directory if directory.is_absolute() else ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    x = frequency["number"]
    for name, values, title, ylabel in [
        ("number_frequency", frequency["count"], "各號碼歷史出現次數", "次數"),
        ("number_zscore", frequency["z_score"], "各號碼相對理論期望 z-score", "z-score"),
        ("hot_cold_ranking", frequency.sort_values("count")["count"], "冷熱號歷史排名（僅描述）", "次數"),
        ("current_gap", frequency["current_gap"], "目前遺漏期數", "期數"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        xx = x if name != "hot_cold_ranking" else frequency.sort_values("count")["number"].astype(str)
        ax.bar(xx, values, color="#2878B5"); ax.set(title=title, xlabel="號碼", ylabel=ylabel)
        if name == "number_frequency": ax.axhline(frequency["expected_count"].iloc[0], color="tomato", linestyle="--", label="理論期望"); ax.legend()
        paths.append(_save(fig, directory, name))
    for column, name, title, xlabel in [
        ("sum", "sum_distribution", "號碼總和分布", "總和"),
        ("odd_count", "odd_distribution", "單雙比（奇數個數）分布", "奇數個數"),
        ("high_count", "high_low_distribution", "大小比（25–49 個數）分布", "大號個數"),
        ("adjacent_pairs", "consecutive_distribution", "連號配對數量分布", "相鄰配對數"),
        ("overlap_previous", "overlap_distribution", "與上一期重複號碼數分布", "重複號碼數"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(patterns[column], bins="auto", color="#45A778", edgecolor="white")
        ax.set(title=title, xlabel=xlabel, ylabel="期數"); paths.append(_save(fig, directory, name))
    heat = np.zeros((49, 49))
    for row in pairs.itertuples():
        a, b = map(int, row.combination.split("-")); heat[a - 1, b - 1] = heat[b - 1, a - 1] = row.count
    fig, ax = plt.subplots(figsize=(9, 8)); image = ax.imshow(heat, cmap="YlOrRd", origin="lower")
    ax.set(title="兩號共現次數熱圖", xlabel="號碼", ylabel="號碼"); fig.colorbar(image, ax=ax, label="共同出現次數")
    paths.append(_save(fig, directory, "pair_cooccurrence_heatmap"))
    matrix = indicator_matrix(data)
    fig, ax = plt.subplots(figsize=(12, 6))
    for number in frequency.nlargest(5, "count")["number"]:
        rolling = pd.Series(matrix[:, number - 1]).rolling(50).mean()
        ax.plot(data["draw_date"], rolling, label=f"{number:02d}")
    ax.axhline(6 / 49, color="black", linestyle="--", label="理論值")
    ax.set(title="熱門號碼 50 期滾動頻率", xlabel="日期", ylabel="頻率"); ax.legend(ncol=3)
    paths.append(_save(fig, directory, "rolling_frequency"))
    return paths


def create_simulation_figures(comparison: pd.DataFrame, simulations: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """建立真實值與模擬分布比較圖。"""
    configure_chinese_font(); directory = Path(output_dir); directory = directory if directory.is_absolute() else ROOT / directory
    paths = []
    for metric in ("max_abs_number_deviation", "max_gap", "hottest_pair_count", "mean_sum"):
        actual = float(comparison.loc[comparison["metric"] == metric, "observed"].iloc[0])
        fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(simulations[metric], bins=40, color="#8E6CBB", alpha=.8)
        ax.axvline(actual, color="red", linewidth=2, label=f"真實值 {actual:.2f}")
        ax.set(title=f"Monte Carlo：{metric}", xlabel=metric, ylabel="模擬次數"); ax.legend()
        paths.append(_save(fig, directory, f"monte_carlo_{metric}"))
    return paths


def create_backtest_figures(summary: pd.DataFrame, yearly: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """建立策略總體、Brier 與年度穩定性圖。"""
    configure_chinese_font(); directory = Path(output_dir); directory = directory if directory.is_absolute() else ROOT / directory
    paths = []
    for column, name, title, ylabel in [
        ("mean_hits", "backtest_mean_hits", "各策略樣本外平均命中數", "平均命中數"),
        ("brier_score", "backtest_brier", "各策略 Brier Score（愈低愈好）", "Brier Score"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5)); ordered = summary.sort_values(column)
        ax.barh(ordered["strategy"], ordered[column], color="#F18F3B"); ax.set(title=title, xlabel=ylabel)
        paths.append(_save(fig, directory, name))
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy, group in yearly.groupby("strategy"):
        ax.plot(group["year"], group["mean_hits"], marker="o", label=strategy)
    ax.set(title="不同年度策略平均命中數", xlabel="年度", ylabel="平均命中數"); ax.legend(ncol=2, fontsize=8)
    paths.append(_save(fig, directory, "backtest_yearly_stability"))
    return paths
