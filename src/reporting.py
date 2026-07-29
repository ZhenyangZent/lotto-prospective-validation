"""將統計結果組成易讀的繁體中文 Markdown 報告。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT


def _fmt_p(value: float) -> str:
    return "< 0.0001" if value < 0.0001 else f"{value:.4f}"


def generate_report(data: pd.DataFrame, results: dict[str, Any], output: str | Path) -> Path:
    """產生完整報告；只陳述現有產出，不把缺少的結果推測成結論。"""
    destination = Path(output); destination = destination if destination.is_absolute() else ROOT / destination
    tables = ROOT / "reports" / "tables"
    frequency = pd.read_csv(tables / "frequency_all.csv")
    pairs = pd.read_csv(tables / "pair_cooccurrence.csv")
    triples = pd.read_csv(tables / "triple_cooccurrence.csv")
    validation = results["validation"]
    theory = results["theory"]
    random = results["randomness"]
    hot = frequency.nlargest(5, "count")["number"].astype(int).tolist()
    cold_gap = frequency.nlargest(5, "current_gap")[["number", "current_gap"]].astype(int).values.tolist()
    raw_pair = int(pairs["significant_raw"].sum())
    fdr_pair = int(pairs["significant_fdr"].sum())
    raw_triple = int(triples["significant_raw"].sum())
    bonf_triple = int(triples["significant_bonferroni"].sum())
    fdr_triple = int(triples["significant_fdr"].sum())
    backtest_path = tables / "backtest_summary.csv"
    backtest = pd.read_csv(backtest_path) if backtest_path.exists() else pd.DataFrame()
    best = backtest.iloc[0] if not backtest.empty else None
    simulation = results.get("simulation")
    sim_text = "尚未執行；請執行 `python -m src.main simulate`。"
    if simulation:
        sim_text = (f"共模擬 {simulation['iterations']:,} 次完整歷史序列；全域異常 p 值近似為 "
                    f"{_fmt_p(simulation['global_anomaly_pvalue'])}。個別指標請見 `simulation_comparison.csv`。")
    model_text = "尚未執行回測。"
    if best is not None:
        adjusted_p = float(best.get("model_selection_p_bonferroni", min(1.0, best["paired_ttest_p"] * 9)))
        significant = bool(adjusted_p < 0.05 and best["difference_vs_uniform"] > 0)
        model_text = (f"樣本外平均命中數最高為 **{best['strategy']}**（{best['mean_hits']:.3f} 個），"
                      f"相對均勻固定 Top-6 差 {best['difference_vs_uniform']:+.3f}，未修正配對 p={_fmt_p(best['paired_ttest_p'])}，"
                      f"九個非均勻策略 Bonferroni 模型選擇修正後 p={_fmt_p(adjusted_p)}。"
                      f"依預先設定 5% 門檻，修正後{'有' if significant else '沒有'}顯著優勢。"
                      f"其 Brier Score 為 {best['brier_score']:.7f}；命中排名若未伴隨實質機率分數改善，不應解讀為可利用訊號。")
    report = f"""# 台灣大樂透歷史資料分析報告

> 存取／產製日期：{results['generated_at']}。本報告只分析歷史統計，不保證、也不暗示任何選號能提高下一期中獎機率。未滿 18 歲不得購買或兌領彩券；請量力而為。

## 1. 執行摘要

本次使用台灣彩券官方年度下載檔，共 {len(data):,} 期，期間為 {data['draw_date'].min().date()} 至 {data['draw_date'].max().date()}。資料驗證{'通過' if validation['valid'] else '未通過'}。歷史頻率較高的五個號碼是 {hot}；這是描述，不代表下一期機率較高。

## 2. 資料來源及期間

來源為台灣彩券「各期開獎結果資料下載」與官方 `Lottery/ResultDownload` API。原始年度 ZIP 保留於 `data/raw/`，清理檔為 `data/processed/lotto649.csv`。官方下載頁說明資料每月 5 日更新至前一個月；本資料實際涵蓋日以上述最大日期為準。

## 3. 規則與理論機率

大樂透由 1–49 選 6 個一般獎號，另開 1 個不與一般獎號重複的特別號，每注新臺幣 50 元。一般獎號組合數為 {theory['total_combinations']:,}，單注頭獎機率為 1/{theory['total_combinations']:,}（{theory['jackpot_probability']:.10%}）。單一指定號碼成為一般獎號的機率為 6/49；至少一組連號的理論機率為 {theory['consecutive_probability']:.2%}。

獎金分為固定與浮動項目，實際規則、保證金額與分配方式可能調整，請以官方當期公告為準；本分析不以獎金預測作選號依據。

## 4. 資料品質

驗證結果：`valid={validation['valid']}`，共 {validation['row_count']:,} 列；錯誤／警告詳見 `validation.json`。檢查涵蓋號碼範圍、期內重複、特別號、日期與期別重複、空值、排序與資料量。年度官方檔可能沒有各獎項中獎注數與單注獎金，這些選填欄位不被捏造。

## 5. 號碼頻率、冷熱號與遺漏

歷史出現次數前五名為 {hot}。目前遺漏較久的五個「號碼、期數」為 {cold_gap}。每個號碼的次數、理論期望、差異、z-score、95% 二項信賴區間、目前與最大遺漏、平均間隔均在 `frequency_all.csv`。

熱號不表示下一期較容易出現；冷號也沒有「該出現」的機制。若各期獨立，遺漏多久不會改變下一期的 6/49 邊際機率（賭徒謬誤）。

## 6. 組合型態

單雙、大小、總和、跨度、相鄰間距、連號、同尾、前期重複、十位區間、質數、平均與標準差均已輸出至 `draw_patterns.csv`。連號並不罕見：至少一組連號的精確機率約 {theory['consecutive_probability']:.2%}。與上一期重複號碼的完整超幾何理論分布記錄於 `analysis_summary.json`。

## 7. 配對與三號共現

兩號配對未修正顯著數為 {raw_pair}，Benjamini–Hochberg FDR 修正後為 {fdr_pair}。三號組合未修正為 {raw_triple}，Bonferroni 後 {bonf_triple}，FDR 後 {fdr_triple}。p 值以固定公平機率的離散二項尾端計算。大量組合必然產生一些隨機極端值，因此不可只挑最熱門組合解讀；若 FDR 與較保守 Bonferroni 結論不同，應以探索性結果看待。

## 8. 隨機性與獨立性

全號碼卡方均勻性檢定 p={_fmt_p(random['chi_square_uniformity']['p_value'])}；前後半期結構檢定 p={_fmt_p(random['half_period_structure']['p_value'])}。逐號 runs、Ljung–Box、遺漏分布及跨期跨號檢定合計未修正顯著 {random['multiple_testing']['raw_significant']} 項，Bonferroni 後 {random['multiple_testing']['bonferroni_significant']} 項，FDR 後 {random['multiple_testing']['fdr_significant']} 項。修正後逐號異常的前後半期穩定性檢查為 `{random.get('stability_checks', [])}`；顯著不自動代表可預測，若同一檢定無法在兩段都通過修正，更不能視為穩定訊號。

## 9. Monte Carlo 全域比較

{sim_text}

## 10. 策略與 walk-forward 回測

系統包含均勻、全期頻率、近期熱號、冷號、混合、EWMA、貝氏收縮、Markov、邏輯斯迴歸及隨機森林。所有預測點只讀取先前資料，禁止隨機切割。

{model_text}

Brier Score、log loss、Top-6／Top-10、命中 0–6 比例、bootstrap 信賴區間及年度穩定性見回測表格與圖表。複雜模型若只在歷史調參後看似較好、而未穩定超越均勻基準，應視為過度擬合風險。

## 11. 對核心問題的回答

- 哪些號碼較常出現：{hot}；僅為本資料期間的歷史排名。
- 差異是否超過合理隨機波動：以卡方、多重比較及 Monte Carlo 結果共同判讀，不以單一 z-score 下結論。
- 哪些號碼遺漏較久：{cold_gap}；遺漏不提高下一期機率。
- 連號是否少見：不算罕見，理論上至少一組連號約 {theory['consecutive_probability']:.2%}。
- 上期號碼再次出現是否罕見：應與超幾何分布比較，圖表與型態表已完成此比較。
- 是否有顯著配對：原始 {raw_pair} 組、FDR 後 {fdr_pair} 組；修正後結果才適合進一步檢視。
- 是否有穩定時間依賴：必須同時通過逐號多重修正、期間穩定性及樣本外回測；單次顯著不足以證明。
- 是否能預測下一期：目前沒有任何理論理由讓歷史冷熱改變公平抽樣機率；只有穩定、樣本外且修正偏差後的優勢才算證據。

## 12. 投注組合與限制

多注產生器避免相同組合並可提高不同注的覆蓋差異。平衡單雙或大小不提高某一確定組合的頭獎機率；只有增加互不重複組合才增加覆蓋率。避開常見生日號碼可能降低多人分獎風險，但不改變開獎機率。

## 13. 最終結論與責任提醒

描述性異常不等於可利用訊號。任何「最佳」策略都是在多模型比較後得到，必須保守看待模型選擇偏差。除非樣本外優勢顯著且跨年度穩定，最合理結論是歷史資料不足以預測下一期。

彩券是高變異、負期望娛樂。請設定可承受的預算，不借貸、不追損，且不要把本系統的實驗性推薦視為投資或保證中獎建議。
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return destination


def load_results(path: str | Path) -> dict[str, Any]:
    source = Path(path); source = source if source.is_absolute() else ROOT / source
    return json.loads(source.read_text(encoding="utf-8"))
