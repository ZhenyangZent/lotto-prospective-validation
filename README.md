# 台灣大樂透歷史資料分析系統

這是一套可重複執行的 Python 3.11+ 專案：從台灣彩券官方年度資料下載、清理與驗證開始，完成理論機率、描述統計、隨機性檢定、Monte Carlo 全域異常比較、十種選號策略與 walk-forward 樣本外回測，最後產生繁體中文報告。

本系統不保證中獎。冷熱號、遺漏、配對與模型排名都是統計實驗；在公平且獨立的開獎機制下，歷史型態不會自動改變下一期機率。

## 快速開始

在本目錄執行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.main all
```

若 Python 是 MSYS/MinGW 版本而套件沒有相容 wheel，請改用 python.org 64 位元 CPython 3.11 或 3.12。

個別命令：

```powershell
python -m src.main download
python -m src.main validate
python -m src.main analyze
python -m src.main simulate
python -m src.main backtest
python -m src.main recommend --tickets 10
python -m src.main report
```

- `download --force`：重新下載所有年度官方 ZIP。
- `download --manual official.xlsx`：匯入手動取得的官方 CSV/Excel。
- `simulate --iterations 10000`：覆寫模擬次數；正式全域比較建議至少 10,000 次。
- `recommend --strategy BayesianShrinkage --tickets 10`：產生互不重複、較分散的實驗組合。
- `recommend --unrestricted`：不做跨注平衡，只依策略權重抽樣。

所有隨機流程使用 `config.yaml` 的固定 seed。模擬與回測可能需要數分鐘；`config.yaml` 可調整批次大小、最少訓練期數、bootstrap、隨機基準重複數及 ML 重訓間隔。

本次在 2,148 期資料上的參考耗時：10,000 次完整序列模擬約 35 秒；十策略、1,848 個樣本外點的完整回測約 1–2 分鐘（硬體與 BLAS 實作不同會有差異）。ML 每 100 期重訓一次；兩次重訓間沿用先前模型，因此只會少用新資訊，不會看到未來資料。

## 分析內容

- 精確機率：13,983,816 組、各獎項、指定一／二／三號、相鄰兩期重複、連號、單雙及大小分布。
- 描述統計：全期、最近 20/50/100/200 期、年度、規則期間頻率；z-score、信賴區間、遺漏與間隔；組合型態、配對與三號共現。
- 隨機性：卡方、逐號自相關、runs、Ljung–Box、互資訊、遺漏幾何分布、前後期結構，以及 Bonferroni/FDR。
- Monte Carlo：每次都模擬和真實資料相同期數的完整公平序列，評估最大號碼偏差、最大遺漏、最熱配對、單雙極端、總和與重複號碼，並計算 empirical p-value。
- 策略：Uniform、全期頻率、近期熱號、冷號、混合、EWMA、Bayesian shrinkage、Markov、Logistic Regression、Random Forest。
- 回測：擴張視窗 walk-forward；Brier、log loss、Top-6、Top-10、0–6 命中分布、bootstrap CI、均勻隨機基準及年度穩定性。

日期特徵只列為探索性特徵；沒有合理物理機制時不得過度解讀。機器學習使用的每個 target 特徵都只由 target 之前的資料建立。

## 輸出

- `data/raw/`：官方 ZIP 與下載 manifest。
- `data/processed/lotto649.csv`：驗證用正規化資料。
- `reports/validation.json`：資料品質檢查。
- `reports/analysis_summary.json`：可機器讀取摘要。
- `reports/tables/`：完整頻率、型態、共現、隨機性、模擬與回測 CSV。
- `reports/figures/`：PNG 圖表。
- `reports/report.md`：繁體中文最終報告。

圖表會依序尋找 Microsoft JhengHei、Noto Sans TC、PingFang TC、Arial Unicode MS。都不存在時使用 DejaVu Sans，部分繁中文字形可能缺漏；請安裝 Noto Sans TC 後重跑。

## 測試

```powershell
pytest
```

測試涵蓋組合數與理論機率、一般與特別號驗證、去重、特徵無未來洩漏、walk-forward 切割、策略合法性、seed 可重現、模擬均值及多注不重複。

## 已知限制

- 年度官方下載檔以每月更新為主，當年度不一定含最近一期。
- 年度主檔未必含各獎項中獎注數與單注獎金；本系統不捏造缺值。
- p-value 不是「公平」的證明，也不是可預測性的證明；它只量化特定虛無模型下的相容程度。
- 近 20 年資料對非常微弱效果仍可能統計力不足；嘗試許多窗口與模型會增加選擇偏差。
- 多注平衡只增加不重複組合覆蓋，不會讓任何一注更容易開出。

## 負責任投注

未滿 18 歲不得購買或兌領彩券。請設定娛樂預算，不借貸、不追損；若投注造成壓力或影響生活，請停止並尋求專業協助。
