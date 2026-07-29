# 第三階段前瞻性鎖模預註冊

- Experiment ID：`TW649-PROSPECTIVE-20260729-V1`
- 版本：`1.0.0`
- 正式模型：`ProspectiveBatchLogistic-v1`
- 預註冊時間：2026-07-29T13:12:26+08:00
- 第一個前瞻目標：115000075（2026-07-31）

## 研究問題

固定模型的完整 49 號邊際機率，能否在未來資料上同時以 Brier Score 與 binary Log Loss 優於每號 6/49 的 ExactUniform？

## 鎖定方法

Batch expanding Logistic Regression；C=0.01、L2、lbfgs、max_iter=500、seed=20260729。數值特徵為 long_z、recent20_z、recent50_z、recent100_z、ewma_z、gap_z、transition_z、in_last_draw，另加號碼 one-hot。EWMA alpha=0.06、transition alpha=30、長期先驗強度=12、收縮=0.10、capped-simplex 總和=6。排序同分時小號優先。

## 資料與時序

唯一結果來源為台灣彩券官方 API/網站。預測只能在目標期開獎前，以目標期以前所有已完成資料重新 fit。遺漏預測不補寫；遠端 commit 未確認者不計正式樣本。官方修正以新事件追加。

## 指標、零假設與成功標準

主要指標為每期模型減均勻基準的 Brier 與 Log Loss；Top-6/10/12 命中為次要指標。第100期使用保留的100組機率向量，在公平6/49下以固定 seed 做至少1,000,000次條件式 Monte Carlo、plus-one p-value及95% Monte Carlo區間，兩個單尾 p 值作 Holm 修正。平均兩種差值皆須小於0、兩個修正 p 均小於0.05、至少60%的20期區塊同方向，且無洩漏或事後預測。Top-6另以 Hypergeometric 卷積核對。

## 階段、限制與停止規則

第1至100個有效預測為第一階段，第101至200個為獨立複驗。每20期可作描述性進度，但第100期前不得作正式結論；第二階段不得與第一階段合併擇優。無論第一階段結果均繼續第二階段。Bug 結束當前版本，開新版本並從零計數，不得修改本版本後續跑。

## 不得事後修改

特徵、模型、超參數、收縮、Top-k排序、指標、成功門檻、Monte Carlo方法、階段界線、缺失/修正/Bug規則皆已鎖定。
