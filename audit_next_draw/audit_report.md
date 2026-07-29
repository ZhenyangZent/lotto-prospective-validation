# 台灣大樂透下一期號碼分析器獨立稽核報告

## 1. 可重現性與範圍

初始專案不是 Git repository，且 `C:\Users\zheny\Music\University\program\彩卷分析\lotto_analysis\next_draw_predictor.py` 不存在。`original_baseline_output.txt` 保存未修改命令、退出碼 2、stderr、耗時與 seed；`file_hashes.json` 將缺檔明列為 `exists=false`。因此不能聲稱重現原版 0.779 命中或對原版函式逐行通過。修正版是新增檔案，不是暗中覆寫。

## 2. 資料驗證

獨立讀取 CSV 得 2,148 期，日期 2007-01-02 至 2026-06-30，SHA-256 `3973dcdc08ae7b96c31b5a4ee4261db501fc9744857e8cc7dd62e9193f50ad34`。一般號碼越界、同期重複、必要空值、重複期別、重複日期、完全重複列、特別號重疊均為 0；最後一列確為最新日期。模型只以 `number_1..number_6` 建標籤。

## 3. 程式與洩漏稽核

審計版的 `build_feature_dataset` 在 t 快照前只更新 t-1；counts、last_seen、EWMA、transition 均沒有讀取 t 標籤。gap 定義為「最近出現後已完整錯過幾期」，故上一期出現為 0。固定 holdout 及 walk-forward 都按完整 `target_index` 群組切割。`normalize_and_shrink_probabilities` 使用 capped-simplex 投影，保證 49 個邊際值在 [0,1] 且總和 6；主設定收縮 0.10 是事前固定，沒有用外層測試 Brier 選擇。

逐期主回測採 `SGDClassifier(loss='log_loss')` 的 expanding `partial_fit`：每期預測後才更新。這是計算可行的線上 Logistic，不等同每期從頭求解完整 batch MLE；固定 holdout 另用標準 `LogisticRegression` Pipeline。此差異是方法限制，不偽裝成 Bug。

## 4. 精確理論

任一事前六號的 H~Hypergeom(49,6,6)。P(H=0..6) 分別為 0.435964975512, 0.413019450485, 0.132378029002, 0.0176504038669, 0.000968619724401, 1.84498995124e-05, 7.15112384202e-08；E(H)=0.734693877551，Var(H)=0.577571845065，中央 95% 整數範圍 [0, 2]。Top-10 與 Top-12 也在 `exact_theory.json`。共同 1,648 期總命中零假設使用逐期離散卷積，並同時保存精確 survival 與常態近似供比較。

## 5. 固定 holdout 與 walk-forward

審計版固定 holdout：424 期，平均命中 0.702830、Brier 0.107507658、Log Loss 0.372006244。主 expanding 線上回測：1648 期，平均命中 0.752427、Brier 0.108164250、Log Loss 0.375330779。差異不是原版 vs 修正版，因為原版缺失；它是審計版兩種估計程序的差異。

## 6. 基準、模型與多模型選擇

共同範圍從第 500 期開始，共 1,648 期。`model_comparison.csv` 保存精確均勻、10,000 seed 分布的配套基準、固定號、長期/近期熱門、冷號、重複/排除上一期、五種 Logistic 消融、完整 Logistic、EWMA、Bayesian、Markov、Random Forest。觀察最佳為 Recent50HotBaseline，平均命中 0.768204。主模型 raw exact p=0.175801、Bonferroni=1、Holm=1、FDR=0.937826。

主模型 Brier 相對均勻差 +0.000709023，Log Loss 差 +0.003554599；負值才代表較好。命中差 paired bootstrap 95% CI [-0.020495, 0.055355]，block-20 CI [-0.021102, 0.056583]。不能把 49×期數當獨立樣本；所有 CI 以期為抽樣群組。

## 7. 公平亂數零假設

第一層實際完成 100,000 條隨機選號序列；平均命中分布見 `baseline_distribution.csv`。第二層實際完成 1,000 份、每份 2,148 期的公平歷史：重建長期、近期、gap、上一期與 EWMA 特徵，在前半從十個預先指定模型選最佳，再到未碰過的後半評估。公平流程平均外層命中 0.734911，95% 範圍 [0.683252, 0.788896]；真實流程位於第 97.50 百分位，family-wise p=0.025974，MC 95% CI [0.0162, 0.0367]。

限制：沒有達到目標 10,000 份完整歷史；最低 1,000 已完成。每份未逐期重跑 sklearn Logistic/RF，因此這個 family-wise p 是預先指定十候選流程的選模修正，不是完全相同最終 ML pipeline 的精確修正。

## 8. 上一期、transition 與 gap

實際相鄰期平均重複 0.736842，公平理論 0.734694。完整模型平均命中 0.752427；移除上一期與 transition 後 0.716626；只用上一期 0.712985。完整數字、前後半與 repeat/exclude 結果在 `previous_draw_analysis.json`、`ablation_results.csv`。小差異不能解讀為條件機率改變或賭徒謬誤有用。

## 9. 穩定性與推薦

主模型高於理論平均的年度為 10/16。不同 min_history、更新頻率、C、刪除末 1/2/5/10 期與 25 次 draw-group bootstrap 的 Top-6 平均 Jaccard=0.443818。最新排序為 1 22 28 36 43 44；證據等級固定為「實驗性排序；未證明提高中獎機率」。

## 10. Top-6 與所有組合的正確解釋

本模型只有 49 個可加總邊際分數。對任何六號集合 S，目標是最大化 sum(s_i, i in S)；交換論證可知，只要集合含較低分而排除較高分，交換後總分上升，所以最優集合就是分數最高六號。這不需要枚舉 13,983,816 組，本稽核也沒有宣稱枚舉。邊際 Logistic 值總和投影為 6，仍不構成「無放回抽六個」的完整聯合分布；最高分組合也未被證明有較高頭獎機率。

## 11. 最終判定

**C－方法有重大驗證缺陷。** 直接原因是指定原程式不存在，原始聲稱無法重現；不是把審計版無洩漏誤寫成原版通過。審計版可重現並提供實驗排序，但沒有同時通過實質 loss 改善、多模型修正、完整相同 ML 零假設與穩定性門檻，所以沒有足夠證據支持歷史資料能預測下一期。

負責任投注：每一注事前固定六號的頭獎機率相同；增加不重複注數只增加覆蓋並同比增加成本。請設定可承受預算，不追損。
