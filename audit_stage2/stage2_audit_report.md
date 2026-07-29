# 台灣大樂透第二階段獨立驗證報告

## 分離判定

`original_delivery_verdict`：**C－原始交付存在重大可重現性缺陷**。原因是初始快照沒有 `next_draw_predictor.py`，原始聲稱無法重現。

`audited_implementation_verdict`：**B－實作可用，但沒有足夠預測證據**。審計版的特徵更新、期別群組、機率投影及未來不變性測試通過；原始交付缺失不再被當成審計版 Bug。

`predictive_evidence_verdict`：**否**。命中、Brier、Log Loss 沒有在多模型與 same-pipeline 修正後同方向顯著改善。

## 統一回測

為使 sklearn Batch 的 1,000 份相同流程可完成，預先固定 inner 100 期（target 2028–2127）與共同 outer 最後 20 期（target 2128–2147）。這段太短、只涵蓋 2026，是主要統計限制。

Batch expanding 每期重新 fit StandardScaler、OneHotEncoder、LogisticRegression，結果為命中 1.100000、Brier 0.107369988、Log Loss 0.371381071。OnlineSGDLogistic 明確使用 partial_fit，結果 0.750000、0.107302378、0.371105004。Fixed holdout 只作補充，沒有把差額解釋成模型改善。

內層選擇規則在看外層前固定為「inner Brier ascending; then inner Log Loss ascending; then fixed simplicity rank」，選到 `FullFeatureBatchLogistic`。正式主結果即該模型，不是外層命中最高者。

## same-pipeline null

實際完成 1,000 份，每份 2,148 期，全部重新建立相同特徵，評估九個基準、八個 C 的 FullFeatureBatchLogistic 與 FullFeatureOnlineSGD，再以同一內層規則選模及執行外層。RandomForest 因完整成本未納入。

plus-one family-wise p 分別為：命中 0.033966（Wilson 95% CI [0.0235929279584269, 0.04598126446708158]）、Brier 0.091908、Log Loss 0.091908。命中在短短 20 期看似突出，但 loss 未同時通過，因此不能判定預測能力。另完成 10,000 份明確標為 simplified 的非 ML 流程，其 p 值不冒充完整 ML 校正。

## Recent50Hot

共同 20 期 Recent50 命中較高但 Brier/Log Loss 較差。另以 1,648 期檢查 45/50/55 視窗、前後半、逐年、排除最佳年度與同分反向排序，結果保存在 `recent50_analysis.csv`。同分採分數降冪、號碼升冪；反向 tie diagnostic 可量測固定號碼順序影響。完整 null 的內層 Brier 規則沒有選到 Recent20/50/100，顯示 Top-6 排序碰巧較好不等於完整機率模型較好。

## 上一期與 gap

上一期比較全部使用相同 Batch expanding outer。Full 與 RemoveBoth 都命中 1.100000；Full 的 Brier/Log Loss 只微幅差 -0.000012348/-0.000057265，樣本與 CI 不支持同時改善三指標。

gap 分箱提供 Wilson CI、Holm 修正及 1,000 次 draw-group permutation。全域 permutation p=0.578422；共同外層 Full、WithoutGap、GapOnly 的命中與 loss 另列。沒有證據支持「久未開較容易出現」。

## 推薦穩定性

完成 1,000 次 draw-group bootstrap、100 次刪除末 1–20 期擾動及全部 C、收縮、min_history、視窗設定。平均 Jaccard=0.220977，95% 範圍=[0.0, 0.5]；最常見組合只有 10/1123。**模型不存在穩定的唯一推薦組合。** 最新六號 1、5、15、22、27、47 只能稱為實驗性排序。

## 未完成

RandomForest 未納入完整 null；完整 same-pipeline 為最低 1,000 而非目標 2,000/5,000；共同 outer 只有 20 期且僅 2026，故跨年度 Batch outer 穩定性不可估；原始程式仍不存在；manifest 依標準排除三個自我引用 metadata，驗證 ZIP payload 數為 manifest rows，ZIP 總 entries 為 rows+3。
