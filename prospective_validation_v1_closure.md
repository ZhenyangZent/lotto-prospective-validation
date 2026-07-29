# 台灣大樂透前瞻驗證 V1 關閉紀錄

- Experiment ID：`TW649-PROSPECTIVE-20260729-V1`
- 關閉時間：`2026-07-29T13:46:51+08:00`
- 關閉原因：`INVALID_REMOTE_TIME_ANCHOR_DESIGN`
- 正式有效樣本數：`0`

V1 第一筆預測已於目標期 115000075（2026-07-31）開獎前在本機建立，預測 commit 為 `1b517e01d015c7168e63af13527e433e8133937f`。然而當時沒有 `origin`，因此無法 push；ledger 中的 `git_commit` 是建立預測前的 parent commit `f0547e07f13367fc4c8c1aa60d2a5673b5dc3b9a`，不是包含預測檔案的 commit。

V1 的遠端確認流程沒有以 `git ls-remote` 取得實際 remote ref OID 並與 prediction commit 比對。能組合 commit URL 或 repository 存在，均不能證明該預測在開獎前已位於第三方遠端。

依 V1 bug policy 與 missing-draw policy：

- V1 正式有效樣本數固定為 0。
- V1 第 115000075 期不得在事後補算為正式樣本。
- 不刪除、不覆寫任何 V1 ledger、預測、機率或預註冊檔案。
- V1 全部紀錄保留作測試與稽核證據。
