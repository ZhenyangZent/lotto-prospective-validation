# 台灣大樂透前瞻驗證 V2 預註冊

- Experiment：`TW649-PROSPECTIVE-20260729-V2`
- 模型：`ProspectiveBatchLogistic-v1`（與 V1 完全相同）
- 第一目標期：115000075（2026-07-31）
- 凍結時間：2026-07-29T13:58:08+08:00

V2 不重新選模型或參數。特徵、C=0.01、L2/lbfgs、seed=20260729、EWMA alpha=0.06、transition alpha=30、長期先驗12、收縮0.10、capped-simplex與評估/成功標準均沿用 V1。

唯一設計修正是兩階段 Git 證據：prediction 事件只記 parent commit 與 `PENDING_REMOTE_ANCHOR`；prediction commit push 後，必須以 `git ls-remote` 得到與 prediction commit 完全相同的 branch OID，並在官方結果仍未公布時追加 remote_anchor。status 只解析合法 anchor，不信任 prediction/result 物件內的 boolean。

跨平台重現要求 Top-6/10/12與完整排名一致、機率總和誤差不超過1e-12、逐號機率 `atol<=1e-12`。相同正式環境另保留嚴格重現。

未錨定、遠端 branch 消失、OID無法證實、開獎後驗證或 late prediction 均不計正式樣本，且不得事後補算。
