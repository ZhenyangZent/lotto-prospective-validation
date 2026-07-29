# 限制

- 原始 next_draw_predictor.py 不存在。
- 共同外層只有最後20期且只涵蓋2026，統計力低。
- 完整 same-pipeline 實際1,000份，未達2,000目標；RandomForest未納入。
- 10,000份 simplified 不含ML，不得替代正式p值。
- 推薦bootstrap完整1,000次，但不能把重抽穩定性解釋為中獎機率。
- manifest排除自身CSV/JSON與validation三個自我引用metadata。
- ZIP 內的 stage2_results_summary.json 對 bundle bytes/SHA 使用自我引用提示值；最終精確值記錄在ZIP外同名summary與 .zip.sha256 sidecar，避免宣稱不可能的自我雜湊固定點。
