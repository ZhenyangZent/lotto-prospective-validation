"""台灣大樂透分析系統命令列介面。"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import random_baseline_distribution, summarize_backtest, walk_forward_backtest, yearly_backtest
from .config import ROOT, load_config
from .data_loader import download_official_years, import_manual, import_official_archives, load_processed
from .descriptive_analysis import descriptive_summary
from .probability import theoretical_summary
from .randomness_tests import randomness_summary
from .reporting import generate_report
from .simulation import empirical_comparison, global_anomaly_pvalue, observed_metrics, simulate_histories
from .strategies import default_strategies, diverse_tickets
from .validation import validate_draws
from .visualization import create_analysis_figures, create_backtest_figures, create_simulation_figures

LOGGER = logging.getLogger("lotto_analysis")


class NumpyEncoder(json.JSONEncoder):
    """將 numpy/pandas scalar 轉成標準 JSON。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
        if isinstance(obj, Path): return str(obj)
        return super().default(obj)


def _path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config["root"] / path


def _results_path(config: dict[str, Any]) -> Path:
    return config["root"] / "reports" / "analysis_summary.json"


def _save_results(config: dict[str, Any], values: dict[str, Any]) -> None:
    path = _results_path(config); path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values)
    current["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2, cls=NumpyEncoder), encoding="utf-8")


def command_download(config: dict[str, Any], force: bool = False, manual: str | None = None) -> pd.DataFrame:
    data_cfg = config["data"]
    if manual:
        LOGGER.info("匯入使用者指定檔案：%s", manual)
        return import_manual(manual, data_cfg["processed_file"])
    archives = download_official_years(
        data_cfg["raw_dir"], int(data_cfg["start_year"]), data_cfg.get("end_year"),
        data_cfg.get("api_url"), force,
    )
    LOGGER.info("可用官方年度檔：%d", len(archives))
    return import_official_archives(data_cfg["raw_dir"], data_cfg["processed_file"])


def command_validate(config: dict[str, Any]) -> dict[str, Any]:
    data = load_processed(config["data"]["processed_file"])
    report = validate_draws(data)
    output = config["root"] / "reports" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if not report.valid:
        raise ValueError(f"資料驗證失敗，詳見 {output}")
    LOGGER.info("資料驗證通過：%d 期，%s 至 %s", report.row_count, report.start_date, report.end_date)
    return report.to_dict()


def command_analyze(config: dict[str, Any]) -> None:
    data = load_processed(config["data"]["processed_file"])
    validation = validate_draws(data, strict=True).to_dict()
    analysis_cfg = config["analysis"]
    description = descriptive_summary(data, list(analysis_cfg["recent_windows"]), float(analysis_cfg["alpha"]))
    random = randomness_summary(data, float(analysis_cfg["alpha"]))
    table_dir = config["root"] / "reports" / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in description["frequencies"].items():
        table.to_csv(table_dir / f"frequency_{name}.csv", index=False, encoding="utf-8-sig")
    description["patterns"].to_csv(table_dir / "draw_patterns.csv", index=False, encoding="utf-8-sig")
    description["pairs"].to_csv(table_dir / "pair_cooccurrence.csv", index=False, encoding="utf-8-sig")
    description["triples"].to_csv(table_dir / "triple_cooccurrence.csv", index=False, encoding="utf-8-sig")
    random["number_tests"].to_csv(table_dir / "randomness_number_tests.csv", index=False, encoding="utf-8-sig")
    random["cross_dependence"].to_csv(table_dir / "cross_dependence.csv", index=False, encoding="utf-8-sig")
    random_json = {key: value for key, value in random.items() if not isinstance(value, pd.DataFrame)}
    _save_results(config, {"validation": validation, "theory": theoretical_summary(), "randomness": random_json})
    create_analysis_figures(data, description["frequencies"]["all"], description["patterns"],
                            description["pairs"], config["root"] / "reports" / "figures")
    LOGGER.info("分析完成：表格 %s；圖表 %s", table_dir, config["root"] / "reports" / "figures")


def command_simulate(config: dict[str, Any], iterations: int | None = None) -> None:
    data = load_processed(config["data"]["processed_file"])
    analysis_cfg = config["analysis"]
    count = int(iterations or analysis_cfg["monte_carlo_iterations"])
    LOGGER.info("模擬 %d 次、每次 %d 期完整公平序列", count, len(data))
    simulations = simulate_histories(len(data), count, int(config["seed"]), int(analysis_cfg["monte_carlo_batch_size"]))
    observed = observed_metrics(data); comparison = empirical_comparison(observed, simulations)
    table_dir = config["root"] / "reports" / "tables"
    simulations.to_csv(table_dir / "monte_carlo_simulations.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(table_dir / "simulation_comparison.csv", index=False, encoding="utf-8-sig")
    global_p = global_anomaly_pvalue(comparison)
    _save_results(config, {"simulation": {"iterations": count, "observed": observed,
                                           "global_anomaly_pvalue": global_p}})
    create_simulation_figures(comparison, simulations, config["root"] / "reports" / "figures")
    LOGGER.info("Monte Carlo 完成；全域異常 p 值近似 %.4g", global_p)


def command_backtest(config: dict[str, Any]) -> None:
    data = load_processed(config["data"]["processed_file"])
    cfg = config["backtest"]; seed = int(config["seed"])
    strategies = default_strategies(seed)
    LOGGER.info("開始 %d 個策略的 walk-forward 回測", len(strategies))
    detailed = walk_forward_backtest(data, strategies, int(cfg["min_train_draws"]), cfg.get("max_predictions"), int(cfg["ml_refit_interval"]))
    summary = summarize_backtest(detailed, int(cfg["bootstrap_iterations"]), seed)
    yearly = yearly_backtest(detailed)
    baseline = random_baseline_distribution(
        int(detailed["target_index"].nunique()), int(cfg["random_baseline_repetitions"]), seed
    )
    table_dir = config["root"] / "reports" / "tables"
    detailed.to_csv(table_dir / "backtest_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(table_dir / "backtest_yearly.csv", index=False, encoding="utf-8-sig")
    _save_results(config, {"backtest": {"random_baseline": baseline, "strategies": summary.to_dict("records")}})
    create_backtest_figures(summary, yearly, config["root"] / "reports" / "figures")
    LOGGER.info("回測完成；最高平均命中策略：%s", summary.iloc[0]["strategy"])


def command_recommend(config: dict[str, Any], tickets: int, strategy_name: str, unrestricted: bool) -> None:
    data = load_processed(config["data"]["processed_file"])
    strategies = {item.name.lower(): item for item in default_strategies(int(config["seed"]))}
    key = strategy_name.lower()
    if key not in strategies:
        raise ValueError(f"未知策略 {strategy_name}；可用：{sorted(strategies)}")
    strategy = strategies[key]
    result = strategy.predict(data)
    combinations = diverse_tickets(strategy, data, tickets, unrestricted=unrestricted)
    print("統計模型推薦／實驗性選號；不保證提高中獎機率。")
    print(f"策略：{result.strategy}；理由：{result.reason}")
    for index, ticket in enumerate(combinations, 1): print(f"{index:02d}: {' '.join(f'{n:02d}' for n in ticket)}")


def command_report(config: dict[str, Any]) -> Path:
    data = load_processed(config["data"]["processed_file"])
    path = _results_path(config)
    if not path.exists():
        raise FileNotFoundError("尚無分析摘要；請先執行 analyze")
    results = json.loads(path.read_text(encoding="utf-8"))
    report = generate_report(data, results, config["root"] / "reports" / "report.md")
    LOGGER.info("報告已產生：%s", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="台灣大樂透歷史資料分析系統")
    parser.add_argument("--config", help="YAML 設定檔路徑")
    parser.add_argument("--verbose", action="store_true", help="顯示除錯訊息")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="下載及匯入官方年度資料")
    download.add_argument("--force", action="store_true", help="覆寫既有年度 ZIP")
    download.add_argument("--manual", help="改為匯入指定的官方 CSV/Excel")
    sub.add_parser("validate", help="驗證處理後資料")
    sub.add_parser("analyze", help="理論、描述與隨機性分析及圖表")
    simulate = sub.add_parser("simulate", help="Monte Carlo 完整序列模擬")
    simulate.add_argument("--iterations", type=int, help="覆寫設定的模擬次數")
    sub.add_parser("backtest", help="walk-forward 策略回測")
    recommend = sub.add_parser("recommend", help="輸出實驗性多注組合")
    recommend.add_argument("--tickets", type=int, default=10)
    recommend.add_argument("--strategy", default="BayesianShrinkage")
    recommend.add_argument("--unrestricted", action="store_true")
    sub.add_parser("report", help="產生繁體中文報告")
    sub.add_parser("all", help="依序下載、驗證、分析、模擬、回測及報告")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    try:
        if args.command == "download": command_download(config, args.force, args.manual)
        elif args.command == "validate": command_validate(config)
        elif args.command == "analyze": command_analyze(config)
        elif args.command == "simulate": command_simulate(config, args.iterations)
        elif args.command == "backtest": command_backtest(config)
        elif args.command == "recommend": command_recommend(config, args.tickets, args.strategy, args.unrestricted)
        elif args.command == "report": command_report(config)
        elif args.command == "all":
            command_download(config); command_validate(config); command_analyze(config)
            command_simulate(config); command_backtest(config); command_report(config)
        return 0
    except Exception:
        LOGGER.exception("命令執行失敗")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
