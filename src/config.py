"""設定檔與共用路徑。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LotteryRules:
    """大樂透基本規則。"""

    min_number: int = 1
    max_number: int = 49
    main_numbers: int = 6
    ticket_price_twd: int = 50


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """讀取 YAML 設定並補上絕對專案根目錄。"""
    config_path = Path(path) if path else ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["root"] = ROOT
    return config


def rules_from_config(config: dict[str, Any]) -> LotteryRules:
    """將設定轉為不可變規則物件。"""
    values = config.get("lottery", {})
    return LotteryRules(
        min_number=int(values.get("min_number", 1)),
        max_number=int(values.get("max_number", 49)),
        main_numbers=int(values.get("main_numbers", 6)),
        ticket_price_twd=int(values.get("ticket_price_twd", 50)),
    )

