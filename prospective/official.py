"""只接受台灣彩券官方 API 的資料取得與存證。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from urllib.parse import urlparse

from next_draw_predictor_audited import NUMBER_COLUMNS, load_lotto_data
from .config import DATA_PATH, ROOT, STATE_DIR, TIMEZONE

API_BASE = "https://api.taiwanlottery.com/TLCAPIWeB"
OFFICIAL_RESULT_ENDPOINT = f"{API_BASE}/Lottery/Lotto649Result"
OFFICIAL_NEXT_ENDPOINT = f"{API_BASE}/Lottery/NextDrawDate"
OFFICIAL_PAGE = "https://www.taiwanlottery.com/lotto/result/lotto649/"
OFFICIAL_HOSTS = {"api.taiwanlottery.com", "www.taiwanlottery.com", "apislb.taiwanlottery.com"}
GAME_CODE = 5118


def validate_official_source(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS:
        raise ValueError("結果來源不是允許的台灣彩券官方 HTTPS host")


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_official_source(url)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("rtCode") != 0:
        raise RuntimeError(f"官方 API 錯誤：{payload}")
    return payload


def normalize_result(item: dict[str, Any]) -> dict[str, Any]:
    drawn = [int(value) for value in item["drawNumberSize"]]
    if len(drawn) != 7 or len(set(drawn)) != 7:
        raise ValueError("官方大樂透結果不是六個一般號加一個特別號")
    return {
        "draw_id": str(item["period"]),
        "draw_date": pd.Timestamp(item["lotteryDate"]).date().isoformat(),
        "numbers": sorted(drawn[:6]),
        "special_number": drawn[6],
        "sales_amount": item.get("sellAmount"),
        "total_prize": item.get("totalAmount"),
    }


def fetch_draw(draw_id: str, fetcher: Callable[..., dict[str, Any]] = _get_json) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    raw = fetcher(OFFICIAL_RESULT_ENDPOINT, {"period": str(draw_id), "pageNum": 1, "pageSize": 10})
    items = raw.get("content", {}).get("lotto649Res", [])
    exact = [item for item in items if str(item.get("period")) == str(draw_id)]
    if len(exact) > 1:
        raise ValueError("官方來源回傳重複期別")
    return (normalize_result(exact[0]) if exact else None), raw


def fetch_month(month: str, fetcher: Callable[..., dict[str, Any]] = _get_json) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = fetcher(OFFICIAL_RESULT_ENDPOINT, {"month": month, "endMonth": month, "pageNum": 1, "pageSize": 200})
    return [normalize_result(item) for item in raw.get("content", {}).get("lotto649Res", [])], raw


def fetch_next_draw(fetcher: Callable[..., dict[str, Any]] = _get_json) -> tuple[dict[str, str], dict[str, Any]]:
    raw = fetcher(OFFICIAL_NEXT_ENDPOINT, None)
    rows = [row for row in raw.get("content", {}).get("nextDrawDateList", []) if int(row.get("gameCode", -1)) == GAME_CODE]
    if len(rows) != 1 or rows[0].get("drawTerm") is None:
        raise RuntimeError("官方 API 未提供唯一的大樂透下一期")
    row = rows[0]
    return {"draw_id": str(row["drawTerm"]), "draw_date": datetime.strptime(str(row["drawDate"]), "%Y%m%d").date().isoformat()}, raw


def save_raw_response(payload: dict[str, Any], stem: str) -> Path:
    directory = STATE_DIR / "official_responses"; directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.json"
    if path.exists():
        suffix = 2
        while (directory / f"{stem}-{suffix}.json").exists(): suffix += 1
        path = directory / f"{stem}-{suffix}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return path


def update_current_month(data_path: str | Path = DATA_PATH, now: datetime | None = None) -> dict[str, Any]:
    """以官方即時結果補上年度下載尚未涵蓋的當月完成期。"""
    current = now or datetime.now()
    month = current.strftime("%Y-%m")
    rows, raw = fetch_month(month)
    raw_path = save_raw_response(raw, f"lotto649-{month}")
    data = pd.read_csv(data_path, encoding="utf-8-sig", dtype={"draw_id": str})
    by_id = {str(row["draw_id"]): row for row in rows}
    existing = set(data["draw_id"].astype(str))
    additions: list[dict[str, Any]] = []
    for draw_id, item in by_id.items():
        if draw_id in existing:
            old = data.loc[data["draw_id"].astype(str).eq(draw_id)].iloc[0]
            if sorted(int(old[column]) for column in NUMBER_COLUMNS) != item["numbers"]:
                raise ValueError(f"官方即時結果與既有資料衝突：{draw_id}")
            continue
        row = {column: None for column in data.columns}
        row.update({"draw_id": draw_id, "draw_date": item["draw_date"], "special_number": item["special_number"],
                    "sales_amount": item["sales_amount"], "total_prize": item["total_prize"],
                    "source": OFFICIAL_PAGE, "source_version": f"live-api-{month}"})
        row.update({column: value for column, value in zip(NUMBER_COLUMNS, item["numbers"])})
        additions.append(row)
    if additions:
        data = pd.concat([data, pd.DataFrame(additions)], ignore_index=True).sort_values(["draw_date", "draw_id"])
        data.to_csv(data_path, index=False, encoding="utf-8-sig")
    validated = load_lotto_data(data_path)
    return {"added": len(additions), "draws": len(validated), "data_end_draw_id": str(validated.iloc[-1]["draw_id"]),
            "data_end_date": pd.Timestamp(validated.iloc[-1]["draw_date"]).date().isoformat(), "raw_response": str(raw_path.relative_to(ROOT))}
