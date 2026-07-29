"""官方檔案欄位正規化與清理。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

NUMBER_COLUMNS = [f"number_{i}" for i in range(1, 7)]
CANONICAL_COLUMNS = [
    "draw_id", "draw_date", *NUMBER_COLUMNS, "special_number",
    "sales_amount", "sales_count", "total_prize", "jackpot", "winner_count_by_tier",
    "prize_by_tier", "source", "source_version",
]

COLUMN_ALIASES = {
    "遊戲名稱": "game_name", "遊戲": "game_name", "期別": "draw_id",
    "開獎日期": "draw_date", "銷售總額": "sales_amount", "銷售金額": "sales_amount",
    "銷售注數": "sales_count", "總獎金": "total_prize", "特別號": "special_number",
    **{f"獎號{i}": f"number_{i}" for i in range(1, 7)},
    **{f"號碼{i}": f"number_{i}" for i in range(1, 7)},
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """移除欄名空白並套用中英文欄位別名。"""
    result = frame.copy()
    result.columns = [str(c).strip().replace("\ufeff", "") for c in result.columns]
    result = result.rename(columns=COLUMN_ALIASES)
    return result


def clean_draws(frame: pd.DataFrame, source: str, source_version: str) -> pd.DataFrame:
    """清理單一來源，保留大樂透並建立排序後的一般獎號。"""
    data = normalize_columns(frame)
    if "game_name" in data:
        games = set(data["game_name"].dropna().astype(str).str.strip())
        if games - {"大樂透"}:
            raise ValueError(f"輸入檔混入其他彩券遊戲：{sorted(games - {'大樂透'})}")
        data = data[data["game_name"].astype(str).str.strip().eq("大樂透")]
    required = ["draw_id", "draw_date", *NUMBER_COLUMNS, "special_number"]
    missing = [c for c in required if c not in data]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}")
    data = data.copy()
    data["draw_id"] = data["draw_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    data["draw_date"] = pd.to_datetime(data["draw_date"], errors="coerce")
    for column in [*NUMBER_COLUMNS, "special_number", "sales_amount", "sales_count", "total_prize", "jackpot"]:
        if column not in data:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")
    sorted_numbers = data[NUMBER_COLUMNS].apply(
        lambda row: sorted(row.astype("Int64").tolist()) if not row.isna().any() else [pd.NA] * 6,
        axis=1,
        result_type="expand",
    )
    sorted_numbers.columns = NUMBER_COLUMNS
    data[NUMBER_COLUMNS] = sorted_numbers
    for column in ["winner_count_by_tier", "prize_by_tier"]:
        if column not in data:
            data[column] = json.dumps({}, ensure_ascii=False)
    data["source"] = source
    data["source_version"] = source_version
    data = data[CANONICAL_COLUMNS].drop_duplicates(subset=["draw_id", "draw_date"], keep="last")
    return data.sort_values(["draw_date", "draw_id"]).reset_index(drop=True)


def merge_draws(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """合併各年度資料並依期別與日期去重。"""
    frames = list(frames)
    if not frames:
        raise ValueError("沒有可合併的大樂透資料")
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(subset=["draw_id", "draw_date"], keep="last").sort_values(
        ["draw_date", "draw_id"]
    ).reset_index(drop=True)


def read_user_file(path: str | Path) -> pd.DataFrame:
    """讀取使用者提供的 CSV 或 Excel。"""
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "cp950"):
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"無法辨識 CSV 編碼：{file_path}")
    if file_path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(file_path)
    raise ValueError("僅支援 CSV、XLS 或 XLSX")
