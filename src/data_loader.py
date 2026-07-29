"""台灣彩券官方年度 ZIP 下載與資料匯入。"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import ROOT
from .data_cleaner import clean_draws, merge_draws, read_user_file

LOGGER = logging.getLogger(__name__)
OFFICIAL_API = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/ResultDownload"
OFFICIAL_PAGE = "https://www.taiwanlottery.com/lotto/history/result_download/"


def available_end_year(today: date | None = None) -> int:
    """官方頁面當年度選項使用西元年；目前年度通常已有逐月更新檔。"""
    return (today or date.today()).year


def _official_download_info(year: int, api_url: str = OFFICIAL_API) -> dict[str, str]:
    response = requests.get(api_url, params={"year": year}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("rtCode") != 0 or not payload.get("content", {}).get("path"):
        raise RuntimeError(f"官方 API 未提供 {year} 年資料：{payload}")
    return payload["content"]


def download_official_years(
    raw_dir: str | Path,
    start_year: int = 2007,
    end_year: int | None = None,
    api_url: str = OFFICIAL_API,
    force: bool = False,
) -> list[Path]:
    """下載官方年度 ZIP，並寫入可稽核的 manifest。"""
    target = Path(raw_dir)
    if not target.is_absolute():
        target = ROOT / target
    target.mkdir(parents=True, exist_ok=True)
    end = end_year or available_end_year()
    downloaded: list[Path] = []
    manifest: dict[str, Any] = {
        "source_page": OFFICIAL_PAGE,
        "api": api_url,
        "accessed_date": date.today().isoformat(),
        "files": [],
    }
    for year in range(start_year, end + 1):
        output = target / f"{year}.zip"
        try:
            info = _official_download_info(year, api_url)
            if force or not output.exists():
                LOGGER.info("下載官方 %s 年資料：%s", year, info["path"])
                with requests.get(info["path"], timeout=120, stream=True) as response:
                    response.raise_for_status()
                    with output.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            handle.write(block)
            with zipfile.ZipFile(output) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("ZIP CRC 驗證失敗")
            downloaded.append(output)
            manifest["files"].append({"year": year, "path": str(output), "url": info["path"]})
        except Exception as exc:  # keep earlier official years usable
            LOGGER.warning("%s 年官方資料無法取得：%s", year, exc)
            manifest["files"].append({"year": year, "error": str(exc)})
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not downloaded:
        raise RuntimeError("沒有下載到任何官方年度資料")
    return downloaded


def _read_lotto_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).name.startswith("大樂透_") and "加開獎項" not in name and name.lower().endswith(".csv")
        ]
        # 2018 等舊 ZIP 沒有設定 UTF-8 filename flag，Python 會以 CP437
        # 顯示亂碼；此時改以 CSV 第一筆遊戲名稱辨識，不依賴檔名。
        if not candidates:
            candidates = []
            for name in archive.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                prefix = archive.read(name)[:2048].decode("utf-8-sig", errors="replace")
                if "\n大樂透," in prefix or "\r\n大樂透," in prefix:
                    candidates.append(name)
        if len(candidates) != 1:
            raise ValueError(f"{path.name} 內大樂透主檔數量不是 1：{candidates}")
        raw = archive.read(candidates[0])
    # 2007–2023 檔案的資料列在正式 13 欄後另有多個空欄；若讓
    # pandas 自動推斷，會把前四欄誤當多層索引而整列左移。
    return pd.read_csv(
        io.BytesIO(raw), encoding="utf-8-sig", usecols=range(13), index_col=False
    )


def import_official_archives(raw_dir: str | Path, output: str | Path) -> pd.DataFrame:
    """從年度 ZIP 擷取大樂透 CSV、清理、合併並輸出 processed CSV。"""
    directory = Path(raw_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    frames: list[pd.DataFrame] = []
    for archive in sorted(directory.glob("20??.zip")):
        try:
            year = archive.stem
            frames.append(clean_draws(_read_lotto_csv_from_zip(archive), OFFICIAL_PAGE, year))
        except Exception as exc:
            LOGGER.error("無法匯入 %s：%s", archive, exc)
            raise
    result = merge_draws(frames)
    destination = Path(output)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8-sig")
    return result


def import_manual(path: str | Path, output: str | Path, source: str = "使用者提供的官方匯出檔") -> pd.DataFrame:
    """匯入手動取得的官方 CSV/Excel；來源名稱由使用者明示。"""
    input_path = Path(path)
    result = clean_draws(read_user_file(input_path), source, date.today().isoformat())
    destination = Path(output)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8-sig")
    return result


def load_processed(path: str | Path) -> pd.DataFrame:
    """讀取已正規化資料並恢復日期與整數型別。"""
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise FileNotFoundError(f"找不到處理後資料：{source}；請先執行 download")
    data = pd.read_csv(source, encoding="utf-8-sig", dtype={"draw_id": str})
    data["draw_date"] = pd.to_datetime(data["draw_date"])
    return data.sort_values(["draw_date", "draw_id"]).reset_index(drop=True)
