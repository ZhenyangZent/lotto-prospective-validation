"""保存可逐位元稽核的台灣彩券官方 precheck 證據。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import requests

from prospective.official import OFFICIAL_RESULT_ENDPOINT, normalize_result, validate_official_source
from .config import STATE_DIR, TIMEZONE


class ResponseLike(Protocol):
    status_code: int
    headers: Any
    content: bytes
    url: str
    def raise_for_status(self) -> None: ...
    def json(self) -> dict[str, Any]: ...


def timestamp() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")


def file_stamp(value: str) -> str:
    return value.replace("-", "").replace(":", "").replace("+", "+")


def fetch_official_precheck(draw_id: str, requester: Any = requests.get,
                            output_dir: str | Path | None = None) -> dict[str, Any]:
    validate_official_source(OFFICIAL_RESULT_ENDPOINT)
    params = {"period": str(draw_id), "pageNum": 1, "pageSize": 10}
    obtained_at = timestamp()
    response: ResponseLike = requester(OFFICIAL_RESULT_ENDPOINT, params=params, timeout=30)
    raw = bytes(response.content)
    status_code = int(response.status_code)
    response.raise_for_status()
    payload = response.json()
    if payload.get("rtCode") != 0:
        raise RuntimeError(f"官方 API 錯誤：{payload}")
    rows = payload.get("content", {}).get("lotto649Res", [])
    exact = [item for item in rows if str(item.get("period")) == str(draw_id)]
    if len(exact) > 1:
        raise ValueError("官方來源回傳重複目標期")
    parsed = normalize_result(exact[0]) if exact else None
    status = "ANNOUNCED" if parsed else "NOT_ANNOUNCED"
    directory = Path(output_dir) if output_dir else STATE_DIR / "official_prechecks"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"official-precheck-{draw_id}-{file_stamp(obtained_at)}"
    raw_path = directory / f"{stem}.raw"
    metadata_path = directory / f"{stem}.json"
    raw_path.write_bytes(raw)
    metadata = {
        "http_retrieved_at": obtained_at,
        "http_status": status_code,
        "request_url": str(response.url),
        "request_endpoint": OFFICIAL_RESULT_ENDPOINT,
        "request_params": params,
        "response_headers": {str(key): str(value) for key, value in response.headers.items()},
        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_response_file": raw_path.name,
        "parsed_result": parsed,
        "target_draw_id": str(draw_id),
        "target_draw_status": status,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"metadata": metadata, "metadata_path": metadata_path, "raw_path": raw_path}
