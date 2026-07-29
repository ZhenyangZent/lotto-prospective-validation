"""資料品質驗證。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .data_cleaner import NUMBER_COLUMNS


@dataclass
class ValidationIssue:
    """單一驗證問題。"""

    severity: str
    code: str
    message: str
    rows: list[int] = field(default_factory=list)


@dataclass
class ValidationReport:
    """可序列化的驗證報告。"""

    valid: bool
    row_count: int
    start_date: str | None
    end_date: str | None
    issues: list[ValidationIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid, "row_count": self.row_count,
            "start_date": self.start_date, "end_date": self.end_date,
            "issues": [asdict(item) for item in self.issues],
        }


def validate_draws(data: pd.DataFrame, strict: bool = False) -> ValidationReport:
    """檢查範圍、重複、空值、遊戲混入與基本時間一致性。"""
    issues: list[ValidationIssue] = []
    required = ["draw_id", "draw_date", *NUMBER_COLUMNS, "special_number"]
    missing_columns = [c for c in required if c not in data]
    if missing_columns:
        issues.append(ValidationIssue("error", "missing_columns", f"缺少欄位：{missing_columns}"))
        report = ValidationReport(False, len(data), None, None, issues)
        if strict:
            raise ValueError(report.to_dict())
        return report

    missing = data[required].isna().any(axis=1)
    if missing.any():
        issues.append(ValidationIssue("error", "null_required", "必要欄位存在空值", data.index[missing].tolist()))
    numbers = data[NUMBER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_range = ~numbers.apply(lambda col: col.between(1, 49)).all(axis=1)
    if bad_range.any():
        issues.append(ValidationIssue("error", "main_out_of_range", "一般獎號不在 1–49", data.index[bad_range].tolist()))
    duplicate_numbers = numbers.nunique(axis=1).ne(6)
    if duplicate_numbers.any():
        issues.append(ValidationIssue("error", "duplicate_main", "同一期一般獎號重複", data.index[duplicate_numbers].tolist()))
    special = pd.to_numeric(data["special_number"], errors="coerce")
    bad_special = ~special.between(1, 49)
    if bad_special.any():
        issues.append(ValidationIssue("error", "special_out_of_range", "特別號不在 1–49", data.index[bad_special].tolist()))
    special_repeated = pd.Series(
        [s in set(row) for s, row in zip(special, numbers.to_numpy())], index=data.index
    )
    if special_repeated.any():
        issues.append(ValidationIssue("error", "special_repeated", "特別號與一般獎號重複", data.index[special_repeated].tolist()))
    for subset, code, message in [
        (["draw_id"], "duplicate_draw_id", "期別重複"),
        (["draw_date"], "duplicate_date", "開獎日期重複"),
    ]:
        duplicated = data.duplicated(subset=subset, keep=False)
        if duplicated.any():
            issues.append(ValidationIssue("error", code, message, data.index[duplicated].tolist()))
    dates = pd.to_datetime(data["draw_date"], errors="coerce")
    invalid_dates = dates.isna()
    if invalid_dates.any():
        issues.append(ValidationIssue("error", "invalid_date", "日期格式無法解析", data.index[invalid_dates].tolist()))
    if len(data) < 50:
        issues.append(ValidationIssue("warning", "small_dataset", "資料少於 50 期，推論力很低"))
    if len(data) > 1 and not dates.is_monotonic_increasing:
        issues.append(ValidationIssue("warning", "date_order", "資料未依日期排序"))
    numeric_ids = pd.to_numeric(data["draw_id"], errors="coerce")
    if numeric_ids.isna().any():
        issues.append(ValidationIssue("warning", "non_numeric_draw_id", "部分期別不是九碼數字，無法檢查缺期"))
    else:
        missing_ids: list[int] = []
        ids = numeric_ids.astype(int)
        # ROC 年（前三碼）內的流水號應連續；跨年自然重設。
        for _, group in ids.groupby(ids.astype(str).str.zfill(9).str[:3]):
            observed = set(group.tolist())
            missing_ids.extend(sorted(set(range(min(observed), max(observed) + 1)) - observed))
        if missing_ids:
            issues.append(ValidationIssue(
                "warning", "missing_draw_ids", f"偵測到 {len(missing_ids)} 個期別缺口：{missing_ids[:20]}"
            ))
    errors = [x for x in issues if x.severity == "error"]
    report = ValidationReport(
        not errors, len(data),
        None if dates.isna().all() else dates.min().date().isoformat(),
        None if dates.isna().all() else dates.max().date().isoformat(), issues,
    )
    if strict and not report.valid:
        raise ValueError(report.to_dict())
    return report
