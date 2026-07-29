import pandas as pd

from src.data_cleaner import clean_draws, merge_draws
from src.validation import validate_draws


def test_valid_data_passes(draws: pd.DataFrame) -> None:
    assert validate_draws(draws).valid


def test_main_and_special_validation(draws: pd.DataFrame) -> None:
    broken = draws.copy()
    broken.loc[0, "number_1"] = 50
    broken.loc[1, "special_number"] = broken.loc[1, "number_1"]
    report = validate_draws(broken)
    codes = {issue.code for issue in report.issues}
    assert {"main_out_of_range", "special_repeated"} <= codes
    assert not report.valid


def test_deduplication() -> None:
    raw = pd.DataFrame([
        {"遊戲名稱": "大樂透", "期別": "1", "開獎日期": "2024/01/01", "獎號1": 6, "獎號2": 5,
         "獎號3": 4, "獎號4": 3, "獎號5": 2, "獎號6": 1, "特別號": 7},
    ])
    cleaned = clean_draws(raw, "official", "v1")
    merged = merge_draws([cleaned, cleaned])
    assert len(merged) == 1
    assert merged.loc[0, [f"number_{i}" for i in range(1, 7)]].tolist() == [1, 2, 3, 4, 5, 6]

