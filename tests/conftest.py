from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def draws() -> pd.DataFrame:
    rows = []
    for index in range(80):
        numbers = [((index * 7 + offset * 8) % 49) + 1 for offset in range(6)]
        numbers = sorted(set(numbers))
        while len(numbers) < 6:
            candidate = (numbers[-1] % 49) + 1
            if candidate not in numbers: numbers.append(candidate)
        special = next(n for n in range(1, 50) if n not in numbers)
        rows.append({
            "draw_id": str(100000000 + index), "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index * 3),
            **{f"number_{i+1}": value for i, value in enumerate(sorted(numbers))},
            "special_number": special, "source": "test", "source_version": "test",
        })
    return pd.DataFrame(rows)

