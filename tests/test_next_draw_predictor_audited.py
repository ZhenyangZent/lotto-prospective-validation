"""審計規格的 24 項強制測試。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from next_draw_predictor_audited import (
    NUMBER_COLUMNS,
    append_recommendation_history,
    build_feature_dataset,
    build_next_features,
    evaluate_model,
    indicator_matrix,
    load_lotto_data,
    normalize_and_shrink_probabilities,
    online_walk_forward_evaluate,
    score_one_draw,
    stable_top_k,
    validate_lotto_data,
)


def synthetic_draws(count: int = 40) -> pd.DataFrame:
    rows = []
    for draw in range(count):
        numbers = sorted((((np.arange(6) * 7 + draw) % 49) + 1).tolist())
        special = next(value for value in range(1, 50) if value not in numbers)
        rows.append({"draw_id": f"{draw + 1:09d}", "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=draw),
                     **{column: numbers[index] for index, column in enumerate(NUMBER_COLUMNS)},
                     "special_number": special})
    return pd.DataFrame(rows)


def test_01_valid_csv_loads(tmp_path):
    path = tmp_path / "valid.csv"; synthetic_draws().to_csv(path, index=False)
    assert len(load_lotto_data(path)) == 40


def test_02_out_of_range_fails():
    data = synthetic_draws(); data.loc[0, "number_1"] = 50
    with pytest.raises(ValueError, match="介於"):
        validate_lotto_data(data)


def test_03_duplicate_number_in_draw_fails():
    data = synthetic_draws(); data.loc[0, "number_2"] = data.loc[0, "number_1"]
    with pytest.raises(ValueError, match="互不相同"):
        validate_lotto_data(data)


def test_04_missing_required_column_fails():
    with pytest.raises(ValueError, match="缺少必要欄位"):
        validate_lotto_data(synthetic_draws().drop(columns="number_6"))


def test_05_empty_data_fails():
    with pytest.raises(ValueError, match="不可為空"):
        validate_lotto_data(synthetic_draws(1).iloc[:0])


def test_06_insufficient_min_history_fails():
    with pytest.raises(ValueError, match="min_history"):
        build_feature_dataset(synthetic_draws(10), min_history=10)


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1])
def test_07_invalid_test_ratio_fails_safely(ratio):
    data = synthetic_draws(); features = build_feature_dataset(data, min_history=5)
    with pytest.raises(ValueError, match="test_ratio"):
        evaluate_model(features, data, test_ratio=ratio)


def test_08_time_split_keeps_all_49_candidates_together():
    data = synthetic_draws(); features = build_feature_dataset(data, min_history=5)
    _, predictions, per_number, _ = evaluate_model(features, data, test_ratio=.25)
    assert per_number.groupby("target_index").size().eq(49).all()
    assert set(predictions["target_index"]) == set(per_number["target_index"])


def test_09_future_change_does_not_change_past_features():
    data = synthetic_draws(35); target = 20
    original = build_feature_dataset(data, min_history=5).query("target_index == @target").reset_index(drop=True)
    changed = data.copy()
    for row in range(target + 1, len(changed)):
        values = sorted((((np.arange(6) * 5 + row + 11) % 49) + 1).tolist())
        changed.loc[row, NUMBER_COLUMNS] = values
    altered = build_feature_dataset(changed, min_history=5).query("target_index == @target").reset_index(drop=True)
    pd.testing.assert_frame_equal(original, altered)


def test_10_transition_update_has_no_target_leakage():
    data = synthetic_draws(25); target = 15
    before = build_feature_dataset(data, 5).query("target_index == @target")["transition_rate"].to_numpy()
    changed = data.copy(); changed.loc[target, NUMBER_COLUMNS] = [2, 9, 16, 23, 30, 37]
    after = build_feature_dataset(changed, 5).query("target_index == @target")["transition_rate"].to_numpy()
    np.testing.assert_allclose(before, after)


def test_11_ewma_update_has_no_target_leakage():
    data = synthetic_draws(25); target = 15
    before = build_feature_dataset(data, 5).query("target_index == @target")["ewma_rate"].to_numpy()
    changed = data.copy(); changed.loc[target, NUMBER_COLUMNS] = [2, 9, 16, 23, 30, 37]
    after = build_feature_dataset(changed, 5).query("target_index == @target")["ewma_rate"].to_numpy()
    np.testing.assert_allclose(before, after)


def test_12_gap_has_no_off_by_one():
    data = synthetic_draws(12)
    features = build_feature_dataset(data, min_history=2)
    target = 5; previous_numbers = set(data.loc[target - 1, NUMBER_COLUMNS])
    snapshot = features.query("target_index == @target").set_index("number")
    assert snapshot.loc[list(previous_numbers), "gap"].eq(0).all()


def test_13_next_and_historical_feature_definitions_match():
    full = synthetic_draws(25); history = full.iloc[:-1].reset_index(drop=True)
    next_features = build_next_features(history).set_index("number")
    historical = build_feature_dataset(full, min_history=5).query("target_index == 24").set_index("number")
    columns = ["long_z", "recent20_z", "recent50_z", "recent100_z", "ewma_z", "gap_z",
               "transition_z", "in_last_draw", "long_rate", "ewma_rate", "gap", "transition_rate"]
    pd.testing.assert_frame_equal(next_features[columns], historical[columns])


def test_14_output_exactly_six_unique_legal_numbers():
    numbers = stable_top_k(np.arange(49, dtype=float), 6)
    assert len(numbers) == len(set(numbers)) == 6 and all(1 <= n <= 49 for n in numbers)


def test_15_same_input_and_seed_is_reproducible():
    data = synthetic_draws(30); features = build_feature_dataset(data, 5)
    first = online_walk_forward_evaluate(features, data, first_target=20, seed=7)[0]
    second = online_walk_forward_evaluate(features, data, first_target=20, seed=7)[0]
    pd.testing.assert_frame_equal(first, second)


def test_16_probabilities_sum_to_six():
    rng = np.random.default_rng(1)
    assert normalize_and_shrink_probabilities(rng.normal(size=49)).sum() == pytest.approx(6)


def test_17_probabilities_are_bounded():
    values = normalize_and_shrink_probabilities(np.r_[[1e9], np.full(48, -1e9)], 0)
    assert np.all((0 <= values) & (values <= 1))


def test_18_brier_matches_manual_example():
    probabilities = np.full(49, 6 / 49)
    result = score_one_draw(probabilities, [1, 2, 3, 4, 5, 6])
    target = np.r_[np.ones(6), np.zeros(43)]
    assert result["brier"] == pytest.approx(np.mean((probabilities - target) ** 2))


def test_19_top6_hits_match_manual_example():
    probabilities = np.arange(49, dtype=float)
    result = score_one_draw(probabilities, [44, 45, 46, 47, 48, 49])
    assert result["hits_top6"] == 6


def test_20_hypergeometric_pmf_sums_to_one():
    denominator = math.comb(49, 6)
    pmf = [math.comb(6, h) * math.comb(43, 6 - h) / denominator for h in range(7)]
    assert sum(pmf) == pytest.approx(1)


def test_21_theoretical_mean_is_36_over_49():
    denominator = math.comb(49, 6)
    pmf = np.array([math.comb(6, h) * math.comb(43, 6 - h) / denominator for h in range(7)])
    assert np.dot(np.arange(7), pmf) == pytest.approx(36 / 49)


def test_22_recommendation_history_is_idempotent(tmp_path):
    path = tmp_path / "history.json"
    record = {"data_sha256": "abc", "model_version": "v1", "numbers": [1, 2, 3, 4, 5, 6]}
    assert append_recommendation_history(path, record) is True
    assert append_recommendation_history(path, record) is False
    assert len(pd.read_json(path)) == 1


def test_23_null_unsorted_and_duplicate_date_handling():
    data = synthetic_draws(); data.loc[0, "number_1"] = np.nan
    with pytest.raises(ValueError, match="空值"):
        validate_lotto_data(data)
    data = synthetic_draws(); data.loc[2, "draw_date"] = data.loc[1, "draw_date"]
    with pytest.raises(ValueError, match="draw_date"):
        validate_lotto_data(data)
    data = synthetic_draws().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="遞增"):
        validate_lotto_data(data)


def test_24_walk_forward_uses_only_strictly_past_targets():
    data = synthetic_draws(35); features = build_feature_dataset(data, 5)
    _, _, metadata = online_walk_forward_evaluate(features, data, first_target=20)
    assert (metadata["max_training_target"] < metadata["test_target"]).all()
