import pandas as pd

from src.feature_engineering import build_number_features, supervised_number_dataset


def test_features_do_not_see_future(draws: pd.DataFrame) -> None:
    first = build_number_features(draws, as_of=50)
    changed_future = draws.copy()
    changed_future.loc[50:, [f"number_{i}" for i in range(1, 7)]] = [1, 2, 3, 4, 5, 6]
    second = build_number_features(changed_future, as_of=50)
    pd.testing.assert_frame_equal(first, second)


def test_supervised_target_is_next_draw(draws: pd.DataFrame) -> None:
    features, labels = supervised_number_dataset(draws.iloc[:35], start=30)
    first_block = features[features["target_index"] == 30]
    actual = set(draws.iloc[30][[f"number_{i}" for i in range(1, 7)]].astype(int))
    predicted_labels = set((first_block.loc[labels[:49].astype(bool), "number"]).astype(int))
    assert predicted_labels == actual

