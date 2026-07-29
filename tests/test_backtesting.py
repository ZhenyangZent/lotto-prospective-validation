import numpy as np
import pandas as pd

from src.backtesting import walk_forward_backtest, walk_forward_splits
from src.strategies import UniformRandomStrategy, default_strategies, diverse_tickets


def test_walk_forward_has_no_leakage() -> None:
    splits = list(walk_forward_splits(20, 10))
    assert splits[0][1] == 10
    assert all(train.max() < target for train, target in splits)
    assert splits[-1][1] == 19


def test_strategies_return_six_legal_unique_numbers(draws: pd.DataFrame) -> None:
    for strategy in default_strategies(42):
        result = strategy.predict(draws)
        assert len(result.numbers) == len(set(result.numbers)) == 6
        assert all(1 <= value <= 49 for value in result.numbers)
        assert np.isclose(result.probabilities.sum(), 6)


def test_seed_and_multi_ticket_reproducibility(draws: pd.DataFrame) -> None:
    one = UniformRandomStrategy(123).tickets(draws, 10)
    two = UniformRandomStrategy(123).tickets(draws, 10)
    assert one == two and len({tuple(x) for x in one}) == 10
    diverse = diverse_tickets(UniformRandomStrategy(123), draws, 10)
    assert len({tuple(x) for x in diverse}) == 10


def test_backtest_uses_expected_target_count(draws: pd.DataFrame) -> None:
    result = walk_forward_backtest(draws, [UniformRandomStrategy(1)], min_train=60)
    assert len(result) == 20
    assert result["target_index"].min() == 60

