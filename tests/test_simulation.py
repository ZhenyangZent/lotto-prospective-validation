import numpy as np

from src.simulation import simulate_histories


def test_simulator_is_reproducible_and_close_to_theory() -> None:
    first = simulate_histories(100, iterations=80, seed=9, batch_size=20)
    second = simulate_histories(100, iterations=80, seed=9, batch_size=20)
    assert first.equals(second)
    # 公平 1..49 六號的總和期望為 6*25=150。
    assert abs(first["mean_sum"].mean() - 150) < 2
