from math import comb, isclose

from src.probability import (
    consecutive_probability, inclusion_probability, jackpot_probability,
    odd_count_distribution, overlap_distribution, prize_probabilities, total_combinations,
)


def test_combinations_and_jackpot() -> None:
    assert total_combinations() == 13_983_816
    assert jackpot_probability() == 1 / 13_983_816


def test_probability_distributions_sum_to_one() -> None:
    assert isclose(sum(overlap_distribution().values()), 1.0)
    assert isclose(sum(odd_count_distribution().values()), 1.0)
    assert isclose(inclusion_probability(1), 6 / 49)
    assert isclose(inclusion_probability(2), comb(47, 4) / comb(49, 6))
    assert 0 < consecutive_probability() < 1


def test_prize_probability_is_valid() -> None:
    probabilities = prize_probabilities()
    assert probabilities["頭獎（6）"] == jackpot_probability()
    assert 0 < sum(probabilities.values()) < 1

