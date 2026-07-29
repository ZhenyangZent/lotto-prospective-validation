"""統一選號策略介面、基準模型與機器學習實驗。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import build_number_features, indicator_matrix, supervised_number_dataset


@dataclass
class StrategyResult:
    """策略輸出。probabilities 是 49 個號碼的邊際機率且總和為 6。"""

    strategy: str
    probabilities: np.ndarray
    numbers: list[int]
    parameters: dict[str, Any]
    reason: str


def _probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values -= values.min(initial=0)
    values += 1e-9
    return values / values.sum() * 6


class Strategy(ABC):
    """所有策略的共同介面。"""

    name = "abstract"

    def __init__(self, seed: int = 0, **parameters: Any) -> None:
        self.seed = seed
        self.parameters = parameters

    @abstractmethod
    def score(self, history: pd.DataFrame) -> np.ndarray:
        """回傳 49 個相對分數。"""

    def predict(self, history: pd.DataFrame) -> StrategyResult:
        probs = _probabilities(self.score(history))
        numbers = (np.argsort(probs)[-6:] + 1).tolist()
        return StrategyResult(self.name, probs, sorted(numbers), {"seed": self.seed, **self.parameters}, self.reason())

    def reason(self) -> str:
        return "僅為歷史統計實驗，不保證提高中獎機率。"

    def tickets(self, history: pd.DataFrame, count: int) -> list[list[int]]:
        """依策略邊際權重抽出互不相同的多注。"""
        probabilities = self.predict(history).probabilities
        probabilities = probabilities / probabilities.sum()
        rng = np.random.default_rng(self.seed)
        tickets: set[tuple[int, ...]] = set()
        attempts = 0
        while len(tickets) < count and attempts < max(1000, count * 100):
            ticket = tuple(sorted((rng.choice(49, 6, replace=False, p=probabilities) + 1).tolist()))
            tickets.add(ticket); attempts += 1
        if len(tickets) < count:
            raise RuntimeError("在指定限制下無法產生足夠的不重複組合")
        return [list(ticket) for ticket in sorted(tickets)]


class UniformRandomStrategy(Strategy):
    name = "UniformRandom"

    def score(self, history: pd.DataFrame) -> np.ndarray:
        return np.ones(49)


class FrequencyRandomStrategy(Strategy):
    name = "FrequencyRandom"

    def score(self, history: pd.DataFrame) -> np.ndarray:
        return indicator_matrix(history).sum(axis=0) + 1


class RecentHotStrategy(Strategy):
    name = "RecentHot"

    def score(self, history: pd.DataFrame) -> np.ndarray:
        window = int(self.parameters.get("window", 50))
        return indicator_matrix(history.tail(window)).sum(axis=0) + 1


class ColdNumberStrategy(Strategy):
    name = "ColdNumber"

    def score(self, history: pd.DataFrame) -> np.ndarray:
        return build_number_features(history)["current_gap"].to_numpy() + 1


class MixedHotColdStrategy(Strategy):
    name = "MixedHotCold"

    def score(self, history: pd.DataFrame) -> np.ndarray:
        features = build_number_features(history)
        hot = features["freq_50"].rank(pct=True).to_numpy()
        cold = features["current_gap"].rank(pct=True).to_numpy()
        middle = 1 - np.abs(hot - 0.5) * 2
        return 0.4 * hot + 0.3 * cold + 0.3 * middle + 0.01


class EWMAFrequencyStrategy(Strategy):
    name = "EWMAFrequency"

    def score(self, history: pd.DataFrame) -> np.ndarray:
        return build_number_features(history)["ewma"].to_numpy() + 1e-3


class BayesianShrinkageStrategy(Strategy):
    name = "BayesianShrinkage"

    def score(self, history: pd.DataFrame) -> np.ndarray:
        matrix = indicator_matrix(history)
        prior_strength = float(self.parameters.get("prior_strength", 49.0))
        return (matrix.sum(axis=0) + prior_strength * 6 / 49) / (len(matrix) + prior_strength)


class MarkovStrategy(Strategy):
    name = "Markov"

    def score(self, history: pd.DataFrame) -> np.ndarray:
        matrix = indicator_matrix(history)
        if len(matrix) < 3:
            return np.ones(49)
        previous = matrix[:-1]; current = matrix[1:]
        last = matrix[-1]
        result = np.zeros(49)
        for idx in range(49):
            state = last[idx]
            mask = previous[:, idx] == state
            result[idx] = (current[mask, idx].sum() + 1) / (mask.sum() + 2)
        return result


FEATURE_COLUMNS = [
    "freq_5", "freq_10", "freq_20", "freq_50", "freq_100", "freq_200",
    "current_gap", "max_gap", "mean_gap", "ewma", "frequency_slope",
    "recent_long_diff", "previous_1", "previous_2",
]


class _MLStrategy(Strategy):
    minimum_training_draws = 60

    @abstractmethod
    def estimator(self):
        """建立 sklearn estimator。"""

    def score(self, history: pd.DataFrame) -> np.ndarray:
        if len(history) < self.minimum_training_draws:
            return np.ones(49)
        start = max(30, len(history) - int(self.parameters.get("training_window", 250)))
        train_source = history.iloc[max(0, start - 30):]
        features, labels = supervised_number_dataset(train_source, start=min(30, len(train_source) - 1))
        if features.empty or len(np.unique(labels)) < 2:
            return np.ones(49)
        model = self.estimator()
        model.fit(features[FEATURE_COLUMNS], labels)
        current = build_number_features(history)
        return model.predict_proba(current[FEATURE_COLUMNS])[:, 1]


class LogisticRegressionStrategy(_MLStrategy):
    name = "LogisticRegression"

    def estimator(self):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=self.seed, class_weight="balanced"))


class RandomForestStrategy(_MLStrategy):
    name = "RandomForest"

    def estimator(self):
        return RandomForestClassifier(
            n_estimators=int(self.parameters.get("n_estimators", 100)),
            max_depth=int(self.parameters.get("max_depth", 5)),
            min_samples_leaf=10, class_weight="balanced", random_state=self.seed, n_jobs=-1,
        )


def default_strategies(seed: int) -> list[Strategy]:
    """規格要求的十個基準／實驗策略。"""
    return [
        UniformRandomStrategy(seed), FrequencyRandomStrategy(seed), RecentHotStrategy(seed, window=50),
        ColdNumberStrategy(seed), MixedHotColdStrategy(seed), EWMAFrequencyStrategy(seed),
        BayesianShrinkageStrategy(seed, prior_strength=49), MarkovStrategy(seed),
        LogisticRegressionStrategy(seed, training_window=250),
        RandomForestStrategy(seed, training_window=250, n_estimators=50, max_depth=5),
    ]


def diverse_tickets(
    strategy: Strategy, history: pd.DataFrame, count: int,
    max_number_uses: int | None = None, unrestricted: bool = False,
) -> list[list[int]]:
    """貪婪產生多注，避免重複並提高注間差異；限制只改覆蓋方式，不改單注頭獎機率。"""
    if unrestricted:
        return strategy.tickets(history, count)
    probs = strategy.predict(history).probabilities
    probs = probs / probs.sum()
    rng = np.random.default_rng(strategy.seed)
    cap = max_number_uses or max(2, int(np.ceil(count * 6 / 49 * 2)))
    usage = np.zeros(49, dtype=int)
    chosen: list[tuple[int, ...]] = []
    for _ in range(count):
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for _ in range(300):
            weights = probs / (1 + usage)
            weights = weights / weights.sum()
            ticket = tuple(sorted((rng.choice(49, 6, replace=False, p=weights) + 1).tolist()))
            if ticket in chosen or any(usage[n - 1] >= cap for n in ticket):
                continue
            overlap = max((len(set(ticket) & set(old)) for old in chosen), default=0)
            candidates.append((float(sum(probs[n - 1] for n in ticket)) - overlap, ticket))
        if not candidates:
            # Relax the cap but never permit a duplicate ticket.
            for _ in range(1000):
                ticket = tuple(sorted((rng.choice(49, 6, replace=False, p=probs) + 1).tolist()))
                if ticket not in chosen:
                    candidates.append((0.0, ticket)); break
        best = max(candidates, key=lambda item: item[0])[1]
        chosen.append(best)
        usage[np.array(best) - 1] += 1
    return [list(ticket) for ticket in chosen]
