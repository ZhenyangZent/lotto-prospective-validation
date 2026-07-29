"""固定的 ProspectiveBatchLogistic-v1 與探索性基準。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from next_draw_predictor_audited import (
    BASE_FEATURE_COLUMNS, NUMBER_COLUMNS, UNIFORM_PROBABILITY,
    build_feature_dataset, build_next_features, indicator_matrix, make_model,
    normalize_and_shrink_probabilities, stable_top_k,
)
from .config import FROZEN_CONFIG, MODEL_VERSION, RANDOM_SEED


def fit_predict(data: pd.DataFrame) -> dict[str, Any]:
    """只以所有已完成期 fit，對下一期產生固定模型機率。"""
    model_config = FROZEN_CONFIG["model"]
    features = build_feature_dataset(
        data, min_history=model_config["minimum_history"],
        ewma_alpha=model_config["ewma_alpha"],
        transition_alpha=model_config["transition_alpha"],
        recent_windows=(20, 50, 100),
    )
    model = make_model(
        c=model_config["C"], seed=RANDOM_SEED,
        feature_columns=BASE_FEATURE_COLUMNS, include_number=True,
    )
    model.fit(features, features["target"])
    next_features = build_next_features(
        data, ewma_alpha=model_config["ewma_alpha"],
        transition_alpha=model_config["transition_alpha"],
        recent_windows=(20, 50, 100),
    ).sort_values("number")
    raw = model.predict_proba(next_features)[:, 1]
    probabilities = normalize_and_shrink_probabilities(raw, model_config["shrink_strength"])
    return {
        "model_version": MODEL_VERSION,
        "probabilities": probabilities,
        "top6": stable_top_k(probabilities, 6),
        "top10": stable_top_k(probabilities, 10),
        "top12": stable_top_k(probabilities, 12),
        "training_draws": len(data),
    }


def baselines(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    matrix = indicator_matrix(data)
    uniform = np.full(49, UNIFORM_PROBABILITY)
    recent = matrix[-min(50, len(matrix)):].mean(axis=0)
    previous = matrix[-1].astype(float)
    return {
        "ExactUniform": {"probabilities": uniform, "top6": list(range(1, 7))},
        "Recent50Hot": {
            "probabilities": normalize_and_shrink_probabilities(recent, 0.10),
            "top6": stable_top_k(recent, 6),
        },
        "PreviousDraw": {
            "probabilities": normalize_and_shrink_probabilities(previous, 0.10),
            "top6": sorted((np.flatnonzero(previous) + 1).tolist()),
        },
    }


def assert_pre_target_data(data: pd.DataFrame, draw_id: str, draw_date: str) -> None:
    target_date = pd.Timestamp(draw_date)
    if (pd.to_datetime(data["draw_date"]) >= target_date).any():
        raise ValueError("模型資料包含目標日期或更晚資料")
    if draw_id in set(data["draw_id"].astype(str)):
        raise ValueError("模型資料已包含目標期別")
    if data.empty or len(data) <= FROZEN_CONFIG["model"]["minimum_history"]:
        raise ValueError("訓練資料不足")
