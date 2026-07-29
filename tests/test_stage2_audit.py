"""第二階段新增的獨立驗證測試。"""
from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from next_draw_predictor_audited import (
    NUMBER_COLUMNS, build_feature_dataset, build_next_features,
    normalize_and_shrink_probabilities, stable_top_k,
)
from audit_stage2.finalize_stage2 import MANIFEST_EXCLUSIONS, validate_zip_against_manifest
from audit_stage2.pipeline import (CANDIDATE_NAMES, INNER_START, OUTER_DRAWS, OUTER_START, SELECTION_RULE,
    baseline_probabilities, batch_expanding, build_feature_dataset_fast, fair_matrix, metric_arrays,
    online_sgd, select_inner_model, simulate_simplified_pipeline, synthetic_dataframe)


def small_data(periods: int = 45) -> pd.DataFrame:
    matrix = np.zeros((periods, 49), dtype=np.int8)
    for index in range(periods):
        chosen = (np.arange(6) * 7 + index) % 49
        matrix[index, chosen] = 1
    return synthetic_dataframe(matrix)


def test_stage2_01_batch_expanding_refits_every_period():
    data = small_data(); features = build_feature_dataset_fast(data, 5)
    _, _, metadata, _ = batch_expanding(data, features, start=40, c=.1)
    assert len(metadata) == 5 and metadata["fit_performed"].all()


def test_stage2_02_batch_and_online_names_are_distinct():
    data = small_data(); features = build_feature_dataset_fast(data, 5)
    batch = batch_expanding(data, features, start=43, c=.1)[0]
    online = online_sgd(data, features, start=43, c=.1)[0]
    assert set(batch["model"]) == {"FullFeatureBatchLogistic"}
    assert set(online["model"]) == {"OnlineSGDLogistic"}


def test_stage2_03_saved_models_use_same_outer_targets():
    batch = pd.read_csv("audit_stage2/batch_walk_forward_predictions.csv")
    online = pd.read_csv("audit_stage2/online_walk_forward_predictions.csv")
    assert batch["target_index"].tolist() == online["target_index"].tolist() == list(range(OUTER_START, OUTER_START + OUTER_DRAWS))


def test_stage2_04_fast_null_features_equal_real_feature_builder():
    data = small_data()
    slow = build_feature_dataset(data, 5)
    fast = build_feature_dataset_fast(data, 5)
    numeric = ["number", "long_z", "recent20_z", "recent50_z", "recent100_z", "ewma_z", "gap_z",
               "transition_z", "in_last_draw", "long_rate", "ewma_rate", "gap", "transition_rate", "target"]
    np.testing.assert_allclose(slow[numeric], fast[numeric], atol=0, rtol=0)


def test_stage2_05_null_and_real_share_selection_rule():
    config = json.load(open("audit_stage2/same_pipeline_null_config.json", encoding="utf-8"))
    actual = json.load(open("audit_stage2/selection_rule.json", encoding="utf-8"))
    assert config["selection_rule"] == actual["rule"] == SELECTION_RULE
    assert tuple(config["candidate_models"]) == CANDIDATE_NAMES


def test_stage2_06_fair_seed_is_reproducible():
    np.testing.assert_array_equal(fair_matrix(123, 50), fair_matrix(123, 50))


def test_stage2_07_same_seed_complete_simplified_result_is_identical():
    assert simulate_simplified_pipeline(1, 991) == simulate_simplified_pipeline(1, 991)


def test_stage2_08_recent50_uses_only_previous_50_draws():
    data = small_data(70); matrix = np.zeros((70,49), dtype=np.int8)
    values = data[NUMBER_COLUMNS].to_numpy(int)-1; matrix[np.arange(70)[:,None], values]=1
    features = build_feature_dataset_fast(data, 5); target=np.array([69])
    first = baseline_probabilities("Recent50HotBaseline", features, matrix, target)
    changed = matrix.copy(); changed[:19] = np.roll(changed[:19], 1, axis=1)
    changed_data = synthetic_dataframe(changed); changed_features = build_feature_dataset_fast(changed_data, 5)
    second = baseline_probabilities("Recent50HotBaseline", changed_features, changed, target)
    np.testing.assert_allclose(first, second)


def test_stage2_09_tie_sorting_is_fixed_and_reproducible():
    scores = np.ones(49)
    assert stable_top_k(scores, 6) == stable_top_k(scores, 6) == [1,2,3,4,5,6]


def test_stage2_10_probability_sum_is_six():
    p = normalize_and_shrink_probabilities(np.linspace(-3, 8, 49), .1)
    assert p.sum() == pytest.approx(6)


def test_stage2_11_probabilities_are_bounded():
    p = normalize_and_shrink_probabilities(np.r_[1e9, np.full(48,-1e9)], 0)
    assert np.all((0 <= p) & (p <= 1))


def test_stage2_12_losses_are_grouped_by_draw():
    probability=np.full((2,49),6/49); actual=np.zeros((2,49),dtype=int); actual[0,:6]=1; actual[1,6:12]=1
    metrics=metric_arrays(probability,actual)
    assert metrics["brier"].shape == (2,) and metrics["log_loss"].shape == (2,)
    assert metrics["brier"][0] == pytest.approx((6/49)*(1-6/49))


def test_stage2_13_outer_targets_do_not_participate_in_selection():
    config=json.load(open("audit_stage2/selection_rule.json",encoding="utf-8"))
    assert config["inner_targets"][1] < config["outer_targets"][0]


def test_stage2_14_inner_validation_does_not_cross_outer_date():
    inner=pd.read_csv("audit_stage2/inner_model_selection.csv")
    assert not inner.empty and INNER_START < OUTER_START and OUTER_START-INNER_START == 100


def test_stage2_15_bootstrap_inputs_have_no_future_target():
    data=small_data(); features=build_feature_dataset_fast(data,5); next_features=build_next_features(data)
    assert features["target_index"].max() == len(data)-1
    assert next_features["history_draws"].eq(len(data)).all()


def test_stage2_16_zip_manifest_validation(tmp_path):
    payload=tmp_path/"payload.txt"; payload.write_text("verified",encoding="utf-8")
    manifest=[{"relative_path":"payload.txt","bytes":payload.stat().st_size,"sha256":hashlib.sha256(payload.read_bytes()).hexdigest()}]
    manifest_path=tmp_path/"file_manifest.json"; manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
    for name in MANIFEST_EXCLUSIONS:
        if name != "file_manifest.json": (tmp_path/name).write_text("metadata",encoding="utf-8")
    archive_path=tmp_path/"bundle.zip"
    with zipfile.ZipFile(archive_path,"w") as archive:
        archive.write(payload,"payload.txt")
        for name in MANIFEST_EXCLUSIONS: archive.write(tmp_path/name,name)
    assert validate_zip_against_manifest(archive_path,manifest_path)["validation_passed"]


def test_stage2_17_same_pipeline_results_have_unique_complete_seeds():
    frame=pd.read_parquet("audit_stage2/same_pipeline_null_results.parquet")
    assert len(frame)==frame.simulation_id.nunique()==frame.seed.nunique()==1000
    assert set(frame.simulation_id)==set(range(1,1001))


def test_stage2_18_all_same_pipeline_outer_counts_match():
    frame=pd.read_parquet("audit_stage2/same_pipeline_null_results.parquet")
    assert frame.outer_prediction_draws.eq(OUTER_DRAWS).all()
