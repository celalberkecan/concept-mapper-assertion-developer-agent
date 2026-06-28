"""Tests for evaluator.py — no API calls needed."""

from concept_mapper.evaluator import compute_summary, evaluate_batch, evaluate_single

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GOLD_CI = {
    "concept_id": "C001",
    "input_topic_parent_concept": "age",
    "concept_level_ci_cp_gold": "CI",
    "concept_level_indicator_model_gold": "NA",
    "gold_indicators_conceptual": "",
    "indicator_count_gold": 0,
}

_GOLD_CP = {
    "concept_id": "C003",
    "input_topic_parent_concept": "fear of crime",
    "concept_level_ci_cp_gold": "CP",
    "concept_level_indicator_model_gold": "formative",
    "gold_indicators_conceptual": "fear of burglary; fear of assault; fear of theft",
    "indicator_count_gold": 3,
}

_PRED_CI_CORRECT = {
    "concept_id": "C001",
    "input_topic": "age",
    "ci_or_cp": "CI",
    "indicator_model": "NA",
    "indicators": [],
}

_PRED_CI_WRONG = {
    "concept_id": "C001",
    "input_topic": "age",
    "ci_or_cp": "CP",          # wrong
    "indicator_model": "formative",
    "indicators": [
        {"name": "x", "definition": "y", "role": "component"},
        {"name": "a", "definition": "b", "role": "component"},
    ],
}

_PRED_CP_CORRECT = {
    "concept_id": "C003",
    "input_topic": "fear of crime",
    "ci_or_cp": "CP",
    "indicator_model": "formative",
    "indicators": [
        {"name": "fear of burglary", "definition": "...", "role": "component"},
        {"name": "fear of assault", "definition": "...", "role": "component"},
        {"name": "fear of theft", "definition": "...", "role": "component"},
    ],
}

_PRED_CP_WRONG_MODEL = {
    "concept_id": "C003",
    "input_topic": "fear of crime",
    "ci_or_cp": "CP",
    "indicator_model": "reflective",  # wrong model
    "indicators": [
        {"name": "fear of burglary", "definition": "...", "role": "manifestation"},
        {"name": "fear of assault", "definition": "...", "role": "manifestation"},
    ],
}


# ---------------------------------------------------------------------------
# evaluate_single — CI rows
# ---------------------------------------------------------------------------


def test_ci_correct_prediction():
    result = evaluate_single(_GOLD_CI, _PRED_CI_CORRECT)
    assert result["ci_cp_correct"] is True
    assert result["indicator_model_correct"] is None   # not applicable for CI
    assert result["indicator_count_abs_diff"] is None  # not applicable for CI
    assert result["pred_indicator_count"] == 0


def test_ci_wrong_prediction():
    result = evaluate_single(_GOLD_CI, _PRED_CI_WRONG)
    assert result["ci_cp_correct"] is False
    assert result["indicator_model_correct"] is None   # still None — gold is CI
    assert result["indicator_count_abs_diff"] is None


# ---------------------------------------------------------------------------
# evaluate_single — CP rows
# ---------------------------------------------------------------------------


def test_cp_fully_correct():
    result = evaluate_single(_GOLD_CP, _PRED_CP_CORRECT)
    assert result["ci_cp_correct"] is True
    assert result["indicator_model_correct"] is True
    assert result["gold_indicator_count"] == 3
    assert result["pred_indicator_count"] == 3
    assert result["indicator_count_abs_diff"] == 0


def test_cp_wrong_indicator_model():
    result = evaluate_single(_GOLD_CP, _PRED_CP_WRONG_MODEL)
    assert result["ci_cp_correct"] is True      # CP vs CP is still correct
    assert result["indicator_model_correct"] is False  # formative vs reflective
    assert result["indicator_count_abs_diff"] == 1  # gold=3, pred=2


def test_cp_indicator_count_diff():
    pred = {**_PRED_CP_CORRECT, "indicators": _PRED_CP_CORRECT["indicators"][:2]}  # only 2
    result = evaluate_single(_GOLD_CP, pred)
    assert result["pred_indicator_count"] == 2
    assert result["gold_indicator_count"] == 3
    assert result["indicator_count_abs_diff"] == 1


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------


def test_evaluate_batch_matches_on_concept_id():
    results = evaluate_batch(
        [_GOLD_CI, _GOLD_CP],
        [_PRED_CI_CORRECT, _PRED_CP_CORRECT],
    )
    assert len(results) == 2
    assert results[0]["ci_cp_correct"] is True
    assert results[1]["ci_cp_correct"] is True


def test_evaluate_batch_missing_prediction():
    results = evaluate_batch([_GOLD_CI], predictions=[])  # no predictions at all
    assert len(results) == 1
    assert results[0]["pred_ci_cp"] is None
    assert results[0]["ci_cp_correct"] is None


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


def test_summary_all_correct():
    records = evaluate_batch(
        [_GOLD_CI, _GOLD_CP],
        [_PRED_CI_CORRECT, _PRED_CP_CORRECT],
    )
    summary = compute_summary(records)
    assert summary["ci_cp_accuracy"] == 1.0
    assert summary["indicator_model_accuracy_cp_only"] == 1.0
    assert summary["mean_indicator_count_abs_diff_cp_only"] == 0.0
    assert summary["n_gold_ci"] == 1
    assert summary["n_gold_cp"] == 1


def test_summary_mixed_results():
    records = evaluate_batch(
        [_GOLD_CI, _GOLD_CP],
        [_PRED_CI_WRONG, _PRED_CP_WRONG_MODEL],
    )
    summary = compute_summary(records)
    assert summary["ci_cp_accuracy"] == 0.5   # CI wrong, CP correct
    assert summary["indicator_model_accuracy_cp_only"] == 0.0
