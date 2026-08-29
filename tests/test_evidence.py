import pytest

from suicide_risk_detection.evidence import (
    mean_phrase_f1,
    phrase_counts,
    phrase_f1_per_post,
    phrases_match,
    validate_verbatim,
)


def test_phrase_match_is_case_insensitive_containment_with_length_limit():
    assert phrases_match("Alpha Beta", "I said alpha beta today")
    assert not phrases_match("one two three four five six seven", "three four")


def test_one_to_one_matching_uses_maximum_matching():
    true_positive, predicted_count, gold_count = phrase_counts(
        ["alpha", "alpha beta"], ["alpha beta", "alpha"]
    )
    assert (true_positive, predicted_count, gold_count) == (2, 2, 2)


def test_empty_evidence_pair_scores_one():
    assert phrase_f1_per_post([], []) == 1.0
    assert mean_phrase_f1([[]], [[]]) == 1.0


def test_verbatim_validation_rejects_normalized_text():
    with pytest.raises(ValueError):
        validate_verbatim("Exact original", ["exact original"])

