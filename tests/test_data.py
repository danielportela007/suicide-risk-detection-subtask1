import pytest

from suicide_risk_detection.data import canonicalize_risk_label, parse_gold_evidence


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        (" indicator ", "Indicator"),
        ("IDEATION", "Ideation"),
        ("Behavior  ", "Behavior"),
        ("attempt", "Attempt"),
    ],
)
def test_canonicalize_risk_label(raw, canonical):
    assert canonicalize_risk_label(raw) == canonical


def test_parse_gold_evidence_treats_none_as_empty_and_dedicated_spans_as_lists():
    assert parse_gold_evidence("none") == []
    assert parse_gold_evidence("first; second") == ["first", "second"]

