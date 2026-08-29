import pandas as pd

from suicide_risk_detection.features import _flatten_units


def test_flatten_units_consumes_configured_serialized_column():
    frame = pd.DataFrame(
        {
            "row_id": ["1", "2"],
            "post": ["First. Second.", "Only one."],
            "post_frases": ["['First.', 'Second.']", "['Only one.']"],
        }
    )
    flat, locations = _flatten_units(frame, "sentence", 40, {"sentence": "post_frases"})
    assert flat == ["First.", "Second.", "Only one."]
    assert flat[locations[0]] == ["First.", "Second."]
    assert flat[locations[1]] == ["Only one."]
