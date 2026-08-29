from pathlib import Path

from suicide_risk_detection.anchors import load_risk_anchors
from suicide_risk_detection.constants import RISK_LABELS


def test_canonical_anchor_files_load_all_risk_targets():
    root = Path(__file__).resolve().parents[1]
    anchors = load_risk_anchors(
        root / "data" / "anchors" / "v1.1.0", ["zero_shot", "meta_prompting"]
    )
    assert set(anchors) == {"zero_shot", "meta_prompting"}
    for anchor_set in anchors.values():
        assert tuple(anchor_set.phrases) == RISK_LABELS
        assert all(len(anchor_set.phrases[label]) == 15 for label in RISK_LABELS)

