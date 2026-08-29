"""Canonical task constants. Never infer label order from workbook contents."""

RISK_LABELS = ("Indicator", "Ideation", "Behavior", "Attempt")
RISK_TO_ID = {label: index for index, label in enumerate(RISK_LABELS)}
ID_TO_RISK = dict(enumerate(RISK_LABELS))

REQUIRED_TRAIN_COLUMNS = (
    "row_id",
    "anon_user_id",
    "post_id",
    "post",
    "suicide risk",
    "evidence for suicide risk level",
)
REQUIRED_TEST_COLUMNS = ("row_id", "anon_user_id", "post_id", "post")

EMPTY_EVIDENCE_MARKERS = frozenset(("", "none", "nan", "null", "n/a"))

