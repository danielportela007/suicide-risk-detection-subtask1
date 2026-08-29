import numpy as np

from suicide_risk_detection.modeling import repeated_stratified_group_splits


def test_group_splits_never_split_a_user():
    groups = np.repeat([f"u{index}" for index in range(12)], 4)
    y = np.tile(np.arange(4), 12)
    splits = repeated_stratified_group_splits(
        y, groups, n_splits=3, n_repeats=2, random_state=2026
    )
    for _, _, train, validation in splits:
        assert not (set(groups[train]) & set(groups[validation]))
