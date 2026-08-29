import numpy as np

from suicide_risk_detection.features import FeatureBundle, load_feature_bundle, save_feature_bundle


def test_feature_archive_uses_pickle_free_string_arrays(tmp_path):
    bundle = FeatureBundle(
        row_ids=np.asarray(["1", "2"], dtype=object),
        user_ids=np.asarray(["u1", "u2"], dtype=object),
        labels=np.asarray([0, 1]),
        blocks={"post_embedding": np.ones((2, 3), dtype=np.float32)},
        feature_names={"post_embedding": ["a", "b", "c"]},
    )
    destination = tmp_path / "features.npz"
    save_feature_bundle(destination, bundle)
    restored = load_feature_bundle(destination)
    assert restored.row_ids.tolist() == ["1", "2"]
    assert restored.user_ids.tolist() == ["u1", "u2"]
