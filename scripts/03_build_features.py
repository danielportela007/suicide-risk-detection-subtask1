#!/usr/bin/env python
"""Build text-free MPNet and risk-anchor feature archives."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys

from suicide_risk_detection.anchors import load_risk_anchors
from suicide_risk_detection.config import (
    load_config,
    resolve_path,
    seed_everything,
    sha256_file,
    write_json,
)
from suicide_risk_detection.data import load_workbook
from suicide_risk_detection.embedding import MPNetEncoder
from suicide_risk_detection.features import build_feature_bundle, save_feature_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subtask1.yaml")
    parser.add_argument("--dataset", choices=("train", "test", "both"), default="both")
    args = parser.parse_args()
    config = load_config(args.config)
    project, embedding, feature_config = (
        config["project"],
        config["embedding"],
        config["features"],
    )
    seed_everything(int(project["seed"]))
    encoder = MPNetEncoder(**embedding)
    anchors_dir = resolve_path(project["anchors_dir"])
    anchors = load_risk_anchors(anchors_dir, feature_config["prompting_strategies"])
    output_dir = resolve_path(project["output_dir"])
    selected = ("train", "test") if args.dataset == "both" else (args.dataset,)
    source_hashes = {}
    for dataset in selected:
        source = resolve_path(project[f"{dataset}_path"])
        frame = load_workbook(source, labeled=dataset == "train")
        bundle = build_feature_bundle(
            frame,
            encoder=encoder,
            anchor_sets=anchors,
            views=feature_config["unit_views"],
            statistics=feature_config["statistics"],
            segment_max_tokens=int(feature_config["segment_max_tokens"]),
            include_post_embedding=bool(feature_config["include_post_embedding"]),
            unit_source_columns=feature_config.get("unit_source_columns"),
        )
        destination = output_dir / f"{dataset}_features.npz"
        save_feature_bundle(destination, bundle)
        source_hashes[dataset] = sha256_file(source)
        print(
            f"Saved {dataset} features ({len(frame)} rows, {len(bundle.blocks)} blocks): {destination}"
        )

    manifest = {
        "model_name": embedding["model_name"],
        "requested_revision": embedding.get("revision"),
        "resolved_revision": encoder.resolved_revision,
        "max_seq_length": embedding["max_seq_length"],
        "embedding_dimension": encoder.dimension,
        "strategies": list(anchors),
        "anchor_versions": {
            key: {
                "prompt_version": value.prompt_version,
                "curation_version": value.curation_version,
                "source_files": value.source_files,
            }
            for key, value in anchors.items()
        },
        "source_sha256": source_hashes,
        "anchor_sha256": {
            path.name: sha256_file(path)
            for path in sorted(anchors_dir.glob("*__risk_levels.json"))
            if path.name.split("__", 1)[0] in anchors
        },
        "seed": project["seed"],
        "configuration": config,
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "sentence-transformers",
                "torch",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(output_dir / "feature_manifest.json", manifest)


if __name__ == "__main__":
    main()
