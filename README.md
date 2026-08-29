# Explainable Suicide Risk Detection — Subtask 1

Reproducible implementation of `PostFrasesV1` for the IEEE Big Data Cup 2026 suicide-risk detection task. Given one Reddit post, the system predicts `Indicator`, `Ideation`, `Behavior`, or `Attempt` and extracts verbatim evidence spans.

This repository covers Subtask 1 only. It does not model risk or protective factors.

## Results

Repeated out-of-fold validation was stratified and grouped by `anon_user_id` using five folds and three repeats.

| Metric | OOF result |
|---|---:|
| Risk Weighted F1 | **0.625747** |
| Risk Macro F1 | 0.551130 |
| Accuracy | 0.621407 |
| Evidence Phrase F1 | **0.493796** |
| Normalized Subtask 1 score | **0.569196** |

These are exploratory OOF results, not held-out test or leaderboard scores. The local Phrase F1 assigns 1.0 when prediction and gold evidence are both empty; this convention must be checked against the official scorer.

## Method

The risk classifier combines a 768-dimensional `all-mpnet-base-v2` post embedding with four 32-dimensional blocks derived from sentence-to-anchor similarities. Two synthetic-anchor strategies (`zero_shot` and `meta_prompting`) and two representations (macro and individual) produce 896 input variables. The selected model applies standardization, fold-local ANOVA selection of 128 variables, and class-balanced logistic regression with `C=0.25`.

Evidence candidates are generated from the untouched post with character offsets. Candidates are ranked by their MPNet similarity to the two anchor families for the predicted class. The selected policy emits at most two spans, requires similarity of at least 0.40, and limits each span to 12 tokens. Every emitted span is checked against the original post.

The complete methodology, metric definitions, experiments, results and limitations are in [DOCUMENTACION_COMPLETA_SUBTASK1.md](DOCUMENTACION_COMPLETA_SUBTASK1.md).

A detailed interpretation of cross-validation stability, configuration comparisons, per-class errors, the confusion matrix, evidence tuning and the exact train/test protocol is available in [ANALISIS_COMPLETO_RESULTADOS.md](ANALISIS_COMPLETO_RESULTADOS.md).

## Repository contents

- `configs/`: complete experiment configuration;
- `data/anchors/`: the 120 synthetic risk-level phrases used by the system;
- `prompts/`: versioned generation prompts and target definitions;
- `scripts/`: validation, feature construction, training, evidence and prediction stages;
- `src/`: reusable implementation;
- `tests/`: unit tests for data, groups, features, spans and metrics;
- `results/`: text-free numerical reports and experiment tables.

## Restricted data

Competition workbooks, text-bearing OOF predictions, the submission CSV, feature archives and trained models are intentionally excluded. To reproduce the experiment, obtain the data through the competition and place these files under `data/raw/`:

```text
train dataset-con-frases-con-segmentosV1.xlsx
test dataset-con-frases.xlsx
```

Their expected SHA-256 hashes are recorded in `results/data_validation.json`. Do not commit, publish or transfer the Reddit text.

## Reproduction

Python 3.11–3.13 is supported. The recorded run used Python 3.12.11 on CPU.

```bash
python -m pip install -e '.[dev]'
python scripts/01_validate_data.py
python scripts/02_prepare_user_splits.py
python scripts/03_build_features.py
python scripts/04_train_risk_models.py
python scripts/05_tune_evidence.py
python scripts/06_predict_submission.py --team-name PostFrasesV1
python scripts/07_evaluate.py
python -m pytest
ruff check .
```

The MPNet revision is pinned in `configs/subtask1.yaml`. Exact dependencies are declared in `pyproject.toml`.

## Intended use

This is a research system for competition evaluation. It is not a clinical diagnosis, professional assessment or autonomous intervention system.
