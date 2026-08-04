# Runbook

How to get from a clean clone to the numbers in the README.

## 0. Environment

The project is managed with [uv](https://docs.astral.sh/uv/). `uv sync`
reconstructs the virtual environment from `uv.lock`; the lock file is the source
of truth, not a `requirements.txt`.

```bash
uv sync
uv run ruff check src tests app     # the verification gate
uv run pytest -q                    # the full suite
```

`.venv/` is never committed and does not travel between machines. Rebuild it on
each one.

## 1. Prove the chain runs, without any data

```bash
uv run python -m fraudq.pipeline --synthetic
```

Takes seconds. It generates transactions with the schema properties the code
depends on, then runs features, temporal split, cross-validation, training,
calibration, capacity-constrained allocation and the policy comparison.

**No number it prints means anything about fraud.** What it proves is that every
stage still connects to the next one. Run it before a long run on real data, and
after any change to the pipeline.

## 2. The data

The dataset is [IEEE-CIS Fraud
Detection](https://www.kaggle.com/c/ieee-fraud-detection). Three prerequisites,
and the third is the one people forget:

1. A Kaggle account.
2. An API token: Kaggle → Settings → API → *Create New Token*. It downloads
   `kaggle.json`, which goes to `~/.kaggle/kaggle.json` with `chmod 600`.
   `KAGGLE_USERNAME` and `KAGGLE_KEY` work instead if you prefer not to keep the
   file on disk.
3. **Accepting the competition rules once, on the web**:
   <https://www.kaggle.com/c/ieee-fraud-detection/rules>. Without this the API
   returns a 403 that does not explain itself.

`unzip` must be installed; the download script uses it.

```bash
./scripts/download_data.sh          # -> data/raw/train_transaction.csv, train_identity.csv
uv run python -m fraudq.data.ingest # -> data/processed/transactions.parquet, identity.parquet
```

The script is idempotent: if the CSVs are already there it does nothing. It
fetches only the two train tables rather than the full ~1.2 GB archive. The
competition test files are ignored on purpose: they have no `isFraud` column, so
every partition here is derived from `train_transaction` by time.

## 3. The pipeline

```bash
uv run python -m fraudq.pipeline
```

Stages, in order, and what each leaves behind:

| Stage | Output |
|---|---|
| Features: UID aggregates and base features | in memory |
| Temporal split: train / embargo / calibration / test | in memory |
| Expanding-window CV, then the final fit | in memory |
| Calibration, chosen on a temporal holdout inside calibration | in memory |
| Scoring of the calibration partition | `reports/scored_calib.parquet` |
| Single-threshold policy tuned on calibration | printed |
| **The single look at the test partition** | `reports/scored_test.parquet` |
| Policy comparison | `reports/policy_comparison.csv` |
| Booster, calibrator, feature list, cost config | `models/artifacts/` |

Runtime and peak memory on the real dataset have not been measured yet. Check
free memory before starting: the training frame is ~590k rows by ~400 columns,
and swapping mid-training is not slow, it is fatal.

## 4. Everything downstream reads the persisted scoring

The sensitivity sweep, the drift report, the results notebook, the API and the
queue simulator all consume `reports/` and `models/artifacts/`. **None of them
re-evaluates the test partition**, which is what keeps the single-look rule
(`docs/design.md` §9, invariant 5) true in practice rather than in principle.

Varying a cost parameter does not re-score anything: predictions are already
made, and only the decision layer changes.

```bash
uv run jupyter lab notebooks/03_results.ipynb   # figures into reports/figures/
uv run streamlit run app/streamlit_app.py       # queue simulator
uv run uvicorn fraudq.api.main:app --port 8000  # scoring endpoint
```

## 5. Container

```bash
docker build -t fraud-review-queue .
docker run --rm -p 8000:8000 -v $PWD/models/artifacts:/models fraud-review-queue
```

Artifacts are mounted, not baked in. The runtime image installs `libgomp1`
because LightGBM fails to import on `slim` without OpenMP, and the error message
never mentions LightGBM.

## 6. If the result is small

The four cost parameters are assumptions, not measurements, and the honest
outcome may be that ranking by value saves little on this dataset. If that
happens, **the parameters do not get adjusted after seeing the result.** The
sensitivity analysis exists to report that fact, and the thesis becomes a
narrower one: under which conditions ranking a review queue by value matters,
and why it does not here.
