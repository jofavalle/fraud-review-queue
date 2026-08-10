# Runbook

How to get from a clean clone to the numbers in the README.

## 0. Environment

The project is managed with [uv](https://docs.astral.sh/uv/). `uv sync`
reconstructs the virtual environment from `uv.lock`; the lock file is the source
of truth, not a `requirements.txt`.

```bash
uv sync
uv run ruff check src tests app            # lint, the verification gate
uv run ruff format --check src tests app   # and formatting, same paths
uv run pytest -q                           # the full suite
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
2. An API token, from <https://www.kaggle.com/settings/api>. The site hands you
   a bare token and tells you to save it at `~/.kaggle/access_token` with
   `chmod 600`; that is the first place the CLI looks. The older
   `~/.kaggle/kaggle.json` still works, and so do the `KAGGLE_API_TOKEN` and
   `KAGGLE_USERNAME` plus `KAGGLE_KEY` environment variables if you would
   rather keep nothing on disk.
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

Measured on 4 cores with 7.5 GiB of RAM: **14.6 minutes wall clock and a peak
resident set of 6.56 GiB**, on a training frame of 410,601 rows by 412 features.
The ingest step before it takes 41 seconds and peaks at 1.8 GiB.

That peak is the number to plan around. On a 7.5 GiB machine it fits only
because the kernel pushes about 3 GiB into swap during the feature stage, and it
recovers; on anything smaller, swapping mid-training is not slow, it is fatal.
Check free memory before starting.

## 4. Everything downstream reads the persisted scoring

The sensitivity sweep, the drift report, the results notebook, the API and the
queue simulator all consume `reports/` and `models/artifacts/`. **None of them
re-evaluates the test partition**, which is what keeps the single-look rule
(`docs/design.md` §9, invariant 5) true in practice rather than in principle.

Varying a cost parameter does not re-score anything: predictions are already
made, and only the decision layer changes. That is why the sweep takes 20
seconds against the pipeline's 15 minutes.

```bash
uv run python -m fraudq.analysis                # sweep and drift into reports/
uv run python -m fraudq.diagnostics --learning-curve   # model diagnostics
uv run jupyter lab notebooks/03_results.ipynb   # figures into reports/figures/
uv run streamlit run app/streamlit_app.py       # queue simulator
uv run uvicorn fraudq.api.main:app --port 8000  # scoring endpoint
```

`fraudq.analysis` writes `sensitivity_tornado.csv`, `drift_by_week.csv` and
`analysis_summary.json`, and the notebook reads all three. Run it before the
notebook: three of the twelve figures have nothing to draw otherwise.

### `fraudq.diagnostics` is the one exception to "reads the persisted scoring"

It reports on the model rather than on the decision layer, and two of its
outputs cannot come from `reports/`: PSI and the correlation matrix need
**feature values**, and the persisted scoring carries identifiers, amount, label
and probability only. So it rebuilds the 412 features from `data/processed/`,
which is what it costs.

```bash
uv run python -m fraudq.diagnostics --cheap-only        # ~10 s, artefacts only
uv run python -m fraudq.diagnostics                     # + feature rebuild, ~3 min
uv run python -m fraudq.diagnostics --learning-curve    # + retrained folds, ~20 min
```

Measured on the reference machine: the full run with the learning curve took
**16 minutes with a peak of 6.66 GiB of RSS**, of which the feature rebuild was
about two minutes and the retrained folds were the rest. The peak sits alongside
the pipeline's own, so the same memory caveat applies. The learning curve is the
long pole because recording PR-AUC on the fit window as well as the validation
window roughly doubles the cost of every boosting round.

**It regenerates no model.** The booster and the calibrator are loaded, never
fitted, and every output is a new file. Two guards abort the run rather than
report a number that would be quietly wrong:

- The features it rebuilds must re-score the test partition to **exactly** the
  values in `scored_test.parquet`. A mismatch means the feature pipeline is not
  deterministic or the artefacts are stale.
- The folds it retrains must choose the number of trees the persisted booster
  already has. That is what entitles the learning curve to be presented as the
  overtraining of the published model.

Run it in `tmux` with an end marker, like the pipeline.

One subtlety worth knowing before reading the code. `run_pipeline` fits the
single-threshold policy on calib and prints the threshold without persisting it,
so `fraudq.analysis` refits it from `scored_calib.parquet`. The grid is fixed and
the tie-break is deterministic, so this reproduces the pipeline's value exactly
rather than approximating it, and nothing has to be re-run to recover it.

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
