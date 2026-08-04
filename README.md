# Fraud Review Queue

**Cost-optimal allocation of a capacity-constrained fraud review queue.**

> **Status: in progress (July 2026).** The design is fixed and the specification
> is complete; results are being produced. Full spec in `docs/design.md`.

---

## The problem

A payment processor cannot manually inspect every transaction, and cannot
automate every decision either. Real fraud operations run a **three-zone
policy**: block automatically above a high score, approve automatically below a
low one, and send the ambiguous middle to a **manual review queue** — a queue
with **finite analyst capacity**.

That makes this an **allocation problem**, not a classification problem: how do
you spend a scarce human resource under uncertainty?

Because the cost of approving fraud scales with the transaction amount, and so
does the cost of blocking a legitimate customer, **the optimal thresholds are
not constants — they depend on the amount.** A single-threshold system is
structurally the wrong policy.

And once capacity binds, the correct quantity to rank the queue by is not the
fraud score but the **value of review**:

```
V = min( p·(a + F),  (1 − p)·(m·a + φ) ) − r
```

— the expected-cost reduction from sending a transaction to a human rather than
deciding it automatically.

`V` peaks at **moderate** probabilities and grows with the amount. Ranking by
score concentrates analysts where the system is already confident, and where a
human therefore adds little. **This project measures what that costs.**

Derivations, assumptions, and project invariants: `docs/design.md`.

---

## The data

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
(Vesta Corporation). ~590k e-commerce transactions, ~3.5 % fraud rate, ~6 months,
~430 features.

**Note:** this is a competition dataset, and the public test labels do not exist.
The competition test files are ignored. All partitions here are derived from
`train_transaction` **by time**.

---

## Key decisions

**Temporal split with a label-availability embargo.**
Fraud labels arrive weeks late via chargebacks. A contiguous train/test split
implicitly assumes instantaneous label availability. A 10-day embargo between
train and calibration simulates production reality.

**No SMOTE, no class weighting.**
Resampling distorts predicted probabilities, and the entire cost layer depends
on `p` being a real probability. Class imbalance is addressed by choosing the
right *metric*, not by altering the training distribution. Reliability curves for
the uncalibrated model, Platt scaling, and isotonic regression are reported side
by side.

**The UID: entity resolution, not leakage.**
A customer identifier can be reconstructed from `card1`, `addr1` and a
normalisation of `D1`. In production, such an identifier exists — recovering it
is legitimate. What *would* be leakage is aggregating the target over it. Every
UID aggregate here is strictly backward-looking.

**A leakage test, written before the features.**
`tests/test_no_future_leakage.py` asserts that features computed over the full
history are identical to features computed over the truncated history, for every
row inside the truncation. Honest features cannot tell the difference.

---

## Results

*Held-out test partition (days 156+). Evaluated once. In progress.*

| Policy | Expected loss / $1k volume | Fraud caught | Fraud missed | Legit blocked |
|---|---|---|---|---|
| Approve everything | — | — | — | — |
| Single score threshold | — | — | — | — |
| Review top-K by score | — | — | — | — |
| **Review top-K by value** | — | — | — | — |

### Model

| Metric | Value |
|---|---|
| PR-AUC | — |
| ROC-AUC | — |
| Brier score (calibrated) | — |
| ECE (calibrated) | — |

---

## Sensitivity

The four cost parameters (`F`, `m`, `φ`, `r`) are **assumptions, not
measurements**. Every conclusion is tested across their plausible ranges, and
reported as a tornado plot.

The analysis answers two questions: whether the conclusion survives the full
range, and which parameter dominates — i.e. what the business should measure
next, rather than what to tune in the model.

*In progress.*

---

## Limitations

- Analyst review is assumed perfect. Real analysts have a detection rate below 1.
- Cost parameters are assumed. Mitigated by the sensitivity analysis, not removed.
- No feedback loop: blocking a transaction means never learning whether it was
  fraud. A deployed system faces selective labelling; this offline evaluation
  does not model it.
- Static capacity. Real queues have shifts, backlogs, and SLA prioritisation.
- The data is from 2017–2018. Fraud patterns have moved.

---

## Run it

```bash
uv sync

# The whole chain on synthetic data: no Kaggle account, no download, seconds.
# It proves the pipeline runs; it says nothing about fraud.
uv run python -m fraudq.pipeline --synthetic
```

With the real data:

```bash
# Requires a Kaggle account, and the competition rules accepted once on the web:
# https://www.kaggle.com/c/ieee-fraud-detection/rules
./scripts/download_data.sh              # -> data/raw/train_*.csv
uv run python -m fraudq.data.ingest     # -> data/processed/*.parquet

uv run python -m fraudq.pipeline        # features -> split -> train -> calibrate -> evaluate
uv run streamlit run app/streamlit_app.py
```

The pipeline writes `reports/scored_calib.parquet`, `reports/scored_test.parquet`,
`reports/policy_comparison.csv` and `models/artifacts/`. Everything downstream —
the sensitivity sweep, the drift report, the API and the queue simulator — reads
those, so **the test partition is scored once and never looked at again**.

The scoring API ships as a container. Model artifacts are mounted, not baked in:
the image is the code, the model is data with its own lifecycle.

```bash
docker build -t fraud-review-queue .
docker run --rm -p 8000:8000 -v $PWD/models/artifacts:/models fraud-review-queue
```

The **queue simulator** lets you move analyst capacity and each cost assumption
and watch the expected loss — and the gap against score-ranked triage — move with
them.

---

## Repository structure

```
src/fraudq/
├── config.py           # every cost and policy assumption, in one place
├── pipeline.py         # the driver: chains the stages below end to end
├── data/               # ingest (CSV → parquet), temporal split, synthetic data
├── features/           # UID construction, backward-looking aggregates
├── models/             # training, probability calibration, artifact persistence
├── policy/             # cost functions, value of review, capacity allocation
├── evaluate/           # metrics, policy comparison, sensitivity, drift
└── api/                # FastAPI scoring endpoint

tests/                  # including the temporal leakage guard
docs/design.md          # cost model, derivations, project invariants
app/                    # queue simulator
scripts/                # Kaggle download
```

---

## License

MIT
