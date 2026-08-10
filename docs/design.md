# Design: Fraud Review Queue

**Status:** in progress
**Last updated:** July 2026

This document specifies the system. It is written ahead of the results, and the
sections describing outputs describe what the project *will* report, not what it
has already shown.

---

## 1. Problem statement

A payment processor cannot manually inspect every transaction, and cannot
automate every decision either. Real fraud operations run a **three-zone
policy**:

| Zone | Action |
|---|---|
| High suspicion | Block automatically |
| Low suspicion | Approve automatically |
| Ambiguous | Send to a **manual review queue** |

The review queue has **finite capacity**. There are N analysts, each resolving
M cases per shift. The system cannot review everything it would like to review.

This turns a classification problem into an **allocation problem**: how do you
spend a scarce human resource under uncertainty?

### The thesis

> The transactions worth reviewing are not the highest-scoring ones.

The highest-scoring transactions are already blocked with confidence, so a human
adds no information. What is worth an analyst's time is the **genuinely
ambiguous and high-value** transaction. Ranking the review queue by fraud score,
which is the naive implementation, sends analysts to the wrong place.

This project quantifies how much money that costs.

---

## 2. Decision model

For a transaction with **calibrated** fraud probability `p` and amount `a`:

| Action | Expected cost |
|---|---|
| **Approve** | `p · (a + F)` |
| **Block** | `(1 − p) · (m · a + φ)` |
| **Review** | `r` |

**Approve:** if the transaction is fraudulent (probability `p`), the issuer
reverses the charge. The merchant loses the full amount plus a fixed chargeback
fee `F`. If it is legitimate, cost is zero.

**Block:** if the transaction was legitimate (probability `1 − p`), the merchant
loses the gross margin `m · a` on that sale, plus a friction cost `φ` (support
contact, churn risk). If it was fraud, cost is zero: the loss was avoided.

**Review:** the analyst's time. Review is assumed to resolve the case correctly.
See [Limitations](#8-limitations).

### 2.1 Parameters

These are **business assumptions, not measurements**. They live in
`src/fraudq/config.py`, and every conclusion in this project is subjected to a
sensitivity sweep over them (§7.2).

| Parameter | Base value | Rationale |
|---|---|---|
| `F`, the chargeback fee | $20 | Card networks charge merchants a fixed fee per chargeback, typically $15-25. |
| `m`, the gross margin | 0.25 | Typical e-commerce margin. Blocking a legitimate sale costs the *profit*, not the price. |
| `φ`, the friction cost | $10 | Cost of a blocked legitimate customer: support contact, probability of not returning. The softest assumption of the four. |
| `r`, the review cost | $2 | Analyst at ~$25/hr fully loaded, ~5 minutes per case. |
| `K`, the daily capacity | 0.5-2 % of volume | A policy lever, explored as a variable. |

### 2.2 Thresholds depend on the amount

Setting the cost of each automated action equal to the cost of review gives the
boundaries of the review region.

**Approve / Review boundary:**

```
p · (a + F) = r    ⟹    p = r / (a + F)
```

A hyperbola, decreasing in `a`. **Interpretation:** the larger the amount, the
lower the suspicion threshold at which auto-approval stops being acceptable.
With $20 at stake you can afford to be tolerant. With $800 you cannot.

**Block / Review boundary:**

```
(1 − p)(m·a + φ) = r    ⟹    p = 1 − r / (m·a + φ)
```

Approaches 1 as `a` grows. **Interpretation:** the more legitimate revenue is at
risk, the more confidence you need before blocking automatically.

**Consequence: the optimal thresholds are not constants.** A single-threshold
system is structurally the wrong policy.

### 2.3 Capacity is what makes the problem interesting

Evaluating those boundaries at the base parameters, the "worth reviewing" region
spans roughly `p ∈ [0.02, 0.95]` at moderate amounts. Without a capacity
constraint, the cost-optimal policy would review an implausibly large fraction
of traffic.

Since it cannot, the queue must be **ranked**. And the correct quantity to rank
by is not the score. It is the **value of review**:

```
V = min( p·(a + F),  (1 − p)·(m·a + φ) ) − r
```

`V` is the expected-cost reduction from sending this transaction to a human
instead of deciding it automatically.

**Policy under capacity K, applied per day:**

1. Compute `V` for every transaction that day.
2. Sort descending.
3. Send the top-K with `V > 0` to review.
4. Decide the rest with the cheaper automated action.

This is a greedy allocation, and it is optimal: a knapsack with unit weights.

### 2.4 Why score is the wrong ranking

For a fixed amount, `V(p, a)` is maximised where the two cost curves cross:

```
p* = (m·a + φ) / ((a + F) + (m·a + φ))
```

At the base parameters this gives `p* ≈ 0.20-0.30`, **nearly independent of the
amount**:

| Amount | `p*` | `V_max` |
|---|---|---|
| $10 | 0.29 | $6.8 |
| $50 | 0.24 | $15.0 |
| $100 | 0.23 | $25.1 |
| $200 | 0.21 | $45.1 |
| $500 | 0.21 | $105.0 |

Value of review **peaks at moderate probabilities and grows strongly with the
amount.** A "review top-K by score" policy concentrates analysts at `p > 0.9`,
where `V` is low: the system is already confident there, and the automated
block is nearly as good as the human.

At the base parameters the two sets, top-K by score and top-K by value, are
**nearly disjoint**. Quantifying the resulting cost gap on held-out data is the
purpose of this project.

*(These are algebraic consequences of the cost model, not empirical findings.
They hold for any model that produces calibrated probabilities.)*

---

## 3. Data

**IEEE-CIS Fraud Detection** (Kaggle; real e-commerce data from Vesta
Corporation).

- ~590,000 training transactions
- ~3.5 % fraud rate
- ~430 columns across two tables

### 3.1 Schema

`train_transaction.csv`:

| Group | Columns | Notes |
|---|---|---|
| Identity | `TransactionID`, `isFraud` | `isFraud` is the target |
| Time | `TransactionDT` | **Seconds** from an unspecified origin. Spans ~6 months. |
| Amount / product | `TransactionAmt`, `ProductCD` | |
| Card | `card1`-`card6` | Anonymised. `card1` acts as a card proxy. |
| Address | `addr1`, `addr2`, `dist1`, `dist2` | |
| Email | `P_emaildomain`, `R_emaildomain` | Purchaser, recipient |
| Counts | `C1`-`C14` | Counts of associated entities (meaning undisclosed) |
| Timedeltas | `D1`-`D15` | Days since prior events. `D1` is load-bearing (§5.2). |
| Matches | `M1`-`M9` | Boolean match flags |
| Vesta | `V1`-`V339` | Proprietary engineered features. Highly correlated, heavily null. |

`train_identity.csv` joins on `TransactionID`: `id_01`-`id_38`, `DeviceType`,
`DeviceInfo`. **It covers only a fraction of transactions.** Left join, keep the
nulls: LightGBM handles NaN natively, and the *absence* of identity data is
itself a signal.

### 3.2 Operational notes

**The public test labels do not exist.** This is a competition dataset;
`test_transaction.csv` has no `isFraud` column. The competition test files are
**ignored entirely**. All partitions in this project are derived from
`train_transaction` by time.

**`TransactionDT` is not a calendar date.** It is a second offset. All work is
done in relative days:

```python
df["day"]  = df["TransactionDT"] // 86400
df["hour"] = (df["TransactionDT"] // 3600) % 24
```

Community-inferred calendar origins exist, but they are inference rather than
documented fact, and they are not needed here.

### 3.3 Storage

- CSV → **Parquet**, once, on ingest.
- Heavy aggregation and joins run in **DuckDB / SQL**, not pandas.

---

## 4. Splits

Sorted by `TransactionDT`, cut by day:

| Partition | Days | Purpose |
|---|---|---|
| **Train** | 0 to 119 | Model fitting, hyperparameter search |
| **Embargo** | 120 to 129 | **Unused** |
| **Calibration + policy** | 130 to 155 | Fit the calibrator; set thresholds; explore `K` |
| **Test** | 156 to the end | Final evaluation. **Looked at once.** |

### 4.1 Never a random split

A `train_test_split(shuffle=True)` on temporal transaction data leaks the future
into training. The result is an inflated AUC that does not survive production.

### 4.2 The embargo

**Fraud labels arrive late.** A chargeback takes weeks or months to materialise.
When a model is retrained in production, labels for the most recent weeks are
not yet available.

A contiguous train/test split implicitly assumes instantaneous label
availability. That assumption is false. The 10-day embargo simulates the gap.

### 4.3 The calibrator sees held-out data only

Fitting the calibrator on the training set produces systematically overconfident
probabilities, because the model has already seen those labels.

**The entire cost layer depends on `p` being a real probability.** A
mis-calibrated model makes the "optimal" policy no longer optimal. This is why
the calibration partition exists as an entity separate from both train and test.

### 4.4 Cross-validation

Hyperparameter search inside the training window uses an **expanding window**,
not KFold:

```
Fold 1: train 0-39   valid 40-59
Fold 2: train 0-59   valid 60-79
Fold 3: train 0-79   valid 80-99
Fold 4: train 0-99   valid 100-119
```

---

## 5. Features

### 5.1 Base features

- `log1p(TransactionAmt)`
- **Decimal part of the amount** (`TransactionAmt % 1`). Currency-converted or
  programmatically generated amounts leave odd signatures here.
- Hour of day
- Email domains: group rare ones, extract the base provider
- Frequency encoding for high-cardinality categoricals (`card1`, `addr1`),
  **computed on train only**

### 5.2 The UID: entity resolution, not leakage

An approximate customer identifier can be reconstructed:

```python
df["D1n"] = df["day"] - df["D1"]              # constant per card
df["uid"] = df["card1"] + "_" + df["addr1"] + "_" + df["D1n"]
```

`D1` approximates "days since the card was first seen". Subtracting it from the
current day recovers a proxy for the **card registration date**, constant for
that card. Combined with `card1` and `addr1`, it identifies a customer.

**Is this leakage?**

**No.** In a production fraud system, a persistent customer identifier *does*
exist. Reconstructing it from anonymised columns recovers information that would
be available at scoring time. That is entity resolution.

**What *is* leakage is aggregating the target over the UID across the full
dataset.** A `mean(isFraud)` grouped by UID looks at future labels.

### 5.3 The rule

> **Every UID aggregate is strictly backward-looking: `expanding()` with
> `.shift(1)`. No full-column statistics. No target encoding outside folds.**

| Feature | Definition |
|---|---|
| `uid_txn_count_prior` | Prior transactions from this UID |
| `uid_seconds_since_last` | Time since this UID's previous transaction |
| `uid_amt_ratio` | Current amount / rolling mean of prior amounts |
| `uid_amt_zscore` | (Amount − prior mean) / prior std |
| `uid_n_devices_prior` | Distinct devices previously seen for this UID |
| `uid_n_emails_prior` | Distinct emails previously seen |
| `uid_velocity_1h/24h/7d` | Transactions in the preceding window |

Each of these is enforced by `tests/test_no_future_leakage.py`, which asserts
that features computed over the full history are identical to features computed
over the truncated history, for every row inside the truncation. Honest features
cannot tell the difference.

### 5.4 The V-columns

339 columns with correlated blocks and shared null patterns. Approach, in
priority order:

1. Group by **null pattern**: columns sharing a pattern belong to the same
   Vesta block.
2. Within each block, keep one representative per highly correlated group.
3. Or hand them all to LightGBM, which is robust to redundant features, and
   report feature importance.

Archaeology on anonymised columns is not where the value of this project lies.

---

## 6. Modelling

### 6.1 Model

**LightGBM.** A logistic regression on a small feature set serves as an honest
baseline. No model zoo.

### 6.2 No resampling

> **No SMOTE. No undersampling. No `scale_pos_weight`.**

All of them **distort predicted probabilities.** A model trained on rebalanced
data predicts probabilities for the rebalanced distribution, not the real one.
Train on a 50/50 resample of a 3.5 % base rate and every predicted probability
is systematically inflated.

**The cost layer requires `p` to be a real probability.** If `p` is meaningless,
`V` is meaningless, and the decision layer, which is the entire point of this
project, collapses.

Class imbalance is not a problem *per se*. LightGBM with `objective="binary"`
and log loss handles a 3.5 % positive rate without difficulty. What needs
adjusting is not the training procedure but the **evaluation metric** (§6.4).

Reliability curves for the uncalibrated model, Platt scaling, and isotonic
regression are reported side by side in `reports/figures/`.

### 6.3 Calibration

Fitted **only** on the calibration partition (days 130-155).

- **Platt scaling** (sigmoid) and **isotonic regression**, compared.
- Diagnostics: **reliability diagram**, **Brier score**, **ECE**.
- **Calibration is reported per score decile, not only in aggregate.** A model
  can be well calibrated on average and badly calibrated in the top 1 %, which
  is exactly where the expensive decisions are made. The average hides the error
  where it matters.

### 6.4 Metrics

| Metric | Used | Why |
|---|---|---|
| Accuracy | no | With 3.5 % positives, "all legitimate" scores 96.5 %. |
| ROC-AUC | Reported, not leading | Optimistic under heavy imbalance. |
| **PR-AUC** | yes | The correct discrimination metric under imbalance. |
| **Precision@K / Recall@K** | yes | With `K` = review capacity. The operational metric. |
| **Brier / ECE** | yes | Without these, the cost layer is not valid. |
| **Expected loss per $1,000 of volume** | yes, the primary one | The only metric a business reads. |

### 6.5 Diagnostics

The metrics above say how good the model is. These say **what it is made of and
whether the number on test can be believed.** They are computed by
`fraudq.diagnostics`, which loads the persisted booster and never fits one.

| Diagnostic | Question it answers |
|---|---|
| Importance by **gain** and by **split** | Which columns carry the score. Promised in §5.4 and delivered here. |
| **Null-pattern blocks** of the V-columns | Whether the Vesta blocks of §5.4 exist, recovered without looking at a value. |
| **Spearman** rank correlation | How much of that importance is redundant. Not Pearson: the C, D and amount columns have long tails, and Pearson would measure the outliers. |
| **ROC and PR curves**, with operating points | The whole trade-off rather than the scalar AUC, and where a capacity-constrained queue actually sits on it. |
| **Two-sample KS** between partitions | Whether two score distributions are the same. |
| **Learning curve** per CV fold | Overtraining, isolated from drift. |

**Two rules attached to these, because both are easy to get wrong.**

1. **A KS between train and test is not a test for overtraining here.** With a
   random split it would be, and that is where the habit comes from. This split
   is temporal (§4.1), so that difference sums overtraining and genuine drift
   and separates neither. Worse, at n in the hundreds of thousands the p-value
   is ~0 for a difference of no operational size, so the test degenerates into
   asking whether two distributions are literally identical. **Report the
   statistic `D`**, which is a distance and does not grow with n, and read the
   p-value as a statement about the sample size.

   Three comparisons are reported so the reader can separate what the single one
   cannot: calib against test (both unseen, so drift alone), train against test,
   and train against calib. Splitting each by class matters too, since the
   overall distributions can differ merely because the fraud rate moved.

2. **The clean overtraining measurement is the learning curve**, where PR-AUC on
   the fit window and on the validation window are recorded per boosting round
   **within the same expanding-window fold** (§4.4). The two windows are
   adjacent in time, which holds drift roughly constant between them, so the gap
   at the chosen iteration is overtraining and not calendar.

**These diagnostics may not change the model.** They are read after the fact,
off artefacts that already exist. A diagnostic that prompted a hyperparameter
change would have turned test into a tuning set, which invariant 5 forbids.

---

## 7. Evaluation

### 7.1 Policies compared

Evaluated on the test partition. **One look.**

| # | Policy | What it represents |
|---|---|---|
| 1 | Approve everything | Baseline loss |
| 2 | Single score threshold | The naive system |
| 3 | Review top-K **by score** | What most implementations do |
| 4 | Review top-K **by value of review** | This project |

**Primary metric: expected loss per $1,000 of transaction volume.**

**The headline number is the gap between (3) and (4)**: the cost of ranking a
review queue by score rather than by expected value.

Also reported per policy: fraud caught, fraud missed, legitimate transactions
blocked, and capacity utilisation.

### 7.2 Sensitivity analysis

The four cost parameters are **assumptions**, not measurements. Reporting a
dollar figure without an error bar is not acceptable.

1. Define a plausible range per parameter: `F ∈ [10, 40]`, `m ∈ [0.15, 0.40]`,
   `φ ∈ [2, 30]`, `r ∈ [1, 5]`.
2. Vary one at a time, holding the rest at base.
3. **Tornado plot**, ordered by how much each moves the estimated saving.

This answers two questions:

- **How robust is the conclusion?** If the (4)-over-(3) saving survives the full
  parameter range, the finding holds independently of the exact assumptions.
- **Which parameter is worth measuring properly?** Whichever dominates the
  tornado is the one where the business's next dollar of effort belongs, and
  that recommendation is an output of this analysis, not an input to it.

**Capacity is swept separately, and it is a different kind of question.** The
four cost parameters are assumptions about a business; capacity is a fact about
an operation, and a reader will want to substitute their own. `daily_capacity_pct`
is therefore swept across `PolicyConfig.capacity_sweep`, fixed in config like the
cost ranges and for the same reason.

The sweep carries one reference point that is not a swept value: **capacity
zero**, where both queue policies collapse to the same automatic rule. Without it
the other rows have no scale, because a queue is not free. Every review costs `r`,
and reviewing a case the automatic rule had already decided correctly buys
nothing, so a queue can cost **more** than not having one. The capacity at which
that stops being true is an output of this sweep.

### 7.3 Drift

- **PSI** by month on the top features.
- PR-AUC and expected cost computed month by month across the test window.

Informs a retraining cadence.

---

## 8. Limitations

- **Analyst review is assumed perfect.** Real analysts have a detection rate
  below 1 and take variable time. A natural extension.
- **Cost parameters are assumed, not measured.** Mitigated by §7.2, not removed.
- **No feedback loop.** Blocking a transaction means never observing whether it
  would have been fraud. A deployed system faces selective labelling; this
  offline evaluation does not model it.
- **Static capacity.** Real queues have shift patterns, backlogs, and SLA
  prioritisation.
- **The data is from 2017-2018.** Fraud patterns have moved.

---

## 9. Project invariants

**These are non-negotiable. Every contributor, human or agent, follows them.**

1. **Splits are strictly temporal** by `TransactionDT`. Never
   `train_test_split(shuffle=True)`. A 10-day embargo separates train from
   calibration, simulating chargeback label delay.

2. **No SMOTE, no undersampling, no `scale_pos_weight`.** The decision layer
   depends on calibrated probabilities; resampling destroys them.

3. **Every UID aggregate is backward-looking**: `expanding()` + `.shift(1)`.
   Any new feature must pass `tests/test_no_future_leakage.py`.

4. **The calibrator is fitted on the calibration partition only**, never on
   data the model was trained on.

5. **The test partition is evaluated once**, at the end. It is not a tuning set.

6. **Heavy aggregation runs in DuckDB / SQL**, not pandas.

7. **All cost and policy parameters live in `src/fraudq/config.py`.** No magic
   numbers scattered through the code.

8. **Config is passed explicitly, never read from a module-level global inside
   a function.** The sensitivity analysis builds variants with
   `dataclasses.replace` and runs the pipeline against each. A function that
   reaches for `CONFIG` directly instead of taking it as an argument silently
   ignores the sweep, and fails without raising.

9. **Logic lives in `src/`.** Notebooks are for exploration and for the final
   report, not for the pipeline.
