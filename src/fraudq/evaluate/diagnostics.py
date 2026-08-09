"""Model diagnostics: what the score is made of, and whether it was overtrained.

The rest of `evaluate/` answers "how good is this model". This module answers
the two questions that come before it: **what is the model looking at**, and
**can the number on test be believed**.

| Diagnostic                | What it settles                                  |
|---------------------------|--------------------------------------------------|
| feature importance        | Which columns carry the score (design.md §5.4).    |
| null-pattern blocks       | Whether the V-columns really are Vesta blocks.    |
| rank correlation          | How much of that importance is redundant.         |
| ROC and PR curves         | The full trade-off, not the scalar AUC.           |
| queue operating points    | Where a policy actually sits on that trade-off.   |
| two-sample KS             | Whether two score distributions are the same.     |
| learning curve            | Overtraining, isolated from drift.                |

## The KS caveat, which changes what may be concluded

In the textbook setup the train/test split is random, so a difference between
the classifier output on train and on test is overtraining and nothing else.
**Here the split is temporal** (design.md §4.1), so that same difference mixes
overtraining with genuine drift and cannot separate them. On top of that, at
n around 385,000 against 75,000 the p-value is ~0 for a difference of no
operational size whatsoever: with samples this large, the test answers "are
these distributions literally identical", which they are not and need not be.

So the KS numbers are reported as three separate comparisons, and only one of
them is about time alone:

- **calib against test**: both partitions are unseen by the model, so what is
  left is drift, cleanly.
- **train against test**: overtraining PLUS drift, summed and not separable.
- **train against calib**: the same sum over a shorter horizon.

The statistic ``D`` is the one to read, since it is a distance and does not
grow with n. **The clean overtraining measurement is `learning_curve_folds`**,
where the two series live in adjacent temporal windows, which is what holds
drift roughly constant between them.

Following the convention of the package, this module returns numbers and never
draws: the figures are built in `notebooks/03_results.ipynb` from the CSVs that
`fraudq.diagnostics` writes. Heavy imports live inside the functions that use
them, so importing this module costs nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The same deterministic top-K mask the published precision@K and recall@K use.
# Reusing it, rather than re-deriving the selection here, is what guarantees an
# operating point lands exactly where `metrics.precision_at_k` says it does.
from fraudq.evaluate.metrics import _top_k_mask

#: Score-distribution comparisons worth making, and the order to report them in.
#: The first is the only one that is about drift alone; see the module docstring.
KS_PAIRS = (("calib", "test"), ("train", "test"), ("train", "calib"))

#: Curves are written to CSV and drawn from there. `roc_curve` returns one point
#: per distinct threshold, which on 75,000 rows is tens of thousands of rows of
#: no visual consequence.
CURVE_MAX_POINTS = 1000


# ---------------------------------------------------------------------------
# What the model is made of
# ---------------------------------------------------------------------------


def importance_table(booster) -> pd.DataFrame:
    """Feature importance by gain and by split, the ranking design.md §5.4 asks for.

    Two importance types, because they disagree and the disagreement is the
    informative part. **Gain** totals how much each split on the feature reduced
    the loss, so it rewards features that decide a lot in a few places.
    **Split** just counts the splits, so it rewards features the trees consult
    constantly to make small adjustments.

    Read `gain` with its known bias in mind: it favours high-cardinality and
    continuous features, which have more places to cut. A frequency-encoded
    identifier will rank higher than a binary flag of the same real usefulness.

    Comes entirely out of the persisted booster and touches no data.

    Returns
    -------
    One row per feature, ordered by gain descending: `rank`, `feature`, `gain`,
    `gain_pct`, `split`, `split_pct`.
    """
    names = list(booster.feature_name())
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)

    out = pd.DataFrame({"feature": names, "gain": gain, "split": split})
    # A booster with no splits at all would divide by zero. It cannot happen in
    # a trained model, but the guard costs nothing and the NaN would be silent.
    out["gain_pct"] = 100.0 * out["gain"] / out["gain"].sum() if gain.sum() > 0 else 0.0
    out["split_pct"] = 100.0 * out["split"] / out["split"].sum() if split.sum() > 0 else 0.0

    out = out.sort_values("gain", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[["rank", "feature", "gain", "gain_pct", "split", "split_pct"]]


def top_features(importance: pd.DataFrame, n: int = 25) -> list[str]:
    """The top `n` feature names by gain, for the correlation and PSI reports."""
    return importance.head(n)["feature"].tolist()


def null_pattern_blocks(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Group columns by their EXACT null pattern: point 1 of design.md §5.4.

    Vesta's V-columns arrive anonymised, but they were engineered in blocks, and
    columns of the same block are null on exactly the same rows because they
    were derived from the same source. Grouping by the null mask recovers that
    structure without looking at a single value, and without the column-by-column
    archaeology the design rules out.

    The signature is the null mask itself, hashed. Two columns land in the same
    block only if they are null on precisely the same rows, which is a stricter
    and more honest criterion than sharing a null RATE.

    Returns
    -------
    One row per block, ordered by size descending: `block`, `n_columns`,
    `null_rate`, `columns` (comma separated).
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.DataFrame(columns=["block", "n_columns", "null_rate", "columns"])

    groups: dict[bytes, list[str]] = {}
    rates: dict[bytes, float] = {}
    for col in present:
        mask = df[col].isna().to_numpy()
        key = mask.tobytes()
        groups.setdefault(key, []).append(col)
        rates[key] = float(mask.mean())

    rows = [
        {"n_columns": len(cols), "null_rate": rates[key], "columns": ",".join(cols)}
        for key, cols in groups.items()
    ]
    out = pd.DataFrame(rows).sort_values(
        ["n_columns", "null_rate"], ascending=[False, True], ignore_index=True
    )
    out.insert(0, "block", np.arange(1, len(out) + 1))
    return out


def correlation_matrix(
    df: pd.DataFrame, features: list[str], method: str = "spearman"
) -> pd.DataFrame:
    """Rank correlation between features, in the order they are given.

    **Spearman and not Pearson, deliberately.** The C-columns, the D-columns and
    the amount are heavily skewed and have long tails, and Pearson on those is
    dominated by a handful of extreme rows: it would measure the outliers rather
    than the relationship. Spearman works on ranks, so it is invariant to any
    monotone transform, which is also the only structure a tree ensemble can
    exploit in the first place.

    The row and column order is the order of `features`, so passing them ranked
    by gain puts the important ones in the top-left corner of the heatmap.
    """
    present = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Features absent from the frame: {missing}.")
    return df[present].corr(method=method)


# ---------------------------------------------------------------------------
# The trade-off curves, and where a policy sits on them
# ---------------------------------------------------------------------------


def _thin(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices into 0..n-1, always keeping both ends."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def roc_points(y_true, scores, max_points: int = CURVE_MAX_POINTS) -> pd.DataFrame:
    """The ROC curve as a table: `fpr`, `tpr`, `threshold`.

    ROC-AUC is already reported as a scalar (metrics.py). The curve is here
    because the scalar averages over the whole range, including the operating
    points no review queue will ever be at. What matters operationally is the
    far left of the curve, where a capacity of 1 % of volume puts you.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    keep = _thin(len(fpr), max_points)
    return pd.DataFrame(
        {"fpr": fpr[keep], "tpr": tpr[keep], "threshold": thresholds[keep]}
    ).reset_index(drop=True)


def pr_points(y_true, scores, max_points: int = CURVE_MAX_POINTS) -> pd.DataFrame:
    """The precision-recall curve as a table: `recall`, `precision`, `threshold`.

    The leading discrimination curve under imbalance (design.md §6.4). Note that
    `precision_recall_curve` returns one more precision/recall pair than it
    returns thresholds, the trailing (1, 0) point; the extra row carries NaN in
    `threshold` rather than being silently dropped.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    thresholds = np.append(np.asarray(thresholds, dtype=float), np.nan)
    keep = _thin(len(precision), max_points)
    return pd.DataFrame(
        {"recall": recall[keep], "precision": precision[keep], "threshold": thresholds[keep]}
    ).reset_index(drop=True)


def operating_point(y_true, review_mask, label: str = "") -> dict:
    """Where a review set sits in ROC and PR space.

    For the review set R that `review_mask` selects:

        tpr = |fraud in R| / |all fraud|
        fpr = |legitimate in R| / |all legitimate|
        precision = |fraud in R| / |R|

    A caveat worth keeping attached to the number: `tpr` counts fraud that
    reached an ANALYST. It is not the "fraud caught" column of the policy table,
    which also counts fraud the automatic rule blocked outside the queue.
    """
    y = np.asarray(y_true, dtype=float)
    mask = np.asarray(review_mask, dtype=bool)

    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    k_eff = int(mask.sum())
    caught = float(y[mask].sum())

    return {
        "queue": label,
        "k": k_eff,
        "fraud_in_queue": int(caught),
        "tpr": caught / n_pos if n_pos else float("nan"),
        "fpr": (k_eff - caught) / n_neg if n_neg else float("nan"),
        "precision": caught / k_eff if k_eff else float("nan"),
    }


def queue_operating_point(y_true, ranking, k: int, label: str = "") -> dict:
    """The operating point of a GLOBAL top-K by `ranking`, over the whole window.

    With `ranking = p` this point lies ON the ROC curve by construction: taking
    the top K by score is a threshold on the score, which is what a ROC point
    is. That makes it the right reference to draw the other points against, and
    `operating_points_table` reports it as such.

    It is a reference and not a policy. The policies allocate PER DAY, because
    analyst capacity renews daily (design.md §2.3), and the union of daily
    queues is not a global threshold cut.
    """
    return operating_point(y_true, _top_k_mask(ranking, k), label=label)


def operating_points_table(
    df: pd.DataFrame,
    cfg,
    capacity_pct: float,
    p_col: str = "p",
    amt_col: str = "TransactionAmt",
    day_col: str = "day",
    target: str = "isFraud",
) -> pd.DataFrame:
    """The three points figure 7 marks on the ROC, and what each one is for.

    | queue                | what it is                                        |
    |----------------------|---------------------------------------------------|
    | `global_topk_by_score` | The reference. A global threshold, so ON the curve. |
    | `daily_topk_by_score`  | Policy 3, allocated per day as it really runs.    |
    | `daily_topk_by_value`  | Policy 4, this project's queue.                   |

    **Reading the three of them together is the point.** The ROC curve is an
    object about global thresholds, and a review queue is not one: capacity
    renews every day, so the same total spend is forced to take the best cases
    of each day rather than the best cases of the window. That constraint alone
    already moves a SCORE-ranked queue off the curve, before any question of how
    the queue is ranked. The distance from the second point to the third is then
    the part that is about ranking, which is the thesis of design.md §2.4.

    The two daily rows come from `simulate_queue` running the same policy
    functions the published results table used, so these points and that table
    describe one and the same allocation rather than two similar ones. `cfg` is
    passed explicitly and never read from a global, which is invariant 8 of §9.
    """
    from fraudq.evaluate.policies import actions_topk_by_score, actions_topk_by_value
    from fraudq.policy.simulate import simulate_queue

    y = df[target].to_numpy()
    rows = []
    for label, policy in (
        ("daily_topk_by_score", actions_topk_by_score),
        ("daily_topk_by_value", actions_topk_by_value),
    ):
        result = simulate_queue(
            df,
            policy,
            cfg,
            capacity_pct,
            p_col=p_col,
            amt_col=amt_col,
            day_col=day_col,
            target=target,
        )
        mask = (result.actions == "review").to_numpy()
        rows.append({**operating_point(y, mask, label=label), "capacity": result.capacity})

    # The reference is drawn at the SAME total spend as the policies, so the
    # three points are comparable rather than three different budgets.
    total_capacity = int(rows[0]["capacity"])
    reference = queue_operating_point(
        y, df[p_col].to_numpy(dtype=float), total_capacity, label="global_topk_by_score"
    )
    reference["capacity"] = total_capacity

    return pd.DataFrame([reference, *rows])


# ---------------------------------------------------------------------------
# Two-sample KS. Read the module docstring before reading the numbers.
# ---------------------------------------------------------------------------


def ks_two_sample(a, b) -> dict:
    """Two-sample Kolmogorov-Smirnov, with the sample sizes attached.

    Wrapped rather than called directly for one reason: `n_a` and `n_b` are half
    the interpretation. The statistic `D` is a distance between the two
    empirical CDFs and does not grow with n; the p-value is a statement about
    those n and, past a few tens of thousands of rows, is ~0 for any difference
    at all. Reporting one without the others invites the wrong conclusion.

    NaN is dropped on both sides.
    """
    from scipy.stats import ks_2samp

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n_a": len(a), "n_b": len(b)}

    result = ks_2samp(a, b)
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def score_ks_table(
    frames: dict[str, pd.DataFrame],
    pairs: tuple[tuple[str, str], ...] = KS_PAIRS,
    score_col: str = "score_raw",
    target: str = "isFraud",
) -> pd.DataFrame:
    """KS between partitions, split by class, the way a HEP overtraining plot is read.

    Splitting by class matters: the overall distributions can differ simply
    because the fraud RATE moved (it does, from 2.84 % to 4.09 % across test
    alone), which says nothing about the score. Comparing fraud against fraud
    and legitimate against legitimate removes that confound and leaves the shape.

    Pairs whose partitions are missing from `frames` are skipped, so a caller
    that only has calib and test still gets a table.

    Returns
    -------
    Rows of `pair`, `subset`, `statistic`, `pvalue`, `n_a`, `n_b`.
    """
    rows = []
    for a_name, b_name in pairs:
        if a_name not in frames or b_name not in frames:
            continue
        a_df, b_df = frames[a_name], frames[b_name]
        for subset, label in (("all", None), ("fraud", 1), ("legit", 0)):
            a = a_df if label is None else a_df[a_df[target] == label]
            b = b_df if label is None else b_df[b_df[target] == label]
            rows.append(
                {
                    "pair": f"{a_name}_vs_{b_name}",
                    "subset": subset,
                    **ks_two_sample(a[score_col], b[score_col]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The clean overtraining measurement
# ---------------------------------------------------------------------------


def learning_curve_folds(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[tuple[int, int], tuple[int, int]]],
    params: dict,
    target: str = "isFraud",
    day_col: str = "day",
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 200,
) -> pd.DataFrame:
    """PR-AUC per boosting round, on the fit set and on the validation set, per fold.

    The gap between the two curves at `best_iteration` IS the overtraining, and
    unlike a train-against-test comparison it is measured between two adjacent
    temporal windows, which is what holds drift roughly constant between them.

    ## Why this does not live in `cv_lightgbm`, and why it reproduces it anyway

    `models/train.py` is not touched. Adding the fit set to its `valid_sets` to
    record the second series would change what the early-stopping callback
    observes, and a change there moves `best_iteration`, therefore
    `n_estimators`, therefore the published model. This function retrains on its
    own and the artefacts are never regenerated.

    It still reproduces `cv_lightgbm` exactly, because the fit set is registered
    under the name `training`, and LightGBM's early stopping skips whichever
    dataset carries the booster's `_train_data_name` (`callback.py`, around line
    425). Verified rather than assumed: on the same folds and parameters both
    paths return identical `best_iteration` per fold. `run_diagnostics` asserts
    the same property against the persisted booster's tree count.

    Returns
    -------
    Long format, ready for a facet plot: `fold`, `iteration`, `fit_ap`,
    `valid_ap`, `gap`, `best_iteration`.
    """
    import lightgbm as lgb

    from fraudq.models.train import _validate_params

    _validate_params(params)
    day = df_train[day_col]
    frames = []

    for i, ((t_lo, t_hi), (v_lo, v_hi)) in enumerate(folds, start=1):
        fit = df_train[(day >= t_lo) & (day <= t_hi)]
        valid = df_train[(day >= v_lo) & (day <= v_hi)]

        dfit = lgb.Dataset(fit[feature_cols], label=fit[target])
        dvalid = lgb.Dataset(valid[feature_cols], label=valid[target], reference=dfit)

        evals: dict = {}
        booster = lgb.train(
            {**params, "objective": "binary", "metric": "average_precision", "verbosity": -1},
            dfit,
            num_boost_round=num_boost_round,
            # The order is the reporting order; the NAME is what matters, since
            # "training" is what makes early stopping ignore that series.
            valid_sets=[dvalid, dfit],
            valid_names=["valid", "training"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.record_evaluation(evals),
            ],
        )

        fit_ap = np.asarray(evals["training"]["average_precision"], dtype=float)
        valid_ap = np.asarray(evals["valid"]["average_precision"], dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "fold": i,
                    "iteration": np.arange(1, len(valid_ap) + 1),
                    "fit_ap": fit_ap,
                    "valid_ap": valid_ap,
                    "gap": fit_ap - valid_ap,
                    "best_iteration": int(booster.best_iteration),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def overtraining_summary(curve: pd.DataFrame) -> pd.DataFrame:
    """One row per fold, read at `best_iteration`: the numbers a report quotes.

    `gap_at_best` is the overtraining of that fold. `valid_ap_at_best` is the
    honest out-of-sample PR-AUC the fold contributes to the CV mean.
    """
    rows = []
    for fold, g in curve.groupby("fold", sort=True):
        best = int(g["best_iteration"].iloc[0])
        at_best = g[g["iteration"] == best]
        # A fold that ran out of rounds before early stopping fired has no row
        # at `best`; fall back to the last iteration recorded.
        at_best = at_best if len(at_best) else g.tail(1)
        rows.append(
            {
                "fold": int(fold),
                "best_iteration": best,
                "rounds_run": int(g["iteration"].max()),
                "fit_ap_at_best": float(at_best["fit_ap"].iloc[0]),
                "valid_ap_at_best": float(at_best["valid_ap"].iloc[0]),
                "gap_at_best": float(at_best["gap"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)
