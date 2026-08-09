"""The diagnostics have to be right about the model, not just produce a figure.

Three of these tests exist because the corresponding claim is made in prose
somewhere and would otherwise be unchecked:

- `test_a_score_ranked_queue_lands_on_the_roc_curve` is what makes figure 7
  legible. The point of that figure is that one queue sits on the curve and the
  other need not, and if the first part were false the figure would be arguing
  nothing.
- `test_recording_the_fit_series_does_not_move_early_stopping` is the property
  that lets `learning_curve_folds` claim it reproduces `cv_lightgbm` without
  `models/train.py` being touched. It is the reason the published model cannot
  move.
- `test_psi_of_a_distribution_against_itself_is_zero` covers `drift.psi`, which
  has been implemented since the start and, until the diagnostics run, had never
  been executed on anything.

Everything here runs on constructed or synthetic data, so it runs in CI without
`data/processed/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from fraudq.data.split import expanding_window_folds
from fraudq.evaluate.diagnostics import (
    KS_PAIRS,
    correlation_matrix,
    importance_table,
    ks_two_sample,
    learning_curve_folds,
    null_pattern_blocks,
    operating_points_table,
    overtraining_summary,
    pr_points,
    queue_operating_point,
    roc_points,
    score_ks_table,
    top_features,
)
from fraudq.evaluate.drift import psi
from fraudq.evaluate.metrics import precision_at_k, recall_at_k

COSTS = SimpleNamespace(F=20.0, m=0.25, phi=10.0, r=2.0)

#: Small enough to train in about a second, large enough that early stopping
#: has something to stop on.
_N_ROWS = 4_000
_N_FEATURES = 8
_LGBM_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "seed": 42,
    "num_threads": 2,
}


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    """A scored partition with signal: `p` correlates with the label."""
    rng = np.random.default_rng(7)
    n = 3_000
    p = rng.beta(0.8, 8.0, size=n)
    return pd.DataFrame(
        {
            "p": p,
            "TransactionAmt": rng.lognormal(4.0, 1.2, size=n),
            "isFraud": rng.binomial(1, np.clip(p * 2.5, 0.0, 0.95)),
        }
    )


@pytest.fixture(scope="module")
def trainable() -> tuple[pd.DataFrame, list[str]]:
    """A frame shaped like the train partition: `day`, `isFraud`, numeric features."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(_N_ROWS, _N_FEATURES))
    logit = 1.3 * X[:, 0] + 0.9 * X[:, 1] - 0.6 * X[:, 2] - 2.4
    cols = [f"f{i}" for i in range(_N_FEATURES)]
    df = pd.DataFrame(X, columns=cols)
    df["day"] = rng.integers(0, 120, _N_ROWS)
    df["isFraud"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return df, cols


@pytest.fixture(scope="module")
def booster(trainable):
    """A real LightGBM booster, so the importance table is read from the real API."""
    import lightgbm as lgb

    df, cols = trainable
    dtrain = lgb.Dataset(df[cols], label=df["isFraud"])
    return lgb.train({**_LGBM_PARAMS, "objective": "binary", "verbosity": -1}, dtrain, 40)


# --------------------------------------------------------------- importance


def test_importance_lists_every_feature_exactly_once(booster, trainable):
    _, cols = trainable
    table = importance_table(booster)
    assert sorted(table["feature"]) == sorted(cols)
    assert table["rank"].tolist() == list(range(1, len(cols) + 1))


def test_gain_percentages_sum_to_one_hundred(booster):
    table = importance_table(booster)
    assert table["gain_pct"].sum() == pytest.approx(100.0)
    assert table["split_pct"].sum() == pytest.approx(100.0)


def test_importance_is_ordered_by_gain_descending(booster):
    gain = importance_table(booster)["gain"].to_numpy()
    assert np.all(np.diff(gain) <= 0)


def test_top_features_follows_the_ranking(booster):
    table = importance_table(booster)
    assert top_features(table, 3) == table.head(3)["feature"].tolist()


# ------------------------------------------------------------ null patterns


def test_null_pattern_blocks_groups_columns_that_share_their_nulls():
    """Two columns belong to the same block only if they are null on the SAME
    rows. Sharing a null rate is not enough, and this frame separates the two:
    `b` and `c` are both null twice, but not in the same places."""
    df = pd.DataFrame(
        {
            "a": [1.0, None, None, 4.0],
            "b": [2.0, None, None, 5.0],  # same pattern as `a`
            "c": [None, 3.0, 6.0, None],  # same RATE as `a`, different pattern
        }
    )
    blocks = null_pattern_blocks(df, ["a", "b", "c"])

    assert len(blocks) == 2
    assert blocks.loc[0, "n_columns"] == 2
    assert blocks.loc[0, "columns"] == "a,b"
    assert blocks["null_rate"].tolist() == [0.5, 0.5]


def test_null_pattern_blocks_ignores_absent_columns():
    df = pd.DataFrame({"a": [1.0, None]})
    assert len(null_pattern_blocks(df, ["a", "not_here"])) == 1
    assert null_pattern_blocks(df, ["not_here"]).empty


# ------------------------------------------------------------- correlation


def test_correlation_matrix_is_symmetric_with_unit_diagonal(trainable):
    df, cols = trainable
    corr = correlation_matrix(df, cols[:4])
    assert corr.shape == (4, 4)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T)


def test_correlation_preserves_the_order_it_is_given(trainable):
    df, cols = trainable
    order = [cols[3], cols[0], cols[1]]
    assert correlation_matrix(df, order).columns.tolist() == order


def test_spearman_sees_a_monotone_transform_that_pearson_misses():
    """The reason the default is Spearman. `y = exp(5x)` is a perfect monotone
    function of `x`, so the rank correlation is exactly 1, while Pearson reports
    a weaker number because the relationship is not a straight line."""
    x = np.linspace(0.0, 1.0, 400)
    df = pd.DataFrame({"x": x, "y": np.exp(5.0 * x)})
    assert correlation_matrix(df, ["x", "y"]).loc["x", "y"] == pytest.approx(1.0)
    assert correlation_matrix(df, ["x", "y"], method="pearson").loc["x", "y"] < 0.95


def test_correlation_raises_on_a_missing_feature(trainable):
    df, cols = trainable
    with pytest.raises(KeyError, match="absent"):
        correlation_matrix(df, [cols[0], "not_a_column"])


# ------------------------------------------------ curves and operating points


def test_roc_points_are_monotone_and_span_the_unit_square(scored):
    roc = roc_points(scored["isFraud"], scored["p"])
    assert np.all(np.diff(roc["fpr"]) >= 0)
    assert np.all(np.diff(roc["tpr"]) >= 0)
    assert (roc["fpr"].iloc[0], roc["tpr"].iloc[0]) == (0.0, 0.0)
    assert (roc["fpr"].iloc[-1], roc["tpr"].iloc[-1]) == (1.0, 1.0)


def test_curves_are_thinned_but_keep_their_ends(scored):
    roc = roc_points(scored["isFraud"], scored["p"], max_points=50)
    assert len(roc) <= 50
    assert (roc["fpr"].iloc[-1], roc["tpr"].iloc[-1]) == (1.0, 1.0)


def test_pr_points_carry_the_trailing_row_with_a_null_threshold(scored):
    """`precision_recall_curve` returns one more point than thresholds. Dropping
    it silently would shorten the curve; carrying it with NaN says what it is."""
    pr = pr_points(scored["isFraud"], scored["p"])
    assert np.isnan(pr["threshold"].iloc[-1])
    assert pr["precision"].iloc[-1] == 1.0
    assert pr["recall"].iloc[-1] == 0.0


def test_operating_point_agrees_with_the_published_precision_and_recall(scored):
    """The operating point must be the same selection `metrics.precision_at_k`
    reports, or the figure and the results table would disagree."""
    y, p, k = scored["isFraud"], scored["p"], 200
    point = queue_operating_point(y, p, k)
    assert point["precision"] == pytest.approx(precision_at_k(y, p, k))
    assert point["tpr"] == pytest.approx(recall_at_k(y, p, k))


def _distance_to_polyline(px: float, py: float, xs, ys) -> float:
    """Shortest distance from a point to the polyline through (xs, ys).

    Distance to the LINE and not to the nearest stored vertex, which is the
    weaker claim and the right one. `roc_curve` drops intermediate vertices that
    are collinear with their neighbours, so a point can sit exactly on the curve
    without being one of the rows in the CSV. Nor does interpolating `tpr` at
    the point's `fpr` work: the ROC has vertical segments, where several frauds
    are caught without a single new false positive, and there `tpr` is not a
    function of `fpr` at all.
    """
    x0, y0 = np.asarray(xs[:-1], dtype=float), np.asarray(ys[:-1], dtype=float)
    dx, dy = np.asarray(xs[1:], dtype=float) - x0, np.asarray(ys[1:], dtype=float) - y0
    length2 = dx**2 + dy**2
    # Projection onto each segment, clamped to its ends. Zero-length segments
    # collapse to their own start, which the clamp already handles.
    t = np.clip(
        np.divide(
            (px - x0) * dx + (py - y0) * dy, length2, out=np.zeros_like(dx), where=length2 > 0
        ),
        0.0,
        1.0,
    )
    return float(np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)).min())


def test_a_score_ranked_queue_lands_on_the_roc_curve(scored):
    """The property figure 7 rests on. Taking the top K by score IS a threshold
    on the score, so the point lies on the curve by construction, and the marker
    the figure draws falls on the line it draws."""
    y, p, k = scored["isFraud"], scored["p"], 200
    point = queue_operating_point(y, p, k)
    roc = roc_points(y, p, max_points=len(scored) + 2)

    distance = _distance_to_polyline(point["fpr"], point["tpr"], roc["fpr"], roc["tpr"])
    assert distance == pytest.approx(0.0, abs=1e-12)


def test_a_value_ranked_queue_need_not_land_on_the_roc_curve(scored):
    """The other half of the same argument: ranking by value selects a different
    set of the same size, so it is free to sit off the curve. If this ever
    stopped being true, the thesis of design.md §2.4 would be empty."""
    from fraudq.policy.costs import value_of_review

    y, p, k = scored["isFraud"], scored["p"], 200
    value = value_of_review(p.to_numpy(), scored["TransactionAmt"].to_numpy(), COSTS)

    by_value = queue_operating_point(y, value, k)
    roc = roc_points(y, p, max_points=len(scored) + 2)

    distance = _distance_to_polyline(by_value["fpr"], by_value["tpr"], roc["fpr"], roc["tpr"])
    assert distance > 1e-6


def test_operating_points_table_reports_both_queues(scored):
    table = operating_points_table(scored, COSTS, capacity=200)
    assert table["queue"].tolist() == ["topk_by_score", "topk_by_value"]
    assert (table["k"] == 200).all()
    assert table[["tpr", "fpr", "precision"]].notna().all().all()


# --------------------------------------------------------------------- KS


def test_ks_matches_scipy(scored):
    """The wrapper adds the sample sizes and drops NaN; it must not change the
    statistic scipy computes."""
    from scipy.stats import ks_2samp

    a = scored.loc[scored["isFraud"] == 1, "p"].to_numpy()
    b = scored.loc[scored["isFraud"] == 0, "p"].to_numpy()
    expected = ks_2samp(a, b)

    result = ks_two_sample(a, b)
    assert result["statistic"] == pytest.approx(expected.statistic)
    assert result["pvalue"] == pytest.approx(expected.pvalue)
    assert (result["n_a"], result["n_b"]) == (len(a), len(b))


def test_ks_of_a_sample_against_itself_is_zero(scored):
    result = ks_two_sample(scored["p"], scored["p"])
    assert result["statistic"] == pytest.approx(0.0)
    assert result["pvalue"] == pytest.approx(1.0)


def test_ks_drops_nulls_on_both_sides():
    result = ks_two_sample([1.0, 2.0, np.nan], [1.0, 2.0])
    assert (result["n_a"], result["n_b"]) == (2, 2)
    assert result["statistic"] == pytest.approx(0.0)


def test_ks_of_an_empty_sample_is_nan_rather_than_an_exception():
    result = ks_two_sample([], [1.0, 2.0])
    assert np.isnan(result["statistic"])
    assert result["n_a"] == 0


def test_score_ks_table_covers_every_pair_and_class(scored):
    frames = {name: scored.rename(columns={"p": "score_raw"}) for name in ("train", "calib")}
    frames["test"] = frames["train"]
    table = score_ks_table(frames)

    assert len(table) == 3 * len(KS_PAIRS)
    assert set(table["subset"]) == {"all", "fraud", "legit"}
    # Comparing identical frames: every distance is zero, by class as well.
    assert table["statistic"].abs().max() == pytest.approx(0.0)


def test_score_ks_table_skips_pairs_whose_partition_is_absent(scored):
    """The cheap phase of the driver only has calib and test, and must still get
    a table rather than a KeyError."""
    renamed = scored.rename(columns={"p": "score_raw"})
    table = score_ks_table({"calib": renamed, "test": renamed})
    assert set(table["pair"]) == {"calib_vs_test"}


# ------------------------------------------------------------------- PSI


def test_psi_of_a_distribution_against_itself_is_zero():
    """`drift.psi` has been implemented since the start and had never run."""
    rng = np.random.default_rng(3)
    sample = rng.lognormal(3.0, 1.0, size=5_000)
    assert psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_when_the_distribution_moves():
    rng = np.random.default_rng(3)
    reference = rng.normal(0.0, 1.0, size=20_000)
    assert psi(reference, rng.normal(0.1, 1.0, size=20_000)) < psi(
        reference, rng.normal(1.0, 1.0, size=20_000)
    )


# --------------------------------------------------------- learning curve


@pytest.fixture(scope="module")
def folds():
    return expanding_window_folds(119, n_folds=2, valid_len=20)


@pytest.fixture(scope="module")
def curve(trainable, folds):
    df, cols = trainable
    return learning_curve_folds(
        df, cols, folds, _LGBM_PARAMS, num_boost_round=200, early_stopping_rounds=20
    )


def test_learning_curve_records_both_series_for_every_fold(curve, folds):
    assert set(curve["fold"]) == {1, 2}
    assert len(folds) == 2
    for _, g in curve.groupby("fold"):
        assert g["iteration"].tolist() == list(range(1, len(g) + 1))
        assert g[["fit_ap", "valid_ap"]].notna().all().all()
        assert np.allclose(g["gap"], g["fit_ap"] - g["valid_ap"])


def test_recording_the_fit_series_does_not_move_early_stopping(trainable, folds, curve):
    """The property that keeps `models/train.py` untouched and the published
    model frozen. `cv_lightgbm` decides `n_estimators` from these iterations, so
    if registering the fit set as an extra valid_set changed them, regenerating
    the artefacts would produce a different model. LightGBM skips whichever
    dataset carries the booster's `_train_data_name`, which is why naming it
    `training` is not cosmetic."""
    from fraudq.models.train import cv_lightgbm

    df, cols = trainable
    reference = cv_lightgbm(
        df, cols, folds, _LGBM_PARAMS, num_boost_round=200, early_stopping_rounds=20
    )
    recovered = curve.groupby("fold")["best_iteration"].first().tolist()
    assert recovered == reference.best_iters


def test_the_fit_series_ends_above_the_validation_series(curve):
    """Not a tautology round by round, but by the last iteration a gradient
    boosting model has always fitted its own training data better than held-out
    data. A curve where this failed would mean the two series are swapped."""
    for _, g in curve.groupby("fold"):
        assert g["fit_ap"].iloc[-1] > g["valid_ap"].iloc[-1]


def test_overtraining_summary_reads_the_curve_at_the_best_iteration(curve):
    summary = overtraining_summary(curve)
    assert summary["fold"].tolist() == [1, 2]
    assert (summary["best_iteration"] <= summary["rounds_run"]).all()

    for row in summary.itertuples():
        at_best = curve[(curve["fold"] == row.fold) & (curve["iteration"] == row.best_iteration)]
        assert row.valid_ap_at_best == pytest.approx(at_best["valid_ap"].iloc[0])
        assert row.gap_at_best == pytest.approx(row.fit_ap_at_best - row.valid_ap_at_best)


def test_learning_curve_refuses_resampling_parameters(trainable, folds):
    """The invariant of design.md §6.2 travels with the diagnostics: a curve
    drawn under `scale_pos_weight` would describe a model this project forbids."""
    df, cols = trainable
    with pytest.raises(ValueError, match="Resampling parameters forbidden"):
        learning_curve_folds(df, cols, folds, {**_LGBM_PARAMS, "scale_pos_weight": 5.0})
