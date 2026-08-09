"""The whole pipeline runs end to end over synthetic data.

It verifies no result of the project: numbers out of an invented dataset say
nothing about real fraud. What it verifies is that the CHAIN does not break
(features, temporal split, CV, training, calibration, scoring, allocation under
capacity and the policy comparison) and that it leaves on disk the artefacts
the results notebook, the simulator and the API consume.

It exists because the cost of discovering a `KeyError` at step 5 depends
entirely on where it is discovered: here it takes seconds, on the machine
training over the 590k real records it costs the whole run.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from fraudq.config import CONFIG
from fraudq.data.synthetic import SYNTHETIC_SPLIT, make_synthetic_transactions
from fraudq.evaluate.policies import POLICY_ORDER
from fraudq.pipeline import run_pipeline

#: Columns `notebooks/03_results.ipynb` and the simulator expect to find.
_SCORED_COLUMNS = {"TransactionID", "day", "TransactionAmt", "isFraud", "score_raw", "p"}


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """One shared run: setting it up is the expensive part, not the assertions."""
    out = tmp_path_factory.mktemp("pipeline")
    cfg = replace(CONFIG, split=replace(CONFIG.split, **SYNTHETIC_SPLIT))
    df = make_synthetic_transactions(n_days=70, txns_per_day=200, seed=7)

    result = run_pipeline(
        df,
        cfg,
        reports_dir=out / "reports",
        models_dir=out / "models" / "artifacts",
        valid_len=10,
        holdout_days=3,
    )
    return result, out, cfg


def test_artifacts_are_written(pipeline_run):
    _, out, _ = pipeline_run
    reports = out / "reports"
    assert (reports / "scored_calib.parquet").exists()
    assert (reports / "scored_test.parquet").exists()
    assert (reports / "policy_comparison.csv").exists()
    # The three the API loads: booster, calibrator and metadata.
    artifacts = out / "models" / "artifacts"
    assert (artifacts / "model.txt").exists()
    assert (artifacts / "calibrator.pkl").exists()
    assert (artifacts / "metadata.json").exists()


def test_scored_frames_have_the_expected_contract(pipeline_run):
    _, out, _ = pipeline_run
    for name in ("scored_calib", "scored_test"):
        scored = pd.read_parquet(out / "reports" / f"{name}.parquet")
        assert _SCORED_COLUMNS <= set(scored.columns), name
        # `p` has to be a real probability: the whole cost layer depends on it,
        # and simulate_queue aborts when it is not.
        assert scored["p"].between(0.0, 1.0).all(), name
        assert not scored["p"].isna().any(), name


def test_all_four_policies_are_compared(pipeline_run):
    result, _, _ = pipeline_run
    assert list(result["comparison"].index) == list(POLICY_ORDER)


def test_capacity_is_respected_by_every_policy(pipeline_run):
    """No policy reviews more than capacity allows.

    `capacity` in the table is the sum of the daily quotas, so the aggregate
    comparison is legitimate: if a single day went over, the total would give
    it away barring exact compensation, and the policies do not compensate.
    """
    result, _, _ = pipeline_run
    comparison = result["comparison"]
    assert (comparison["reviews"] <= comparison["capacity"]).all()


def test_the_queue_is_actually_exercised(pipeline_run):
    """The degenerate case this test exists to refuse.

    At low daily volume `int(n * capacity_pct)` falls to zero: nothing gets
    reviewed, policies 3 and 4 become identical, and the smoke would pass
    without having tested the allocation, the central piece of the project.
    """
    result, _, _ = pipeline_run
    comparison = result["comparison"]
    assert comparison.loc["topk_by_value", "capacity"] > 0
    assert comparison.loc["topk_by_value", "reviews"] > 0


def test_config_reaches_the_cost_layer(pipeline_run):
    """The cost assumptions reach the decision, they do not stay in the config.

    Invariant 8 of design.md seen from outside: if the review cost were
    ignored, multiplying it by twenty would change nothing. The comparison is
    redone over the already-persisted scoring, so it never looks at test again.
    """
    result, out, cfg = pipeline_run
    from fraudq.evaluate.policies import compare_policies

    scored_test = pd.read_parquet(out / "reports" / "scored_test.parquet")
    expensive = replace(cfg.cost, review_cost=cfg.cost.review_cost * 20.0)
    with_expensive_reviews = compare_policies(
        scored_test, expensive, cfg.policy.daily_capacity_pct, result["threshold"]
    )
    assert (
        with_expensive_reviews.loc["topk_by_value", "total_cost"]
        != result["comparison"].loc["topk_by_value", "total_cost"]
    )
