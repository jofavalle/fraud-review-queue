"""Temporal split: a structural property, checked mechanically.

The partitions do not overlap, the embargo exists and nothing uses it, and the
expanding-window folds never validate on the past.

It does not depend on the real config.py: it uses a duck-typed synthetic config
with the same attributes (train_end_day, embargo_days, calib_end_day). The real
SplitConfig satisfies it by construction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from fraudq.data.split import expanding_window_folds, fold_frames, split_by_day


@pytest.fixture
def cfg():
    # The values of design.md §4: train 0-119, a 10-day embargo, calib to 155.
    return SimpleNamespace(train_end_day=119, embargo_days=10, calib_end_day=155)


@pytest.fixture
def df_days():
    # One record per day, days 0 to 180, with an id to trace rows.
    days = list(range(181))
    return pd.DataFrame({"TransactionID": days, "day": days})


def test_partitions_are_disjoint_and_cover(df_days, cfg):
    parts = split_by_day(df_days, cfg)

    ids = pd.concat([p["TransactionID"] for p in parts.values()])
    assert len(ids) == len(df_days), "the partitions do not cover the whole dataset"
    assert ids.is_unique, "some rows are in more than one partition"


def test_temporal_order_and_embargo(df_days, cfg):
    parts = split_by_day(df_days, cfg)

    # Strict temporal order between partitions.
    assert parts["train"]["day"].max() < parts["calib"]["day"].min()
    assert parts["calib"]["day"].max() < parts["test"]["day"].min()

    # The embargo exists, spans exactly embargo_days, and separates the two.
    assert len(parts["embargo"]) == cfg.embargo_days
    assert parts["embargo"]["day"].min() == parts["train"]["day"].max() + 1
    assert parts["embargo"]["day"].max() == parts["calib"]["day"].min() - 1

    # The boundaries are the ones in the design.
    assert parts["train"]["day"].max() == 119
    assert parts["calib"]["day"].min() == 130
    assert parts["calib"]["day"].max() == 155
    assert parts["test"]["day"].min() == 156


def test_incoherent_config_raises(df_days):
    bad = SimpleNamespace(train_end_day=150, embargo_days=10, calib_end_day=155)
    with pytest.raises(ValueError):
        split_by_day(df_days, bad)


def test_expanding_window_folds_match_design():
    # design.md §4.4, literally.
    folds = expanding_window_folds(train_end_day=119, n_folds=4, valid_len=20)
    assert folds == [
        ((0, 39), (40, 59)),
        ((0, 59), (60, 79)),
        ((0, 79), (80, 99)),
        ((0, 99), (100, 119)),
    ]


def test_folds_never_validate_with_the_past(df_days, cfg):
    parts = split_by_day(df_days, cfg)
    folds = expanding_window_folds(cfg.train_end_day)

    for fit, valid in fold_frames(parts["train"], folds):
        # The fit stretch ends STRICTLY before the validation one begins.
        assert fit["day"].max() < valid["day"].min()
        # And all of it lives inside train: no embargo, no calib, no test.
        assert valid["day"].max() <= cfg.train_end_day


def test_folds_that_do_not_fit_raise():
    with pytest.raises(ValueError):
        expanding_window_folds(train_end_day=59, n_folds=4, valid_len=20)
