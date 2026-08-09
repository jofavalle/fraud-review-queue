"""Temporal split with an embargo: train / embargo / calib / test.

## The design (design.md §4)

Ordered by time and cut by day:

    train    : days 0 .. train_end_day
    embargo  : train_end_day+1 .. train_end_day+embargo_days   (NEVER USED)
    calib    : .. calib_end_day        (calibrator and policy)
    test     : calib_end_day+1 .. end  (final evaluation, ONE look)

The three non-negotiable rules (§4.2): never a random split; the embargo
simulates the real delay of the labels, which arrive as chargebacks; the
calibrator is fitted on data the model never saw.

## Coupling with config.py

`config.py`, the single source for these numbers, defines `SplitConfig` with
`train_end_day`, `embargo_days`, `calib_end_day` and the derived days. This
module is **duck-typed**: it accepts any object with those attributes, and uses
the derived ones (`calib_start_day`, `test_start_day`) if they exist or
computes them if they do not. That way the module imposes nothing the config
does not already declare.
"""

from __future__ import annotations

import pandas as pd

_REQUIRED = ("train_end_day", "embargo_days", "calib_end_day")

PARTS = ("train", "embargo", "calib", "test")


def _boundaries(cfg) -> tuple[int, int, int, int]:
    """(train_end, calib_start, calib_end, test_start), validated."""
    missing = [a for a in _REQUIRED if not hasattr(cfg, a)]
    if missing:
        raise AttributeError(f"SplitConfig has no attributes {missing}; see config.py.")
    train_end = int(cfg.train_end_day)
    calib_start = int(getattr(cfg, "calib_start_day", train_end + int(cfg.embargo_days) + 1))
    calib_end = int(cfg.calib_end_day)
    test_start = int(getattr(cfg, "test_start_day", calib_end + 1))

    if not (train_end < calib_start <= calib_end < test_start):
        raise ValueError(
            "Incoherent boundaries: train_end < calib_start <= calib_end < "
            f"test_start is required; got ({train_end}, {calib_start}, "
            f"{calib_end}, {test_start})."
        )
    if calib_start - train_end - 1 != int(cfg.embargo_days):
        raise ValueError(
            f"The embargo does not add up: calib_start_day ({calib_start}) is not "
            f"train_end_day + embargo_days + 1 ({train_end} + {cfg.embargo_days} + 1)."
        )
    return train_end, calib_start, calib_end, test_start


def split_by_day(df: pd.DataFrame, cfg, day_col: str = "day") -> dict[str, pd.DataFrame]:
    """Partition `df` by day into train / embargo / calib / test.

    Returns a dict with the four partitions, as copies with a reset index. The
    postconditions are checked right here: they are disjoint and they cover all
    of `df`. The embargo is returned ONLY so that its non-use can be verified;
    no legitimate consumer should touch it.
    """
    if day_col not in df.columns:
        raise KeyError(f"Column '{day_col}' is missing; the ingestion derives it.")

    train_end, calib_start, calib_end, test_start = _boundaries(cfg)
    day = df[day_col]

    parts = {
        "train": df[day <= train_end],
        "embargo": df[(day > train_end) & (day < calib_start)],
        "calib": df[(day >= calib_start) & (day <= calib_end)],
        "test": df[day >= test_start],
    }

    total = sum(len(p) for p in parts.values())
    if total != len(df):
        raise AssertionError(f"The partitions do not cover the dataset: {total} != {len(df)}.")
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def expanding_window_folds(
    train_end_day: int,
    n_folds: int = 4,
    valid_len: int = 20,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Expanding-window folds INSIDE train (design.md §4.4).

    Returns [((train_lo, train_hi), (valid_lo, valid_hi)), ...] in days, both
    ends inclusive. For train_end_day=119, n_folds=4, valid_len=20:

        fold 1: train 0-39,  valid 40-59
        fold 2: train 0-59,  valid 60-79
        fold 3: train 0-79,  valid 80-99
        fold 4: train 0-99,  valid 100-119

    A random KFold here would be the cardinal sin of this dataset (§4.1):
    validating on the past a model trained on the future.
    """
    span = n_folds * valid_len
    if span > train_end_day + 1 - valid_len:
        raise ValueError(
            f"{n_folds} folds of {valid_len} days do not fit in 0..{train_end_day} "
            "while leaving at least an initial stretch to train on."
        )
    folds = []
    for i in range(n_folds):
        valid_lo = train_end_day + 1 - (n_folds - i) * valid_len
        valid_hi = valid_lo + valid_len - 1
        folds.append(((0, valid_lo - 1), (valid_lo, valid_hi)))
    return folds


def fold_frames(
    df_train: pd.DataFrame,
    folds: list[tuple[tuple[int, int], tuple[int, int]]],
    day_col: str = "day",
):
    """Yield (df_fit, df_valid) per fold, out of the train DataFrame."""
    day = df_train[day_col]
    for (t_lo, t_hi), (v_lo, v_hi) in folds:
        fit = df_train[(day >= t_lo) & (day <= t_hi)]
        valid = df_train[(day >= v_lo) & (day <= v_hi)]
        yield fit.reset_index(drop=True), valid.reset_index(drop=True)
