"""Test del split temporal — completo (columna delegable del plan §5.3).

Va en: fraud-review-queue/tests/test_split_no_temporal_overlap.py

A diferencia del test de leakage (que escribes tú), este es un test mecánico de
una propiedad estructural: las particiones no se solapan, el embargo existe y
está vacío de usos, y los folds de ventana expansiva nunca validan con el
pasado. Aún así: léelo y sé capaz de explicar cada aserción.

No depende de tu config.py real: usa un config sintético duck-typed con los
mismos atributos (train_end_day, embargo_days, calib_end_day). Tu SplitConfig
real lo satisface por construcción.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from fraudq.data.split import expanding_window_folds, fold_frames, split_by_day


@pytest.fixture
def cfg():
    # Mismos valores que design.md §4.1: train 0-119, embargo 10 días, calib hasta 155.
    return SimpleNamespace(train_end_day=119, embargo_days=10, calib_end_day=155)


@pytest.fixture
def df_days():
    # Un registro por día, días 0..180, con un id para rastrear filas.
    days = list(range(181))
    return pd.DataFrame({"TransactionID": days, "day": days})


def test_partitions_are_disjoint_and_cover(df_days, cfg):
    parts = split_by_day(df_days, cfg)

    ids = pd.concat([p["TransactionID"] for p in parts.values()])
    assert len(ids) == len(df_days), "las particiones no cubren todo el dataset"
    assert ids.is_unique, "hay filas en más de una partición"


def test_temporal_order_and_embargo(df_days, cfg):
    parts = split_by_day(df_days, cfg)

    # Orden temporal estricto entre particiones.
    assert parts["train"]["day"].max() < parts["calib"]["day"].min()
    assert parts["calib"]["day"].max() < parts["test"]["day"].min()

    # El embargo existe, tiene exactamente embargo_days días, y separa train de calib.
    assert len(parts["embargo"]) == cfg.embargo_days
    assert parts["embargo"]["day"].min() == parts["train"]["day"].max() + 1
    assert parts["embargo"]["day"].max() == parts["calib"]["day"].min() - 1

    # Los cortes son los del diseño.
    assert parts["train"]["day"].max() == 119
    assert parts["calib"]["day"].min() == 130
    assert parts["calib"]["day"].max() == 155
    assert parts["test"]["day"].min() == 156


def test_incoherent_config_raises(df_days):
    bad = SimpleNamespace(train_end_day=150, embargo_days=10, calib_end_day=155)
    with pytest.raises(ValueError):
        split_by_day(df_days, bad)


def test_expanding_window_folds_match_design():
    # design.md §4.3, literal.
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
        # El tramo de ajuste termina ESTRICTAMENTE antes de que empiece el de validación.
        assert fit["day"].max() < valid["day"].min()
        # Y todo vive dentro de train: ni embargo, ni calib, ni test.
        assert valid["day"].max() <= cfg.train_end_day


def test_folds_that_do_not_fit_raise():
    with pytest.raises(ValueError):
        expanding_window_folds(train_end_day=59, n_folds=4, valid_len=20)
