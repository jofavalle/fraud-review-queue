"""Split temporal con embargo — train / embargo / calib / test.

Va en: fraud-review-queue/src/fraudq/data/split.py

## El diseño (design.md §4)

Ordenado por tiempo y cortado por día:

    train    : días 0 .. train_end_day
    embargo  : train_end_day+1 .. train_end_day+embargo_days   (NO SE USA)
    calib    : .. calib_end_day        (calibrador + política)
    test     : calib_end_day+1 .. fin  (evaluación final, UNA mirada)

Las tres reglas no negociables (§4.2): jamás un split aleatorio; el embargo
simula el retraso real de las etiquetas (chargebacks); el calibrador se ajusta
en datos que el modelo nunca vio.

## Acoplamiento con config.py

`config.py` (en el repo, fuente única §9.1) define `SplitConfig` con
`train_end_day`, `embargo_days`, `calib_end_day` y derivados. Este módulo está
**duck-typed**: acepta cualquier objeto con esos atributos, y usa los derivados
(`calib_start_day`, `test_start_day`) si existen, o los calcula si no. Así el
módulo no impone nada que tu config no declare ya.
"""

from __future__ import annotations

import pandas as pd

_REQUIRED = ("train_end_day", "embargo_days", "calib_end_day")

PARTS = ("train", "embargo", "calib", "test")


def _boundaries(cfg) -> tuple[int, int, int, int]:
    """(train_end, calib_start, calib_end, test_start), validados."""
    missing = [a for a in _REQUIRED if not hasattr(cfg, a)]
    if missing:
        raise AttributeError(
            f"SplitConfig sin atributos {missing}; ver design.md §9.1 / config.py."
        )
    train_end = int(cfg.train_end_day)
    calib_start = int(getattr(cfg, "calib_start_day",
                              train_end + int(cfg.embargo_days) + 1))
    calib_end = int(cfg.calib_end_day)
    test_start = int(getattr(cfg, "test_start_day", calib_end + 1))

    if not (train_end < calib_start <= calib_end < test_start):
        raise ValueError(
            "Cortes incoherentes: se requiere train_end < calib_start <= "
            f"calib_end < test_start; recibido ({train_end}, {calib_start}, "
            f"{calib_end}, {test_start})."
        )
    if calib_start - train_end - 1 != int(cfg.embargo_days):
        raise ValueError(
            f"El embargo no cuadra: calib_start_day ({calib_start}) no es "
            f"train_end_day + embargo_days + 1 ({train_end} + {cfg.embargo_days} + 1)."
        )
    return train_end, calib_start, calib_end, test_start


def split_by_day(df: pd.DataFrame, cfg, day_col: str = "day") -> dict[str, pd.DataFrame]:
    """Particiona `df` por día en train / embargo / calib / test.

    Devuelve un dict con las cuatro particiones (copias, índice reseteado).
    Postcondiciones verificadas aquí mismo: disjuntas y cubren todo `df`.
    El embargo se devuelve SOLO para poder verificar que no se usa — ningún
    consumidor legítimo debería tocarlo.
    """
    if day_col not in df.columns:
        raise KeyError(f"Falta la columna '{day_col}' (la deriva la ingesta).")

    train_end, calib_start, calib_end, test_start = _boundaries(cfg)
    day = df[day_col]

    parts = {
        "train":   df[day <= train_end],
        "embargo": df[(day > train_end) & (day < calib_start)],
        "calib":   df[(day >= calib_start) & (day <= calib_end)],
        "test":    df[day >= test_start],
    }

    total = sum(len(p) for p in parts.values())
    if total != len(df):
        raise AssertionError(
            f"Las particiones no cubren el dataset: {total} != {len(df)}."
        )
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def expanding_window_folds(
    train_end_day: int,
    n_folds: int = 4,
    valid_len: int = 20,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Folds de ventana expansiva DENTRO de train (design.md §4.3).

    Devuelve [((train_lo, train_hi), (valid_lo, valid_hi)), ...] en días,
    ambos extremos inclusivos. Para train_end_day=119, n_folds=4, valid_len=20:

        fold 1: train 0-39,  valid 40-59
        fold 2: train 0-59,  valid 60-79
        fold 3: train 0-79,  valid 80-99
        fold 4: train 0-99,  valid 100-119

    KFold aleatorio aquí sería el pecado capital del dataset (§4.2): validar
    con el pasado un modelo entrenado con el futuro.
    """
    span = n_folds * valid_len
    if span > train_end_day + 1 - valid_len:
        raise ValueError(
            f"No caben {n_folds} folds de {valid_len} días en 0..{train_end_day} "
            "dejando al menos un tramo inicial de entrenamiento."
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
    """Genera (df_fit, df_valid) por fold, a partir del DataFrame de train."""
    day = df_train[day_col]
    for (t_lo, t_hi), (v_lo, v_hi) in folds:
        fit = df_train[(day >= t_lo) & (day <= t_hi)]
        valid = df_train[(day >= v_lo) & (day <= v_hi)]
        yield fit.reset_index(drop=True), valid.reset_index(drop=True)
