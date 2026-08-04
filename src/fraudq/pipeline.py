"""Driver del pipeline: de los parquet de la ingesta a la tabla de políticas.

    python -m fraudq.pipeline                 # datos reales, data/processed/
    python -m fraudq.pipeline --synthetic     # dataset sintético, sin Kaggle

Encadena lo que los módulos de `src/fraudq/` hacen por separado:

    features -> split temporal -> CV -> entrenamiento -> calibración
             -> scoring -> umbral en calib -> comparación de políticas en test

y deja en disco los artefactos que consumen el notebook de resultados, el
simulador de cola y la API:

    reports/scored_calib.parquet     scores y p sobre la partición de calibración
    reports/scored_test.parquet      idem sobre test, tras la única evaluación
    reports/policy_comparison.csv    la tabla de las cuatro políticas
    models/artifacts/                booster, calibrador y metadatos

Sobre la partición de test: se toca UNA vez, al final, y el resultado se
persiste inmediatamente. El análisis de sensibilidad y el de drift trabajan
sobre ese parquet y no vuelven a mirarla. Volver a correr este script sobre los
mismos datos es legítimo; usar el número de test para elegir un hiperparámetro,
un umbral o un supuesto de costo no lo es (`docs/design.md` §9, invariante 5).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from fraudq.config import CONFIG, DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, Config
from fraudq.data.split import expanding_window_folds, split_by_day
from fraudq.data.synthetic import SYNTHETIC_SPLIT, make_synthetic_transactions
from fraudq.evaluate.metrics import brier_score, ece, pr_auc, roc_auc
from fraudq.evaluate.policies import compare_policies, fit_single_threshold, headline_savings
from fraudq.features.build import FrequencyEncoder, add_base_features, build_features
from fraudq.models.calibrate import (
    compare_calibrators,
    fit_isotonic,
    fit_platt,
    temporal_calibration_split,
)
from fraudq.models.persist import save_artifacts
from fraudq.models.train import cv_lightgbm, predict_scores, train_final_lgbm

#: Salidas del pipeline. `config.py` define FIGURES_DIR = reports/figures, así que
#: reports/ es su padre; los artefactos del modelo cuelgan de models/.
REPORTS_DIR = FIGURES_DIR.parent
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

#: Columnas que nunca son features: identificadores, el target y las que solo
#: sirven para particionar. `uid` queda fuera a propósito: es una llave de
#: agrupación, y meterla como número sería darle al modelo un identificador de
#: cliente en crudo.
_NOT_FEATURES = frozenset({"TransactionID", "isFraud", "TransactionDT", "day", "uid"})

_LGBM_PARAM_FIELDS = (
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_lambda",
)


def lgbm_params(model_cfg) -> dict:
    """Traduce `ModelConfig` a los parámetros que entiende LightGBM.

    Se pasan explícitos y no con `asdict`: `n_estimators`, `metric` y
    `calibration_method` gobiernan el flujo, no el booster, y colarlos en el
    diccionario haría que LightGBM los ignorase en silencio.
    """
    params = {f: getattr(model_cfg, f) for f in _LGBM_PARAM_FIELDS}
    params["seed"] = model_cfg.random_state
    params["num_threads"] = model_cfg.n_jobs
    return params


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Todas las columnas numéricas que no son identificador ni target."""
    return [
        c for c in df.columns if c not in _NOT_FEATURES and pd.api.types.is_numeric_dtype(df[c])
    ]


def prepare(df_raw: pd.DataFrame, cfg: Config) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Features y partición temporal.

    El encoder de frecuencia se ajusta SOLO sobre train y se aplica a las cuatro
    particiones: es lo que impide que la distribución de calibración y test se
    filtre a la representación con la que se entrena.
    """
    feats = build_features(df_raw)
    df = df_raw.merge(
        feats.drop(columns=["TransactionDT", "TransactionAmt"]),
        on="TransactionID",
        how="left",
    )
    df = add_base_features(df)

    parts = split_by_day(df, cfg.split)

    encoder = FrequencyEncoder().fit(parts["train"])
    parts = {name: encoder.transform(part) for name, part in parts.items()}

    feature_cols = select_feature_columns(parts["train"])
    return parts, feature_cols


def train(parts: dict[str, pd.DataFrame], feature_cols: list[str], cfg: Config, valid_len: int):
    """CV de ventana expansiva para fijar n_estimators, y ajuste final."""
    folds = expanding_window_folds(
        cfg.split.train_end_day, n_folds=cfg.split.n_cv_folds, valid_len=valid_len
    )
    params = lgbm_params(cfg.model)
    cv = cv_lightgbm(parts["train"], feature_cols, folds, params)
    print(f"  CV: {cv.summary()}")

    booster = train_final_lgbm(parts["train"], feature_cols, params, cv.n_estimators)
    return booster, cv


def calibrate(booster, parts: dict[str, pd.DataFrame], feature_cols: list[str], holdout_days: int):
    """Elige calibrador en un holdout temporal de calib y lo reajusta sobre todo calib.

    El calibrador se ajusta solo con datos que el modelo no vio nunca
    (`docs/design.md` §4.3). Toda la capa de costos depende de que ``p`` sea una
    probabilidad de verdad: un modelo sobreconfiado hace que la política
    "óptima" deje de serlo.
    """
    fit_df, hold_df = temporal_calibration_split(parts["calib"], holdout_days=holdout_days)
    fit_scores = predict_scores(booster, fit_df, feature_cols)
    hold_scores = predict_scores(booster, hold_df, feature_cols)

    table = compare_calibrators(fit_scores, fit_df["isFraud"], hold_scores, hold_df["isFraud"])
    print(table.to_string(float_format=lambda v: f"{v:.5f}"))

    # Se compara sobre el holdout y se elige por Brier, que penaliza a la vez
    # calibración y discriminación.
    winner = str(table["brier"].idxmin()) if "brier" in table.columns else "isotonic"
    if winner == "raw":
        # Puede pasar y no es un fallo: significa que el score ya estaba bien
        # calibrado. Se usa el isotónico igualmente para tener un objeto con la
        # misma interfaz aguas abajo, y queda constancia en la salida.
        print("  El score crudo gana en Brier; se usa isotónica por uniformidad de interfaz.")
        winner = "isotonic"
    print(f"  Calibrador elegido: {winner}")

    calib_scores = predict_scores(booster, parts["calib"], feature_cols)
    fit_fn = fit_platt if winner == "platt" else fit_isotonic
    calibrator = fit_fn(calib_scores, parts["calib"]["isFraud"])
    return calibrator, calib_scores, table


def _scored_frame(df: pd.DataFrame, scores, p) -> pd.DataFrame:
    cols = ["TransactionID", "day", "TransactionAmt", "isFraud"]
    out = df[cols].copy()
    out["score_raw"] = scores
    out["p"] = p
    return out


def run_pipeline(
    df_raw: pd.DataFrame,
    cfg: Config,
    reports_dir: Path,
    models_dir: Path,
    valid_len: int = 20,
    holdout_days: int = 6,
) -> dict:
    """Corre la cadena completa y devuelve las cifras de cabecera."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Features y partición temporal")
    parts, feature_cols = prepare(df_raw, cfg)
    for name, part in parts.items():
        print(f"  {name:>8}: {len(part):>7,} filas, días {part['day'].min()}-{part['day'].max()}")
    print(f"  {len(feature_cols)} features")

    print("[2/6] Entrenamiento")
    booster, cv = train(parts, feature_cols, cfg, valid_len)

    print("[3/6] Calibración")
    calibrator, calib_scores, calib_table = calibrate(booster, parts, feature_cols, holdout_days)

    print("[4/6] Scoring de calibración")
    p_calib = calibrator.predict(calib_scores)
    scored_calib = _scored_frame(parts["calib"], calib_scores, p_calib)
    scored_calib.to_parquet(reports_dir / "scored_calib.parquet", index=False)

    threshold = fit_single_threshold(scored_calib, cfg.cost)
    print(f"  Umbral de la política de score único, ajustado en calib: {threshold:.4f}")

    print("[5/6] Evaluación en test (la única mirada)")
    test_scores = predict_scores(booster, parts["test"], feature_cols)
    p_test = calibrator.predict(test_scores)
    scored_test = _scored_frame(parts["test"], test_scores, p_test)
    scored_test.to_parquet(reports_dir / "scored_test.parquet", index=False)

    y_test = scored_test["isFraud"].to_numpy()
    model_metrics = {
        "pr_auc": pr_auc(y_test, p_test),
        "roc_auc": roc_auc(y_test, p_test),
        "brier": brier_score(y_test, p_test),
        "ece": ece(y_test, p_test),
    }
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in model_metrics.items()))

    comparison = compare_policies(scored_test, cfg.cost, cfg.policy.daily_capacity_pct, threshold)
    comparison.to_csv(reports_dir / "policy_comparison.csv")
    print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))

    savings = headline_savings(comparison)
    print(
        f"  Ahorro de rankear por valor frente a rankear por score: "
        f"{savings['savings_per_1k']:.4f} por cada $1,000"
    )

    print("[6/6] Artefactos")
    artifacts = save_artifacts(models_dir, booster, calibrator, feature_cols, cfg.cost)
    print(f"  {artifacts}")

    return {
        "cv": cv,
        "calibration": calib_table,
        "threshold": threshold,
        "model_metrics": model_metrics,
        "comparison": comparison,
        "savings": savings,
        "feature_cols": feature_cols,
    }


def load_processed(processed_dir: Path) -> pd.DataFrame:
    """Carga el parquet de la ingesta, con identity unida si existe."""
    transactions = processed_dir / "transactions.parquet"
    if not transactions.exists():
        raise SystemExit(
            f"No encuentro {transactions}.\n"
            "Ejecuta antes:  ./scripts/download_data.sh  &&  python -m fraudq.data.ingest\n"
            "O prueba la cadena sin datos reales:  python -m fraudq.pipeline --synthetic"
        )
    df = pd.read_parquet(transactions)

    identity = processed_dir / "identity.parquet"
    if identity.exists():
        # Left join y se conservan los nulos: la ausencia de datos de identidad
        # es en sí misma una señal, y LightGBM trata NaN de forma nativa.
        df = df.merge(pd.read_parquet(identity), on="TransactionID", how="left")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Usa transacciones sintéticas en vez de los parquet de data/processed.",
    )
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    cfg = CONFIG
    valid_len = 20

    if args.synthetic:
        print("Dataset SINTÉTICO: verifica que la cadena corre, no produce resultados.\n")
        df_raw = make_synthetic_transactions()
        # El split de producción no cabe en un dataset de pocos días.
        cfg = dataclasses.replace(cfg, split=dataclasses.replace(cfg.split, **SYNTHETIC_SPLIT))
        valid_len = 10
    else:
        df_raw = load_processed(args.processed_dir)
        print(f"{len(df_raw):,} transacciones desde {args.processed_dir}\n")

    run_pipeline(
        df_raw,
        cfg,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        valid_len=valid_len,
        holdout_days=3 if args.synthetic else 6,
    )


if __name__ == "__main__":
    main()
