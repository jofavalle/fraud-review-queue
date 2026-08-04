"""El pipeline completo corre de punta a punta sobre datos sintéticos.

No verifica ningún resultado del proyecto: los números que salen de un dataset
inventado no dicen nada sobre el fraude real. Verifica que la CADENA no se
rompe --features, split temporal, CV, entrenamiento, calibración, scoring,
asignación bajo capacidad y comparación de políticas-- y que deja en disco los
artefactos que consumen el notebook de resultados, el simulador y la API.

Existe porque el coste de descubrir un `KeyError` en el paso 5 es muy distinto
según dónde se descubra: aquí tarda segundos, en la máquina que entrena con los
590k registros reales cuesta la corrida entera.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from fraudq.config import CONFIG
from fraudq.data.synthetic import SYNTHETIC_SPLIT, make_synthetic_transactions
from fraudq.evaluate.policies import POLICY_ORDER
from fraudq.pipeline import run_pipeline

#: Columnas que `notebooks/03_results.ipynb` y el simulador esperan encontrar.
_SCORED_COLUMNS = {"TransactionID", "day", "TransactionAmt", "isFraud", "score_raw", "p"}


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Una sola corrida compartida: montarla es lo caro, no las aserciones."""
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
    # El trío que carga la API: booster, calibrador y metadatos.
    artifacts = out / "models" / "artifacts"
    assert (artifacts / "model.txt").exists()
    assert (artifacts / "calibrator.pkl").exists()
    assert (artifacts / "metadata.json").exists()


def test_scored_frames_have_the_expected_contract(pipeline_run):
    _, out, _ = pipeline_run
    for name in ("scored_calib", "scored_test"):
        scored = pd.read_parquet(out / "reports" / f"{name}.parquet")
        assert _SCORED_COLUMNS <= set(scored.columns), name
        # `p` tiene que ser una probabilidad de verdad: la capa de costos entera
        # depende de ello, y simulate_queue aborta si no lo es.
        assert scored["p"].between(0.0, 1.0).all(), name
        assert not scored["p"].isna().any(), name


def test_all_four_policies_are_compared(pipeline_run):
    result, _, _ = pipeline_run
    assert list(result["comparison"].index) == list(POLICY_ORDER)


def test_capacity_is_respected_by_every_policy(pipeline_run):
    """Ninguna política revisa más de lo que la capacidad permite.

    `capacity` en la tabla es la suma de los cupos diarios, así que la
    comparación agregada es legítima: si un solo día se pasara, el total lo
    delataría salvo compensación exacta, y las políticas no compensan.
    """
    result, _, _ = pipeline_run
    comparison = result["comparison"]
    assert (comparison["reviews"] <= comparison["capacity"]).all()


def test_the_queue_is_actually_exercised(pipeline_run):
    """El caso degenerado que este test existe para no dejar pasar.

    Con volumen diario bajo, `int(n * capacity_pct)` cae a cero: no se revisa
    nada, las políticas 3 y 4 se vuelven idénticas y el smoke pasaría sin haber
    probado la asignación, que es la pieza central del proyecto.
    """
    result, _, _ = pipeline_run
    comparison = result["comparison"]
    assert comparison.loc["topk_by_value", "capacity"] > 0
    assert comparison.loc["topk_by_value", "reviews"] > 0


def test_config_reaches_the_cost_layer(pipeline_run):
    """Los supuestos de costo llegan a la decisión, no se quedan en el config.

    Es el invariante 8 de design.md visto desde fuera: si el coste de revisión
    fuera ignorado, multiplicarlo por veinte no cambiaría nada. La comparación
    se rehace sobre el scoring ya persistido, así que no vuelve a mirar el test.
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
