"""Persistencia de artefactos: booster + calibrador + metadatos — delegable.

Va en: fraud-review-queue/src/fraudq/models/persist.py

El contrato entre el entrenamiento (Días 4-5) y el despliegue (Día 8): lo que
la API y el Streamlit cargan es EXACTAMENTE lo que la evaluación del Día 6
usó. Un directorio de artefactos contiene:

    model.txt        — LightGBM booster (formato nativo)
    calibrator.pkl   — PlattCalibrator o IsotonicCalibrator (Día 5)
    metadata.json    — feature_cols + snapshot de los parámetros de costo

El snapshot de costos va en los metadatos a propósito: la API decide con los
MISMOS parámetros congelados con los que se midió el resultado del README. Si
el negocio cambia F o φ, se regeneran artefactos — no se editan a mano.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

MODEL_FILE = "model.txt"
CALIBRATOR_FILE = "calibrator.pkl"
METADATA_FILE = "metadata.json"

_COST_KEYS = ("F", "m", "phi", "r")


def save_artifacts(dir_path, booster, calibrator, feature_cols, cost_cfg) -> Path:
    """Guarda el trío de artefactos. `cost_cfg` es duck-typed (F, m, phi, r)."""
    out = Path(dir_path)
    out.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(out / MODEL_FILE))
    with open(out / CALIBRATOR_FILE, "wb") as f:
        pickle.dump(calibrator, f)
    metadata = {
        "feature_cols": list(feature_cols),
        "costs": {k: float(getattr(cost_cfg, k)) for k in _COST_KEYS},
    }
    with open(out / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    return out


def load_artifacts(dir_path):
    """Carga (booster, calibrator, feature_cols, cost_cfg) desde un directorio.

    `cost_cfg` vuelve como SimpleNamespace(F, m, phi, r): la misma interfaz
    duck-typed que consume policy/costs.py.
    """
    import lightgbm as lgb

    src = Path(dir_path)
    for name in (MODEL_FILE, CALIBRATOR_FILE, METADATA_FILE):
        if not (src / name).exists():
            raise FileNotFoundError(
                f"Falta {name} en {src}. ¿Corriste save_artifacts tras el Día 6?"
            )

    booster = lgb.Booster(model_file=str(src / MODEL_FILE))
    with open(src / CALIBRATOR_FILE, "rb") as f:
        calibrator = pickle.load(f)
    metadata = json.loads((src / METADATA_FILE).read_text())
    cost_cfg = SimpleNamespace(**metadata["costs"])
    return booster, calibrator, metadata["feature_cols"], cost_cfg
