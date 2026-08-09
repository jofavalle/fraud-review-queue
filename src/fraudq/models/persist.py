"""Artefact persistence: booster, calibrator and metadata.

The contract between training and deployment: what the API and the Streamlit
app load is EXACTLY what the evaluation used. An artefact directory holds:

    model.txt        LightGBM booster, in its native format
    calibrator.pkl   PlattCalibrator or IsotonicCalibrator
    metadata.json    feature_cols plus a snapshot of the cost parameters

The cost snapshot goes into the metadata deliberately: the API decides with the
SAME frozen parameters the README result was measured under. If the business
changes F or phi, the artefacts are regenerated rather than hand-edited.
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
    """Save the three artefacts. `cost_cfg` is duck-typed on (F, m, phi, r)."""
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
    """Load (booster, calibrator, feature_cols, cost_cfg) from a directory.

    `cost_cfg` comes back as SimpleNamespace(F, m, phi, r), the same duck-typed
    interface policy/costs.py consumes.
    """
    import lightgbm as lgb

    src = Path(dir_path)
    for name in (MODEL_FILE, CALIBRATOR_FILE, METADATA_FILE):
        if not (src / name).exists():
            raise FileNotFoundError(f"{name} is missing from {src}. Was save_artifacts run?")

    booster = lgb.Booster(model_file=str(src / MODEL_FILE))
    with open(src / CALIBRATOR_FILE, "rb") as f:
        calibrator = pickle.load(f)
    metadata = json.loads((src / METADATA_FILE).read_text())
    cost_cfg = SimpleNamespace(**metadata["costs"])
    return booster, calibrator, metadata["feature_cols"], cost_cfg
