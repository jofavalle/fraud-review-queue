"""Central configuration for the fraud review queue.

Every business assumption in this project lives here. No magic numbers
elsewhere in the codebase.

This is a requirement, not a style preference: the sensitivity analysis
(docs/design.md §7.2) sweeps these parameters, which is only possible if they
are addressable from one place and passed explicitly into the code that uses
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# This file is at src/fraudq/config.py, so the project root is three parents up.
# Resolving it this way means the code works regardless of the working directory
# it is invoked from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

SECONDS_PER_DAY: int = 86_400


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConfig:
    """Expected cost of each action, for a transaction with calibrated fraud
    probability ``p`` and amount ``a``.

        approve:  p * (a + F)
        block:    (1 - p) * (m * a + phi)
        review:   r

    These are assumptions, not measurements. Every one is swept in the
    sensitivity analysis; see ``SENSITIVITY_RANGES``.
    """

    #: Fixed fee charged to the merchant by the card network when a fraudulent
    #: transaction is reversed. Industry range is roughly $15-25. This is
    #: charged *on top of* losing the transaction amount itself.
    chargeback_fee: float = 20.0

    #: Gross margin on a legitimate sale. Blocking a legitimate transaction
    #: costs the merchant the profit that sale would have made, not the full
    #: price. Typical e-commerce margin.
    gross_margin: float = 0.25

    #: Cost of blocking a legitimate customer: support contact, plus the
    #: probability that the customer does not return.
    friction_cost: float = 10.0

    #: Cost of one manual review. An analyst at ~$25/hr fully loaded, spending
    #: ~5 minutes on a case.
    review_cost: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.gross_margin < 1.0:
            raise ValueError(f"gross_margin must be in (0, 1), got {self.gross_margin}")
        for name in ("chargeback_fee", "friction_cost", "review_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    # Aliases matching the notation in docs/design.md, so that the policy code
    # reads the same as the derivations it implements.
    @property
    def F(self) -> float:  # noqa: N802
        return self.chargeback_fee

    @property
    def m(self) -> float:
        return self.gross_margin

    @property
    def phi(self) -> float:
        return self.friction_cost

    @property
    def r(self) -> float:
        return self.review_cost


#: Plausible ranges for the tornado plot (docs/design.md §7.2).
SENSITIVITY_RANGES: dict[str, tuple[float, float]] = {
    "chargeback_fee": (10.0, 40.0),
    "gross_margin": (0.15, 0.40),
    "friction_cost": (2.0, 30.0),
    "review_cost": (1.0, 5.0),
}


# --------------------------------------------------------------------------
# Temporal splits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Strictly temporal partitioning by ``TransactionDT``. Never a random
    split. See docs/design.md §4.
    """

    #: Last day (inclusive) of the model training window.
    train_end_day: int = 119

    #: Days between train and calibration that are used for nothing.
    #:
    #: Fraud labels arrive late: a chargeback takes weeks or months to
    #: materialise. When a model is retrained in production, labels for the
    #: most recent weeks do not yet exist. A contiguous train/test split
    #: implicitly assumes instantaneous label availability. This embargo
    #: simulates the real gap.
    embargo_days: int = 10

    #: Last day (inclusive) of the calibration and policy-tuning window. The
    #: calibrator is fitted here, on data the model has never seen. Thresholds
    #: and capacity are explored here.
    calib_end_day: int = 155

    #: Expanding-window CV folds inside the training partition. Expanding
    #: window, not KFold: see docs/design.md §4.4.
    n_cv_folds: int = 4

    @property
    def calib_start_day(self) -> int:
        return self.train_end_day + self.embargo_days + 1

    @property
    def test_start_day(self) -> int:
        return self.calib_end_day + 1

    def __post_init__(self) -> None:
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be non-negative")
        if self.calib_start_day > self.calib_end_day:
            raise ValueError(
                "Embargo overruns the calibration window: "
                f"calibration starts on day {self.calib_start_day} "
                f"but ends on day {self.calib_end_day}"
            )


# --------------------------------------------------------------------------
# Review policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    """The capacity-constrained review queue.

    Capacity is a fraction of daily transaction volume rather than an absolute
    headcount, so the policy transfers across volume regimes.
    """

    #: Fraction of each day's transactions the analyst team can review.
    daily_capacity_pct: float = 0.01

    #: Capacities swept when producing the capacity-vs-saving curve.
    capacity_sweep: tuple[float, ...] = (0.002, 0.005, 0.01, 0.02, 0.05)

    def __post_init__(self) -> None:
        if not 0.0 < self.daily_capacity_pct <= 1.0:
            raise ValueError(f"daily_capacity_pct must be in (0, 1], got {self.daily_capacity_pct}")

    def daily_capacity(self, n_transactions_today: int) -> int:
        """Reviews available today, given the day's transaction volume."""
        return int(n_transactions_today * self.daily_capacity_pct)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """LightGBM configuration.

    ``scale_pos_weight``, ``is_unbalance`` and any form of resampling are
    deliberately absent. They distort predicted probabilities, and the decision
    layer requires ``p`` to be a real probability. Class imbalance is addressed
    by the choice of metric, not by altering the training distribution.

    See docs/design.md §6.2.
    """

    objective: str = "binary"
    metric: str = "average_precision"
    n_estimators: int = 2_000
    learning_rate: float = 0.03
    num_leaves: int = 64
    min_child_samples: int = 100

    #: Row subsampling. LightGBM ignores ``subsample`` entirely unless
    #: ``subsample_freq`` is greater than zero, and issues no warning when it
    #: does. The two must be set together.
    subsample: float = 0.8
    subsample_freq: int = 1

    colsample_bytree: float = 0.6
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 100
    random_state: int = 42
    n_jobs: int = -1

    #: "sigmoid" (Platt) or "isotonic". Both are fitted and compared.
    calibration_method: str = "isotonic"


# --------------------------------------------------------------------------
# Top-level bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """The full configuration, passed explicitly into the code that uses it."""

    cost: CostConfig = field(default_factory=CostConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


#: Default configuration. Variants are built with ``dataclasses.replace``,
#: which is why every config class is frozen.
CONFIG = Config()
