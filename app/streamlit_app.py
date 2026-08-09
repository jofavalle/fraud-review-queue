"""The queue simulator (Streamlit).

    streamlit run app/streamlit_app.py

Controls: K, F, m, phi, r. Outputs: loss per $1,000, frauds caught and missed,
legitimate transactions blocked, THE SAVING against the score-ranked queue, and
the map of decision regions in the (p, amount) plane.

It consumes the PERSISTED scoring (reports/scored_test.parquet): the
predictions are frozen and the sliders move ONLY the decision layer. Exploring
here is legitimate. The number in the README comes from the single evaluation
on test with the base parameters of config.py, not from this demo.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from fraudq.evaluate.policies import (
    compare_policies,
    fit_single_threshold,
    headline_savings,
)
from fraudq.policy.costs import cost_approve, cost_block, value_of_review

DEFAULT_DATA = "reports/scored_test.parquet"

st.set_page_config(page_title="Fraud review queue simulator", layout="wide")
st.title("Fraud review queue: a capacity-constrained simulator")
st.caption(
    "Predictions are frozen, scored once on the test partition. Sliders move "
    "only the decision layer: cost assumptions and review capacity."
)


@st.cache_data(show_spinner="Loading scored transactions…")
def load_scored(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = {"day", "TransactionAmt", "isFraud", "p"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"The parquet is missing {sorted(missing)}. Is it the scored one?")
    return df


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    data_path = st.text_input("Scored parquet", value=DEFAULT_DATA)

    st.header("Review capacity")
    k_pct = st.slider("Daily capacity K (% of volume)", 0.1, 5.0, 1.0, 0.1) / 100

    st.header("Cost assumptions")
    F = st.slider("Chargeback fee F ($)", 5.0, 60.0, 20.0, 1.0)
    m = st.slider("Gross margin m", 0.05, 0.60, 0.25, 0.01)
    phi = st.slider("Friction cost φ ($)", 0.0, 40.0, 10.0, 1.0)
    r = st.slider("Review cost r ($)", 0.5, 10.0, 2.0, 0.5)

cfg = SimpleNamespace(F=F, m=m, phi=phi, r=r)

if not Path(data_path).exists():
    st.warning(
        f"`{data_path}` not found. Run `python -m fraudq.pipeline`, which "
        "persists the scoring. The demo needs that file."
    )
    st.stop()

df = load_scored(data_path)


# ---------------------------------------------------------------- simulation
@st.cache_data(show_spinner="Simulating four policies…")
def run(F: float, m: float, phi: float, r: float, k_pct: float) -> tuple:
    cfg = SimpleNamespace(F=F, m=m, phi=phi, r=r)
    t = fit_single_threshold(df, cfg)
    comparison = compare_policies(df, cfg, k_pct, t)
    return comparison, headline_savings(comparison), t


comparison, savings, threshold = run(F, m, phi, r, k_pct)
by_value = comparison.loc["topk_by_value"]
by_score = comparison.loc["topk_by_score"]

# ------------------------------------------------------------ the headline number
st.subheader("What score-ranking leaves on the table")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Savings vs. score-ranked queue",
    f"${savings['savings_total']:,.0f}",
    delta=f"${savings['savings_per_1k']:.2f} per $1,000",
)
c2.metric("Loss per $1,000 (value-ranked)", f"${by_value['cost_per_1k']:.2f}")
c3.metric(
    "Frauds caught",
    f"{int(by_value['frauds_caught']):,}",
    delta=int(by_value["frauds_caught"] - by_score["frauds_caught"]),
)
c4.metric(
    "Legit blocked",
    f"{int(by_value['legit_blocked']):,}",
    delta=int(by_value["legit_blocked"] - by_score["legit_blocked"]),
    delta_color="inverse",
)

st.dataframe(
    comparison.style.format(
        {
            "total_cost": "${:,.0f}",
            "cost_per_1k": "${:.2f}",
            "utilization": "{:.1%}",
            "frauds_caught": "{:.0f}",
            "frauds_missed": "{:.0f}",
            "legit_blocked": "{:.0f}",
            "reviews": "{:.0f}",
            "capacity": "{:.0f}",
        }
    ),
    use_container_width=True,
)
st.caption(
    f"Single-threshold policy refit on this data at t = {threshold:.3f} for "
    "each parameter change (it is the naive competitor, so it gets to adapt too)."
)

# --------------------------------------------- decision regions in (p, amount)
st.subheader("Decision regions in the (probability, amount) plane")

col_fig, col_txt = st.columns([2, 1])
with col_fig:
    import matplotlib.pyplot as plt

    p_grid = np.linspace(0.001, 0.999, 300)
    a_grid = np.geomspace(5, 2000, 300)
    P, A = np.meshgrid(p_grid, a_grid)

    approve_c = cost_approve(P, A, cfg)
    block_c = cost_block(P, A, cfg)
    v = value_of_review(P, A, cfg)
    # 0=approve, 1=review (V>0, no capacity constraint, §2.4), 2=block
    region = np.where(v > 0, 1, np.where(approve_c <= block_c, 0, 2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.contourf(
        P,
        A,
        region,
        levels=[-0.5, 0.5, 1.5, 2.5],
        alpha=0.35,
        colors=["#2a9d8f", "#e9c46a", "#e76f51"],
    )
    ax.set_yscale("log")
    ax.set_xlabel("Calibrated fraud probability p")
    ax.set_ylabel("Transaction amount ($, log)")
    ax.text(0.03, a_grid[5], "approve", fontsize=9)
    ax.text(0.45, a_grid[150], "review (V > 0)", fontsize=9)
    ax.text(0.9, a_grid[5], "block", fontsize=9, ha="center")
    st.pyplot(fig, clear_figure=True)

with col_txt:
    st.markdown(
        "- The boundaries **depend on the amount**: a single threshold on the "
        "score is structurally the wrong policy (§2.4).\n"
        "- The yellow region is where a human adds something; under finite "
        "capacity, the queue takes the **highest V** inside it.\n"
        "- Move phi or r and watch the review region breathe. That is the "
        "sensitivity analysis, live."
    )

# ----------------------------------------------------------------- by day
st.subheader("Daily realised cost: value-ranked against score-ranked")
# compare_policies returns the summary; for the daily series the two queues
# that matter are simulated again.
from fraudq.evaluate.policies import actions_topk_by_score, actions_topk_by_value  # noqa: E402
from fraudq.policy.simulate import simulate_queue  # noqa: E402


@st.cache_data(show_spinner=False)
def daily_series(F: float, m: float, phi: float, r: float, k_pct: float) -> pd.DataFrame:
    cfg = SimpleNamespace(F=F, m=m, phi=phi, r=r)
    a = simulate_queue(df, actions_topk_by_score, cfg, k_pct).per_day
    b = simulate_queue(df, actions_topk_by_value, cfg, k_pct).per_day
    out = (
        a[["day", "cost"]]
        .rename(columns={"cost": "score-ranked"})
        .merge(b[["day", "cost"]].rename(columns={"cost": "value-ranked"}), on="day")
    )
    return out.set_index("day")


st.line_chart(daily_series(F, m, phi, r, k_pct))
