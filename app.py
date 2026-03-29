"""Streamlit app for the PoTS Decision Engine."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta as beta_dist

from engine import (
    load_portfolio, compute_pots, apply_interim, propagate_failure,
    smart_decision, capital_allocation, monte_carlo_portfolio,
    build_repurpose_table, PHASE_THRESHOLDS,
)

st.set_page_config(page_title="PoTS Decision Engine", layout="wide")
st.title("Biotech Portfolio PoTS Decision Engine")
st.caption("A Bayesian, portfolio-level decision engine for go/kill/repurpose decisions")

# --- Load data ---
df, repurpose_map, correlation = load_portfolio("data/assets.csv", "data/repurpose_map.csv")
assets = df["asset"].tolist()
n = len(assets)
indications = df["indication"].tolist()
phases = df["phase"].tolist()
pathways = df["target_pathway"].tolist()
alpha_prior = df["prior_successes"].values + 1.0
beta_prior = df["prior_failures"].values + 1.0
revenue = df["estimated_revenue_m"].values.astype(float)
cost = df["estimated_cost_m"].values.astype(float)

# --- Sidebar controls ---
st.sidebar.header("Controls")
thresh_scale = st.sidebar.slider("Threshold scale", 0.5, 2.0, 1.0, 0.1,
                                  help="Multiplier on phase-dependent kill/go thresholds")
budget = st.sidebar.slider("Budget ($M)", 100, 800, 350, 50)
use_interim = st.sidebar.toggle("Include interim data", value=True)

st.sidebar.divider()
st.sidebar.header("Failure Propagation")
failed_asset = st.sidebar.selectbox("Asset that fails", assets, index=3)
penalty = st.sidebar.slider("Failure penalty", 1, 10, 5)
do_propagate = st.sidebar.toggle("Simulate failure event", value=False)

st.sidebar.divider()
st.sidebar.header("Monte Carlo")
n_sims = st.sidebar.slider("Simulations", 1000, 50000, 10000, 1000)

# --- Interim data ---
INTERIM = {
    "Asset A": (1, 3),
    "Asset B": (2, 4),
    "Asset C": (0, 2),
    "Asset D": (0, 3),
    "Asset E": (1, 2),
}

# --- Compute ---
if use_interim:
    alpha_post, beta_post = apply_interim(alpha_prior, beta_prior, assets, INTERIM)
else:
    alpha_post, beta_post = alpha_prior.copy(), beta_prior.copy()

if do_propagate:
    fidx = assets.index(failed_asset)
    alpha_post, beta_post = propagate_failure(fidx, penalty, alpha_post, beta_post, correlation)

mean_val, lo_val, hi_val = compute_pots(alpha_post, beta_post)
mean_prior, lo_prior, hi_prior = compute_pots(alpha_prior, beta_prior)

decision_df = smart_decision(assets, phases, mean_val, lo_val, hi_val,
                              indications, repurpose_map, thresh_scale)
enpv, funded, remaining = capital_allocation(mean_val, revenue, cost, budget)

# ============================================================
# Section 1: Portfolio overview
# ============================================================
st.header("1. Portfolio Overview")
col1, col2 = st.columns([3, 2])
with col1:
    st.dataframe(df[["asset", "phase", "indication", "target_pathway",
                      "estimated_revenue_m", "estimated_cost_m"]], hide_index=True)
with col2:
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation, x=assets, y=assets,
        colorscale="Blues", zmin=0, zmax=1,
        text=np.round(correlation, 2), texttemplate="%{text}",
    ))
    fig_corr.update_layout(title="Correlation Matrix", height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

with st.expander("Repurpose Candidates"):
    st.dataframe(repurpose_map, hide_index=True)

# ============================================================
# Section 2: Prior distributions
# ============================================================
st.header("2. Prior PoTS Distributions")
fig_prior = make_subplots(rows=1, cols=n, subplot_titles=[f"{a}\n{ind}" for a, ind in zip(assets, indications)])
x = np.linspace(0, 1, 200)
for i in range(n):
    pdf = beta_dist.pdf(x, alpha_prior[i], beta_prior[i])
    fig_prior.add_trace(go.Scatter(x=x, y=pdf, fill="tozeroy", fillcolor="rgba(70,130,180,0.4)",
                                    line=dict(color="steelblue"), showlegend=False), row=1, col=i+1)
    fig_prior.add_vline(x=mean_prior[i], line_dash="dash", line_color="red", row=1, col=i+1)
fig_prior.update_layout(height=250, margin=dict(t=40, b=20))
st.plotly_chart(fig_prior, use_container_width=True)

# ============================================================
# Section 3: Prior vs Posterior
# ============================================================
if use_interim:
    st.header("3. Bayesian Updating After Interim Data")
    fig_update = make_subplots(rows=1, cols=n, subplot_titles=assets)
    for i in range(n):
        pdf_prior = beta_dist.pdf(x, alpha_prior[i], beta_prior[i])
        a_post, b_post = alpha_post[i], beta_post[i]
        if do_propagate:
            # recompute without propagation for the overlay
            a_tmp, b_tmp = apply_interim(alpha_prior, beta_prior, assets, INTERIM)
            pdf_post = beta_dist.pdf(x, a_tmp[i], b_tmp[i])
        else:
            pdf_post = beta_dist.pdf(x, a_post, b_post)
        fig_update.add_trace(go.Scatter(x=x, y=pdf_prior, fill="tozeroy",
                                         fillcolor="rgba(70,130,180,0.3)",
                                         line=dict(color="steelblue"), name="Prior",
                                         showlegend=(i == 0)), row=1, col=i+1)
        fig_update.add_trace(go.Scatter(x=x, y=pdf_post, fill="tozeroy",
                                         fillcolor="rgba(255,165,0,0.3)",
                                         line=dict(color="darkorange"), name="Posterior",
                                         showlegend=(i == 0)), row=1, col=i+1)
    fig_update.update_layout(height=250, margin=dict(t=40, b=20))
    st.plotly_chart(fig_update, use_container_width=True)

# ============================================================
# Section 4: Failure propagation
# ============================================================
if do_propagate:
    st.header("4. Correlated Failure Propagation")
    st.warning(f"Simulating: **{failed_asset}** fails (penalty = {penalty} virtual failures)")
    # Compute pre-propagation values for comparison
    if use_interim:
        a_pre, b_pre = apply_interim(alpha_prior, beta_prior, assets, INTERIM)
    else:
        a_pre, b_pre = alpha_prior.copy(), beta_prior.copy()
    mean_pre, _, _ = compute_pots(a_pre, b_pre)

    fig_prop = go.Figure()
    fig_prop.add_trace(go.Bar(x=assets, y=mean_pre, name="Before failure", marker_color="steelblue"))
    fig_prop.add_trace(go.Bar(x=assets, y=mean_val, name="After propagation", marker_color="#e74c3c",
                               text=[f"{d:+.3f}" for d in mean_val - mean_pre], textposition="outside"))
    fig_prop.update_layout(barmode="group", height=400, yaxis_title="PoTS",
                            title=f"Failure Propagation: {failed_asset} fails")
    st.plotly_chart(fig_prop, use_container_width=True)

# ============================================================
# Section 5: Portfolio decisions
# ============================================================
st.header("5. Portfolio Decisions")

DEC_COLORS = {"Go": "#2ecc71", "Kill": "#e74c3c", "Repurpose": "#f39c12"}

fig_dec = go.Figure()
fig_dec.add_trace(go.Bar(
    x=assets, y=mean_val,
    marker_color=[DEC_COLORS[d] for d in decision_df["Decision"]],
    error_y=dict(type="data", symmetric=False,
                 array=hi_val - mean_val, arrayminus=mean_val - lo_val,
                 color="grey", thickness=1.5),
    text=decision_df["Decision"], textposition="outside",
    textfont=dict(size=12, color=[DEC_COLORS[d] for d in decision_df["Decision"]]),
))

# Phase-specific threshold lines
for i in range(n):
    kt = decision_df["Kill thresh"].iloc[i]
    gt = decision_df["Go thresh"].iloc[i]
    fig_dec.add_shape(type="line", x0=i - 0.4, x1=i + 0.4, y0=kt, y1=kt,
                       line=dict(color="#e74c3c", width=2))
    fig_dec.add_shape(type="line", x0=i - 0.4, x1=i + 0.4, y0=gt, y1=gt,
                       line=dict(color="#2ecc71", width=2))

fig_dec.update_layout(
    height=500, yaxis_title="PoTS (mean + 90% CI)", yaxis_range=[0, 0.9],
    xaxis=dict(ticktext=[f"{a}<br>{ind}<br>({ph})" for a, ind, ph in zip(assets, indications, phases)],
               tickvals=list(range(n))),
    title="Phase-Dependent Thresholds (90% CI)",
    showlegend=False,
)
st.plotly_chart(fig_dec, use_container_width=True)

st.dataframe(decision_df[["Asset", "Phase", "Current Indication", "PoTS", "90% CI",
                            "Kill thresh", "Go thresh", "Decision",
                            "Suggested Repurpose", "Repurpose PoTS", "Reason"]],
             hide_index=True)

# ============================================================
# Section 6: Capital allocation
# ============================================================
st.header("6. Capital Allocation")

c1, c2, c3 = st.columns(3)
c1.metric("Budget", f"${budget}M")
c2.metric("Allocated", f"${budget - remaining:.0f}M")
c3.metric("Remaining", f"${remaining:.0f}M")

order = np.argsort(-enpv)
fig_cap = go.Figure()
fig_cap.add_trace(go.Bar(
    y=[f"{assets[i]} - {indications[i]}" for i in order],
    x=enpv[order],
    orientation="h",
    marker_color=["#2ecc71" if funded[i] else "#cccccc" for i in order],
    text=[f"${enpv[i]:.0f}M{' - FUNDED' if funded[i] else ''}" for i in order],
    textposition=["inside" if enpv[i] > max(enpv) * 0.3 else "outside" for i in order],
    textfont=dict(color=["white" if enpv[i] > max(enpv) * 0.3 else "black" for i in order]),
))
fig_cap.add_vline(x=0, line_color="black", line_width=1)
fig_cap.update_layout(height=350, xaxis_title="Expected NPV ($M)",
                       title=f"Capital Allocation (Budget: ${budget}M)", margin=dict(l=180))
st.plotly_chart(fig_cap, use_container_width=True)

# ============================================================
# Section 7: Monte Carlo
# ============================================================
st.header("7. Monte Carlo Simulation")


@st.cache_data
def cached_mc(alpha_tuple, beta_tuple, corr_tuple, rev_tuple, cost_tuple, ns):
    return monte_carlo_portfolio(
        np.array(alpha_tuple), np.array(beta_tuple),
        np.array(corr_tuple).reshape(int(len(corr_tuple)**0.5), -1),
        np.array(rev_tuple), np.array(cost_tuple), ns)


pots_samples, portfolio_value = cached_mc(
    tuple(alpha_post), tuple(beta_post), tuple(correlation.flatten()),
    tuple(revenue), tuple(cost), n_sims)

median_val = np.median(portfolio_value)
p5 = np.percentile(portfolio_value, 5)
p95 = np.percentile(portfolio_value, 95)

def fmt_money(v):
    neg = v < 0
    av = abs(v)
    s = f"${av/1000:.1f}B" if av >= 1000 else f"${av:.0f}M"
    return f"-{s}" if neg else s

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Median", fmt_money(median_val))
mc2.metric("5th pctile", fmt_money(p5))
mc3.metric("95th pctile", fmt_money(p95))
mc4.metric("P(positive)", f"{(portfolio_value > 0).mean():.0%}")

fig_mc = make_subplots(rows=1, cols=2, subplot_titles=["Portfolio Value Distribution", "PoTS Samples per Asset"])

fig_mc.add_trace(go.Histogram(x=portfolio_value, nbinsx=60, marker_color="steelblue",
                                name="Portfolio value"), row=1, col=1)
fig_mc.add_vline(x=0, line_color="red", line_width=2, row=1, col=1)
fig_mc.add_vline(x=median_val, line_color="orange", line_dash="dash", line_width=2, row=1, col=1)

for i in range(n):
    fig_mc.add_trace(go.Violin(y=pots_samples[:, i], name=assets[i],
                                marker_color="steelblue", showlegend=False), row=1, col=2)

fig_mc.update_layout(height=400, margin=dict(t=40, b=20))
st.plotly_chart(fig_mc, use_container_width=True)

# ============================================================
# Section 8: Repurposing deep dive
# ============================================================
st.header("8. Repurposing Landscape")

repurpose_df = build_repurpose_table(assets, indications, mean_val, repurpose_map)

y_labels = []
y_values = []
y_colors = []
for i, asset in enumerate(assets):
    y_labels.append(f"{asset}: {indications[i]} (original)")
    y_values.append(mean_val[i])
    y_colors.append("steelblue")
    cands = repurpose_df[repurpose_df["Asset"] == asset]
    for _, row in cands.iterrows():
        y_labels.append(f"  -> {row['Alt Indication']} (sim={row['Similarity']:.2f})")
        y_values.append(row["Repurpose PoTS"])
        y_colors.append("#9b59b6" if row["Viable"] else "#cccccc")

fig_rep = go.Figure()
fig_rep.add_trace(go.Bar(y=y_labels, x=y_values, orientation="h", marker_color=y_colors))
fig_rep.add_vline(x=0.15, line_color="#e74c3c", line_dash="dash", annotation_text="Viability")
fig_rep.update_layout(height=500, xaxis_title="PoTS", title="Original vs Alternative Indications",
                       yaxis=dict(autorange="reversed"), margin=dict(l=280))
st.plotly_chart(fig_rep, use_container_width=True)

with st.expander("Full Repurpose Table"):
    st.dataframe(repurpose_df, hide_index=True)
