# Biotech Portfolio Decision Engine

A Bayesian engine for go/kill/repurpose decisions across a small biopharma portfolio. Built to explore how adaptive probability-of-technical-success (PoTS) models can sharpen capital allocation decisions in early-stage drug development.

**View the notebook**: [portfolio_demo.ipynb](https://github.com/ChristopherSNelson/PoTS_decisionTool/blob/main/portfolio_demo.ipynb)

---

## What it does

Each asset in the portfolio has a Beta-distributed prior over PoTS, updated as interim trial data comes in. The engine then makes phase-dependent decisions:

- **Go** - posterior mean exceeds the phase threshold; advance and commit capital
- **Continue** - uncertainty zone; gather more data before deciding
- **Kill** - posterior mean falls below kill threshold; stop and reallocate
- **Repurpose** - biology suggests a better indication; pivot rather than kill

Phase III assets face a stricter bar than Phase I - more capital at risk means higher evidence requirements.

![Portfolio Decisions](plots/portfolio_decisions.png)

---

## Key features

- **Bayesian updating** - Beta-Binomial posteriors combine historical priors with interim readouts
- **Correlated failure propagation** - when one asset fails, correlated assets absorb virtual failures weighted by a shared-biology correlation matrix
- **Biology-driven repurposing** - Continue-zone assets pivot to an alternative indication if mechanistic similarity is strong (>= 0.65); repurpose PoTS = original PoTS * similarity score
- **eNPV capital allocation** - greedy allocation under a fixed budget, ranked by risk-adjusted expected value
- **Monte Carlo simulation** - 10,000 portfolio simulations via Gaussian copula to respect asset correlations

---

## Portfolio snapshot (simulated data)

| Asset | Phase | Indication | PoTS | Decision |
|-------|-------|------------|------|----------|
| A | II | SLE | 25.0% | Repurpose (Lupus Nephritis) |
| B | II | Lupus Nephritis | 33.3% | Go |
| C | I | Type 2 Diabetes | 16.7% | Go |
| D | III | NAFLD | 41.7% | Go |
| E | I | Rare Haem (PNH) | 25.0% | Go |

*All data is simulated. This is a methodological demonstration, not a clinical claim.*

---

## Run it

```bash
pip install numpy pandas matplotlib scipy ipywidgets
jupyter notebook portfolio_demo.ipynb
```
