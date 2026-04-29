"""Computation engine for the PoTS decision tool. No UI dependencies."""

import numpy as np
import pandas as pd
from scipy.stats import beta, betabinom
from scipy.linalg import cholesky
from scipy.stats import norm

PHASE_THRESHOLDS = {
    'Phase I':   {'kill': 0.10, 'go': 0.15},
    'Phase II':  {'kill': 0.10, 'go': 0.25},
    'Phase III': {'kill': 0.15, 'go': 0.40},
}


def load_portfolio(assets_path: str, repurpose_path: str):
    """Load portfolio data and build correlation matrix."""
    df = pd.read_csv(assets_path)
    repurpose_map = pd.read_csv(repurpose_path)

    n = len(df)
    groups = df['correlation_group'].values
    correlation = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if groups[i] == groups[j]:
                correlation[i, j] = correlation[j, i] = 0.6
            else:
                correlation[i, j] = correlation[j, i] = 0.15

    return df, repurpose_map, correlation


def compute_pots(alpha: np.ndarray, beta_param: np.ndarray, ci: float = 0.90):
    """Posterior mean and credible interval."""
    tail = (1 - ci) / 2
    mean = beta.mean(alpha, beta_param)
    lo = beta.ppf(tail, alpha, beta_param)
    hi = beta.ppf(1 - tail, alpha, beta_param)
    return mean, lo, hi


def apply_interim(alpha_prior: np.ndarray, beta_prior: np.ndarray,
                  assets: list[str], interim: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply interim trial data to priors."""
    alpha_post = alpha_prior.copy()
    beta_post = beta_prior.copy()
    for i, asset in enumerate(assets):
        if asset in interim:
            s, t = interim[asset]
            alpha_post[i] += s
            beta_post[i] += (t - s)
    return alpha_post, beta_post


def propagate_failure(failed_idx: int, penalty: float,
                      alpha: np.ndarray, beta_param: np.ndarray,
                      corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Propagate failure through correlation matrix."""
    virtual_failures = corr[failed_idx] * penalty
    return alpha.copy(), beta_param + virtual_failures


def smart_decision(assets: list[str], phases: list[str],
                   mean: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   indications: list[str], repurpose_map: pd.DataFrame,
                   thresh_scale: float = 1.0) -> pd.DataFrame:
    """CI-based decisions with phase-dependent thresholds and repurposing."""
    results = []
    for i, asset in enumerate(assets):
        phase = phases[i]
        kill_t = PHASE_THRESHOLDS[phase]['kill'] * thresh_scale
        go_t = PHASE_THRESHOLDS[phase]['go'] * thresh_scale

        candidates = repurpose_map[repurpose_map['asset'] == asset].copy()
        best_candidate = None
        if len(candidates) > 0:
            candidates['repurpose_pots'] = mean[i] * candidates['similarity_score']
            best_candidate = candidates.loc[candidates['repurpose_pots'].idxmax()]

        # DECISION LOGIC:
        # Go: Mean exceeds the Go threshold - advance to next phase
        if mean[i] > go_t:
            dec, alt_ind, alt_pots, reason = 'Go', '-', None, '-'

        # Kill: Posterior mean below kill threshold - no repurpose rescue
        # (repurpose_pots = mean * similarity <= mean < kill_t, so any alt indication also fails)
        elif mean[i] < kill_t:
            dec, alt_ind, alt_pots, reason = 'Kill', '-', None, '-'

        # Continue: between thresholds - pivot to alt indication if mechanistic overlap is strong
        else:
            if best_candidate is not None and best_candidate['similarity_score'] >= 0.65:
                dec = 'Repurpose'
                alt_ind = best_candidate['alt_indication']
                alt_pots = best_candidate['repurpose_pots']
                reason = best_candidate['reason']
            else:
                dec, alt_ind, alt_pots, reason = 'Continue', '-', None, '-'

        results.append({
            'Asset': asset, 'Phase': phase,
            'Current Indication': indications[i],
            'PoTS': round(mean[i], 3),
            '90% CI': f'[{lo[i]:.3f}, {hi[i]:.3f}]',
            'Kill thresh': round(kill_t, 3),
            'Go thresh': round(go_t, 3),
            'Decision': dec,
            'Suggested Repurpose': alt_ind,
            'Repurpose PoTS': round(alt_pots, 3) if alt_pots else '-',
            'Reason': reason,
        })
    return pd.DataFrame(results)


def capital_allocation(pots_mean: np.ndarray, revenue: np.ndarray,
                       cost: np.ndarray, budget: float,
                       decisions: list | None = None):
    """Greedy eNPV-ranked allocation under budget. Skips Kill decisions if provided."""
    enpv = pots_mean * revenue - cost
    order = np.argsort(-enpv)
    funded = np.zeros(len(pots_mean), dtype=bool)
    remaining = budget
    for idx in order:
        eligible = decisions is None or decisions[idx] in ('Go', 'Continue', 'Repurpose')
        if eligible and enpv[idx] > 0 and cost[idx] <= remaining:
            funded[idx] = True
            remaining -= cost[idx]
    return enpv, funded, remaining


def monte_carlo_portfolio(alpha: np.ndarray, beta_param: np.ndarray,
                          corr: np.ndarray, revenue: np.ndarray,
                          cost: np.ndarray, n_sims: int = 10000):
    """Correlated portfolio simulation via Gaussian copula."""
    n_assets = len(alpha)
    L = cholesky(corr, lower=True)
    z = np.random.default_rng(42).standard_normal((n_sims, n_assets))
    corr_z = z @ L.T
    u = norm.cdf(corr_z)

    pots_samples = np.zeros_like(u)
    for i in range(n_assets):
        pots_samples[:, i] = beta.ppf(u[:, i], alpha[i], beta_param[i])

    rng = np.random.default_rng(123)
    success = (rng.random((n_sims, n_assets)) < pots_samples).astype(float)
    portfolio_value = (success * revenue - cost).sum(axis=1)
    return pots_samples, portfolio_value


def _build_corr_matrix(groups: np.ndarray, within_corr: float,
                        between_corr: float = 0.15) -> np.ndarray:
    n = len(groups)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            val = within_corr if groups[i] == groups[j] else between_corr
            corr[i, j] = corr[j, i] = val
    return corr


def stress_test(
    alpha: np.ndarray,
    beta_param: np.ndarray,
    groups: np.ndarray,
    revenue: np.ndarray,
    cost: np.ndarray,
    n_sims: int = 5000,
) -> dict:
    """Tornado diagram inputs and Bear/Base/Bull scenario results.

    Varies one assumption at a time (tornado) and three in combination
    (scenarios).  Returns raw portfolio value arrays so the caller can
    compute any statistics it needs.
    """
    base_corr = _build_corr_matrix(groups, 0.6)

    def run(corr=None, rev_mult=1.0, cost_mult=1.0):
        c = base_corr if corr is None else corr
        _, pv = monte_carlo_portfolio(
            alpha, beta_param, c, revenue * rev_mult, cost * cost_mult, n_sims)
        return pv

    base_pv = run()

    # Tornado: one factor at a time - return raw arrays so caller picks any stat
    factors = {
        'Within-group correlation\n(0.4 vs 0.8)': (
            run(corr=_build_corr_matrix(groups, 0.4)),
            run(corr=_build_corr_matrix(groups, 0.8)),
        ),
        'Revenue assumption\n(-20% vs +20%)': (
            run(rev_mult=0.80),
            run(rev_mult=1.20),
        ),
        'Development cost\n(-20% vs +20%)': (
            run(cost_mult=1.20),   # cost up = value down
            run(cost_mult=0.80),   # cost down = value up
        ),
    }

    # Scenarios: combine assumptions
    scenarios = {
        'Bear': run(corr=_build_corr_matrix(groups, 0.8), rev_mult=0.80, cost_mult=1.20),
        'Base': base_pv,
        'Bull': run(corr=_build_corr_matrix(groups, 0.4), rev_mult=1.20, cost_mult=0.80),
    }

    return {'base_pv': base_pv,
            'factors': factors, 'scenarios': scenarios}


def build_repurpose_table(assets: list[str], indications: list[str],
                          mean: np.ndarray, repurpose_map: pd.DataFrame) -> pd.DataFrame:
    """Full repurpose analysis for all assets."""
    rows = []
    for i, asset in enumerate(assets):
        candidates = repurpose_map[repurpose_map['asset'] == asset]
        for _, cand in candidates.iterrows():
            new_pots = mean[i] * cand['similarity_score']
            rows.append({
                'Asset': asset,
                'Original Indication': indications[i],
                'Original PoTS': round(mean[i], 3),
                'Alt Indication': cand['alt_indication'],
                'Similarity': cand['similarity_score'],
                'Repurpose PoTS': round(new_pots, 3),
                'Viable': new_pots > 0.15,
                'Reason': cand['reason'],
            })
    return pd.DataFrame(rows)


def value_of_information(
    alpha: np.ndarray,
    beta_param: np.ndarray,
    phases: list[str],
    revenue: np.ndarray,
    cost: np.ndarray,
    sample_sizes: np.ndarray | None = None,
    cost_per_patient: float = 0.0,
) -> dict:
    """Compute EVSI and decision resolution probability per asset.

    For each asset and each candidate additional sample size N, uses the
    Beta-Binomial predictive distribution to enumerate all possible
    outcomes k = 0..N.  For each outcome the posterior is updated and:

      - Decision resolution: does the new mean cross a Go or Kill threshold?
      - EVSI: E_k[max(new_mean*R - C, 0)] - max(current_mean*R - C, 0)

    Returns a dict keyed by asset index, each containing arrays of
    resolution_prob, evsi, go_prob, kill_prob over sample_sizes.
    """
    if sample_sizes is None:
        sample_sizes = np.array([5, 10, 15, 20, 30, 50, 75, 100])

    n_assets = len(alpha)
    results = {}

    for i in range(n_assets):
        a, b = alpha[i], beta_param[i]
        phase = phases[i]
        kill_t = PHASE_THRESHOLDS[phase]['kill']
        go_t = PHASE_THRESHOLDS[phase]['go']

        current_mean = a / (a + b)
        current_best = max(current_mean * revenue[i] - cost[i], 0.0)

        resolution_probs = []
        evsi_values = []
        go_probs = []
        kill_probs = []

        for N in sample_sizes:
            k_values = np.arange(0, N + 1)
            pred_probs = betabinom.pmf(k_values, N, a, b)

            new_means = (a + k_values) / (a + b + N)

            is_go = new_means > go_t
            is_kill = new_means < kill_t

            # Optimal decision value for each possible outcome
            optimal_value = np.maximum(new_means * revenue[i] - cost[i], 0.0)

            resolution_probs.append((pred_probs * (is_go | is_kill)).sum())
            evsi_values.append(max((pred_probs * optimal_value).sum() - current_best, 0.0))
            go_probs.append((pred_probs * is_go).sum())
            kill_probs.append((pred_probs * is_kill).sum())

        evsi_arr = np.array(evsi_values)
        net_evsi = np.maximum(evsi_arr - cost_per_patient * sample_sizes, 0.0)

        results[i] = {
            'resolution_prob': np.array(resolution_probs),
            'evsi': evsi_arr,
            'net_evsi': net_evsi,
            'go_prob': np.array(go_probs),
            'kill_prob': np.array(kill_probs),
            'current_mean': current_mean,
            'kill_t': kill_t,
            'go_t': go_t,
        }

    return results, sample_sizes
