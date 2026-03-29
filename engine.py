"""Computation engine for the PoTS decision tool. No UI dependencies."""

import numpy as np
import pandas as pd
from scipy.stats import beta
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

        # Calculate Repurpose PoTS with a Safety De-risking Bonus (from literature)
        # Repurposed drugs often have ~2-3x higher overall success because safety is de-risked.
        SAFETY_BONUS = 1.4  # Conservative 40% boost for established safety/PK
        
        candidates = repurpose_map[repurpose_map['asset'] == asset].copy()
        best_candidate = None
        if len(candidates) > 0:
            # We apply the similarity score but also the safety bonus
            candidates['repurpose_pots'] = np.minimum(0.95, mean[i] * candidates['similarity_score'] * SAFETY_BONUS)
            best_candidate = candidates.loc[candidates['repurpose_pots'].idxmax()]

        # DECISION LOGIC:
        # Go: Mean exceeds the Go threshold - advance to next phase
        if mean[i] > go_t:
            dec, alt_ind, alt_pots, reason = 'Go', '-', None, '-'

        # Kill: Posterior mean falls below the Kill threshold
        elif mean[i] < kill_t:
            if best_candidate is not None and best_candidate['repurpose_pots'] > mean[i] * 1.2:
                dec = 'Repurpose'
                alt_ind = best_candidate['alt_indication']
                alt_pots = best_candidate['repurpose_pots']
                reason = best_candidate['reason']
            else:
                dec, alt_ind, alt_pots, reason = 'Kill', '-', None, '-'

        # Continue: between thresholds - keep running, gather more data
        else:
            if best_candidate is not None and best_candidate['repurpose_pots'] > mean[i] * 1.2:
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
                       cost: np.ndarray, budget: float):
    """Greedy eNPV-ranked allocation under budget."""
    enpv = pots_mean * revenue - cost
    order = np.argsort(-enpv)
    funded = np.zeros(len(pots_mean), dtype=bool)
    remaining = budget
    for idx in order:
        if enpv[idx] > 0 and cost[idx] <= remaining:
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
