#!/usr/bin/env python
"""
Evaluation metrics and statistical inference.

  · PCC, Kendall τ_b
  · FEP: per-series correlation → mean weighted by number of ligands
    (wmPCC / wmKτ)
  · Confidence intervals: BCa confidence interval (bias-corrected and
    accelerated bootstrap), n=10,000
      FEP  resampling unit = series (ligands within a series kept intact)
      OOD  resampling unit = sample
  · Between-config comparison: BCa CI of the paired difference (the paired
    difference is explicitly required; overlap of the two intervals is not
    used as the criterion)
  · Scaling law: log-linear fit + R²

BCa implementation notes (consistent with Efron & Tibshirani 1993):
  z0 = Φ⁻¹( #{θ*<θ̂} / B )
  a  = Σ(θ̄_(·) − θ_(i))³ / (6·[Σ(θ̄_(·) − θ_(i))²]^{3/2}), where θ_(i) are the
       leave-one-out jackknife values
  α₁ = Φ( z0 + (z0+z_{α/2}) / (1 − a(z0+z_{α/2})) ), α₂ likewise with z_{1−α/2}
  CI = ( bootstrap quantile α₁, bootstrap quantile α₂ )
Degeneracy guard: when all θ* are equal, or the jackknife variance is 0, fall
back to the percentile method and flag it in the result.
"""
import numpy as np
from scipy import stats

N_BOOT = 10000


# ───────────────────────────── Basic metrics ─────────────────────────────

def pcc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def ktau(y, p):
    """Kendall τ_b (scipy default variant='b', corrected for ties)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3:
        return np.nan
    t = stats.kendalltau(y, p, variant='b')
    return float(t.correlation)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


# ─────────────────── FEP: per-series → weighted mean ───────────────────

def series_stats(series_ids, y, p):
    """Return list[(n_i, r_i, tau_i)], computed per series."""
    series_ids = np.asarray(series_ids)
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = []
    for s in pd_unique(series_ids):
        m = series_ids == s
        out.append((int(m.sum()), pcc(y[m], p[m]), ktau(y[m], p[m]), str(s)))
    return out


def pd_unique(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def weighted_mean(units, key):
    """units: list[(n, r, tau, name)]; key ∈ {1,2} selects r or tau. Series
    with NaN are dropped and the remaining weights renormalized."""
    n = np.array([u[0] for u in units], float)
    v = np.array([u[key] for u in units], float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    return float(np.sum(n[ok] * v[ok]) / np.sum(n[ok]))


# ───────────────────────────── BCa Bootstrap ─────────────────────────────

def bca_ci(units, stat_fn, n_boot=N_BOOT, alpha=0.05, seed=0):
    """
    units    : list of independent resamplable units (series tuples for FEP,
               sample indices for OOD)
    stat_fn  : takes a sublist of units → scalar statistic
    Returns dict(point, lo, hi, method)
    """
    units = list(units)
    n = len(units)
    theta = stat_fn(units)
    if not np.isfinite(theta) or n < 3:
        return {'point': float(theta), 'lo': np.nan, 'hi': np.nan, 'method': 'undefined'}

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot = np.array([stat_fn([units[i] for i in row]) for row in idx], float)
    boot = boot[np.isfinite(boot)]
    if len(boot) < 100 or np.std(boot) < 1e-12:
        return {'point': float(theta), 'lo': float(np.nanmin(boot)) if len(boot) else np.nan,
                'hi': float(np.nanmax(boot)) if len(boot) else np.nan,
                'method': 'degenerate'}

    # Bias correction z0
    frac = float(np.mean(boot < theta))
    frac = min(max(frac, 1.0 / (2 * len(boot))), 1 - 1.0 / (2 * len(boot)))
    z0 = stats.norm.ppf(frac)

    # Acceleration a (leave-one-out jackknife)
    jack = np.array([stat_fn([units[i] for i in range(n) if i != k]) for k in range(n)], float)
    jack = jack[np.isfinite(jack)]
    d = jack.mean() - jack
    den = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = float(np.sum(d ** 3) / den) if den > 1e-30 else 0.0

    out = {}
    for tag, q in (('lo', alpha / 2), ('hi', 1 - alpha / 2)):
        z = stats.norm.ppf(q)
        adj = z0 + (z0 + z) / max(1e-12, (1 - a * (z0 + z)))
        out[tag] = float(np.percentile(boot, 100 * stats.norm.cdf(adj)))
    method = 'BCa'
    if out['lo'] > out['hi']:
        out['lo'], out['hi'] = out['hi'], out['lo']
        method = 'BCa(swapped)'
    return {'point': float(theta), 'lo': out['lo'], 'hi': out['hi'], 'method': method}


# ───────────────── Wrappers for the two evaluation sets ─────────────────

def fep_metrics(series_ids, y, p, n_boot=N_BOOT, seed=0):
    """wmPCC / wmKτ + series-level BCa CI."""
    units = series_stats(series_ids, y, p)
    res = {'n_series': len(units), 'n_ligands': int(sum(u[0] for u in units)),
           'per_series': [{'series': u[3], 'n': u[0], 'pcc': u[1], 'ktau': u[2]}
                          for u in units]}
    res['wmPCC'] = bca_ci(units, lambda U: weighted_mean(U, 1), n_boot, seed=seed)
    res['wmKtau'] = bca_ci(units, lambda U: weighted_mean(U, 2), n_boot, seed=seed + 1)
    return res


def ood_metrics(y, p, n_boot=N_BOOT, seed=0):
    """PCC / Kendall τ_b / RMSE / MAE + sample-level BCa CI."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    units = list(zip(y, p))
    f = {'PCC': lambda U: pcc([u[0] for u in U], [u[1] for u in U]),
         'Ktau': lambda U: ktau([u[0] for u in U], [u[1] for u in U]),
         'RMSE': lambda U: rmse([u[0] for u in U], [u[1] for u in U]),
         'MAE': lambda U: mae([u[0] for u in U], [u[1] for u in U])}
    return {'n': len(y), **{k: bca_ci(units, fn, n_boot, seed=seed + i)
                            for i, (k, fn) in enumerate(f.items())}}


# ───────────────── Paired comparison between configs ─────────────────

def paired_diff_ci(units_a, units_b, stat_fn, n_boot=N_BOOT, alpha=0.05, seed=0):
    """
    CI of the difference between two configs on **the same set of units**:
    both configs use the same resampled indices, which preserves the pairing.
    units_a[i] and units_b[i] must correspond to the same unit.
    """
    assert len(units_a) == len(units_b)
    pair = list(zip(units_a, units_b))
    return bca_ci(pair,
                  lambda P: stat_fn([p[0] for p in P]) - stat_fn([p[1] for p in P]),
                  n_boot, alpha, seed)


def scaling_fit(sizes, values):
    """Log-linear fit value = k·log10(size) + b; returns k, b, R²."""
    s = np.log10(np.asarray(sizes, float))
    v = np.asarray(values, float)
    ok = np.isfinite(s) & np.isfinite(v)
    if ok.sum() < 3:
        return {'k': np.nan, 'b': np.nan, 'r2': np.nan, 'n': int(ok.sum())}
    k, b = np.polyfit(s[ok], v[ok], 1)
    pred = k * s[ok] + b
    ss_res = np.sum((v[ok] - pred) ** 2)
    ss_tot = np.sum((v[ok] - v[ok].mean()) ** 2)
    return {'k': float(k), 'b': float(b),
            'r2': float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan,
            'n': int(ok.sum())}
