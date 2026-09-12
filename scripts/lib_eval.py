#!/usr/bin/env python
"""
评估指标与统计推断（PLAN §6.2 / §6.3 / §6.5）。

  · PCC、Kendall τ_b
  · FEP：逐系列相关系数 → 以配体数加权平均（wmPCC / wmKτ）
  · 置信区间：BCa Bootstrap（偏差校正 + 加速），n=10,000
      FEP  重采样单位 = 系列（保持系列内配体完整）
      OOD  重采样单位 = 样本
  · 配置间比较：配对差值的 BCa CI（PLAN §6.5 明确要求配对差值，不用区间重叠判断）
  · Scaling Law：log-linear 拟合 + R²

BCa 实现要点（与 Efron & Tibshirani 1993 一致）：
  z0 = Φ⁻¹( #{θ*<θ̂} / B )
  a  = Σ(θ̄_(·) − θ_(i))³ / (6·[Σ(θ̄_(·) − θ_(i))²]^{3/2})，θ_(i) 为留一 jackknife 值
  α₁ = Φ( z0 + (z0+z_{α/2}) / (1 − a(z0+z_{α/2})) )，α₂ 同理用 z_{1−α/2}
  CI = ( bootstrap 分位数 α₁, bootstrap 分位数 α₂ )
退化保护：θ* 全等、或 jackknife 方差为 0 时，回退到百分位法并在结果中标注。
"""
import numpy as np
from scipy import stats

N_BOOT = 10000


# ───────────────────────────── 基础指标 ─────────────────────────────

def pcc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def ktau(y, p):
    """Kendall τ_b（scipy 默认 variant='b'，对并列值有校正）。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3:
        return np.nan
    t = stats.kendalltau(y, p, variant='b')
    return float(t.correlation)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


# ─────────────────────── FEP：系列级 → 加权平均 ───────────────────────

def series_stats(series_ids, y, p):
    """返回 list[(n_i, r_i, tau_i)]，逐系列计算。"""
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
    """units: list[(n, r, tau, name)]；key ∈ {1,2} 选 r 或 tau。NaN 系列剔除并按剩余权重归一。"""
    n = np.array([u[0] for u in units], float)
    v = np.array([u[key] for u in units], float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    return float(np.sum(n[ok] * v[ok]) / np.sum(n[ok]))


# ───────────────────────────── BCa Bootstrap ─────────────────────────────

def bca_ci(units, stat_fn, n_boot=N_BOOT, alpha=0.05, seed=0):
    """
    units    : 可重采样的独立单位列表（FEP 传系列元组，OOD 传样本索引）
    stat_fn  : 接受 units 的子列表 → 标量统计量
    返回 dict(point, lo, hi, method)
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

    # 偏差校正 z0
    frac = float(np.mean(boot < theta))
    frac = min(max(frac, 1.0 / (2 * len(boot))), 1 - 1.0 / (2 * len(boot)))
    z0 = stats.norm.ppf(frac)

    # 加速 a（留一 jackknife）
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


# ───────────────── 面向两个评测集的封装 ─────────────────

def fep_metrics(series_ids, y, p, n_boot=N_BOOT, seed=0):
    """wmPCC / wmKτ + 系列级 BCa CI。"""
    units = series_stats(series_ids, y, p)
    res = {'n_series': len(units), 'n_ligands': int(sum(u[0] for u in units)),
           'per_series': [{'series': u[3], 'n': u[0], 'pcc': u[1], 'ktau': u[2]}
                          for u in units]}
    res['wmPCC'] = bca_ci(units, lambda U: weighted_mean(U, 1), n_boot, seed=seed)
    res['wmKtau'] = bca_ci(units, lambda U: weighted_mean(U, 2), n_boot, seed=seed + 1)
    return res


def ood_metrics(y, p, n_boot=N_BOOT, seed=0):
    """PCC / Kendall τ_b / RMSE / MAE + 样本级 BCa CI。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    units = list(zip(y, p))
    f = {'PCC': lambda U: pcc([u[0] for u in U], [u[1] for u in U]),
         'Ktau': lambda U: ktau([u[0] for u in U], [u[1] for u in U]),
         'RMSE': lambda U: rmse([u[0] for u in U], [u[1] for u in U]),
         'MAE': lambda U: mae([u[0] for u in U], [u[1] for u in U])}
    return {'n': len(y), **{k: bca_ci(units, fn, n_boot, seed=seed + i)
                            for i, (k, fn) in enumerate(f.items())}}


# ───────────────── 配置间配对比较（PLAN §6.5） ─────────────────

def paired_diff_ci(units_a, units_b, stat_fn, n_boot=N_BOOT, alpha=0.05, seed=0):
    """
    两配置在**同一批单位**上的差值 CI：重采样时两配置用同一组下标，
    保证配对性。units_a[i] 与 units_b[i] 必须对应同一个单位。
    """
    assert len(units_a) == len(units_b)
    pair = list(zip(units_a, units_b))
    return bca_ci(pair,
                  lambda P: stat_fn([p[0] for p in P]) - stat_fn([p[1] for p in P]),
                  n_boot, alpha, seed)


def scaling_fit(sizes, values):
    """log-linear 拟合 value = k·log10(size) + b，返回 k、b、R²。"""
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
