"""Multiple-testing correction and false discovery rate (FDR) estimation.

Implemented
-----------
* Family-wise : Bonferroni, Holm (step-down), Šidák
* FDR         : Benjamini-Hochberg (BH, 1995), Benjamini-Yekutieli (BY, 2001, arbitrary dependence)
* Storey q-values with pi0 estimation (smoother or bootstrap; Storey & Tibshirani 2003, Storey 2004)
* Local FDR (lfdr) on the probit scale (as in the R ``qvalue`` package)
* Hierarchical / two-stage FDR for QTLs (TreeQTL-style; Peterson et al. 2016): Simes per feature -> BH across
  features -> BH within selected features at an adjusted level
* Permutation-based empirical FDR (observed vs. pooled-null statistics)
* Effective number of tests per locus from LD eigenvalues (eigenMT; Davis et al. 2016)
* pi1 replication statistic (1 - pi0)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def _clean(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    return p, ok


def bonferroni(p, n_tests: Optional[int] = None):
    p, ok = _clean(p)
    m = n_tests or ok.sum()
    out = np.full_like(p, np.nan)
    out[ok] = np.minimum(p[ok] * m, 1)
    return out


def sidak(p, n_tests: Optional[int] = None):
    p, ok = _clean(p)
    m = n_tests or ok.sum()
    out = np.full_like(p, np.nan)
    out[ok] = 1 - (1 - p[ok]) ** m
    return out


def holm(p):
    p, ok = _clean(p)
    q = p[ok]; m = len(q)
    o = np.argsort(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[o])
    res = np.empty(m); res[o] = np.minimum(adj, 1)
    out = np.full_like(p, np.nan); out[ok] = res
    return out


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p, ok = _clean(p)
    q = p[ok]; m = len(q)
    if m == 0:
        return p
    o = np.argsort(q)[::-1]
    adj = np.minimum.accumulate(q[o] * m / np.arange(m, 0, -1))
    res = np.empty(m); res[o] = np.minimum(adj, 1)
    out = np.full_like(p, np.nan); out[ok] = res
    return out


def by(p):
    """Benjamini-Yekutieli adjusted p-values (valid under arbitrary dependence)."""
    p, ok = _clean(p)
    m = ok.sum()
    c = np.sum(1.0 / np.arange(1, m + 1))
    out = bh(p) * c
    return np.minimum(out, 1)


# ----------------------------------------------------------------------------------------------------------------
# Storey q-value
# ----------------------------------------------------------------------------------------------------------------
def estimate_pi0(p, method: str = "auto", lambdas=None, n_boot: int = 100, seed: int = 0) -> float:
    """Proportion of true nulls.

    ``method``: "smoother" (Storey & Tibshirani 2003; low-df smooth of pi0(lambda) evaluated at the largest
    lambda), "bootstrap" (Storey 2004), "fixed" (lambda = 0.5; Storey 2002) or "auto" (smoother when there are
    >= 1000 p-values, otherwise the more stable fixed-lambda estimate – the smoother's extrapolation to
    lambda = 0.95 is noisy and tends to underestimate pi0 for a few hundred tests)."""
    p = np.asarray(p, float); p = p[~np.isnan(p)]
    m = len(p)
    if m == 0:
        return 1.0
    if method == "auto":
        method = "smoother" if m >= 1000 else "fixed"
    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.05)
    lambdas = np.asarray(lambdas)
    if method == "fixed" or m < 50:
        return float(np.clip(np.mean(p >= 0.5) / 0.5, 1.0 / m, 1.0))
    pi0 = np.array([np.mean(p >= l) / (1 - l) for l in lambdas])
    if method == "smoother":
        # quadratic least-squares smooth (3 parameters ~ smooth.spline(df = 3) in the qvalue package)
        coef = np.polyfit(lambdas, pi0, 2)
        sm = np.polyval(coef, lambdas)
        est = sm[-1]
    elif method == "bootstrap":
        rng = np.random.default_rng(seed)
        minpi0 = np.quantile(pi0, 0.1)
        mse = np.zeros(len(lambdas))
        for _ in range(n_boot):
            pb = rng.choice(p, m, replace=True)
            pib = np.array([np.mean(pb >= l) / (1 - l) for l in lambdas])
            mse += (pib - minpi0) ** 2
        est = pi0[np.argmin(mse)]
    else:
        raise ValueError(method)
    return float(np.clip(est, 1.0 / m, 1.0))


def qvalue(p, pi0: Optional[float] = None, pi0_method: str = "auto") -> np.ndarray:
    """Storey q-values."""
    p, ok = _clean(p)
    if pi0 is None:
        pi0 = estimate_pi0(p[ok], pi0_method)
    q = p[ok]; m = len(q)
    if m == 0:
        return p
    o = np.argsort(q)[::-1]
    adj = np.minimum.accumulate(pi0 * m * q[o] / np.arange(m, 0, -1))
    res = np.empty(m); res[o] = np.minimum(adj, 1)
    out = np.full_like(p, np.nan); out[ok] = res
    return out


def local_fdr(p, pi0: Optional[float] = None, eps: float = 1e-8) -> np.ndarray:
    """Local FDR via kernel density of probit-transformed p-values (qvalue::lfdr approach)."""
    p, ok = _clean(p)
    x = stats.norm.ppf(np.clip(p[ok], eps, 1 - eps))
    if pi0 is None:
        pi0 = estimate_pi0(p[ok])
    kde = stats.gaussian_kde(x)
    f = kde(x)
    lf = pi0 * stats.norm.pdf(x) / np.maximum(f, 1e-300)
    o = np.argsort(p[ok])
    lf_sorted = np.minimum(np.maximum.accumulate(lf[o]), 1)
    res = np.empty_like(lf); res[o] = lf_sorted
    out = np.full_like(p, np.nan); out[ok] = res
    return out


def pi1(p) -> float:
    """Replication rate pi1 = 1 - pi0 (Storey). Use on p-values of discovery hits looked up in a second dataset."""
    p = np.asarray(p, float); p = p[~np.isnan(p)]
    if len(p) < 10:
        return float("nan")
    return 1.0 - estimate_pi0(p, "auto")


# ----------------------------------------------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------------------------------------------
METHODS = ("bh", "by", "bonferroni", "holm", "sidak", "qvalue", "lfdr")


def adjust(p, method: str = "bh") -> np.ndarray:
    method = method.lower()
    f = {"bh": bh, "fdr_bh": bh, "by": by, "bonferroni": bonferroni, "holm": holm, "sidak": sidak,
         "qvalue": qvalue, "storey": qvalue, "lfdr": local_fdr}[method]
    return f(p)


def add_all_corrections(df: pd.DataFrame, p_col: str = "pval", methods=METHODS, prefix: str = "") -> pd.DataFrame:
    """Append one column per correction method (e.g. ``bh``, ``qvalue``) to a results table."""
    out = df.copy()
    for m in methods:
        try:
            out[f"{prefix}{m}"] = adjust(out[p_col].values, m)
        except Exception:  # lfdr can fail on tiny inputs
            out[f"{prefix}{m}"] = np.nan
    return out


# ----------------------------------------------------------------------------------------------------------------
# QTL-specific procedures
# ----------------------------------------------------------------------------------------------------------------
def simes(p) -> float:
    p = np.sort(np.asarray(p, float)[~np.isnan(p)])
    m = len(p)
    if m == 0:
        return np.nan
    return float(np.min(p * m / np.arange(1, m + 1)))


def hierarchical_fdr(df: pd.DataFrame, group_col: str = "feature_id", p_col: str = "pval",
                     level1: float = 0.05, level2: float = 0.05, group_p: Optional[pd.Series] = None) -> pd.DataFrame:
    """Two-level hierarchical FDR (TreeQTL, Peterson et al. 2016, Bioinformatics 32:i45).

    Level 1: per-feature p (Simes by default, or supplied e.g. beta-approximated permutation p) -> BH across features.
    Level 2: within each selected feature, BH at level ``level2 * R / G`` (R selected features of G tested),
    which controls the expected average proportion of false variants among selected features.
    Adds columns ``group_p``, ``group_q``, ``group_selected``, ``within_q``, ``hier_significant``.
    """
    if group_p is None:
        group_p = df.groupby(group_col)[p_col].apply(simes)
    gq = pd.Series(bh(group_p.values), index=group_p.index)
    sel = gq[gq <= level1].index
    G, R = len(gq), len(sel)
    out = df.copy()
    out["group_p"] = out[group_col].map(group_p)
    out["group_q"] = out[group_col].map(gq)
    out["group_selected"] = out[group_col].isin(sel)
    out["within_q"] = np.nan
    thr = level2 * R / G if G else 0
    for g, idx in out[out.group_selected].groupby(group_col).groups.items():
        out.loc[idx, "within_q"] = bh(out.loc[idx, p_col].values)
    out["hier_significant"] = out.group_selected & (out.within_q <= thr)
    out.attrs["level2_threshold"] = thr
    return out


def permutation_fdr(observed, null, larger_is_more_significant: bool = False) -> np.ndarray:
    """Empirical FDR for each observed statistic using a pooled permutation null.

    FDR(t) = [#null <= t / n_perm_sets] / #observed <= t   (for p-values; flip for test statistics).
    ``null`` may be any array; its size divided by len(observed) defines the number of permutation sets.
    """
    obs = np.asarray(observed, float)
    nul = np.sort(np.asarray(null, float).ravel())
    if larger_is_more_significant:
        obs, nul = -obs, np.sort(-nul)
    B = len(nul) / len(obs)
    o = np.argsort(obs)
    so = obs[o]
    n_null = np.searchsorted(nul, so, side="right") / B
    n_obs = np.arange(1, len(so) + 1)
    fdr = np.minimum(n_null / n_obs, 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    out = np.empty_like(fdr); out[o] = fdr
    return out


def eigenmt_meff(G: np.ndarray, var_thresh: float = 0.995, window: int = 200) -> float:
    """Effective number of independent tests in a locus (eigenMT, Davis et al. 2016 AJHG 98:216).

    ``G`` = variants x samples. For > ``window`` variants, Meff is summed over consecutive windows."""
    m = G.shape[0]
    if m <= 1:
        return float(m)
    total = 0.0
    for s in range(0, m, window):
        X = G[s:s + window]
        X = X[X.std(1) > 0]
        if X.shape[0] <= 1:
            total += X.shape[0]
            continue
        C = np.corrcoef(X)
        ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
        cum = np.cumsum(ev) / ev.sum()
        total += int(np.searchsorted(cum, var_thresh) + 1)
    return float(total)


def fdr_summary(p, alphas=(0.01, 0.05, 0.1)) -> pd.DataFrame:
    """Number of discoveries at several thresholds for every method (handy for reports)."""
    p = np.asarray(p, float)
    rows = []
    for m in METHODS:
        try:
            a = adjust(p, m)
        except Exception:
            continue
        rows.append({"method": m, **{f"n_at_{al}": int(np.nansum(a <= al)) for al in alphas}})
    rows.append({"method": "uncorrected", **{f"n_at_{al}": int(np.nansum(p <= al)) for al in alphas}})
    df = pd.DataFrame(rows)
    df.attrs["pi0"] = estimate_pi0(p)
    return df
