"""Statistical analysis suite: calibration, enrichment, effect-size and power analyses for QTL results."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .fdr import estimate_pi0, fdr_summary


# ----------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------
def genomic_inflation(p, quantile: float = 0.5, n_boot: int = 200, seed: int = 0) -> Dict[str, float]:
    """Genomic inflation factor lambda_GC (median chi2 / 0.4549) with bootstrap 95% CI; ``quantile`` lets you
    compute lambda at other quantiles (e.g. 0.1 for the upper tail)."""
    p = np.asarray(p, float); p = p[(p > 0) & (p <= 1)]
    chi = stats.chi2.isf(p, 1)
    exp_q = stats.chi2.isf(1 - quantile, 1)
    lam = np.quantile(chi, quantile) / exp_q
    rng = np.random.default_rng(seed)
    bs = [np.quantile(rng.choice(chi, len(chi)), quantile) / exp_q for _ in range(n_boot)] if len(chi) > 10 else [np.nan]
    return {"lambda": float(lam), "ci_low": float(np.nanpercentile(bs, 2.5)), "ci_high": float(np.nanpercentile(bs, 97.5)), "n": int(len(p))}


def qq_data(p, max_points: int = 20000, ci: float = 0.95) -> pd.DataFrame:
    """Expected vs observed -log10(p) with pointwise beta-order-statistic confidence band, thinned for plotting
    (all top-1000 points kept)."""
    p = np.sort(np.asarray(p, float)[~np.isnan(p)])
    p = np.clip(p, 1e-300, 1)
    n = len(p)
    i = np.arange(1, n + 1)
    keep = np.unique(np.concatenate([np.arange(min(1000, n)), np.unique(np.round(np.geomspace(1, n, max_points)).astype(int) - 1)]))
    i_k = i[keep]
    lo = stats.beta.ppf((1 - ci) / 2, i_k, n - i_k + 1)
    hi = stats.beta.ppf((1 + ci) / 2, i_k, n - i_k + 1)
    return pd.DataFrame({"expected": -np.log10(i_k / (n + 1)), "observed": -np.log10(p[keep]),
                         "ci_low": -np.log10(hi), "ci_high": -np.log10(lo)})


def pvalue_uniformity(p) -> Dict[str, float]:
    """KS test of p-values against Uniform(0,1) (use on permuted/null scans to check calibration)."""
    p = np.asarray(p, float); p = p[~np.isnan(p)]
    ks = stats.kstest(p, "uniform")
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "pi0": estimate_pi0(p)}


# ----------------------------------------------------------------------------------------------------------------
# Effect sizes / architecture
# ----------------------------------------------------------------------------------------------------------------
def variance_explained(slope, se, af, n: int, dof: Optional[int] = None) -> np.ndarray:
    """Proportion of phenotypic variance explained by a variant, from t (r^2 = t^2 / (t^2 + dof))."""
    t = np.asarray(slope) / np.asarray(se)
    dof = dof or n - 2
    return t ** 2 / (t ** 2 + dof)


def maf_effect_relationship(df: pd.DataFrame, slope_col: str = "slope", af_col: str = "af") -> Dict[str, float]:
    """Spearman correlation between MAF and |effect| – a negative correlation is the classic signature of
    purifying selection on regulatory variants."""
    maf = np.minimum(df[af_col], 1 - df[af_col])
    rho, p = stats.spearmanr(maf, np.abs(df[slope_col]))
    return {"spearman_rho": float(rho), "pval": float(p), "n": int(len(df))}


def tss_distance_enrichment(sig: pd.DataFrame, background: pd.DataFrame, col: str = "tss_distance",
                            bins: Sequence[int] = (0, 1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)) -> pd.DataFrame:
    """Enrichment of significant QTL variants by |distance to TSS| bins vs. all tested pairs (Fisher exact per bin)
    plus a global KS test of the distance distributions (stored in attrs)."""
    a = np.abs(sig[col].values); b = np.abs(background[col].values)
    edges = np.asarray(bins + (np.inf,)) if isinstance(bins, tuple) else np.append(bins, np.inf)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s_in = int(((a >= lo) & (a < hi)).sum()); s_out = len(a) - s_in
        b_in = int(((b >= lo) & (b < hi)).sum()); b_out = len(b) - b_in
        orr, p = stats.fisher_exact([[s_in, s_out], [b_in, b_out]]) if len(a) else (np.nan, np.nan)
        rows.append({"bin_low": lo, "bin_high": hi, "n_sig": s_in, "n_background": b_in,
                     "frac_sig": s_in / max(len(a), 1), "frac_background": b_in / max(len(b), 1),
                     "odds_ratio": orr, "fisher_p": p})
    out = pd.DataFrame(rows)
    if len(a) and len(b):
        ks = stats.ks_2samp(a, b)
        out.attrs.update(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                         median_abs_distance_sig=float(np.median(a)), median_abs_distance_bg=float(np.median(b)))
    return out


def direction_bias(slopes) -> Dict[str, float]:
    """Binomial sign test of positive vs negative effects."""
    s = np.asarray(slopes); s = s[s != 0]
    k = int((s > 0).sum())
    return {"n_pos": k, "n_neg": int(len(s) - k), "binom_p": float(stats.binomtest(k, len(s)).pvalue) if len(s) else np.nan}


# ----------------------------------------------------------------------------------------------------------------
# Power
# ----------------------------------------------------------------------------------------------------------------
def qtl_power(n, maf, beta, alpha: float = 5e-8, n_covariates: int = 0) -> np.ndarray:
    """Analytical power of an additive QTL test (phenotype in SD units).

    Non-centrality lambda = n * h^2 / (1 - h^2) with h^2 = 2 maf (1-maf) beta^2."""
    n, maf, beta = np.broadcast_arrays(np.asarray(n, float), np.asarray(maf, float), np.asarray(beta, float))
    h2 = np.clip(2 * maf * (1 - maf) * beta ** 2, 0, 0.99)
    ncp = (n - n_covariates - 2) * h2 / (1 - h2)
    crit = stats.chi2.isf(alpha, 1)
    return stats.ncx2.sf(crit, 1, ncp)


def power_grid(n_values=(100, 200, 300, 500, 1000), maf_values=(0.05, 0.1, 0.2, 0.3, 0.5),
               beta_values=(0.1, 0.2, 0.3, 0.5, 0.8), alpha: float = 1e-5) -> pd.DataFrame:
    rows = [{"n": n, "maf": m, "beta": b, "power": float(qtl_power(n, m, b, alpha))}
            for n in n_values for m in maf_values for b in beta_values]
    return pd.DataFrame(rows)


def minimum_detectable_effect(n, maf, alpha: float = 1e-5, power: float = 0.8) -> float:
    lo, hi = 1e-4, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if qtl_power(n, maf, mid, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------------------------------------------
# Layer-level descriptive statistics
# ----------------------------------------------------------------------------------------------------------------
def describe_layer(raw: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    """Per-feature descriptive stats: missingness, mean/median, CV (on raw), and normalized SD."""
    R = raw.loc[normalized.index] if set(normalized.index) <= set(raw.index) else raw
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = R.std(1) / R.mean(1)
    return pd.DataFrame({"missing_frac": R.isna().mean(1) if R.isna().values.any() else (R == 0).mean(1),
                         "mean_raw": R.mean(1), "median_raw": R.median(1), "cv_raw": cv,
                         "sd_normalized": normalized.std(1)})


def bootstrap_effect_ci(g: np.ndarray, y: np.ndarray, C: Optional[np.ndarray] = None, n_boot: int = 1000,
                        seed: int = 0) -> Dict[str, float]:
    """Nonparametric bootstrap CI for an additive genotype effect (robust to heteroscedasticity)."""
    rng = np.random.default_rng(seed)
    n = len(g)
    X = np.column_stack([np.ones(n), g] + ([] if C is None else [C]))
    est = np.linalg.lstsq(X, y, rcond=None)[0][1]
    bs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        bs[i] = np.linalg.lstsq(X[k], y[k], rcond=None)[0][1]
    return {"beta": float(est), "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)),
            "boot_se": float(bs.std())}


def summarize_cis(perm: pd.DataFrame, nominal: Optional[pd.DataFrame] = None, fdr: float = 0.05,
                  sig_pairs: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Headline numbers for one layer's cis scan."""
    out = {"n_features_tested": int(len(perm)),
           "n_significant_features": int((perm.qval <= fdr).sum()),
           "frac_significant_features": float((perm.qval <= fdr).mean()) if len(perm) else np.nan,
           "pi1_features": float(1 - estimate_pi0(perm.pval_beta.values)) if len(perm) >= 50 else np.nan,
           "n_sig_eigenmt_bh": int((perm.get("qval_eigenmt_bh", pd.Series(dtype=float)) <= fdr).sum()),
           "median_abs_tss_distance_leads": float(perm.loc[perm.qval <= fdr, "tss_distance"].abs().median()) if (perm.qval <= fdr).any() else np.nan,
           "beta_perm_concordance_spearman": float(stats.spearmanr(perm.pval_perm, perm.pval_beta)[0]) if len(perm) > 3 else np.nan}
    if nominal is not None and len(nominal):
        out["n_pairs_tested"] = int(len(nominal))
        out["lambda_gc_nominal"] = genomic_inflation(nominal.pval.values)["lambda"]
    if sig_pairs is not None:
        out["n_significant_pairs"] = int(len(sig_pairs))
        out["n_significant_variants"] = int(sig_pairs.variant_id.nunique())
    return out
