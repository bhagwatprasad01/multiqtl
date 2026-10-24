"""Quality control and normalization for genotypes, RNA-seq and LC-MS proteomics data."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ================================================================================================================
# Genotype QC
# ================================================================================================================
def hwe_exact(n_het: int, n_hom1: int, n_hom2: int) -> float:
    """Exact Hardy-Weinberg test (Wigginton, Cutler & Abecasis 2005, AJHG 76:887)."""
    n_homr, n_homc = min(n_hom1, n_hom2), max(n_hom1, n_hom2)
    n_rare = 2 * n_homr + n_het
    n = n_het + n_homr + n_homc
    if n == 0 or n_rare == 0:
        return 1.0
    probs = np.zeros(n_rare + 1)
    mid = int(n_rare * (2 * n - n_rare) / (2 * n))
    if (n_rare - mid) % 2:
        mid += 1
    probs[mid] = 1.0
    h, hr, hc = mid, (n_rare - mid) // 2, n - mid - (n_rare - mid) // 2
    while h >= 2:
        probs[h - 2] = probs[h] * h * (h - 1) / (4.0 * (hr + 1) * (hc + 1))
        h -= 2; hr += 1; hc += 1
    h, hr, hc = mid, (n_rare - mid) // 2, n - mid - (n_rare - mid) // 2
    while h <= n_rare - 2:
        probs[h + 2] = probs[h] * 4.0 * hr * hc / ((h + 2.0) * (h + 1.0))
        h += 2; hr -= 1; hc -= 1
    probs /= probs.sum()
    return float(min(1.0, probs[probs <= probs[n_het] * (1 + 1e-8)].sum()))


def hwe_chisq(G: np.ndarray) -> np.ndarray:
    """Vectorized 1-df chi-square HWE test on hard-called genotypes (rows = variants)."""
    R = np.rint(G)
    n0 = np.nansum(R == 0, 1); n1 = np.nansum(R == 1, 1); n2 = np.nansum(R == 2, 1)
    n = n0 + n1 + n2
    p = (2 * n2 + n1) / np.maximum(2 * n, 1)
    q = 1 - p
    e0, e1, e2 = n * q * q, 2 * n * p * q, n * p * p
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = (n0 - e0) ** 2 / e0 + (n1 - e1) ** 2 / e1 + (n2 - e2) ** 2 / e2
    chi = np.nan_to_num(chi, nan=0.0, posinf=0.0)
    return stats.chi2.sf(chi, 1)


def genotype_qc(G: pd.DataFrame, maf_min: float = 0.05, call_rate_min: float = 0.95,
                hwe_p_min: float = 1e-6, hwe_method: str = "chisq") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter variants on MAF, call rate and HWE; mean-impute remaining missing dosages.

    Returns (filtered genotype DataFrame, per-variant QC table)."""
    X = G.values.astype(float)
    call = 1 - np.isnan(X).mean(1)
    af = np.nanmean(X, 1) / 2
    maf = np.minimum(af, 1 - af)
    if hwe_method == "exact":
        R = np.rint(X)
        hwe = np.array([hwe_exact(int(np.nansum(r == 1)), int(np.nansum(r == 0)), int(np.nansum(r == 2))) for r in R])
    else:
        hwe = hwe_chisq(X)
    qc = pd.DataFrame({"af": af, "maf": maf, "call_rate": call, "hwe_p": hwe}, index=G.index)
    qc["pass"] = (qc.maf >= maf_min) & (qc.call_rate >= call_rate_min) & (qc.hwe_p >= hwe_p_min)
    sub = G.loc[qc["pass"]]
    vals = np.array(sub.values, dtype=float, copy=True)
    m = np.isnan(vals)
    if m.any():
        rowmean = np.nanmean(vals, 1)
        vals[m] = np.take(rowmean, np.where(m)[0])
    return pd.DataFrame(vals, index=sub.index, columns=sub.columns), qc


# ================================================================================================================
# Phenotype normalization
# ================================================================================================================
def inverse_normal_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Rank-based inverse normal transform per feature (row); ties averaged, NaN preserved (Blom offset 0.5)."""
    X = df.values.astype(float)
    out = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        ok = ~np.isnan(X[i])
        r = stats.rankdata(X[i, ok])
        out[i, ok] = stats.norm.ppf((r - 0.5) / ok.sum())
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalize columns (samples) to a common distribution."""
    ranks = df.rank(axis=0, method="first")
    mean_sorted = np.sort(df.values, axis=0).mean(1)
    out = ranks.apply(lambda c: pd.Series(np.interp(c.values, np.arange(1, len(mean_sorted) + 1), mean_sorted), index=c.index))
    return out


def tmm_factors(counts: np.ndarray, logratio_trim: float = 0.3, sum_trim: float = 0.05) -> np.ndarray:
    """TMM normalization factors (Robinson & Oshlack 2010), mirroring edgeR::calcNormFactors defaults."""
    counts = np.asarray(counts, float)
    lib = counts.sum(0)
    f75 = np.quantile(counts / lib, 0.75, axis=0)
    ref = int(np.argmin(np.abs(f75 - f75.mean())))
    nR, refc = lib[ref], counts[:, ref]
    fac = np.ones(counts.shape[1])
    for j in range(counts.shape[1]):
        obs, nO = counts[:, j], lib[j]
        k = (obs > 0) & (refc > 0)
        o, r = obs[k], refc[k]
        logR = np.log2((o / nO) / (r / nR))
        absE = 0.5 * (np.log2(o / nO) + np.log2(r / nR))
        v = (nO - o) / nO / o + (nR - r) / nR / r
        n = len(logR)
        if n == 0:
            continue
        loL, hiL = np.floor(n * logratio_trim) + 1, n + 1 - (np.floor(n * logratio_trim) + 1)
        loS, hiS = np.floor(n * sum_trim) + 1, n + 1 - (np.floor(n * sum_trim) + 1)
        rl, ra = stats.rankdata(logR), stats.rankdata(absE)
        keep = (rl >= loL) & (rl <= hiL) & (ra >= loS) & (ra <= hiS)
        if keep.sum() > 0:
            fac[j] = 2 ** (np.sum(logR[keep] / v[keep]) / np.sum(1 / v[keep]))
    return fac / np.exp(np.mean(np.log(fac)))


def normalize_expression(counts: pd.DataFrame, method: str = "tmm_int", min_count: int = 6,
                         min_cpm: float = 0.5, min_frac: float = 0.2) -> Tuple[pd.DataFrame, dict]:
    """RNA-seq count normalization (GTEx-style).

    method: ``tmm_int`` (TMM-CPM -> inverse-normal; default, GTEx v8), ``tmm_log`` (log2 TMM-CPM),
            ``log`` (log2(x+1) for already normalized TPM/FPKM), ``int`` (INT only), ``none``.
    Genes are kept if count >= ``min_count`` and CPM >= ``min_cpm`` in >= ``min_frac`` of samples (count methods).
    """
    qc = {"n_features_in": counts.shape[0]}
    if method in ("tmm_int", "tmm_log"):
        C = counts.values.astype(float)
        fac = tmm_factors(C)
        eff = C.sum(0) * fac
        cpm = C / eff * 1e6
        keep = ((C >= min_count).mean(1) >= min_frac) & ((cpm >= min_cpm).mean(1) >= min_frac)
        cpm = pd.DataFrame(cpm[keep], index=counts.index[keep], columns=counts.columns)
        qc.update(tmm_factors=dict(zip(counts.columns, fac)), n_features_out=int(keep.sum()))
        out = inverse_normal_transform(cpm) if method == "tmm_int" else np.log2(cpm + 1)
    elif method == "log":
        out = np.log2(counts + 1)
    elif method == "int":
        out = inverse_normal_transform(counts)
    else:
        out = counts.copy()
    qc.setdefault("n_features_out", out.shape[0])
    return out, qc


def impute_proteomics(X: pd.DataFrame, method: str = "downshift", shift: float = 1.8, width: float = 0.3,
                      seed: int = 0) -> pd.DataFrame:
    """Left-censored (MNAR) imputation for log-scale LC-MS intensities.

    ``downshift`` – Perseus-style: draw from N(mean - shift*sd, (width*sd)^2) per sample.
    ``min``       – per-feature minimum; ``halfmin`` – per-feature minimum minus 1 (log2 half-minimum);
    ``knn``       – mean of the 10 nearest features (Euclidean on complete overlap).
    """
    rng = np.random.default_rng(seed)
    V = X.values.copy()
    miss = np.isnan(V)
    if not miss.any():
        return X.copy()
    if method == "downshift":
        mu, sd = np.nanmean(V, 0), np.nanstd(V, 0)
        for j in range(V.shape[1]):
            k = miss[:, j]
            V[k, j] = rng.normal(mu[j] - shift * sd[j], width * sd[j], k.sum())
    elif method in ("min", "halfmin"):
        mn = np.nanmin(V, 1) - (1.0 if method == "halfmin" else 0.0)
        V[miss] = np.take(mn, np.where(miss)[0])
    elif method == "knn":
        filled = np.where(miss, np.nanmean(V, 1, keepdims=True), V)
        Z = (filled - filled.mean(1, keepdims=True)) / (filled.std(1, keepdims=True) + 1e-9)
        D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1) if Z.shape[0] <= 3000 else None
        for i in np.where(miss.any(1))[0]:
            d = D[i] if D is not None else ((Z - Z[i]) ** 2).sum(1)
            nn = np.argsort(d)[1:11]
            for j in np.where(miss[i])[0]:
                vals = V[nn, j][~np.isnan(V[nn, j])]
                V[i, j] = vals.mean() if len(vals) else np.nanmin(V[i])
    else:
        raise ValueError(method)
    return pd.DataFrame(V, index=X.index, columns=X.columns)


def normalize_proteomics(intens: pd.DataFrame, log_transform: bool = True, max_missing: float = 0.3,
                         impute: str = "downshift", center: str = "median", final: str = "int",
                         seed: int = 0) -> Tuple[pd.DataFrame, dict]:
    """LC-MS protein abundance pipeline: log2 -> missingness filter -> sample centering -> MNAR imputation -> INT.

    Zero intensities are treated as missing. ``center`` in {"median", "quantile", "none"};
    ``final`` in {"int", "zscore", "none"}.
    """
    X = intens.replace(0, np.nan).astype(float)
    if log_transform:
        X = np.log2(X)
    qc = {"n_features_in": X.shape[0], "missing_rate_raw": float(X.isna().values.mean())}
    keep = X.isna().mean(1) <= max_missing
    X = X.loc[keep]
    if center == "median":
        med = X.median(0)
        X = X - med + med.mean()
    elif center == "quantile":
        X = quantile_normalize(X)
    X = impute_proteomics(X, method=impute, seed=seed)
    qc.update(n_features_out=int(keep.sum()), n_removed_missing=int((~keep).sum()))
    if final == "int":
        X = inverse_normal_transform(X)
    elif final == "zscore":
        X = X.sub(X.mean(1), axis=0).div(X.std(1), axis=0)
    return X, qc


# ================================================================================================================
# Covariates / hidden factors
# ================================================================================================================
def encode_covariates(C: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """One-hot encode categorical covariates (drop first level), impute numeric NaN by mean."""
    if C is None or C.shape[1] == 0:
        return None
    parts = []
    for c in C.columns:
        s = C[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            parts.append(pd.get_dummies(s.astype(str), prefix=c, drop_first=True).astype(float))
        else:
            parts.append(s.astype(float).fillna(s.mean()).to_frame())
    out = pd.concat(parts, axis=1)
    return out.loc[:, out.std() > 0]


def pca(X: np.ndarray, k: int):
    """PCA of a features x samples matrix; returns (sample scores n x k, variance explained)."""
    Z = X - X.mean(1, keepdims=True)
    Z = Z / (Z.std(1, keepdims=True) + 1e-12)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ve = S ** 2 / np.sum(S ** 2)
    return Vt[:k].T * S[:k], ve[:k]


def choose_n_pcs_be(X: np.ndarray, n_perm: int = 20, alpha: float = 0.05, max_k: int = 50, seed: int = 0) -> int:
    """Buja & Eyuboglu (1992) permutation method for the number of significant PCs (as in PCAForQTL, Zhou 2022)."""
    rng = np.random.default_rng(seed)
    Z = X - X.mean(1, keepdims=True)
    Z = Z / (Z.std(1, keepdims=True) + 1e-12)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    ve = s / s.sum()
    K = min(max_k, len(ve))
    null = np.zeros((n_perm, K))
    for b in range(n_perm):
        P = np.array([rng.permutation(r) for r in Z])
        sp = np.linalg.svd(P, compute_uv=False) ** 2
        null[b] = (sp / sp.sum())[:K]
    pvals = (1 + (null >= ve[:K]).sum(0)) / (n_perm + 1)
    k = 0
    while k < K and pvals[k] <= alpha:
        k += 1
    return k


def hidden_factor_covariates(pheno: pd.DataFrame, n: str | int = "auto", known: Optional[pd.DataFrame] = None,
                             max_k: int = 30, seed: int = 0) -> pd.DataFrame:
    """Phenotype PCs as hidden-confounder covariates (PEER/PCA). ``n='auto'`` uses Buja-Eyuboglu, ``'elbow'``
    uses the max-distance elbow on the scree curve. Known covariates are regressed out first, so PCs are
    orthogonal additions."""
    X = pheno.values.astype(float)
    if known is not None and known.shape[1]:
        Q = covariate_basis(known.values)
        X = X - (X @ Q) @ Q.T
    if n == "auto":
        k = choose_n_pcs_be(X, max_k=max_k, seed=seed)
    elif n == "elbow":
        _, ve = pca(X, min(max_k, X.shape[1] - 1))
        x = np.arange(len(ve)); y = ve
        line = (y[-1] - y[0]) / (x[-1] - x[0])
        d = np.abs(line * x - y + y[0]) / np.sqrt(line ** 2 + 1)
        k = int(np.argmax(d)) + 1
    else:
        k = int(n)
    k = max(k, 0)
    if k == 0:
        return pd.DataFrame(index=pheno.columns)
    scores, _ = pca(X, k)
    return pd.DataFrame(scores, index=pheno.columns, columns=[f"PC{i + 1}" for i in range(k)])


def covariate_basis(C: Optional[np.ndarray], n: Optional[int] = None) -> np.ndarray:
    """Orthonormal basis Q (n x k) for [intercept, covariates]; used to residualize by projection."""
    if C is None or (hasattr(C, "shape") and C.shape[1] == 0):
        return np.ones((n, 1)) / np.sqrt(n)
    C = np.asarray(C, float)
    M = np.column_stack([np.ones(C.shape[0]), C])
    from scipy.linalg import qr
    Q, R, _ = qr(M, mode="economic", pivoting=True)   # pivoting -> robust rank detection
    d = np.abs(np.diag(R))
    rank = int(np.sum(d > 1e-8 * d.max()))
    return Q[:, :rank]
