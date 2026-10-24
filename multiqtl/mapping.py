"""Vectorized QTL mapping engine (cis nominal, cis permutation, conditional, trans, and interaction QTLs).

All association tests are exact multiple linear regressions computed through the Frisch–Waugh–Lovell theorem:
genotypes and phenotypes are residualized on the covariate basis (intercept + covariates) once, after which the
per-variant regression of phenotype on genotype reduces to a correlation. With ``k`` covariate columns
(including the intercept), the residual degrees of freedom are ``n - k - 1`` and

    t = r * sqrt(dof / (1 - r^2)),     beta = r * ||y_res|| / ||g_res||,     se = beta / t.

This gives the same estimates as fitting ``y ~ g + covariates`` by OLS for every pair, but in matrix form.
Permutation p-values follow FastQTL / tensorQTL: the minimum nominal p across a feature's cis window is compared
with its permutation null, which is summarized by a fitted Beta distribution (Ongen et al. 2016).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .fdr import bh, eigenmt_meff, qvalue
from .preprocess import covariate_basis


# ================================================================================================================
# Low-level helpers
# ================================================================================================================
def residualize(M: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Project out the covariate space (rows of M are variables, columns samples)."""
    return M - (M @ Q) @ Q.T


def _normalize_rows(M: np.ndarray):
    norm = np.sqrt((M * M).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        Mn = M / norm[:, None]
    Mn[~np.isfinite(Mn)] = 0.0
    return Mn, norm


def r_to_t(r, dof):
    r = np.clip(r, -1 + 1e-15, 1 - 1e-15)
    return r * np.sqrt(dof / (1 - r * r))


def t_to_p(t, dof):
    return 2 * stats.t.sf(np.abs(t), dof)


def r2_to_p(r2, dof):
    r2 = np.clip(r2, 0, 1 - 1e-15)
    return t_to_p(np.sqrt(dof * r2 / (1 - r2)), dof)


def p_to_r2(p, dof):
    t = stats.t.isf(p / 2, dof)
    return t * t / (dof + t * t)


def _fit_beta(minp: np.ndarray):
    """MLE fit of Beta(a, b) to permutation min-p values (method-of-moments start / fallback)."""
    x = np.clip(minp, 1e-300, 1 - 1e-12)
    m, v = x.mean(), x.var()
    common = m * (1 - m) / max(v, 1e-12) - 1
    a0, b0 = max(m * common, 1e-3), max((1 - m) * common, 1e-3)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a, b, _, _ = stats.beta.fit(x, a0, b0, floc=0, fscale=1)
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
            raise ValueError
    except Exception:
        a, b = a0, b0
    return a, b


class CisIndex:
    """Fast lookup of variants within a window of a genomic anchor (variants must be sorted by chrom/pos)."""

    def __init__(self, variants: pd.DataFrame):
        self.chrom_idx: Dict[str, tuple] = {}
        chrom = variants["chrom"].astype(str).values
        pos = variants["pos"].values.astype(np.int64)
        for c in pd.unique(chrom):
            idx = np.where(chrom == c)[0]
            o = np.argsort(pos[idx], kind="stable")
            self.chrom_idx[c] = (pos[idx][o], idx[o])

    def window(self, chrom: str, start: int, end: Optional[int], w: int) -> np.ndarray:
        if str(chrom) not in self.chrom_idx:
            return np.array([], int)
        pos, idx = self.chrom_idx[str(chrom)]
        e = end if end is not None and not pd.isna(end) else start
        lo = np.searchsorted(pos, min(start, e) - w, "left")
        hi = np.searchsorted(pos, max(start, e) + w, "right")
        return idx[lo:hi]


# ================================================================================================================
# Prepared mapping context
# ================================================================================================================
class QTLMapper:
    """Holds residualized genotype and phenotype matrices for one molecular layer.

    Parameters
    ----------
    genotypes  : variants x samples dosage DataFrame (QC'd, no NaN)
    variants   : variant annotation (chrom, pos)
    phenotypes : features x samples normalized phenotype DataFrame
    annot      : feature annotation (chrom, start[, end, strand])
    covariates : samples x covariates numeric DataFrame (already encoded) or None
    """

    def __init__(self, genotypes: pd.DataFrame, variants: pd.DataFrame, phenotypes: pd.DataFrame,
                 annot: pd.DataFrame, covariates: Optional[pd.DataFrame] = None, name: str = "qtl"):
        if list(genotypes.columns) != list(phenotypes.columns):
            phenotypes = phenotypes[genotypes.columns]
        if phenotypes.isna().values.any():
            warnings.warn("Phenotype NaNs found – mean-imputing per feature")
            phenotypes = phenotypes.apply(lambda r: r.fillna(r.mean()), axis=1)
        self.name = name
        self.samples = list(genotypes.columns)
        self.n = len(self.samples)
        self.variant_ids = np.asarray(genotypes.index)
        self.variants = variants.loc[genotypes.index]
        self.feature_ids = np.asarray(phenotypes.index)
        self.annot = annot.loc[phenotypes.index]
        self.G = genotypes.values.astype(np.float64)
        self.P = phenotypes.values.astype(np.float64)
        self.C = None if covariates is None else covariates.loc[self.samples].values.astype(float)
        self.Q = covariate_basis(self.C, self.n)
        self.dof = self.n - self.Q.shape[1] - 1
        self.Gr = residualize(self.G, self.Q)
        self.Pr = residualize(self.P, self.Q)
        self.Gn, self.Gnorm = _normalize_rows(self.Gr)
        self.Pn, self.Pnorm = _normalize_rows(self.Pr)
        self.af = self.G.mean(1) / 2
        self.cis = CisIndex(self.variants)
        self.var_pos = self.variants["pos"].values.astype(np.int64)
        self.var_chrom = self.variants["chrom"].astype(str).values

    # ------------------------------------------------------------------------------------------------------------
    def _tss_distance(self, fi: int, vidx: np.ndarray) -> np.ndarray:
        """Signed distance to the TSS: ``start`` for + strand, ``end`` for - strand (when an end is given);
        negative = upstream."""
        a = self.annot.iloc[fi]
        minus = "strand" in a.index and a["strand"] == "-"
        tss = int(a["end"]) if minus and "end" in a.index and not pd.isna(a["end"]) else int(a["start"])
        d = self.var_pos[vidx] - tss
        return -d if minus else d

    def _assoc(self, fi: int, vidx: np.ndarray):
        r = self.Gn[vidx] @ self.Pn[fi]
        t = r_to_t(r, self.dof)
        p = t_to_p(t, self.dof)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = r * self.Pnorm[fi] / self.Gnorm[vidx]
            se = np.sqrt((1 - r * r) / self.dof) * self.Pnorm[fi] / self.Gnorm[vidx]
        return r, t, p, slope, se

    def cis_variants(self, fi: int, window: int) -> np.ndarray:
        a = self.annot.iloc[fi]
        return self.cis.window(str(a["chrom"]), int(a["start"]), a.get("end", None) if "end" in a.index else None, window)

    # ------------------------------------------------------------------------------------------------------------
    def map_cis_nominal(self, window: int = 1_000_000, pval_max: float = 1.0) -> pd.DataFrame:
        """All cis variant–feature pairs (optionally only those with p <= ``pval_max``)."""
        out = []
        for fi, fid in enumerate(self.feature_ids):
            v = self.cis_variants(fi, window)
            if len(v) == 0:
                continue
            r, t, p, slope, se = self._assoc(fi, v)
            k = p <= pval_max
            if not k.any():
                continue
            out.append(pd.DataFrame({
                "feature_id": fid, "variant_id": self.variant_ids[v[k]], "chrom": self.var_chrom[v[k]],
                "pos": self.var_pos[v[k]], "tss_distance": self._tss_distance(fi, v[k]), "af": self.af[v[k]],
                "r": r[k], "r2": r[k] ** 2, "slope": slope[k], "slope_se": se[k], "tstat": t[k], "pval": p[k]}))
        res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        res.attrs.update(layer=self.name, dof=self.dof, window=window)
        return res

    # ------------------------------------------------------------------------------------------------------------
    def _perm_minp(self, Gn_win: np.ndarray, pn: np.ndarray, perms: np.ndarray, dof: int) -> np.ndarray:
        Pp = pn[perms]                          # n_perm x n
        R = Gn_win @ Pp.T                       # m x n_perm
        return r2_to_p((R * R).max(0), dof)

    def map_cis_permutation(self, window: int = 1_000_000, n_perm: int = 1000, seed: int = 123,
                            eigenmt: bool = True, fdr: float = 0.05) -> pd.DataFrame:
        """Feature-level ("eGene"/"pGene") test with beta-approximated permutation p-values.

        Columns: lead variant stats, ``pval_perm`` (empirical), ``pval_beta`` (Beta-approximated, recommended),
        ``pval_eigenmt`` (min-p x Meff), ``qval`` (Storey, on pval_beta), ``qval_bh``, and
        ``pval_nominal_threshold`` – the per-feature nominal p threshold for calling significant variant pairs
        (tensorQTL procedure)."""
        rng = np.random.default_rng(seed)
        perms = np.array([rng.permutation(self.n) for _ in range(n_perm)])
        rows = []
        for fi, fid in enumerate(self.feature_ids):
            v = self.cis_variants(fi, window)
            if len(v) == 0:
                continue
            r, t, p, slope, se = self._assoc(fi, v)
            j = int(np.nanargmin(p))
            minp_null = self._perm_minp(self.Gn[v], self.Pn[fi], perms, self.dof)
            a, b = _fit_beta(minp_null)
            row = {"feature_id": fid, "n_variants": len(v), "variant_id": self.variant_ids[v[j]],
                   "chrom": self.var_chrom[v[j]], "pos": self.var_pos[v[j]],
                   "tss_distance": int(self._tss_distance(fi, v[[j]])[0]), "af": self.af[v[j]],
                   "slope": slope[j], "slope_se": se[j], "r2": r[j] ** 2, "pval_nominal": p[j],
                   "pval_perm": (1 + np.sum(minp_null <= p[j])) / (n_perm + 1),
                   "pval_beta": stats.beta.cdf(p[j], a, b), "beta_shape1": a, "beta_shape2": b}
            if eigenmt:
                meff = eigenmt_meff(self.G[v])
                row["meff"] = meff
                row["pval_eigenmt"] = min(1.0, p[j] * meff)
            rows.append(row)
        res = pd.DataFrame(rows)
        if len(res) == 0:
            return res
        res["qval"] = qvalue(res["pval_beta"].values)
        res["qval_bh"] = bh(res["pval_beta"].values)
        if eigenmt:
            res["qval_eigenmt_bh"] = bh(res["pval_eigenmt"].values)
        res = add_nominal_thresholds(res, fdr)
        res.attrs.update(layer=self.name, n_perm=n_perm, window=window, fdr=fdr)
        return res

    # ------------------------------------------------------------------------------------------------------------
    def map_independent(self, perm_res: pd.DataFrame, window: int = 1_000_000, n_perm: int = 1000,
                        fdr: float = 0.05, max_signals: int = 10, seed: int = 123) -> pd.DataFrame:
        """Forward–backward stepwise conditional mapping of independent cis signals (as tensorQTL map_independent).

        A signal is retained while its beta-approximated permutation p-value <= the study-wide feature-level
        threshold derived from ``perm_res`` (the p_beta threshold at the requested FDR)."""
        thr = perm_res.attrs.get("pval_beta_threshold")
        if thr is None:
            thr = add_nominal_thresholds(perm_res, fdr).attrs["pval_beta_threshold"]
        sig = perm_res[perm_res["qval"] <= fdr]
        rng = np.random.default_rng(seed)
        perms = np.array([rng.permutation(self.n) for _ in range(n_perm)])
        fidx = {f: i for i, f in enumerate(self.feature_ids)}
        vidx_of = {v: i for i, v in enumerate(self.variant_ids)}
        rows = []

        def scan(fi, v, extra):
            C = self.C if self.C is not None else np.empty((self.n, 0))
            if extra:
                C = np.column_stack([C] + [self.G[e] for e in extra])
            Q = covariate_basis(C, self.n)
            dof = self.n - Q.shape[1] - 1
            Gn, Gnorm = _normalize_rows(residualize(self.G[v], Q))
            pr = residualize(self.P[fi][None, :], Q)[0]
            pnorm = np.linalg.norm(pr)
            pn = pr / pnorm
            r = Gn @ pn
            p = r2_to_p(r * r, dof)
            p[np.isin(v, extra)] = 1.0
            j = int(np.argmin(p))
            minp = self._perm_minp(Gn, pn, perms, dof)
            a, b = _fit_beta(minp)
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = r[j] * pnorm / Gnorm[j]
                se = np.sqrt((1 - r[j] ** 2) / dof) * pnorm / Gnorm[j]
            return v[j], p[j], stats.beta.cdf(p[j], a, b), slope, se

        for fid in sig["feature_id"]:
            fi = fidx[fid]
            v = self.cis_variants(fi, window)
            lead = vidx_of[sig.loc[sig.feature_id == fid, "variant_id"].iloc[0]]
            selected = [lead]
            while len(selected) < max_signals:
                vj, pj, pb, *_ = scan(fi, v, selected)
                if pb > thr:
                    break
                selected.append(vj)
            # backward pass
            final = []
            for k in range(len(selected)):
                others = [s for i, s in enumerate(selected) if i != k]
                vj, pj, pb, sl, se = scan(fi, v, others)
                if pb <= thr:
                    final.append((vj, pj, pb, sl, se))
            for rank, (vj, pj, pb, sl, se) in enumerate(final, 1):
                rows.append({"feature_id": fid, "rank": rank, "variant_id": self.variant_ids[vj],
                             "chrom": self.var_chrom[vj], "pos": self.var_pos[vj], "af": self.af[vj],
                             "slope": sl, "slope_se": se, "pval_nominal": pj, "pval_beta": pb})
        res = pd.DataFrame(rows)
        res.attrs.update(layer=self.name, threshold=thr)
        return res

    # ------------------------------------------------------------------------------------------------------------
    def map_trans(self, pval_threshold: float = 1e-5, cis_exclusion: int = 5_000_000, chunk: int = 5000,
                  n_perm: int = 1, null_sample: int = 100_000, seed: int = 7) -> pd.DataFrame:
        """Genome-wide association of every variant with every feature, keeping pairs with p <= threshold.

        * pairs within ``cis_exclusion`` of the feature anchor are flagged ``is_cis`` (and excluded from trans FDR)
        * ``n_perm`` sample-label permutations (shared across features to preserve their correlation) give an
          empirical FDR for trans hits: FDR(p) = (#null pairs <= p / n_perm) / #observed pairs <= p
        * a random sample of all p-values is stored in ``res.attrs['pval_sample']`` for genome-wide QQ / lambda."""
        rng = np.random.default_rng(seed)
        r2_thr = p_to_r2(pval_threshold, self.dof)
        f_chrom = self.annot["chrom"].astype(str).values
        f_start = self.annot["start"].values.astype(np.int64)
        hits, null_hits, sample = [], [], []
        total = self.Gn.shape[0] * self.Pn.shape[0]
        frac = min(1.0, null_sample / max(total, 1))
        perms = [rng.permutation(self.n) for _ in range(n_perm)]
        for s in range(0, self.Gn.shape[0], chunk):
            Gc = self.Gn[s:s + chunk]
            R = Gc @ self.Pn.T                                   # variants x features
            if frac > 0:
                m = rng.random(R.shape) < frac
                sample.append(r2_to_p(R[m] ** 2, self.dof))
            vi, fi = np.where(R * R >= r2_thr)
            vi_g = vi + s
            r = R[vi, fi]
            for pm in perms:
                Rn = Gc @ self.Pn[:, pm].T
                a, b = np.where(Rn * Rn >= r2_thr)
                same = (self.var_chrom[a + s] == f_chrom[b]) & (np.abs(self.var_pos[a + s] - f_start[b]) <= cis_exclusion)
                null_hits.append(r2_to_p(Rn[a, b][~same] ** 2, self.dof))
            if len(vi) == 0:
                continue
            t = r_to_t(r, self.dof)
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = r * self.Pnorm[fi] / self.Gnorm[vi_g]
                se = np.abs(slope / t)
            hits.append(pd.DataFrame({
                "variant_id": self.variant_ids[vi_g], "feature_id": self.feature_ids[fi],
                "variant_chrom": self.var_chrom[vi_g], "variant_pos": self.var_pos[vi_g],
                "feature_chrom": f_chrom[fi], "feature_start": f_start[fi], "af": self.af[vi_g],
                "slope": slope, "slope_se": se, "r2": r * r, "pval": t_to_p(t, self.dof)}))
        res = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(
            columns=["variant_id", "feature_id", "variant_chrom", "variant_pos", "feature_chrom", "feature_start",
                     "af", "slope", "slope_se", "r2", "pval"])
        res["is_cis"] = (res.variant_chrom == res.feature_chrom) & \
                        ((res.variant_pos - res.feature_start).abs() <= cis_exclusion)
        nul = np.sort(np.concatenate(null_hits)) if null_hits else np.array([])
        tr = ~res.is_cis.values
        fdr_col = np.full(len(res), np.nan)
        if tr.any():
            ptr = res.pval.values[tr]
            o = np.argsort(ptr)
            n_null = np.searchsorted(nul, ptr[o], "right") / max(n_perm, 1)
            f = np.minimum(n_null / np.arange(1, len(o) + 1), 1)
            f = np.minimum.accumulate(f[::-1])[::-1]
            tmp = np.empty_like(f); tmp[o] = f
            fdr_col[tr] = tmp
        res["fdr_perm"] = fdr_col
        res["bonferroni"] = np.minimum(res.pval * total, 1)
        res = res.sort_values("pval").reset_index(drop=True)
        res.attrs.update(layer=self.name, n_tests=total, pval_threshold=pval_threshold,
                         pval_sample=np.concatenate(sample) if sample else np.array([]), dof=self.dof)
        return res

    # ------------------------------------------------------------------------------------------------------------
    def map_cis_interaction(self, interaction: np.ndarray, window: int = 1_000_000, name: str = "interaction",
                            maf_min_per_group: float = 0.0) -> pd.DataFrame:
        """Genotype x covariate interaction QTLs (e.g. age-dependent / ontogeny QTLs, sex- or disease-specific QTLs).

        Model: y ~ covariates + i + g + g:i   (g and i mean-centered before forming g:i).
        Reports main-effect and interaction betas, SEs and p-values; per feature the top interaction variant is
        corrected by eigenMT and then by BH across features (``top`` attribute)."""
        i = np.asarray(interaction, float)
        i = i - i.mean()
        C = self.C if self.C is not None else np.empty((self.n, 0))
        Q = covariate_basis(np.column_stack([C, i]), self.n)
        dof = self.n - Q.shape[1] - 2
        rows, tops = [], []
        for fi, fid in enumerate(self.feature_ids):
            v = self.cis_variants(fi, window)
            if len(v) == 0:
                continue
            g = self.G[v] - self.G[v].mean(1, keepdims=True)
            X1 = residualize(g, Q)
            X2 = residualize(g * i[None, :], Q)
            y = residualize(self.P[fi][None, :], Q)[0]
            a11, a22, a12 = (X1 * X1).sum(1), (X2 * X2).sum(1), (X1 * X2).sum(1)
            b1, b2 = X1 @ y, X2 @ y
            det = a11 * a22 - a12 ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                bg = (a22 * b1 - a12 * b2) / det
                bgi = (a11 * b2 - a12 * b1) / det
                rss = y @ y - (bg * b1 + bgi * b2)
                s2 = rss / dof
                se_g = np.sqrt(s2 * a22 / det)
                se_gi = np.sqrt(s2 * a11 / det)
            p_g = t_to_p(bg / se_g, dof)
            p_gi = t_to_p(bgi / se_gi, dof)
            df = pd.DataFrame({"feature_id": fid, "variant_id": self.variant_ids[v], "pos": self.var_pos[v],
                               "tss_distance": self._tss_distance(fi, v), "af": self.af[v],
                               "b_g": bg, "b_g_se": se_g, "pval_g": p_g,
                               f"b_gx{name}": bgi, f"b_gx{name}_se": se_gi, "pval_gxi": p_gi})
            rows.append(df)
            j = int(np.nanargmin(p_gi))
            meff = eigenmt_meff(self.G[v])
            tops.append({**df.iloc[j].to_dict(), "meff": meff, "pval_gxi_eigenmt": min(1, p_gi[j] * meff)})
        res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        top = pd.DataFrame(tops)
        if len(top):
            top["qval_bh"] = bh(top["pval_gxi_eigenmt"].values)
        res.attrs.update(layer=self.name, interaction=name, dof=dof)
        top.attrs.update(layer=self.name, interaction=name)
        return res, top

    # ------------------------------------------------------------------------------------------------------------
    def residual_phenotype(self, feature_id) -> np.ndarray:
        return self.Pr[np.where(self.feature_ids == feature_id)[0][0]]

    def genotype(self, variant_id) -> np.ndarray:
        return self.G[np.where(self.variant_ids == variant_id)[0][0]]


# ================================================================================================================
def add_nominal_thresholds(res: pd.DataFrame, fdr: float = 0.05) -> pd.DataFrame:
    """tensorQTL procedure: find the p_beta cut-off between significant and non-significant features at ``fdr``
    and convert it to a feature-specific nominal p threshold via the inverse Beta CDF."""
    res = res.copy()
    sig = res.loc[res.qval <= fdr, "pval_beta"]
    non = res.loc[res.qval > fdr, "pval_beta"]
    if len(sig) == 0:
        thr = 0.0
    elif len(non) == 0:
        thr = float(sig.max())
    else:
        thr = 0.5 * (float(sig.max()) + float(non.min()))
    res["pval_nominal_threshold"] = stats.beta.ppf(thr, res.beta_shape1, res.beta_shape2) if thr > 0 else 0.0
    res["significant"] = res.qval <= fdr
    res.attrs["pval_beta_threshold"] = thr
    return res


def clump_trans(trans: pd.DataFrame, window: int = 1_000_000, fdr: Optional[float] = 0.05) -> pd.DataFrame:
    """Collapse trans hits into independent loci: per feature, greedily keep the most significant variant and
    drop hits on the same chromosome within ``window`` bp. Adds ``n_variants_in_locus``."""
    t = trans[~trans.is_cis]
    if fdr is not None:
        t = t[t.fdr_perm <= fdr]
    rows = []
    for fid, g in t.sort_values("pval").groupby("feature_id", sort=False):
        taken = []
        for r in g.itertuples():
            hit = next((k for k in taken if k[0] == r.variant_chrom and abs(k[1] - r.variant_pos) <= window), None)
            if hit is None:
                taken.append([r.variant_chrom, r.variant_pos, r.Index, 1])
            else:
                hit[3] += 1
        for c, p, idx, n in taken:
            rows.append({**t.loc[idx].to_dict(), "n_variants_in_locus": n})
    return pd.DataFrame(rows, columns=list(t.columns) + ["n_variants_in_locus"]).sort_values("pval").reset_index(drop=True)


def significant_pairs(nominal: pd.DataFrame, perm: pd.DataFrame, fdr: float = 0.05) -> pd.DataFrame:
    """All variant–feature pairs passing the per-feature nominal threshold for features significant at ``fdr``."""
    sig = perm.loc[perm.qval <= fdr, ["feature_id", "pval_nominal_threshold"]]
    m = nominal.merge(sig, on="feature_id")
    return m[m.pval <= m.pval_nominal_threshold].reset_index(drop=True)
