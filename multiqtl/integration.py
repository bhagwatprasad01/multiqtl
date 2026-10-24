"""Genomics–transcriptomics–proteomics integration: eQTL/pQTL sharing, colocalization, Mendelian randomization,
SMR/HEIDI, mediation and mRNA–protein concordance."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from .fdr import bh, estimate_pi0, pi1, qvalue
from .preprocess import covariate_basis


# ================================================================================================================
# Protein <-> gene mapping
# ================================================================================================================
def protein_gene_map(protein_annot: pd.DataFrame, gene_annot: pd.DataFrame, gene_col: str = "gene_id") -> pd.DataFrame:
    """Map protein features to transcript features via a shared ``gene_id`` column (or identical feature IDs).

    One-to-many mappings (protein groups, isoforms) are kept as separate rows."""
    if gene_col in protein_annot.columns:
        pm = protein_annot[[gene_col]].reset_index().rename(columns={protein_annot.index.name or "index": "protein_id"})
        pm.columns = ["protein_id", gene_col]
        pm = pm.assign(**{gene_col: pm[gene_col].astype(str).str.split(r"[;,]")}).explode(gene_col)
    else:
        pm = pd.DataFrame({"protein_id": protein_annot.index, gene_col: protein_annot.index})
    if gene_col in gene_annot.columns:
        gm = gene_annot[[gene_col]].reset_index()
        gm.columns = ["gene_feature_id", gene_col]
    else:
        gm = pd.DataFrame({"gene_feature_id": gene_annot.index, gene_col: gene_annot.index})
    m = pm.merge(gm, on=gene_col)
    return m[["protein_id", "gene_feature_id", gene_col]].drop_duplicates().reset_index(drop=True)


# ================================================================================================================
# mRNA–protein correlation
# ================================================================================================================
def mrna_protein_correlation(rna: pd.DataFrame, prot: pd.DataFrame, mapping: pd.DataFrame,
                             covariate_Q: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Across-sample Pearson and Spearman correlation of each protein with its transcript (optionally on
    covariate-residualized values), with BH correction."""
    common = [s for s in rna.columns if s in prot.columns]
    rows = []
    for _, m in mapping.iterrows():
        if m.protein_id not in prot.index or m.gene_feature_id not in rna.index:
            continue
        x = rna.loc[m.gene_feature_id, common].values.astype(float)
        y = prot.loc[m.protein_id, common].values.astype(float)
        if covariate_Q is not None:
            x = x - covariate_Q @ (covariate_Q.T @ x)
            y = y - covariate_Q @ (covariate_Q.T @ y)
        ok = ~(np.isnan(x) | np.isnan(y))
        pr = stats.pearsonr(x[ok], y[ok]); sr = stats.spearmanr(x[ok], y[ok])
        rows.append({"protein_id": m.protein_id, "gene_feature_id": m.gene_feature_id, "n": int(ok.sum()),
                     "pearson_r": pr[0], "pearson_p": pr[1], "spearman_rho": sr[0], "spearman_p": sr[1]})
    df = pd.DataFrame(rows)
    if len(df):
        df["pearson_bh"] = bh(df.pearson_p.values)
        df["spearman_bh"] = bh(df.spearman_p.values)
    return df


# ================================================================================================================
# eQTL vs pQTL comparison
# ================================================================================================================
def estimate_null_z_correlation(z1, z2, p1, p2, p_min: float = 0.2) -> float:
    """Correlation of z-scores among null pairs (both p > p_min) – captures sample overlap / shared noise
    between the two QTL scans (used to correct the heterogeneity test)."""
    k = (p1 > p_min) & (p2 > p_min) & np.isfinite(z1) & np.isfinite(z2)
    if k.sum() < 50:
        return 0.0
    return float(np.corrcoef(z1[k], z2[k])[0, 1])


def compare_qtls(eqtl_nominal: pd.DataFrame, pqtl_nominal: pd.DataFrame, mapping: pd.DataFrame,
                 eqtl_perm: Optional[pd.DataFrame] = None, pqtl_perm: Optional[pd.DataFrame] = None,
                 fdr: float = 0.05) -> Dict[str, object]:
    """Systematic comparison of cis-eQTLs and cis-pQTLs for matched gene/protein pairs.

    Returns a dict with
      * ``pairs``        – merged variant-level statistics for every gene-protein pair (shared variants)
      * ``leads``        – per gene-protein pair: lead eQTL/pQTL, lookup of each lead in the other layer,
                           direction concordance, heterogeneity test, and a sharing category
      * ``summary``      – counts, pi1 replication in both directions, effect-size correlation
    Sharing categories: ``shared_concordant``, ``shared_discordant``, ``eQTL_only`` (transcript effect
    buffered at the protein level), ``pQTL_only`` (candidate post-transcriptional / protein-specific
    mechanism), ``none``.
    """
    e = eqtl_nominal.rename(columns={"feature_id": "gene_feature_id"})
    p = pqtl_nominal.rename(columns={"feature_id": "protein_id"})
    keep = ["variant_id", "af", "slope", "slope_se", "pval", "tss_distance"]
    e = e[["gene_feature_id"] + keep]
    p = p[["protein_id"] + keep[:-1]]
    pairs = (mapping[["protein_id", "gene_feature_id"]]
             .merge(e, on="gene_feature_id")
             .merge(p, on=["protein_id", "variant_id"], suffixes=("_e", "_p")))
    pairs["z_e"] = pairs.slope_e / pairs.slope_se_e
    pairs["z_p"] = pairs.slope_p / pairs.slope_se_p
    rho = estimate_null_z_correlation(pairs.z_e.values, pairs.z_p.values, pairs.pval_e.values, pairs.pval_p.values)
    sig_e = set(eqtl_perm.loc[eqtl_perm.qval <= fdr, "feature_id"]) if eqtl_perm is not None else None
    sig_p = set(pqtl_perm.loc[pqtl_perm.qval <= fdr, "feature_id"]) if pqtl_perm is not None else None
    thr_e = eqtl_perm.set_index("feature_id")["pval_nominal_threshold"] if eqtl_perm is not None else None
    thr_p = pqtl_perm.set_index("feature_id")["pval_nominal_threshold"] if pqtl_perm is not None else None

    rows = []
    for (pid, gid), g in pairs.groupby(["protein_id", "gene_feature_id"]):
        ie, ip = g.pval_e.idxmin(), g.pval_p.idxmin()
        le, lp = g.loc[ie], g.loc[ip]
        is_e = (gid in sig_e) if sig_e is not None else le.pval_e < 1e-5
        is_p = (pid in sig_p) if sig_p is not None else lp.pval_p < 1e-5
        # standardized effects (phenotypes are INT-normalized; slopes on comparable SD scale)
        b1, s1, b2, s2 = le.slope_e, le.slope_se_e, le.slope_p, le.slope_se_p
        zdiff = (b1 - b2) / np.sqrt(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
        concord = np.sign(le.slope_e) == np.sign(le.slope_p)
        # lookup p of each lead in the other layer; "replicates" if nominal p < 0.05 and same direction
        if is_e and is_p:
            cat = "shared_concordant" if concord else "shared_discordant"
        elif is_e:
            cat = "eQTL_only"
        elif is_p:
            cat = "pQTL_only"
        else:
            cat = "none"
        rows.append({
            "protein_id": pid, "gene_feature_id": gid, "n_variants": len(g),
            "eqtl_lead": le.variant_id, "eqtl_lead_p": le.pval_e, "eqtl_lead_slope": le.slope_e,
            "eqtl_lead_slope_se": le.slope_se_e, "eqtl_lead_pqtl_p": le.pval_p, "eqtl_lead_pqtl_slope": le.slope_p,
            "eqtl_lead_pqtl_slope_se": le.slope_se_p,
            "pqtl_lead": lp.variant_id, "pqtl_lead_p": lp.pval_p, "pqtl_lead_slope": lp.slope_p,
            "pqtl_lead_eqtl_p": lp.pval_e, "pqtl_lead_eqtl_slope": lp.slope_e,
            "same_lead": le.variant_id == lp.variant_id,
            "eGene": is_e, "pGene": is_p, "direction_concordant": bool(concord),
            "het_z": zdiff, "het_p": 2 * stats.norm.sf(abs(zdiff)),
            "ratio_p_over_e": b2 / b1 if b1 != 0 else np.nan, "category": cat})
    leads = pd.DataFrame(rows)
    if len(leads):
        leads["het_bh"] = bh(leads.het_p.values)
    summary = {"n_gene_protein_pairs": len(leads), "null_z_correlation_rho": rho}
    if len(leads):
        vc = leads.category.value_counts().to_dict()
        summary.update({f"n_{k}": int(v) for k, v in vc.items()})
        e_sig = leads[leads.eGene]; p_sig = leads[leads.pGene]
        summary["pi1_eqtl_in_pqtl"] = pi1(e_sig.eqtl_lead_pqtl_p.values) if len(e_sig) >= 10 else np.nan
        summary["pi1_pqtl_in_eqtl"] = pi1(p_sig.pqtl_lead_eqtl_p.values) if len(p_sig) >= 10 else np.nan
        sh = leads[leads.eGene & leads.pGene]
        if len(sh) >= 3:
            summary["slope_pearson_r_shared"] = float(stats.pearsonr(sh.eqtl_lead_slope, sh.eqtl_lead_pqtl_slope)[0])
            summary["direction_concordance_shared"] = float(sh.direction_concordant.mean())
        if len(e_sig) >= 3:
            summary["direction_concordance_eqtl_leads"] = float((np.sign(e_sig.eqtl_lead_slope) == np.sign(e_sig.eqtl_lead_pqtl_slope)).mean())
            summary["sign_test_p"] = float(stats.binomtest(int((np.sign(e_sig.eqtl_lead_slope) == np.sign(e_sig.eqtl_lead_pqtl_slope)).sum()), len(e_sig)).pvalue)
    return {"pairs": pairs, "leads": leads, "summary": summary}


# ================================================================================================================
# Bayesian colocalization (coloc.abf, Giambartolomei et al. 2014, PLoS Genet 10:e1004383)
# ================================================================================================================
def _labf(beta, se, W):
    V = se ** 2
    r = W ** 2 / (W ** 2 + V)
    z = beta / se
    return 0.5 * (np.log(1 - r) + r * z * z)


def coloc_abf(beta1, se1, beta2, se2, p1: float = 1e-4, p2: float = 1e-4, p12: float = 1e-5,
              W1: float = 0.15, W2: float = 0.15, credible: float = 0.95) -> Dict[str, object]:
    """Approximate Bayes factor colocalization under a single-causal-variant assumption.

    W is the prior SD of the effect on the (standardized) trait scale; 0.15 is coloc's default for quantitative
    traits with sdY = 1, appropriate for INT-normalized molecular phenotypes.
    Returns PP.H0–H4, per-SNP posterior for a shared variant (H4) and its credible set."""
    l1 = _labf(np.asarray(beta1, float), np.asarray(se1, float), W1)
    l2 = _labf(np.asarray(beta2, float), np.asarray(se2, float), W2)
    ok = np.isfinite(l1) & np.isfinite(l2)
    l1, l2 = l1[ok], l2[ok]
    s1, s2, s12 = logsumexp(l1), logsumexp(l2), logsumexp(l1 + l2)
    lH = np.array([0.0, np.log(p1) + s1, np.log(p2) + s2,
                   np.log(p1) + np.log(p2) + (s1 + s2 + np.log1p(-min(np.exp(s12 - s1 - s2), 1 - 1e-12))),
                   np.log(p12) + s12])
    pp = np.exp(lH - logsumexp(lH))
    snp_pp = np.exp((l1 + l2) - s12)
    o = np.argsort(-snp_pp)
    cs = o[: int(np.searchsorted(np.cumsum(snp_pp[o]), credible) + 1)]
    return {"PP.H0": pp[0], "PP.H1": pp[1], "PP.H2": pp[2], "PP.H3": pp[3], "PP.H4": pp[4],
            "nsnps": int(ok.sum()), "snp_pp_h4": snp_pp, "credible_set_idx": np.where(ok)[0][cs], "ok_idx": np.where(ok)[0]}


def coloc_all(pairs: pd.DataFrame, min_snps: int = 10, **priors) -> pd.DataFrame:
    """Run coloc.abf for every gene–protein pair from ``compare_qtls(...)['pairs']``.

    Interpretation (Giambartolomei 2014): PP.H4 > 0.8 strong evidence of a shared causal variant;
    PP.H3 > 0.8 distinct causal variants; PP.H1/H2 large = association in one trait only."""
    rows = []
    for (pid, gid), g in pairs.groupby(["protein_id", "gene_feature_id"]):
        if len(g) < min_snps:
            continue
        r = coloc_abf(g.slope_e.values, g.slope_se_e.values, g.slope_p.values, g.slope_se_p.values, **priors)
        top = g.variant_id.values[r["ok_idx"][np.argmax(r["snp_pp_h4"])]]
        cs = g.variant_id.values[r["credible_set_idx"]]
        rows.append({"protein_id": pid, "gene_feature_id": gid, "nsnps": r["nsnps"],
                     **{k: r[k] for k in ["PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4"]},
                     "top_shared_variant": top, "top_snp_pp_h4": float(np.max(r["snp_pp_h4"])),
                     "credible_set_size": len(cs), "credible_set": ";".join(map(str, cs[:50]))})
    df = pd.DataFrame(rows)
    if len(df):
        hyp = df[["PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4"]].values
        df["max_hypothesis"] = np.array(["H0", "H1", "H2", "H3", "H4"])[hyp.argmax(1)]
        df["colocalized"] = df["PP.H4"] > 0.8
    return df


def coloc_prior_sensitivity(beta1, se1, beta2, se2, p12_grid=None, p1=1e-4, p2=1e-4) -> pd.DataFrame:
    """PP.H4 / PP.H3 across a grid of p12 priors (coloc sensitivity analysis)."""
    if p12_grid is None:
        p12_grid = np.logspace(-8, -4, 17)
    out = []
    for p12 in p12_grid:
        r = coloc_abf(beta1, se1, beta2, se2, p1=p1, p2=p2, p12=min(p12, p1 * 0.999))
        out.append({"p12": p12, **{k: r[k] for k in ["PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4"]}})
    return pd.DataFrame(out)


# ================================================================================================================
# Mendelian randomization (transcript -> protein, or any exposure -> outcome)
# ================================================================================================================
def mr_wald_ratio(bx, sx, by, sy):
    b = by / bx
    se = np.sqrt(sy ** 2 / bx ** 2 + by ** 2 * sx ** 2 / bx ** 4)   # second-order delta method
    return b, se, 2 * stats.norm.sf(abs(b / se))


def mr_ivw(bx, sx, by, sy, random_effects: bool = True):
    bx, by, sy = map(np.asarray, (bx, by, sy))
    w = bx ** 2 / sy ** 2
    b = np.sum(bx * by / sy ** 2) / np.sum(w)
    se = 1 / np.sqrt(np.sum(w))
    k = len(bx)
    Q = float(np.sum((by - b * bx) ** 2 / sy ** 2))
    if random_effects and k > 1:
        se *= max(1.0, np.sqrt(Q / (k - 1)))
    p = 2 * (stats.t.sf(abs(b / se), k - 1) if k > 2 else stats.norm.sf(abs(b / se)))
    return {"beta": b, "se": se, "pval": p, "Q": Q, "Q_df": k - 1, "Q_p": stats.chi2.sf(Q, k - 1) if k > 1 else np.nan}


def mr_egger(bx, sx, by, sy):
    bx, by, sy = map(np.asarray, (bx, by, sy))
    s = np.sign(bx); bx, by = bx * s, by * s       # orient instruments to positive exposure effect
    k = len(bx)
    if k < 3:
        return {"beta": np.nan, "se": np.nan, "pval": np.nan, "intercept": np.nan, "intercept_se": np.nan, "intercept_p": np.nan}
    W = 1 / sy ** 2
    X = np.column_stack([np.ones(k), bx])
    XtW = X.T * W
    cov = np.linalg.inv(XtW @ X)
    coef = cov @ XtW @ by
    res = by - X @ coef
    sigma = np.sqrt(np.sum(W * res ** 2) / (k - 2))
    se = np.sqrt(np.diag(cov)) * max(1.0, sigma)
    pv = 2 * stats.t.sf(np.abs(coef / se), k - 2)
    return {"beta": coef[1], "se": se[1], "pval": pv[1], "intercept": coef[0], "intercept_se": se[0], "intercept_p": pv[0]}


def _weighted_median(b, w):
    o = np.argsort(b); b, w = b[o], w[o] / w.sum()
    cum = np.cumsum(w) - 0.5 * w
    j = np.searchsorted(cum, 0.5)
    if j == 0:
        return b[0]
    if j >= len(b):
        return b[-1]
    return b[j - 1] + (b[j] - b[j - 1]) * (0.5 - cum[j - 1]) / (cum[j] - cum[j - 1])


def mr_weighted_median(bx, sx, by, sy, n_boot: int = 1000, seed: int = 0):
    """Bowden et al. 2016 (Genet Epidemiol 40:304); SE by parametric bootstrap."""
    bx, sx, by, sy = map(lambda a: np.asarray(a, float), (bx, sx, by, sy))
    if len(bx) < 3:
        return {"beta": np.nan, "se": np.nan, "pval": np.nan}
    biv = by / bx
    w = 1 / (sy / np.abs(bx)) ** 2
    est = _weighted_median(biv, w)
    rng = np.random.default_rng(seed)
    boots = [_weighted_median(rng.normal(by, sy) / rng.normal(bx, sx), w) for _ in range(n_boot)]
    se = float(np.std(boots))
    return {"beta": est, "se": se, "pval": 2 * stats.norm.sf(abs(est / se)) if se > 0 else np.nan}


def mendelian_randomization(bx, sx, by, sy, n_boot: int = 1000) -> pd.DataFrame:
    """Run Wald ratio (k = 1) or IVW / MR-Egger / weighted median (k >= 2/3) for one exposure–outcome pair."""
    k = len(bx)
    rows = []
    if k == 1:
        b, se, p = mr_wald_ratio(bx[0], sx[0], by[0], sy[0])
        rows.append({"method": "wald_ratio", "beta": b, "se": se, "pval": p, "n_iv": 1})
        return pd.DataFrame(rows)
    ivw = mr_ivw(bx, sx, by, sy)
    rows.append({"method": "ivw", "n_iv": k, **ivw})
    eg = mr_egger(bx, sx, by, sy)
    rows.append({"method": "egger", "n_iv": k, **eg})
    wm = mr_weighted_median(bx, sx, by, sy, n_boot=n_boot)
    rows.append({"method": "weighted_median", "n_iv": k, **wm})
    return pd.DataFrame(rows)


# ================================================================================================================
# SMR + HEIDI (Zhu et al. 2016, Nat Genet 48:481)
# ================================================================================================================
def smr_test(z_exp: float, z_out: float):
    """SMR statistic T = z_x^2 z_y^2 / (z_x^2 + z_y^2), approx chi2(1)."""
    T = z_exp ** 2 * z_out ** 2 / (z_exp ** 2 + z_out ** 2)
    return T, stats.chi2.sf(T, 1)


def heidi_test(b_exp, se_exp, b_out, se_out, ld: np.ndarray, top: int, p_exp_max: float = 1.57e-3,
               ld_min: float = 0.05, ld_max: float = 0.9, max_snps: int = 20, min_snps: int = 3):
    """Approximate HEIDI heterogeneity test.

    For SNPs in moderate LD with the top cis-SNP (r^2 between ``ld_min`` and ``ld_max``) and strong exposure
    association, compares b_xy(i) with b_xy(top) using a multivariate Wald test with LD-structured covariance
    built from delta-method variances (independent-sample assumption). Significant p (< 0.05) = linkage
    rather than pleiotropy/causality."""
    b_exp, se_exp, b_out, se_out = map(lambda a: np.asarray(a, float), (b_exp, se_exp, b_out, se_out))
    z = b_exp / se_exp
    p = 2 * stats.norm.sf(np.abs(z))
    r2 = ld[top] ** 2
    cand = np.where((p <= p_exp_max) & (r2 >= ld_min) & (r2 <= ld_max))[0]
    cand = cand[cand != top]
    if len(cand) < min_snps:
        return np.nan, len(cand)
    cand = cand[np.argsort(p[cand])][:max_snps]
    idx = np.concatenate([[top], cand])
    bxy = b_out[idx] / b_exp[idx]
    # delta-method covariance with LD
    R = ld[np.ix_(idx, idx)]
    zo = (b_out / se_out)[idx]
    zx = z[idx]
    cov = R * (np.outer(se_out[idx], se_out[idx]) / np.outer(b_exp[idx], b_exp[idx])) \
        + R * np.outer(bxy, bxy) / np.outer(zx, zx)
    d = bxy[1:] - bxy[0]
    D = np.column_stack([-np.ones(len(d)), np.eye(len(d))])
    Sd = D @ cov @ D.T
    try:
        T = float(d @ np.linalg.solve(Sd + 1e-10 * np.eye(len(d)), d))
    except np.linalg.LinAlgError:
        return np.nan, len(cand)
    return float(stats.chi2.sf(T, len(d))), len(cand)


def smr_heidi_all(pairs: pd.DataFrame, genotypes: pd.DataFrame, p_exp_max: float = 5e-8,
                  exposure: str = "e", outcome: str = "p") -> pd.DataFrame:
    """SMR (+HEIDI) for every gene–protein pair with a top cis-eQTL below ``p_exp_max`` (exposure = transcript)."""
    rows = []
    for (pid, gid), g in pairs.groupby(["protein_id", "gene_feature_id"]):
        g = g.reset_index(drop=True)
        top = int(g[f"pval_{exposure}"].idxmin())
        if g.loc[top, f"pval_{exposure}"] > p_exp_max:
            continue
        bx, sx = g[f"slope_{exposure}"].values, g[f"slope_se_{exposure}"].values
        byy, sy = g[f"slope_{outcome}"].values, g[f"slope_se_{outcome}"].values
        T, p = smr_test(bx[top] / sx[top], byy[top] / sy[top])
        b, se, _ = mr_wald_ratio(bx[top], sx[top], byy[top], sy[top])
        G = genotypes.loc[g.variant_id.values].values
        ld = np.corrcoef(G)
        ld = np.nan_to_num(ld)
        hp, nh = heidi_test(bx, sx, byy, sy, ld, top)
        rows.append({"protein_id": pid, "gene_feature_id": gid, "top_variant": g.loc[top, "variant_id"],
                     "b_smr": b, "se_smr": se, "T_smr": T, "p_smr": p, "p_heidi": hp, "n_heidi_snps": nh})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_smr_bh"] = bh(df.p_smr.values)
        df["pass_heidi"] = ~(df.p_heidi < 0.05)
    return df


# ================================================================================================================
# Mediation: does the transcript mediate the pQTL?
# ================================================================================================================
def _ols(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    dof = X.shape[0] - X.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return coef, np.sqrt(np.diag(cov))


def mediation(g: np.ndarray, m: np.ndarray, y: np.ndarray, C: Optional[np.ndarray] = None,
              n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Single-mediator analysis  g -> m (mRNA) -> y (protein), adjusting for covariates C.

    a  : g -> m;    b : m -> y | g;    c' : direct g -> y | m;    c : total g -> y
    Indirect effect a*b with Sobel test and percentile bootstrap CI; proportion mediated = a*b / c."""
    n = len(g)
    base = np.ones((n, 1)) if C is None or C.shape[1] == 0 else np.column_stack([np.ones(n), C])

    def fit(idx):
        Xa = np.column_stack([base[idx], g[idx]])
        (ca, sa) = _ols(Xa, m[idx])
        Xb = np.column_stack([base[idx], g[idx], m[idx]])
        (cb, sb) = _ols(Xb, y[idx])
        (cc, sc) = _ols(Xa, y[idx])
        return ca[-1], sa[-1], cb[-1], sb[-1], cb[-2], sb[-2], cc[-1], sc[-1]

    a, sa, b, sb, cp, scp, c, sc = fit(np.arange(n))
    ind = a * b
    sobel_se = np.sqrt(b ** 2 * sa ** 2 + a ** 2 * sb ** 2)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    Xa_full = np.column_stack([base, g])
    Xb_full = np.column_stack([base, g, m])
    for s0 in range(0, n_boot, 250):                      # batched normal equations (fast)
        B = min(250, n_boot - s0)
        idx = rng.integers(0, n, (B, n))
        Xa, Xb, mm, yy = Xa_full[idx], Xb_full[idx], m[idx], y[idx]
        ca = np.linalg.solve(np.einsum("bni,bnj->bij", Xa, Xa), np.einsum("bni,bn->bi", Xa, mm)[..., None])[..., 0]
        cb = np.linalg.solve(np.einsum("bni,bnj->bij", Xb, Xb), np.einsum("bni,bn->bi", Xb, yy)[..., None])[..., 0]
        boots[s0:s0 + B] = ca[:, -1] * cb[:, -1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return {"a": a, "a_se": sa, "b": b, "b_se": sb, "direct": cp, "direct_se": scp, "direct_p": 2 * stats.norm.sf(abs(cp / scp)),
            "total": c, "total_se": sc, "total_p": 2 * stats.norm.sf(abs(c / sc)),
            "indirect": ind, "sobel_z": ind / sobel_se, "sobel_p": 2 * stats.norm.sf(abs(ind / sobel_se)),
            "indirect_ci_low": lo, "indirect_ci_high": hi, "indirect_p_boot": max(p_boot, 1 / n_boot),
            "prop_mediated": ind / c if c != 0 else np.nan}


def mediation_all(leads: pd.DataFrame, genotypes: pd.DataFrame, rna: pd.DataFrame, prot: pd.DataFrame,
                  covariates: Optional[pd.DataFrame] = None, n_boot: int = 1000, only_pgenes: bool = True) -> pd.DataFrame:
    """Mediation analysis at each pGene's lead pQTL (variant -> transcript -> protein)."""
    sel = leads[leads.pGene] if only_pgenes else leads
    samples = list(genotypes.columns)
    C = None if covariates is None else covariates.loc[samples].values
    rows = []
    for _, r in sel.iterrows():
        g = genotypes.loc[r.pqtl_lead, samples].values.astype(float)
        m = rna.loc[r.gene_feature_id, samples].values.astype(float)
        y = prot.loc[r.protein_id, samples].values.astype(float)
        res = mediation(g, m, y, C, n_boot=n_boot)
        rows.append({"protein_id": r.protein_id, "gene_feature_id": r.gene_feature_id, "variant_id": r.pqtl_lead,
                     "category": r.category, **res})
    df = pd.DataFrame(rows)
    if len(df):
        df["sobel_bh"] = bh(df.sobel_p.values)
        df["mechanism"] = np.where((df.sobel_bh <= 0.05) & (df.direct_p > 0.05), "transcript_mediated",
                           np.where(df.sobel_bh <= 0.05, "partially_mediated", "protein_specific"))
    return df


# ================================================================================================================
# LD helper
# ================================================================================================================
def ld_matrix(genotypes: pd.DataFrame, variant_ids: Sequence[str]) -> pd.DataFrame:
    G = genotypes.loc[list(variant_ids)].values.astype(float)
    R = np.nan_to_num(np.corrcoef(G))
    return pd.DataFrame(R, index=variant_ids, columns=variant_ids)
