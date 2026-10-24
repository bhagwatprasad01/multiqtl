"""Publication-quality figures (vector PDF/SVG/EPS and 600-dpi PNG/TIFF by default).

Design rules: thin marks, recessive grid/axes, a fixed colour-blind-validated categorical order (never cycled),
single-hue sequential ramps for magnitude (e.g. LD r^2), one y-axis per panel, text in ink colours rather than
series colours, and a legend whenever two or more series are shown.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .stats import genomic_inflation, qq_data

# ---- palette (validated categorical order; see README "Figure design") --------------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NEUTRAL = "#b9b8b2"
SEQ_CMAP = LinearSegmentedColormap.from_list("mq_blue", SEQ_BLUE)
# ordinal ramp for marks on a white surface: starts at a step that keeps >= 2:1 contrast
LD_CMAP = LinearSegmentedColormap.from_list("mq_ld", SEQ_BLUE[1:])
DIV_CMAP = LinearSegmentedColormap.from_list("mq_div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a8a", "#a32a2a"])

DEFAULT_FORMATS = ("png", "pdf", "svg", "tiff")
DEFAULT_DPI = 600


def set_style(font_size: float = 8, font_family: str = "DejaVu Sans"):
    """Journal-style defaults (8 pt text, 0.6 pt axes, Type-42 fonts so PDF/SVG text stays editable)."""
    plt.rcParams.update({
        "font.family": font_family, "font.size": font_size, "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size, "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.prop_cycle": matplotlib.cycler(color=CAT)})


set_style()


def save_figure(fig, stem: str, formats: Sequence[str] = DEFAULT_FORMATS, dpi: int = DEFAULT_DPI,
                close: bool = True) -> List[str]:
    """Save one figure in several formats. TIFF is LZW-compressed; raster formats use ``dpi``."""
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    paths = []
    for f in formats:
        f = f.lower().lstrip(".")
        path = f"{stem}.{f}"
        kw = {"bbox_inches": "tight", "pad_inches": 0.03}
        if f in ("png", "jpg", "jpeg", "tif", "tiff"):
            kw["dpi"] = dpi
        if f in ("tif", "tiff"):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kw)
        paths.append(path)
    if close:
        plt.close(fig)
    return paths


def _new(w=3.5, h=3.0, ax=None):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=(w, h))


def _nlp(p):
    return -np.log10(np.clip(np.asarray(p, float), 1e-300, 1))


# ================================================================================================================
def manhattan(df: pd.DataFrame, chrom_col="chrom", pos_col="pos", p_col="pval", threshold: Optional[float] = None,
              highlight: Optional[Iterable] = None, highlight_col: str = "variant_id", label_col: Optional[str] = None,
              n_labels: int = 6, title: str = "", ax=None, width=7.0, height=2.6):
    """Manhattan plot. Chromosomes alternate between two steps of one hue; highlighted hits use the second
    categorical slot; the threshold is a dashed reference line."""
    fig, ax = _new(width, height, ax)
    d = df[[chrom_col, pos_col, p_col] + ([highlight_col] if highlight_col in df.columns else [])
           + ([label_col] if label_col and label_col in df.columns else [])].copy()
    d["_c"] = d[chrom_col].astype(str)

    def ckey(c):
        c = c.replace("chr", "")
        return (0, int(c)) if c.isdigit() else (1, {"X": 23, "Y": 24, "MT": 25, "M": 25}.get(c, 99))
    chroms = sorted(d._c.unique(), key=ckey)
    offs, ticks, x0 = {}, [], 0
    for c in chroms:
        mx = d.loc[d._c == c, pos_col].max()
        offs[c] = x0
        ticks.append(x0 + mx / 2)
        x0 += mx * 1.02
    d["_x"] = d[pos_col] + d._c.map(offs)
    d["_y"] = _nlp(d[p_col])
    for k, c in enumerate(chroms):
        s = d[d._c == c]
        ax.scatter(s._x, s._y, s=3, lw=0, color=SEQ_BLUE[4] if k % 2 == 0 else SEQ_BLUE[2], rasterized=True)
    if highlight is not None and highlight_col in d.columns:
        h = d[d[highlight_col].isin(set(highlight))]
        ax.scatter(h._x, h._y, s=8, lw=0.4, edgecolor="white", color=CAT[1], zorder=3, label="significant", rasterized=True)
    if threshold:
        ax.axhline(-np.log10(threshold), color=INK2, lw=0.7, ls=(0, (4, 3)))
    if label_col and label_col in d.columns:
        top = d.sort_values("_y", ascending=False).drop_duplicates(label_col).head(n_labels)
        for _, r in top.iterrows():
            ax.annotate(str(r[label_col]), (r._x, r._y), xytext=(2, 2), textcoords="offset points", fontsize=6, color=INK2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(chroms, fontsize=6)
    ax.set_xlim(-x0 * 0.005, x0 * 1.005)
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Chromosome"); ax.set_ylabel(r"$-\log_{10}(P)$")
    if title:
        ax.set_title(title, loc="left")
    return fig


def qq_plot(pvals: Dict[str, np.ndarray], title: str = "", ax=None, show_lambda: bool = True):
    """QQ plot of one or more p-value sets with a 95% confidence band for the first set."""
    fig, ax = _new(3.2, 3.2, ax)
    mx = 0
    for k, (name, p) in enumerate(pvals.items()):
        q = qq_data(p)
        if k == 0:
            ax.fill_between(q.expected, q.ci_low, q.ci_high, color=GRID, lw=0, label="95% CI (null)")
        lam = genomic_inflation(p)["lambda"] if show_lambda else None
        lab = f"{name} (λ = {lam:.2f})" if lam is not None else name
        ax.scatter(q.expected, q.observed, s=6, lw=0, color=CAT[k], label=lab, rasterized=True)
        mx = max(mx, q.expected.max())
    ax.plot([0, mx], [0, mx], color=INK2, lw=0.7, ls=(0, (4, 3)))
    ax.set_xlabel(r"Expected $-\log_{10}(P)$"); ax.set_ylabel(r"Observed $-\log_{10}(P)$")
    ax.legend(loc="upper left", markerscale=2)
    if title:
        ax.set_title(title, loc="left")
    return fig


def pvalue_histogram(p, pi0: Optional[float] = None, bins: int = 40, title: str = "", ax=None):
    fig, ax = _new(3.2, 2.6, ax)
    p = np.asarray(p, float); p = p[~np.isnan(p)]
    ax.hist(p, bins=bins, range=(0, 1), color=CAT[0], edgecolor="white", lw=0.5, density=True)
    if pi0 is not None:
        ax.axhline(pi0, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.text(0.98, pi0, f"π₀ = {pi0:.2f}", ha="right", va="bottom", color=INK2, transform=ax.get_yaxis_transform())
    ax.set_xlabel("P value"); ax.set_ylabel("Density")
    if title:
        ax.set_title(title, loc="left")
    return fig


def fdr_method_comparison(summary: pd.DataFrame, alpha_col: str = "n_at_0.05", title: str = "Discoveries by correction method"):
    fig, ax = _new(3.6, 2.6)
    s = summary.sort_values(alpha_col)
    ax.barh(s.method, s[alpha_col], color=CAT[0], height=0.6)
    for y, v in enumerate(s[alpha_col]):
        ax.text(v, y, f" {v}", va="center", fontsize=6, color=INK2)
    ax.set_xlabel(f"Discoveries ({alpha_col.replace('n_at_', 'α = ')})")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    return fig


# ================================================================================================================
def locus_plot(tracks: Dict[str, pd.DataFrame], lead_variant: str, ld: Optional[pd.Series] = None,
               title: str = "", feature_pos: Optional[int] = None, annotation: str = "", credible_set=None):
    """Stacked regional association plots (e.g. eQTL over pQTL) sharing the genomic x-axis.

    Each track is a nominal table for one feature (variant_id, pos, pval). Points are coloured by LD r^2 with
    ``lead_variant`` on a single-hue sequential ramp; the lead is a diamond; credible-set members are ringed."""
    n = len(tracks)
    fig, axes = plt.subplots(n, 1, figsize=(4.6, 1.7 * n + 0.4), sharex=True)
    axes = np.atleast_1d(axes)
    sc = None
    for ax, (name, t) in zip(axes, tracks.items()):
        t = t.sort_values("pos")
        r2 = (ld.reindex(t.variant_id).values ** 2) if ld is not None else np.zeros(len(t))
        r2 = np.nan_to_num(r2)
        o = np.argsort(r2)
        x = t.pos.values[o] / 1e6; y = _nlp(t.pval.values[o])
        sc = ax.scatter(x, y, c=r2[o], cmap=LD_CMAP, vmin=0, vmax=1, s=9, lw=0.3, edgecolor="white")
        if credible_set is not None:
            cs = t[t.variant_id.isin(set(credible_set))]
            ax.scatter(cs.pos / 1e6, _nlp(cs.pval), s=26, facecolor="none", edgecolor=CAT[1], lw=0.8,
                       label="95% credible set")
        L = t[t.variant_id == lead_variant]
        if len(L):
            ax.scatter(L.pos / 1e6, _nlp(L.pval), marker="D", s=30, color=CAT[7], edgecolor=INK, lw=0.5, zorder=4,
                       label=f"{lead_variant}")
        if feature_pos is not None:
            ax.axvline(feature_pos / 1e6, color=INK2, lw=0.6, ls=":")
        ax.set_ylabel(r"$-\log_{10}(P)$")
        ax.set_title(name, loc="left", fontsize=8)
        ax.set_ylim(0, None)
    axes[0].legend(loc="upper right", fontsize=6)
    axes[-1].set_xlabel("Position (Mb)")
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=9)
    cb = fig.colorbar(sc, ax=list(axes), fraction=0.03, pad=0.02)
    cb.set_label(r"LD $r^2$ with lead", fontsize=7); cb.outline.set_linewidth(0.4)
    if annotation:
        axes[0].text(0.01, 0.95, annotation, transform=axes[0].transAxes, va="top", fontsize=6.5, color=INK2)
    return fig


def effect_concordance(leads: pd.DataFrame, title: str = "eQTL vs pQTL effects at eQTL lead variants"):
    """Scatter of transcript vs protein effect at each lead eQTL variant, by sharing category (≤3 groups +
    neutral 'other' so all-pairs colour separation holds)."""
    fig, ax = _new(3.4, 3.2)
    d = leads.copy()
    groups = [("shared_concordant", "Shared (concordant)", CAT[0]), ("eQTL_only", "eQTL only", CAT[1]),
              ("pQTL_only", "pQTL only", CAT[2])]
    other = d[~d.category.isin([g[0] for g in groups])]
    ax.scatter(other.eqtl_lead_slope, other.eqtl_lead_pqtl_slope, s=6, color=NEUTRAL, lw=0, label="Other")
    for key, lab, col in groups:
        s = d[d.category == key]
        if len(s):
            ax.scatter(s.eqtl_lead_slope, s.eqtl_lead_pqtl_slope, s=14, color=col, lw=0.4, edgecolor="white",
                       label=f"{lab} (n={len(s)})")
    lim = np.nanmax(np.abs(np.r_[d.eqtl_lead_slope, d.eqtl_lead_pqtl_slope])) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.6, ls=(0, (4, 3)))
    ax.axhline(0, color=MUTED, lw=0.5); ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("eQTL effect (SD per allele)"); ax.set_ylabel("pQTL effect (SD per allele)")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title(title, loc="left", fontsize=8)
    return fig


def tss_distance_plot(dists: Dict[str, np.ndarray], window: int = 1_000_000, bins: int = 60, title: str = ""):
    fig, ax = _new(3.6, 2.6)
    edges = np.linspace(-window, window, bins + 1) / 1e3
    for k, (name, d) in enumerate(dists.items()):
        ax.hist(np.asarray(d) / 1e3, bins=edges, histtype="step", lw=1.4, color=CAT[k], density=True, label=f"{name} (n={len(d)})")
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("Distance of lead variant to TSS (kb)"); ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.set_title(title or "Lead variant position relative to TSS", loc="left", fontsize=8)
    return fig


def genotype_effect_plot(g: np.ndarray, layers: Dict[str, np.ndarray], variant: str = "", feature: str = "",
                         alleles=("REF", "ALT")):
    """Box + jitter of each layer's (residualized) phenotype by genotype class; one panel per layer."""
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n + 0.3, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    gc = np.clip(np.rint(g), 0, 2).astype(int)
    labels = [f"{alleles[0]}/{alleles[0]}", f"{alleles[0]}/{alleles[1]}", f"{alleles[1]}/{alleles[1]}"]
    rng = np.random.default_rng(1)
    for ax, (k, (name, y)) in zip(axes, enumerate(layers.items())):
        data = [y[gc == c] for c in range(3)]
        bp = ax.boxplot(data, positions=range(3), widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1), boxprops=dict(lw=0.6, color=INK2),
                        whiskerprops=dict(lw=0.6, color=INK2), capprops=dict(lw=0.6, color=INK2))
        for b in bp["boxes"]:
            b.set_facecolor(SEQ_BLUE[0] if k == 0 else "#fbd9c9")
        for c in range(3):
            ax.scatter(c + rng.uniform(-0.15, 0.15, len(data[c])), data[c], s=4, lw=0, alpha=0.7, color=CAT[k])
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{l}\n(n={len(d)})" for l, d in zip(labels, data)], fontsize=6)
        ax.set_title(name, loc="left", fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Normalized abundance")
    fig.suptitle(f"{variant} — {feature}", x=0.02, ha="left", fontsize=8)
    fig.tight_layout()
    return fig


def coloc_barplot(coloc: pd.DataFrame, top: int = 30, label_col: str = "protein_id", title: str = "Colocalization posteriors"):
    """Stacked horizontal bars of PP.H0–H4 for the top features by PP.H4 (H4 = shared causal variant)."""
    d = coloc.sort_values("PP.H4", ascending=False).head(top).iloc[::-1]
    fig, ax = _new(4.2, 0.16 * len(d) + 0.9)
    left = np.zeros(len(d))
    names = {"PP.H0": "H0 none", "PP.H1": "H1 eQTL only", "PP.H2": "H2 pQTL only", "PP.H3": "H3 distinct", "PP.H4": "H4 shared"}
    cols = {"PP.H0": NEUTRAL, "PP.H1": CAT[1], "PP.H2": CAT[2], "PP.H3": CAT[3], "PP.H4": CAT[0]}
    for h in ["PP.H4", "PP.H3", "PP.H1", "PP.H2", "PP.H0"]:
        ax.barh(np.arange(len(d)), d[h], left=left, color=cols[h], height=0.75, edgecolor="white", lw=0.5, label=names[h])
        left += d[h].values
    ax.axvline(0.8, color=INK2, lw=0.6, ls=(0, (4, 3)))
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d[label_col], fontsize=6)
    ax.set_xlim(0, 1); ax.set_xlabel("Posterior probability")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 if len(d) > 10 else -0.25), ncol=5, fontsize=5.5, handlelength=1)
    ax.set_title(title, loc="left", fontsize=8)
    return fig


def mr_scatter(bx, sx, by, sy, results: Optional[pd.DataFrame] = None, labels=None,
               exposure: str = "mRNA", outcome: str = "protein", title: str = ""):
    fig, ax = _new(3.3, 3.0)
    bx, sx, by, sy = map(lambda a: np.asarray(a, float), (bx, sx, by, sy))
    s = np.sign(bx); bx, by = bx * s, by * s
    ax.errorbar(bx, by, xerr=1.96 * sx, yerr=1.96 * sy, fmt="o", ms=3, color=INK2, ecolor=GRID, elinewidth=0.8, capsize=0)
    xs = np.linspace(0, bx.max() * 1.1, 10)
    if results is not None:
        style = {"ivw": (CAT[0], "-"), "egger": (CAT[1], "--"), "weighted_median": (CAT[2], ":"), "wald_ratio": (CAT[0], "-")}
        for _, r in results.iterrows():
            if not np.isfinite(r.beta):
                continue
            icpt = r.get("intercept", 0.0) if r.method == "egger" else 0.0
            c, ls = style.get(r.method, (CAT[3], "-"))
            ax.plot(xs, icpt + r.beta * xs, color=c, ls=ls, lw=1.2, label=f"{r.method}: β={r.beta:.2f} (P={r.pval:.1e})")
        ax.legend(fontsize=6, loc="upper left")
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xlabel(f"Variant effect on {exposure}"); ax.set_ylabel(f"Variant effect on {outcome}")
    if title:
        ax.set_title(title, loc="left", fontsize=8)
    return fig


def cis_trans_map(trans: pd.DataFrame, variants: pd.DataFrame, annot: pd.DataFrame, fdr: float = 0.05,
                  title: str = "Variant–feature associations"):
    """Genomic position of the variant (x) vs. of the feature (y); the diagonal is cis."""
    def cum(df_chrom, df_pos, ref_chrom, ref_pos):
        chroms = sorted(set(ref_chrom), key=lambda c: (0, int(c)) if str(c).isdigit() else (1, str(c)))
        sizes = pd.Series(ref_pos).groupby(np.asarray(ref_chrom)).max()
        offs, x0 = {}, 0
        for c in chroms:
            offs[c] = x0; x0 += sizes[c] * 1.02
        return np.asarray(df_pos) + pd.Series(df_chrom).map(offs).values, offs, x0
    allc = np.r_[variants.chrom.astype(str), annot.chrom.astype(str)]
    allp = np.r_[variants.pos, annot.start]
    x, offs, tot = cum(trans.variant_chrom.astype(str), trans.variant_pos, allc, allp)
    y, _, _ = cum(trans.feature_chrom.astype(str), trans.feature_start, allc, allp)
    fig, ax = _new(3.6, 3.6)
    sig = trans.fdr_perm.fillna(0).values <= fdr
    cis = trans.is_cis.values
    ax.scatter(x[cis] / 1e6, y[cis] / 1e6, s=4, color=SEQ_BLUE[2], lw=0, label="cis", rasterized=True)
    ax.scatter(x[~cis & sig] / 1e6, y[~cis & sig] / 1e6, s=14, color=CAT[1], lw=0.4, edgecolor="white", label=f"trans (FDR ≤ {fdr})")
    for o in offs.values():
        ax.axvline(o / 1e6, color=GRID, lw=0.5); ax.axhline(o / 1e6, color=GRID, lw=0.5)
    ax.grid(False)
    ticks = [(offs[c] + (list(offs.values()) + [tot])[i + 1]) / 2 / 1e6 for i, c in enumerate(offs)]
    ax.set_xticks(ticks); ax.set_xticklabels(list(offs), fontsize=6)
    ax.set_yticks(ticks); ax.set_yticklabels(list(offs), fontsize=6)
    ax.set_xlim(0, tot / 1e6); ax.set_ylim(0, tot / 1e6)
    ax.set_xlabel("Variant position"); ax.set_ylabel("Feature position")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title(title, loc="left", fontsize=8)
    return fig


def mediation_forest(med: pd.DataFrame, top: int = 30, label_col: str = "protein_id",
                     title: str = "Indirect (mRNA-mediated) effect of lead pQTL"):
    d = med.reindex(med.indirect.abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig, ax = _new(3.8, 0.17 * len(d) + 0.9)
    y = np.arange(len(d))
    sig = (d.indirect_ci_low > 0) | (d.indirect_ci_high < 0)
    ax.hlines(y, d.indirect_ci_low, d.indirect_ci_high, color=np.where(sig, CAT[0], NEUTRAL), lw=1.2)
    ax.scatter(d.indirect, y, s=14, color=np.where(sig, CAT[0], NEUTRAL), zorder=3, lw=0)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(d[label_col], fontsize=6)
    ax.set_xlabel("Indirect effect a×b (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=8)
    return fig


def volcano(df: pd.DataFrame, slope_col="slope", p_col="pval_nominal", sig_col: Optional[str] = "significant",
            label_col: Optional[str] = "feature_id", n_labels: int = 10, title: str = ""):
    fig, ax = _new(3.4, 3.0)
    y = _nlp(df[p_col]); x = df[slope_col]
    s = df[sig_col].astype(bool).values if sig_col and sig_col in df.columns else np.zeros(len(df), bool)
    ax.scatter(x[~s], y[~s], s=5, color=NEUTRAL, lw=0, label="not significant")
    ax.scatter(x[s], y[s], s=9, color=CAT[0], lw=0, label="significant")
    if label_col and label_col in df.columns:
        for _, r in df.assign(_y=y).sort_values("_y", ascending=False).head(n_labels).iterrows():
            ax.annotate(str(r[label_col]), (r[slope_col], r._y), xytext=(2, 1), textcoords="offset points", fontsize=5.5, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_xlabel("Lead variant effect (SD per allele)"); ax.set_ylabel(r"$-\log_{10}(P)$")
    ax.legend(loc="upper left", fontsize=6)
    if title:
        ax.set_title(title, loc="left", fontsize=8)
    return fig


def interaction_plot(g: np.ndarray, y: np.ndarray, cov: np.ndarray, cov_name: str = "age",
                     variant: str = "", feature: str = ""):
    """Phenotype vs interaction covariate with a separate OLS line per genotype class (3 groups)."""
    fig, ax = _new(3.4, 2.8)
    gc = np.clip(np.rint(g), 0, 2).astype(int)
    for c, lab in enumerate(["0", "1", "2"]):
        k = gc == c
        if k.sum() < 3:
            continue
        ax.scatter(cov[k], y[k], s=6, lw=0, color=CAT[c], alpha=0.75)
        b = np.polyfit(cov[k], y[k], 1)
        xs = np.linspace(cov.min(), cov.max(), 20)
        ax.plot(xs, np.polyval(b, xs), color=CAT[c], lw=1.6, label=f"{lab} alt alleles (n={k.sum()})")
    ax.set_xlabel(cov_name); ax.set_ylabel("Normalized abundance")
    ax.legend(fontsize=6, loc="best")
    ax.set_title(f"{variant} × {cov_name} — {feature}", loc="left", fontsize=8)
    return fig


def power_heatmap(grid: pd.DataFrame, n: int, title: str = ""):
    d = grid[grid.n == n].pivot(index="maf", columns="beta", values="power")
    fig, ax = _new(3.4, 2.6)
    ax.grid(False)
    im = ax.imshow(d.values, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto", origin="lower")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(d.shape[1])); ax.set_xticklabels(d.columns)
    ax.set_yticks(range(d.shape[0])); ax.set_yticklabels(d.index)
    ax.set_xlabel("Effect size (SD per allele)"); ax.set_ylabel("MAF")
    cb = fig.colorbar(im, ax=ax, fraction=0.05); cb.set_label("Power", fontsize=7); cb.outline.set_linewidth(0.4)
    ax.set_title(title or f"Power at n = {n}", loc="left", fontsize=8)
    return fig


def correlation_histogram(corr: pd.DataFrame, col: str = "spearman_rho", title: str = "mRNA–protein correlation"):
    fig, ax = _new(3.2, 2.6)
    v = corr[col].dropna()
    ax.hist(v, bins=40, range=(-1, 1), color=CAT[0], edgecolor="white", lw=0.5)
    ax.axvline(v.median(), color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.text(v.median(), ax.get_ylim()[1] * 0.95, f" median = {v.median():.2f}", fontsize=6.5, color=INK2, va="top")
    ax.set_xlabel("Spearman ρ (across samples)"); ax.set_ylabel("Gene–protein pairs")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=8)
    return fig


def category_bar(counts: Dict[str, int], title: str = "eQTL / pQTL sharing"):
    fig, ax = _new(3.4, 2.4)
    order = [k for k in ["shared_concordant", "shared_discordant", "eQTL_only", "pQTL_only"] if k in counts]
    vals = [counts[k] for k in order]
    ax.bar(range(len(order)), vals, color=CAT[0], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=6.5)
    ax.set_ylabel("Gene–protein pairs")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=8)
    return fig
