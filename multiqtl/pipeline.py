"""End-to-end pipeline: QC -> normalization -> covariates -> cis/trans/interaction mapping -> FDR -> integration
-> statistics -> figures -> tables + HTML report."""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import fdr as F
from . import integration as I
from . import plotting as PL
from . import preprocess as PP
from . import stats as ST
from .export import HTMLReport, write_excel, write_json, write_table
from .io import Molecular, QTLDataset, load_dataset
from .mapping import QTLMapper, clump_trans, significant_pairs

DEFAULT_CONFIG: Dict = {
    "output_dir": "multiqtl_results",
    "genotype_qc": {"maf_min": 0.05, "call_rate_min": 0.95, "hwe_p_min": 1e-6, "hwe_method": "chisq"},
    "layers": {},   # name -> {type: rnaseq|proteomics|normalized, data, annot, + normalization options}
    "covariates": {"known": None, "genotype_pcs": 3, "hidden_factors": "auto", "max_hidden": 30},
    "cis": {"window": 1_000_000, "n_perm": 1000, "fdr": 0.05, "nominal_pval_max": 1.0, "independent": True,
            "eigenmt": True, "seed": 123},
    "trans": {"enabled": True, "pval_threshold": 1e-5, "cis_exclusion": 5_000_000, "n_perm": 1, "fdr": 0.05},
    "interaction": {"enabled": False, "covariate": None, "layers": None},
    "integration": {"transcript_layer": None, "protein_layer": None, "coloc": {"p1": 1e-4, "p2": 1e-4, "p12": 1e-5},
                    "mediation_boot": 1000, "smr_p_exp_max": 1e-5},
    "fdr_methods": ["bh", "by", "bonferroni", "holm", "qvalue", "lfdr"],
    "output": {"table_formats": ["tsv", "xlsx"], "figure_formats": ["png", "pdf", "svg", "tiff"], "dpi": 600,
               "n_locus_plots": 6, "report": True},
}


def _merge(a: Dict, b: Dict) -> Dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> Dict:
    """Load a YAML or JSON config; relative paths are resolved against the config file's folder."""
    from .config_tools import read_config_file
    return _merge(DEFAULT_CONFIG, read_config_file(path))


class InputError(Exception):
    """Raised when input files fail validation; the message is a plain-language checklist."""


class Log:
    def __init__(self, path):
        self.t0 = time.time()
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:  # e.g. Windows consoles with a legacy code page
            print(line.encode("ascii", "replace").decode(), flush=True)
        self.fh.write(line + "\n"); self.fh.flush()


# ================================================================================================================
def run_pipeline(config: Dict, dataset: Optional[QTLDataset] = None) -> Dict:
    cfg = _merge(DEFAULT_CONFIG, config)
    out = cfg["output_dir"]
    for d in ["tables", "figures", "figures/locus"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    log = Log(os.path.join(out, "multiqtl.log"))
    figfmt, dpi = cfg["output"]["figure_formats"], cfg["output"]["dpi"]
    tfmt = [f for f in cfg["output"]["table_formats"] if f != "xlsx"]
    want_xlsx = "xlsx" in cfg["output"]["table_formats"]
    R: Dict = {"tables": {}, "summary": {}, "figures": []}

    def fig(f, name, caption=""):
        paths = PL.save_figure(f, os.path.join(out, "figures", name), figfmt, dpi)
        R["figures"].append((paths, caption))
        return paths

    def table(df, name):
        R["tables"][name] = df
        if tfmt:
            write_table(df, os.path.join(out, "tables", name), tfmt)

    # ---------------------------------------------------------------------------------------------------- load
    if dataset is None:
        from .config_tools import check_inputs, format_report
        checks = check_inputs(cfg)
        if any(level == "ERROR" for level, _ in checks):
            raise InputError("Input check failed:\n" + format_report(checks))
        layers = {k: {"data": v["data"], "annot": v["annot"]} for k, v in cfg["layers"].items()}
        dataset = load_dataset(cfg["genotypes"], cfg.get("variants"), layers, cfg["covariates"].get("known"))
    ds = dataset
    log(f"Loaded {ds.genotypes.shape[0]} variants, {ds.qc['n_samples']} samples, layers: "
        + ", ".join(f"{k} ({v.data.shape[0]})" for k, v in ds.layers.items()))

    # ---------------------------------------------------------------------------------------------- genotype QC
    G, vqc = PP.genotype_qc(ds.genotypes, **cfg["genotype_qc"])
    V = ds.variants.loc[G.index]
    order = V.assign(_c=V.chrom.astype(str)).sort_values(["_c", "pos"]).index
    G, V = G.loc[order], V.loc[order]
    table(vqc.reset_index(), "genotype_qc")
    R["summary"]["genotype_qc"] = {"variants_in": int(len(vqc)), "variants_pass": int(vqc["pass"].sum()),
                                   "fail_maf": int((vqc.maf < cfg["genotype_qc"]["maf_min"]).sum()),
                                   "fail_call_rate": int((vqc.call_rate < cfg["genotype_qc"]["call_rate_min"]).sum()),
                                   "fail_hwe": int((vqc.hwe_p < cfg["genotype_qc"]["hwe_p_min"]).sum())}
    log(f"Genotype QC: {len(G)}/{len(vqc)} variants pass")

    # -------------------------------------------------------------------------------------------- normalization
    for name, L in ds.layers.items():
        spec = cfg["layers"].get(name, {})
        typ = spec.get("type", "normalized")
        raw = L.raw if L.raw is not None else L.data
        if typ == "rnaseq":
            L.data, q = PP.normalize_expression(raw, spec.get("method", "tmm_int"))
        elif typ == "proteomics":
            L.data, q = PP.normalize_proteomics(raw, log_transform=spec.get("log_transform", True),
                                                max_missing=spec.get("max_missing", 0.3), impute=spec.get("impute", "downshift"),
                                                center=spec.get("center", "median"), final=spec.get("final", "int"))
        else:
            L.data, q = (PP.inverse_normal_transform(raw) if spec.get("int", True) else raw), {}
        L.annot = L.annot.loc[L.data.index]
        L.qc.update(q)
        table(ST.describe_layer(raw, L.data).reset_index().rename(columns={"index": "feature_id"}), f"{name}_feature_qc")
        log(f"[{name}] normalized ({typ}): {L.data.shape[0]} features retained")

    # --------------------------------------------------------------------------------------------- covariates
    known = PP.encode_covariates(ds.covariates)
    gpcs = None
    ngpc = cfg["covariates"]["genotype_pcs"]
    if ngpc:
        sc, ve = PP.pca(G.values, ngpc)
        gpcs = pd.DataFrame(sc, index=G.columns, columns=[f"genoPC{i + 1}" for i in range(ngpc)])
        R["summary"]["genotype_pc_variance"] = ve.tolist()
    base_cov = pd.concat([x for x in [known, gpcs] if x is not None], axis=1) if (known is not None or gpcs is not None) else None
    layer_cov = {}
    for name, L in ds.layers.items():
        hf = PP.hidden_factor_covariates(L.data, cfg["covariates"]["hidden_factors"], known=base_cov,
                                         max_k=cfg["covariates"]["max_hidden"])
        hf.columns = [f"{name}_{c}" for c in hf.columns]
        layer_cov[name] = pd.concat([base_cov, hf], axis=1) if base_cov is not None else hf
        R["summary"].setdefault("covariates", {})[name] = {"n_known": 0 if known is None else known.shape[1],
                                                           "n_genotype_pcs": ngpc, "n_hidden_factors": hf.shape[1]}
        log(f"[{name}] covariates: {layer_cov[name].shape[1]} ({hf.shape[1]} hidden factors)")

    # ---------------------------------------------------------------------------------------------------- mapping
    cis = cfg["cis"]
    mappers, perm, nominal, sigp, indep, trans = {}, {}, {}, {}, {}, {}
    for name, L in ds.layers.items():
        M = QTLMapper(G, V, L.data, L.annot, layer_cov[name], name=name)
        mappers[name] = M
        nominal[name] = M.map_cis_nominal(cis["window"], cis["nominal_pval_max"])
        perm[name] = M.map_cis_permutation(cis["window"], cis["n_perm"], cis["seed"], cis["eigenmt"], cis["fdr"])
        sigp[name] = significant_pairs(nominal[name], perm[name], cis["fdr"])
        # full FDR suite on feature-level p-values
        perm[name] = F.add_all_corrections(perm[name], "pval_beta", cfg["fdr_methods"], prefix="fdr_")
        hier = F.hierarchical_fdr(nominal[name], "feature_id", "pval", cis["fdr"], cis["fdr"],
                                  group_p=perm[name].set_index("feature_id")["pval_beta"])
        nominal[name]["hier_significant"] = hier["hier_significant"].values
        table(perm[name], f"{name}_cis_feature_level")
        table(sigp[name], f"{name}_cis_significant_pairs")
        table(nominal[name], f"{name}_cis_nominal")
        fs = F.fdr_summary(perm[name].pval_beta.values)
        table(fs, f"{name}_fdr_method_summary")
        R["summary"][f"{name}_cis"] = {**ST.summarize_cis(perm[name], nominal[name], cis["fdr"], sigp[name]),
                                       "n_hierarchical_pairs": int(hier.hier_significant.sum()),
                                       "pi0_feature_level": fs.attrs["pi0"]}
        log(f"[{name}] cis: {R['summary'][f'{name}_cis']['n_significant_features']} significant features, "
            f"{len(sigp[name])} significant pairs")
        if cis.get("independent"):
            indep[name] = M.map_independent(perm[name], cis["window"], min(cis["n_perm"], 1000), cis["fdr"], seed=cis["seed"])
            table(indep[name], f"{name}_cis_independent")
            if len(indep[name]):
                R["summary"][f"{name}_cis"]["n_independent_signals"] = int(len(indep[name]))
                R["summary"][f"{name}_cis"]["n_features_multiple_signals"] = int((indep[name].groupby("feature_id").size() > 1).sum())
        if cfg["trans"]["enabled"]:
            tc = cfg["trans"]
            trans[name] = M.map_trans(tc["pval_threshold"], tc["cis_exclusion"], n_perm=tc["n_perm"])
            table(trans[name], f"{name}_trans")
            loci = clump_trans(trans[name], fdr=tc["fdr"])
            table(loci, f"{name}_trans_loci")
            tr = trans[name][~trans[name].is_cis]
            R["summary"][f"{name}_trans"] = {"n_tests": int(trans[name].attrs["n_tests"]),
                                             "n_trans_pairs_p_thr": int(len(tr)),
                                             "n_trans_fdr": int((tr.fdr_perm <= tc["fdr"]).sum()),
                                             "n_trans_loci_fdr": int(len(loci)),
                                             "n_trans_bonferroni": int((tr.bonferroni <= 0.05).sum()),
                                             "lambda_gc_genomewide": ST.genomic_inflation(trans[name].attrs["pval_sample"])["lambda"]}
            log(f"[{name}] trans: {R['summary'][f'{name}_trans']['n_trans_fdr']} trans pairs in {len(loci)} loci at FDR {tc['fdr']}")

    # ----------------------------------------------------------------------------------------------- interaction
    ic = cfg["interaction"]
    inter = {}
    if ic["enabled"] and ic["covariate"] and ds.covariates is not None:
        ivals = ds.covariates.loc[G.columns, ic["covariate"]].astype(float).values
        for name in (ic["layers"] or list(ds.layers)):
            M = mappers[name]
            # drop the interaction covariate from the adjustment set (it is re-added as a main effect)
            res, top = M.map_cis_interaction(ivals, cis["window"], name=ic["covariate"])
            inter[name] = top
            table(top, f"{name}_interaction_{ic['covariate']}_top")
            sig_i = res[res.feature_id.isin(top.loc[top.qval_bh <= cis["fdr"], "feature_id"])]
            table(sig_i.sort_values("pval_gxi").head(100000), f"{name}_interaction_{ic['covariate']}_pairs_sigfeatures")
            R["summary"][f"{name}_interaction"] = {"covariate": ic["covariate"], "n_features_tested": int(len(top)),
                                                   "n_significant_features": int((top.qval_bh <= cis["fdr"]).sum())}
            log(f"[{name}] G x {ic['covariate']}: {R['summary'][f'{name}_interaction']['n_significant_features']} features")

    # ----------------------------------------------------------------------------------------------- integration
    ig = cfg["integration"]
    tl, pl = ig.get("transcript_layer"), ig.get("protein_layer")
    integ = {}
    if tl and pl and tl in ds.layers and pl in ds.layers:
        rna, prot = ds.layers[tl], ds.layers[pl]
        mp = I.protein_gene_map(prot.annot, rna.annot)
        mp = mp[mp.protein_id.isin(prot.data.index) & mp.gene_feature_id.isin(rna.data.index)]
        table(mp, "protein_gene_map")
        Qb = PP.covariate_basis(None if base_cov is None else base_cov.loc[G.columns].values, len(G.columns))
        corr = I.mrna_protein_correlation(rna.data, prot.data, mp, Qb)
        table(corr, "mrna_protein_correlation")
        cmp_ = I.compare_qtls(nominal[tl], nominal[pl], mp, perm[tl], perm[pl], cis["fdr"])
        leads = cmp_["leads"]
        if len(leads):
            ldr = [np.corrcoef(G.loc[a].values, G.loc[b].values)[0, 1] for a, b in zip(leads.eqtl_lead, leads.pqtl_lead)]
            leads["lead_ld_r2"] = np.square(ldr)
        table(leads, "eqtl_pqtl_comparison")
        log(f"Integration: {len(mp)} gene–protein pairs; categories {leads.category.value_counts().to_dict() if len(leads) else {}}")
        co = I.coloc_all(cmp_["pairs"], **ig["coloc"])
        table(co, "coloc_eqtl_pqtl")
        smr = I.smr_heidi_all(cmp_["pairs"], G, p_exp_max=ig["smr_p_exp_max"])
        table(smr, "smr_heidi_mrna_to_protein")
        # multi-instrument MR using independent eQTL signals
        mr_rows = []
        if tl in indep and len(indep[tl]):
            pairs_idx = cmp_["pairs"].set_index(["gene_feature_id", "variant_id"])
            for gid, ivs in indep[tl].groupby("feature_id"):
                for pid in mp.loc[mp.gene_feature_id == gid, "protein_id"]:
                    sub = [(gid, v) for v in ivs.variant_id if (gid, v) in pairs_idx.index]
                    if not sub:
                        continue
                    rows = pairs_idx.loc[sub]
                    rows = rows[rows.protein_id == pid]
                    if len(rows) == 0:
                        continue
                    res = I.mendelian_randomization(rows.slope_e.values, rows.slope_se_e.values, rows.slope_p.values, rows.slope_se_p.values)
                    mr_rows.append(res.assign(gene_feature_id=gid, protein_id=pid))
        mr = pd.concat(mr_rows, ignore_index=True) if mr_rows else pd.DataFrame()
        if len(mr):
            mr["pval_bh_within_method"] = mr.groupby("method").pval.transform(lambda p: F.bh(p.values))
        table(mr, "mendelian_randomization_mrna_to_protein")
        med = I.mediation_all(leads, G, rna.data, prot.data, base_cov, n_boot=ig["mediation_boot"]) if len(leads) else pd.DataFrame()
        table(med, "mediation_pqtl_via_mrna")
        integ = {"map": mp, "corr": corr, "cmp": cmp_, "coloc": co, "smr": smr, "mr": mr, "mediation": med}
        s = dict(cmp_["summary"])
        s.update({"median_mrna_protein_spearman": float(corr.spearman_rho.median()) if len(corr) else np.nan,
                  "n_colocalized_pp4_0.8": int(co.colocalized.sum()) if len(co) else 0,
                  "n_distinct_pp3_0.8": int((co["PP.H3"] > 0.8).sum()) if len(co) else 0,
                  "n_smr_sig_bh": int((smr.p_smr_bh <= 0.05).sum()) if len(smr) else 0,
                  "n_smr_sig_pass_heidi": int(((smr.p_smr_bh <= 0.05) & smr.pass_heidi).sum()) if len(smr) else 0,
                  "mediation_mechanisms": med.mechanism.value_counts().to_dict() if len(med) else {}})
        R["summary"]["integration"] = s
        log(f"coloc: {s['n_colocalized_pp4_0.8']} with PP.H4>0.8; SMR: {s['n_smr_sig_bh']}; mediation: {s['mediation_mechanisms']}")

    # ------------------------------------------------------------------------------------------------- statistics
    stat = {}
    for name in ds.layers:
        if len(sigp[name]):
            enr = ST.tss_distance_enrichment(sigp[name], nominal[name])
            table(enr, f"{name}_tss_distance_enrichment")
            stat[f"{name}_tss_ks_p"] = enr.attrs.get("ks_p")
        lead_sig = perm[name][perm[name].qval <= cis["fdr"]]
        if len(lead_sig) > 5:
            stat[f"{name}_maf_vs_effect"] = ST.maf_effect_relationship(lead_sig)
            stat[f"{name}_direction_bias"] = ST.direction_bias(lead_sig.slope)
    pg = ST.power_grid(n_values=tuple(sorted({ds.qc["n_samples"], 100, 300, 500, 1000})), alpha=1e-5)
    table(pg, "power_grid")
    stat["min_detectable_effect_maf0.2"] = ST.minimum_detectable_effect(ds.qc["n_samples"], 0.2)
    R["summary"]["statistics"] = stat

    # ---------------------------------------------------------------------------------------------------- figures
    log("Rendering figures")
    names = list(ds.layers)
    fig(PL.qq_plot({n: nominal[n].pval.values for n in names}, "cis nominal P values"), "qq_cis_nominal",
        "QQ plot of all cis variant–feature tests per layer with 95% null band.")
    fig(PL.qq_plot({n: perm[n].pval_beta.values for n in names}, "Feature-level (beta-approx.) P values", show_lambda=False),
        "qq_feature_level", "Feature-level permutation (beta-approximated) P values.")
    for n in names:
        fig(PL.pvalue_histogram(perm[n].pval_beta, F.estimate_pi0(perm[n].pval_beta.values), title=f"{n}: feature-level P"),
            f"{n}_pvalue_histogram", "Histogram of feature-level P values; dashed line = estimated π₀.")
        fig(PL.fdr_method_comparison(F.fdr_summary(perm[n].pval_beta.values), title=f"{n}: discoveries by correction method"),
            f"{n}_fdr_methods", "Number of significant features under each multiple-testing procedure.")
        lab = "symbol" if "symbol" in ds.layers[n].annot.columns else None
        nm = nominal[n].merge(ds.layers[n].annot[[lab]] if lab else ds.layers[n].annot[[]], left_on="feature_id", right_index=True, how="left") if lab else nominal[n]
        fig(PL.manhattan(nm, threshold=None, highlight=sigp[n].variant_id, label_col=lab, title=f"{n}: cis associations"),
            f"{n}_manhattan_cis", "Manhattan plot of cis tests (orange = significant pairs).")
        pv = perm[n].merge(ds.layers[n].annot[[lab]], left_on="feature_id", right_index=True) if lab else perm[n]
        fig(PL.volcano(pv, label_col=lab or "feature_id", title=f"{n}: lead variant per feature"), f"{n}_volcano",
            "Lead-variant effect vs significance per feature.")
        if n in trans and len(trans[n]):
            fig(PL.cis_trans_map(trans[n], V, ds.layers[n].annot, cfg["trans"]["fdr"], title=f"{n}: cis/trans map"),
                f"{n}_cis_trans_map", "Variant vs feature position; off-diagonal orange points are trans-QTLs.")
            fig(PL.qq_plot({f"{n} genome-wide": trans[n].attrs["pval_sample"]}, f"{n}: genome-wide (random sample)"),
                f"{n}_qq_genomewide", "QQ plot of a random sample of all genome-wide tests.")
    fig(PL.tss_distance_plot({n: perm[n].loc[perm[n].qval <= cis["fdr"], "tss_distance"].values for n in names},
                             cis["window"]), "tss_distance", "Lead variant distance to TSS for significant features.")
    sz = ds.qc["n_samples"]
    fig(PL.power_heatmap(pg, sz), "power_heatmap", f"Analytical power at n = {sz}, α = 1e-5.")

    if integ:
        leads = integ["cmp"]["leads"]
        fig(PL.effect_concordance(leads), "eqtl_pqtl_effect_concordance",
            "Transcript vs protein effect at each lead eQTL, coloured by sharing category.")
        fig(PL.category_bar({k: int(v) for k, v in leads.category.value_counts().items()}), "eqtl_pqtl_categories",
            "Gene–protein pairs by eQTL/pQTL sharing category.")
        fig(PL.correlation_histogram(integ["corr"]), "mrna_protein_correlation", "Per-gene mRNA–protein Spearman correlation.")
        if len(integ["coloc"]):
            lbl = integ["coloc"].merge(ds.layers[pl].annot[["symbol"]], left_on="protein_id", right_index=True, how="left") \
                if "symbol" in ds.layers[pl].annot.columns else integ["coloc"].assign(symbol=integ["coloc"].protein_id)
            fig(PL.coloc_barplot(lbl, label_col="symbol"), "coloc_posteriors", "Colocalization posteriors (top 30 by PP.H4).")
        if len(integ["mediation"]):
            md = integ["mediation"].merge(ds.layers[pl].annot[["symbol"]], left_on="protein_id", right_index=True, how="left") \
                if "symbol" in ds.layers[pl].annot.columns else integ["mediation"].assign(symbol=integ["mediation"].protein_id)
            fig(PL.mediation_forest(md, label_col="symbol"), "mediation_forest", "Indirect (mRNA-mediated) effects with bootstrap CI.")
        if len(integ["mr"]):
            multi = integ["mr"][integ["mr"].method == "ivw"].sort_values("pval").head(1)
            if len(multi):
                gid, pid = multi.gene_feature_id.iloc[0], multi.protein_id.iloc[0]
                pr = integ["cmp"]["pairs"]
                rows = pr[(pr.gene_feature_id == gid) & (pr.protein_id == pid) & pr.variant_id.isin(indep[tl].loc[indep[tl].feature_id == gid, "variant_id"])]
                fig(PL.mr_scatter(rows.slope_e, rows.slope_se_e, rows.slope_p, rows.slope_se_p,
                                  integ["mr"][(integ["mr"].gene_feature_id == gid) & (integ["mr"].protein_id == pid)],
                                  title=f"MR: {gid} mRNA → {pid}"), "mr_example", "Multi-instrument MR example.")
        # locus plots: strongest colocalized + strongest distinct + pQTL-only examples
        co = integ["coloc"]
        picks = []
        if len(co):
            picks += list(co.sort_values("PP.H4", ascending=False).head(cfg["output"]["n_locus_plots"] // 2).itertuples())
            picks += list(co[co["PP.H3"] > 0.5].sort_values("PP.H3", ascending=False).head(cfg["output"]["n_locus_plots"] // 2).itertuples())
        pr = integ["cmp"]["pairs"]
        for r in picks:
            r = co.loc[r.Index]
            sub = pr[(pr.protein_id == r.protein_id) & (pr.gene_feature_id == r.gene_feature_id)]
            sub = sub.assign(pos=V.loc[sub.variant_id, "pos"].values)
            lead = sub.loc[sub.pval_p.idxmin(), "variant_id"]
            ld = I.ld_matrix(G, sub.variant_id.values)[lead]
            tracks = {f"eQTL · {r.gene_feature_id}": sub[["variant_id", "pos", "pval_e"]].rename(columns={"pval_e": "pval"}),
                      f"pQTL · {r.protein_id}": sub[["variant_id", "pos", "pval_p"]].rename(columns={"pval_p": "pval"})}
            ann = f"PP.H4 = {r['PP.H4']:.2f}   PP.H3 = {r['PP.H3']:.2f}"
            f = PL.locus_plot(tracks, lead, ld, title=f"{r.protein_id} / {r.gene_feature_id}",
                              feature_pos=int(ds.layers[pl].annot.loc[r.protein_id, "start"]), annotation=ann,
                              credible_set=str(r.credible_set).split(";"))
            fig(f, f"locus/{r.protein_id}_{r.gene_feature_id}", "Regional association with LD to pQTL lead.")
            g = G.loc[lead].values
            Mt, Mp = mappers[tl], mappers[pl]
            f = PL.genotype_effect_plot(g, {"mRNA (residualized)": Mt.residual_phenotype(r.gene_feature_id),
                                            "Protein (residualized)": Mp.residual_phenotype(r.protein_id)},
                                        variant=lead, feature=f"{r.gene_feature_id} / {r.protein_id}")
            fig(f, f"locus/{r.protein_id}_{lead}_genotype_effect", "Covariate-adjusted abundance by genotype.")

    for name, top in inter.items():
        if len(top) and (top.qval_bh <= cis["fdr"]).any():
            t = top.sort_values("pval_gxi").iloc[0]
            M = mappers[name]
            iv = ds.covariates.loc[G.columns, cfg["interaction"]["covariate"]].astype(float).values
            fig(PL.interaction_plot(G.loc[t.variant_id].values, M.residual_phenotype(t.feature_id), iv,
                                    cfg["interaction"]["covariate"], t.variant_id, t.feature_id),
                f"{name}_interaction_top", "Top genotype × covariate interaction.")

    # ----------------------------------------------------------------------------------------------------- export
    write_json(R["summary"], os.path.join(out, "summary.json"))
    with open(os.path.join(out, "config_used.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    if want_xlsx:
        log("Writing Excel workbook")
        keep = {k: v for k, v in R["tables"].items() if not k.endswith("_cis_nominal") and k != "genotype_qc" and len(v) <= 1_000_000}
        write_excel(keep, os.path.join(out, "multiqtl_results.xlsx"), readme=TABLE_DOCS_FOR(keep))
    if cfg["output"]["report"]:
        build_report(R, cfg, ds, os.path.join(out, "report.html"))
    log(f"Done. Results in {out}")
    R["config"] = cfg
    R["mappers"] = mappers
    return R


# ================================================================================================================
TABLE_DOCS = {
    "cis_feature_level": "One row per feature: lead variant, nominal/permutation/beta-approximated/eigenMT P, Storey q, all FDR corrections (fdr_*), per-feature nominal threshold.",
    "cis_significant_pairs": "All variant–feature pairs below the feature's nominal threshold for significant features (tensorQTL procedure).",
    "cis_independent": "Conditionally independent cis signals from forward–backward stepwise regression.",
    "cis_nominal": "Every cis test (TSV only; omitted from Excel for size).",
    "fdr_method_summary": "Number of significant features at alpha 0.01/0.05/0.10 under each correction method.",
    "trans_loci": "Independent trans loci (per feature, most significant variant per 1 Mb) at the trans FDR.",
    "trans": "Genome-wide pairs below the trans P threshold with permutation FDR and Bonferroni; is_cis flags pairs within the exclusion window.",
    "interaction": "Genotype × covariate interaction QTLs (top variant per feature, eigenMT then BH).",
    "feature_qc": "Per-feature missingness, raw mean/median/CV and normalized SD.",
    "tss_distance_enrichment": "Fisher-test enrichment of significant pairs by distance-to-TSS bin.",
    "protein_gene_map": "Protein-to-transcript mapping used for integration.",
    "mrna_protein_correlation": "Across-sample Pearson/Spearman mRNA–protein correlation (covariate-residualized), BH-adjusted.",
    "eqtl_pqtl_comparison": "Per gene–protein pair: lead eQTL/pQTL, cross-lookup, direction concordance, heterogeneity z (sample-overlap corrected), sharing category.",
    "coloc_eqtl_pqtl": "coloc.abf posteriors PP.H0–H4, top shared variant, 95% credible set.",
    "smr_heidi_mrna_to_protein": "SMR test (mRNA → protein) with approximate HEIDI heterogeneity test.",
    "mendelian_randomization_mrna_to_protein": "Wald / IVW / MR-Egger / weighted-median MR using independent cis-eQTL instruments.",
    "mediation_pqtl_via_mrna": "Mediation of each lead pQTL through its transcript: a, b, direct, total, indirect with Sobel and bootstrap CI.",
    "power_grid": "Analytical power by sample size, MAF and effect size (alpha = 1e-5).",
}


def TABLE_DOCS_FOR(tables):
    out = {}
    for k in tables:
        doc = next((v for key, v in TABLE_DOCS.items() if k.endswith(key) or k == key), "")
        out[k[:31]] = doc
    return out


def build_report(R: Dict, cfg: Dict, ds: QTLDataset, path: str):
    s = R["summary"]
    rep = HTMLReport("MultiQTL — genomics × transcriptomics × proteomics QTL report",
                     f"{ds.qc['n_samples']} samples · {s['genotype_qc']['variants_pass']} variants · layers: {', '.join(ds.layers)}")
    k = {"Samples": ds.qc["n_samples"], "Variants (QC pass)": s["genotype_qc"]["variants_pass"]}
    for n in ds.layers:
        c = s[f"{n}_cis"]
        k[f"{n}: significant features"] = f"{c['n_significant_features']} / {c['n_features_tested']}"
    if "integration" in s:
        k["Colocalized (PP.H4 > 0.8)"] = s["integration"]["n_colocalized_pp4_0.8"]
        k["Median mRNA–protein ρ"] = s["integration"]["median_mrna_protein_spearman"]
    rep.kpis(k)

    def figs(prefix_list):
        items = []
        for paths, cap in R["figures"]:
            png = [p for p in paths if p.endswith(".png")]
            if png and any(os.path.basename(png[0]).startswith(pf) for pf in prefix_list):
                items.append((png[0], cap))
        return items

    rep.h2("1. Quality control and normalization")
    rep.table(pd.DataFrame([s["genotype_qc"]]))
    rep.table(pd.DataFrame([{"layer": n, **{kk: vv for kk, vv in L.qc.items() if not isinstance(vv, dict)}} for n, L in ds.layers.items()]))
    rep.table(pd.DataFrame([{"layer": n, **v} for n, v in s["covariates"].items()]))

    rep.h2("2. cis-QTL mapping")
    rep.table(pd.DataFrame([{"layer": n, **{kk: vv for kk, vv in s[f'{n}_cis'].items()}} for n in ds.layers]))
    rep.figures(figs(["qq_cis", "qq_feature", "tss_distance"]))
    for n in ds.layers:
        rep.h3(f"{n}")
        rep.figures(figs([f"{n}_manhattan", f"{n}_volcano", f"{n}_pvalue", f"{n}_fdr"]))
        top = R["tables"][f"{n}_cis_feature_level"].sort_values("pval_beta").head(15)
        rep.table(top[[c for c in ["feature_id", "variant_id", "tss_distance", "af", "slope", "slope_se", "pval_nominal",
                                   "pval_beta", "pval_eigenmt", "qval", "fdr_bh", "fdr_by", "fdr_bonferroni"] if c in top]])

    rep.h2("3. False discovery rate analysis")
    rep.p("Feature-level P values are beta-approximated permutation P values (FastQTL/tensorQTL). Storey q-values "
          "control the FDR across features; variant-level significance uses each feature's nominal threshold derived "
          "from the Beta fit. eigenMT (LD-based effective tests), hierarchical TreeQTL-style FDR, BH, BY, Holm, Bonferroni "
          "and local FDR are reported side by side for sensitivity.")
    for n in ds.layers:
        rep.h3(n)
        rep.table(R["tables"][f"{n}_fdr_method_summary"])

    trans_layers = [n for n in ds.layers if f"{n}_trans" in s]
    if trans_layers:
        rep.h2("4. trans-QTL mapping")
        rep.table(pd.DataFrame([{"layer": n, **s[f"{n}_trans"]} for n in trans_layers]))
        rep.figures(figs([f"{n}_cis_trans" for n in trans_layers] + [f"{n}_qq_genome" for n in trans_layers]))
        for n in trans_layers:
            t = R["tables"][f"{n}_trans"]
            t = t[(~t.is_cis) & (t.fdr_perm <= cfg["trans"]["fdr"])]
            if len(t):
                rep.h3(f"{n}: trans hits (FDR ≤ {cfg['trans']['fdr']})")
                rep.table(t.head(25))

    inter = [n for n in ds.layers if f"{n}_interaction" in s]
    if inter:
        rep.h2(f"5. Genotype × {cfg['interaction']['covariate']} interaction QTLs")
        rep.table(pd.DataFrame([{"layer": n, **s[f"{n}_interaction"]} for n in inter]))
        rep.figures(figs([f"{n}_interaction" for n in inter]))
        for n in inter:
            t = R["tables"][f"{n}_interaction_{cfg['interaction']['covariate']}_top"].sort_values("pval_gxi_eigenmt").head(10)
            rep.table(t)

    if "integration" in s:
        rep.h2("6. eQTL–pQTL integration")
        si = s["integration"]
        rep.kpis({"Gene–protein pairs": si.get("n_gene_protein_pairs"),
                  "π₁ (eQTL leads in pQTL)": si.get("pi1_eqtl_in_pqtl"), "π₁ (pQTL leads in eQTL)": si.get("pi1_pqtl_in_eqtl"),
                  "Direction concordance (eQTL leads)": si.get("direction_concordance_eqtl_leads"),
                  "SMR significant + pass HEIDI": si.get("n_smr_sig_pass_heidi")})
        rep.figures(figs(["eqtl_pqtl_effect", "eqtl_pqtl_categories", "mrna_protein", "coloc_post", "mediation_forest", "mr_example"]))
        rep.h3("Colocalization (top by PP.H4)")
        rep.table(R["tables"]["coloc_eqtl_pqtl"].sort_values("PP.H4", ascending=False).drop(columns=["credible_set"]).head(20))
        rep.h3("Mediation of pQTLs through mRNA")
        if len(R["tables"]["mediation_pqtl_via_mrna"]):
            rep.table(R["tables"]["mediation_pqtl_via_mrna"].sort_values("sobel_p")[[
                "protein_id", "variant_id", "category", "a", "b", "direct", "total", "indirect", "indirect_ci_low",
                "indirect_ci_high", "prop_mediated", "sobel_bh", "mechanism"]].head(20))
        rep.h3("Example loci")
        rep.figures([(p[0], c) for p, c in [([q for q in paths if q.endswith(".png")], cap) for paths, cap in R["figures"]]
                     if p and "/locus/" in p[0]][:8])

    rep.h2("7. Statistical summary and power")
    st = s["statistics"]
    rows = [{"statistic": kk, "value": json.dumps(vv) if isinstance(vv, dict) else vv} for kk, vv in st.items()]
    rep.table(pd.DataFrame(rows))
    rep.figures(figs(["power_heatmap"]))

    rep.h2("Methods")
    rep.list([
        "Genotype QC: MAF, call rate and HWE (χ² or Wigginton exact) filters; residual missing dosages mean-imputed.",
        "RNA-seq: TMM normalization (edgeR algorithm), expression filter, rank-based inverse normal transform (GTEx v8).",
        "Proteomics: log2, missingness filter, median centring, left-censored (down-shift) imputation, inverse normal transform.",
        "Covariates: known covariates (one-hot for categorical), genotype PCs, and phenotype PCs chosen by the Buja–Eyuboglu permutation rule.",
        "Association: OLS of phenotype on dosage with covariates via Frisch–Waugh–Lovell residualization; cis window ±"
        f"{cfg['cis']['window'] // 1000} kb of the feature start.",
        f"Feature-level significance: {cfg['cis']['n_perm']} permutations, Beta-approximated P (Ongen 2016), Storey q-values; "
        "independent signals by forward–backward conditional regression.",
        "trans: genome-wide scan with permutation-based empirical FDR and Bonferroni.",
        "Integration: coloc.abf (Giambartolomei 2014), SMR/HEIDI-approx (Zhu 2016), MR (Wald, IVW random effects, MR-Egger, weighted median), "
        "mediation (Sobel + bootstrap), sample-overlap-corrected effect heterogeneity, π₁ replication.",
    ])
    rep.save(path)
    return path
