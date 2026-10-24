"""Benchmark a MultiQTL run on simulated data against the simulation's ground truth.

usage: python validate_against_truth.py <sim_dir> <results_dir>
"""
import sys

import numpy as np
import pandas as pd

sim, res = sys.argv[1], sys.argv[2]
truth = pd.read_csv(f"{sim}/truth_cis.tsv", sep="\t")
special = pd.read_csv(f"{sim}/truth_trans_interaction.tsv", sep="\t")
G = pd.read_csv(f"{sim}/genotypes.tsv.gz", sep="\t", index_col=0)
T = lambda n: pd.read_csv(f"{res}/tables/{n}.tsv", sep="\t")


def ld_r2(a, b):
    if a not in G.index or b not in G.index:
        return np.nan
    x, y = G.loc[a].values, G.loc[b].values
    k = ~(np.isnan(x) | np.isnan(y))
    return np.corrcoef(x[k], y[k])[0, 1] ** 2


def prf(called, true, universe):
    called, true = set(called) & set(universe), set(true) & set(universe)
    tp = len(called & true)
    return {"called": len(called), "true": len(true), "TP": tp, "FDP": 1 - tp / max(len(called), 1),
            "recall": tp / max(len(true), 1)}


rows = []
# ---- eGenes
e = T("expression_cis_feature_level")
true_e = truth.loc[truth.eqtl_variant.notna(), "gene_feature_id"]
rows.append({"test": "eGenes (q<=0.05)", **prf(e.loc[e.qval <= 0.05, "feature_id"], true_e, e.feature_id)})
sig = e[e.qval <= 0.05].merge(truth, left_on="feature_id", right_on="gene_feature_id")
sig = sig[sig.eqtl_variant.notna()]
lead_ok = np.mean([ld_r2(a, b) >= 0.8 for a, b in zip(sig.variant_id, sig.eqtl_variant)])
# ---- pGenes: true pQTL = direct protein variant OR mediated eQTL (coupling > 0)
p = T("protein_cis_feature_level")
tp_mask = truth.protein_id.notna() & (truth.pqtl_variant.notna() | ((truth.eqtl_variant.notna()) & (truth.coupling > 0)))
rows.append({"test": "pGenes (q<=0.05)", **prf(p.loc[p.qval <= 0.05, "feature_id"], truth.loc[tp_mask, "protein_id"], p.feature_id)})
# ---- category accuracy
cmp_ = T("eqtl_pqtl_comparison").merge(truth, on=["gene_feature_id", "protein_id"], how="left")
exp_cat = {"mediated": "shared_concordant", "eqtl_only": "eQTL_only", "pqtl_only": "pQTL_only", "no_cis_qtl": "none",
           "distinct": "shared_concordant|shared_discordant"}
cmp_["expected"] = cmp_.mechanism.map(exp_cat)
cmp_["match"] = [str(c) in str(x).split("|") for c, x in zip(cmp_.category, cmp_.expected)]
conf = pd.crosstab(cmp_.mechanism, cmp_.category)
# ---- coloc
co = T("coloc_eqtl_pqtl").merge(truth, on=["gene_feature_id", "protein_id"])
coloc_tab = co.groupby("mechanism")[["PP.H3", "PP.H4"]].median().round(2)
coloc_tab["frac_H4>0.8"] = co.groupby("mechanism")["PP.H4"].apply(lambda s: (s > 0.8).mean()).round(2)
coloc_tab["frac_H3>0.8"] = co.groupby("mechanism")["PP.H3"].apply(lambda s: (s > 0.8).mean()).round(2)
# ---- mediation
med = T("mediation_pqtl_via_mrna").merge(truth, on=["gene_feature_id", "protein_id"])
med_tab = pd.crosstab(med.mechanism_y if "mechanism_y" in med else med.mechanism, med.mechanism_x if "mechanism_x" in med else med.category)
# ---- trans
tr = T("protein_trans")
tr = tr[(~tr.is_cis) & (tr.fdr_perm <= 0.05)]
tt = special[special.type == "trans"]
V = pd.read_csv(f"{sim}/variants.tsv", sep="\t", index_col=0)


def near(v1, v2, w=1_000_000):   # same locus = same chromosome within 1 Mb
    return V.loc[v1, "chrom"] == V.loc[v2, "chrom"] and abs(V.loc[v1, "pos"] - V.loc[v2, "pos"]) <= w


found = sum(any(r.feature_id == t.protein_id and near(r.variant_id, t.variant_id) for r in tr.itertuples()) for t in tt.itertuples())
loci = T("protein_trans_loci") if __import__("os").path.exists(f"{res}/tables/protein_trans_loci.tsv") else tr
loci = loci[(~loci.is_cis) & (loci.fdr_perm <= 0.05)] if "is_cis" in loci else loci
fp_trans = sum(not any(r.feature_id == t.protein_id and near(r.variant_id, t.variant_id) for t in tt.itertuples()) for r in loci.itertuples())
# ---- interaction
it = T("protein_interaction_age_top")
ti = special[special.type == "gxage"]
rows.append({"test": "G x age pQTL features", **prf(it.loc[it.qval_bh <= 0.05, "feature_id"], ti.protein_id, it.feature_id)})

print("=== Feature-level discovery vs truth ===")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"\nLead eQTL variant in LD (r2>=0.8) with true causal variant: {lead_ok:.1%} of true-positive eGenes")
print(f"\ntrans-pQTLs: recovered {found}/{len(tt)} simulated loci; {len(loci)} trans loci called "
      f"({len(tr)} variant pairs), {fp_trans} loci not matching a simulated trans effect")
print("\n=== Sharing category: truth (rows) vs called (columns) ===")
print(conf.to_string())
print(f"Category agreement (excluding 'distinct'): {cmp_[cmp_.mechanism != 'distinct'].match.mean():.1%}")
print("\n=== coloc by true mechanism (median PP) ===")
print(coloc_tab.to_string())
print("\n=== Mediation call (columns) by true mechanism (rows) ===")
print(med_tab.to_string())
