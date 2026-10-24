"""Simulate a multi-omic QTL study with known ground truth (for validation, tutorials and benchmarking).

Architecture per gene/protein pair (each a distinct regulatory mechanism):
  * ``mediated``   – cis-eQTL whose effect propagates to the protein via mRNA (shared eQTL/pQTL, coloc H4)
  * ``eqtl_only``  – cis-eQTL, protein level buffered (no transcript -> protein coupling for that gene)
  * ``pqtl_only``  – variant acts on the protein only (e.g. coding/PTM/stability; post-transcriptional)
  * ``distinct``   – eQTL and pQTL driven by different causal variants in the same window (coloc H3)
  * ``null``       – no cis genetic effect
Plus a few trans-pQTLs and genotype x age interaction pQTLs (ontogeny-type effects).
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


def simulate_genotypes(n_samples: int, n_chrom: int = 3, n_var_per_chrom: int = 2000, chrom_len: int = 30_000_000,
                       rng=None):
    """Haplotype-based genotypes with LD: latent AR(1) Gaussian per haplotype, thresholded at allele frequency."""
    rng = rng or np.random.default_rng(0)
    Gs, V = [], []
    for c in range(1, n_chrom + 1):
        pos = np.sort(rng.choice(np.arange(10_000, chrom_len), n_var_per_chrom, replace=False))
        maf = rng.uniform(0.05, 0.5, n_var_per_chrom)
        gaps = np.diff(pos, prepend=pos[0])
        rho = np.exp(-gaps / 40_000.0)                       # LD decays with distance
        rho[rng.random(n_var_per_chrom) < 0.08] = 0          # recombination hotspots
        H = np.empty((2 * n_samples, n_var_per_chrom))
        z = rng.standard_normal(2 * n_samples)
        for j in range(n_var_per_chrom):
            z = rho[j] * z + np.sqrt(1 - rho[j] ** 2) * rng.standard_normal(2 * n_samples)
            H[:, j] = z
        thr = norm.ppf(maf)
        A = (H < thr).astype(float)
        G = A[:n_samples] + A[n_samples:]
        Gs.append(G.T)
        V.append(pd.DataFrame({"chrom": str(c), "pos": pos, "ref": "A", "alt": "G"}))
    G = np.vstack(Gs)
    V = pd.concat(V, ignore_index=True)
    V.index = [f"rs{1000000 + i}" for i in range(len(V))]
    V.index.name = "variant_id"
    return G, V


def simulate_study(outdir: str = None, n_samples: int = 300, n_chrom: int = 3, n_var_per_chrom: int = 2000,
                   n_genes: int = 300, n_proteins: int = 200, seed: int = 42, missing_genotype_rate: float = 0.002) -> Dict:
    rng = np.random.default_rng(seed)
    samples = [f"S{i:04d}" for i in range(n_samples)]
    G, V = simulate_genotypes(n_samples, n_chrom, n_var_per_chrom, rng=rng)
    chrom_len = int(V.pos.max())

    # ---- genes --------------------------------------------------------------------------------------------
    gchrom = rng.integers(1, n_chrom + 1, n_genes).astype(str)
    gstart = rng.integers(1_500_000, chrom_len - 1_500_000, n_genes)
    genes = pd.DataFrame({"chrom": gchrom, "start": gstart, "end": gstart + rng.integers(5_000, 80_000, n_genes),
                          "strand": rng.choice(["+", "-"], n_genes)},
                         index=pd.Index([f"ENSG{100000 + i}" for i in range(n_genes)], name="feature_id"))
    genes["gene_id"] = genes.index
    genes["symbol"] = [f"GENE{i + 1}" for i in range(n_genes)]

    # ---- covariates ---------------------------------------------------------------------------------------
    age = rng.uniform(0, 18, n_samples)                      # e.g. pediatric ontogeny cohort (years)
    sex = rng.choice(["F", "M"], n_samples)
    batch = rng.choice(["B1", "B2", "B3"], n_samples)
    cov = pd.DataFrame({"age": np.round(age, 2), "sex": sex, "batch": batch}, index=pd.Index(samples, name="sample_id"))
    sexn = (sex == "M").astype(float)
    bmat = pd.get_dummies(batch).values.astype(float)

    def cis_candidates(chrom, start, w=300_000):
        idx = np.where((V.chrom.values == chrom) & (np.abs(V.pos.values - start) <= w))[0]
        return idx

    # ---- assign mechanisms ---------------------------------------------------------------------------------
    mech = np.array(["no_cis_qtl"] * n_genes, dtype=object)
    prot_genes = np.arange(n_proteins)                     # first n_proteins genes have a measured protein
    perm = rng.permutation(prot_genes)
    k1, k2, k3, k4 = int(0.25 * n_proteins), int(0.15 * n_proteins), int(0.15 * n_proteins), int(0.08 * n_proteins)
    mech[perm[:k1]] = "mediated"
    mech[perm[k1:k1 + k2]] = "eqtl_only"
    mech[perm[k1 + k2:k1 + k2 + k3]] = "pqtl_only"
    mech[perm[k1 + k2 + k3:k1 + k2 + k3 + k4]] = "distinct"
    rest = np.arange(n_proteins, n_genes)
    mech[rng.choice(rest, int(0.4 * len(rest)), replace=False)] = "eqtl_no_protein"

    truth_rows = []
    E = np.zeros((n_genes, n_samples))
    Pextra = np.zeros((n_proteins, n_samples))
    coupling = rng.uniform(0.5, 0.8, n_genes)               # mRNA -> protein coupling (SD units)
    for i in range(n_genes):
        cand = cis_candidates(genes.chrom.iloc[i], genes.start.iloc[i])
        maf = np.minimum(G[cand].mean(1) / 2, 1 - G[cand].mean(1) / 2)
        cand = cand[maf >= 0.1]
        if len(cand) < 3:
            mech[i] = "no_cis_qtl"
            continue
        m = mech[i]
        e_var = p_var = None
        be = rng.choice([-1, 1]) * rng.uniform(0.35, 0.8)
        bp = rng.choice([-1, 1]) * rng.uniform(0.35, 0.8)
        if m in ("mediated", "eqtl_only", "eqtl_no_protein"):
            e_var = rng.choice(cand)
        elif m == "pqtl_only":
            p_var = rng.choice(cand)
        elif m == "distinct":
            e_var = rng.choice(cand)
            far = cand[np.abs(V.pos.values[cand] - V.pos.values[e_var]) > 150_000]
            far = far[np.abs(np.corrcoef(G[far], G[e_var][None])[-1, :-1]) < 0.1] if len(far) else far
            if len(far) == 0:
                mech[i] = "mediated"; m = "mediated"
            else:
                p_var = rng.choice(far)
        if e_var is not None:
            g = G[e_var] - G[e_var].mean()
            E[i] += be * g / g.std()
        if p_var is not None and i < n_proteins:
            g = G[p_var] - G[p_var].mean()
            Pextra[i] += bp * g / g.std()
        truth_rows.append({"gene_feature_id": genes.index[i], "protein_id": f"P{i:05d}" if i < n_proteins else None,
                           "mechanism": m, "eqtl_variant": V.index[e_var] if e_var is not None else None,
                           "eqtl_beta_sd": be if e_var is not None else 0.0,
                           "pqtl_variant": V.index[p_var] if p_var is not None else (V.index[e_var] if m == "mediated" else None),
                           "pqtl_direct_beta_sd": bp if p_var is not None else 0.0,
                           "coupling": 0.0 if m == "eqtl_only" else coupling[i]})
    truth = pd.DataFrame(truth_rows)

    # ---- transcript levels -> counts ------------------------------------------------------------------------
    H_e = rng.standard_normal((5, n_samples))                # hidden factors
    Le = rng.normal(0, 0.35, (n_genes, 5))
    noise_e = rng.standard_normal((n_genes, n_samples))
    Ez = (E + Le @ H_e
          + 0.25 * np.outer(rng.standard_normal(n_genes), sexn)
          + 0.2 * np.outer(rng.standard_normal(n_genes), (age - 9) / 5)
          + rng.normal(0, 0.2, (n_genes, 3)) @ bmat.T)
    latent_mrna = Ez + noise_e                               # SD-ish units
    base = rng.uniform(3, 10, n_genes)                       # log2 CPM baseline
    logcpm = base[:, None] + 0.6 * latent_mrna
    lib = rng.uniform(15e6, 40e6, n_samples)
    counts = rng.poisson(np.exp2(logcpm) * lib / 1e6)
    rna = pd.DataFrame(counts, index=genes.index, columns=samples)

    # ---- proteins -------------------------------------------------------------------------------------------
    pid = [f"P{i:05d}" for i in range(n_proteins)]
    prot_annot = genes.iloc[:n_proteins][["chrom", "start", "end", "strand", "gene_id", "symbol"]].copy()
    prot_annot.index = pd.Index(pid, name="feature_id")
    lm = (latent_mrna[:n_proteins] - latent_mrna[:n_proteins].mean(1, keepdims=True)) / latent_mrna[:n_proteins].std(1, keepdims=True)
    cpl = coupling[:n_proteins].copy()
    cpl[mech[:n_proteins] == "eqtl_only"] = 0.0
    H_p = rng.standard_normal((4, n_samples))
    Lp = rng.normal(0, 0.35, (n_proteins, 4))
    Pz = cpl[:, None] * lm + Pextra + Lp @ H_p + 0.15 * np.outer(rng.standard_normal(n_proteins), (age - 9) / 5) \
        + rng.normal(0, 0.15, (n_proteins, 3)) @ bmat.T + 0.8 * rng.standard_normal((n_proteins, n_samples))

    # trans pQTLs
    trans_rows = []
    nonnull = np.where(mech[:n_proteins] == "no_cis_qtl")[0]
    for t in rng.choice(nonnull, min(5, len(nonnull)), replace=False):
        pc = prot_annot.chrom.iloc[t]
        cand = np.where((V.chrom.values != pc) & (np.minimum(G.mean(1) / 2, 1 - G.mean(1) / 2) > 0.2))[0]
        v = rng.choice(cand)
        g = (G[v] - G[v].mean()) / G[v].std()
        b = rng.choice([-1, 1]) * 0.6
        Pz[t] += b * g
        trans_rows.append({"protein_id": pid[t], "variant_id": V.index[v], "beta_sd": b, "type": "trans"})
    # genotype x age interaction pQTLs (ontogeny)
    agez = (age - age.mean()) / age.std()
    for t in rng.choice(np.where(mech[:n_proteins] == "no_cis_qtl")[0], 5, replace=False):
        cand = cis_candidates(prot_annot.chrom.iloc[t], prot_annot.start.iloc[t])
        cand = cand[np.minimum(G[cand].mean(1) / 2, 1 - G[cand].mean(1) / 2) > 0.2]
        if len(cand) == 0:
            continue
        v = rng.choice(cand)
        g = (G[v] - G[v].mean()) / G[v].std()
        Pz[t] += 0.6 * g * agez
        trans_rows.append({"protein_id": pid[t], "variant_id": V.index[v], "beta_sd": 0.6, "type": "gxage"})
    special = pd.DataFrame(trans_rows)

    logint = 22 + 1.5 * rng.standard_normal(n_proteins)[:, None] + 0.8 * Pz
    # MNAR missingness (low-abundance values more often missing) + 1% MCAR
    pmiss = 1 / (1 + np.exp((logint - 19.0) * 2.0))
    miss = (rng.random(logint.shape) < pmiss) | (rng.random(logint.shape) < 0.01)
    inten = np.exp2(logint)
    inten[miss] = np.nan
    prot = pd.DataFrame(inten, index=pid, columns=samples)

    # ---- genotype missingness ---------------------------------------------------------------------------------
    Gm = G.copy().astype(float)
    Gm[rng.random(Gm.shape) < missing_genotype_rate] = np.nan
    geno = pd.DataFrame(Gm, index=V.index, columns=samples)

    out ={"genotypes": geno, "variants": V, "rna_counts": rna, "gene_annot": genes, "protein_intensity": prot,
           "protein_annot": prot_annot, "covariates": cov, "truth": truth, "truth_special": special}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        geno.to_csv(f"{outdir}/genotypes.tsv.gz", sep="\t", float_format="%.0f")
        V.to_csv(f"{outdir}/variants.tsv", sep="\t")
        rna.to_csv(f"{outdir}/rna_counts.tsv.gz", sep="\t")
        genes.to_csv(f"{outdir}/gene_annotation.tsv", sep="\t")
        prot.to_csv(f"{outdir}/protein_intensity.tsv.gz", sep="\t", float_format="%.1f")
        prot_annot.to_csv(f"{outdir}/protein_annotation.tsv", sep="\t")
        cov.to_csv(f"{outdir}/covariates.tsv", sep="\t")
        truth.to_csv(f"{outdir}/truth_cis.tsv", sep="\t", index=False)
        special.to_csv(f"{outdir}/truth_trans_interaction.tsv", sep="\t", index=False)
    return out
