"""Config helpers for standalone use: templates, path resolution, and plain-language input checks."""
from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TEMPLATE_YAML = """\
# MultiQTL configuration
# Paths may be absolute or relative to THIS file's folder.
# Lines starting with # are comments. Remove a layer block if you only have one data type.

output_dir: results

genotypes: data/genotypes.vcf.gz          # VCF (.vcf / .vcf.gz) OR a dosage table (variants x samples, 0/1/2)
variants:                                 # only for dosage tables: variant_id, chrom, pos

layers:
  expression:                             # transcript layer (remove block if not available)
    type: rnaseq                          # rnaseq = raw counts | normalized = already normalized values
    data: data/rna_counts.tsv             # genes x samples
    annot: data/gene_annotation.tsv       # feature_id, chrom, start, end, strand, gene_id, symbol
  protein:                                # protein layer (remove block if not available)
    type: proteomics                      # proteomics = raw LC-MS intensities | normalized
    data: data/protein_intensity.tsv      # proteins x samples; 0 or empty = not detected
    annot: data/protein_annotation.tsv    # feature_id, chrom, start, end, strand, gene_id, symbol

covariates:
  known: data/covariates.tsv              # samples x covariates (age, sex, batch ...), or leave empty
  genotype_pcs: 3
  hidden_factors: auto

cis: {window: 1000000, n_perm: 1000, fdr: 0.05}
trans: {enabled: true, pval_threshold: 1.0e-5, fdr: 0.05}
interaction: {enabled: false, covariate: age, layers: [protein]}
integration: {transcript_layer: expression, protein_layer: protein}

output:
  table_formats: [tsv, xlsx]
  figure_formats: [png, pdf, svg, tiff]
  dpi: 600
"""


def write_template(path: str = "multiqtl_config.yaml") -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(TEMPLATE_YAML)
    return path


def read_config_file(path: str) -> Dict:
    """Read YAML or JSON. YAML needs PyYAML; JSON works with the standard library alone."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if path.lower().endswith(".json"):
        cfg = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as e:
            raise SystemExit("Reading a .yaml config needs PyYAML (pip install pyyaml), or save the config as .json") from e
        cfg = yaml.safe_load(text) or {}
    return resolve_paths(cfg, os.path.dirname(os.path.abspath(path)))


def resolve_paths(cfg: Dict, base: str) -> Dict:
    """Make every input/output path absolute relative to the config file's folder."""
    cfg = copy.deepcopy(cfg)

    def fix(p):
        if not p or not isinstance(p, str):
            return p
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    for k in ("genotypes", "variants", "output_dir"):
        if k in cfg:
            cfg[k] = fix(cfg[k])
    for spec in (cfg.get("layers") or {}).values():
        for k in ("data", "annot"):
            if k in spec:
                spec[k] = fix(spec[k])
    if cfg.get("covariates") and cfg["covariates"].get("known"):
        cfg["covariates"]["known"] = fix(cfg["covariates"]["known"])
    return cfg


# ================================================================================================================
def check_inputs(cfg: Dict, max_rows: int = 2000) -> List[Tuple[str, str]]:
    """Validate a config and its input files without running the analysis.

    Returns a list of (level, message) with level in {"ERROR", "WARNING", "OK"}."""
    from .io import _standardize_annot, read_table, read_vcf_dosage
    out: List[Tuple[str, str]] = []
    E = lambda m: out.append(("ERROR", m))
    W = lambda m: out.append(("WARNING", m))
    OK = lambda m: out.append(("OK", m))

    def exists(label, p):
        if not p:
            E(f"{label}: no file given")
            return False
        if not os.path.exists(p):
            E(f"{label}: file not found -> {p}")
            return False
        return True

    # ---- genotypes
    gsamples = None
    g = cfg.get("genotypes")
    if exists("genotypes", g):
        try:
            if g.endswith((".vcf", ".vcf.gz")):
                G, V = read_vcf_dosage(g)
                OK(f"genotypes: VCF with {G.shape[0]} variants x {G.shape[1]} samples")
            else:
                G = read_table(g)
                vals = G.iloc[:max_rows].apply(pd.to_numeric, errors="coerce")
                if vals.isna().values.mean() > 0.5:
                    E("genotypes: values are not numeric dosages (expected 0, 1, 2 or decimals between 0 and 2)")
                elif np.nanmax(vals.values) > 2.01 or np.nanmin(vals.values) < -0.01:
                    E("genotypes: dosages must lie between 0 and 2")
                else:
                    OK(f"genotypes: {G.shape[0]} variants x {G.shape[1]} samples")
                if exists("variants", cfg.get("variants")):
                    V = _standardize_annot(read_table(cfg["variants"]))
                    miss = [c for c in ("chrom", "pos") if c not in V.columns]
                    if miss:
                        E(f"variants: missing column(s) {miss} (need variant_id, chrom, pos)")
                    else:
                        n_ann = len(set(G.index) & set(V.index))
                        (OK if n_ann == len(G) else W)(f"variants: {n_ann}/{len(G)} genotype rows have a position")
            gsamples = set(map(str, G.columns))
        except Exception as e:
            E(f"genotypes: could not read file ({e.__class__.__name__}: {e})")

    # ---- covariates
    csamples = None
    cv = (cfg.get("covariates") or {}).get("known")
    if cv:
        if exists("covariates", cv):
            C = read_table(cv)
            csamples = set(map(str, C.index))
            OK(f"covariates: {C.shape[1]} columns ({', '.join(map(str, C.columns[:8]))}) for {C.shape[0]} samples")
            ic = (cfg.get("interaction") or {})
            if ic.get("enabled") and ic.get("covariate") not in C.columns:
                E(f"interaction: covariate '{ic.get('covariate')}' is not a column of the covariate file")
            if ic.get("enabled") and ic.get("covariate") in C.columns and not pd.api.types.is_numeric_dtype(C[ic["covariate"]]):
                E(f"interaction: covariate '{ic['covariate']}' must be numeric")
    elif (cfg.get("interaction") or {}).get("enabled"):
        E("interaction: enabled but no covariate file given")

    # ---- layers
    layers = cfg.get("layers") or {}
    if not layers:
        E("layers: at least one molecular layer (expression and/or protein) is required")
    lsamples, annots = {}, {}
    for name, spec in layers.items():
        typ = spec.get("type", "normalized")
        if typ not in ("rnaseq", "proteomics", "normalized"):
            E(f"{name}: type must be rnaseq, proteomics or normalized (got '{typ}')")
        if exists(f"{name} data", spec.get("data")):
            D = read_table(spec["data"])
            num = D.iloc[:max_rows].apply(pd.to_numeric, errors="coerce")
            if num.isna().values.all():
                E(f"{name} data: no numeric values found – first column must be the feature ID, then one column per sample")
            else:
                if typ == "rnaseq" and (num.fillna(0).values % 1 != 0).mean() > 0.2:
                    W(f"{name} data: type is rnaseq but values are not integer counts – use type: normalized for TPM/FPKM/log values")
                if typ == "proteomics":
                    W_frac = float((num.isna() | (num == 0)).values.mean())
                    OK(f"{name} data: {W_frac:.0%} missing/zero values (filtered & imputed automatically)")
                OK(f"{name} data: {D.shape[0]} features x {D.shape[1]} samples")
            lsamples[name] = set(map(str, D.columns))
        if exists(f"{name} annotation", spec.get("annot")):
            A = _standardize_annot(read_table(spec["annot"]))
            miss = [c for c in ("chrom", "start") if c not in A.columns]
            if miss:
                E(f"{name} annotation: missing column(s) {miss} (need feature_id, chrom, start)")
            else:
                OK(f"{name} annotation: {len(A)} features with positions")
            annots[name] = A
            if name in lsamples and spec.get("data") and os.path.exists(spec["data"]):
                ids = set(read_table(spec["data"]).index)
                n = len(ids & set(A.index))
                (OK if n == len(ids) else (E if n == 0 else W))(f"{name}: {n}/{len(ids)} features found in the annotation")

    # ---- sample overlap
    sets = {"genotypes": gsamples, "covariates": csamples, **lsamples}
    sets = {k: v for k, v in sets.items() if v}
    if len(sets) >= 2:
        common = set.intersection(*sets.values())
        if len(common) < 10:
            E(f"samples: only {len(common)} sample IDs shared across files – column/row names must match exactly")
            ref_name, ref = next(iter(sets.items()))
            for k, v in sets.items():
                if k != ref_name:
                    ex_a, ex_b = sorted(ref)[:2], sorted(v)[:2]
                    E(f"samples: {ref_name} vs {k}: {len(ref & v)} shared (e.g. {ex_a} vs {ex_b})")
        else:
            OK(f"samples: {len(common)} shared across {', '.join(sets)}")
            for k, v in sets.items():
                if len(v - common):
                    W(f"samples: {len(v - common)} {k} samples not in all other files (they will be dropped)")
        if len(common) and len(common) < 50:
            W("samples: fewer than 50 shared samples – power will be low")

    # ---- integration
    ig = cfg.get("integration") or {}
    tl, pl = ig.get("transcript_layer"), ig.get("protein_layer")
    if tl or pl:
        if tl not in layers or pl not in layers:
            W("integration: transcript_layer and protein_layer must both name layers – integration will be skipped")
        elif pl in annots and tl in annots:
            if "gene_id" not in annots[pl].columns:
                E("integration: the protein annotation needs a gene_id column linking each protein to its gene")
            elif "gene_id" in annots[tl].columns:
                pg = set(annots[pl].gene_id.astype(str).str.split(r"[;,]").explode())
                n = len(pg & set(annots[tl].gene_id.astype(str)))
                (OK if n else E)(f"integration: {n} proteins link to a transcript through gene_id")
            else:
                n = len(set(annots[pl].gene_id.astype(str)) & set(map(str, annots[tl].index)))
                (OK if n else E)(f"integration: {n} proteins link to transcript feature IDs through gene_id")
    if not cfg.get("output_dir"):
        W("output_dir not set – results go to ./multiqtl_results")
    return out


def format_report(items: List[Tuple[str, str]]) -> str:
    sym = {"ERROR": "✗", "WARNING": "!", "OK": "✓"}
    lines = [f" {sym[l]} {l:<7} {m}" for l, m in items]
    n_e = sum(l == "ERROR" for l, _ in items)
    n_w = sum(l == "WARNING" for l, _ in items)
    lines.append("")
    lines.append("Ready to run." if n_e == 0 else f"{n_e} error(s) must be fixed before running.")
    if n_w:
        lines.append(f"{n_w} warning(s) – the run will proceed, but check them.")
    return "\n".join(lines)
