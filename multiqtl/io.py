"""Input/output: genotype, phenotype (transcript / protein) and covariate loaders.

Conventions used throughout MultiQTL
------------------------------------
* Genotypes   : DataFrame, rows = variants, columns = samples, values = alt-allele dosage (0-2, NaN = missing)
* Variants    : DataFrame indexed by variant_id with columns ``chrom``, ``pos`` (+ optional ``ref``, ``alt``)
* Phenotypes  : DataFrame, rows = features (genes / proteins), columns = samples
* Feature annot: DataFrame indexed by feature_id with ``chrom``, ``start`` (TSS / gene start, used as the cis
                anchor), optional ``end``, ``strand``, and ``gene_id`` (links a protein to its coding gene)
* Covariates  : DataFrame, rows = samples, columns = covariates
"""
from __future__ import annotations

import gzip
import io as _io
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------------------------------
# Generic table reader
# ----------------------------------------------------------------------------------------------------------------
def read_table(path: str, index_col: Optional[int | str] = 0, **kw) -> pd.DataFrame:
    """Read TSV / CSV / TXT (optionally .gz), XLSX or Parquet based on file extension.

    Row and column labels are always returned as stripped strings, so numeric sample IDs (e.g. 1001) match
    across files; a UTF-8 byte-order mark (common in files saved from Excel) is ignored. For .txt files the
    delimiter (tab, comma or semicolon) is detected automatically."""
    p = path.lower()
    if p.endswith(".parquet") or p.endswith(".pq"):
        df = pd.read_parquet(path)
        if index_col is not None:
            df = df.set_index(df.columns[index_col] if isinstance(index_col, int) else index_col)
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        df = pd.read_excel(path, index_col=index_col, **kw)
    else:
        stem = p[:-3] if p.endswith(".gz") else p
        if stem.endswith(".csv"):
            sep = ","
        elif stem.endswith((".tsv", ".tab")):
            sep = "\t"
        else:
            sep = None                                   # sniff delimiter (.txt etc.)
        engine = "python" if sep is None else "c"
        df = pd.read_csv(path, sep=sep, index_col=index_col, encoding="utf-8-sig", engine=engine, **kw)
    if index_col is not None:
        df.index = df.index.map(lambda x: str(x).strip())
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# VCF reader (dosage)
# ----------------------------------------------------------------------------------------------------------------
def read_vcf_dosage(path: str, field: str = "auto", chunksize: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read a (b)gzipped or plain VCF into a dosage matrix.

    ``field`` = "DS" (imputed dosage), "GT" (hard calls), or "auto" (DS if present, otherwise GT).
    Multi-allelic records are skipped. Returns (genotypes, variants).
    """
    samples, rows, meta = None, [], []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            f = line.rstrip("\n").split("\t")
            if "," in f[4]:
                continue
            fmt = f[8].split(":")
            use = field
            if use == "auto":
                use = "DS" if "DS" in fmt else "GT"
            idx = fmt.index(use)
            vals = np.empty(len(samples))
            for j, s in enumerate(f[9:]):
                parts = s.split(":")
                v = parts[idx] if idx < len(parts) else "."
                if use == "DS":
                    vals[j] = np.nan if v in (".", "") else float(v)
                else:
                    a = v.replace("|", "/").split("/")
                    vals[j] = np.nan if "." in a else sum(int(x) > 0 for x in a)
            vid = f[2] if f[2] != "." else f"{f[0]}:{f[1]}:{f[3]}:{f[4]}"
            rows.append(vals)
            meta.append((vid, f[0], int(f[1]), f[3], f[4]))
    variants = pd.DataFrame(meta, columns=["variant_id", "chrom", "pos", "ref", "alt"]).set_index("variant_id")
    geno = pd.DataFrame(np.vstack(rows) if rows else np.empty((0, len(samples))), index=variants.index, columns=samples)
    return geno, variants


def read_plink_raw(path: str) -> pd.DataFrame:
    """Read a PLINK ``--recode A`` .raw file -> dosage matrix (variants x samples)."""
    df = pd.read_csv(path, sep=r"\s+")
    df.index = df["IID"].astype(str)
    df = df.drop(columns=[c for c in ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"] if c in df.columns])
    df.columns = [c.rsplit("_", 1)[0] for c in df.columns]
    return df.T


def write_vcf_dosage(geno: pd.DataFrame, variants: pd.DataFrame, path: str):
    """Minimal VCF writer (DS field) – useful for interoperability and tests."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Alt dosage\">\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(geno.columns) + "\n")
        for vid, row in geno.iterrows():
            v = variants.loc[vid]
            ds = "\t".join("." if np.isnan(x) else f"{x:.3f}" for x in row.values)
            fh.write(f"{v['chrom']}\t{int(v['pos'])}\t{vid}\t{v.get('ref', 'A')}\t{v.get('alt', 'G')}\t.\tPASS\t.\tDS\t{ds}\n")


# ----------------------------------------------------------------------------------------------------------------
# Dataset container
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Molecular:
    """One molecular layer (e.g. RNA-seq or LC-MS proteomics)."""
    name: str
    data: pd.DataFrame          # features x samples (normalized once preprocessing has run)
    annot: pd.DataFrame         # feature annotation
    raw: Optional[pd.DataFrame] = None
    qc: Dict[str, object] = field(default_factory=dict)


@dataclass
class QTLDataset:
    genotypes: pd.DataFrame
    variants: pd.DataFrame
    layers: Dict[str, Molecular]
    covariates: Optional[pd.DataFrame] = None
    qc: Dict[str, object] = field(default_factory=dict)

    @property
    def samples(self):
        return list(self.genotypes.columns)

    def align(self) -> "QTLDataset":
        """Restrict every component to the intersection of samples (keeps genotype order)."""
        common = set(self.genotypes.columns)
        for L in self.layers.values():
            common &= set(L.data.columns)
        if self.covariates is not None:
            common &= set(self.covariates.index)
        order = [s for s in self.genotypes.columns if s in common]
        if len(order) < 10:
            raise ValueError(f"Only {len(order)} samples shared across genotype/phenotype/covariate files")
        self.genotypes = self.genotypes[order]
        for L in self.layers.values():
            L.data = L.data[order]
            if L.raw is not None:
                L.raw = L.raw[[c for c in order if c in L.raw.columns]]
        if self.covariates is not None:
            self.covariates = self.covariates.loc[order]
        self.variants = self.variants.loc[self.genotypes.index]
        for L in self.layers.values():
            missing = set(L.data.index) - set(L.annot.index)
            if missing:
                L.data = L.data.drop(index=list(missing))
            L.annot = L.annot.loc[L.data.index]
        self.qc["n_samples"] = len(order)
        return self


def _standardize_annot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("chr", "chromosome", "#chr", "seqname"):
            ren[c] = "chrom"
        elif lc in ("tss", "position", "start_pos", "gene_start"):
            ren[c] = "start"
        elif lc in ("bp", "position_bp"):
            ren[c] = "pos"
        elif lc in ("gene", "gene_name", "ensembl_gene_id", "geneid"):
            ren[c] = "gene_id"
    df = df.rename(columns=ren)
    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)
    return df


def load_dataset(genotypes: str, variants: Optional[str], layers: Dict[str, Dict[str, str]],
                 covariates: Optional[str] = None) -> QTLDataset:
    """Load everything from file paths.

    ``layers`` = {"expression": {"data": "rna_counts.tsv", "annot": "genes.tsv"},
                  "protein":    {"data": "protein_intensity.tsv", "annot": "proteins.tsv"}}
    ``genotypes`` may be a VCF (then ``variants`` is ignored) or a dosage table + ``variants`` table.
    """
    if genotypes.endswith((".vcf", ".vcf.gz")):
        G, V = read_vcf_dosage(genotypes)
    else:
        G = read_table(genotypes)
        V = _standardize_annot(read_table(variants))
    V["chrom"] = V["chrom"].astype(str).str.replace("chr", "", regex=False)
    Ls = {}
    for name, spec in layers.items():
        d = read_table(spec["data"])
        a = _standardize_annot(read_table(spec["annot"]))
        Ls[name] = Molecular(name=name, data=d, annot=a, raw=d.copy())
    C = read_table(covariates) if covariates else None
    return QTLDataset(genotypes=G, variants=V, layers=Ls, covariates=C).align()
