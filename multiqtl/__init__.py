"""MultiQTL — integrated genomics × transcriptomics × proteomics QTL analysis.

Modules
-------
io           loaders for VCF / dosage tables / PLINK .raw, phenotype and covariate tables
preprocess   genotype QC (MAF, call rate, HWE), RNA-seq (TMM + INT) and LC-MS proteomics normalization,
             covariates and hidden factors
mapping      vectorized cis (nominal, permutation, conditional), trans and interaction QTL mapping
fdr          BH, BY, Bonferroni, Holm, Šidák, Storey q-value, local FDR, hierarchical FDR, permutation FDR, eigenMT
stats        calibration (lambda_GC, QQ), enrichment, effect-size architecture, power
integration  eQTL/pQTL comparison, coloc.abf, SMR/HEIDI, Mendelian randomization, mediation, mRNA–protein correlation
plotting     publication-quality figures (PNG/TIFF at 600 dpi, vector PDF/SVG/EPS)
export       TSV/CSV/Parquet/JSON/formatted Excel, self-contained HTML report
pipeline     end-to-end orchestration from a YAML config
simulate     simulated multi-omic QTL study with ground truth
"""
__version__ = "0.1.0"

from . import fdr, integration, io, mapping, plotting, preprocess, stats  # noqa: F401
from .mapping import QTLMapper, significant_pairs  # noqa: F401
from .pipeline import run_pipeline, load_config  # noqa: F401
