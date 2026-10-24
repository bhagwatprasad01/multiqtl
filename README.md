# MultiQTL

Integrated **genomics × transcriptomics × proteomics** QTL analysis in pure Python (NumPy/SciPy/pandas/matplotlib).
One YAML config takes you from genotypes, RNA-seq counts and LC-MS protein intensities to cis/trans/interaction
eQTLs and pQTLs, a full multiple-testing suite, eQTL–pQTL integration (colocalization, SMR/HEIDI, Mendelian
randomization, mediation), and publication-ready tables, figures and an HTML report.

```
genotypes (VCF / dosage / PLINK .raw) ─┐
RNA-seq counts ──── TMM → INT ─────────┼─► covariates (known + genotype PCs + hidden factors)
LC-MS intensities ─ log2 → filter →    │        │
                    MNAR impute → INT ─┘        ▼
                          cis nominal · cis permutation (beta-approx) · conditional signals · trans · G×E
                                                │
                                   FDR suite (q-value, BH, BY, eigenMT, hierarchical, permutation, lfdr…)
                                                │
                    eQTL–pQTL integration: sharing, π₁, coloc.abf, SMR/HEIDI, MR, mediation, mRNA–protein ρ
                                                │
                     TSV/CSV/Parquet/JSON · formatted Excel · 600-dpi PNG/TIFF · PDF/SVG/EPS · HTML report
```

## Install and run

MultiQTL is a standard Python package: it needs only Python ≥ 3.9 and common scientific libraries, runs offline on
Windows, macOS and Linux, and has three equivalent front ends – a **browser interface**, a **command line** and a
**Python API**. Non-programmers should start with **[USER_GUIDE.md](USER_GUIDE.md)**.

| Route | Install | Start |
|---|---|---|
| Windows, one click | double-click `install\install_windows.bat` | `install\start_multiqtl_windows.bat` |
| macOS / Linux | `bash install/install_mac_linux.sh` | `install/start_multiqtl_mac_linux.command` |
| pip | `pip install -r requirements.txt && pip install -e .` | `multiqtl gui` |
| conda | `conda env create -f environment.yml` | `conda activate multiqtl && multiqtl gui` |
| Docker | `docker build -t multiqtl .` | `docker run --rm -v "$PWD":/data multiqtl run /data/config.yaml` |

`python -m multiqtl …` works everywhere the `multiqtl` launcher is not on PATH.

```bash
multiqtl gui                              # browser interface (local only, http://127.0.0.1:8765)
multiqtl demo                             # simulate a study and analyse it end to end
multiqtl init my.yaml                     # commented settings template
multiqtl check my.yaml                    # plain-language validation of every input file
multiqtl run my.yaml                      # full pipeline
multiqtl run my.yaml -o out --n-perm 10000 --dpi 1200 --formats png,pdf,tiff,eps
multiqtl fdr results.tsv adjusted.tsv --p-col pval --group-col feature_id   # any table, all corrections
multiqtl coloc eqtl_sumstats.tsv pqtl_sumstats.tsv --p12 5e-6              # variant_id, beta, se
multiqtl power --n 100,300,1000 --alpha 5e-8
```

`examples/tutorial.ipynb` walks through the Python API step by step; `templates/` holds example input files.
Settings files may be YAML or JSON, and relative paths inside them are resolved from the file's own folder.

## Inputs

| Input | Format |
|---|---|
| Genotypes | `.vcf(.gz)` (DS or GT), or dosage table (variants × samples, 0–2) + variant table (`variant_id, chrom, pos[, ref, alt]`), or PLINK `--recode A` `.raw` via `io.read_plink_raw` |
| Molecular layers | features × samples. `type: rnaseq` (raw counts), `proteomics` (raw intensities, 0/NA = missing) or `normalized` |
| Feature annotation | `feature_id, chrom, start` (TSS/gene start = cis anchor) `[, end, strand]`, and `gene_id` linking proteins to transcripts (`;`-separated for protein groups) |
| Covariates | samples × covariates; categorical columns are one-hot encoded automatically |

Tables may be TSV, CSV, TXT (delimiter auto-detected; optionally gzipped), XLSX or Parquet. Samples are matched by
name across all inputs (numeric IDs are fine; files saved from Excel with a byte-order mark are handled).

## Analyses

**Preprocessing.** Genotype QC (MAF, call rate, HWE χ² or Wigginton exact test), mean imputation of residual
missing dosages. RNA-seq: TMM (edgeR algorithm) → expression filter → rank inverse-normal transform (GTEx v8).
Proteomics: log2 → missingness filter → median/quantile centring → left-censored imputation (Perseus down-shift,
min, half-min or kNN) → INT/z-score. Hidden confounders as phenotype PCs, number chosen by the Buja–Eyuboglu
permutation rule (`auto`), the scree elbow, or fixed.

**Mapping** (`mapping.QTLMapper`). Exact OLS (`y ~ g + covariates`) for every pair via Frisch–Waugh–Lovell
residualization, so millions of tests are matrix products.
- *cis nominal* — every variant within ±window of the feature.
- *cis permutation* — FastQTL/tensorQTL feature-level test: empirical and Beta-approximated permutation P, eigenMT
  P, Storey q, and the per-feature nominal threshold used to call significant variant pairs.
- *independent signals* — forward–backward conditional regression.
- *trans* — chunked genome-wide scan; empirical FDR from sample-label permutations (shared across features to keep
  their correlation), Bonferroni, and clumping into independent trans loci.
- *interaction QTLs* — genotype × covariate (age/ontogeny, sex, disease, cell-type proportion…), eigenMT + BH.

**FDR and multiple testing** (`fdr`). Bonferroni, Holm, Šidák, Benjamini–Hochberg, Benjamini–Yekutieli,
Storey q-values (π₀ by smoother, bootstrap or fixed λ; `auto` picks fixed λ below 1,000 tests where the
smoother's extrapolation is unstable), local FDR, TreeQTL-style hierarchical FDR, permutation FDR, eigenMT
effective tests, π₁ replication. Every feature-level table carries all corrections side by side (`fdr_*`).

**Integration** (`integration`).
- Protein ↔ transcript mapping, covariate-adjusted mRNA–protein correlation.
- eQTL vs pQTL: lead cross-lookup, direction concordance + sign test, π₁ in both directions, effect
  heterogeneity z corrected for sample overlap (ρ estimated from null z-scores), categories
  *shared / eQTL-only (buffered) / pQTL-only (post-transcriptional) / discordant*.
- **coloc.abf** (PP.H0–H4, 95% credible set, prior sensitivity).
- **SMR** with an approximate HEIDI test (LD from your genotypes).
- **Mendelian randomization** mRNA → protein: Wald ratio, IVW (random effects), MR-Egger, weighted median, using
  independent cis-eQTL instruments.
- **Mediation** of each lead pQTL through its transcript: a, b, direct, total, indirect, Sobel and bootstrap CI,
  proportion mediated, and a mechanism call (*transcript-mediated / partially mediated / protein-specific*).

**Statistics** (`stats`). λ_GC with bootstrap CI, QQ with null band, KS uniformity, TSS-distance enrichment,
MAF–effect relationship, direction bias, variance explained, bootstrap effect CIs, analytical power and minimum
detectable effect.

## Outputs (`output_dir/`)

| Path | Content |
|---|---|
| `tables/*.tsv` (+ csv / tsv.gz / parquet / json) | every result table; see the README sheet of the workbook |
| `multiqtl_results.xlsx` | formatted workbook (frozen headers, filters, scientific P formats, README sheet) |
| `figures/` | each figure in PNG + TIFF (LZW) at 600 dpi and vector PDF/SVG (EPS optional); editable text (Type 42) |
| `figures/locus/` | regional eQTL/pQTL plots coloured by LD, credible sets, genotype-effect plots |
| `report.html` | self-contained report: QC, cis, FDR, trans, G×E, integration, statistics, methods |
| `summary.json`, `config_used.json`, `multiqtl.log` | machine-readable summary and full provenance |

Figures: Manhattan, QQ (per layer and genome-wide), P-value histograms with π₀, discoveries by FDR method,
volcano, TSS distance, cis/trans map, eQTL-vs-pQTL effect scatter, sharing categories, mRNA–protein correlation,
coloc posteriors, mediation forest, MR scatter, G×E plot, power heatmap. Colours use a fixed colour-blind-validated
categorical order and single-hue ramps for magnitude.

## Python API

```python
from multiqtl import preprocess as PP, fdr as F, integration as I
from multiqtl.mapping import QTLMapper

G, qc = PP.genotype_qc(genotypes)
X, _ = PP.normalize_proteomics(protein_intensity)
cov = PP.encode_covariates(covariates)
M = QTLMapper(G, variants.loc[G.index], X, protein_annot.loc[X.index], cov, name="protein")
perm = M.map_cis_permutation(window=1_000_000, n_perm=1000)
perm["bh"] = F.bh(perm.pval_beta)
res = I.coloc_abf(beta_e, se_e, beta_p, se_p)
```

## Validation (simulated study, 300 samples, 6,000 variants, 300 genes, 200 proteins)

Benchmark from `examples/validate_against_truth.py` on the bundled example run:

| Check | Result |
|---|---|
| eGenes at q ≤ 0.05 | recall 100%, FDP 7.5% |
| pGenes at q ≤ 0.05 | recall 88%, FDP 3.7% |
| Lead variant in LD (r² ≥ 0.8) with the causal variant | 99.3% |
| G × age pQTLs | 5/5, no false positives |
| trans-pQTL loci | 4/5 recovered, 1 false locus |
| coloc, mediated (shared causal) genes | median PP.H4 = 0.98 |
| coloc, distinct causal variants | PP.H3 > 0.8 in 14/14 |
| mediation, mediated vs protein-specific | 35/37 and 27/28 classified correctly |

Across six replicate simulations the mean eGene FDP at q ≤ 0.05 was 5.9% (recall 99.8%); feature-level null
P values are uniform. For a stricter feature-level call use `fdr_bh` or `qval_eigenmt_bh` (2.9% FDP in the same run).
Regression estimates match `numpy.linalg.lstsq` to 10 decimal places (`tests/`).

```bash
python -m unittest discover -s tests -v     # 23 tests: OLS equivalence, FDR control, coloc, MR, input checks, GUI, end-to-end
```

## Limitations

- Unrelated samples assumed: no kinship/linear-mixed-model correction. Remove relatives or include
  genotype PCs; for family cohorts a LMM tool is the right choice.
- coloc.abf assumes at most one causal variant per trait in the region; use the conditional signals table to
  spot loci with several.
- HEIDI is an approximation (delta-method covariance, independent-sample assumption). eQTLs and pQTLs measured
  on the same individuals violate that assumption; treat HEIDI as a flag, not a test.
- Mediation assumes no unmeasured mRNA–protein confounding beyond the covariates.
- All genotypes are held in memory; split very large panels by chromosome.
- Parquet output requires `pyarrow`.

## Key references

Ongen et al. 2016 (FastQTL) · Taylor-Weiner et al. 2019 (tensorQTL) · Davis et al. 2016 (eigenMT) ·
Storey & Tibshirani 2003 · Benjamini & Hochberg 1995 · Benjamini & Yekutieli 2001 · Peterson et al. 2016 (TreeQTL) ·
Giambartolomei et al. 2014 (coloc) · Zhu et al. 2016 (SMR/HEIDI) · Bowden et al. 2015/2016 (MR-Egger, weighted
median) · Robinson & Oshlack 2010 (TMM) · Wigginton et al. 2005 (HWE) · Buja & Eyuboglu 1992 · GTEx Consortium 2020.

MIT licence.
