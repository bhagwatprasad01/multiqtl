# MultiQTL user guide

This guide is for anyone who wants to run eQTL / pQTL analyses without writing code. Everything runs on your own
computer; no account, internet service or AI assistant is needed once MultiQTL is installed.

---

## 1. Install (once)

You need **Python 3.9 or newer**.

| Your computer | Steps |
|---|---|
| **Windows** | 1. Install Python from <https://www.python.org/downloads/> and tick **“Add python.exe to PATH”**.<br>2. Unzip MultiQTL.<br>3. Double-click `install\install_windows.bat`. |
| **macOS** | 1. Install Python from <https://www.python.org/downloads/>.<br>2. Unzip MultiQTL, open **Terminal** in the folder.<br>3. Run `bash install/install_mac_linux.sh`. |
| **Linux** | `bash install/install_mac_linux.sh` from the unzipped folder. |
| **Conda users** | `conda env create -f environment.yml` then `conda activate multiqtl`. |
| **Docker** | `docker build -t multiqtl .` (see the header of `Dockerfile`). |
| **Institutional cluster / HPC** | `pip install --user -r requirements.txt && pip install --user --no-deps .` then use the command line (section 4). |

The installers create a private environment in the folder `.venv`, so nothing else on your computer is changed.
To uninstall, delete the MultiQTL folder.

---

## 2. Start MultiQTL

- **Windows:** double-click `install\start_multiqtl_windows.bat`
- **macOS:** double-click `install/start_multiqtl_mac_linux.command` (first time: right-click → Open)
- **Linux / any terminal:** `python -m multiqtl gui`

Your web browser opens at `http://127.0.0.1:8765`. This page is served by your own computer – the files you
choose are copied into a run folder in `MultiQTL_runs` in your home folder and never leave the machine.

**Try it first:** press **Run demo**. MultiQTL simulates a study with known true QTLs and analyses it
(about 2–3 minutes), so you can see every output before preparing your own data.

---

## 3. Prepare your files

Open the examples in `templates/` with Excel to see the exact layout. Save your own files as **CSV**,
**tab-separated text**, or **Excel (.xlsx)**. The first column always holds the IDs; the first row holds column
names. Sample names must be spelled **identically** in every file.

| File | Rows | Columns | Template |
|---|---|---|---|
| Genotypes | variants | one per sample, values 0 / 1 / 2 (count of the alternative allele; decimals allowed for imputed dosage) — or simply provide a **VCF** | `genotypes_dosage.csv` |
| Variant positions *(only with a genotype table)* | variants | `variant_id, chrom, pos` (+ `ref, alt`) | `variants.csv` |
| Transcript data | genes | one per sample: raw read counts, or already normalized values | `rna_counts.csv` |
| Gene annotation | genes | `feature_id, chrom, start, end, strand, gene_id, symbol` | `gene_annotation.csv` |
| Protein data | proteins | one per sample: raw LC-MS intensities (0 or blank = not detected) or normalized values | `protein_intensity.csv` |
| Protein annotation | proteins | same as genes; **`gene_id` must match the gene annotation** so proteins link to their transcripts | `protein_annotation.csv` |
| Covariates *(recommended)* | samples | e.g. `age, sex, batch`; text columns are handled automatically | `covariates.csv` |

Notes
- `chrom` may be written `1` or `chr1`; positions must use the **same genome build** as the genotypes.
- For a protein group mapping to several genes, separate gene IDs with `;`.
- You can analyse transcripts only, proteins only, or both. Integration analyses run when both are given.
- Recommended minimum: 80–100 samples; see the power heat map in every report.

---

## 4. Command line (optional, for scripting and large studies)

```bash
python -m multiqtl init my_study.yaml     # writes a commented settings file – edit the file paths
python -m multiqtl check my_study.yaml    # checks every input file and explains any problem
python -m multiqtl run my_study.yaml      # runs everything
python -m multiqtl demo                   # simulated example end to end
```

Paths in the settings file can be relative to the file itself, so a project folder can be moved or shared.
Every run made with the browser interface also saves a `config.yaml` in its run folder: re-running that file
reproduces the analysis exactly.

---

## 5. What you get

Each run folder contains `results/` with:

| Output | Open with | What it is |
|---|---|---|
| `report.html` | any web browser | the overview: QC, results, figures, methods text you can adapt |
| `multiqtl_results.xlsx` | Excel / LibreOffice | every results table, one per sheet, with a README sheet |
| `tables/` | Excel, R, Python | the same tables as TSV/CSV/JSON/Parquet (the full cis table is here only) |
| `figures/` | any viewer; Illustrator/Inkscape for PDF/SVG | PNG and TIFF at 600 dpi, PDF/SVG vector files with editable text |
| `summary.json`, `config_used.json`, `multiqtl.log` | text editor | key numbers and the exact settings used |

**Reading the main tables**

| Column | Meaning |
|---|---|
| `qval` | gene/protein-level false discovery rate (Storey q-value). `qval ≤ 0.05` = significant eGene / pGene |
| `fdr_bh`, `fdr_by`, `fdr_bonferroni`, `qval_eigenmt_bh` … | the same test under stricter or alternative corrections |
| `pval_nominal_threshold` | per-gene P cut-off used to list all significant variants for that gene |
| `slope` | effect per copy of the alternative allele, in standard-deviation units of the normalized abundance |
| `PP.H4` (coloc) | probability that transcript and protein share the same causal variant (> 0.8 = strong) |
| `PP.H3` (coloc) | probability of two different causal variants |
| `category` | shared / eQTL-only (protein buffered) / pQTL-only (protein-specific mechanism) |
| `mechanism` (mediation) | whether the pQTL acts through mRNA (`transcript_mediated`) or not (`protein_specific`) |

---

## 6. Troubleshooting

| Message | Fix |
|---|---|
| `only 0 sample IDs shared across files` | sample names differ between files (spaces, prefixes, `S01` vs `S1`). The check lists examples from each file. |
| `file not found` | the path in the settings file is wrong; relative paths start from the settings file's folder |
| `values are not numeric dosages` | the genotype table holds letters (A/G). Convert to 0/1/2 or supply a VCF |
| `type is rnaseq but values are not integer counts` | choose “Already normalized” for TPM/FPKM/log data |
| `protein annotation needs a gene_id column` | add the gene each protein comes from, using the same IDs as the gene annotation |
| `Python not found` (Windows) | re-install Python with “Add python.exe to PATH” ticked |
| Browser does not open | open `http://127.0.0.1:8765` manually; if the port is busy: `python -m multiqtl gui --port 9000` |
| Analysis is slow | reduce permutations to 500 for a first look; split genotypes by chromosome for very large panels |

---

## 7. Citing the methods

The **Methods** section of `report.html` names each procedure; key references are listed in `README.md`.
Report the MultiQTL version (`python -m multiqtl --version`) and attach `config_used.json` as supplementary
material for full reproducibility.
