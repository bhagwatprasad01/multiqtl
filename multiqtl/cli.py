"""Command-line interface.

    multiqtl gui                      point-and-click interface in your web browser
    multiqtl demo [folder]            simulate a study and run the full analysis on it
    multiqtl init [config.yaml]       write a commented config template to fill in
    multiqtl check config.yaml        validate input files without running
    multiqtl run config.yaml          run the full pipeline
    multiqtl simulate folder          write simulated input files with ground truth
    multiqtl fdr | coloc | power      stand-alone utilities

``python -m multiqtl <command>`` works identically when the ``multiqtl`` launcher is not on PATH.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="multiqtl", description="Integrated eQTL / pQTL mapping and multi-omic integration",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--version", action="store_true", help="print version and exit")
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("gui", help="open the point-and-click interface in your browser")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--workspace", help="folder for runs (default ~/MultiQTL_runs)")
    g.add_argument("--no-browser", action="store_true")

    d = sub.add_parser("demo", help="simulate a study and analyse it end to end")
    d.add_argument("folder", nargs="?", default="multiqtl_demo")
    d.add_argument("--n-perm", type=int, default=1000)

    i = sub.add_parser("init", help="write a commented config template")
    i.add_argument("path", nargs="?", default="multiqtl_config.yaml")

    c0 = sub.add_parser("check", help="validate a config and its input files")
    c0.add_argument("config")

    r = sub.add_parser("run", help="run the full pipeline from a YAML/JSON config")
    r.add_argument("config")
    r.add_argument("-o", "--output-dir")
    r.add_argument("--n-perm", type=int)
    r.add_argument("--dpi", type=int)
    r.add_argument("--formats", help="figure formats, comma-separated (png,pdf,svg,tiff,eps)")

    s = sub.add_parser("simulate", help="write a simulated multi-omic study with ground truth")
    s.add_argument("outdir")
    s.add_argument("--samples", type=int, default=300)
    s.add_argument("--genes", type=int, default=300)
    s.add_argument("--proteins", type=int, default=200)
    s.add_argument("--seed", type=int, default=42)

    f = sub.add_parser("fdr", help="add multiple-testing corrections to any results table")
    f.add_argument("table"); f.add_argument("output")
    f.add_argument("--p-col", default="pval")
    f.add_argument("--methods", default="bh,by,bonferroni,holm,qvalue,lfdr")
    f.add_argument("--group-col", help="also run hierarchical (TreeQTL-style) FDR grouped by this column")

    c = sub.add_parser("coloc", help="coloc.abf on two summary-statistic tables (variant_id, beta, se)")
    c.add_argument("trait1"); c.add_argument("trait2")
    c.add_argument("--p1", type=float, default=1e-4); c.add_argument("--p2", type=float, default=1e-4)
    c.add_argument("--p12", type=float, default=1e-5)
    c.add_argument("--beta-col", default="beta"); c.add_argument("--se-col", default="se")

    p = sub.add_parser("power", help="analytical QTL power grid")
    p.add_argument("--n", default="100,200,300,500,1000"); p.add_argument("--alpha", type=float, default=1e-5)
    p.add_argument("-o", "--output", default="power_grid.tsv")

    a = ap.parse_args(argv)
    if a.version:
        from . import __version__
        print(f"multiqtl {__version__}")
        return 0
    if a.cmd is None:
        ap.print_help()
        return 0

    try:
        return _dispatch(a)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        from .pipeline import InputError
        if isinstance(e, (InputError, FileNotFoundError, ValueError)):
            print(f"\n{e}\n", file=sys.stderr)
            print("Tip: `multiqtl check <config>` lists every input problem in plain language.", file=sys.stderr)
            return 1
        raise


def _dispatch(a):
    if a.cmd == "gui":
        from .gui import serve
        serve(a.port, a.workspace, not a.no_browser)
    elif a.cmd == "init":
        from .config_tools import write_template
        if os.path.exists(a.path):
            print(f"{a.path} already exists – not overwritten.")
            return 1
        write_template(a.path)
        print(f"Wrote {a.path}. Edit the file paths, then run:\n  multiqtl check {a.path}\n  multiqtl run {a.path}")
    elif a.cmd == "check":
        from .config_tools import format_report, check_inputs
        from .pipeline import load_config
        items = check_inputs(load_config(a.config))
        print(format_report(items))
        return 1 if any(l == "ERROR" for l, _ in items) else 0
    elif a.cmd == "demo":
        from .pipeline import run_pipeline
        from .simulate import simulate_study
        data = os.path.join(a.folder, "data")
        print(f"Simulating a study in {data} ...")
        simulate_study(data)
        cfg = {"output_dir": os.path.join(a.folder, "results"),
               "genotypes": os.path.join(data, "genotypes.tsv.gz"), "variants": os.path.join(data, "variants.tsv"),
               "layers": {"expression": {"type": "rnaseq", "data": os.path.join(data, "rna_counts.tsv.gz"),
                                         "annot": os.path.join(data, "gene_annotation.tsv")},
                          "protein": {"type": "proteomics", "data": os.path.join(data, "protein_intensity.tsv.gz"),
                                      "annot": os.path.join(data, "protein_annotation.tsv")}},
               "covariates": {"known": os.path.join(data, "covariates.tsv")},
               "cis": {"n_perm": a.n_perm}, "trans": {"n_perm": 2},
               "interaction": {"enabled": True, "covariate": "age", "layers": ["protein"]},
               "integration": {"transcript_layer": "expression", "protein_layer": "protein"}}
        run_pipeline(cfg)
        print(f"\nOpen {os.path.join(a.folder, 'results', 'report.html')} in a web browser.")
    elif a.cmd == "run":
        from .pipeline import load_config, run_pipeline
        cfg = load_config(a.config)
        if a.output_dir: cfg["output_dir"] = os.path.abspath(a.output_dir)
        if a.n_perm: cfg["cis"]["n_perm"] = a.n_perm
        if a.dpi: cfg["output"]["dpi"] = a.dpi
        if a.formats: cfg["output"]["figure_formats"] = a.formats.split(",")
        run_pipeline(cfg)
    elif a.cmd == "simulate":
        from .simulate import simulate_study
        simulate_study(a.outdir, n_samples=a.samples, n_genes=a.genes, n_proteins=a.proteins, seed=a.seed)
        print(f"Simulated study written to {a.outdir}")
    elif a.cmd == "fdr":
        from . import fdr as F
        from .io import read_table
        df = read_table(a.table, index_col=None)
        df = F.add_all_corrections(df, a.p_col, a.methods.split(","))
        if a.group_col:
            df = F.hierarchical_fdr(df, a.group_col, a.p_col)
        sep = "," if a.output.endswith(".csv") else "\t"
        df.to_csv(a.output, sep=sep, index=False)
        print(F.fdr_summary(df[a.p_col].values).to_string(index=False))
    elif a.cmd == "coloc":
        from .integration import coloc_abf
        from .io import read_table
        t1, t2 = read_table(a.trait1), read_table(a.trait2)
        m = t1.join(t2, lsuffix="_1", rsuffix="_2", how="inner")
        res = coloc_abf(m[f"{a.beta_col}_1"], m[f"{a.se_col}_1"], m[f"{a.beta_col}_2"], m[f"{a.se_col}_2"], a.p1, a.p2, a.p12)
        for h in ["PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4"]:
            print(f"{h}\t{res[h]:.4f}")
        print("credible_set\t" + ",".join(m.index[res["credible_set_idx"]]))
    elif a.cmd == "power":
        from .stats import power_grid
        g = power_grid(n_values=[int(x) for x in a.n.split(",")], alpha=a.alpha)
        g.to_csv(a.output, sep="\t", index=False)
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
