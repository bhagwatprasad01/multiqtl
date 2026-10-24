"""Unit and calibration tests (run: python -m unittest discover -s tests -v)."""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from multiqtl import fdr as F
from multiqtl import integration as I
from multiqtl import preprocess as PP
from multiqtl import stats as ST
from multiqtl.mapping import QTLMapper


def toy(n=200, m=300, seed=0, effect=0.6, n_feat=40):
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, rng.uniform(0.1, 0.5, m)[:, None], (m, n)).astype(float)
    V = pd.DataFrame({"chrom": "1", "pos": np.sort(rng.choice(np.arange(1, 50_000_000), m, replace=False))},
                     index=[f"v{i}" for i in range(m)])
    C = pd.DataFrame({"c1": rng.normal(size=n), "c2": rng.normal(size=n)}, index=[f"s{j}" for j in range(n)])
    P = rng.normal(size=(n_feat, n)) + 0.5 * C.c1.values
    causal = {}
    for f in range(n_feat // 2):                      # first half have a cis effect
        start = int(rng.integers(5_000_000, 45_000_000))
        cand = np.where(np.abs(V.pos.values - start) < 200_000)[0]
        if len(cand):
            v = rng.choice(cand)
            g = (G[v] - G[v].mean()) / G[v].std()
            P[f] += effect * g
            causal[f"f{f}"] = (V.index[v], start)
    starts = [causal.get(f"f{f}", (None, int(rng.integers(5_000_000, 45_000_000))))[1] for f in range(n_feat)]
    A = pd.DataFrame({"chrom": "1", "start": starts}, index=[f"f{f}" for f in range(n_feat)])
    cols = C.index
    return (pd.DataFrame(G, index=V.index, columns=cols), V, pd.DataFrame(P, index=A.index, columns=cols), A, C, causal)


class TestRegression(unittest.TestCase):
    def test_matches_ols(self):
        G, V, P, A, C, _ = toy()
        M = QTLMapper(G, V, P, A, C)
        nom = M.map_cis_nominal(window=1_000_000)
        r = nom.iloc[7]
        y = P.loc[r.feature_id].values
        X = np.column_stack([np.ones(len(y)), G.loc[r.variant_id].values, C.values])
        beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        dof = len(y) - X.shape[1]
        s2 = res[0] / dof
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        p = 2 * stats.t.sf(abs(beta[1] / se), dof)
        self.assertAlmostEqual(r.slope, beta[1], places=10)
        self.assertAlmostEqual(r.slope_se, se, places=10)
        self.assertAlmostEqual(r.pval / p, 1.0, places=8)

    def test_interaction_matches_ols(self):
        G, V, P, A, C, _ = toy(seed=3)
        rng = np.random.default_rng(1)
        age = rng.uniform(0, 18, G.shape[1])
        M = QTLMapper(G, V, P, A, C)
        res, top = M.map_cis_interaction(age, 1_000_000, name="age")
        r = res.iloc[3]
        y = P.loc[r.feature_id].values
        g = G.loc[r.variant_id].values; g = g - g.mean(); a = age - age.mean()
        X = np.column_stack([np.ones(len(y)), C.values, a, g, g * a])
        b, rs, *_ = np.linalg.lstsq(X, y, rcond=None)
        dof = len(y) - X.shape[1]
        se = np.sqrt(rs[0] / dof * np.linalg.inv(X.T @ X)[-1, -1])
        self.assertAlmostEqual(r.b_gxage, b[-1], places=8)
        self.assertAlmostEqual(r.b_gxage_se, se, places=8)


class TestPermutation(unittest.TestCase):
    def test_detects_and_calibrates(self):
        G, V, P, A, C, causal = toy(n=250, seed=5)
        M = QTLMapper(G, V, P, A, C)
        perm = M.map_cis_permutation(1_000_000, n_perm=300)
        sig = set(perm.loc[perm.qval <= 0.05, "feature_id"])
        self.assertGreaterEqual(len(sig & set(causal)), 0.8 * len(causal))
        null = perm[~perm.feature_id.isin(causal)]
        # beta-approximated p tracks the empirical permutation p
        k = perm.pval_perm > 5 / 301                       # outside the permutation floor
        self.assertGreater(stats.spearmanr(perm.pval_perm[k], perm.pval_beta[k])[0], 0.9)
        self.assertLess(np.abs(perm.pval_perm[k] - perm.pval_beta[k]).mean(), 0.05)
        self.assertLess((null.pval_beta < 0.05).mean(), 0.2)


class TestFDR(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.null = rng.uniform(size=8000)
        self.alt = stats.norm.sf(rng.normal(3.0, 1, 2000))
        self.p = np.r_[self.null, self.alt]
        self.is_null = np.r_[np.ones(8000, bool), np.zeros(2000, bool)]

    def test_bh_matches_reference(self):
        p = np.array([0.01, 0.04, 0.03, 0.005, 0.2])
        # reference values as returned by R p.adjust(p, "BH") / p.adjust(p, "holm")
        np.testing.assert_allclose(F.bh(p), [0.025, 0.05, 0.05, 0.025, 0.2])
        np.testing.assert_allclose(F.holm(p), [0.04, 0.09, 0.09, 0.025, 0.2])
        np.testing.assert_allclose(F.by(p), np.minimum(F.bh(p) * sum(1 / np.arange(1, 6)), 1))

    def test_fdr_control(self):
        for fn in (F.bh, F.qvalue, F.by):
            q = fn(self.p)
            d = q <= 0.05
            fdp = (d & self.is_null).sum() / max(d.sum(), 1)
            self.assertLess(fdp, 0.08, fn.__name__)

    def test_pi0(self):
        self.assertAlmostEqual(F.estimate_pi0(self.p, "smoother"), 0.8, delta=0.06)
        self.assertAlmostEqual(F.estimate_pi0(self.p, "bootstrap"), 0.8, delta=0.06)
        self.assertAlmostEqual(F.estimate_pi0(self.null), 1.0, delta=0.08)

    def test_lfdr_range(self):
        l = F.local_fdr(self.p)
        self.assertTrue(np.all((l >= 0) & (l <= 1)))
        self.assertLess(np.median(l[~self.is_null]), np.median(l[self.is_null]))

    def test_permutation_fdr(self):
        rng = np.random.default_rng(2)
        obs = np.r_[rng.uniform(size=900), rng.uniform(0, 1e-4, 100)]
        null = rng.uniform(size=10_000)
        f = F.permutation_fdr(obs, null)
        self.assertTrue(np.all(f[900:] < 0.05))

    def test_hierarchical(self):
        df = pd.DataFrame({"feature_id": np.repeat(np.arange(100), 20), "pval": np.random.default_rng(3).uniform(size=2000)})
        df.loc[df.feature_id < 10, "pval"] /= 1e6
        out = F.hierarchical_fdr(df)
        self.assertEqual(set(out.loc[out.hier_significant, "feature_id"]) - set(range(10)), set())

    def test_eigenmt(self):
        rng = np.random.default_rng(4)
        base = rng.normal(size=(5, 300))
        G = np.repeat(base, 10, axis=0) + 0.01 * rng.normal(size=(50, 300))
        self.assertLessEqual(F.eigenmt_meff(G), 6)


class TestColoc(unittest.TestCase):
    def test_shared_and_distinct(self):
        n = 50
        se = np.full(n, 0.05)
        b1 = np.zeros(n); b2 = np.zeros(n)
        b1[10] = 0.4; b2[10] = 0.35
        r = I.coloc_abf(b1, se, b2, se)
        self.assertGreater(r["PP.H4"], 0.9)
        b2 = np.zeros(n); b2[40] = 0.35
        r = I.coloc_abf(b1, se, b2, se)
        self.assertGreater(r["PP.H3"], 0.9)
        r = I.coloc_abf(b1, se, np.zeros(n), se)
        self.assertGreater(r["PP.H1"], 0.9)
        self.assertAlmostEqual(sum(r[f"PP.H{i}"] for i in range(5)), 1.0, places=10)


class TestMR(unittest.TestCase):
    def test_ivw_egger_wm(self):
        rng = np.random.default_rng(0)
        bx = rng.uniform(0.1, 0.5, 20); sx = np.full(20, 0.02)
        by = 0.6 * bx + rng.normal(0, 0.01, 20); sy = np.full(20, 0.01)
        res = I.mendelian_randomization(bx, sx, by, sy, n_boot=200).set_index("method")
        for m in ("ivw", "egger", "weighted_median"):
            self.assertAlmostEqual(res.loc[m, "beta"], 0.6, delta=0.06)
        self.assertGreater(res.loc["egger", "intercept_p"], 0.01)
        b, se, p = I.mr_wald_ratio(0.5, 0.05, 0.3, 0.05)
        self.assertAlmostEqual(b, 0.6)

    def test_mediation(self):
        rng = np.random.default_rng(1)
        n = 400
        g = rng.binomial(2, 0.3, n).astype(float)
        m = 0.5 * g + rng.normal(size=n)
        y = 0.6 * m + rng.normal(size=n)
        r = I.mediation(g, m, y, n_boot=300)
        self.assertAlmostEqual(r["indirect"], 0.3, delta=0.1)
        self.assertGreater(r["direct_p"], 0.01)
        self.assertGreater(r["indirect_ci_low"], 0)          # bootstrap CI excludes zero


class TestPreprocess(unittest.TestCase):
    def test_hwe(self):
        self.assertGreater(PP.hwe_exact(50, 25, 25), 0.5)
        self.assertLess(PP.hwe_exact(0, 50, 50), 1e-10)

    def test_int(self):
        df = pd.DataFrame(np.random.default_rng(0).exponential(size=(5, 500)))
        out = PP.inverse_normal_transform(df)
        self.assertLess(abs(out.values.mean()), 0.01)
        self.assertGreater(stats.shapiro(out.iloc[0])[1], 0.05)

    def test_tmm_scale_invariance(self):
        rng = np.random.default_rng(0)
        base = rng.gamma(2, 50, 1000)
        C = np.column_stack([rng.poisson(base * s) for s in (1, 2, 1, 3)])
        f = PP.tmm_factors(C)
        self.assertTrue(np.all(np.abs(np.log2(f)) < 0.1))

    def test_proteomics(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(2 ** rng.normal(20, 2, (100, 60)))
        X[X < 2 ** 17] = np.nan
        out, qc = PP.normalize_proteomics(X)
        self.assertFalse(out.isna().values.any())
        self.assertLessEqual(qc["n_features_out"], 100)


class TestStats(unittest.TestCase):
    def test_lambda_and_power(self):
        p = np.random.default_rng(0).uniform(size=50_000)
        self.assertAlmostEqual(ST.genomic_inflation(p)["lambda"], 1.0, delta=0.03)
        self.assertGreater(ST.qtl_power(300, 0.3, 0.5, 1e-5), 0.9)
        self.assertLess(ST.qtl_power(100, 0.05, 0.1, 1e-5), 0.05)


class TestEndToEnd(unittest.TestCase):
    def test_small_pipeline(self):
        from multiqtl.simulate import simulate_study
        from multiqtl.io import Molecular, QTLDataset
        from multiqtl.pipeline import run_pipeline
        d = simulate_study(None, n_samples=150, n_var_per_chrom=400, n_genes=60, n_proteins=40, seed=7)
        ds = QTLDataset(d["genotypes"], d["variants"],
                        {"expression": Molecular("expression", d["rna_counts"], d["gene_annot"], d["rna_counts"]),
                         "protein": Molecular("protein", d["protein_intensity"], d["protein_annot"], d["protein_intensity"])},
                        d["covariates"]).align()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"output_dir": tmp, "cis": {"n_perm": 100}, "covariates": {"hidden_factors": 2},
                   "layers": {"expression": {"type": "rnaseq"}, "protein": {"type": "proteomics"}},
                   "interaction": {"enabled": True, "covariate": "age", "layers": ["protein"]},
                   "integration": {"transcript_layer": "expression", "protein_layer": "protein", "mediation_boot": 100},
                   "output": {"figure_formats": ["png", "svg"], "dpi": 72, "table_formats": ["tsv", "xlsx", "json"]}}
            R = run_pipeline(cfg, ds)
            self.assertTrue(os.path.exists(os.path.join(tmp, "report.html")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "multiqtl_results.xlsx")))
            self.assertGreater(R["summary"]["expression_cis"]["n_significant_features"], 5)


if __name__ == "__main__":
    unittest.main()


class TestStandaloneUse(unittest.TestCase):
    """Features for users running MultiQTL on their own: tolerant file reading, config handling, GUI form."""

    def test_read_table_variants(self):
        from multiqtl.io import read_table
        df = pd.DataFrame({"feature_id": ["g1", "g2"], "1001": [1, 2], "1002": [3, 4]})
        with tempfile.TemporaryDirectory() as t:
            p1, p2 = os.path.join(t, "a.csv"), os.path.join(t, "b.txt")
            with open(p1, "w", encoding="utf-8-sig") as fh:      # Excel-style BOM
                fh.write(df.to_csv(index=False))
            df.to_csv(p2, sep=";", index=False)                    # semicolon-delimited .txt
            for p in (p1, p2):
                x = read_table(p)
                self.assertEqual(list(x.index), ["g1", "g2"])
                self.assertEqual(list(x.columns), ["1001", "1002"])

    def test_config_paths_relative_to_file(self):
        from multiqtl.config_tools import read_config_file
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "c.json")
            with open(p, "w") as fh:
                json.dump({"genotypes": "data/g.vcf", "layers": {"x": {"data": "d.tsv", "annot": "a.tsv"}}}, fh)
            cfg = read_config_file(p)
            self.assertEqual(cfg["genotypes"], os.path.join(t, "data", "g.vcf"))
            self.assertEqual(cfg["layers"]["x"]["annot"], os.path.join(t, "a.tsv"))

    def test_check_inputs_reports_problems(self):
        from multiqtl.config_tools import check_inputs
        from multiqtl.simulate import simulate_study
        with tempfile.TemporaryDirectory() as t:
            simulate_study(t, n_samples=60, n_var_per_chrom=100, n_genes=20, n_proteins=10, seed=3)
            P = pd.read_csv(f"{t}/protein_intensity.tsv.gz", sep="\t", index_col=0)
            P.columns = ["X" + c for c in P.columns]
            P.to_csv(f"{t}/bad.tsv", sep="\t")
            cfg = {"genotypes": f"{t}/genotypes.tsv.gz", "variants": f"{t}/variants.tsv",
                   "layers": {"protein": {"type": "proteomics", "data": f"{t}/bad.tsv", "annot": f"{t}/protein_annotation.tsv"}},
                   "covariates": {"known": f"{t}/covariates.tsv"}, "interaction": {"enabled": True, "covariate": "weight"}}
            msgs = " ".join(m for l, m in check_inputs(cfg) if l == "ERROR")
            self.assertIn("sample IDs shared", msgs)
            self.assertIn("'weight'", msgs)
            cfg["layers"]["protein"]["data"] = f"{t}/protein_intensity.tsv.gz"
            cfg["interaction"]["covariate"] = "age"
            self.assertFalse([m for l, m in check_inputs(cfg) if l == "ERROR"])

    def test_gui_form_to_config(self):
        from multiqtl import gui
        with tempfile.TemporaryDirectory() as t:
            f = {"genotypes": ("g.vcf", b"x"), "prot_data": ("p.tsv", b"x"), "prot_annot": ("a.tsv", b"x"),
                 "prot_type": "proteomics", "window_kb": "500", "n_perm": "200", "fdr": "0.1", "interaction": "age",
                 "hidden": "5", "fig_pdf": "on", "tab_xlsx": "on", "dpi": "1200"}
            os.makedirs(os.path.join(t, "inputs"))
            cfg = gui.build_config(f, t)
            self.assertEqual(cfg["cis"]["window"], 500_000)
            self.assertEqual(cfg["covariates"]["hidden_factors"], 5)
            self.assertEqual(cfg["output"]["figure_formats"], ["pdf"])
            self.assertTrue(cfg["interaction"]["enabled"])
            self.assertNotIn("integration", cfg)
            self.assertTrue(os.path.exists(cfg["genotypes"]))
