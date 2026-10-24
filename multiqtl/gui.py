"""Point-and-click interface: a small local web app (standard library only) that runs in your browser.

    multiqtl gui                 # opens http://127.0.0.1:8765
    multiqtl gui --port 9000 --workspace D:/qtl_runs --no-browser

Nothing leaves your computer: the server listens on 127.0.0.1 only and uploaded files are copied into a run
folder under the workspace (default: ~/MultiQTL_runs). Each run folder also gets a config.yaml, so the same
analysis can be repeated later from the command line with `multiqtl run <run folder>/config.yaml`.
"""
from __future__ import annotations

import datetime as _dt
import email
import email.policy
import html
import json
import os
import shutil
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse

from . import __version__

JOBS = {}           # job_id -> {"dir", "status", "error"}
WORKSPACE = os.path.join(os.path.expanduser("~"), "MultiQTL_runs")

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--surface:#fcfcfb;--page:#f4f4f1;--accent:#256abf;--accent-ink:#fff;
--err:#b42323;--warn:#8a5a00;--ok:#0b7a0b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--surface:#1a1a19;
--page:#0d0d0d;--accent:#6da7ec;--accent-ink:#0b0b0b;--err:#ff8a8a;--warn:#f0c060;--ok:#7fd67f}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 -apple-system,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
main{max-width:920px;margin:0 auto;padding:28px 16px 64px}h1{font-size:26px;margin:0}h2{font-size:17px;margin:0 0 4px}
.sub{color:var(--ink2);margin:4px 0 22px}.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 18px 8px;margin:0 0 16px}
.hint{color:var(--ink2);font-size:13px;margin:0 0 12px}label{display:block;font-weight:600;font-size:13.5px;margin:10px 0 4px}
label small{font-weight:400;color:var(--ink2)}input[type=file],input[type=text],input[type=number],select{width:100%;padding:8px;border:1px solid var(--line);
border-radius:8px;background:var(--page);color:var(--ink);font:inherit}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0 16px}
.checks label{display:inline-flex;gap:6px;align-items:center;font-weight:400;margin:4px 16px 8px 0}
.btns{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}button,.btn{background:var(--accent);color:var(--accent-ink);border:0;border-radius:8px;
padding:11px 18px;font:600 15px/1 inherit;cursor:pointer;text-decoration:none;display:inline-block}button.secondary,.btn.secondary{background:transparent;
color:var(--accent);border:1px solid var(--accent)}pre{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px;
overflow:auto;font-size:12.5px;max-height:420px;white-space:pre-wrap}.ERROR{color:var(--err)}.WARNING{color:var(--warn)}.OK{color:var(--ok)}
ul.checks-out{list-style:none;padding:0}ul.checks-out li{padding:3px 0;border-bottom:1px solid var(--line)}code{font-size:13px}
.status{font-weight:600}a{color:var(--accent)}
"""


def page(title, body, refresh=None):
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"{meta}<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>").encode("utf-8")


def file_input(name, label, hint="", required=False):
    req = " <small>(required)</small>" if required else " <small>(optional)</small>"
    return (f"<label for='{name}'>{label}{req}</label><input type='file' id='{name}' name='{name}'>"
            + (f"<p class='hint'>{hint}</p>" if hint else ""))


FORM = f"""
<h1>MultiQTL</h1><p class='sub'>eQTL · pQTL · multi-omic integration — v{__version__} · runs locally, nothing is uploaded to the internet</p>
<form method='post' action='/submit' enctype='multipart/form-data'>
<div class='card'><h2>1 · Genotypes</h2><p class='hint'>A VCF (.vcf / .vcf.gz), or a table with one row per variant, one column per sample, values 0/1/2.</p>
{file_input('genotypes', 'Genotype file', required=True)}
{file_input('variants', 'Variant positions', 'Only needed for a genotype table: columns variant_id, chrom, pos.')}
</div>
<div class='card'><h2>2 · Transcript data</h2><p class='hint'>Rows = genes, columns = samples. Leave empty if you only have protein data.</p>
{file_input('rna_data', 'Expression matrix')}
<div class='row'><div><label>Data type</label><select name='rna_type'><option value='rnaseq'>Raw RNA-seq counts</option>
<option value='normalized'>Already normalized (TPM, log-CPM, microarray…)</option></select></div></div>
{file_input('rna_annot', 'Gene annotation', 'Columns: feature_id, chrom, start (TSS), end, strand, gene_id, symbol.')}
</div>
<div class='card'><h2>3 · Protein data</h2><p class='hint'>Rows = proteins, columns = samples. Leave empty if you only have transcript data.</p>
{file_input('prot_data', 'Protein abundance matrix')}
<div class='row'><div><label>Data type</label><select name='prot_type'><option value='proteomics'>Raw LC-MS intensities (0/blank = not detected)</option>
<option value='normalized'>Already normalized</option></select></div>
<div><label>Missing-value imputation</label><select name='impute'><option value='downshift'>Down-shifted normal (Perseus)</option>
<option value='min'>Minimum</option><option value='halfmin'>Half-minimum</option><option value='knn'>k-nearest neighbours</option></select></div></div>
{file_input('prot_annot', 'Protein annotation', 'Columns: feature_id, chrom, start, end, strand, gene_id (links to the gene), symbol.')}
</div>
<div class='card'><h2>4 · Covariates</h2>
{file_input('covariates', 'Covariate table', 'Rows = samples; columns such as age, sex, batch. Text columns are encoded automatically.')}
<div class='row'><div><label>Genotype PCs</label><input type='number' name='genotype_pcs' value='3' min='0' max='20'></div>
<div><label>Hidden factors</label><select name='hidden'><option value='auto'>Automatic (permutation rule)</option><option value='elbow'>Scree elbow</option>
<option value='0'>None</option><option value='5'>5</option><option value='10'>10</option><option value='15'>15</option></select></div></div>
</div>
<div class='card'><h2>5 · Analysis settings</h2>
<div class='row'>
<div><label>cis window (kb, each side)</label><input type='number' name='window_kb' value='1000' min='10'></div>
<div><label>Permutations</label><input type='number' name='n_perm' value='1000' min='100' step='100'></div>
<div><label>FDR threshold</label><input type='number' name='fdr' value='0.05' step='0.01' min='0.001' max='0.5'></div>
<div><label>Minor allele frequency ≥</label><input type='number' name='maf' value='0.05' step='0.01' min='0' max='0.5'></div>
</div>
<div class='checks'><label><input type='checkbox' name='trans' checked> trans-QTL scan</label>
<label><input type='checkbox' name='independent' checked> independent signals</label></div>
<div class='row'><div><label>Interaction covariate <small>(e.g. age; blank = off)</small></label><input type='text' name='interaction' placeholder='age'></div></div>
</div>
<div class='card'><h2>6 · Output</h2>
<div class='checks'><b style='font-size:13.5px'>Figures:</b>
<label><input type='checkbox' name='fig_png' checked> PNG</label><label><input type='checkbox' name='fig_tiff' checked> TIFF</label>
<label><input type='checkbox' name='fig_pdf' checked> PDF</label><label><input type='checkbox' name='fig_svg' checked> SVG</label>
<label><input type='checkbox' name='fig_eps'> EPS</label></div>
<div class='checks'><b style='font-size:13.5px'>Tables:</b>
<label><input type='checkbox' name='tab_tsv' checked> TSV</label><label><input type='checkbox' name='tab_csv'> CSV</label>
<label><input type='checkbox' name='tab_xlsx' checked> Excel</label><label><input type='checkbox' name='tab_json'> JSON</label>
<label><input type='checkbox' name='tab_parquet'> Parquet</label></div>
<div class='row'><div><label>Raster resolution (dpi)</label><select name='dpi'><option>300</option><option selected>600</option><option>1200</option></select></div>
<div><label>Run name <small>(optional)</small></label><input type='text' name='run_name' placeholder='my_study'></div></div>
</div>
<div class='btns'><button type='submit' name='action' value='check' class='secondary'>Check inputs only</button>
<button type='submit' name='action' value='run'>Run analysis</button></div>
</form>
<div class='card'><h2>No data yet?</h2><p class='hint'>Run the full workflow on a simulated study (300 samples, transcripts + proteins, known true QTLs).</p>
<form method='post' action='/demo'><div class='btns' style='margin-top:0'><button class='secondary'>Run demo</button></div></form></div>
<p class='hint'>Runs are saved in <code>{{workspace}}</code>.</p>
"""


# ================================================================================================================
def _parse_multipart(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    body = handler.rfile.read(length)
    ctype = handler.headers.get("Content-Type", "")
    msg = email.message_from_bytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body, policy=email.policy.HTTP)
    fields = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fname = part.get_filename()
        data = part.get_payload(decode=True) or b""
        if fname:
            if data:
                fields[name] = (os.path.basename(fname), data)
        else:
            fields[name] = data.decode("utf-8", "replace").strip()
    return fields


def _new_job_dir(name: str = "") -> tuple:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40]
    jid = f"{stamp}_{safe}" if safe else stamp
    d = os.path.join(WORKSPACE, jid)
    os.makedirs(os.path.join(d, "inputs"), exist_ok=True)
    return jid, d


def build_config(f: dict, job_dir: str) -> dict:
    """Translate form fields + uploaded files into a pipeline config (paths inside the run folder)."""
    def save(key):
        if key not in f or not isinstance(f[key], tuple):
            return None
        fname, data = f[key]
        path = os.path.join(job_dir, "inputs", f"{key}__{fname}")
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    cfg = {"output_dir": os.path.join(job_dir, "results"), "genotypes": save("genotypes"), "variants": save("variants"),
           "layers": {}, "covariates": {"known": save("covariates")}}
    rd, ra = save("rna_data"), save("rna_annot")
    pdat, pa = save("prot_data"), save("prot_annot")
    if rd:
        cfg["layers"]["expression"] = {"type": f.get("rna_type", "rnaseq"), "data": rd, "annot": ra}
    if pdat:
        cfg["layers"]["protein"] = {"type": f.get("prot_type", "proteomics"), "data": pdat, "annot": pa,
                                    "impute": f.get("impute", "downshift")}
    hidden = f.get("hidden", "auto")
    cfg["covariates"].update({"genotype_pcs": int(f.get("genotype_pcs") or 3),
                              "hidden_factors": int(hidden) if hidden.isdigit() else hidden})
    fdr = float(f.get("fdr") or 0.05)
    cfg["genotype_qc"] = {"maf_min": float(f.get("maf") or 0.05)}
    cfg["cis"] = {"window": int(float(f.get("window_kb") or 1000) * 1000), "n_perm": int(f.get("n_perm") or 1000),
                  "fdr": fdr, "independent": "independent" in f}
    cfg["trans"] = {"enabled": "trans" in f, "fdr": fdr}
    ic = (f.get("interaction") or "").strip()
    cfg["interaction"] = {"enabled": bool(ic), "covariate": ic or None, "layers": None}
    if rd and pdat:
        cfg["integration"] = {"transcript_layer": "expression", "protein_layer": "protein"}
    figs = [x for x in ("png", "tiff", "pdf", "svg", "eps") if f"fig_{x}" in f] or ["png"]
    tabs = [x for x in ("tsv", "csv", "xlsx", "json", "parquet") if f"tab_{x}" in f] or ["tsv"]
    cfg["output"] = {"figure_formats": figs, "table_formats": tabs, "dpi": int(f.get("dpi") or 600)}
    return cfg


def _write_config(cfg: dict, job_dir: str):
    """Save a portable config next to the inputs (relative paths) for command-line re-runs."""
    def rel(p):
        return os.path.relpath(p, job_dir) if isinstance(p, str) and os.path.isabs(p) else p
    c = json.loads(json.dumps(cfg))
    for k in ("genotypes", "variants", "output_dir"):
        c[k] = rel(c.get(k))
    for spec in c["layers"].values():
        spec["data"], spec["annot"] = rel(spec["data"]), rel(spec["annot"])
    c["covariates"]["known"] = rel(c["covariates"]["known"])
    try:
        import yaml
        with open(os.path.join(job_dir, "config.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump(c, fh, sort_keys=False)
    except ImportError:
        with open(os.path.join(job_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=2)


def _run_job(jid: str, cfg: dict):
    from .pipeline import run_pipeline
    job = JOBS[jid]
    job["status"] = "running"
    try:
        run_pipeline(cfg)
        res = cfg["output_dir"]
        shutil.make_archive(os.path.join(job["dir"], "results"), "zip", res)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e) if e.__class__.__name__ == "InputError" else traceback.format_exc()


def _run_demo(jid: str):
    from .simulate import simulate_study
    job = JOBS[jid]
    job["status"] = "simulating data"
    d = job["dir"]
    simulate_study(os.path.join(d, "inputs"))
    i = os.path.join(d, "inputs")
    cfg = {"output_dir": os.path.join(d, "results"), "genotypes": f"{i}/genotypes.tsv.gz", "variants": f"{i}/variants.tsv",
           "layers": {"expression": {"type": "rnaseq", "data": f"{i}/rna_counts.tsv.gz", "annot": f"{i}/gene_annotation.tsv"},
                      "protein": {"type": "proteomics", "data": f"{i}/protein_intensity.tsv.gz", "annot": f"{i}/protein_annotation.tsv"}},
           "covariates": {"known": f"{i}/covariates.tsv"},
           "trans": {"n_perm": 2}, "interaction": {"enabled": True, "covariate": "age", "layers": ["protein"]},
           "integration": {"transcript_layer": "expression", "protein_layer": "protein"},
           "output": {"figure_formats": ["png", "pdf", "svg"], "table_formats": ["tsv", "xlsx"], "dpi": 300}}
    _write_config(cfg, d)
    _run_job(jid, cfg)


# ================================================================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = f"MultiQTL/{__version__}"

    def log_message(self, fmt, *args):   # quiet console
        pass

    def _send(self, data: bytes, ctype="text/html; charset=utf-8", code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, url):
        self.send_response(303)
        self.send_header("Location", url)
        self.end_headers()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(page("MultiQTL", FORM.replace("{workspace}", html.escape(WORKSPACE))))
        if path.startswith("/job/"):
            return self._job_page(path.split("/")[2])
        if path.startswith("/files/"):
            parts = path.split("/", 3)
            if len(parts) == 4 and parts[2] in JOBS:
                return self._serve_file(JOBS[parts[2]]["dir"], parts[3])
        self._send(page("Not found", "<h1>Not found</h1><p><a href='/'>Back</a></p>"), code=404)

    def _serve_file(self, root, rel):
        full = os.path.realpath(os.path.join(root, rel))
        if not full.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(full):
            return self._send(b"not found", "text/plain", 404)
        ext = os.path.splitext(full)[1].lower()
        ctype = {".html": "text/html; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml", ".pdf": "application/pdf",
                 ".zip": "application/zip", ".tsv": "text/tab-separated-values", ".json": "application/json",
                 ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}.get(ext, "application/octet-stream")
        with open(full, "rb") as fh:
            data = fh.read()
        extra = {"Content-Disposition": f'attachment; filename="{os.path.basename(full)}"'} if ext in (".zip", ".xlsx") else None
        self._send(data, ctype, extra=extra)

    def _job_page(self, jid):
        job = JOBS.get(jid)
        if not job:
            return self._send(page("Unknown run", "<h1>Unknown run</h1><p><a href='/'>Back</a></p>"), code=404)
        log = ""
        lp = os.path.join(job["dir"], "results", "multiqtl.log")
        if os.path.exists(lp):
            with open(lp, encoding="utf-8") as fh:
                log = fh.read()[-6000:]
        st = job["status"]
        body = f"<h1>Run {html.escape(jid)}</h1><p class='sub'>Folder: <code>{html.escape(job['dir'])}</code></p>"
        body += f"<p class='status'>Status: {html.escape(st)}</p>"
        if job.get("checks"):
            body += "<div class='card'><h2>Input check</h2><ul class='checks-out'>" + "".join(
                f"<li class='{l}'>{'✓' if l == 'OK' else ('!' if l == 'WARNING' else '✗')} {html.escape(m)}</li>" for l, m in job["checks"]) + "</ul></div>"
        if st == "done":
            q = quote(jid)
            body += (f"<div class='btns'><a class='btn' href='/files/{q}/results/report.html' target='_blank'>Open report</a>"
                     f"<a class='btn secondary' href='/files/{q}/results.zip'>Download all results (.zip)</a>"
                     f"<a class='btn secondary' href='/files/{q}/results/multiqtl_results.xlsx'>Excel workbook</a></div>"
                     f"<p class='hint'>Re-run from a terminal with <code>multiqtl run \"{html.escape(os.path.join(job['dir'], 'config.yaml'))}\"</code></p>")
        if st == "failed":
            body += f"<div class='card'><h2 class='ERROR'>The run stopped</h2><pre>{html.escape(job.get('error', ''))}</pre></div>"
        if st == "checked":
            body += "<p><a href='/'>Back to the form</a> to run the analysis.</p>"
        if log:
            body += f"<h2>Log</h2><pre>{html.escape(log)}</pre>"
        body += "<p><a href='/'>New analysis</a></p>"
        self._send(page(f"MultiQTL – {jid}", body, refresh=3 if st not in ("done", "failed", "checked") else None))

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/demo":
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            jid, d = _new_job_dir("demo")
            JOBS[jid] = {"dir": d, "status": "queued"}
            threading.Thread(target=_run_demo, args=(jid,), daemon=True).start()
            return self._redirect(f"/job/{quote(jid)}")
        if path == "/submit":
            from .config_tools import check_inputs
            from .pipeline import DEFAULT_CONFIG, _merge
            f = _parse_multipart(self)
            jid, d = _new_job_dir(f.get("run_name", ""))
            cfg = _merge(DEFAULT_CONFIG, build_config(f, d))
            checks = check_inputs(cfg)
            _write_config(cfg, d)
            JOBS[jid] = {"dir": d, "status": "queued", "checks": checks}
            if any(l == "ERROR" for l, _ in checks):
                JOBS[jid]["status"] = "failed"
                JOBS[jid]["error"] = "Fix the errors listed under 'Input check' and submit again."
            elif f.get("action") == "check":
                JOBS[jid]["status"] = "checked"
            else:
                threading.Thread(target=_run_job, args=(jid, cfg), daemon=True).start()
            return self._redirect(f"/job/{quote(jid)}")
        self._send(b"not found", "text/plain", 404)


def serve(port: int = 8765, workspace: str = None, open_browser: bool = True, host: str = "127.0.0.1"):
    """Start the interface. ``host`` stays 127.0.0.1 (this computer only) except inside Docker."""
    global WORKSPACE
    if workspace:
        WORKSPACE = os.path.abspath(os.path.expanduser(workspace))
    os.makedirs(WORKSPACE, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}/"
    print(f"MultiQTL is running at {url}\nRuns are saved in {WORKSPACE}\nPress Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        httpd.server_close()
