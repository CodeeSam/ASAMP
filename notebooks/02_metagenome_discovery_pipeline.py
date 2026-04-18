# Notebook 2 — Metagenome AMP Discovery Pipeline
#
# Performs the full metagenome discovery pipeline for candidate AMPs:
#
#   1.  Load sample metadata from SRRs.csv
#   2.  Download mixed SRR and ERR accessions (single-end and paired-end)
#   3.  Trim reads with fastp
#   4.  Assemble reads with MEGAHIT
#   5.  Predict ORFs with MetaProdigal
#   6.  Filter peptides to 10–50 aa
#   7.  Deduplicate peptide sequences
#   8.  Cluster with CD-HIT
#   9.  Predict on cluster representatives only
#   10. Rank and prioritize candidates
#   11. Save publication-ready tables and figures
#
# Designed for: Kaggle, reproducibility, GitHub export, publication outputs.


# 0. Reproducible setup

import os
import re
import gc
import io
import csv
import sys
import json
import time
import math
import gzip
import glob
import shutil
import random
import hashlib
import zipfile
import textwrap
import logging
import pathlib
import subprocess
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import requests
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Python:", sys.version)
print("Working directory:", os.getcwd())


# 1. Configuration

CONFIG = {
    # ------------------------------------------------------------------
    # INPUT PATHS — update these to match your Kaggle dataset structure.
    #
    # On Kaggle, datasets are mounted at /kaggle/input/<dataset-slug>/.
    # To discover your exact slug and file paths, run in a notebook cell:
    #   import os; print(os.listdir("/kaggle/input"))
    # ------------------------------------------------------------------

    # Path to your sample sheet CSV (must have accession and body_site columns).
    "sample_sheet": "/kaggle/input/<your-dataset-slug>/accession_list.csv",
    "body_site_col": "body_site",
    "accession_col": "SRR_ID",

    # Path to the saved model artifact from Notebook 1.
    # If running both notebooks in the same Kaggle session, the model and
    # tokenizer are already in memory and this path is not used (see Section 18).
    # Otherwise, save Notebook 1's output as a Kaggle model and point here:
    #   e.g. "/kaggle/input/<your-model-slug>"
    "model_dir": "/kaggle/input/<your-model-slug>",

    # /kaggle/working is the writable output directory on Kaggle — no change needed.
    "work_dir": "/kaggle/working",
    "project_name": "metagenome_amp_discovery",
    "min_len_aa": 10,
    "max_len_aa": 50,

    # processing controls
    "run_download": True,
    "run_fastp": True,
    "run_megahit": True,
    "run_metaprodigal": True,
    "run_cdhit": True,
    "run_prediction": True,

    # batching / safety
    "limit_samples": None,   # set to an integer for pilot runs, e.g. 8
    "overwrite": False,
    "skip_existing": True,

    # fastp
    "fastp_threads": 4,
    "fastp_min_length": 50,

    # megahit
    "megahit_threads": 4,
    "megahit_min_contig_len": 200,

    # prodigal
    "prodigal_mode": "meta",

    # cdhit
    "cdhit_identity": 0.90,
    "cdhit_word_size": 5,
    "cdhit_threads": 4,
    "cdhit_memory_mb": 3200,

    # prediction
    "batch_size": 32,
    "pred_threshold": 0.50,
    "high_conf_threshold": 0.90,

    # figures
    "fig_dpi": 300
}

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
BASE = pathlib.Path(CONFIG["work_dir"]) / CONFIG["project_name"]
DIRS = {
    "base": BASE,
    "logs": BASE / "logs",
    "data": BASE / "data",
    "downloads": BASE / "data" / "downloads",
    "trimmed": BASE / "data" / "trimmed",
    "assemblies": BASE / "data" / "assemblies",
    "orfs": BASE / "data" / "orfs",
    "peptides": BASE / "data" / "peptides",
    "cluster": BASE / "data" / "cluster",
    "results": BASE / "results",
    "figures": BASE / "figures",
    "supplementary": BASE / "supplementary",
    "manifests": BASE / "manifests",
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

with open(DIRS["manifests"] / "notebook2_config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

print("Project base:", BASE)


# 2. Logging helpers

LOG_FILE = DIRS["logs"] / f"notebook2_{RUN_ID}.log"

logger = logging.getLogger("nb2")
logger.setLevel(logging.INFO)
logger.handlers = []

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

logger.info("Notebook 2 started.")
logger.info(f"Config saved to: {DIRS['manifests'] / 'notebook2_config.json'}")


# 3. Utility functions

def run_cmd(cmd, log_prefix="CMD", cwd=None, env=None, check=True):
    """Run a shell command safely and log stdout/stderr."""
    if isinstance(cmd, str):
        shell = True
        printable = cmd
    else:
        shell = False
        printable = " ".join(cmd)

    logger.info(f"[{log_prefix}] Running: {printable}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if proc.stdout.strip():
        logger.info(f"[{log_prefix}] STDOUT:\n{proc.stdout[:5000]}")
    if proc.stderr.strip():
        logger.info(f"[{log_prefix}] STDERR:\n{proc.stderr[:5000]}")
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {printable}")
    return proc

def sha256_of_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def maybe_write_records(records, out_fasta):
    count = 0
    with open(out_fasta, "w") as handle:
        for rec in records:
            SeqIO.write(rec, handle, "fasta")
            count += 1
    return count

def is_single_end(accession: str) -> bool:
    """Conservative accession-level check using ENA filereport."""
    url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport?"
        f"accession={accession}&result=read_run&fields=fastq_ftp,fastq_aspera,fastq_md5,library_layout&format=json"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError(f"No ENA metadata returned for accession {accession}")
    row = rows[0]
    layout = str(row.get("library_layout", "")).upper()
    fastq_ftp = row.get("fastq_ftp", "")
    parts = [x for x in str(fastq_ftp).split(";") if x.strip()]
    if layout == "SINGLE":
        return True
    if layout == "PAIRED":
        return False
    return len(parts) == 1

def get_ena_fastq_urls(accession: str):
    """Return downloadable HTTPS URLs from ENA for an accession."""
    url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport?"
        f"accession={accession}&result=read_run&fields=fastq_ftp,fastq_md5,library_layout&format=json"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError(f"No ENA file report found for accession: {accession}")
    row = rows[0]
    ftp_paths = [x for x in str(row.get("fastq_ftp", "")).split(";") if x.strip()]
    md5s = [x for x in str(row.get("fastq_md5", "")).split(";") if x.strip()]
    https_urls = []
    for p in ftp_paths:
        if p.startswith("ftp.sra.ebi.ac.uk/"):
            https_urls.append("https://" + p)
        elif p.startswith("ftp://"):
            https_urls.append(p.replace("ftp://", "https://"))
        else:
            https_urls.append("https://" + p)
    return {
        "layout": str(row.get("library_layout", "")).upper(),
        "urls": https_urls,
        "md5s": md5s
    }

def download_file(url: str, out_path: str, chunk_size=8 * 1024 * 1024):
    """Stream download a file over HTTPS."""
    out_path = str(out_path)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path

def parse_cdhit_clstr(clstr_path: str) -> pd.DataFrame:
    """Parse CD-HIT .clstr file into a table with cluster membership."""
    rows = []
    cluster_id = None
    with open(clstr_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[1])
                continue
            m = re.search(r">(.+?)\.\.\.", line)
            if m is None:
                continue
            seq_id = m.group(1)
            is_rep = line.endswith("*")
            rows.append({
                "cluster_id": cluster_id,
                "seq_id": seq_id,
                "is_representative": is_rep
            })
    return pd.DataFrame(rows)

def compute_peptide_properties(seq: str) -> Dict[str, float]:
    """Publication-friendly peptide descriptors."""
    pa = ProteinAnalysis(seq)
    aromaticity = pa.aromaticity()
    instability = pa.instability_index()
    gravy = pa.gravy()
    mw = pa.molecular_weight()
    pI = pa.isoelectric_point()
    charge_pH74 = pa.charge_at_pH(7.4)
    return {
        "length": len(seq),
        "aromaticity": aromaticity,
        "instability_index": instability,
        "gravy": gravy,
        "molecular_weight": mw,
        "pI": pI,
        "charge_pH74": charge_pH74
    }


# 4. Install system tools

def apt_install(packages):
    cmd = f"apt-get update -y && apt-get install -y {' '.join(packages)}"
    run_cmd(cmd, log_prefix="APT", check=True)

def pip_install(packages):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    run_cmd(cmd, log_prefix="PIP", check=True)

pip_install([
    "transformers>=4.39.0",
    "sentencepiece",
    "torch",
    "scikit-learn",
    "umap-learn"
])

apt_install([
    "fastp",
    "megahit",
    "prodigal",
    "cd-hit",
    "seqtk"
])

logger.info("Tool installation completed.")


# 5. Load sample sheet

samples = pd.read_csv(CONFIG["sample_sheet"])

required_cols = [CONFIG["accession_col"], CONFIG["body_site_col"]]
missing = [c for c in required_cols if c not in samples.columns]
if missing:
    raise ValueError(f"Missing required columns in sample sheet: {missing}")

samples = samples[[CONFIG["accession_col"], CONFIG["body_site_col"]]].copy()
samples.rename(columns={
    CONFIG["accession_col"]: "accession",
    CONFIG["body_site_col"]: "body_site"
}, inplace=True)

samples["accession"] = samples["accession"].astype(str).str.strip()
samples["body_site"] = samples["body_site"].astype(str).str.strip()

samples = samples.dropna().drop_duplicates().reset_index(drop=True)

if CONFIG["limit_samples"] is not None:
    samples = samples.iloc[:int(CONFIG["limit_samples"])].copy()

samples["accession_prefix"] = samples["accession"].str.extract(r"^([A-Z]+)")
samples["sample_id"] = samples["accession"]

samples.to_csv(DIRS["manifests"] / "sample_manifest_input.csv", index=False)

print(samples.head())
print("\nN samples:", len(samples))
print("\nBy body site:")
print(samples["body_site"].value_counts())


# 6. Figure helpers

FIG_DPI = CONFIG["fig_dpi"]

def savefig(path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

plt.figure(figsize=(7, 4))
samples["body_site"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Body site")
plt.ylabel("Number of samples")
plt.title("Sample distribution by body site")
savefig(DIRS["figures"] / "fig01_sample_distribution.png")


# 7. Download reads from ENA

download_records = []

for _, row in samples.iterrows():
    accession = row["accession"]
    body_site = row["body_site"]

    sample_dir = DIRS["downloads"] / accession
    sample_dir.mkdir(parents=True, exist_ok=True)

    try:
        ena_info = get_ena_fastq_urls(accession)
        urls = ena_info["urls"]
        md5s = ena_info["md5s"]
        layout = ena_info["layout"]

        if len(urls) == 0:
            raise ValueError(f"No FASTQ URLs found for {accession}")

        local_files = []
        for i, url in enumerate(urls):
            fname = os.path.basename(url)
            out_path = sample_dir / fname
            if out_path.exists() and CONFIG["skip_existing"]:
                logger.info(f"Skipping existing download: {out_path}")
            else:
                download_file(url, out_path)
            local_files.append(str(out_path))

        record = {
            "accession": accession,
            "body_site": body_site,
            "layout": layout,
            "n_fastq_files": len(local_files),
            "fastq_files": ";".join(local_files)
        }
        for j, md5 in enumerate(md5s):
            record[f"md5_{j+1}"] = md5
        download_records.append(record)

        logger.info(f"Downloaded {accession} ({layout}) -> {len(local_files)} file(s)")
    except Exception as e:
        logger.error(f"Download failed for {accession}: {e}")
        download_records.append({
            "accession": accession,
            "body_site": body_site,
            "layout": "UNKNOWN",
            "n_fastq_files": 0,
            "fastq_files": "",
            "error": str(e)
        })

download_df = pd.DataFrame(download_records)
download_df.to_csv(DIRS["results"] / "download_manifest.csv", index=False)

print(download_df.head())
print(download_df["layout"].value_counts(dropna=False))


# 8. Trim reads with fastp

trim_records = []

for _, row in download_df.iterrows():
    accession = row["accession"]
    body_site = row["body_site"]
    n_files = int(row["n_fastq_files"]) if pd.notna(row["n_fastq_files"]) else 0
    fastq_files = [x for x in str(row["fastq_files"]).split(";") if x.strip()]

    if n_files == 0 or len(fastq_files) == 0:
        trim_records.append({
            "accession": accession,
            "body_site": body_site,
            "trim_status": "skipped_no_download"
        })
        continue

    out_dir = DIRS["trimmed"] / accession
    out_dir.mkdir(parents=True, exist_ok=True)

    json_report = out_dir / f"{accession}_fastp.json"
    html_report = out_dir / f"{accession}_fastp.html"

    try:
        if len(fastq_files) == 1:
            in1 = fastq_files[0]
            out1 = out_dir / f"{accession}_trimmed.fastq.gz"
            cmd = [
                "fastp",
                "-i", in1,
                "-o", str(out1),
                "-j", str(json_report),
                "-h", str(html_report),
                "--length_required", str(CONFIG["fastp_min_length"]),
                "-w", str(CONFIG["fastp_threads"])
            ]
            run_cmd(cmd, log_prefix=f"FASTP-{accession}")
            trim_records.append({
                "accession": accession,
                "body_site": body_site,
                "layout": "SINGLE",
                "trim_status": "ok",
                "trimmed_files": str(out1),
                "fastp_json": str(json_report),
                "fastp_html": str(html_report)
            })
        elif len(fastq_files) == 2:
            in1, in2 = fastq_files
            out1 = out_dir / f"{accession}_R1_trimmed.fastq.gz"
            out2 = out_dir / f"{accession}_R2_trimmed.fastq.gz"
            cmd = [
                "fastp",
                "-i", in1,
                "-I", in2,
                "-o", str(out1),
                "-O", str(out2),
                "-j", str(json_report),
                "-h", str(html_report),
                "--length_required", str(CONFIG["fastp_min_length"]),
                "-w", str(CONFIG["fastp_threads"])
            ]
            run_cmd(cmd, log_prefix=f"FASTP-{accession}")
            trim_records.append({
                "accession": accession,
                "body_site": body_site,
                "layout": "PAIRED",
                "trim_status": "ok",
                "trimmed_files": f"{out1};{out2}",
                "fastp_json": str(json_report),
                "fastp_html": str(html_report)
            })
        else:
            raise ValueError(f"Unexpected number of FASTQ files for {accession}: {len(fastq_files)}")
    except Exception as e:
        logger.error(f"fastp failed for {accession}: {e}")
        trim_records.append({
            "accession": accession,
            "body_site": body_site,
            "trim_status": "error",
            "error": str(e)
        })

trim_df = pd.DataFrame(trim_records)
trim_df.to_csv(DIRS["results"] / "trim_manifest.csv", index=False)

print(trim_df.head())


# 9. Parse fastp reports and make QC figure

qc_rows = []
for _, row in trim_df.iterrows():
    if row.get("trim_status") != "ok":
        continue
    fp = row.get("fastp_json", "")
    if not fp or not os.path.exists(fp):
        continue
    try:
        with open(fp, "r") as f:
            js = json.load(f)
        summary = js.get("summary", {})
        before = summary.get("before_filtering", {})
        after = summary.get("after_filtering", {})
        qc_rows.append({
            "accession": row["accession"],
            "body_site": row["body_site"],
            "layout": row.get("layout", ""),
            "reads_before": before.get("total_reads", np.nan),
            "reads_after": after.get("total_reads", np.nan),
            "bases_before": before.get("total_bases", np.nan),
            "bases_after": after.get("total_bases", np.nan),
            "q20_rate_after": after.get("q20_rate", np.nan),
            "q30_rate_after": after.get("q30_rate", np.nan),
            "gc_content_after": after.get("gc_content", np.nan),
        })
    except Exception as e:
        logger.error(f"Could not parse fastp report for {row['accession']}: {e}")

qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(DIRS["supplementary"] / "table_s1_qc_summary.csv", index=False)

if len(qc_df):
    plt.figure(figsize=(8, 4))
    plt.hist(qc_df["reads_after"].dropna(), bins=20)
    plt.xlabel("Reads after trimming")
    plt.ylabel("Number of samples")
    plt.title("Distribution of reads after trimming")
    savefig(DIRS["figures"] / "fig02_reads_after_trimming_distribution.png")

    plt.figure(figsize=(8, 4))
    plt.scatter(qc_df["reads_before"], qc_df["reads_after"])
    plt.xlabel("Reads before trimming")
    plt.ylabel("Reads after trimming")
    plt.title("Read counts before vs after trimming")
    savefig(DIRS["figures"] / "fig03_reads_before_vs_after.png")

print(qc_df.head())


# 10. Assemble with MEGAHIT

assemblies_dir = DIRS["assemblies"]
if assemblies_dir.exists():
    shutil.rmtree(assemblies_dir)
assemblies_dir.mkdir(parents=True, exist_ok=True)

assembly_rows = []

for _, row in trim_df.iterrows():
    accession = row["accession"]
    body_site = row["body_site"]

    if row.get("trim_status") != "ok":
        assembly_rows.append({
            "accession": accession,
            "body_site": body_site,
            "assembly_status": "skipped_no_trim"
        })
        continue

    trimmed_files = [x for x in str(row["trimmed_files"]).split(";") if x.strip()]
    out_dir = DIRS["assemblies"] / accession
    contigs_fa = out_dir / "final.contigs.fa"

    if contigs_fa.exists() and CONFIG["skip_existing"]:
        logger.info(f"Skipping MEGAHIT for existing sample: {accession}")
        assembly_rows.append({
            "accession": accession,
            "body_site": body_site,
            "assembly_status": "ok_existing",
            "contigs_fa": str(contigs_fa)
        })
        continue

    # MEGAHIT requires output dir does NOT already exist
    if out_dir.exists():
        shutil.rmtree(out_dir)
        logger.info(f"Removed pre-existing assembly directory for rerun: {out_dir}")

    try:
        if len(trimmed_files) == 1:
            cmd = [
                "megahit",
                "-r", trimmed_files[0],
                "-o", str(out_dir),
                "--min-contig-len", str(CONFIG["megahit_min_contig_len"]),
                "-t", str(CONFIG["megahit_threads"])
            ]
        elif len(trimmed_files) == 2:
            cmd = [
                "megahit",
                "-1", trimmed_files[0],
                "-2", trimmed_files[1],
                "-o", str(out_dir),
                "--min-contig-len", str(CONFIG["megahit_min_contig_len"]),
                "-t", str(CONFIG["megahit_threads"])
            ]
        else:
            raise ValueError(f"Unexpected trimmed file count for {accession}: {len(trimmed_files)}")

        run_cmd(cmd, log_prefix=f"MEGAHIT-{accession}")

        if not contigs_fa.exists():
            raise FileNotFoundError(f"MEGAHIT finished but contigs file not found: {contigs_fa}")

        assembly_rows.append({
            "accession": accession,
            "body_site": body_site,
            "assembly_status": "ok",
            "contigs_fa": str(contigs_fa)
        })

    except Exception as e:
        logger.error(f"MEGAHIT failed for {accession}: {e}")
        assembly_rows.append({
            "accession": accession,
            "body_site": body_site,
            "assembly_status": "error",
            "error": str(e)
        })

assembly_df = pd.DataFrame(assembly_rows)
assembly_df.to_csv(DIRS["results"] / "assembly_manifest.csv", index=False)

print(assembly_df.head())
print(assembly_df["assembly_status"].value_counts(dropna=False))


# 11. Assembly statistics

def fasta_lengths(path):
    lens = []
    for rec in SeqIO.parse(path, "fasta"):
        lens.append(len(rec.seq))
    return lens

def n50(lengths):
    if not lengths:
        return 0
    lengths = sorted(lengths, reverse=True)
    total = sum(lengths)
    csum = 0
    for L in lengths:
        csum += L
        if csum >= total / 2:
            return L
    return 0

assembly_stats = []
for _, row in assembly_df.iterrows():
    if str(row.get("assembly_status", "")).startswith("ok"):
        contigs_fa = row["contigs_fa"]
        if os.path.exists(contigs_fa):
            lens = fasta_lengths(contigs_fa)
            assembly_stats.append({
                "accession": row["accession"],
                "body_site": row["body_site"],
                "n_contigs": len(lens),
                "total_assembly_bp": int(sum(lens)),
                "contig_n50": int(n50(lens)),
                "mean_contig_len": float(np.mean(lens)) if len(lens) else 0.0,
                "median_contig_len": float(np.median(lens)) if len(lens) else 0.0
            })

assembly_stats_df = pd.DataFrame(assembly_stats)
assembly_stats_df.to_csv(DIRS["supplementary"] / "table_s2_assembly_stats.csv", index=False)

if len(assembly_stats_df):
    plt.figure(figsize=(8, 4))
    plt.hist(assembly_stats_df["n_contigs"], bins=20)
    plt.xlabel("Number of contigs")
    plt.ylabel("Number of samples")
    plt.title("Assembly contig counts by sample")
    savefig(DIRS["figures"] / "fig04_assembly_contig_counts.png")

    plt.figure(figsize=(8, 4))
    plt.hist(assembly_stats_df["contig_n50"], bins=20)
    plt.xlabel("Contig N50")
    plt.ylabel("Number of samples")
    plt.title("Assembly N50 distribution")
    savefig(DIRS["figures"] / "fig05_assembly_n50_distribution.png")

print(assembly_stats_df.head())


# 12. ORF calling with MetaProdigal

orf_rows = []

for _, row in assembly_df.iterrows():
    accession = row["accession"]
    body_site = row["body_site"]

    if not str(row.get("assembly_status", "")).startswith("ok"):
        orf_rows.append({
            "accession": accession,
            "body_site": body_site,
            "orf_status": "skipped_no_assembly"
        })
        continue

    contigs_fa = row["contigs_fa"]
    out_dir = DIRS["orfs"] / accession
    out_dir.mkdir(parents=True, exist_ok=True)

    faa_out = out_dir / f"{accession}.orfs.faa"
    gff_out = out_dir / f"{accession}.orfs.gff"
    nuc_out = out_dir / f"{accession}.orfs.fna"

    if faa_out.exists() and CONFIG["skip_existing"]:
        logger.info(f"Skipping existing MetaProdigal output for {accession}")
        orf_rows.append({
            "accession": accession,
            "body_site": body_site,
            "orf_status": "ok_existing",
            "orf_faa": str(faa_out),
            "orf_gff": str(gff_out),
            "orf_fna": str(nuc_out)
        })
        continue

    try:
        cmd = [
            "prodigal",
            "-i", contigs_fa,
            "-a", str(faa_out),
            "-d", str(nuc_out),
            "-f", "gff",
            "-o", str(gff_out),
            "-p", CONFIG["prodigal_mode"]
        ]
        run_cmd(cmd, log_prefix=f"PRODIGAL-{accession}")

        if not faa_out.exists():
            raise FileNotFoundError(f"Expected ORF amino acid FASTA not found: {faa_out}")

        orf_rows.append({
            "accession": accession,
            "body_site": body_site,
            "orf_status": "ok",
            "orf_faa": str(faa_out),
            "orf_gff": str(gff_out),
            "orf_fna": str(nuc_out)
        })
    except Exception as e:
        logger.error(f"MetaProdigal failed for {accession}: {e}")
        orf_rows.append({
            "accession": accession,
            "body_site": body_site,
            "orf_status": "error",
            "error": str(e)
        })

orf_df = pd.DataFrame(orf_rows)
orf_df.to_csv(DIRS["results"] / "orf_manifest.csv", index=False)

print(orf_df.head())


# 13. Filter peptides to 10–50 aa and standardize IDs

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

peptide_rows = []
all_filtered_records = []

for _, row in orf_df.iterrows():
    accession = row["accession"]
    body_site = row["body_site"]

    if not str(row.get("orf_status", "")).startswith("ok"):
        peptide_rows.append({
            "accession": accession,
            "body_site": body_site,
            "peptide_status": "skipped_no_orf"
        })
        continue

    faa = row["orf_faa"]
    out_fa = DIRS["peptides"] / f"{accession}.filtered_{CONFIG['min_len_aa']}_{CONFIG['max_len_aa']}aa.faa"

    kept = []
    n_total = 0
    n_kept = 0

    for rec in SeqIO.parse(faa, "fasta"):
        n_total += 1
        seq = str(rec.seq).replace("*", "").strip().upper()
        if not (CONFIG["min_len_aa"] <= len(seq) <= CONFIG["max_len_aa"]):
            continue
        if not AA_RE.match(seq):
            continue

        # traceable ID: accession|bodysite|original_id
        new_id = f"{accession}|{body_site}|{rec.id}"
        new_rec = SeqRecord(Seq(seq), id=new_id, description="")
        kept.append(new_rec)
        all_filtered_records.append(new_rec)
        n_kept += 1

    maybe_write_records(kept, out_fa)

    peptide_rows.append({
        "accession": accession,
        "body_site": body_site,
        "peptide_status": "ok",
        "n_orfs_total": n_total,
        "n_peptides_filtered": n_kept,
        "filtered_faa": str(out_fa)
    })

peptide_df = pd.DataFrame(peptide_rows)
peptide_df.to_csv(DIRS["supplementary"] / "table_s3_orf_and_peptide_counts_per_sample.csv", index=False)

all_filtered_fa = DIRS["peptides"] / "all_filtered_peptides.faa"
maybe_write_records(all_filtered_records, all_filtered_fa)

print(peptide_df.head())
print("Total filtered peptides:", len(all_filtered_records))


# 14. Figures: ORF and peptide counts

if len(peptide_df):
    tmp = peptide_df.dropna(subset=["n_orfs_total", "n_peptides_filtered"]).copy()

    plt.figure(figsize=(8, 4))
    plt.hist(tmp["n_orfs_total"], bins=20)
    plt.xlabel("Predicted ORFs per sample")
    plt.ylabel("Number of samples")
    plt.title("Distribution of ORF counts per sample")
    savefig(DIRS["figures"] / "fig06_orf_count_distribution.png")

    plt.figure(figsize=(8, 4))
    plt.hist(tmp["n_peptides_filtered"], bins=20)
    plt.xlabel("Filtered 10–50 aa peptides per sample")
    plt.ylabel("Number of samples")
    plt.title("Distribution of short peptide counts per sample")
    savefig(DIRS["figures"] / "fig07_filtered_peptide_count_distribution.png")

lengths = [len(rec.seq) for rec in SeqIO.parse(all_filtered_fa, "fasta")]
if lengths:
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=range(CONFIG["min_len_aa"], CONFIG["max_len_aa"] + 2))
    plt.xlabel("Peptide length (aa)")
    plt.ylabel("Count")
    plt.title("Length distribution of filtered metagenomic peptides")
    savefig(DIRS["figures"] / "fig08_filtered_peptide_length_distribution.png")


# 15. Exact deduplication

dedup_map = {}
dedup_rows = []

for rec in SeqIO.parse(all_filtered_fa, "fasta"):
    seq = str(rec.seq)
    if seq not in dedup_map:
        dedup_map[seq] = {
            "representative_id": rec.id,
            "sequence": seq,
            "members": [rec.id]
        }
    else:
        dedup_map[seq]["members"].append(rec.id)

dedup_records = []
for i, (seq, payload) in enumerate(dedup_map.items(), start=1):
    rep_id = f"DEDUPSEQ_{i:07d}|orig={payload['representative_id']}"
    dedup_records.append(SeqRecord(Seq(seq), id=rep_id, description=""))

dedup_fa = DIRS["cluster"] / "all_filtered_peptides.dedup.faa"
maybe_write_records(dedup_records, dedup_fa)

dedup_members = []
for seq, payload in dedup_map.items():
    for member in payload["members"]:
        dedup_members.append({
            "sequence": seq,
            "dedup_representative_original_id": payload["representative_id"],
            "member_id": member
        })
dedup_members_df = pd.DataFrame(dedup_members)
dedup_members_df.to_csv(DIRS["supplementary"] / "table_s4_dedup_membership.csv", index=False)

summary = {
    "n_filtered_peptides_total": int(sum(1 for _ in SeqIO.parse(all_filtered_fa, "fasta"))),
    "n_unique_sequences_after_exact_dedup": len(dedup_records)
}
save_json(summary, DIRS["results"] / "dedup_summary.json")

print(summary)


# 16. Cluster unique peptides with CD-HIT

cdhit_out = DIRS["cluster"] / "all_filtered_peptides.dedup.cdhit"
cdhit_clstr = str(cdhit_out) + ".clstr"

if not pathlib.Path(str(cdhit_out)).exists() or not pathlib.Path(cdhit_clstr).exists() or not CONFIG["skip_existing"]:
    cmd = [
        "cd-hit",
        "-i", str(dedup_fa),
        "-o", str(cdhit_out),
        "-c", str(CONFIG["cdhit_identity"]),
        "-n", str(CONFIG["cdhit_word_size"]),
        "-T", str(CONFIG["cdhit_threads"]),
        "-M", str(CONFIG["cdhit_memory_mb"]),
        "-d", "0"
    ]
    run_cmd(cmd, log_prefix="CDHIT")
else:
    logger.info("Skipping CD-HIT because output already exists.")

cluster_df = parse_cdhit_clstr(cdhit_clstr)
cluster_df.to_csv(DIRS["results"] / "cluster_membership.csv", index=False)

rep_fa = str(cdhit_out)
rep_records = list(SeqIO.parse(rep_fa, "fasta"))
print("Cluster representatives:", len(rep_records))

cluster_sizes = cluster_df.groupby("cluster_id").size().reset_index(name="cluster_size")
cluster_sizes.to_csv(DIRS["supplementary"] / "table_s5_cluster_sizes.csv", index=False)
print(cluster_sizes.head())


# 17. Cluster figures

if len(cluster_sizes):
    plt.figure(figsize=(8, 4))
    plt.hist(cluster_sizes["cluster_size"], bins=30)
    plt.xlabel("Cluster size")
    plt.ylabel("Number of clusters")
    plt.title("Distribution of CD-HIT cluster sizes")
    savefig(DIRS["figures"] / "fig09_cluster_size_distribution.png")

    plt.figure(figsize=(8, 4))
    plt.hist(np.log10(cluster_sizes["cluster_size"]), bins=30)
    plt.xlabel("log10(cluster size)")
    plt.ylabel("Number of clusters")
    plt.title("Log-scaled distribution of cluster sizes")
    savefig(DIRS["figures"] / "fig10_cluster_size_log_distribution.png")


# 18. Reuse trained model already in memory

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

if "model" not in globals():
    raise RuntimeError("Model object not found in memory.")
if "tokenizer" not in globals():
    raise RuntimeError("Tokenizer object not found in memory.")

model = model.to(DEVICE)
model.eval()

if "cfg" in globals() and hasattr(cfg, "max_length_tokens"):
    MAX_LEN = int(cfg.max_length_tokens)
else:
    MAX_LEN = int(CONFIG["max_len_aa"])

print("Using in-memory trained model.")
print("MAX_LEN:", MAX_LEN)


# 19. Prepare cluster representatives for inference

rep_rows = []
for rec in SeqIO.parse(rep_fa, "fasta"):
    seq = str(rec.seq)
    rep_rows.append({
        "rep_id": rec.id,
        "sequence": seq,
        "length": len(seq)
    })

rep_df = pd.DataFrame(rep_rows)
rep_df = rep_df.merge(
    cluster_df[cluster_df["is_representative"] == True][["cluster_id", "seq_id"]],
    left_on="rep_id",
    right_on="seq_id",
    how="left"
).drop(columns=["seq_id"])

rep_df = rep_df.merge(cluster_sizes, on="cluster_id", how="left")
rep_df.to_csv(DIRS["results"] / "cluster_representatives_pre_prediction.csv", index=False)

print(rep_df.head())
print("N representatives:", len(rep_df))


# 20. Predict AMP probability on representatives only

@torch.no_grad()
def predict_probabilities(sequences, batch_size=32):
    probs = []

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        formatted = [" ".join(list(seq)) for seq in batch]

        toks = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        toks = {
            k: v.to(DEVICE)
            for k, v in toks.items()
            if k in ["input_ids", "attention_mask"]
        }

        outputs = model(**toks)

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        elif logits.ndim == 2 and logits.shape[1] == 2:
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        probs.extend(p.tolist())

        del toks, outputs, logits
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return np.array(probs)


if CONFIG["run_prediction"]:
    probs = predict_probabilities(rep_df["sequence"].tolist(), batch_size=CONFIG["batch_size"])
    rep_df["Prob_AMP"] = probs
    rep_df["Pred_Class"] = (rep_df["Prob_AMP"] >= CONFIG["pred_threshold"]).astype(int)
    rep_df["High_Confidence"] = (rep_df["Prob_AMP"] >= CONFIG["high_conf_threshold"]).astype(int)
else:
    raise RuntimeError("Prediction disabled in config.")

rep_df.to_csv(DIRS["results"] / "cluster_representatives_predictions.csv", index=False)

print(rep_df.sort_values("Prob_AMP", ascending=False).head(10))


# 21. Add peptide properties and provenance

dedup_rep_to_members = (
    dedup_members_df.groupby("dedup_representative_original_id")["member_id"]
    .apply(list).to_dict()
)

prop_rows = []
for _, row in rep_df.iterrows():
    props = compute_peptide_properties(row["sequence"])
    merged = dict(row.to_dict())
    merged.update(props)
    prop_rows.append(merged)

pred_props_df = pd.DataFrame(prop_rows)

def extract_orig_id(rep_id):
    m = re.search(r"\|orig=(.+)$", rep_id)
    return m.group(1) if m else None

pred_props_df["orig_id"] = pred_props_df["rep_id"].apply(extract_orig_id)
pred_props_df["orig_accession"] = pred_props_df["orig_id"].str.split("|").str[0]
pred_props_df["orig_body_site"] = pred_props_df["orig_id"].str.split("|").str[1]

pred_props_df.to_csv(DIRS["supplementary"] / "table_s6_representative_predictions_with_properties.csv", index=False)

print(pred_props_df.head())


# 22. Prediction figures

plt.figure(figsize=(8, 4))
plt.hist(pred_props_df["Prob_AMP"], bins=30)
plt.xlabel("Predicted AMP probability")
plt.ylabel("Number of representatives")
plt.title("AMP probability distribution across cluster representatives")
savefig(DIRS["figures"] / "fig11_probability_distribution.png")

plt.figure(figsize=(8, 4))
plt.hist(pred_props_df.loc[pred_props_df["Pred_Class"] == 1, "Prob_AMP"], bins=30)
plt.xlabel("Predicted AMP probability")
plt.ylabel("Count")
plt.title("Probability distribution among predicted positives")
savefig(DIRS["figures"] / "fig12_probability_distribution_predicted_positives.png")

if pred_props_df["orig_body_site"].notna().any():
    tmp = pred_props_df.groupby("orig_body_site")["Pred_Class"].sum().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    tmp.plot(kind="bar")
    plt.xlabel("Body site")
    plt.ylabel("Predicted AMP-positive representatives")
    plt.title("Predicted AMP-positive representatives by body site")
    savefig(DIRS["figures"] / "fig13_positive_representatives_by_body_site.png")

hc = pred_props_df[pred_props_df["High_Confidence"] == 1].copy()
if len(hc):
    plt.figure(figsize=(8, 4))
    plt.hist(hc["charge_pH74"], bins=30)
    plt.xlabel("Charge at pH 7.4")
    plt.ylabel("Count")
    plt.title("Charge distribution of high-confidence candidates")
    savefig(DIRS["figures"] / "fig14_highconf_charge_distribution.png")

    plt.figure(figsize=(8, 4))
    plt.hist(hc["gravy"], bins=30)
    plt.xlabel("GRAVY")
    plt.ylabel("Count")
    plt.title("Hydrophobicity distribution of high-confidence candidates")
    savefig(DIRS["figures"] / "fig15_highconf_gravy_distribution.png")


# Top candidates — tiered shortlists


# Tier 1: high probability + physicochemical filters
tier1 = pred_props_df[
    (pred_props_df["Prob_AMP"] >= 0.9) &
    (pred_props_df["charge_pH74"] >= 2) &
    (pred_props_df["gravy"] <= 0.5)
].copy()

# Tier 2: moderate probability + charge filter
tier2 = pred_props_df[
    (pred_props_df["Prob_AMP"] >= 0.8) &
    (pred_props_df["charge_pH74"] >= 1)
].copy()

# Tier 3: all predicted positives at default threshold
tier3 = pred_props_df[pred_props_df["Pred_Class"] == 1].copy()

tier1 = tier1.sort_values("Prob_AMP", ascending=False)
top_candidates = tier1.head(30)

tier1.to_csv(DIRS["results"] / "tier1_high_confidence_candidates.csv", index=False)
tier2.to_csv(DIRS["results"] / "tier2_moderate_candidates.csv", index=False)
top_candidates.to_csv(DIRS["results"] / "top30_final_candidates.csv", index=False)

summary = {
    "total_representatives": len(pred_props_df),
    "predicted_positive": int((pred_props_df["Pred_Class"] == 1).sum()),
    "high_confidence": len(tier1),
    "moderate_confidence": len(tier2),
}

print(summary)


# 23. Expand predictions back to all cluster members

cluster_rep_map = (
    cluster_df[cluster_df["is_representative"] == True][["cluster_id", "seq_id"]]
    .rename(columns={"seq_id": "rep_id"})
)

full_cluster_map = cluster_df.merge(cluster_rep_map, on="cluster_id", how="left")
full_pred_df = full_cluster_map.merge(
    pred_props_df,
    on=["cluster_id", "rep_id"],
    how="left",
    suffixes=("", "_rep")
)

full_member_table = full_pred_df[[
    "cluster_id", "cluster_size", "seq_id", "is_representative", "rep_id",
    "sequence", "Prob_AMP", "Pred_Class", "High_Confidence",
    "length", "charge_pH74", "gravy", "pI", "molecular_weight",
    "orig_accession", "orig_body_site"
]].copy()

full_member_table.to_csv(DIRS["supplementary"] / "table_s7_cluster_member_predictions.csv", index=False)

print(full_member_table.head())


# 24. Final candidate shortlist


shortlist = pred_props_df[
    (pred_props_df["Prob_AMP"] >= 0.9) &
    (pred_props_df["charge_pH74"] >= 2) &
    (pred_props_df["gravy"] <= 0.5)
].copy()

shortlist = shortlist.sort_values(
    ["Prob_AMP", "cluster_size", "length"],
    ascending=[False, False, True]
).reset_index(drop=True)

shortlist["candidate_rank"] = np.arange(1, len(shortlist) + 1)

shortlist.to_csv(DIRS["results"] / "final_candidate_shortlist.csv", index=False)

shortlist_ids = set(shortlist["rep_id"].tolist())
shortlist_records = [rec for rec in SeqIO.parse(rep_fa, "fasta") if rec.id in shortlist_ids]
shortlist_fa = DIRS["results"] / "final_candidate_shortlist.faa"
maybe_write_records(shortlist_records, shortlist_fa)

print(shortlist.head(20))
print("Shortlist size:", len(shortlist))


# Quick pipeline integrity check

checks = {
    "samples": "samples" in globals(),
    "download_df": "download_df" in globals(),
    "trim_df": "trim_df" in globals(),
    "assembly_df": "assembly_df" in globals(),
    "orf_df": "orf_df" in globals(),
    "dedup_records": "dedup_records" in globals(),
    "rep_df": "rep_df" in globals(),
    "pred_props_df": "pred_props_df" in globals(),
    "shortlist": "shortlist" in globals(),
}

print("Variable presence:")
for k, v in checks.items():
    print(f"{k}: {v}")

print("\nSanity checks:")
print("rep_df size:", len(rep_df) if "rep_df" in globals() else "MISSING")
print("pred_props_df size:", len(pred_props_df) if "pred_props_df" in globals() else "MISSING")
print("shortlist size:", len(shortlist) if "shortlist" in globals() else "MISSING")

if "pred_props_df" in globals():
    print("\nPrediction columns present:",
          all(col in pred_props_df.columns for col in ["Prob_AMP", "Pred_Class", "High_Confidence"]))


# 25. Supplementary tables manifest

supp_manifest = pd.DataFrame([
    {"table_id": "Table S1", "file": str(DIRS["supplementary"] / "table_s1_qc_summary.csv"), "description": "Read QC summary after fastp"},
    {"table_id": "Table S2", "file": str(DIRS["supplementary"] / "table_s2_assembly_stats.csv"), "description": "Assembly statistics per sample"},
    {"table_id": "Table S3", "file": str(DIRS["supplementary"] / "table_s3_orf_and_peptide_counts_per_sample.csv"), "description": "ORF and filtered short-peptide counts per sample"},
    {"table_id": "Table S4", "file": str(DIRS["supplementary"] / "table_s4_dedup_membership.csv"), "description": "Exact deduplication membership"},
    {"table_id": "Table S5", "file": str(DIRS["supplementary"] / "table_s5_cluster_sizes.csv"), "description": "CD-HIT cluster sizes"},
    {"table_id": "Table S6", "file": str(DIRS["supplementary"] / "table_s6_representative_predictions_with_properties.csv"), "description": "Representative predictions with physicochemical properties"},
    {"table_id": "Table S7", "file": str(DIRS["supplementary"] / "table_s7_cluster_member_predictions.csv"), "description": "Cluster member predictions expanded from representatives"},
])
supp_manifest.to_csv(DIRS["supplementary"] / "supplementary_manifest.csv", index=False)
print(supp_manifest)


# 26. Figure manifest and run summary

figure_files = sorted([str(p) for p in DIRS["figures"].glob("*") if p.is_file()])
fig_manifest = pd.DataFrame({"figure_file": figure_files})
fig_manifest.to_csv(DIRS["figures"] / "figures_manifest.csv", index=False)

run_summary = {
    "run_id": RUN_ID,
    "n_samples_input": int(len(samples)),
    "n_samples_download_ok": int((download_df["n_fastq_files"] > 0).sum()) if "n_fastq_files" in download_df.columns else 0,
    "n_samples_trim_ok": int((trim_df["trim_status"] == "ok").sum()) if "trim_status" in trim_df.columns else 0,
    "n_samples_assembly_ok": int(assembly_df["assembly_status"].astype(str).str.startswith("ok").sum()) if "assembly_status" in assembly_df.columns else 0,
    "n_samples_orf_ok": int(orf_df["orf_status"].astype(str).str.startswith("ok").sum()) if "orf_status" in orf_df.columns else 0,
    "n_filtered_peptides_total": int(sum(1 for _ in SeqIO.parse(all_filtered_fa, "fasta"))) if os.path.exists(all_filtered_fa) else 0,
    "n_exact_unique_peptides": int(len(dedup_records)),
    "n_cluster_representatives": int(len(rep_df)),
    "n_predicted_positive_representatives": int((pred_props_df["Pred_Class"] == 1).sum()) if "Pred_Class" in pred_props_df.columns else 0,
    "n_high_confidence_representatives": int((pred_props_df["High_Confidence"] == 1).sum()) if "High_Confidence" in pred_props_df.columns else 0,
    "final_shortlist_size": int(len(shortlist))
}
save_json(run_summary, DIRS["results"] / "run_summary.json")
print(pd.DataFrame([run_summary]))
