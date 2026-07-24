# Notebook 1 — AMP Classifier Training with ProtBert

# Install dependencies (Kaggle)

import subprocess

def run_cmd(cmd, check=True):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=check)

run_cmd("python -m pip install -q --upgrade pip")
run_cmd("python -m pip install -q biopython pandas numpy scipy scikit-learn matplotlib tqdm pyyaml nbformat requests")
run_cmd("python -m pip install -q transformers accelerate sentencepiece")
run_cmd("apt-get update -qq")
run_cmd("apt-get install -y -qq cd-hit")


# Imports

import os
import re
import sys
import json
import time
import math
import random
import platform
import hashlib
import requests
import subprocess
from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)


# Configuration

@dataclass
class Config:
    # ------------------------------------------------------------------
    # INPUT PATHS — update these to match your Kaggle dataset structure.
    #
    # On Kaggle, datasets are mounted at:
    #   /kaggle/input/<dataset-slug>/
    # ------------------------------------------------------------------
    amp_fasta: str = "/kaggle/input/<your-dataset-slug>/full_AMPs.fasta"
    srr_csv: str = "/kaggle/input/<your-dataset-slug>/SRRs.csv"

    # UniProt non-AMP acquisition
    download_nonamp_from_uniprot: bool = True
    nonamp_fasta: str = "/kaggle/working/non_AMPs_filtered.fasta"
    nonamp_raw_fasta: str = "/kaggle/working/non_AMPs_raw_unfiltered.fasta"
    uniprot_metadata_tsv: str = "/kaggle/working/non_AMPs_uniprot_metadata.tsv"
    uniprot_query_log_json: str = "/kaggle/working/non_AMPs_uniprot_query_log.json"
    uniprot_base_url: str = "https://rest.uniprot.org/uniprotkb/search"
    uniprot_page_size: int = 500
    uniprot_request_timeout: int = 120
    nonamp_target_multiplier: float = 1.0
    exclusion_keywords: tuple = (
        "antimicrobial", "antibacterial", "antibiotic", "defensin", "bacteriocin",
        "microbicidal", "antifungal", "antiviral", "antiparasitic", "host-defense",
        "host defense", "toxin", "venom", "fungicidal", "microbiocidal", "kill bacteria",
        "kills bacteria", "antimycobacterial", "lantibiotic", "bactericidal", "virucidal"
    )

    # Optional: provide pre-assembled external benchmark FASTAs instead
    # of letting the notebook build them from DBAASP + UniProt.
    # Leave as empty strings "" to use the auto-build path.
    external_amp_fasta: str = ""
    external_nonamp_fasta: str = ""

    # All outputs are written here. /kaggle/working is writable on Kaggle.
    output_root: str = "/kaggle/working/amp_protbert_training"

    min_len: int = 10
    max_len: int = 50
    valid_aas: str = "ACDEFGHIKLMNPQRSTVWY"

    cdhit_identity: float = 0.80
    cdhit_word_length: int = 5
    cdhit_memory_mb: int = 32000
    cdhit_threads: int = 2

    test_size: float = 0.15
    val_size: float = 0.15
    split_seed: int = 42

    model_name: str = "Rostlab/prot_bert"
    max_length_tokens: int = 52
    dropout: float = 0.20
    classifier_hidden_dim: int = 256
    trainable_encoder_layers: int = 2

    seed: int = 42
    batch_size: int = 16
    num_workers: int = 2
    epochs: int = 6
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.02
    early_stopping_patience: int = 3
    monitor_metric: str = "val_mcc"
    use_amp: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
print(cfg)


# Reproducibility and paths

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)

OUT = Path(cfg.output_root)
DIRS = {
    "root": OUT,
    "data": OUT / "data",
    "tables": OUT / "tables",
    "figures": OUT / "figures",
    "models": OUT / "models",
    "logs": OUT / "logs",
    "splits": OUT / "splits",
    "configs": OUT / "configs",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

print("Output root:", OUT)
print("AMP FASTA:", cfg.amp_fasta)
print("SRR CSV:", cfg.srr_csv)
print("UniProt non-AMP FASTA:", cfg.nonamp_fasta)


# Helper utilities

VALID_AA_SET = set(cfg.valid_aas)

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def clean_sequence(seq: str) -> str:
    return str(seq).upper().strip().replace("*", "")

def is_valid_sequence(seq: str) -> bool:
    return len(seq) > 0 and set(seq).issubset(VALID_AA_SET)

def fasta_to_df(fasta_path: str, label: int, source_name: str) -> pd.DataFrame:
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")
    rows = []
    for rec in SeqIO.parse(str(path), "fasta"):
        seq = clean_sequence(rec.seq)
        rows.append({
            "record_id": str(rec.id),
            "description": str(rec.description),
            "sequence": seq,
            "label": int(label),
            "source": source_name,
            "length": len(seq),
            "seq_sha1": sha1_text(seq),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No sequences parsed from {path}")
    return df

def filter_short_peptides(df: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    out = df.copy()
    out["valid_seq"] = out["sequence"].map(is_valid_sequence)
    out = out[out["valid_seq"]].copy()
    out = out[(out["length"] >= min_len) & (out["length"] <= max_len)].copy()
    return out.reset_index(drop=True)

def deduplicate_by_sequence(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["label", "record_id"])
          .drop_duplicates(subset=["seq_sha1"], keep="first")
          .reset_index(drop=True)
    )

def write_fasta_from_df(df: pd.DataFrame, fasta_path: Path, id_col: str = "uid") -> None:
    with open(fasta_path, "w") as handle:
        for _, row in df.iterrows():
            handle.write(f">{row[id_col]}\n{row['sequence']}\n")

def spaced_sequence(seq: str) -> str:
    return " ".join(list(seq))

def aa_frequency_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    amino_acids = list(cfg.valid_aas)
    rows = []
    for grp, sub in df.groupby(group_col):
        total = sum(len(s) for s in sub["sequence"])
        counts = Counter("".join(sub["sequence"].tolist()))
        freqs = {aa: counts.get(aa, 0) / total if total else 0.0 for aa in amino_acids}
        freqs[group_col] = grp
        rows.append(freqs)
    return pd.DataFrame(rows)

def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}")

def capture_environment(out_path: Path) -> None:
    pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": cfg.device,
        "pip_freeze": pip_freeze.splitlines(),
    }
    save_json(env, out_path)

def text_has_exclusion_keyword(text: str, exclusion_keywords: tuple) -> bool:
    text = str(text).lower()
    return any(k.lower() in text for k in exclusion_keywords)

def write_fasta(df: pd.DataFrame, fasta_path: str, id_col: str = "record_id", seq_col: str = "sequence", desc_col: Optional[str] = "description") -> None:
    fasta_path = Path(fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as handle:
        for _, row in df.iterrows():
            rid = str(row[id_col])
            desc = str(row[desc_col]) if desc_col and desc_col in row and pd.notna(row[desc_col]) else rid
            handle.write(f">{rid} {desc}\n{row[seq_col]}\n")

def balanced_downsample_binary(df: pd.DataFrame, seed: int, target_multiplier: float = 1.0) -> pd.DataFrame:
    counts = df["label"].value_counts().to_dict()
    if set(counts.keys()) != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1, got counts: {counts}")
    pos_n = counts[1]
    neg_n = counts[0]
    target_neg = int(max(1, round(pos_n * target_multiplier)))
    if neg_n <= target_neg:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos_df = df[df["label"] == 1].copy()
    neg_df = df[df["label"] == 0].sample(n=target_neg, random_state=seed).copy()
    out = pd.concat([pos_df, neg_df], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def fetch_uniprot_nonamp_records(cfg: Config) -> pd.DataFrame:
    session = requests.Session()
    fields = ",".join([
        "accession",
        "id",
        "protein_name",
        "organism_name",
        "length",
        "cc_function",
        "sequence",
    ])
    params = {
        "query": f"reviewed:true AND length:[{cfg.min_len} TO {cfg.max_len}]",
        "format": "json",
        "fields": fields,
        "size": str(cfg.uniprot_page_size),
    }

    all_rows = []
    page = 0
    next_url = cfg.uniprot_base_url

    while next_url:
        page += 1
        print(f"Fetching UniProt page {page}: {next_url}")
        resp = session.get(next_url, params=params if page == 1 else None, timeout=cfg.uniprot_request_timeout)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        print(f"  records returned: {len(results)}")

        for item in results:
            accession = item.get("primaryAccession", "")
            entry_name = item.get("uniProtkbId", "")
            protein_desc = ""
            pd_obj = item.get("proteinDescription", {})
            if "recommendedName" in pd_obj:
                protein_desc = pd_obj["recommendedName"].get("fullName", {}).get("value", "")
            if not protein_desc and "submissionNames" in pd_obj and pd_obj["submissionNames"]:
                protein_desc = pd_obj["submissionNames"][0].get("fullName", {}).get("value", "")
            organism_name = item.get("organism", {}).get("scientificName", "")
            seq = item.get("sequence", {}).get("value", "")
            length = item.get("sequence", {}).get("length", len(seq))

            comments = item.get("comments", [])
            function_texts = []
            for c in comments:
                if c.get("commentType") == "FUNCTION":
                    texts = c.get("texts", [])
                    function_texts.extend([t.get("value", "") for t in texts])

            text_blob = " ".join([protein_desc, organism_name, " ".join(function_texts), accession, entry_name])
            if text_has_exclusion_keyword(text_blob, cfg.exclusion_keywords):
                continue

            seq = clean_sequence(seq)
            if not (cfg.min_len <= len(seq) <= cfg.max_len):
                continue
            if not is_valid_sequence(seq):
                continue

            all_rows.append({
                "record_id": accession,
                "entry_name": entry_name,
                "description": protein_desc,
                "organism_name": organism_name,
                "function_text": " ".join(function_texts),
                "sequence": seq,
                "length": len(seq),
                "label": 0,
                "source": "UniProt_SwissProt_query",
                "seq_sha1": sha1_text(seq),
            })

        link_header = resp.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        next_url = match.group(1) if match else None

    if not all_rows:
        raise ValueError("UniProt query returned zero usable non-AMP sequences after filtering.")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["seq_sha1"]).reset_index(drop=True)
    return df


# Figure helpers

FIG_DPI = 300
FIG_W = 7
FIG_H = 5

plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = DIRS["figures"] / f"{stem}.png"
    pdf_path = DIRS["figures"] / f"{stem}.pdf"
    svg_path = DIRS["figures"] / f"{stem}.svg"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures: {png_path.name}, {pdf_path.name}, {svg_path.name}")

def plot_bar_counts(df: pd.DataFrame, x_col: str, title: str, stem: str, source_table_name: str) -> None:
    counts = df[x_col].value_counts().reset_index()
    counts.columns = [x_col, "count"]
    save_csv(counts, DIRS["tables"] / source_table_name)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(counts[x_col].astype(str), counts["count"])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Count")
    save_figure(fig, stem)

def plot_length_distribution(df: pd.DataFrame, group_col: str, stem: str, source_table_name: str) -> None:
    save_csv(df[[group_col, "length"]].copy(), DIRS["tables"] / source_table_name)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    groups = list(df[group_col].dropna().unique())
    for grp in groups:
        vals = df.loc[df[group_col] == grp, "length"].values
        ax.hist(vals, bins=25, alpha=0.55, label=str(grp))
    ax.set_title("Sequence length distribution")
    ax.set_xlabel("Peptide length (aa)")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    save_figure(fig, stem)

def plot_aa_composition(freq_df: pd.DataFrame, group_col: str, stem: str, source_table_name: str) -> None:
    save_csv(freq_df, DIRS["tables"] / source_table_name)
    amino_acids = list(cfg.valid_aas)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(amino_acids))
    width = 0.8 / max(1, len(freq_df))
    for i, (_, row) in enumerate(freq_df.iterrows()):
        vals = [row[aa] for aa in amino_acids]
        ax.bar(x + i * width, vals, width=width, label=str(row[group_col]))
    ax.set_xticks(x + width * (len(freq_df) - 1) / 2)
    ax.set_xticklabels(amino_acids)
    ax.set_title("Amino acid composition by class")
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    save_figure(fig, stem)


# Download and prepare UniProt non-AMP pseudo-negatives

if cfg.download_nonamp_from_uniprot:
    nonamp_uniprot_df = fetch_uniprot_nonamp_records(cfg)
    nonamp_uniprot_df = nonamp_uniprot_df.sort_values(["record_id", "seq_sha1"]).reset_index(drop=True)

    save_tsv = Path(cfg.uniprot_metadata_tsv)
    save_tsv.parent.mkdir(parents=True, exist_ok=True)
    nonamp_uniprot_df.to_csv(save_tsv, sep="\t", index=False)

    write_fasta(
        nonamp_uniprot_df[["record_id", "description", "sequence"]].copy(),
        cfg.nonamp_raw_fasta,
        id_col="record_id",
        seq_col="sequence",
        desc_col="description",
    )
    write_fasta(
        nonamp_uniprot_df[["record_id", "description", "sequence"]].copy(),
        cfg.nonamp_fasta,
        id_col="record_id",
        seq_col="sequence",
        desc_col="description",
    )

    query_log = {
        "query_mode": "UniProt_REST_search_json",
        "base_url": cfg.uniprot_base_url,
        "query": f"reviewed:true AND length:[{cfg.min_len} TO {cfg.max_len}]",
        "page_size": cfg.uniprot_page_size,
        "exclusion_keywords": list(cfg.exclusion_keywords),
        "records_after_filtering_and_deduplication": int(len(nonamp_uniprot_df)),
        "output_fasta": cfg.nonamp_fasta,
        "output_metadata_tsv": cfg.uniprot_metadata_tsv,
    }
    with open(cfg.uniprot_query_log_json, "w") as f:
        json.dump(query_log, f, indent=2)

    print(nonamp_uniprot_df.head())
    print(f"Saved UniProt negative FASTA to: {cfg.nonamp_fasta}")
    print(f"Saved UniProt metadata TSV to: {cfg.uniprot_metadata_tsv}")
else:
    print("Skipping UniProt download because cfg.download_nonamp_from_uniprot=False")
    if not Path(cfg.nonamp_fasta).exists():
        raise FileNotFoundError(f"Configured non-AMP FASTA not found: {cfg.nonamp_fasta}")


# Load and preprocess training data

amp_df = fasta_to_df(cfg.amp_fasta, label=1, source_name="AMP")
nonamp_df = fasta_to_df(cfg.nonamp_fasta, label=0, source_name="NonAMP")

raw_counts = pd.DataFrame([
    {"dataset": "AMP_raw", "n": len(amp_df)},
    {"dataset": "NonAMP_raw", "n": len(nonamp_df)},
])
print(raw_counts)

amp_df = filter_short_peptides(amp_df, cfg.min_len, cfg.max_len)
nonamp_df = filter_short_peptides(nonamp_df, cfg.min_len, cfg.max_len)

amp_df = deduplicate_by_sequence(amp_df)
nonamp_df = deduplicate_by_sequence(nonamp_df)

pre_balance_counts = pd.DataFrame([
    {"dataset": "AMP_filtered_dedup", "n": len(amp_df)},
    {"dataset": "NonAMP_filtered_dedup_prebalance", "n": len(nonamp_df)},
])
print(pre_balance_counts)

combined_before_balance = pd.concat([amp_df, nonamp_df], ignore_index=True)
combined_before_balance = deduplicate_by_sequence(combined_before_balance)

amp_df = combined_before_balance[combined_before_balance["label"] == 1].copy().reset_index(drop=True)
nonamp_df = combined_before_balance[combined_before_balance["label"] == 0].copy().reset_index(drop=True)

balanced_df = pd.concat([amp_df, nonamp_df], ignore_index=True)
balanced_df = balanced_downsample_binary(balanced_df, seed=cfg.seed, target_multiplier=cfg.nonamp_target_multiplier)
full_df = balanced_df.copy().reset_index(drop=True)
full_df["uid"] = [f"seq_{i:07d}" for i in range(1, len(full_df) + 1)]

summary_df = pd.DataFrame([
    {"dataset": "AMP_final", "n": int((full_df['label'] == 1).sum())},
    {"dataset": "NonAMP_final", "n": int((full_df['label'] == 0).sum())},
    {"dataset": "Combined_final_balanced", "n": len(full_df)},
])
print(summary_df)

save_csv(raw_counts, DIRS["tables"] / "table_s1_raw_counts.csv")
save_csv(pre_balance_counts, DIRS["tables"] / "table_s2_prebalance_counts.csv")
save_csv(summary_df, DIRS["tables"] / "table_s3_final_counts.csv")
save_csv(full_df, DIRS["tables"] / "table_s4_training_sequences_precluster.csv")


# Exploratory figures before clustering

plot_bar_counts(
    full_df.assign(label_name=full_df["label"].map({1: "AMP", 0: "NonAMP"})),
    x_col="label_name",
    title="Class distribution after filtering and exact deduplication",
    stem="fig_01_class_distribution",
    source_table_name="fig_01_class_distribution_source.csv",
)

plot_length_distribution(
    full_df.assign(label_name=full_df["label"].map({1: "AMP", 0: "NonAMP"})),
    group_col="label_name",
    stem="fig_02_length_distribution",
    source_table_name="fig_02_length_distribution_source.csv",
)

aa_freq = aa_frequency_table(
    full_df.assign(label_name=full_df["label"].map({1: "AMP", 0: "NonAMP"})),
    group_col="label_name",
)
plot_aa_composition(
    aa_freq,
    group_col="label_name",
    stem="fig_03_aa_composition",
    source_table_name="fig_03_aa_composition_source.csv",
)


# Redundancy control with CD-HIT + custom handling for 10-aa peptides

full_df = full_df.copy()
full_df["seq_len"] = full_df["sequence"].astype(str).str.len()

full_df_long = full_df[full_df["seq_len"] > 10].copy().reset_index(drop=True)
full_df_len10 = full_df[full_df["seq_len"] == 10].copy().reset_index(drop=True)

print("Total sequences:", len(full_df))
print("Sequences >10 aa for CD-HIT:", len(full_df_long))
print("Sequences ==10 aa for custom clustering:", len(full_df_len10))

combined_fasta = DIRS["data"] / "combined_training_sequences_gt10.fasta"
write_fasta_from_df(full_df_long, combined_fasta, id_col="uid")

cluster_out = DIRS["data"] / "combined_training_sequences_gt10_cdhit.fasta"
cluster_log = DIRS["logs"] / "cdhit.log"

cdhit_cmd = (
    f"cd-hit -i {combined_fasta} "
    f"-o {cluster_out} "
    f"-c {cfg.cdhit_identity} "
    f"-n {cfg.cdhit_word_length} "
    f"-M {cfg.cdhit_memory_mb} "
    f"-T {cfg.cdhit_threads} "
    f"-d 0 > {cluster_log} 2>&1"
)

print(cdhit_cmd)
subprocess.run(cdhit_cmd, shell=True, check=True)
print("CD-HIT completed.")

cluster_clstr = Path(str(cluster_out) + ".clstr")
if not cluster_clstr.exists():
    raise FileNotFoundError(f"Missing CD-HIT cluster file: {cluster_clstr}")


# Parse CD-HIT clusters + custom clustering for 10-aa sequences

import itertools

def parse_cdhit_clstr(clstr_path: Path) -> pd.DataFrame:
    rows = []
    current_cluster = None

    with open(clstr_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">Cluster"):
                current_cluster = int(line.replace(">Cluster", "").strip())
                continue

            m = re.search(r">(.+?)\.\.\.", line)
            if m is None:
                continue

            seq_id = m.group(1).strip()
            is_rep = line.endswith("*")

            rows.append({
                "cluster_id": f"cdhit_cluster_{current_cluster}",
                "uid": seq_id,
                "is_representative": bool(is_rep),
            })

    cluster_df = pd.DataFrame(rows)

    if cluster_df.empty and len(full_df_long) > 0:
        raise ValueError(f"No clusters parsed from {clstr_path}")

    if not cluster_df.empty:
        cluster_df["uid"] = cluster_df["uid"].astype(str).str.strip()

    return cluster_df


def seq_identity_equal_length(seq1: str, seq2: str) -> float:
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be equal length for this identity function.")
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


def cluster_len10_sequences(df_len10: pd.DataFrame, identity_threshold: float = 0.8) -> pd.DataFrame:
    """
    Cluster 10-aa peptides by single-linkage using pairwise identity >= threshold.
    Returns columns: cluster_id, uid, is_representative
    """
    if df_len10.empty:
        return pd.DataFrame(columns=["cluster_id", "uid", "is_representative"])

    uids = df_len10["uid"].tolist()
    seqs = dict(zip(df_len10["uid"], df_len10["sequence"]))

    parent = {uid: uid for uid in uids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(uids)):
        uid_i = uids[i]
        seq_i = seqs[uid_i]
        for j in range(i + 1, len(uids)):
            uid_j = uids[j]
            seq_j = seqs[uid_j]
            if seq_identity_equal_length(seq_i, seq_j) >= identity_threshold:
                union(uid_i, uid_j)

    groups = {}
    for uid in uids:
        root = find(uid)
        groups.setdefault(root, []).append(uid)

    rows = []
    for idx, members in enumerate(groups.values()):
        members = sorted(members)
        rep_uid = members[0]
        cluster_id = f"len10_cluster_{idx}"
        for uid in members:
            rows.append({
                "cluster_id": cluster_id,
                "uid": uid,
                "is_representative": uid == rep_uid
            })

    return pd.DataFrame(rows)


cluster_map_long = parse_cdhit_clstr(cluster_clstr) if len(full_df_long) > 0 else pd.DataFrame(
    columns=["cluster_id", "uid", "is_representative"]
)

cluster_map_len10 = cluster_len10_sequences(
    full_df_len10,
    identity_threshold=cfg.cdhit_identity
)

cluster_map = pd.concat([cluster_map_long, cluster_map_len10], ignore_index=True)

cluster_map["uid"] = cluster_map["uid"].astype(str).str.strip()
full_df["uid"] = full_df["uid"].astype(str).str.strip()

full_ids = set(full_df["uid"])
cluster_ids = set(cluster_map["uid"])

missing_in_clusters = sorted(full_ids - cluster_ids)
extra_in_clusters = sorted(cluster_ids - full_ids)

print(f"N training sequences in full_df: {len(full_ids)}")
print(f"N assigned sequence IDs from combined clustering: {len(cluster_ids)}")
print(f"Missing IDs after clustering: {len(missing_in_clusters)}")
print(f"Extra IDs in cluster map: {len(extra_in_clusters)}")

if missing_in_clusters:
    print("First 10 missing IDs:", missing_in_clusters[:10])
    raise ValueError("Some sequences were still not assigned a cluster.")

stale_cols = [
    "cluster_id", "is_representative",
    "cluster_id_x", "cluster_id_y",
    "is_representative_x", "is_representative_y"
]
drop_cols = [c for c in stale_cols if c in full_df.columns]
if drop_cols:
    full_df = full_df.drop(columns=drop_cols)

full_df = full_df.merge(cluster_map, on="uid", how="left", validate="one_to_one")

if full_df["cluster_id"].isna().any():
    bad = full_df.loc[full_df["cluster_id"].isna(), "uid"].tolist()[:10]
    raise ValueError(f"Some sequences still have no cluster after merge. Examples: {bad}")

cluster_stats = (
    full_df.groupby("cluster_id")
    .agg(
        cluster_size=("uid", "count"),
        n_amp=("label", lambda x: int((x == 1).sum())),
        n_nonamp=("label", lambda x: int((x == 0).sum())),
    )
    .reset_index()
)

cluster_stats["mixed_label_cluster"] = (
    (cluster_stats["n_amp"] > 0) & (cluster_stats["n_nonamp"] > 0)
)

save_csv(full_df, DIRS["tables"] / "table_s4_training_sequences_with_clusters.csv")
save_csv(cluster_stats, DIRS["tables"] / "table_s5_cluster_stats.csv")

print(cluster_stats.head())


# Remove mixed-label clusters to avoid ambiguous supervision

mixed_clusters = cluster_stats.loc[cluster_stats["mixed_label_cluster"], "cluster_id"].tolist()
mixed_cluster_df = cluster_stats[cluster_stats["mixed_label_cluster"]].copy()
save_csv(mixed_cluster_df, DIRS["tables"] / "table_s6_mixed_label_clusters_removed.csv")

if mixed_clusters:
    print(f"Removing {len(mixed_clusters)} mixed-label clusters.")
    filtered_df = full_df[~full_df["cluster_id"].isin(mixed_clusters)].copy()
else:
    print("No mixed-label clusters detected.")
    filtered_df = full_df.copy()

filtered_df = filtered_df.reset_index(drop=True)
filtered_cluster_stats = (
    filtered_df.groupby("cluster_id")
               .agg(
                   cluster_size=("uid", "count"),
                   label=("label", "first"),
               )
               .reset_index()
)

safety = filtered_df.groupby("cluster_id")["label"].nunique().reset_index(name="n_unique_labels")
if (safety["n_unique_labels"] > 1).any():
    raise ValueError("Mixed-label clusters still present after filtering.")

# Cluster-aware stratified train/val/test split

cluster_level = filtered_cluster_stats.copy()

train_clusters, temp_clusters = train_test_split(
    cluster_level,
    test_size=(cfg.test_size + cfg.val_size),
    random_state=cfg.split_seed,
    stratify=cluster_level["label"],
)

temp_rel_test = cfg.test_size / (cfg.test_size + cfg.val_size)

val_clusters, test_clusters = train_test_split(
    temp_clusters,
    test_size=temp_rel_test,
    random_state=cfg.split_seed,
    stratify=temp_clusters["label"],
)

split_map = {}
for cid in train_clusters["cluster_id"]:
    split_map[cid] = "train"
for cid in val_clusters["cluster_id"]:
    split_map[cid] = "val"
for cid in test_clusters["cluster_id"]:
    split_map[cid] = "test"

filtered_df["split"] = filtered_df["cluster_id"].map(split_map)

if filtered_df["split"].isna().any():
    raise ValueError("Some sequences did not receive a split label.")

overlap = {
    "train_val": set(train_clusters["cluster_id"]) & set(val_clusters["cluster_id"]),
    "train_test": set(train_clusters["cluster_id"]) & set(test_clusters["cluster_id"]),
    "val_test": set(val_clusters["cluster_id"]) & set(test_clusters["cluster_id"]),
}
for key, value in overlap.items():
    if value:
        raise ValueError(f"Cluster overlap detected in {key}: {list(value)[:5]}")

split_summary = (
    filtered_df.groupby(["split", "label"])
               .size()
               .reset_index(name="n_sequences")
)
cluster_split_summary = (
    cluster_level.assign(split=cluster_level["cluster_id"].map(split_map))
                 .groupby(["split", "label"])
                 .size()
                 .reset_index(name="n_clusters")
)

save_csv(filtered_df, DIRS["splits"] / "training_sequences_with_splits.csv")
save_csv(split_summary, DIRS["splits"] / "split_sequence_summary.csv")
save_csv(cluster_split_summary, DIRS["splits"] / "split_cluster_summary.csv")
save_json(split_map, DIRS["splits"] / "cluster_to_split_map.json")

print(split_summary)
print(cluster_split_summary)


# Tokenization and dataset

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, do_lower_case=False)

class PeptideDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = spaced_sequence(row["sequence"])
        enc = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            "uid": row["uid"],
        }

train_df = filtered_df[filtered_df["split"] == "train"].copy()
val_df = filtered_df[filtered_df["split"] == "val"].copy()
test_df = filtered_df[filtered_df["split"] == "test"].copy()

train_loader = DataLoader(
    PeptideDataset(train_df, tokenizer, cfg.max_length_tokens),
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    PeptideDataset(val_df, tokenizer, cfg.max_length_tokens),
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    PeptideDataset(test_df, tokenizer, cfg.max_length_tokens),
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# Lightweight ProtBert classifier

class ProtBertClassifier(nn.Module):
    def __init__(self, model_name: str, classifier_hidden_dim: int, dropout: float, trainable_encoder_layers: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        for p in self.encoder.parameters():
            p.requires_grad = False

        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
            n_layers = len(layers)
            for layer in layers[max(0, n_layers - trainable_encoder_layers):]:
                for p in layer.parameters():
                    p.requires_grad = True

        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            for p in self.encoder.pooler.parameters():
                p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits, cls_embedding

model = ProtBertClassifier(
    model_name=cfg.model_name,
    classifier_hidden_dim=cfg.classifier_hidden_dim,
    dropout=cfg.dropout,
    trainable_encoder_layers=cfg.trainable_encoder_layers,
).to(cfg.device)

n_total = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {n_total:,}")
print(f"Trainable params: {n_trainable:,}")


# Loss, optimizer, scheduler

criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
)

num_training_steps = len(train_loader) * cfg.epochs
num_warmup_steps = int(cfg.warmup_ratio * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and torch.cuda.is_available()))


# Metrics and evaluation helpers

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan

    return metrics

@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    losses, all_true, all_prob, all_pred, all_uid, all_embeddings = [], [], [], [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, cls_embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        losses.append(loss.item())
        all_true.extend(labels.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())
        all_pred.extend((probs >= 0.5).long().cpu().numpy().tolist())
        all_uid.extend(batch["uid"])
        all_embeddings.append(cls_embedding.cpu().numpy())

    metrics = compute_metrics(all_true, all_prob, threshold=0.5)
    metrics["loss"] = float(np.mean(losses)) if losses else np.nan

    out_df = pd.DataFrame({
        "uid": all_uid,
        "y_true": all_true,
        "y_prob": all_prob,
        "y_pred": all_pred,
    })

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 1))
    return metrics, out_df, embeddings


# Training loop with checkpointing and early stopping

best_metric = -np.inf
best_epoch = -1
patience_counter = 0

history = []
best_ckpt_path = DIRS["models"] / "best_model.pt"
last_ckpt_path = DIRS["models"] / "last_model.pt"

start_time = time.time()

for epoch in range(1, cfg.epochs + 1):
    model.train()
    running_losses = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.use_amp and torch.cuda.is_available())):
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_losses.append(loss.item())
        pbar.set_postfix({"train_loss": f"{np.mean(running_losses):.4f}"})

    train_metrics, _, _ = evaluate_model(model, train_loader, criterion, cfg.device)
    val_metrics, _, _ = evaluate_model(model, val_loader, criterion, cfg.device)

    epoch_record = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "train_mcc": train_metrics["mcc"],
        "train_auroc": train_metrics["auroc"],
        "train_auprc": train_metrics["auprc"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_mcc": val_metrics["mcc"],
        "val_auroc": val_metrics["auroc"],
        "val_auprc": val_metrics["auprc"],
    }
    history.append(epoch_record)

    current_metric = epoch_record[cfg.monitor_metric]
    print(epoch_record)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(cfg),
        "history": history,
    }, last_ckpt_path)

    if current_metric > best_metric:
        best_metric = current_metric
        best_epoch = epoch
        patience_counter = 0
        # best_model.pt saves weights only (no optimizer/scheduler state) to keep
        # the file compact (~420 MB vs ~1.68 GB). Optimizer state is in last_model.pt.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "history": history,
        }, best_ckpt_path)
        print(f"Saved new best checkpoint at epoch {epoch} with {cfg.monitor_metric}={current_metric:.5f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{cfg.early_stopping_patience}")

    if patience_counter >= cfg.early_stopping_patience:
        print("Early stopping triggered.")
        break

elapsed = time.time() - start_time
print(f"Training finished in {elapsed/60:.2f} minutes. Best epoch: {best_epoch}")


# Load best model and evaluate

history_df = pd.DataFrame(history)
save_csv(history_df, DIRS["logs"] / "training_history.csv")

checkpoint = torch.load(best_ckpt_path, map_location=cfg.device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_metrics, val_pred_df, val_embeddings = evaluate_model(model, val_loader, criterion, cfg.device)
test_metrics, test_pred_df, test_embeddings = evaluate_model(model, test_loader, criterion, cfg.device)

val_out = val_df.merge(val_pred_df, on="uid", how="left")
test_out = test_df.merge(test_pred_df, on="uid", how="left")

metrics_df = pd.DataFrame([
    {"split": "val", **val_metrics},
    {"split": "test", **test_metrics},
])

print(metrics_df)

save_csv(metrics_df, DIRS["tables"] / "table_s7_final_metrics.csv")
save_csv(val_out, DIRS["tables"] / "table_s8_validation_predictions.csv")
save_csv(test_out, DIRS["tables"] / "table_s9_test_predictions.csv")


# Core evaluation figures

def plot_training_curves(history_df: pd.DataFrame):
    save_csv(history_df, DIRS["tables"] / "fig_04_training_curves_source.csv")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Val loss")
    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    save_figure(fig, "fig_04_loss_curves")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(history_df["epoch"], history_df["train_auprc"], label="Train AUPRC")
    ax.plot(history_df["epoch"], history_df["val_auprc"], label="Val AUPRC")
    ax.set_title("Training and validation AUPRC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUPRC")
    ax.legend(frameon=False)
    save_figure(fig, "fig_05_auprc_curves")

def plot_roc_pr(y_true, y_prob, stem_prefix: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    save_csv(pd.DataFrame({"fpr": fpr, "tpr": tpr}), DIRS["tables"] / f"{stem_prefix}_roc_source.csv")
    ax.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False)
    save_figure(fig, f"{stem_prefix}_roc")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    save_csv(pd.DataFrame({"precision": precision, "recall": recall}), DIRS["tables"] / f"{stem_prefix}_pr_source.csv")
    ax.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    ax.set_title("Precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    save_figure(fig, f"{stem_prefix}_pr")

def plot_confusion(y_true, y_pred, stem: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["NonAMP", "AMP"], columns=["Pred_NonAMP", "Pred_AMP"])
    save_csv(cm_df.reset_index(names="true_label"), DIRS["tables"] / f"{stem}_source.csv")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["NonAMP", "AMP"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["NonAMP", "AMP"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, stem)

def plot_calibration(y_true, y_prob, stem: str):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    save_csv(pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos}), DIRS["tables"] / f"{stem}_source.csv")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_title("Calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.legend(frameon=False)
    save_figure(fig, stem)

def plot_score_distribution(pred_df: pd.DataFrame, stem: str):
    save_csv(pred_df[["y_true", "y_prob"]], DIRS["tables"] / f"{stem}_source.csv")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for lbl, name in [(0, "NonAMP"), (1, "AMP")]:
        vals = pred_df.loc[pred_df["y_true"] == lbl, "y_prob"].values
        ax.hist(vals, bins=25, alpha=0.55, label=name)
    ax.set_title("Predicted score distribution")
    ax.set_xlabel("Predicted AMP probability")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    save_figure(fig, stem)

plot_training_curves(history_df)
plot_roc_pr(test_out["y_true"].values, test_out["y_prob"].values, "fig_06_test")
plot_confusion(test_out["y_true"].values, test_out["y_pred"].values, "fig_07_test_confusion_matrix")
plot_calibration(test_out["y_true"].values, test_out["y_prob"].values, "fig_08_test_calibration")
plot_score_distribution(test_out, "fig_09_test_score_distribution")


# Embedding visualization on the test set

emb_df = test_out[["uid", "label", "sequence", "length"]].copy()
emb_df["label_name"] = emb_df["label"].map({1: "AMP", 0: "NonAMP"})

if len(test_embeddings) >= 6:
    n_pca = min(50, test_embeddings.shape[1], max(2, len(test_embeddings) - 1))
    emb_pca = PCA(n_components=n_pca, random_state=cfg.seed).fit_transform(test_embeddings)

    perplexity = max(5, min(30, len(emb_pca) // 5))
    emb_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=cfg.seed,
    ).fit_transform(emb_pca)

    emb_plot_df = emb_df.copy()
    emb_plot_df["x"] = emb_2d[:, 0]
    emb_plot_df["y"] = emb_2d[:, 1]
    save_csv(emb_plot_df, DIRS["tables"] / "fig_10_test_embedding_tsne_source.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for lbl, name in [(0, "NonAMP"), (1, "AMP")]:
        sub = emb_plot_df[emb_plot_df["label"] == lbl]
        ax.scatter(sub["x"], sub["y"], s=25, alpha=0.7, label=name)
    ax.set_title("t-SNE of ProtBert test-set embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False)
    save_figure(fig, "fig_10_test_embedding_tsne")
else:
    print("Too few test samples for stable t-SNE; skipping embedding figure.")

# Build external benchmark set: 250 AMP vs 250 non-AMP

import os
import re
import json
import random
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

EXT_CFG = {
    # Update this path to point to your DBAASP MIC dataset file on Kaggle.
    # If your dataset slug is "my-amp-data" and the file is in the root,
    # the path would be: /kaggle/input/my-amp-data/Dataset_AMPs_MICs.csv
    "dbaasp_csv": "/kaggle/input/<your-dataset-slug>/Dataset_AMPs_MICs.csv",
    "out_dir": str(DIRS["data"] / "external_benchmark"),
    "min_len": 10,
    "max_len": 50,
    "n_amp": 250,
    "n_nonamp": 250,
    "uniprot_batch_size": 500,
    "uniprot_max_records_to_collect": 5000,
}

ext_out = Path(EXT_CFG["out_dir"])
ext_out.mkdir(parents=True, exist_ok=True)

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

def fasta_to_df_simple(fasta_path: str, label: int, source_name: str) -> pd.DataFrame:
    rows = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).strip().upper()
        rows.append({
            "uid": str(rec.id),
            "sequence": seq,
            "label": label,
            "source": source_name
        })
    return pd.DataFrame(rows)

def write_fasta_from_dataframe(df: pd.DataFrame, out_fasta: str, id_col="uid", seq_col="sequence"):
    records = []
    for _, row in df.iterrows():
        records.append(SeqRecord(Seq(row[seq_col]), id=str(row[id_col]), description=""))
    with open(out_fasta, "w") as handle:
        SeqIO.write(records, handle, "fasta")
    print(f"Saved FASTA: {out_fasta}")

def clean_sequence_column(df: pd.DataFrame, seq_col="sequence") -> pd.DataFrame:
    df = df.copy()
    df[seq_col] = (
        df[seq_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    return df

def filter_standard_short_peptides(df: pd.DataFrame, seq_col="sequence", min_len=10, max_len=50) -> pd.DataFrame:
    df = df.copy()
    df["length"] = df[seq_col].str.len()
    df = df[df["length"].between(min_len, max_len)].copy()
    df = df[df[seq_col].apply(lambda s: bool(AA_RE.match(s)))].copy()
    return df.reset_index(drop=True)

def deduplicate_by_sequence_only(df: pd.DataFrame, seq_col="sequence") -> pd.DataFrame:
    return df.drop_duplicates(subset=[seq_col]).reset_index(drop=True)

def remove_sequence_overlap(df: pd.DataFrame, forbidden_sequences: set, seq_col="sequence") -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df[~df[seq_col].isin(forbidden_sequences)].copy().reset_index(drop=True)
    print(f"Removed {before - len(df)} overlapping sequences.")
    return df

def sample_n_or_fail(df: pd.DataFrame, n: int, random_state=42) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Not enough sequences after filtering. Needed {n}, found {len(df)}.")
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)

def parse_uniprot_fasta(text: str):
    records = []
    handle = StringIO(text)
    for rec in SeqIO.parse(handle, "fasta"):
        records.append({
            "uid": rec.id,
            "sequence": str(rec.seq).strip().upper(),
            "source": "UniProt_external_nonAMP"
        })
    return pd.DataFrame(records)

train_amp_df = amp_df.copy()
train_nonamp_df = nonamp_df.copy()

train_amp_df = clean_sequence_column(train_amp_df, "sequence")
train_nonamp_df = clean_sequence_column(train_nonamp_df, "sequence")

train_amp_df = filter_standard_short_peptides(
    train_amp_df, seq_col="sequence",
    min_len=EXT_CFG["min_len"], max_len=EXT_CFG["max_len"]
)
train_nonamp_df = filter_standard_short_peptides(
    train_nonamp_df, seq_col="sequence",
    min_len=EXT_CFG["min_len"], max_len=EXT_CFG["max_len"]
)

train_amp_seqs = set(train_amp_df["sequence"])
train_nonamp_seqs = set(train_nonamp_df["sequence"])
all_train_seqs = train_amp_seqs | train_nonamp_seqs

print("Training AMP sequences in leakage screen:", len(train_amp_seqs))
print("Training non-AMP sequences in leakage screen:", len(train_nonamp_seqs))

# Build external AMP set from DBAASP CSV
dbaasp_raw = pd.read_csv(EXT_CFG["dbaasp_csv"])
print("Raw DBAASP rows:", len(dbaasp_raw))
print("Columns:", list(dbaasp_raw.columns))

seq_candidates = ["SEQUENCE", "Sequence", "sequence"]
activity_candidates = ["ACTIVITY", "Activity", "activity"]

seq_col = next((c for c in seq_candidates if c in dbaasp_raw.columns), None)
activity_col = next((c for c in activity_candidates if c in dbaasp_raw.columns), None)

if seq_col is None:
    raise ValueError("Could not find a sequence column in DBAASP CSV.")
if activity_col is None:
    raise ValueError("Could not find an activity column in DBAASP CSV.")

ext_amp = dbaasp_raw.copy()
ext_amp = ext_amp.rename(columns={seq_col: "sequence", activity_col: "activity"})
ext_amp = clean_sequence_column(ext_amp, "sequence")

ext_amp["activity"] = ext_amp["activity"].astype(str).str.strip()
ext_amp = ext_amp[ext_amp["activity"].notna()].copy()
ext_amp = ext_amp[ext_amp["activity"] != ""].copy()
ext_amp = ext_amp[ext_amp["activity"].str.lower() != "nan"].copy()

ext_amp = filter_standard_short_peptides(
    ext_amp, seq_col="sequence",
    min_len=EXT_CFG["min_len"], max_len=EXT_CFG["max_len"]
)

ext_amp = deduplicate_by_sequence_only(ext_amp, seq_col="sequence")

print("\nCleaning external AMP set...")
ext_amp = remove_sequence_overlap(ext_amp, all_train_seqs, seq_col="sequence")

ext_amp = ext_amp.reset_index(drop=True)
ext_amp["uid"] = [f"ext_amp_{i:04d}" for i in range(1, len(ext_amp) + 1)]
ext_amp["label"] = 1
ext_amp["source"] = "DBAASP_external_AMP"

print("External AMP candidates after cleaning:", len(ext_amp))

ext_amp_final = sample_n_or_fail(ext_amp, EXT_CFG["n_amp"], random_state=RANDOM_STATE)

# Download external non-AMPs from UniProt
print("\nDownloading UniProt non-AMP candidates...")

uniprot_query = (
    '(reviewed:true) AND '
    '(length:[10 TO 50]) AND '
    'NOT (keyword:"Antimicrobial" OR keyword:"Antibiotic" OR keyword:"Host defense" '
    'OR protein_name:defensin OR protein_name:bacteriocin OR protein_name:antimicrobial '
    'OR protein_name:antibacterial OR protein_name:antifungal OR protein_name:antiviral '
    'OR protein_name:toxin OR protein_name:venom)'
)

base_url = "https://rest.uniprot.org/uniprotkb/search"

def fetch_uniprot_fasta_batches(query, batch_size=500, max_records=5000):
    collected = []
    cursor = None
    n_total = 0

    while True:
        params = {
            "query": query,
            "format": "fasta",
            "size": batch_size,
        }
        if cursor:
            params["cursor"] = cursor

        r = requests.get(base_url, params=params, timeout=120)
        r.raise_for_status()

        text = r.text.strip()
        if not text:
            break

        batch_df = parse_uniprot_fasta(text)
        if len(batch_df) == 0:
            break

        collected.append(batch_df)
        n_total += len(batch_df)
        print(f"Collected {n_total} UniProt candidates so far...")

        link_header = r.headers.get("Link", "")
        next_cursor = None
        if 'rel="next"' in link_header:
            m = re.search(r'[?&]cursor=([^&>]+)', link_header)
            if m:
                next_cursor = m.group(1)

        if not next_cursor or n_total >= max_records:
            break

        cursor = next_cursor

    if len(collected) == 0:
        return pd.DataFrame(columns=["uid", "sequence", "source"])

    return pd.concat(collected, ignore_index=True)

ext_nonamp = fetch_uniprot_fasta_batches(
    uniprot_query,
    batch_size=EXT_CFG["uniprot_batch_size"],
    max_records=EXT_CFG["uniprot_max_records_to_collect"]
)

print("Raw UniProt non-AMP candidates:", len(ext_nonamp))

ext_nonamp = clean_sequence_column(ext_nonamp, "sequence")
ext_nonamp = filter_standard_short_peptides(
    ext_nonamp, seq_col="sequence",
    min_len=EXT_CFG["min_len"], max_len=EXT_CFG["max_len"]
)
ext_nonamp = deduplicate_by_sequence_only(ext_nonamp, seq_col="sequence")

print("\nCleaning external non-AMP set against training data...")
ext_nonamp = remove_sequence_overlap(ext_nonamp, all_train_seqs, seq_col="sequence")

print("Removing overlap with external AMP sequences...")
ext_nonamp = remove_sequence_overlap(ext_nonamp, set(ext_amp_final["sequence"]), seq_col="sequence")

ext_nonamp = ext_nonamp.reset_index(drop=True)
ext_nonamp["uid"] = [f"ext_nonamp_{i:04d}" for i in range(1, len(ext_nonamp) + 1)]
ext_nonamp["label"] = 0
ext_nonamp["source"] = "UniProt_external_nonAMP"

print("External non-AMP candidates after cleaning:", len(ext_nonamp))

ext_nonamp_final = sample_n_or_fail(ext_nonamp, EXT_CFG["n_nonamp"], random_state=RANDOM_STATE)

# Final overlap safety checks
amp_seqs_final = set(ext_amp_final["sequence"])
nonamp_seqs_final = set(ext_nonamp_final["sequence"])

assert len(amp_seqs_final & train_amp_seqs) == 0
assert len(amp_seqs_final & train_nonamp_seqs) == 0
assert len(nonamp_seqs_final & train_amp_seqs) == 0
assert len(nonamp_seqs_final & train_nonamp_seqs) == 0
assert len(amp_seqs_final & nonamp_seqs_final) == 0

print("\nFinal leakage checks passed.")

external_df = pd.concat([ext_amp_final, ext_nonamp_final], ignore_index=True)
external_df = external_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

external_csv = ext_out / "external_dataset_250v250.csv"
external_amp_csv = ext_out / "external_AMPs_250.csv"
external_nonamp_csv = ext_out / "external_nonAMPs_250.csv"

external_amp_fasta = ext_out / "external_AMPs_250.fasta"
external_nonamp_fasta = ext_out / "external_nonAMPs_250.fasta"

external_df.to_csv(external_csv, index=False)
ext_amp_final.to_csv(external_amp_csv, index=False)
ext_nonamp_final.to_csv(external_nonamp_csv, index=False)

write_fasta_from_dataframe(ext_amp_final[["uid", "sequence"]], str(external_amp_fasta))
write_fasta_from_dataframe(ext_nonamp_final[["uid", "sequence"]], str(external_nonamp_fasta))

manifest = {
    "n_external_amp_final": len(ext_amp_final),
    "n_external_nonamp_final": len(ext_nonamp_final),
    "external_csv": str(external_csv),
    "external_amp_fasta": str(external_amp_fasta),
    "external_nonamp_fasta": str(external_nonamp_fasta),
    "uniprot_query": uniprot_query,
    "random_state": RANDOM_STATE,
}
with open(ext_out / "external_benchmark_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\nSaved external benchmark set.")
print(manifest)


# External benchmark evaluation

external_dir = DIRS["data"] / "external_benchmark"
external_amp_fasta = external_dir / "external_AMPs_250.fasta"
external_nonamp_fasta = external_dir / "external_nonAMPs_250.fasta"

if not external_amp_fasta.exists():
    raise FileNotFoundError(f"Missing external AMP FASTA: {external_amp_fasta}")
if not external_nonamp_fasta.exists():
    raise FileNotFoundError(f"Missing external non-AMP FASTA: {external_nonamp_fasta}")

def build_external_df(amp_fasta: str, nonamp_fasta: str) -> pd.DataFrame:
    amp_ext = fasta_to_df(amp_fasta, label=1, source_name="External_AMP")
    nonamp_ext = fasta_to_df(nonamp_fasta, label=0, source_name="External_NonAMP")

    ext = pd.concat([amp_ext, nonamp_ext], ignore_index=True)
    ext = filter_short_peptides(ext, cfg.min_len, cfg.max_len)
    ext = deduplicate_by_sequence(ext)

    ext = ext.copy().reset_index(drop=True)
    ext["uid"] = [f"external_{i:06d}" for i in range(1, len(ext) + 1)]

    return ext

external_df = build_external_df(str(external_amp_fasta), str(external_nonamp_fasta))

print("External dataset size:", len(external_df))
print(external_df["label"].value_counts())

save_csv(external_df, DIRS["tables"] / "table_s10_external_dataset.csv")
print(external_df.head())

def predict_df(model, df: pd.DataFrame, tokenizer, device: str):
    ds = PeptideDataset(df, tokenizer, cfg.max_length_tokens)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    metrics, pred_df, embeddings = evaluate_model(model, loader, criterion, device)
    merged = df.merge(pred_df, on="uid", how="left")
    return merged, metrics, embeddings

external_out, external_metrics, external_embeddings = predict_df(
    model=model,
    df=external_df,
    tokenizer=tokenizer,
    device=cfg.device
)

print("External benchmark metrics")
for k, v in external_metrics.items():
    print(f"{k}: {v}")

save_csv(external_out, DIRS["tables"] / "table_s11_external_benchmark_predictions.csv")
save_csv(pd.DataFrame([external_metrics]), DIRS["tables"] / "table_s12_external_benchmark_metrics.csv")

# External benchmark figures
def plot_external_roc_pr(y_true, y_prob, prefix="fig_11_external"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(fpr, tpr, label=f"AUROC = {roc_auc_score(y_true, y_prob):.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("External benchmark ROC curve")
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, f"{prefix}_roc")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(rec, prec, label=f"AUPRC = {average_precision_score(y_true, y_prob):.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("External benchmark Precision-Recall curve")
    ax.legend(loc="lower left", frameon=False)
    save_figure(fig, f"{prefix}_pr")

def plot_external_confusion(y_true, y_pred, stem="fig_12_external_confusion_matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("External benchmark confusion matrix")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-AMP", "AMP"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-AMP", "AMP"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, stem)

def plot_external_score_distribution(df, stem="fig_13_external_score_distribution"):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(df.loc[df["y_true"] == 1, "y_prob"], bins=25, alpha=0.7, label="AMP")
    ax.hist(df.loc[df["y_true"] == 0, "y_prob"], bins=25, alpha=0.7, label="Non-AMP")
    ax.set_xlabel("Predicted AMP probability")
    ax.set_ylabel("Count")
    ax.set_title("External benchmark score distribution")
    ax.legend(frameon=False)
    save_figure(fig, stem)

plot_external_roc_pr(
    external_out["y_true"].values,
    external_out["y_prob"].values,
    prefix="fig_11_external"
)

plot_external_confusion(
    external_out["y_true"].values,
    external_out["y_pred"].values,
    stem="fig_12_external_confusion_matrix"
)

plot_external_score_distribution(
    external_out,
    stem="fig_13_external_score_distribution"
)

# Supplementary physicochemical table for test predictions

def safe_physchem(seq: str) -> dict:
    pa = ProteinAnalysis(seq)
    try:
        gravy = pa.gravy()
    except Exception:
        gravy = np.nan
    try:
        pI = pa.isoelectric_point()
    except Exception:
        pI = np.nan
    try:
        mw = pa.molecular_weight()
    except Exception:
        mw = np.nan
    aa_pct = pa.amino_acids_percent
    charge_pH74 = ProteinAnalysis(seq).charge_at_pH(7.4)
    return {
        "gravy": gravy,
        "pI": pI,
        "molecular_weight": mw,
        "charge_pH74": charge_pH74,
        "frac_hydrophobic": float(sum(aa_pct.get(x, 0.0) for x in ["A", "V", "I", "L", "M", "F", "W", "Y"])),
    }

phys_rows = []
for _, row in test_out.iterrows():
    feat = safe_physchem(row["sequence"])
    feat.update({
        "uid": row["uid"],
        "label": row["label"],
        "y_prob": row["y_prob"],
        "length": row["length"],
        "sequence": row["sequence"],
    })
    phys_rows.append(feat)

phys_df = pd.DataFrame(phys_rows)
save_csv(phys_df, DIRS["tables"] / "table_s12_test_physicochemical_properties.csv")
print(phys_df.head())


# Save model assets and reproducibility manifest

tokenizer.save_pretrained(DIRS["models"] / "tokenizer")
save_json(asdict(cfg), DIRS["configs"] / "config.json")

run_summary = {
    "best_epoch": int(checkpoint["epoch"]),
    "best_monitor_metric": float(best_metric),
    "monitor_metric_name": cfg.monitor_metric,
    "n_train": int(len(train_df)),
    "n_val": int(len(val_df)),
    "n_test": int(len(test_df)),
    "n_total_after_filters": int(len(filtered_df)),
    "n_clusters_after_filters": int(filtered_df["cluster_id"].nunique()),
    "n_mixed_label_clusters_removed": int(len(mixed_clusters)),
    "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    "total_params": int(sum(p.numel() for p in model.parameters())),
}
save_json(run_summary, DIRS["logs"] / "run_summary.json")
capture_environment(DIRS["logs"] / "environment.json")

print("Notebook 1 outputs ready under:", OUT)
