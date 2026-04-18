"""
export_kaggle_outputs.py
------------------------
Run this at the end of a Kaggle session to zip all outputs into a single
archive that can be downloaded from the Kaggle output panel.

Usage (inside a Kaggle notebook cell):
    %run scripts/export_kaggle_outputs.py
"""

from pathlib import Path
import shutil
import json
import os
import pandas as pd

EXPORT_DIR = Path("/kaggle/working/final_export_amp_project")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

SUBDIRS = {
    "figures": EXPORT_DIR / "figures",
    "tables": EXPORT_DIR / "tables",
    "logs": EXPORT_DIR / "logs",
    "models": EXPORT_DIR / "models",
    "data": EXPORT_DIR / "data",
    "manifests": EXPORT_DIR / "manifests",
}

for p in SUBDIRS.values():
    p.mkdir(parents=True, exist_ok=True)

print("Export directory:", EXPORT_DIR)


def safe_copy(src, dst_dir):
    src = Path(src)
    dst_dir = Path(dst_dir)
    if src.exists():
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        print(f"Copied: {src.name}")
    else:
        print(f"Missing, skipped: {src}")


# Training outputs root
NB1_ROOT = Path("/kaggle/working/amp_protbert_training")

# Figures
for f in (NB1_ROOT / "figures").glob("*"):
    if f.is_file():
        safe_copy(f, SUBDIRS["figures"])

# Tables
for f in (NB1_ROOT / "tables").glob("*"):
    if f.is_file():
        safe_copy(f, SUBDIRS["tables"])

# Logs
for f in (NB1_ROOT / "logs").glob("*"):
    if f.is_file():
        safe_copy(f, SUBDIRS["logs"])

# External benchmark data
external_candidates = [
    NB1_ROOT / "data" / "external_benchmark" / "external_AMPs_250.fasta",
    NB1_ROOT / "data" / "external_benchmark" / "external_nonAMPs_250.fasta",
    NB1_ROOT / "data" / "external_benchmark" / "external_dataset_250v250.csv",
    NB1_ROOT / "data" / "external_benchmark" / "external_benchmark_manifest.json",
]
for f in external_candidates:
    safe_copy(f, SUBDIRS["data"])

# Model/checkpoint files
search_roots = [Path("/kaggle/working")]
model_patterns = ["*.pt", "*.bin", "*.safetensors", "*.json", "*.model", "*.txt"]
copied = set()

for root in search_roots:
    for pattern in model_patterns:
        for f in root.rglob(pattern):
            f_str = str(f)
            if any(x in f_str.lower() for x in [
                "best", "checkpoint", "model", "tokenizer", "label_config", "config"
            ]):
                if f_str not in copied:
                    safe_copy(f, SUBDIRS["models"])
                    copied.add(f_str)

print(f"\nTotal model-related files copied: {len(copied)}")

# Summary
summary = {
    "project": "AMP ProtBert training + external benchmark",
    "export_dir": str(EXPORT_DIR),
    "figures_count": len(list(SUBDIRS["figures"].glob("*"))),
    "tables_count": len(list(SUBDIRS["tables"].glob("*"))),
    "logs_count": len(list(SUBDIRS["logs"].glob("*"))),
    "models_count": len(list(SUBDIRS["models"].glob("*"))),
    "data_count": len(list(SUBDIRS["data"].glob("*"))),
}

with open(SUBDIRS["manifests"] / "final_export_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nExport summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# Verification: list all exported files
print("\nExported files:")
for section, folder in SUBDIRS.items():
    files = sorted([p.name for p in folder.glob("*") if p.is_file()])
    print(f"\n[{section}] ({len(files)} files)")
    for name in files[:20]:
        print(f"  - {name}")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")
