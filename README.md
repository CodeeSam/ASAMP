# ASAMP

A reproducible pipeline for **antimicrobial peptide (AMP) discovery** from metagenomic data.

The repository contains two notebooks:

1. **Notebook 1** — Fine-tunes a lightweight ProtBert classifier on curated AMP sequences and evaluates it on held-out and external benchmark data.
2. **Notebook 2** — Applies the trained classifier to raw metagenomic reads: download → trim → assemble → call ORFs → filter → cluster → predict → rank candidates.

---

## Repository structure

```
amp-protbert-discovery/
├── notebooks/
│   ├── 01_amp_classifier_training.py        # AMP classifier training & evaluation
│   └── 02_metagenome_discovery_pipeline.py  # Metagenome discovery pipeline
├── configs/
│   └── default_config.json                  # Reference copy of all hyperparameters
├── docs/
│   └── output_files.md                      # Description of every output file
├── scripts/
│   └── export_kaggle_outputs.py             # Helper to zip & export Kaggle outputs
└── README.md
```

---

## Quickstart

### Prerequisites

| Category | Tools / Packages |
|---|---|
| Python | ≥ 3.9 |
| Deep learning | PyTorch, Transformers (HuggingFace), sentencepiece |
| Bioinformatics | biopython, CD-HIT, fastp, MEGAHIT, Prodigal |
| ML / analysis | scikit-learn, pandas, numpy, matplotlib |

All Python dependencies are installed inside the notebooks at runtime (Kaggle-compatible). System tools (`cd-hit`, `fastp`, `megahit`, `prodigal`) are installed via `apt-get`.

### Running on Kaggle

1. Upload your data files to Kaggle dataset named; in this work, we used:
   - `full_AMPs.fasta` — positive training sequences
   - `SRRs.csv` — metagenomic sample accessions with body site annotations

2. Upload the DBAASP peptide dataset (for external benchmark):
   - `Dataset_AMPs_MICs.csv`

3. Run **Notebook 1** (`01_amp_classifier_training.py`) first. It trains the model and saves:
   - `best_model.pt` — best checkpoint (weights only, ~420 MB)
   - `last_model.pt` — last checkpoint (weights + optimizer state)
   - tokenizer files

4. Run **Notebook 2** (`02_metagenome_discovery_pipeline.py`) in the same Kaggle session (so the trained `model` and `tokenizer` objects are already in memory), or point `model_dir` to a saved Kaggle model artifact.

---

## Model details

| Setting | Value |
|---|---|
| Base model | `Rostlab/prot_bert` |
| Trainable layers | Last 2 encoder layers + pooler |
| Classifier | 2-layer MLP with GELU, hidden dim 256 |
| Sequence length filter | 10–50 amino acids |
| Clustering identity | 80% (CD-HIT) for training splits |
| Split strategy | Cluster-aware stratified (train / val / test = 70 / 15 / 15%) |
| Prediction threshold | 0.5 |
| Early stopping metric | `val_mcc` (patience = 3) |
| Label smoothing | 0.02 |

---

## Dataset construction

**Positives:** curated AMP sequences from the project-specific FASTA file (originally sourced from APD3 Database [Wang et al., 2016]; https://doi.org/10.1093/nar/gkv1278).

**Pseudo-negatives:** programmatically downloaded from UniProt Swiss-Prot via the REST API. Sequences matching any antimicrobial-related keyword in their name, function annotation, or organism field are excluded before use.

**Redundancy control:**
- Sequences are filtered to 10–50 aa and exact-deduplicated.
- CD-HIT (80% identity) is run on sequences > 10 aa; 10-aa sequences receive a custom pairwise identity clustering step.
- Clusters containing both AMP and non-AMP members are removed before splitting.

**External benchmark:** 250 AMPs from DBAASP (Pirtskhalava et al., 2021; https://doi.org/10.1093/nar/gkaa991) and 250 non-AMPs from a fresh UniProt query, with strict leakage removal against all training sequences.

---

## Outputs

### Notebook 1

| Path | Description |
|---|---|
| `models/best_model.pt` | Best checkpoint by val MCC (weights only) |
| `models/last_model.pt` | Last checkpoint (weights + optimizer state) |
| `models/tokenizer/` | Saved tokenizer |
| `tables/table_s1_raw_counts.csv` | Raw sequence counts |
| `tables/table_s7_final_metrics.csv` | Val and test metrics |
| `tables/table_s9_test_predictions.csv` | Per-sequence test predictions |
| `tables/table_s11_external_benchmark_predictions.csv` | External benchmark predictions |
| `figures/fig_06_test_roc.*` | Test ROC curve (PNG, PDF, SVG) |
| `figures/fig_07_test_confusion_matrix.*` | Test confusion matrix |
| `figures/fig_10_test_embedding_tsne.*` | t-SNE of test-set ProtBert embeddings |
| `logs/training_history.csv` | Per-epoch train/val metrics |
| `logs/run_summary.json` | Summary statistics for the run |
| `logs/environment.json` | Full pip freeze + platform info |
| `configs/config.json` | Frozen hyperparameter config |

### Notebook 2

| Path | Description |
|---|---|
| `results/download_manifest.csv` | Download status per accession |
| `results/assembly_manifest.csv` | Assembly status per sample |
| `results/cluster_representatives_predictions.csv` | Predictions on CD-HIT representatives |
| `results/tier1_high_confidence_candidates.csv` | Prob ≥ 0.9, charge ≥ 2, GRAVY ≤ 0.5 |
| `results/tier2_moderate_candidates.csv` | Prob ≥ 0.8, charge ≥ 1 |
| `results/top30_final_candidates.csv` | Top 30 from Tier 1 |
| `results/final_candidate_shortlist.csv` | Full ranked shortlist |
| `results/final_candidate_shortlist.faa` | FASTA of shortlisted candidates |
| `supplementary/table_s1_qc_summary.csv` | Read QC summary (fastp) |
| `supplementary/table_s2_assembly_stats.csv` | Assembly statistics |
| `supplementary/table_s6_representative_predictions_with_properties.csv` | Predictions + physicochemical properties |

---

## Reproducibility

- All random seeds are fixed (`seed = 42`).
- `torch.backends.cudnn.deterministic = True` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are set.
- The full pip environment is captured to `logs/environment.json` at the end of Notebook 1.
- The UniProt query used to generate the negative training set is logged to `non_AMPs_uniprot_query_log.json`.
- CD-HIT cluster membership and train/val/test split maps are saved as CSV and JSON for full traceability.

---

## Citation

If you use this code or pipeline, please cite the associated manuscript (details to be added on publication).

---

## License

MIT License. See `LICENSE` for details.
