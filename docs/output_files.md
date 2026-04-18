# Output Files Reference

This document describes every file produced by the two notebooks.

---

## Notebook 1 — AMP Classifier Training

All outputs are written under `/kaggle/working/amp_protbert_training/` (configurable via `cfg.output_root`).

### `models/`

| File | Description |
|---|---|
| `best_model.pt` | Best checkpoint selected by `val_mcc`. Stores `model_state_dict`, `config`, and `history`. Weights only — no optimizer state — keeping the file around 420 MB. |
| `last_model.pt` | Checkpoint from the final training epoch. Stores `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `config`, and `history`. Use this to resume training. |
| `tokenizer/` | Saved HuggingFace tokenizer directory (vocab, config, special tokens). |

### `configs/`

| File | Description |
|---|---|
| `config.json` | Frozen copy of the `Config` dataclass serialized to JSON. Every hyperparameter used in the run is recorded here. |

### `logs/`

| File | Description |
|---|---|
| `training_history.csv` | Per-epoch train and val metrics: loss, accuracy, precision, recall, F1, MCC, AUROC, AUPRC. |
| `run_summary.json` | High-level summary: best epoch, best metric value, dataset sizes, cluster counts, parameter counts. |
| `environment.json` | Full `pip freeze` output plus Python version, platform, and PyTorch version. Enables exact environment reconstruction. |
| `cdhit.log` | stdout/stderr from the CD-HIT run. |

### `splits/`

| File | Description |
|---|---|
| `training_sequences_with_splits.csv` | Every post-filtering sequence with its assigned cluster ID and split (`train` / `val` / `test`). |
| `split_sequence_summary.csv` | Sequence counts by split and label. |
| `split_cluster_summary.csv` | Cluster counts by split and label. |
| `cluster_to_split_map.json` | Mapping from `cluster_id` → `train` / `val` / `test`. |

### `tables/`

| File | Description |
|---|---|
| `table_s1_raw_counts.csv` | Raw sequence counts before any filtering. |
| `table_s2_prebalance_counts.csv` | Counts after filtering and deduplication, before class balancing. |
| `table_s3_final_counts.csv` | Final balanced dataset counts (AMP, NonAMP, combined). |
| `table_s4_training_sequences_precluster.csv` | All sequences before CD-HIT clustering. |
| `table_s4_training_sequences_with_clusters.csv` | All sequences after cluster assignment. |
| `table_s5_cluster_stats.csv` | Per-cluster size, AMP count, NonAMP count, and mixed-label flag. |
| `table_s6_mixed_label_clusters_removed.csv` | Clusters removed because they contained both AMP and non-AMP sequences. |
| `table_s7_final_metrics.csv` | Val and test metrics at the 0.5 threshold. |
| `table_s8_validation_predictions.csv` | Per-sequence validation predictions (`y_true`, `y_prob`, `y_pred`). |
| `table_s9_test_predictions.csv` | Per-sequence test predictions. |
| `table_s10_external_dataset.csv` | External benchmark sequences after filtering and leakage removal. |
| `table_s11_external_benchmark_predictions.csv` | Per-sequence external benchmark predictions. |
| `table_s12_external_benchmark_metrics.csv` | External benchmark aggregate metrics. |
| `table_s12_test_physicochemical_properties.csv` | Test-set sequences with GRAVY, pI, MW, charge at pH 7.4, and hydrophobic fraction. |

### `figures/`

All figures are saved in PNG, PDF, and SVG formats.

| Stem | Description |
|---|---|
| `fig_01_class_distribution` | Bar chart of AMP vs NonAMP counts after filtering. |
| `fig_02_length_distribution` | Peptide length histograms by class. |
| `fig_03_aa_composition` | Amino acid frequency by class. |
| `fig_04_loss_curves` | Train and val loss per epoch. |
| `fig_05_auprc_curves` | Train and val AUPRC per epoch. |
| `fig_06_test_roc` | Test-set ROC curve with AUROC. |
| `fig_06_test_pr` | Test-set precision-recall curve with AUPRC. |
| `fig_07_test_confusion_matrix` | Test-set confusion matrix. |
| `fig_08_test_calibration` | Test-set calibration curve. |
| `fig_09_test_score_distribution` | Predicted probability histogram by true class. |
| `fig_10_test_embedding_tsne` | t-SNE of ProtBert CLS embeddings on the test set. |
| `fig_11_external_roc` | External benchmark ROC curve. |
| `fig_11_external_pr` | External benchmark precision-recall curve. |
| `fig_12_external_confusion_matrix` | External benchmark confusion matrix. |
| `fig_13_external_score_distribution` | External benchmark score distribution. |

### Root working directory (`/kaggle/working/`)

| File | Description |
|---|---|
| `non_AMPs_filtered.fasta` | Filtered non-AMP FASTA downloaded from UniProt. |
| `non_AMPs_raw_unfiltered.fasta` | Raw UniProt sequences before keyword exclusion (for audit purposes). |
| `non_AMPs_uniprot_metadata.tsv` | Full UniProt metadata (accession, organism, function text, sequence) for the downloaded non-AMPs. |
| `non_AMPs_uniprot_query_log.json` | Exact query parameters, exclusion keywords, and record count used to construct the negative set. |

---

## Notebook 2 — Metagenome Discovery Pipeline

All outputs are written under `/kaggle/working/metagenome_amp_discovery/`.

### `results/`

| File | Description |
|---|---|
| `download_manifest.csv` | Download status, layout, and local file paths per accession. |
| `trim_manifest.csv` | fastp trimming status and output paths per sample. |
| `assembly_manifest.csv` | MEGAHIT assembly status and contig FASTA paths per sample. |
| `orf_manifest.csv` | MetaProdigal ORF calling status and output paths per sample. |
| `dedup_summary.json` | Total filtered peptide count and unique sequence count after exact deduplication. |
| `cluster_membership.csv` | CD-HIT cluster membership table (`cluster_id`, `seq_id`, `is_representative`). |
| `cluster_representatives_pre_prediction.csv` | Cluster representatives before inference. |
| `cluster_representatives_predictions.csv` | Cluster representatives after inference (`Prob_AMP`, `Pred_Class`, `High_Confidence`). |
| `tier1_high_confidence_candidates.csv` | Candidates with Prob_AMP ≥ 0.9, charge at pH 7.4 ≥ 2, GRAVY ≤ 0.5. |
| `tier2_moderate_candidates.csv` | Candidates with Prob_AMP ≥ 0.8, charge at pH 7.4 ≥ 1. |
| `top30_final_candidates.csv` | Top 30 candidates from Tier 1 ranked by probability. |
| `final_candidate_shortlist.csv` | Complete ranked shortlist (Tier 1 criteria, all candidates). |
| `final_candidate_shortlist.faa` | FASTA of shortlisted candidates. |
| `run_summary.json` | End-to-end pipeline counts: samples, assemblies, peptides, clusters, predictions, shortlist size. |

### `supplementary/`

| File | Description |
|---|---|
| `table_s1_qc_summary.csv` | Read QC statistics (reads before/after trimming, Q20, Q30, GC content) per sample. |
| `table_s2_assembly_stats.csv` | Assembly statistics (contig count, total bp, N50, mean/median length) per sample. |
| `table_s3_orf_and_peptide_counts_per_sample.csv` | ORF counts and filtered short-peptide counts per sample. |
| `table_s4_dedup_membership.csv` | Exact deduplication membership (original IDs → representative ID). |
| `table_s5_cluster_sizes.csv` | CD-HIT cluster size distribution. |
| `table_s6_representative_predictions_with_properties.csv` | Predictions with physicochemical properties (GRAVY, pI, MW, charge, aromaticity, instability index) and provenance (accession, body site). |
| `table_s7_cluster_member_predictions.csv` | Predictions expanded from representatives to all cluster members. |
| `supplementary_manifest.csv` | Index of all supplementary table files. |

### `figures/`

| File | Description |
|---|---|
| `fig01_sample_distribution.png/.pdf` | Sample counts by body site. |
| `fig02_reads_after_trimming_distribution.png/.pdf` | Distribution of read counts after fastp. |
| `fig03_reads_before_vs_after.png/.pdf` | Scatter of read counts before vs after trimming. |
| `fig04_assembly_contig_counts.png/.pdf` | Distribution of contig counts per sample. |
| `fig05_assembly_n50_distribution.png/.pdf` | N50 distribution across assemblies. |
| `fig06_orf_count_distribution.png/.pdf` | ORF counts per sample. |
| `fig07_filtered_peptide_count_distribution.png/.pdf` | Filtered short-peptide counts per sample. |
| `fig08_filtered_peptide_length_distribution.png/.pdf` | Length distribution of all filtered peptides. |
| `fig09_cluster_size_distribution.png/.pdf` | CD-HIT cluster size distribution. |
| `fig10_cluster_size_log_distribution.png/.pdf` | Log-scaled cluster size distribution. |
| `fig11_probability_distribution.png/.pdf` | AMP probability distribution across all representatives. |
| `fig12_probability_distribution_predicted_positives.png/.pdf` | AMP probability distribution among predicted positives only. |
| `fig13_positive_representatives_by_body_site.png/.pdf` | Predicted AMP-positive representatives by body site. |
| `fig14_highconf_charge_distribution.png/.pdf` | Charge distribution of high-confidence candidates. |
| `fig15_highconf_gravy_distribution.png/.pdf` | GRAVY distribution of high-confidence candidates. |

### `manifests/`

| File | Description |
|---|---|
| `notebook2_config.json` | Frozen copy of the `CONFIG` dictionary. |
| `sample_manifest_input.csv` | Loaded sample sheet (accession + body site). |
| `logs/notebook2_<run_id>.log` | Full run log with timestamps. |
