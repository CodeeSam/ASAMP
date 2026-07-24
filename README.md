# ASAMP

**ASAMP** is a reproducible, leakage-aware computational framework for
**antimicrobial peptide (AMP) discovery from public metagenomic sequencing data**.

The repository contains four primary workflow scripts:

1. **AMP classifier training** — Curates AMP and pseudo-negative sequences,
   performs similarity-aware partitioning, fine-tunes ProtBert and evaluates the
   resulting classifier.
2. **Metagenomic discovery pipeline** — Downloads public metagenomic reads,
   performs quality control, assembly, ORF prediction, peptide filtering,
   clustering, prediction and candidate prioritization.
3. **External comparator benchmarking** — Runs Macrel, amPEPpy and ampir on the
   shared 500-sequence benchmark and calculates standardized classification
   metrics.
4. **BLASTp figure generation** — Produces the six-panel publication figure from
   an exported NCBI BLASTp alignment hit table.

ClassAMP and AMPScanner v2 were evaluated separately through their web servers
using default settings. They are not executed by the programmatic benchmarking
script.

---

## Repository structure

```text
ASAMP/
├──Supplementary_Figures
├──Supplementary_Tables
├── configs/
│   └── default_config.json
├── docs/
│   └── output_files.md
├──input data
│   ├── full_AMPs.fasta — curated positive AMP sequences originally sourced fromAPD3
│   ├── SRRs.csv — public metagenomic run accessions and body-site annotations
│   ├── Dataset_AMPs_MICs.csv — DBAASP-derived AMP records used to construct theexternal benchmark
│   ├── Y5SY57DB014-Alignment-HitTable.csv — exported BLASTp alignment hit table
│   └── For_external_validation — labelled external benchmark CSV containing sequence and label columns
├── notebooks/
│   ├── 01_amp_classifier_training.py
│   ├── 02_metagenome_discovery_pipeline.py
│   └── 03_external_benchmarking.py
|   └── 04_plot_blastp_results.py
├── scripts/
│   ├── export_kaggle_outputs.py
├── .gitignore
├── LICENSE
└── README.md
```

Generated models, tables, figures and logs are written to the output directories
defined by the individual scripts and are not required to be tracked by Git.

---

## Workflow overview

### 1. AMP classifier development

The first workflow:

- loads curated AMP sequences;
- constructs a pseudo-negative set from reviewed UniProt Swiss-Prot records;
- filters sequences to 10–50 amino acids;
- removes exact duplicates within and across classes;
- balances the classes at a 1:1 ratio;
- clusters the combined dataset at 80% sequence identity;
- removes mixed-label clusters;
- assigns complete clusters to training, validation and test partitions;
- selectively fine-tunes ProtBert; and
- evaluates the classifier on internal and external datasets.

### 2. Metagenomic AMP discovery

The second workflow begins with publicly available human microbiome metagenomic
reads and performs:

```text
Public metagenomic reads
        ↓
Quality control with fastp
        ↓
Assembly with MEGAHIT
        ↓
ORF prediction with MetaProdigal
        ↓
Peptide translation and 10–50-aa filtering
        ↓
Exact deduplication
        ↓
CD-HIT clustering at 90% identity
        ↓
ASAMP prediction on cluster representatives
        ↓
Physicochemical prioritization and candidate ranking
```

### 3. External comparator benchmarking

The third workflow evaluates the following tools programmatically:

- Macrel
- amPEPpy
- ampir

All three are evaluated on the same labelled benchmark using a 0.50
classification threshold. The script saves aligned per-sequence predictions,
raw tool outputs, summary metrics and environment information.

### 4. BLASTp result visualization

The fourth workflow generates the publication figure from a headerless,
13-column BLASTp alignment hit table exported from the NCBI Protein BLAST web
interface. It does **not** submit BLAST searches itself.

---

## Requirements

### Core environment

| Category | Tools or packages |
|---|---|
| Python | Python 3.10 or later; the study was run in Python 3.12 |
| Deep learning | PyTorch, Transformers, sentencepiece |
| Data analysis | pandas, NumPy, scikit-learn |
| Sequence analysis | Biopython, CD-HIT, SeqKit/Seqtk |
| Metagenomic processing | fastp, MEGAHIT, Prodigal/MetaProdigal |
| Data retrieval | SRA Toolkit and ENA REST services |
| Plotting | Matplotlib |

### External benchmarking dependencies

| Tool | Requirement |
|---|---|
| Macrel | Macrel command-line executable |
| amPEPpy | `ampep` executable and pretrained `amPEP.model` |
| ampir | R, `Rscript` and the `ampir` R package |

The original comparator runs used:

- Macrel 1.6.0
- amPEPpy 1.1.0
- ampir 1.1.0
- scikit-learn 1.6.1

The bundled amPEPpy estimators emitted an `InconsistentVersionWarning` because
they had been serialized under scikit-learn 1.4.0. The benchmarking script does
not suppress this warning.

The Kaggle workflows install their Python and system dependencies at runtime.
For local use, install the required packages and ensure external executables are
available on `PATH`.

---

## Input data

The study used the following main inputs:

- `full_AMPs.fasta` — curated positive AMP sequences originally sourced from
  APD3;
- `SRRs.csv` — public metagenomic run accessions and body-site annotations;
- `Dataset_AMPs_MICs.csv` — DBAASP-derived AMP records used to construct the
  external benchmark;
- a labelled external benchmark CSV containing `sequence` and `label` columns;
- an exported BLASTp alignment hit table containing 13 tabular fields.

The labelled benchmark uses:

```text
label = 1  → AMP
label = 0  → non-AMP
```

---

## Quickstart

### 1. Train the ASAMP classifier

Run:

```bash
python notebooks/01_amp_classifier_training.py
```

Update the paths in the script or configuration file to point to the required
input datasets.

The workflow saves:

- the best and final model checkpoints;
- tokenizer files;
- cluster assignments and split maps;
- validation, test and external predictions;
- performance tables and figures;
- training history;
- run configuration; and
- environment information.

### 2. Run the metagenomic discovery workflow

Run:

```bash
python notebooks/02_metagenome_discovery_pipeline.py
```

The script can reuse the trained `model` and `tokenizer` objects in the same
Kaggle session. When run in a separate session, configure `model_dir` to point
to the saved model artifact.

### 3. Run programmatic external benchmarking

The benchmark input CSV must contain at least:

```text
sequence
label
```

Run:

```bash
python notebooks/03_external_benchmarking.py \
    --input data/For_external_validation.csv \
    --ampeppy-model /path/to/amPEPpy/pretrained_models/amPEP.model \
    --output-dir results/external_benchmark
```

An existing sequence identifier can be preserved with:

```bash
--id-column seq_id
```

The script expects 500 sequences by default. Disable this size check with:

```bash
--expected-size 0
```

Use `--overwrite` to replace an existing output directory.

### 4. Generate the BLASTp summary figure

The input must be a headerless CSV containing these fields in order:

```text
qseqid, sseqid, pident, length, mismatch, gapopen,
qstart, qend, sstart, send, evalue, bitscore, qcovs
```

Run:

```bash
python scripts/04_plot_blastp_results.py \
    --input supplementary/tables11_alignment-HitTable.csv \
    --output-prefix figures/Figure_19_BLASTp_Results \
    --total-candidates 243 \
    --highlight-query AMP_0206 \
    --dpi 500
```

The script writes:

- a vector PDF;
- a 500-dpi PNG; and
- a candidate-level best-hit summary CSV.

---

## Model details

| Setting | Value |
|---|---|
| Base model | `Rostlab/prot_bert` |
| Trainable encoder layers | Final 2 layers |
| Additional trainable components | Pooler and classification head |
| Classifier | Two-layer MLP with GELU; hidden dimension 256 |
| Sequence length range | 10–50 amino acids |
| Training-data clustering | 80% identity |
| Metagenomic peptide clustering | 90% identity |
| Split strategy | Cluster-aware stratified split |
| Train/validation/test ratio | 70% / 15% / 15% |
| Random seed | 42 |
| Default prediction threshold | 0.50 |
| Standalone external ASAMP threshold | 0.524, selected on validation data |
| Early-stopping metric | Validation MCC |
| Early-stopping patience | 3 epochs |
| Label smoothing | 0.02 |

---

## Dataset construction

### Positive class

Positive sequences were obtained from the project AMP FASTA file, originally
curated from APD3:

> Wang G, Li X, Wang Z. APD3: the antimicrobial peptide database as a tool for
> research and education. *Nucleic Acids Research*. 2016;44(D1):D1087–D1093.  
> https://doi.org/10.1093/nar/gkv1278

### Pseudo-negative class

Pseudo-negative sequences were downloaded programmatically from reviewed
UniProt Swiss-Prot records through the UniProt REST API. Short sequences
containing antimicrobial-related annotations or exclusion terms were removed
before model development.

### Redundancy and leakage control

- Sequences were restricted to 10–50 amino acids.
- Exact duplicates were removed within each class.
- Exact sequences occurring in both classes were removed.
- The combined balanced dataset was clustered at 80% identity.
- Sequences longer than 10 residues were clustered with CD-HIT.
- Ten-residue peptides were processed using a custom equal-length identity
  clustering procedure.
- Clusters containing both AMP and non-AMP labels were removed.
- Complete clusters were assigned to only one of the training, validation or
  test partitions.

After balancing, the pre-clustering dataset contained 5,430 sequences. Removal
of three mixed-label clusters excluded 133 sequences, leaving 5,297 sequences
for model development.

---

## External benchmark

The final shared benchmark contained:

- 250 AMP sequences from DBAASP; and
- 250 non-AMP sequences obtained from a separate UniProt Swiss-Prot query.

Exact sequence matches to the ASAMP training dataset were removed. Near-identical
overlap between the training and external datasets was not independently
filtered.

The same 500 sequences were evaluated with:

- ASAMP
- Macrel
- amPEPpy
- ampir
- ClassAMP
- AMPScanner v2

Macrel, amPEPpy and ampir were run programmatically. ClassAMP and AMPScanner v2
were evaluated through their respective web servers using default settings in
early May 2026. Their returned predictions were matched to the shared benchmark
labels before calculating accuracy, precision, recall, F1 score and Matthews
correlation coefficient.

To support full comparator traceability, raw ClassAMP and AMPScanner v2 outputs
may be stored in a repository directory such as:

```text
results/external_benchmark/webserver_outputs/
├── classamp_webserver_predictions.csv
└── ampscanner_v2_webserver_predictions.csv
```

---

## Outputs

### Classifier-training workflow

| Path | Description |
|---|---|
| `models/best_model.pt` | Best checkpoint selected by validation MCC |
| `models/last_model.pt` | Final checkpoint including optimizer state |
| `models/tokenizer/` | Saved ProtBert tokenizer |
| `tables/table_s1_raw_counts.csv` | Initial sequence counts |
| `tables/table_s3_final_counts.csv` | Counts after filtering and balancing |
| `tables/table_s4_training_sequences_with_clusters.csv` | Training sequences and cluster assignments |
| `tables/table_s5_cluster_stats.csv` | Cluster-level statistics |
| `tables/table_s6_mixed_label_clusters_removed.csv` | Mixed-label clusters excluded before splitting |
| `tables/table_s7_final_metrics.csv` | Validation and test metrics |
| `tables/table_s8_validation_predictions.csv` | Per-sequence validation predictions |
| `tables/table_s9_test_predictions.csv` | Per-sequence test predictions |
| `tables/table_s10_external_dataset.csv` | Shared external dataset |
| `tables/table_s11_external_benchmark_predictions.csv` | ASAMP external predictions |
| `figures/fig_06_test.*` | Internal test ROC and precision-recall curves |
| `figures/fig_07_test_confusion_matrix.*` | Internal test confusion matrix |
| `figures/fig_10_test_embedding_tsne.*` | t-SNE projection of test embeddings |
| `logs/training_history.csv` | Per-epoch training and validation metrics |
| `logs/run_summary.json` | Run-level summary |
| `logs/environment.json` | Python, platform and package information |
| `configs/config.json` | Frozen run configuration |

### Metagenomic-discovery workflow

| Path | Description |
|---|---|
| `results/download_manifest.csv` | Download status for each public accession |
| `results/assembly_manifest.csv` | Assembly status for each sample |
| `results/cluster_representatives_predictions.csv` | Predictions on 90%-identity representatives |
| `results/tier1_high_confidence_candidates.csv` | Probability ≥0.90, charge ≥+2 and GRAVY ≤0.5 |
| `results/tier2_moderate_candidates.csv` | Probability ≥0.80 and charge ≥+1 |
| `results/top30_final_candidates.csv` | Top 30 Tier 1 candidates |
| `results/final_candidate_shortlist.csv` | Ranked final shortlist |
| `results/final_candidate_shortlist.faa` | FASTA representation of the shortlist |
| `supplementary/table_s1_qc_summary.csv` | fastp quality-control summary |
| `supplementary/table_s2_assembly_stats.csv` | Assembly statistics |
| `supplementary/table_s3_orf_and_peptide_counts_per_sample.csv` | ORF and peptide counts |
| `supplementary/table_s4_dedup_membership.csv` | Exact-deduplication membership |
| `supplementary/table_s5_cluster_sizes.csv` | CD-HIT cluster sizes |
| `supplementary/table_s6_representative_predictions_with_properties.csv` | Representative predictions and properties |
| `supplementary/table_s7_cluster_member_predictions.csv` | Predictions expanded to cluster members |

### Programmatic comparator benchmarking

| Path | Description |
|---|---|
| `benchmark_input_normalized.csv` | Validated and normalized benchmark |
| `benchmark_input.fasta` | FASTA used by comparator tools |
| `programmatic_predictions.csv` | Aligned predictions from Macrel, amPEPpy and ampir |
| `programmatic_benchmark_metrics.csv` | Accuracy, precision, recall, F1 and MCC |
| `benchmark_environment.json` | Software versions and executable locations |
| `benchmark_run.log` | Commands, standard output and errors |
| `raw/macrel/` | Raw Macrel outputs |
| `raw/ampeppy_predictions.tsv` | Raw amPEPpy output |
| `raw/ampir_output.csv` | Raw ampir output |
| `raw/run_ampir.R` | Generated R script used to run ampir |

### BLASTp figure generation

For an output prefix such as `figures/Figure_19_BLASTp_Results`, the script
creates:

| Path | Description |
|---|---|
| `figures/Figure_19_BLASTp_Results.pdf` | Vector publication figure |
| `figures/Figure_19_BLASTp_Results.png` | High-resolution raster figure |
| `figures/Figure_19_BLASTp_Results_best_hits.csv` | Best hit and identity category for each matched query |

---

## Reproducibility

- Random seeds are fixed at 42.
- Cluster-aware splitting prevents individual clusters from being divided
  across training, validation and test partitions.
- Deterministic CUDA/cuDNN settings are enabled where supported.
- Model, dataset and processing configurations are saved with each run.
- UniProt query details are recorded for pseudo-negative construction.
- CD-HIT membership and cluster-to-split assignments are saved.
- Raw comparator outputs are retained before metric calculation.
- The external benchmarking script aligns predictions using sequence
  identifiers rather than relying only on row order.
- The BLASTp plotting script saves the candidate-level best-hit table used in
  the figure.

---

## Study outputs

The ASAMP workflow identified:

- 1,104 AMP-positive cluster representatives from 8,092 representatives;
- 243 Tier 1 high-confidence candidates after physicochemical prioritization;
  and
- 180 of 243 Tier 1 candidates with no significant NCBI nr match under the
  specified BLASTp search conditions.

These are computationally prioritized candidates and require downstream
structural and experimental validation.

---

## Citation

When using this repository, cite the associated ASAMP manuscript. Full citation
details and DOI will be added after publication.

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.
