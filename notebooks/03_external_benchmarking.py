#!/usr/bin/env python3
"""
ASAMP external benchmarking for programmatically evaluated comparator tools.

This script reproduces the shared-benchmark predictions and metric calculations
for the three tools that were run programmatically in the ASAMP study:

    1. Macrel
    2. amPEPpy
    3. ampir

ClassAMP and AMPScanner v2 were evaluated separately through their web servers
and are therefore not executed by this script.

Expected input
--------------
A CSV file containing at least:

    sequence    Amino-acid sequence
    label       Ground-truth binary label: 1 = AMP, 0 = non-AMP

An existing sequence identifier column may be supplied with --id-column.
Otherwise, deterministic identifiers seq_0, seq_1, ... are generated.

Original study settings
-----------------------
- Shared benchmark: 500 sequences (250 AMP and 250 non-AMP)
- Decision threshold: 0.50 for all three tools
- Macrel: local command-line prediction
- amPEPpy: local command-line prediction with the bundled pretrained model
- ampir: R package, model="mature"

Required software
-----------------
Python packages:
    numpy
    pandas
    scikit-learn

External tools:
    macrel
    ampep           # amPEPpy command-line executable
    Rscript
    R package ampir

Example
-------
python notebooks/03_external_benchmarking.py \
    --input data/For_external_validation.csv \
    --ampeppy-model /path/to/amPEPpy/pretrained_models/amPEP.model \
    --output-dir results/external_benchmark

Outputs
-------
- benchmark_input_normalized.csv
- benchmark_input.fasta
- programmatic_predictions.csv
- programmatic_benchmark_metrics.csv
- benchmark_environment.json
- benchmark_run.log
- raw/macrel/*
- raw/ampeppy_predictions.tsv
- raw/ampir_output.csv
- raw/run_ampir.R

Notes
-----
Macrel reports predicted AMP entries in its prediction file. Sequences absent
from that file are treated as non-AMP, matching the original notebook workflow.

The original amPEPpy run used amPEPpy 1.1.0 under scikit-learn 1.6.1. Its
bundled estimators emitted an InconsistentVersionWarning because they had been
serialized under scikit-learn 1.4.0. This script does not suppress that warning.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Macrel, amPEPpy and ampir on the shared ASAMP external "
            "benchmark and calculate classification metrics."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV containing benchmark sequences and binary labels.",
    )
    parser.add_argument(
        "--ampeppy-model",
        required=True,
        type=Path,
        help="Path to amPEPpy's pretrained amPEP.model file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external_benchmark"),
        help="Directory for predictions, metrics and raw outputs.",
    )
    parser.add_argument(
        "--sequence-column",
        default="sequence",
        help="Input CSV column containing amino-acid sequences.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Input CSV column containing binary labels.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help=(
            "Optional existing sequence-ID column. When omitted, IDs are "
            "generated as seq_0, seq_1, ..."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold used to convert scores to labels.",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=500,
        help=(
            "Expected benchmark size. Set to 0 to disable the size check. "
            "The ASAMP study used 500 sequences."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("asamp_external_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"Required executable '{name}' was not found on PATH."
        )
    return executable


def run_command(
    command: list[str],
    logger: logging.Logger,
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    logger.info("Running: %s", " ".join(command))
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if process.stdout.strip():
        logger.info("STDOUT:\n%s", process.stdout.strip())
    if process.stderr.strip():
        logger.info("STDERR:\n%s", process.stderr.strip())

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: "
            f"{' '.join(command)}"
        )
    return process


def normalise_sequence(sequence: object) -> str:
    return "".join(str(sequence).upper().split()).replace("*", "")


def load_benchmark(
    input_path: Path,
    sequence_column: str,
    label_column: str,
    id_column: str | None,
    expected_size: int,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {input_path}")

    dataframe = pd.read_csv(input_path)

    required_columns = {sequence_column, label_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if id_column is not None and id_column not in dataframe.columns:
        raise ValueError(f"ID column not found in input CSV: {id_column}")

    dataframe = dataframe.copy().reset_index(drop=True)
    dataframe["sequence"] = dataframe[sequence_column].map(normalise_sequence)
    dataframe["label"] = pd.to_numeric(
        dataframe[label_column], errors="raise"
    ).astype(int)

    if id_column is None:
        dataframe["seq_id"] = [
            f"seq_{index}" for index in range(len(dataframe))
        ]
    else:
        dataframe["seq_id"] = dataframe[id_column].astype(str).str.strip()

    if dataframe["seq_id"].eq("").any():
        raise ValueError("One or more sequence identifiers are empty.")
    if dataframe["seq_id"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["seq_id"].duplicated(), "seq_id"
        ].tolist()[:10]
        raise ValueError(f"Duplicate sequence identifiers found: {duplicates}")

    if not set(dataframe["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary values encoded as 0 or 1.")

    empty_sequences = dataframe["sequence"].eq("")
    if empty_sequences.any():
        raise ValueError(
            f"{int(empty_sequences.sum())} benchmark sequences are empty."
        )

    invalid_rows: list[tuple[str, str]] = []
    for seq_id, sequence in zip(
        dataframe["seq_id"], dataframe["sequence"]
    ):
        invalid_residues = sorted(set(sequence).difference(VALID_AMINO_ACIDS))
        if invalid_residues:
            invalid_rows.append((seq_id, "".join(invalid_residues)))

    if invalid_rows:
        preview = ", ".join(
            f"{seq_id}: {residues}"
            for seq_id, residues in invalid_rows[:10]
        )
        raise ValueError(
            "Non-canonical amino-acid symbols were found. "
            f"Examples: {preview}"
        )

    if expected_size > 0 and len(dataframe) != expected_size:
        raise ValueError(
            f"Expected {expected_size} benchmark sequences, "
            f"but found {len(dataframe)}."
        )

    class_counts = dataframe["label"].value_counts().to_dict()
    if class_counts.get(0, 0) == 0 or class_counts.get(1, 0) == 0:
        raise ValueError("Both AMP and non-AMP classes must be present.")

    return dataframe[["seq_id", "sequence", "label"]]


def write_fasta(dataframe: pd.DataFrame, fasta_path: Path) -> None:
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in dataframe.itertuples(index=False):
            handle.write(f">{row.seq_id}\n{row.sequence}\n")


def probabilities_to_labels(
    probabilities: Iterable[float],
    threshold: float,
) -> np.ndarray:
    values = np.asarray(list(probabilities), dtype=float)
    return (values >= threshold).astype(int)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def verify_prediction_ids(
    prediction_ids: set[str],
    expected_ids: set[str],
    tool_name: str,
    *,
    allow_missing: bool,
) -> None:
    unexpected = prediction_ids.difference(expected_ids)
    if unexpected:
        preview = sorted(unexpected)[:10]
        raise ValueError(
            f"{tool_name} returned unexpected sequence IDs: {preview}"
        )

    if not allow_missing:
        missing = expected_ids.difference(prediction_ids)
        if missing:
            preview = sorted(missing)[:10]
            raise ValueError(
                f"{tool_name} did not return predictions for "
                f"{len(missing)} sequences. Examples: {preview}"
            )


def run_macrel(
    fasta_path: Path,
    raw_dir: Path,
    benchmark_ids: list[str],
    threshold: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    require_executable("macrel")
    output_dir = raw_dir / "macrel"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "macrel",
            "peptides",
            "--fasta",
            str(fasta_path.resolve()),
            "--output",
            str(output_dir.resolve()),
        ],
        logger,
    )

    prediction_file = output_dir / "macrel.out.prediction.gz"
    if not prediction_file.exists():
        candidates = sorted(output_dir.glob("*.prediction.gz"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Could not uniquely identify the Macrel prediction file "
                f"inside {output_dir}."
            )
        prediction_file = candidates[0]

    raw_predictions = pd.read_csv(
        prediction_file,
        sep=r"\s+",
        compression="gzip",
        comment="#",
        engine="python",
    )

    required_columns = {"Access", "AMP_probability"}
    missing = required_columns.difference(raw_predictions.columns)
    if missing:
        raise ValueError(
            f"Macrel output is missing columns: {sorted(missing)}"
        )

    raw_predictions["seq_id"] = (
        raw_predictions["Access"].astype(str).str.strip()
    )
    raw_predictions["probability"] = pd.to_numeric(
        raw_predictions["AMP_probability"], errors="raise"
    )

    if raw_predictions["seq_id"].duplicated().any():
        raise ValueError("Macrel returned duplicate sequence identifiers.")

    expected_ids = set(benchmark_ids)
    verify_prediction_ids(
        set(raw_predictions["seq_id"]),
        expected_ids,
        "Macrel",
        allow_missing=True,
    )

    # Macrel's prediction table contains AMP calls. Entries absent from the
    # table are treated as non-AMP, matching the original notebook analysis.
    probability_map = dict(
        zip(raw_predictions["seq_id"], raw_predictions["probability"])
    )
    probabilities = np.array(
        [probability_map.get(seq_id, 0.0) for seq_id in benchmark_ids],
        dtype=float,
    )
    predictions = probabilities_to_labels(probabilities, threshold)

    return pd.DataFrame(
        {
            "seq_id": benchmark_ids,
            "macrel_probability": probabilities,
            "macrel_prediction": predictions,
        }
    )


def run_ampeppy(
    fasta_path: Path,
    model_path: Path,
    raw_dir: Path,
    benchmark_ids: list[str],
    threshold: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    require_executable("ampep")
    if not model_path.exists():
        raise FileNotFoundError(
            f"amPEPpy model file not found: {model_path}"
        )

    output_file = raw_dir / "ampeppy_predictions.tsv"

    run_command(
        [
            "ampep",
            "predict",
            "-i",
            str(fasta_path.resolve()),
            "-o",
            str(output_file.resolve()),
            "-m",
            str(model_path.resolve()),
        ],
        logger,
    )

    if not output_file.exists():
        raise FileNotFoundError(
            f"amPEPpy output was not created: {output_file}"
        )

    # Despite the filename supplied to the CLI, amPEPpy writes a tab-separated
    # output table.
    raw_predictions = pd.read_csv(output_file, sep="\t")
    column_map = {
        column.lower().strip(): column
        for column in raw_predictions.columns
    }

    if "seq_id" not in column_map:
        raise ValueError(
            "amPEPpy output does not contain a seq_id column."
        )

    raw_predictions["seq_id"] = (
        raw_predictions[column_map["seq_id"]].astype(str).str.strip()
    )

    if raw_predictions["seq_id"].duplicated().any():
        raise ValueError("amPEPpy returned duplicate sequence identifiers.")

    expected_ids = set(benchmark_ids)
    verify_prediction_ids(
        set(raw_predictions["seq_id"]),
        expected_ids,
        "amPEPpy",
        allow_missing=False,
    )

    if "probability_amp" in column_map:
        raw_predictions["probability"] = pd.to_numeric(
            raw_predictions[column_map["probability_amp"]],
            errors="raise",
        )
        raw_predictions["prediction"] = probabilities_to_labels(
            raw_predictions["probability"], threshold
        )
    elif "predicted" in column_map:
        label_map = {
            "amp": 1,
            "nonamp": 0,
            "non-amp": 0,
            "positive": 1,
            "negative": 0,
            "1": 1,
            "0": 0,
        }
        mapped = (
            raw_predictions[column_map["predicted"]]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(label_map)
        )
        if mapped.isna().any():
            values = sorted(
                raw_predictions.loc[
                    mapped.isna(), column_map["predicted"]
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Unrecognised amPEPpy predicted labels: {values}"
            )
        raw_predictions["prediction"] = mapped.astype(int)
        raw_predictions["probability"] = np.nan
    else:
        raise ValueError(
            "amPEPpy output has neither probability_AMP nor predicted."
        )

    aligned = (
        pd.DataFrame({"seq_id": benchmark_ids})
        .merge(
            raw_predictions[
                ["seq_id", "probability", "prediction"]
            ],
            on="seq_id",
            how="left",
            validate="one_to_one",
        )
    )

    return aligned.rename(
        columns={
            "probability": "ampeppy_probability",
            "prediction": "ampeppy_prediction",
        }
    )


def run_ampir(
    fasta_path: Path,
    raw_dir: Path,
    benchmark_ids: list[str],
    threshold: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    require_executable("Rscript")

    r_script = raw_dir / "run_ampir.R"
    output_file = raw_dir / "ampir_output.csv"

    fasta_for_r = fasta_path.resolve().as_posix().replace('"', '\\"')
    output_for_r = output_file.resolve().as_posix().replace('"', '\\"')

    r_script.write_text(
        "\n".join(
            [
                "library(ampir)",
                "",
                f'faa_df <- read_faa("{fasta_for_r}")',
                'results <- predict_amps(faa_df, model = "mature")',
                f'write.csv(results, "{output_for_r}", row.names = FALSE)',
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_command(["Rscript", str(r_script.resolve())], logger)

    if not output_file.exists():
        raise FileNotFoundError(
            f"ampir output was not created: {output_file}"
        )

    raw_predictions = pd.read_csv(output_file)
    required_columns = {"seq_name", "prob_AMP"}
    missing = required_columns.difference(raw_predictions.columns)
    if missing:
        raise ValueError(
            f"ampir output is missing columns: {sorted(missing)}"
        )

    raw_predictions["seq_id"] = (
        raw_predictions["seq_name"].astype(str).str.strip()
    )
    raw_predictions["probability"] = pd.to_numeric(
        raw_predictions["prob_AMP"], errors="raise"
    )

    if raw_predictions["seq_id"].duplicated().any():
        raise ValueError("ampir returned duplicate sequence identifiers.")

    expected_ids = set(benchmark_ids)
    verify_prediction_ids(
        set(raw_predictions["seq_id"]),
        expected_ids,
        "ampir",
        allow_missing=False,
    )

    raw_predictions["prediction"] = probabilities_to_labels(
        raw_predictions["probability"], threshold
    )

    aligned = (
        pd.DataFrame({"seq_id": benchmark_ids})
        .merge(
            raw_predictions[
                ["seq_id", "probability", "prediction"]
            ],
            on="seq_id",
            how="left",
            validate="one_to_one",
        )
    )

    return aligned.rename(
        columns={
            "probability": "ampir_probability",
            "prediction": "ampir_prediction",
        }
    )


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_ampir_version(logger: logging.Logger) -> str | None:
    if shutil.which("Rscript") is None:
        return None
    process = subprocess.run(
        [
            "Rscript",
            "-e",
            'cat(as.character(packageVersion("ampir")))',
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        logger.warning(
            "Could not determine the installed ampir version: %s",
            process.stderr.strip(),
        )
        return None
    return process.stdout.strip() or None


def save_environment(
    output_path: Path,
    threshold: float,
    logger: logging.Logger,
) -> None:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "threshold": threshold,
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "macrel": package_version("macrel"),
            "amPEPpy": package_version("amPEPpy"),
            "ampir": get_ampir_version(logger),
        },
        "executables": {
            "macrel": shutil.which("macrel"),
            "ampep": shutil.which("ampep"),
            "Rscript": shutil.which("Rscript"),
        },
    }
    output_path.write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )


def prepare_output_directory(
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    return output_dir, raw_dir


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    output_dir, raw_dir = prepare_output_directory(
        args.output_dir, args.overwrite
    )
    logger = configure_logging(output_dir / "benchmark_run.log")

    logger.info("Loading benchmark: %s", args.input)
    benchmark = load_benchmark(
        input_path=args.input,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        id_column=args.id_column,
        expected_size=args.expected_size,
    )

    class_counts = benchmark["label"].value_counts().sort_index()
    logger.info(
        "Loaded %d sequences: %d non-AMP and %d AMP.",
        len(benchmark),
        int(class_counts.get(0, 0)),
        int(class_counts.get(1, 0)),
    )

    normalised_csv = output_dir / "benchmark_input_normalized.csv"
    fasta_path = output_dir / "benchmark_input.fasta"
    benchmark.to_csv(normalised_csv, index=False)
    write_fasta(benchmark, fasta_path)

    sequence_ids = benchmark["seq_id"].tolist()
    y_true = benchmark["label"].to_numpy(dtype=int)

    logger.info("Running Macrel.")
    macrel = run_macrel(
        fasta_path=fasta_path,
        raw_dir=raw_dir,
        benchmark_ids=sequence_ids,
        threshold=args.threshold,
        logger=logger,
    )

    logger.info("Running amPEPpy.")
    ampeppy = run_ampeppy(
        fasta_path=fasta_path,
        model_path=args.ampeppy_model,
        raw_dir=raw_dir,
        benchmark_ids=sequence_ids,
        threshold=args.threshold,
        logger=logger,
    )

    logger.info("Running ampir.")
    ampir = run_ampir(
        fasta_path=fasta_path,
        raw_dir=raw_dir,
        benchmark_ids=sequence_ids,
        threshold=args.threshold,
        logger=logger,
    )

    predictions = (
        benchmark.merge(macrel, on="seq_id", validate="one_to_one")
        .merge(ampeppy, on="seq_id", validate="one_to_one")
        .merge(ampir, on="seq_id", validate="one_to_one")
    )

    tool_columns = {
        "Macrel": "macrel_prediction",
        "amPEPpy": "ampeppy_prediction",
        "ampir": "ampir_prediction",
    }

    metric_rows: list[dict[str, float | str]] = []
    for tool_name, prediction_column in tool_columns.items():
        y_pred = predictions[prediction_column].to_numpy(dtype=int)
        row: dict[str, float | str] = {"Tool": tool_name}
        row.update(compute_metrics(y_true, y_pred))
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows).set_index("Tool")
    metrics = metrics[
        ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"]
    ]

    predictions_path = output_dir / "programmatic_predictions.csv"
    metrics_path = output_dir / "programmatic_benchmark_metrics.csv"
    environment_path = output_dir / "benchmark_environment.json"

    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path)
    save_environment(environment_path, args.threshold, logger)

    logger.info("Final benchmark results:\n%s", metrics.to_string())
    logger.info("Saved predictions: %s", predictions_path)
    logger.info("Saved metrics: %s", metrics_path)
    logger.info("Saved environment: %s", environment_path)
    logger.info("Raw tool outputs: %s", raw_dir)

    print("\n=== PROGRAMMATIC BENCHMARK RESULTS ===")
    print(metrics.to_string())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.getLogger("asamp_external_benchmark").exception(
            "Benchmarking failed: %s", error
        )
        raise
