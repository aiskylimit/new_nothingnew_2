#!/usr/bin/env python3
"""Prepare local benchmark snapshots using the OPSD parquet/save_to_disk layout."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Value


EVAL_DATASETS = ("aime25", "aime26", "hmmt25")


def prepare_eval_dataset(name, raw_root, output_root):
    source = raw_root / "eval" / name
    search_root = source / "data" if (source / "data").is_dir() else source
    files = sorted(search_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {source}; see download.txt")
    destination = output_root / "eval" / name
    if destination.exists():
        raise FileExistsError(f"Prepared dataset already exists: {destination}")
    table = pa.concat_tables([pq.read_table(str(path)) for path in files]).replace_schema_metadata(None)
    dataset = Dataset(table)
    if not {"problem", "answer"}.issubset(dataset.column_names):
        raise ValueError(f"{name} must contain problem and answer: {dataset.column_names}")
    id_column = "problem_idx" if "problem_idx" in dataset.column_names else "id"

    def normalize(row, index):
        return {
            "problem_id": row.get(id_column, index),
            "problem": row["problem"],
            "answer": str(row["answer"]),
        }

    features = Features({
        "problem_id": dataset.features.get(id_column, Value("int64")),
        "problem": Value("string"),
        "answer": Value("string"),
    })
    dataset = dataset.map(normalize, with_indices=True, remove_columns=dataset.column_names,
                          features=features)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))
    print(f"Saved {len(dataset):,} rows to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--only_eval", choices=EVAL_DATASETS)
    args = parser.parse_args()
    for name in (args.only_eval,) if args.only_eval else EVAL_DATASETS:
        prepare_eval_dataset(name, args.raw_root.expanduser().resolve(), args.output_root.expanduser().resolve())


if __name__ == "__main__":
    main()
