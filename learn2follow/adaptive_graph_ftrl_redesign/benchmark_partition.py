"""Question-only benchmark fingerprints; no answers or grading enter control."""

import hashlib
import os
from pathlib import Path


def benchmark_question_ids(tasks, dataset_dir=None):
    if not tasks:
        return {"benchmark_question_ids": [], "benchmark_sources": []}
    from datasets import DatasetDict, load_from_disk

    dataset_dir = dataset_dir or os.environ.get("EVAL_DATASET_DIR")
    if not dataset_dir:
        raise ValueError("EVAL_DATASET_DIR is required for local benchmark loading")

    sources = {
        "aime24": ("HuggingFaceH4/aime_2024", "train", "problem"),
        "aime25": ("yentinglin/aime_2025", "train", "problem"),
        "aime26": ("MathArena/aime_2026", "train", "problem"),
        "hmmt25": ("MathArena/hmmt_feb_2025", "train", "problem"),
    }
    ids, metadata = set(), []
    for task in tasks:
        if task not in sources:
            raise ValueError(f"Benchmark {task} is not configured; add its actual question schema")
        repo, split, key = sources[task]
        path = Path(dataset_dir).expanduser().resolve() / task
        if not path.is_dir():
            raise FileNotFoundError(f"Prepared benchmark does not exist: {path}")
        data = load_from_disk(str(path))
        if isinstance(data, DatasetDict):
            data = data[split]
        for row in data:
            if not isinstance(row.get(key), str):
                raise ValueError(f"{repo}: missing question field {key}")
            ids.add(hashlib.sha256(row[key].strip().encode()).hexdigest())
        metadata.append(
            {
                "task": task,
                "repo": repo,
                "path": str(path),
                "split": split,
                "count": len(data),
                "fingerprint": getattr(data, "_fingerprint", None),
            }
        )
    return {"benchmark_question_ids": sorted(ids), "benchmark_sources": metadata}
