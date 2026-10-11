"""Question-only benchmark fingerprints; no answers or grading enter control."""

import hashlib


def benchmark_question_ids(tasks):
    if not tasks:
        return {"benchmark_question_ids": [], "benchmark_sources": []}
    from datasets import load_dataset

    sources = {
        "aime24": ("HuggingFaceH4/aime_2024", "train", "problem"),
        "aime25": ("yentinglin/aime_2025", "train", "problem"),
    }
    ids, metadata = set(), []
    for task in tasks:
        if task not in sources:
            raise ValueError(f"Benchmark {task} is not configured; add its actual question schema")
        repo, split, key = sources[task]
        data = load_dataset(repo, split=split)
        for row in data:
            if not isinstance(row.get(key), str):
                raise ValueError(f"{repo}: missing question field {key}")
            ids.add(hashlib.sha256(row[key].strip().encode()).hexdigest())
        metadata.append(
            {
                "task": task,
                "repo": repo,
                "split": split,
                "count": len(data),
                "fingerprint": getattr(data, "_fingerprint", None),
            }
        )
    return {"benchmark_question_ids": sorted(ids), "benchmark_sources": metadata}
