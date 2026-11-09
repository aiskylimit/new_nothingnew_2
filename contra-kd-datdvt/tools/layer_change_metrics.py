"""Small, dependency-free aggregation helpers for checkpoint diagnostics."""

import math


def uniform_schedule(teacher_layers, student_layers, count, allow_duplicates=False):
    """Equivalent to utils.get_distillation_schedule without importing utils/DeepSpeed."""
    if not 2 <= count <= min(teacher_layers, student_layers):
        raise ValueError("Need at least two distinct aligned interior layers")
    import numpy as np

    teacher = [int(x) for x in np.linspace(
        0, teacher_layers, count + 2, endpoint=True, dtype=int
    )[1:-1]]
    student = [int(x) for x in np.linspace(
        0, student_layers, count + 2, endpoint=True, dtype=int
    )[1:-1]]
    if not allow_duplicates and (len(set(teacher)) != count or len(set(student)) != count):
        raise ValueError("Layer schedule contains duplicate indices; use --allow-duplicate-layer-schedule to reproduce it")
    return teacher, student


def format_prompt(row):
    """Use the GPT-2 templates in tools/process_data_dolly.py verbatim."""
    instruction = row["instruction"]
    context = row.get("input", "")
    if not isinstance(instruction, str) or not isinstance(context, str):
        raise ValueError("instruction and input must be strings")
    if not context:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    )


def encode_example(tokenizer, row, max_length, max_prompt_length):
    """Mirror process_data_dolly + LMTrainDataset's shifted response-label mask.

    Position len(prompt)-1 predicts the first response token. The final EOS is
    a target, not an input. Keep the original processor's full-text tokenization
    followed by slicing at the untruncated prompt boundary.
    """
    prompt = format_prompt(row)
    response = row["output"]
    if isinstance(response, list):
        if not response:
            raise ValueError("Empty reference list")
        response = response[0]
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Nonempty reference output required")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + response, add_special_tokens=False)
    full_ids.append(tokenizer.eos_token_id)
    response_ids = full_ids[len(prompt_ids):]
    prompt_ids = prompt_ids[:max_prompt_length]
    tokens = (prompt_ids + response_ids)[:max_length]
    if not prompt_ids or len(tokens) <= len(prompt_ids):
        raise ValueError("No response targets remain after truncation")
    positions = list(range(len(prompt_ids) - 1, len(tokens) - 1))
    return tokens[:-1], positions


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def vector_for_example(record, transition=None):
    """Sufficient statistics: example-balanced cosine, pooled relative L2."""
    transitions = record["transitions"]
    if transition is not None:
        transitions = [transitions[transition]]
    valid = sum(t["valid_tokens"] for t in transitions)
    cosine = ratio(sum(t["cosine_sum"] for t in transitions), valid)
    return [
        cosine if cosine is not None else 0.0,
        int(cosine is not None),
        sum(t["delta_error_sq"] for t in transitions),
        sum(t["teacher_delta_sq"] for t in transitions),
    ]


def aggregate(records, transition=None):
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    vectors = [vector_for_example(row, transition) for row in records]
    totals = [math.fsum(v[i] for v in vectors) for i in range(4)]
    transitions = [
        t for row in records
        for t in (row["transitions"] if transition is None else [row["transitions"][transition]])
    ]
    n = sum(t["tokens"] for t in transitions)
    valid = sum(t["valid_tokens"] for t in transitions)
    return {
        "examples": len(records),
        "transition_token_pairs": n,
        "cosine_valid_token_pairs": valid,
        "degenerate_fraction": 1.0 - valid / n if n else None,
        "transition_cosine_distance": ratio(totals[0], totals[1]),
        "relative_transition_error": ratio(totals[2], totals[3]),
        "relative_state_error": ratio(
            math.fsum(row["state_error_sq"] for row in records),
            math.fsum(row["teacher_state_sq"] for row in records),
        ) if transition is None else None,
    }


def paired_bootstrap(baseline, augmented, samples, seed, transition=None):
    """Paired example resampling; difference = augmented - baseline (lower wins).

    NumPy is loaded lazily so preprocessing/aggregation tests need only stdlib.
    These CIs quantify example variation for the available fixed checkpoints,
    not training-seed uncertainty.
    """
    import numpy as np

    if [r["id"] for r in baseline] != [r["id"] for r in augmented]:
        raise ValueError("Paired bootstrap requires identical, ordered example IDs")
    a = np.asarray([vector_for_example(r, transition) for r in baseline], dtype=np.float64)
    b = np.asarray([vector_for_example(r, transition) for r in augmented], dtype=np.float64)
    rng = np.random.default_rng(seed)
    differences = []
    for start in range(0, samples, 64):
        indices = rng.integers(0, len(a), size=(min(64, samples - start), len(a)))
        av, bv = a[indices].sum(axis=1), b[indices].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            differences.append(np.stack([
                bv[:, 0] / bv[:, 1] - av[:, 0] / av[:, 1],
                bv[:, 2] / bv[:, 3] - av[:, 2] / av[:, 3],
            ], axis=1))
    draws = np.concatenate(differences, axis=0)
    point_a, point_b = aggregate(baseline, transition), aggregate(augmented, transition)
    result = {}
    for col, key in enumerate(["transition_cosine_distance", "relative_transition_error"]):
        finite = draws[:, col][np.isfinite(draws[:, col])]
        result[key] = {
            "augmented_minus_baseline": (
                point_b[key] - point_a[key]
                if point_b[key] is not None and point_a[key] is not None else None
            ),
            "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
            "valid_bootstrap_samples": len(finite),
        }
    return result
