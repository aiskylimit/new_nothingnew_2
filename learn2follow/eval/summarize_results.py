import argparse
import csv
import json
from pathlib import Path
import re


def checkpoint_step(checkpoint_dir):
    if not checkpoint_dir:
        return 0
    match = re.search(r"checkpoint-(\d+)$", str(checkpoint_dir))
    return int(match.group(1)) if match else None


def cumulative_training_tokens(checkpoint_dir):
    if not checkpoint_dir:
        return 0
    state_path = Path(checkpoint_dir) / "follow_the_leader_state.json"
    if not state_path.exists():
        return None
    with state_path.open(encoding="utf-8") as file:
        return json.load(file).get("cumulative_generated_tokens")


def result_row(path):
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    checkpoint_dir = result.get("checkpoint_dir")
    return {
        "result_file": str(path),
        "base_model": result.get("base_model"),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_step": checkpoint_step(checkpoint_dir),
        "cumulative_training_tokens": cumulative_training_tokens(checkpoint_dir),
        "dataset": result.get("dataset"),
        "seed": result.get("seed"),
        "enable_thinking": result.get("enable_thinking"),
        "val_n": result.get("val_n"),
        "average_at_n_pct": result.get("average_at_n_pct"),
        "pass_at_n_pct": result.get("pass_at_n_pct"),
        "majority_vote_at_n_pct": result.get("majority_vote_at_n_pct"),
        "format_rate": result.get("format_rate"),
        "average_output_tokens": result.get("average_output_tokens"),
        "length_truncation_rate": result.get("length_truncation_rate"),
        "total_output_tokens": result.get("total_output_tokens"),
        "generation_seconds": result.get("generation_seconds"),
        "generation_tokens_per_second": result.get("generation_tokens_per_second"),
        "temperature": result.get("temperature"),
        "top_p": result.get("top_p"),
        "top_k": result.get("top_k"),
        "min_p": result.get("min_p"),
        "max_new_tokens": result.get("max_new_tokens"),
        "lora_loaded": result.get("lora_loaded"),
    }


def main():
    parser = argparse.ArgumentParser(description="Export evaluation JSON files to a plotting CSV.")
    parser.add_argument("--results_dir", default="eval_results")
    parser.add_argument("--output", default="eval_results/summary.csv")
    args = parser.parse_args()

    result_paths = sorted(Path(args.results_dir).glob("*.json"))
    rows = [result_row(path) for path in result_paths]
    if not rows:
        raise RuntimeError(f"No evaluation JSON files found in {args.results_dir}.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} evaluation rows to {output_path}")


if __name__ == "__main__":
    main()
