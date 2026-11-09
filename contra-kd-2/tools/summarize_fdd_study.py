import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description="Summarize the GPT-2 FDD seed study")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    train_seeds = (1024, 2048, 4096)
    eval_seeds = (10, 20, 30, 40, 50)
    datasets = {"dolly": "dolly", "self_inst": "self_inst", "vicuna": "vicuna", "sinst": "sinst_11_"}
    metric_names = ("exact_match", "rougeL", "lm_loss", "mean_gen_length")
    generation = {}
    for method in ("fdd", "contra_fdd"):
        generation[method] = {}
        for benchmark, data_name in datasets.items():
            per_seed = {}
            for seed in train_seeds:
                path = args.results / "generation" / method / str(seed) / f"{data_name}-512"
                scores = []
                for eval_seed in eval_seeds:
                    with (path / str(eval_seed) / "metrics.json").open() as f:
                        scores.append(json.load(f))
                per_seed[seed] = {name: mean(score[name] for score in scores) for name in metric_names}
            generation[method][benchmark] = {
                "mean_by_train_seed": per_seed,
                "mean_across_train_seeds": {name: mean(run[name] for run in per_seed.values()) for name in metric_names},
                "sample_std_across_train_seeds": {name: stdev(run[name] for run in per_seed.values()) for name in metric_names},
            }

    mse_seed = 1024
    layer_changes = {}
    for benchmark in datasets:
        relative_path = Path("layer_changes") / str(mse_seed) / benchmark / "layer_changes.json"
        with (args.results / relative_path).open() as f:
            scores = json.load(f)
        layer_changes[benchmark] = {
            "details": str(relative_path),
            "num_response_tokens": scores["num_response_tokens"],
            "mean_transition_mse": scores["mean_transition_mse"],
        }
    summary = {
        "train_seeds": train_seeds,
        "generation_eval_seeds": eval_seeds,
        "generation": generation,
        "layer_changes": {"train_seed": mse_seed, "datasets": layer_changes},
    }
    temporary = args.results / "summary.json.tmp"
    with temporary.open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    temporary.replace(args.results / "summary.json")


if __name__ == "__main__":
    main()
