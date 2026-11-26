"""Post-hoc NLL by region of existing checkpoints (uniform / ALG), on the same tokens as the hard runs.

--data-path is train-hard-h59.jsonl: its token_tags carry the regions and the tokens H59 drops. Every
--checkpoint is <epoch>=<dir or model>; missing local directories are skipped with a warning.
"""

import argparse
from pathlib import Path

from sgl.hardsel import common as C
from sgl.training.dataset import MaskedSFTDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tag", required=True, help="name of this model / seed, e.g. uniform__seed43")
    parser.add_argument("--checkpoint", action="append", required=True, help="<epoch>=<checkpoint dir>")
    parser.add_argument("--output", required=True, help="epoch_nll.json")
    parser.add_argument("--limit", type=int, help="first N traces only (smoke test)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM

    from sgl.hardsel.runlog import region_nll

    dataset = MaskedSFTDataset(args.data_path)
    if args.limit:
        dataset.records = dataset.records[: args.limit]
    out = Path(args.output)
    done = C.read_json(out)["epochs"] if out.is_file() else {}
    for entry in args.checkpoint:
        epoch, directory = entry.split("=", 1)
        if epoch in done:
            continue
        if not Path(directory, "config.json").is_file():
            print(f"WARNING {args.tag} epoch {epoch}: no checkpoint at {directory}, skipped")
            continue
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            directory, dtype=torch.bfloat16 if device == "cuda" else torch.float32, attn_implementation="sdpa"
        ).to(device)
        model.config.use_cache = False
        done[epoch] = {**region_nll(model, dataset), "checkpoint": directory}
        print(f"{args.tag} epoch {epoch}: nll all {done[epoch]['dataset_nll_all']:.4f}, "
              f"h59-dropped {done[epoch].get('dataset_nll_h59_dropped', float('nan')):.4f}")
        C.write_json(out, {"tag": args.tag, "epochs": done,
                           "provenance": C.provenance({"data": args.data_path})})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
