#!/usr/bin/env python3
"""Verify the pre-built GRIT artifacts (snapshot of fushinguyenex/GRIT-artifacts).

Offline replacement for the prepare stage. Checks, in order:
  1. every file listed in SHA256SUMS.json exists and matches its sha256,
  2. manifests: pinned base revision, float32 statistics, top-64, row counts,
  3. the local policy tokenizer fingerprint equals the one stored with the contexts,
  4. the projector covers every MLP Linear and is disjoint from KL/eval prompts,
  5. (optional, GPU) base-logit spot check: KL(local base || stored base top-64)
     on a few monitor contexts must be ~0, i.e. the local weights are the ones
     the statistics were computed from.
A stamp file records a passed verification so re-runs skip the slow hashing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grit.model_profiles import LARGE, profile_for_policy  # noqa: E402
from scripts.preservation_data import preservation_paths  # noqa: E402

T0 = time.time()


def tick(label):
    print(f"[{time.time() - T0:8.1f}s] {label}", flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(root, model_id):
    paths = preservation_paths(root / "data", root / "artifacts", model_id)
    return {
        "task_file": root / "data" / "task" / "task_train.parquet",
        "evaluation_file": root / "data" / "eval" / "evaluation_prompts.parquet",
        "preserve_file": Path(paths["preserve_file"]),
        "projector_contexts": Path(paths["projector_contexts"]),
        "projectors_path": Path(paths["projectors_path"]),
    }


def check_checksums(root, sums):
    files = sums["files"]
    for index, (relative, expected) in enumerate(sorted(files.items()), 1):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing artifact: {path}")
        size_gb = path.stat().st_size / 1e9
        tick(f"sha256 {index}/{len(files)} {relative} ({size_gb:.2f} GB)")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"sha256 mismatch for {relative}: {actual} != {expected}")


def check_manifests(paths, profile, *, top_k, dtype):
    monitor = json.loads((paths["preserve_file"].parent / "manifest.json").read_text())
    projector = json.loads((paths["projector_contexts"].parent / "manifest.json").read_text())
    for name, manifest in (("monitor", monitor), ("projector", projector)):
        if manifest["base_revision"] != profile.policy_revision:
            raise ValueError(f"{name} manifest base_revision {manifest['base_revision']} != {profile.policy_revision}")
        if manifest["generation_dtype"] != dtype:
            raise ValueError(f"{name} manifest generation_dtype {manifest['generation_dtype']} != {dtype}")
    if monitor["base_statistics_dtype"] != dtype or monitor["base_top_k"] != top_k:
        raise ValueError("monitor statistics must be top-64 float32")
    if monitor["rows"] != 1920 or projector["rows"] != 1000:
        raise ValueError(f"unexpected row counts: monitor={monitor['rows']} projector={projector['rows']}")
    tick(f"manifests ok: monitor {monitor['domain_counts']}, projector {projector['domain_counts']}, "
         f"full_vs_incremental={monitor['full_vs_incremental_max_abs_diff']:.2e}")
    return monitor


def check_tokenizer(model_path, monitor):
    from transformers import AutoTokenizer
    from scripts.preservation_data import tokenizer_fingerprint

    fingerprint = tokenizer_fingerprint(AutoTokenizer.from_pretrained(model_path))
    if fingerprint != monitor["tokenizer_sha256"]:
        raise ValueError(f"local tokenizer {fingerprint} != artifact tokenizer {monitor['tokenizer_sha256']}; "
                         "the downloaded model snapshot differs from the pinned revision")
    tick(f"tokenizer fingerprint ok ({fingerprint[:12]})")


def check_projectors(paths, profile):
    import torch
    from datasets import load_dataset
    from scripts.preservation_data import validate_monitoring_split

    tick("loading projectors (CPU)")
    payload = torch.load(paths["projectors_path"], map_location="cpu", weights_only=False, mmap=True)
    if payload.get("base_revision") != profile.policy_revision:
        raise ValueError(f"projector base_revision {payload.get('base_revision')} != {profile.policy_revision}")
    names = sorted(payload["projectors"])
    if not names or any("mlp" not in name for name in names):
        raise ValueError("projector artifact must cover MLP Linear modules only")
    preserve = load_dataset("parquet", data_files=str(paths["preserve_file"]), split="train")
    evaluation = load_dataset("parquet", data_files=str(paths["evaluation_file"]), split="train")
    validate_monitoring_split(payload, preserve, evaluation)
    tick(f"projectors ok: {len(names)} modules, disjoint from {len(preserve)} KL and {len(evaluation)} eval prompts")
    return names


def check_base_logits(model_path, paths, *, contexts, max_length, top_k, tolerance):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.train_grit import base_kl_at_current_policy

    if not torch.cuda.is_available():
        raise RuntimeError("--logit-check needs a GPU")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tick("loading FP32 base model for the logit spot check")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa").to(device).eval()
    rows = load_dataset("parquet", data_files=str(paths["preserve_file"]), split="train")
    rows = [rows[index] for index in range(0, len(rows), max(len(rows) // contexts, 1))][:contexts]
    mean, peak = base_kl_at_current_policy(model, tokenizer, rows, max_length=max_length,
                                           top_k=top_k, device=device)
    del model
    torch.cuda.empty_cache()
    if mean > tolerance:
        raise ValueError(f"base KL to stored statistics {mean:.2e} > {tolerance:.0e}: "
                         "local weights differ from the pinned revision")
    tick(f"base logits ok: KL mean {mean:.2e}, max {peak:.2e} over {len(rows)} contexts")
    return {"kl_mean": mean, "kl_max": peak, "contexts": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", required=True,
                        help="Snapshot folder, e.g. .../prepared/Qwen--Qwen2.5-3B-Instruct")
    parser.add_argument("--model-path", required=True, help="Local policy snapshot (Qwen2.5-3B-Instruct)")
    parser.add_argument("--paths-json", help="Write the resolved trainer input paths here")
    parser.add_argument("--logit-check", type=int, default=0, metavar="N",
                        help="Also compare base logits on N monitor contexts (GPU); 0 disables")
    parser.add_argument("--logit-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-preserve-length", type=int, default=2304)
    parser.add_argument("--force", action="store_true", help="Ignore the verification stamp")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    profile = profile_for_policy(args.model_path)
    if profile != LARGE:
        raise SystemExit("verify_artifacts.py supports the pinned Qwen2.5-3B profile only")
    sums = json.loads((root / "SHA256SUMS.json").read_text())
    if sums.get("model_path") != profile.policy:
        raise SystemExit(f"artifact model {sums.get('model_path')} != {profile.policy}")
    paths = artifact_paths(root, profile.policy)
    if args.paths_json:
        Path(args.paths_json).write_text(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))

    stamp = root / ".grit_verified.json"
    sizes = {name: (root / name).stat().st_size for name in sums["files"] if (root / name).is_file()}
    if not args.force and stamp.is_file():
        previous = json.loads(stamp.read_text())
        if previous.get("sizes") == sizes and previous.get("model_path") == str(Path(args.model_path).resolve()) \
                and (not args.logit_check or previous.get("logits")):
            tick(f"artifacts already verified ({stamp}); skipping")
            return

    check_checksums(root, sums)
    monitor = check_manifests(paths, profile, top_k=64, dtype="float32")
    check_tokenizer(args.model_path, monitor)
    check_projectors(paths, profile)
    logits = None
    if args.logit_check:
        logits = check_base_logits(args.model_path, paths, contexts=args.logit_check,
                                   max_length=args.max_preserve_length, top_k=64,
                                   tolerance=args.logit_tolerance)
    try:
        stamp.write_text(json.dumps({"sizes": sizes, "model_path": str(Path(args.model_path).resolve()),
                                     "logits": logits}, indent=2))
    except OSError as error:
        tick(f"note: could not write stamp ({error}); verification will rerun next time")
    tick("ALL ARTIFACT CHECKS PASSED")


if __name__ == "__main__":
    main()
