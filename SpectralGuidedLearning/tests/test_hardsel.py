"""ALG-hard (sgl.hardsel) on the offline P-ALIGN sample: selection and its checks, logged training with a
tiny student, post-hoc NLL, run checklist and the report built from fake generations."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sgl.allocation.build import build_example
from sgl.data.prepare import build_record
from sgl.hardsel import common as C
from sgl.hardsel import nll_eval, report, runs, select
from sgl.robustness import runs as w4_runs

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"
TOKENIZER = str(ASSETS / "tokenizer")


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    root = tmp_path_factory.mktemp("hard")
    rows = read_jsonl(ASSETS / "palign_sample.jsonl")
    records = [build_record(tokenizer, r["input"], r["output"], 32768) for r in rows]
    for index, record in enumerate(records):
        record["id"] = index
    data = root / "train-segmented.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in records))
    rng = random.Random(0)
    gains = {str(r["id"]): [rng.gauss(0, 1) for _ in r["steps"]] for r in records}
    (root / "gains.json").write_text(json.dumps(gains))
    with (root / "train-e7.jsonl").open("w") as handle:
        for r in records:
            example, _ = build_example(r, [1.0] * len(r["steps"]), gains[str(r["id"])], 1.0, "iwc-stable", 1.0, 0.5,
                                       2.0, 1e-8)
            handle.write(json.dumps(example) + "\n")
    select.main(["--data-path", str(data), "--tokenizer", TOKENIZER, "--data-dir", str(root / "data"),
                 "--root", str(root / "hard"), "--gains", str(root / "gains.json"),
                 "--reference-dataset", str(root / "train-e7.jsonl")])
    return root, records, gains


def test_selection_rules(workspace):
    root, records, gains = workspace
    out = root / "hard" / "selection"
    for arm in C.ARMS:
        assert json.loads((out / arm / "select_check.json").read_text())["n_failures"] == 0
        assert json.loads((out / arm / "manifest.json").read_text())["budget_source"] == "fixed"
    check = json.loads((out / "alg-h59" / "select_check.json").read_text())["all_ones_mask"]
    assert check["reference_dataset_enforced"] and check["max_abs_diff_vs_reference_dataset"] < 1e-6

    h59, r59, h30 = (pd.read_parquet(out / arm / "steps.parquet") for arm in ("h59", "r59", "h30"))
    for trace_id, group in h59.groupby("trace_id"):
        length = group.ell.sum()
        for table, p in ((h59, 0.59), (h30, 0.30)):
            kept = table[(table.trace_id == trace_id) & table.kept]
            assert p * length - 1e-9 <= kept.ell.sum() <= p * length + group.ell.max()
        assert group[group.kept].u.min() >= group[~group.kept].u.max() if (~group.kept).any() else True
        random_kept = r59[(r59.trace_id == trace_id) & r59.kept].ell.sum()
        assert 0 <= random_kept - group[group.kept].ell.sum() <= group.ell.max()

    for arm in C.ARMS:
        for record, row in zip(records, read_jsonl(root / "data" / f"train-hard-{arm}.jsonl")):
            start, end = record["response_token_span"]
            mask, weights, tags = np.array(row["loss_mask"]), np.array(row["loss_weights"]), np.array(row["token_tags"])
            assert row["input_ids"] == record["input_ids"] and not mask[:start].any()
            assert mask[end:].all() and (weights[end:] == 1).all()  # the stop token
            kept = (tags & C.TAG_KEPT) > 0
            dropped = (tags & C.TAG_DROPPED) > 0
            assert ((kept | dropped)[start:end]).all() and not (kept & dropped).any()
            assert (mask[start:end] == kept[start:end]).all() and (weights[mask == 0] == 0).all()
            assert abs(weights.sum() / mask.sum() - 1) < 1e-9
            if arm == "h59":
                assert (((tags & C.TAG_H59_DROPPED) > 0) == dropped).all()

    stats = json.loads((out / "h59" / "selection_stats.json").read_text())
    assert 0.59 <= stats["kept_token_frac"] < 1 and len(stats["kept_token_frac_by_position_decile"]) == 10
    assert stats["loss_coefficient_share_reference"]["uniform"]["prefix"] > 0


def test_sgl_gate_budget_and_w4_gains(workspace, tmp_path):
    root, records, gains = workspace
    from sgl.selection.build_masks import emit_masked_dataset

    rng = random.Random(1)
    strengths = {r["id"]: [rng.random() for _ in r["steps"]] for r in records}
    emit_masked_dataset(records, strengths, 0.6, tmp_path / "train-spectral.jsonl")
    # no answer_gain.json: the gains come from a W4 P0/S0 signal directory
    signals = tmp_path / "w4"
    signals.mkdir()
    pd.DataFrame([{"trace_id": r["id"], "k": k + 1, "u": u} for r in records
                  for k, u in enumerate(gains[str(r["id"])])]).to_parquet(signals / "steps.parquet")
    pd.DataFrame([{"trace_id": r["id"], "has_gain": True} for r in records]).to_parquet(signals / "traces.parquet")
    (signals / "summary.json").write_text(json.dumps({"failures": []}))
    arguments = ["--data-path", str(root / "train-segmented.jsonl"), "--tokenizer", TOKENIZER,
                 "--data-dir", str(tmp_path / "data"), "--root", str(tmp_path / "hard"),
                 "--gains", str(tmp_path / "none.json"), "--w4-signals", str(signals),
                 "--sgl-mask", str(tmp_path / "train-spectral.jsonl")]
    select.main(arguments)
    manifest = json.loads((tmp_path / "hard/selection/h59/manifest.json").read_text())
    assert manifest["budget_source"] == "sgl_gate" and manifest["label"] == "H-SGL"
    assert manifest["gain"]["source"] == "w4_rescored_P0_S0"
    gate = {row["id"]: row["loss_mask"] for row in read_jsonl(tmp_path / "train-spectral.jsonl")}
    traces = pd.read_parquet(tmp_path / "hard/selection/h59/traces.parquet").set_index("trace_id")
    for record in records:
        start, end = record["response_token_span"]
        assert traces.kept_tokens[record["id"]] >= sum(gate[record["id"]][start:end])
    assert (tmp_path / "hard/analysis/selection_overlap.json").is_file()

    (signals / "summary.json").write_text(json.dumps({"failures": ["regression"]}))
    with pytest.raises(SystemExit):
        select.main(arguments)


def test_decision_rule():
    assert C.apply_rule(2.0, 1.0, (0.5, 3.5)) == "soft_better"
    assert C.apply_rule(2.0, 1.0, (-0.5, 3.5)) == "leans_soft"
    assert C.apply_rule(1.5, 2.0, (0.5, 2.5)) == "leans_soft"
    assert C.apply_rule(0.4, 1.0, (-1, 2)) == "equivalent"
    assert C.apply_rule(-1.5, 1.0, (-3, -0.5)) == "hard_better"
    assert [C.needs_group_c(r) for r in C.DECISION_RULE["rows"]] == [False, True, True, True]


def fake_raw(directory: Path, rate: float, seed: int) -> None:
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    for name, n in (("aime25", 6), ("aime24", 6), ("amc12", 8), ("math500", 10)):
        with (directory / f"{name}.jsonl").open("w") as handle:
            for i in range(n):
                texts = ["\\boxed{7}" if rng.random() < rate else "\\boxed{8}" for _ in range(3)]
                handle.write(json.dumps({"id": i, "gold": "7", "task_type": "math", "generations": [
                    {"text": t, "finish_reason": "stop", "n_tokens": 5} for t in texts]}) + "\n")


@pytest.mark.regression
def test_logged_training_checklist_and_report(workspace, tmp_path_factory):
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from sgl.training import train

    root, records, _ = workspace
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model_dir = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=8192)
                     ).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    hard = root / "hard"
    for arm in ("h59", "alg-h59", "r59"):
        run = hard / "runs" / C.run_id(arm, 42)
        train.main([
            "--model-name", str(model_dir), "--data-path", str(root / "data" / f"train-hard-{arm}.jsonl"),
            "--output-dir", str(run / "checkpoints/epoch3"), "--epochs", "3", "--learning-rate", "1e-3",
            "--min-learning-rate", "1e-4", "--warmup-ratio", "0.1", "--per-device-batch-size", "1",
            "--gradient-accumulation-steps", "2", "--seed", "42", "--save-strategy", "no",
            "--model-dtype", "float32", "--run-log-dir", str(run / "train"), "--selection-log", "--epoch-nll",
            "--epoch-checkpoints-dir", str(run / "checkpoints"), "--param-check-step", "2",
        ])
        steps = read_jsonl(run / "train/train_log.jsonl")
        stats = json.loads((hard / "selection" / arm / "selection_stats.json").read_text())
        for line in steps:
            for key in ("step", "lr", "loss", "nll_kept", "nll_all_unweighted", "n_tok_total", "n_tok_kept", "Z",
                        "kept_frac_batch", "sum_w_over_Z", "grad_norm", "grad_clipped", "step_time_s"):
                assert key in line, key
            assert abs(line["sum_w_over_Z"] - 1) < 1e-5 and line["Z"] == line["n_tok_kept"] + 2  # 2 stop tokens
            if arm != "alg-h59":
                assert line["w_max"] == 1 and abs(line["loss"] - line["nll_kept"]) < 1e-4
        kept = sum(x["n_tok_kept"] for x in steps) / sum(x["n_tok_total"] for x in steps)
        assert abs(kept - stats["kept_token_frac"]) < 1e-9
        epochs = read_jsonl(run / "train/epoch_log.jsonl")
        assert [x["epoch"] for x in epochs] == [0, 1, 2, 3]
        for key in ("dataset_nll_all", "dataset_nll_kept", "dataset_nll_dropped", "dataset_nll_h59_dropped"):
            assert all(key in x for x in epochs)
        assert (run / "checkpoints/epoch1/config.json").is_file()
        assert json.loads((run / "train/data_check.json").read_text())["n_mismatched_tokens"] == 0

        fake_raw(run / "raw", {"h59": 0.4, "alg-h59": 0.6, "r59": 0.3}[arm], 1)
        w4_runs.main(["eval", "--run-dir", str(run)])
        runs.main(["--run-dir", str(run), "--selection-root", str(hard / "selection"), "--n-samples", "90"])
        check = json.loads((run / "run_check.json").read_text())
        failed = {k for k, v in check.items() if v is False} - {"git_clean", "all_ok", "params_changed_at_step_64"}
        assert not failed, failed

    # post-hoc NLL of a "uniform" checkpoint, then the report with fake E0 / E7 generations
    nll_eval.main(["--data-path", str(root / "data/train-hard-h59.jsonl"), "--tag", "uniform__seed42",
                   "--checkpoint", f"0={model_dir}", "--checkpoint", f"3={model_dir}", "--checkpoint", "2=missing",
                   "--output", str(hard / "posthoc_nll/uniform__seed42/epoch_nll.json")])
    fake_raw(root / "e0/raw", 0.45, 2)
    fake_raw(root / "e7/raw", 0.8, 3)
    common = ["--root", str(hard), "--w4-root", str(root / "w4"), "--uniform-run", str(root / "e0"),
              "--alg-run", str(root / "e7")]
    report.main(["rule", *common])
    report.main(["report", *common])
    analysis = hard / "analysis"
    decision = json.loads((analysis / "decision.json").read_text())
    assert decision["rule_written_at"] and decision["conclusion"] == "soft_better" and not decision["needs_group_c"]
    assert decision["pass@1"]["sigma_source"] == "pending" and "h59_minus_r59" in decision["secondary"]
    with pytest.raises(SystemExit) as exit_info:
        report.main(["needs-c", "--root", str(hard)])
    assert exit_info.value.code == 1
    for name in ("table_r_h1.md", "table_r_h2.md", "fig_h1_dropped_nll.png", "fig_h2_kept_by_position.pdf",
                 "results_master.csv"):
        assert (analysis / name).stat().st_size > 0, name
    assert "H59" in (analysis / "table_r_h1.md").read_text()
    paired = json.loads((hard / "runs" / C.run_id("h59", 42) / "eval/paired_vs_baselines.json").read_text())
    assert paired["alg"]["either"]["delta_avg_p1"] < 0 and "r59" in paired

    # a weak ALG: the rule asks for group C
    fake_raw(root / "e7/raw", 0.4, 1)
    report.main(["report", *common])
    report.main(["rule", *common])  # the rule block is written once
    assert json.loads((analysis / "decision.json").read_text())["needs_group_c"]
    with pytest.raises(SystemExit) as exit_info:
        report.main(["needs-c", "--root", str(hard)])
    assert exit_info.value.code == 0
