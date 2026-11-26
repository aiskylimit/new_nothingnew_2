"""Positional-bias experiment (sgl.positional) on the offline P-ALIGN sample: tables and sanity checks,
analyses, logged training with the tiny student, evaluation report and the decision rule."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sgl.data.prepare import build_record
from sgl.positional import analysis, common, decide, evalreport, tables

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS / "tokenizer"))
    root = tmp_path_factory.mktemp("positional")
    rows = [json.loads(line) for line in (ASSETS / "palign_sample.jsonl").open()]
    records = [build_record(tokenizer, r["input"], r["output"], 32768) for r in rows]
    for index, record in enumerate(records):
        record["id"] = index
    rng = random.Random(0)
    gains = {str(r["id"]): [rng.gauss(0, 1) for _ in r["steps"]] for r in records}
    (root / "train-segmented.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / "gains.json").write_text(json.dumps(gains))
    out = root / "results" / "positional"
    tables.main(["--data-path", str(root / "train-segmented.jsonl"), "--gains", str(root / "gains.json"),
                 "--tokenizer", str(ASSETS / "tokenizer"), "--output-dir", str(out)])
    analysis.main(["--root", str(out)])
    return root, out, records, gains


def test_weights_and_sanity(workspace):
    root, out, records, _ = workspace
    sanity = json.loads((out / "analysis/sanity_weights.json").read_text())
    assert sanity["failures"] == []
    assert sanity["r6"]["r6_monotonic_violations"] == 0 and sanity["r8"]["r8_multiset_preserved"]
    steps = pd.read_parquet(out / "tables/step_table.parquet")
    for scheme in tables.SCHEMES:
        mass = (steps.ell * steps[f"w_{scheme}"]).groupby(steps.trace_id).sum()
        np.testing.assert_allclose(mass, steps.ell.groupby(steps.trace_id).sum(), rtol=1e-9)
    assert (steps.n_tok_prefix + steps.n_tok_cont == steps.ell).all()
    assert steps.groupby("trace_id").has_marker.sum().eq(1).all()
    # the training file matches what sgl.allocation.build writes for ALERD, except for the weights
    row = json.loads((root / "train-positional-r6.jsonl").read_text().splitlines()[0])
    end = records[0]["response_token_span"][1]
    assert row["loss_weights"][end:] == [1.0] * (len(row["input_ids"]) - end)
    assert set(row["token_source"]) == {0, 1, 2}
    assert tables.dataset_max_diff(str(root / "train-positional-r6.jsonl"), records, steps.assign(w_alerd=steps.w_r6)) == 0


def test_alerd_weights_equal_the_published_pipeline(workspace):
    from sgl.allocation.build import build_example

    _, out, records, gains = workspace
    steps = pd.read_parquet(out / "tables/step_table.parquet")
    record = records[0]
    example, _ = build_example(record, [1.0] * len(record["steps"]), gains["0"], 1.0, "iwc-stable",
                               common.TEMPERATURE, common.INTERPOLATION, common.CLIP, common.EPSILON)
    ours = next(tables.training_records(records[:1], steps, "w_alerd"))
    assert ours["loss_mask"] == example["loss_mask"]
    np.testing.assert_allclose(ours["loss_weights"], example["loss_weights"], rtol=0, atol=1e-12)


def test_analyses(workspace):
    _, out, *_ = workspace
    a1 = json.loads((out / "analysis/a1_summary.json").read_text())
    assert len(a1["R_by_decile"]) == 10
    mass = pd.read_csv(out / "analysis/a1_decile_mass.csv").query("scheme == 'alerd'")
    assert mass.mass_unif.sum() == pytest.approx(1) and mass.mass_scheme.sum() == pytest.approx(1)
    assert (mass.R_ci_lo <= mass.R_ci_hi).all()
    a2 = json.loads((out / "analysis/a2_summary.json").read_text())
    assert a2["median_rho_index"] == pytest.approx(a2["median_rho"])
    for name in ("a1_decile_mass", "a1b_profiles", "a2_rho_hist", "a3_argmax_pos_hist", "a4_by_segment"):
        assert (out / f"analysis/{name}.png").is_file() and (out / f"analysis/{name}.pdf").is_file()
    assert 0 < json.loads((out / "analysis/a4_summary.json").read_text())["prefix_token_share_unif"] < 1


def test_spearman_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    x, y = rng.integers(0, 5, 40), rng.random(40)
    assert analysis.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)


@pytest.mark.regression
def test_logged_training_run(workspace, tmp_path_factory):
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from sgl.training import train

    root, out, *_ = workspace
    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS / "tokenizer"))
    model_dir = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=8192,
    )).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    run_dir = out / "runs" / "r6"
    train.main([
        "--model-name", str(model_dir), "--data-path", str(root / "train-positional-r6.jsonl"),
        "--output-dir", str(run_dir / "checkpoint"), "--run-log-dir", str(run_dir), "--epochs", "1",
        "--per-device-batch-size", "1", "--gradient-accumulation-steps", "4", "--learning-rate", "1e-3",
        "--min-learning-rate", "1e-4", "--warmup-ratio", "0.1", "--save-strategy", "no",
        "--model-dtype", "float32",
    ])
    check = json.loads((run_dir / "data_check.json").read_text())
    assert check["n_tokens_weight_mismatch"] == 0 and len(check["samples"]) == 3
    lines = [json.loads(x) for x in (run_dir / "train_log.jsonl").read_text().splitlines()]
    assert len(lines) == 4  # 16 traces / 4
    for line in lines:
        assert line["grad_norm"] > 0 and line["lr"] is not None
        assert 0.8 < line["sum_w_over_Z"] < 1.2 and line["nll_prefix"] > 0 and line["nll_cont"] > 0
    assert json.loads((run_dir / "train_summary.json").read_text())["n_optimizer_steps"] == 4
    assert json.loads((run_dir / "config.json").read_text())["hyperparameters"]["effective_batch"] == 4


def _write_raw(directory: Path, correct_share: float, seed: int):
    rng = random.Random(seed)
    directory.mkdir(parents=True)
    for name in common.BENCHMARK_ORDER:
        with (directory / f"{name}.jsonl").open("w") as handle:
            for problem in range(30):
                generations = [{"text": f"so \\boxed{{{7 if rng.random() < correct_share else 8}}}",
                                "finish_reason": "stop", "n_tokens": 10} for _ in range(3)]
                handle.write(json.dumps({"id": problem, "gold": "7", "task_type": "math",
                                         "generations": generations}) + "\n")


def test_eval_report_and_decision(workspace, tmp_path):
    pytest.importorskip("math_verify")
    _, out, *_ = workspace
    _write_raw(tmp_path / "base_raw", 0.4, 0)
    _write_raw(tmp_path / "r6_raw", 0.4, 1)
    evalreport.main(["report", "--raw-dir", str(tmp_path / "base_raw"), "--out-dir", str(out / "runs/palign/eval")])
    evalreport.main(["report", "--raw-dir", str(tmp_path / "r6_raw"), "--out-dir", str(out / "runs/r6/eval"),
                     "--baseline", f"palign={out / 'runs/palign/eval'}"])
    metrics = json.loads((out / "runs/r6/eval/metrics.json").read_text())
    assert metrics["average"]["pass@3"] >= metrics["average"]["pass@1"]
    paired = json.loads((out / "runs/r6/eval/paired_vs_baselines.json").read_text())["palign"]
    assert paired["delta_avg_p1_ci95_lo"] <= paired["delta_avg_p1"] <= paired["delta_avg_p1_ci95_hi"]
    decide.main(["--root", str(out)])
    decision = json.loads((out / "decision.json").read_text())
    assert decision["decided_at"] and decision["rule_row_triggered"]
    assert "Position-only" in (out / "report.md").read_text()


@pytest.mark.parametrize("rho,p1,p3,runs", [
    (0.1, 40.9, 52.0, []),             # small rho, about uniform
    (0.1, 39.0, 50.0, []),             # small rho, lower
    (0.1, 41.8, 53.0, []),             # above uniform, clearly below ALERD
    (0.4, 40.0, 50.0, ["r7"]),         # substantial rho
    (0.1, 42.5, 54.0, ["r7", "r8"]),   # about equal to ALERD
    (0.4, 42.5, 54.0, ["r7", "r8"]),
])
def test_decision_rule(rho, p1, p3, runs):
    assert decide.apply_rule(rho, p1, p3)[1] == runs


def test_decision_rule_says_when_the_table_does_not_cover_a_case():
    assert "NOT COVERED" in decide.apply_rule(0.1, 41.77, 54.5)[0]
