"""W4 probe / segmentation robustness (sgl.robustness) on the offline P-ALIGN sample: segmentations,
signals and training files, rank stability, logged training with the tiny student, evaluation records."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sgl.data.prepare import build_record
from sgl.robustness import common as C
from sgl.robustness import runs, segment, signals, stability
from sgl.segment import split
from sgl.segment.sentence import split_math_aware

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"
TOKENIZER = str(ASSETS / "tokenizer")


def test_math_aware_split_keeps_formulas_and_list_numbers_whole():
    text = (
        "We get $x = 3. Then y$ inside. Next we use \\[a.\nB\\] here. Then:\n1. First item. "
        "Second sentence.\n**2.** Bold item."
    )
    pieces = split_math_aware(text)
    assert "".join(pieces) == text
    assert not any(p.startswith("Then y$") or p.startswith("B\\]") for p in pieces)
    assert not any(p.startswith("First") or p.startswith("Bold") for p in pieces)
    assert "Second sentence.\n**2.** Bold item." in pieces
    sentence = split(text, "sentence")
    assert any(p.startswith("Then y$") for p in sentence) and any(p.startswith("First") for p in sentence)


def test_math_aware_stray_dollar_does_not_swallow_the_trace():
    text = "It costs $5. Then we stop.\n\nNext paragraph. Done."
    assert split_math_aware(text) == split(text, "sentence")


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    root = tmp_path_factory.mktemp("w4")
    rows = [json.loads(line) for line in (ASSETS / "palign_sample.jsonl").open()]
    records = [build_record(tokenizer, r["input"], r["output"], 32768) for r in rows]
    for index, record in enumerate(records):
        record["id"] = index
    data = root / "train-segmented.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in records))
    for seg in C.SEGMENTATIONS:
        segment.main(
            [
                "--data-path",
                str(data),
                "--seg",
                seg,
                "--tokenizer",
                TOKENIZER,
                "--output",
                str(root / f"segmented-{seg}.jsonl"),
            ]
        )

    rng = random.Random(0)
    out = root / "results" / "w4"

    def fake_scores(probe, seg, scale):
        """Gains with a consistent log p(answer) trajectory and the meta file answer_gain writes."""
        segmented = [json.loads(x) for x in (root / f"segmented-{seg}.jsonl").open()]
        gains, logp, longest = {}, {}, {}
        for r in segmented:
            u = [rng.gauss(0, scale) for _ in r["steps"]]
            gains[str(r["id"])] = u
            logp[str(r["id"])] = list(np.cumsum([-5.0] + u))
            longest[str(r["id"])] = r["response_token_span"][1] + 20
        raw = out / "signals" / C.signal_name(probe, seg) / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "gains.json").write_text(json.dumps(gains))
        (raw / "logp.json").write_text(json.dumps(logp))
        (raw / "meta.json").write_text(
            json.dumps(
                {
                    "probe": C.PROBES.get(probe),
                    "probe_token_ids": [1, 2],
                    "max_probe_context_len": longest,
                    "seconds": 1.5,
                }
            )
        )
        return raw

    for probe, seg in C.SCORED:
        fake_scores(probe, seg, 1.0)

    def build(probe, seg, *extra):
        members = C.PAVG_MEMBERS if probe == "Pavg" else (probe,)
        raws = [out / "signals" / C.signal_name(p, seg) / "raw" for p in members]
        signals.main(
            [
                "--data-path",
                str(root / f"segmented-{seg}.jsonl"),
                "--probe",
                probe,
                "--seg",
                seg,
                "--tokenizer",
                TOKENIZER,
                "--out-dir",
                str(out / "signals" / C.signal_name(probe, seg)),
                "--dataset-output",
                str(root / f"train-w4-{probe}-{seg}.jsonl"),
                "--gains",
                *[str(r / "gains.json") for r in raws],
                "--logp",
                *[str(r / "logp.json") for r in raws],
                "--meta",
                *[str(r / "meta.json") for r in raws],
                *extra,
            ]
        )

    reference = out / "signals" / C.signal_name("P0", "S0") / "raw" / "gains.json"
    build("P0", "S0", "--reference-gains", str(reference), "--uniform-output", str(root / "train-w4-uniform.jsonl"))
    for probe, seg in C.VARIANTS:
        if (probe, seg) != ("P0", "S0"):
            build(probe, seg)
    stability.main(["--root", str(out), "--n-random", "10"])
    return root, out, records


def test_segmentations_cover_every_response_token(workspace):
    root, _, records = workspace
    for seg in C.SEGMENTATIONS:
        check = json.loads(Path(f"{root}/segmented-{seg}.jsonl.check.json").read_text())
        assert check["n_traces_with_coverage_errors"] == 0 and check["stop_token_outside_steps"]
    assert json.loads((root / "segmented-S0.jsonl.check.json").read_text())["s0_regression_mismatched_traces"] == 0
    s2a = [json.loads(x) for x in (root / "segmented-S2a.jsonl").open()]
    assert all(s["token_end"] - s["token_start"] <= 64 for r in s2a for s in r["steps"])
    counts = {
        seg: json.loads((root / f"segmented-{seg}.jsonl.check.json").read_text())["n_steps"] for seg in C.SEGMENTATIONS
    }
    assert counts["S2b"] < counts["S2a"]  # S1 vs S0 depends on the data (the sample has more paragraphs)


def test_signals_weights_and_training_file(workspace):
    root, out, records = workspace
    directory = out / "signals" / C.signal_name("P0", "S0")
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["failures"] == [] and summary["checks"]["token_coverage_ok"]
    assert summary["checks"]["max_abs_budget_minus_1"] < 1e-9
    assert summary["checks"]["max_abs_sum_u_minus_bK_b0"] < 1e-9
    assert summary["checks"]["regression_vs_reference_gains"]["max_abs_diff"] == 0
    assert "default_reference" in summary["checks"]
    steps = pd.read_parquet(directory / "steps.parquet")
    for column in (
        "b",
        "u",
        "z",
        "v",
        "clipped",
        "r",
        "w_hat",
        "w",
        "region",
        "has_final_boxed",
        "contains_answer_exact",
        "contains_answer_normalized",
        "char_start",
        "text",
    ):
        assert column in steps
    assert set(steps.region) <= {"prefix", "continuation", "crosses_marker", "no_marker"}
    traces = pd.read_parquet(directory / "traces.parquet")
    assert (traces.B - 1).abs().max() < 1e-9 and (traces.N > 0).all()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["probe"][0]["token_ids"] == [1, 2] and manifest["segmentation"]["kind"] == "rule"

    # the training file is what sgl.allocation.build writes for ALERD with these gains
    from sgl.allocation.build import build_example

    gains = json.loads((directory / "raw" / "gains.json").read_text())
    row = json.loads((root / "train-w4-P0-S0.jsonl").read_text().splitlines()[0])
    example, _ = build_example(
        records[0],
        [1.0] * len(records[0]["steps"]),
        gains["0"],
        1.0,
        "iwc-stable",
        C.TEMPERATURE,
        C.INTERPOLATION,
        C.CLIP,
        C.EPSILON,
    )
    assert row["loss_mask"] == example["loss_mask"]
    np.testing.assert_allclose(row["loss_weights"], example["loss_weights"], rtol=0, atol=1e-12)
    end = records[0]["response_token_span"][1]
    assert set(row["token_tags"][end:]) == {signals.TAG_EOS}
    assert all(t & signals.TAG_STEP for t in row["token_tags"][records[0]["response_token_span"][0] : end])
    uniform = json.loads((root / "train-w4-uniform.jsonl").read_text().splitlines()[0])
    assert set(uniform["loss_weights"][records[0]["response_token_span"][0] :]) == {1.0}


def test_pavg_is_the_mean_of_the_four_probes(workspace):
    _, out, _ = workspace
    steps = {
        p: pd.read_parquet(out / "signals" / C.signal_name(p, "S0") / "steps.parquet")
        for p in (*C.PAVG_MEMBERS, "Pavg")
    }
    mean = np.mean([steps[p].u.to_numpy() for p in C.PAVG_MEMBERS], axis=0)
    np.testing.assert_allclose(steps["Pavg"].u.to_numpy(), mean)


def test_rank_stability(workspace):
    _, out, _ = workspace
    summary = pd.read_csv(out / "analysis" / "rank_stability_summary.csv")
    assert set(summary.variant) == {"P0", "P1", "P2", "P3", "Pavg", "S1", "S2a", "S2b", "S3"}
    pavg = pd.read_parquet(out / "analysis" / "rank_stability" / "Pavg__S0.parquet")
    assert pavg.spearman_rho.median() > 0.2  # Pavg contains P0 itself
    s3 = pd.read_parquet(out / "analysis" / "rank_stability" / "P0__S3.parquet")
    assert s3.token_spearman_rho.between(-1, 1).all() and s3.top10pct_mass_overlap.between(0, 1).all()
    assert "Random" in (out / "analysis" / "table_r_w4a.md").read_text()


def test_jaccard_and_rho_helpers():
    assert stability.jaccard({0, 1, 2}, {1, 2, 3}) == pytest.approx(0.5)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert stability.rho(x, x) == pytest.approx(1) and stability.rho(x, -x) == pytest.approx(-1)
    assert stability.top_mass_overlap(np.arange(100.0), np.arange(100.0)) == pytest.approx(1)


@pytest.mark.regression
def test_logged_w4_training_run(workspace, tmp_path_factory):
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from sgl.training import train

    root, out, _ = workspace
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model_dir = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8192,
        )
    ).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    run_dir = out / "runs" / C.run_id("alg", "P1", "S0", 42)
    train.main(
        [
            "--model-name",
            str(model_dir),
            "--data-path",
            str(root / "train-w4-P1-S0.jsonl"),
            "--output-dir",
            str(run_dir / "checkpoints/epoch2"),
            "--run-log-dir",
            str(run_dir / "train"),
            "--epochs",
            "2",
            "--per-device-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "4",
            "--learning-rate",
            "1e-3",
            "--min-learning-rate",
            "1e-4",
            "--warmup-ratio",
            "0.1",
            "--save-strategy",
            "no",
            "--model-dtype",
            "float32",
            "--epoch-nll",
            "--epoch-checkpoints-dir",
            str(run_dir / "checkpoints"),
            "--param-check-step",
            "2",
        ]
    )
    lines = [json.loads(x) for x in (run_dir / "train/train_log.jsonl").read_text().splitlines()]
    assert len(lines) == 8
    for line in lines:
        assert abs(line["sum_w_over_Z"] - 1) < 1e-6 and line["w_max"] >= 1 >= line["w_mean"] - 1e-6
        for key in ("nll_prefix", "nll_cont", "nll_junction", "nll_answer_steps", "nll_non_answer_steps", "nll_eos"):
            assert key in line
    epochs = [json.loads(x) for x in (run_dir / "train/epoch_log.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in epochs] == [1, 2]
    assert epochs[1]["dataset_nll_all"] < epochs[0]["dataset_nll_all"] + 1
    assert "dataset_nll_answer_steps" in epochs[0] and epochs[0]["mean_loss_weighted"] > 0
    assert (run_dir / "checkpoints/epoch1/config.json").is_file() and not (
        run_dir / "checkpoints/epoch2/checkpoint-1"
    ).exists()
    health = [json.loads(x) for x in (run_dir / "train/health.jsonl").read_text().splitlines()]
    assert health[0]["global_step"] == 2 and health[0]["changed_fraction"] > 0.5


def _write_raw(directory: Path, seed: int):
    rng = random.Random(seed)
    directory.mkdir(parents=True)
    sizes = {"aime25": 30, "aime24": 30, "amc12": 83, "math500": 500}
    for name in C.BENCHMARK_ORDER:
        with (directory / f"{name}.jsonl").open("w") as handle:
            for problem in range(sizes[name]):
                generations = [
                    {"text": f"so \\boxed{{{7 if rng.random() < 0.5 else 8}}}", "finish_reason": "stop", "n_tokens": 10}
                    for _ in range(3)
                ]
                handle.write(
                    json.dumps({"id": problem, "gold": "7", "task_type": "math", "generations": generations}) + "\n"
                )


def test_eval_records_checklist_and_master_table(workspace, tmp_path):
    pytest.importorskip("math_verify")
    _, out, _ = workspace
    root = tmp_path / "w4"
    run_dir = root / "runs" / C.run_id("alg", "P1", "S0", 42)
    _write_raw(run_dir / "raw", 0)
    runs.main(["eval", "--run-dir", str(run_dir)])
    samples = [json.loads(x) for x in (run_dir / "eval/samples.jsonl").read_text().splitlines()]
    assert len(samples) == C.N_EVAL_ROWS
    for key in (
        "sampling_seed",
        "generation",
        "num_tokens",
        "finish_reason",
        "correct_mathverify",
        "correct_oat",
        "correct",
        "truncated",
        "repetition_loop",
        "emits_marker",
        "has_boxed",
        "reference_answer",
    ):
        assert key in samples[0]
    runs.main(
        ["finalize", "--run-dir", str(run_dir), "--signals-dir", str(out / "signals" / C.signal_name("P1", "S0"))]
    )
    check = json.loads((run_dir / "run_check.json").read_text())
    assert check["samples_complete"] and check["metrics_recomputable"] and not check["step_log_present"]
    runs.main(["master", "--root", str(root)])
    master = pd.read_csv(root / "analysis/results_master.csv")
    assert master.loc[0, "probe"] == "P1" and 0 <= master.loc[0, "average_p1"] <= 100
    assert "ALG, P1" in (root / "analysis/table_r_w4b.md").read_text()


def test_repetition_loop_detector():
    assert runs.repetition_loop("so " + "we get x = 2 and then " * 6)
    assert not runs.repetition_loop("A normal ending with \\boxed{7}.")
