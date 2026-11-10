import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from adaptive_graph_ftrl_redesign.config import load_config
from adaptive_graph_ftrl_redesign.cli import parser
from adaptive_graph_ftrl_redesign.experiment_config import prepare


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh"


def test_experiment_help_and_new_training_cli():
    result = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "DUEL_PROBLEMS=8 HELDOUT_PROBLEMS=128 WARMUP_STEPS=5" in result.stdout
    assert "scripts/train_adaptive_graph_ftrl.sh" in result.stdout
    args = parser().parse_args(["--control-pool-size=256", "--control-batch-size=32", "--eval-seed=123"])
    assert (args.control_pool_size, args.control_batch_size, args.eval_seed) == (256, 32, 123)


@pytest.mark.parametrize(
    "case",
    [
        "defaults",
        "custom",
        "train_failure",
        "eval_failure",
        "short_horizon",
        "hyperparameters",
        "cli_override",
        "yaml",
        "generated",
    ],
)
def test_experiment_delegates_to_existing_runner_with_storage_and_config(tmp_path, case):
    capture = tmp_path / "calls.jsonl"
    fake = tmp_path / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-m', 'adaptive_graph_ftrl_redesign.experiment_config']:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}] + args)\n"
        "if args[:1] == ['-c']:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}] + args)\n"
        "with Path(os.environ['TEST_CAPTURE']).open('a') as f:\n"
        "    f.write(json.dumps({'args': args, 'env': dict(os.environ)}) + '\\n')\n"
        "if args[:2] == ['-m', 'torch.distributed.run']:\n"
        "    print('mock training')\n"
        "    sys.exit(int(os.environ['TEST_TRAIN_STATUS']))\n"
        "if args[:2] == ['-m', 'adaptive_graph_ftrl_redesign.lifecycle']:\n"
        "    sys.exit(int(os.environ['TEST_TRAIN_STATUS'] != '0'))\n"
        "assert args[:2] == ['-m', 'adaptive_graph_ftrl_redesign.evaluate']\n"
        "print('mock evaluation')\n"
        "sys.exit(int(os.environ['TEST_EVAL_STATUS']))\n"
    )
    fake.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "DUEL_PROBLEMS",
            "HELDOUT_PROBLEMS",
            "WARMUP_STEPS",
            "MIN_VALID_CONTROLS",
            "MAX_STEPS",
            "SAVE_EVERY",
            "EVAL_STEPS",
            "EVAL_DATASETS",
            "CONFIG",
            "TRAIN_SEED",
            "EVAL_SEED",
            "EVAL_GPU_MEMORY_UTILIZATION",
            "FTRL_RHO",
            "FTRL_ETA",
            "FTRL_MIN_WEIGHT",
            "DELTA_RMS_DECAY",
            "DELTA_RMS_FLOOR",
            "DELTA_SCALED_CLIP",
            "WINDOW_STEPS",
            "ACTOR_WARMUP_WINDOWS",
            "CONTROLLER_LR",
            "EXPOSURE_TEMPERATURE",
            "LEARNING_RATE",
            "LAMBDA_GT",
            "GAMMA_DISCOUNT",
        }
    }
    run, work = tmp_path / "run with spaces", tmp_path / "runtime with spaces"
    work.mkdir()
    sentinel = work / "keep-existing-cache"
    sentinel.write_text("unchanged")
    env.update(
        PYTHON_BIN=str(fake),
        RUN_DIR=str(run),
        EVAL_WORK_DIR=str(work),
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        TEST_CAPTURE=str(capture),
        TEST_TRAIN_STATUS="7" if case == "train_failure" else "0",
        TEST_EVAL_STATUS="9" if case == "eval_failure" else "0",
    )
    if case == "custom":
        env.update(
            DUEL_PROBLEMS="32",
            HELDOUT_PROBLEMS="256",
            WARMUP_STEPS="0",
            MAX_STEPS="100",
            EVAL_STEPS="50,75,100",
            CUDA_VISIBLE_DEVICES="4,5,6,7",
            TRAIN_SEED="123",
            EVAL_SEED="456",
        )
    elif case == "short_horizon":
        env.update(MAX_STEPS="30")
    cli = []
    if case in {"hyperparameters", "cli_override"}:
        env.update(
            FTRL_RHO="0.8",
            DELTA_RMS_DECAY="0.7",
            DELTA_RMS_FLOOR="0.002",
            DELTA_SCALED_CLIP="0.5",
            WINDOW_STEPS="3",
            ACTOR_WARMUP_WINDOWS="1",
            CONTROLLER_LR="0.02",
            EXPOSURE_TEMPERATURE="0.1",
            LEARNING_RATE="2e-6",
            LAMBDA_GT="0.03",
            GAMMA_DISCOUNT="0.8",
        )
        if case == "cli_override":
            cli = ["--ftrl-rho=0.6"]
    elif case == "yaml":
        source = tmp_path / "custom.yaml"
        source.write_text("ftrl_rho: 0.73\ndelta_rms_floor: 0.003\n")
        env["CONFIG"] = str(source)
    elif case == "generated":
        env.pop("RUN_DIR")
        env.pop("EVAL_WORK_DIR")
        env["RUN_ROOT"] = str(tmp_path / "outputs")
    result = subprocess.run(
        ["bash", str(SCRIPT), *cli], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    expected_status = 7 if case == "train_failure" else 9 if case == "eval_failure" else 0
    assert result.returncode == expected_status, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == (2 if case == "train_failure" else 3)
    train = calls[0]["args"]
    options = parser().parse_args(train[train.index("adaptive_graph_ftrl_redesign.train") + 1 :])
    cfg = load_config(
        options.config,
        {k: v for k, v in vars(options).items() if k not in {"config", "resume", "validate_only"}},
    ).validate(4)
    custom = case == "custom"
    assert cfg.control_batch_size == (32 if custom else 8)
    assert cfg.control_pool_size == (256 if custom else 128)
    assert cfg.full_opsd_warmup_steps == (0 if custom else 5)
    assert cfg.teacher_selection == "hard" and cfg.mask_teacher_include_final_answer
    assert cfg.eval_seed == (456 if custom else 42)
    assert cfg.seed == (123 if custom else 42)
    if case == "generated":
        run = Path(cfg.output_dir)
        work = Path(calls[0]["env"]["EVAL_WORK_DIR"])
        assert "rho0.9" in run.name and "s42_es42_n75" in run.name
        assert work.parent == run.parent / ".runtime"
        assert run.name not in str(work)
    assert cfg.output_dir == str(run)
    assert options.config == str(run / "config.yaml")
    assert load_config(options.config) == cfg
    assert (run / "experiment.json").is_file()
    if case in {"hyperparameters", "cli_override"}:
        assert cfg.ftrl_rho == (0.6 if case == "cli_override" else 0.8)
        assert (cfg.delta_rms_decay, cfg.delta_rms_floor, cfg.delta_scaled_clip) == (0.7, 0.002, 0.5)
        assert (cfg.window_steps, cfg.warmup_windows, cfg.controller_lr) == (3, 1, 0.02)
        assert (cfg.exposure_temperature, cfg.learning_rate, cfg.lambda_gt, cfg.gamma_discount) == (
            0.1,
            2e-6,
            0.03,
            0.8,
        )
    else:
        assert cfg.ftrl_rho == (0.73 if case == "yaml" else 0.9)
    forwarded = calls[0]["env"]
    assert forwarded["TMPDIR"] == forwarded["TMP"] == forwarded["TEMP"] == str(work / "tmp")
    for key, subdir in {
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "VLLM_CACHE_ROOT": "vllm",
        "UV_CACHE_DIR": "uv",
    }.items():
        assert forwarded[key] == str(work / subdir)
    assert sentinel.read_text() == "unchanged"
    assert "mock training" in (run / "train_eval.log").read_text()
    if case != "train_failure":
        evaluation = calls[-1]["args"]
        assert evaluation[evaluation.index("--datasets") + 1] == "aime24,aime25"
        steps = "50,75,100" if custom else "25,30" if case == "short_horizon" else "25,50,75"
        assert evaluation[evaluation.index("--steps") + 1] == steps
        assert evaluation[evaluation.index("--seed") + 1] == str(cfg.eval_seed)
        assert evaluation[evaluation.index("--gpus") + 1] == env["CUDA_VISIBLE_DEVICES"]


@pytest.mark.parametrize(
    "override",
    [
        {"DUEL_PROBLEMS": "129"},
        {"DUEL_PROBLEMS": "4"},
        {"HELDOUT_PROBLEMS": "bad"},
        {"WARMUP_STEPS": "76"},
        {"EVAL_STEPS": "50,100"},
        {"EVAL_STEPS": "26"},
        {"EVAL_DATASETS": "typo"},
        {"SAVE_EVERY": "0"},
    ],
)
def test_invalid_experiment_parameters_fail_before_launch(tmp_path, override):
    env = {
        **os.environ,
        "RUN_DIR": str(tmp_path / "run"),
        "DUEL_PROBLEMS": "8",
        "HELDOUT_PROBLEMS": "128",
        "MIN_VALID_CONTROLS": "8",
        "WARMUP_STEPS": "5",
        "MAX_STEPS": "75",
        "SAVE_EVERY": "25",
        "EVAL_STEPS": "25,50,75",
        "EVAL_DATASETS": "aime24,aime25",
        **override,
    }
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not (tmp_path / "run").exists()


def test_experiment_refuses_existing_logs(tmp_path):
    log = tmp_path / "train_eval.log"
    log.write_text("previous run")
    env = {**os.environ, "RUN_DIR": str(tmp_path), "MAX_STEPS": "75", "EVAL_STEPS": "75"}
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert log.read_text() == "previous run"


def test_generated_name_and_snapshot_use_resolved_config(tmp_path):
    args = [
        "--config=adaptive_graph_ftrl_redesign/configs/hard_experiment.yaml",
        "--ftrl-rho=0.8",
        "--seed=123",
        "--eval-seed=456",
        "--lambda-gt=0.02",
    ]
    output = prepare(args, tmp_path)
    assert "rho0.8_rms0.9_floor0.0001_clip1_duel8_pool128_min8_w5_H5_aw3" in output.name
    assert "lgt0.02" in output.name and "s123_es456_n75" in output.name
    assert len(output.name.encode()) <= 255
    cfg = load_config(output / "config.yaml").validate(4)
    assert cfg.output_dir == str(output) and cfg.ftrl_rho == 0.8
    first = json.loads((output / "experiment.json").read_text())["config_hash"]
    other = prepare([*args, "--adam-beta2=0.98"], tmp_path)
    assert first != json.loads((other / "experiment.json").read_text())["config_hash"]
    with pytest.raises(FileExistsError):
        prepare(args, tmp_path, str(output))


@pytest.mark.parametrize(
    "option",
    [
        "--ftrl-rho=1.1",
        "--delta-rms-decay=1",
        "--delta-rms-floor=0",
        "--delta-scaled-clip=0",
        "--controller-lr=nan",
        "--window-steps=0",
    ],
)
def test_invalid_hyperparameters_fail_before_creating_run(tmp_path, option):
    with pytest.raises(ValueError):
        prepare([option], tmp_path)
    assert list(tmp_path.iterdir()) == []
