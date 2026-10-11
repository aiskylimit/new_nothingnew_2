import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from adaptive_graph_ftrl_redesign.eval_preflight import check_checkpoints, check_storage


def test_generic_script_help_does_not_require_uv_or_checkpoint(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_eval.sh"
    result = subprocess.run(
        ["bash", str(script), "--help"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert "EVAL_DATASET=aime25" in result.stdout
    assert "EVAL_STEPS=25,50" in result.stdout


@pytest.fixture
def evaluation_run(tmp_path):
    run = tmp_path / "run with spaces"
    for step in (25, 50):
        checkpoint = run / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(json.dumps({"r": 64}))
        save_file({"test_lora": torch.zeros(1)}, checkpoint / "adapter_model.safetensors")
    return run


def test_preflight_checks_all_adapters_before_loading_model(evaluation_run):
    check_checkpoints(evaluation_run, ["25", "50"])
    with pytest.raises(ValueError, match="positive"):
        check_checkpoints(evaluation_run, ["25", ""])
    (evaluation_run / "checkpoint-50/adapter_model.safetensors").write_bytes(b"broken")
    with pytest.raises(Exception):
        check_checkpoints(evaluation_run, ["25", "50"])


def test_preflight_detects_low_disk_space_and_inodes(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=1024))
    with pytest.raises(RuntimeError, match="EVAL_WORK_DIR"):
        check_storage({"TMPDIR": tmp_path}, min_free_gib=10)
    monkeypatch.setattr("os.statvfs", lambda path: SimpleNamespace(f_favail=0, f_files=1))
    with pytest.raises(RuntimeError, match="inodes"):
        check_storage({"TMPDIR": tmp_path}, min_free_gib=0)


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "aime24",
        "aime26",
        "invalid_dataset",
        "failed_eval",
        "missing_50",
        "preflight_only",
        "low_space",
    ],
)
def test_real_graph_scripts_and_preflight_share_environment_and_preserve_errors(
    evaluation_run, tmp_path, case
):
    root = Path(__file__).resolve().parents[2]
    capture = tmp_path / "calls.jsonl"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args[:4] == ['run', 'python', '-m', 'adaptive_graph_ftrl_redesign.eval_preflight']:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}] + args[2:])\n"
        "assert args[:3] == ['run', 'python', 'eval/evaluate_math.py']\n"
        "with Path(os.environ['TEST_CAPTURE']).open('a') as f:\n"
        "    f.write(json.dumps({'argv': args, 'env': dict(os.environ)}) + '\\n')\n"
        "print('mock graph_based evaluation')\n"
        "sys.exit(int(os.environ['TEST_EVAL_STATUS']))\n"
    )
    uv.chmod(0o755)
    if case == "missing_50":
        (evaluation_run / "checkpoint-50/adapter_model.safetensors").unlink()
    work = tmp_path / "compiler workspace"
    existing_cache = work / "triton/keep-existing-cache"
    existing_cache.parent.mkdir(parents=True)
    existing_cache.write_text("do not clean user caches")
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "RUN_DIR": str(evaluation_run),
        "EVAL_WORK_DIR": str(work),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "UV_PROJECT_ENVIRONMENT": "working_se_environment",
        "EVAL_MIN_FREE_GIB": "1e30" if case == "low_space" else "0",
        "EVAL_MIN_FREE_INODES": "0",
        "EVAL_PREFLIGHT_ONLY": str(int(case == "preflight_only")),
        "EVAL_STEPS": "25,50",
        "EVAL_DATASET": "invalid"
        if case == "invalid_dataset"
        else (case if case in {"aime24", "aime26"} else "aime25"),
        "TEST_CAPTURE": str(capture),
        "TEST_EVAL_STATUS": "7" if case == "failed_eval" else "0",
        "CC": "/nonexistent/preflight-must-not-invoke-a-compiler",
    }
    if case == "aime24":
        env.update(
            EVAL_STEPS="50",
            TEMPERATURE="0.6",
            VAL_N="8",
            EVAL_SEED="123",
            ENABLE_THINKING="0",
            EVAL_GPU_MEMORY_UTILIZATION="0.8",
            MAX_MODEL_LEN="32768",
            MAX_NEW_TOKENS="2048",
        )
    result = subprocess.run(
        [
            "bash",
            str(
                root
                / "adaptive_graph_ftrl_redesign/scripts"
                / ("run_eval.sh" if case == "aime24" else "run_eval_aime25_25_50.sh")
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert existing_cache.read_text() == "do not clean user caches"
    if case in {"missing_50", "low_space", "invalid_dataset"}:
        assert result.returncode != 0
        assert not capture.exists(), result.stdout + result.stderr
        return
    assert result.returncode == (7 if case == "failed_eval" else 0), result.stdout + result.stderr
    assert "Preflight OK" in result.stdout
    assert not list(work.rglob("gcc-probe-*"))
    if case == "preflight_only":
        assert not capture.exists()
        return
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == (1 if case in {"failed_eval", "aime24"} else 2)
    for call, step in zip(calls, [50] if case == "aime24" else [25, 50]):
        args, forwarded = call["argv"], call["env"]
        expected = {
            "--checkpoint_dir": str(evaluation_run / f"checkpoint-{step}"),
            "--base_model": "Qwen/Qwen3-4B",
            "--dataset": "aime25",
            "--temperature": "1.0",
            "--top_p": "0.95",
            "--top_k": "-1",
            "--min_p": "0.0",
            "--presence_penalty": "0.0",
            "--val_n": "12",
            "--max_new_tokens": "38912",
            "--max_model_len": "40960",
            "--tensor_parallel_size": "4",
            "--seed": "42",
            "--gpu_memory_utilization": "0.9",
        }
        dataset = case if case in {"aime24", "aime26"} else "aime25"
        expected["--dataset"] = dataset
        if case == "aime24":
            expected.update(
                {
                    "--dataset": dataset,
                    "--temperature": "0.6",
                    "--val_n": "8",
                    "--seed": "123",
                    "--gpu_memory_utilization": "0.8",
                    "--max_model_len": "32768",
                    "--max_new_tokens": "2048",
                }
            )
        assert ("--no_thinking" if case == "aime24" else "--enable_thinking") in args
        for flag, value in expected.items():
            assert args[args.index(flag) + 1] == value
        assert forwarded["TMPDIR"] == forwarded["TMP"] == forwarded["TEMP"] == str(work / "tmp")
        assert forwarded["TRITON_CACHE_DIR"] == str(work / "triton")
        assert forwarded["TORCHINDUCTOR_CACHE_DIR"] == str(work / "torchinductor")
        assert forwarded["VLLM_CACHE_ROOT"] == str(work / "vllm")
        assert forwarded["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert forwarded["UV_PROJECT_ENVIRONMENT"] == "working_se_environment"
        assert args[args.index("--output_file") + 1] == str(
            evaluation_run / f"eval_{dataset}/checkpoint-{step}_{dataset}.json"
        )
        log = evaluation_run / f"eval_{dataset}/checkpoint-{step}_{dataset}.log"
        assert "mock graph_based evaluation" in log.read_text()
