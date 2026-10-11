import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from adaptive_graph_ftrl_redesign.evaluate import main
from adaptive_graph_ftrl_redesign.eval_adapter import eval_command, run_eval


def test_redesign_uses_se_launcher_by_default(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--run-dir",
            "/run/eta1",
            "--steps",
            "25,50",
            "--dry-run",
        ],
    )
    main()
    output = capsys.readouterr().out
    settings = json.loads(output.splitlines()[0])
    assert settings["launcher"] == "se-opsd"
    assert settings["gpu_memory_utilization"] == 0.9
    assert settings["seed"] == 42
    assert settings["temperature"] == 1.0
    assert settings["top_p"] == 0.95 and settings["top_k"] == -1
    assert settings["samples_per_prompt"] == 12 and settings["max_new_tokens"] == 38912
    assert output.count("bash graph_based_scripts/run_graph_based_eval_opsd_4gpu.sh") == 2
    assert "CHECKPOINT_DIR=/run/eta1/checkpoint-25" in output
    assert "CHECKPOINT_DIR=/run/eta1/checkpoint-50" in output
    assert "CUDA_VISIBLE_DEVICES=4,5,6,7" in output


def test_redesign_can_opt_into_active_python(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--run-dir",
            "/run",
            "--launcher",
            "python",
            "--dry-run",
        ],
    )
    main()
    output = capsys.readouterr().out
    assert '"launcher": "python"' in output
    assert "graph_based_scripts/run_graph_based_eval_opsd_4gpu.sh" not in output
    assert "eval/evaluate_math.py" in output


@pytest.mark.parametrize("uv_environment", [None, "custom_eval_env"])
def test_se_scripts_forward_exact_recipe_without_gpu(monkeypatch, tmp_path, uv_environment):
    # Exercise the real SE shell scripts, replacing only uv's final execution.
    capture = tmp_path / "capture.json"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['TEST_EVAL_CAPTURE']).write_text(json.dumps("
        "{'argv': sys.argv[1:], 'env': dict(os.environ)}))\n"
        "sys.exit(int(os.environ.get('TEST_EVAL_EXIT', '0')))\n"
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_EVAL_CAPTURE", str(capture))
    monkeypatch.setenv("TEST_EVAL_EXIT", "7")
    if uv_environment is None:
        monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", uv_environment)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    result = run_eval(
        "/run with spaces/checkpoint-25",
        "aime25",
        launcher="se-opsd",
        gpus="4,5,6,7",
        gpu_memory_utilization=0.9,
        output_file=str(tmp_path / "results with spaces.json"),
    )
    assert result == 7
    recorded = json.loads(capture.read_text())
    argv = recorded["argv"]
    assert argv[:3] == ["run", "python", "eval/evaluate_math.py"]
    expected = {
        "--base_model": "Qwen/Qwen3-4B",
        "--checkpoint_dir": "/run with spaces/checkpoint-25",
        "--dataset": "aime25",
        "--max_new_tokens": "38912",
        "--temperature": "1.0",
        "--top_p": "0.95",
        "--top_k": "-1",
        "--min_p": "0.0",
        "--presence_penalty": "0.0",
        "--tensor_parallel_size": "4",
        "--val_n": "12",
        "--seed": "42",
        "--gpu_memory_utilization": "0.9",
        "--output_file": str(tmp_path / "results with spaces.json"),
    }
    for flag, value in expected.items():
        assert argv[argv.index(flag) + 1] == value
    assert "--enable_thinking" in argv
    assert recorded["env"]["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert recorded["env"]["UV_PROJECT_ENVIRONMENT"] == (uv_environment or "dsd")


def test_se_launcher_uses_manifest_recipe(monkeypatch, tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "model": "custom/model",
                    "eval_temperature": 0.8,
                    "eval_top_p": 0.9,
                    "eval_top_k": 20,
                    "eval_samples": 3,
                    "eval_max_new_tokens": 2048,
                    "eval_seed": 123,
                }
            }
        )
    )
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").touch()
    runner = Mock(return_value=0)
    monkeypatch.setattr("adaptive_graph_ftrl_redesign.evaluation.run_eval", runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--run-dir",
            str(tmp_path),
            "--steps",
            "25",
            "--gpu-memory-utilization",
            "0.4",
        ],
    )
    main()
    settings = runner.call_args.kwargs
    assert settings["launcher"] == "se-opsd"
    assert settings["model"] == "custom/model"
    assert settings["temperature"] == 0.8 and settings["top_p"] == 0.9
    assert settings["top_k"] == 20 and settings["samples"] == 3
    assert settings["max_new_tokens"] == 2048 and settings["seed"] == 123
    assert settings["gpu_memory_utilization"] == 0.4


def test_se_launcher_rejects_unknown_launcher():
    with pytest.raises(ValueError, match="launcher"):
        eval_command("/run/checkpoint-25", "aime25", launcher="unknown")
