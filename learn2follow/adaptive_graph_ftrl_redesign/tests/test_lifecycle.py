import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest

from adaptive_graph_ftrl_redesign.checkpoint import save_checkpoint
from adaptive_graph_ftrl_redesign.lifecycle import (
    record_training_complete,
    shutdown_runtime,
    validate_completion,
)
from adaptive_graph_ftrl_redesign.rollout import VLLMRollouts
from .fixtures import tiny_trainer


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    trainer = tiny_trainer(tmp_path / "run", total_updates=2)
    trainer.run_update()
    trainer.run_update()
    save_checkpoint(trainer)
    launch_id = uuid.uuid4().hex
    monkeypatch.setenv("GRAPH_TRAIN_LAUNCH_ID", launch_id)
    record_training_complete(trainer)
    return trainer, launch_id


def test_completion_requires_current_launch_all_steps_and_final_adapter(completed_run):
    trainer, launch_id = completed_run
    root = trainer.config.output_dir
    validate_completion(root, launch_id, 2, 1)
    with pytest.raises(FileNotFoundError):
        validate_completion(root, uuid.uuid4().hex, 2, 1)
    with pytest.raises(ValueError, match="Rank"):
        validate_completion(root, launch_id, 1, 1)
    with pytest.raises(ValueError, match="Rank"):
        validate_completion(root, launch_id, 2, 4)
    (Path(root) / "checkpoint-2/adapter_model.safetensors").write_bytes(b"broken")
    with pytest.raises(Exception):
        validate_completion(root, launch_id, 2, 1)


def test_completion_refuses_partial_training_and_no_marker_for_direct_runs(tmp_path, monkeypatch):
    trainer = tiny_trainer(tmp_path, total_updates=4)
    monkeypatch.setenv("GRAPH_TRAIN_LAUNCH_ID", uuid.uuid4().hex)
    trainer.run_update()
    with pytest.raises(RuntimeError, match="all optimizer steps"):
        record_training_complete(trainer)
    assert not (tmp_path / ".train_completion").exists()
    monkeypatch.delenv("GRAPH_TRAIN_LAUNCH_ID")
    record_training_complete(trainer)


@pytest.mark.parametrize("fail_close", [False, True])
def test_engine_closes_before_process_group_even_on_shutdown_error(monkeypatch, fail_close):
    events = []

    def close():
        events.append("engine")
        if fail_close:
            raise RuntimeError("engine shutdown failed")

    trainer = SimpleNamespace(
        rollout_engine=SimpleNamespace(native_runtime=False, close=close),
        device=SimpleNamespace(type="cpu"),
    )
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.destroy_process_group", lambda: events.append("training_group"))
    if fail_close:
        with pytest.raises(RuntimeError, match="engine shutdown"):
            shutdown_runtime(trainer, 0)
    else:
        shutdown_runtime(trainer, 0)
        assert trainer.rollout_engine is None
    assert events == ["engine", "training_group"]


def test_native_groups_close_after_engine_before_training_group(monkeypatch):
    events = []
    module = SimpleNamespace(
        destroy_model_parallel=lambda: events.append("model_groups"),
        destroy_distributed_environment=lambda: events.append("vllm_world"),
    )
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", module)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.destroy_process_group", lambda: events.append("training_group"))
    trainer = SimpleNamespace(
        rollout_engine=SimpleNamespace(native_runtime=True, close=lambda: events.append("engine")),
        device=SimpleNamespace(type="cpu"),
    )
    shutdown_runtime(trainer, 0)
    assert events == ["engine", "model_groups", "vllm_world", "training_group"]


@pytest.mark.parametrize("version", ["v1", "v0", "direct"])
def test_rollout_close_is_idempotent_and_rejects_generate(version):
    shutdown = Mock()
    if version == "v1":
        llm = SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown))
    elif version == "v0":
        llm = SimpleNamespace(model_executor=SimpleNamespace(shutdown=shutdown))
    else:
        llm = SimpleNamespace(shutdown=shutdown)
    rollout = VLLMRollouts.__new__(VLLMRollouts)
    rollout.engine = SimpleNamespace(llm_engine=llm)
    rollout.close()
    rollout.close()
    shutdown.assert_called_once_with()
    assert rollout.engine is rollout.trainer is rollout.adapter is rollout.synced_step is None
    with pytest.raises(RuntimeError, match="closed"):
        rollout.generate([[1]], 10)


@pytest.mark.parametrize(
    "train_status,complete,corrupt,eval_status,expected_status,expect_eval",
    [
        (0, True, False, 0, 0, True),
        (1, True, False, 0, 1, True),
        (1, False, False, 0, 1, False),
        (0, False, False, 0, 1, False),
        (1, True, True, 0, 1, False),
        (0, True, False, 7, 7, True),
    ],
)
def test_train_eval_script_does_not_hide_failures_or_trust_stale_checkpoints(
    completed_run,
    tmp_path,
    train_status,
    complete,
    corrupt,
    eval_status,
    expected_status,
    expect_eval,
):
    trainer, _ = completed_run
    root = Path(__file__).resolve().parents[2]
    capture = tmp_path / "eval.json"
    fake = tmp_path / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "if sys.argv[1:3] == ['-m', 'torch.distributed.run']:\n"
        "    if os.environ['TEST_COMPLETE'] == '1':\n"
        "        from types import SimpleNamespace\n"
        "        from adaptive_graph_ftrl_redesign.lifecycle import record_training_complete\n"
        "        for rank in range(4):\n"
        "            record_training_complete(SimpleNamespace(step=2, pending=None, rank=rank, world=4, "
        "config=SimpleNamespace(total_updates=2, output_dir=os.environ['RUN_DIR'])))\n"
        "    if os.environ['TEST_CORRUPT'] == '1':\n"
        "        (Path(os.environ['RUN_DIR']) / 'checkpoint-2/adapter_model.safetensors').write_bytes(b'broken')\n"
        "    sys.exit(int(os.environ['TEST_TRAIN_STATUS']))\n"
        "if sys.argv[1:3] == ['-m', 'adaptive_graph_ftrl_redesign.evaluate']:\n"
        "    Path(os.environ['TEST_CAPTURE']).write_text(json.dumps(sys.argv[3:]))\n"
        "    sys.exit(int(os.environ['TEST_EVAL_STATUS']))\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])\n"
    )
    fake.chmod(0o755)
    result = subprocess.run(
        ["bash", str(root / "adaptive_graph_ftrl_redesign/scripts/run_train_then_eval.sh")],
        env={
            **os.environ,
            "PYTHON_BIN": str(fake),
            "RUN_DIR": trainer.config.output_dir,
            "MAX_STEPS": "2",
            "EVAL_STEPS": "2",
            "EVAL_DATASETS": "aime25",
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
            "TEST_CAPTURE": str(capture),
            "TEST_TRAIN_STATUS": str(train_status),
            "TEST_COMPLETE": str(int(complete)),
            "TEST_CORRUPT": str(int(corrupt)),
            "TEST_EVAL_STATUS": str(eval_status),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert capture.exists() is expect_eval
    if expect_eval:
        args = json.loads(capture.read_text())
        assert args[args.index("--gpus") + 1] == "4,5,6,7"
        assert args[args.index("--datasets") + 1] == "aime25"
        assert args[args.index("--seed") + 1] == "42"
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.9"
    if train_status and expect_eval:
        assert "after verified completion" in result.stderr
