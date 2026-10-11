"""The pruned branch must run without any legacy source package installed."""

import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
REMOVED = {
    "adaptive_graph_ftrl_opsd", "full_masked_graph_opsd", "graph_based",
    "graph_free", "se_opsd", "se_opsd_lite", "experiments", "ablation",
}


def test_no_legacy_imports_or_missing_relative_modules():
    package = ROOT / "adaptive_graph_ftrl_redesign"
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not {alias.name.split(".")[0] for alias in node.names} & REMOVED, path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in REMOVED, path
                if node.level and node.module:
                    target = path.parent
                    for _ in range(node.level - 1):
                        target = target.parent
                    target = target.joinpath(*node.module.split("."))
                    assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), path


def test_entrypoints_import_with_legacy_packages_blocked():
    script = f"""
import importlib.abc
import sys
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {REMOVED!r}:
            raise AssertionError('Unexpected legacy dependency: ' + fullname)
sys.meta_path.insert(0, BlockLegacy())
from adaptive_graph_ftrl_redesign import train, evaluate, smoke_test
from adaptive_graph_ftrl_redesign.rollout import VLLMRollouts
from adaptive_graph_ftrl_redesign.checkpoint import restore_checkpoint, save_checkpoint
train.parser().parse_args(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--teacher-selection" in result.stdout


def test_evaluation_launchers_still_exist():
    for name in ("run_graph_based_eval.sh", "run_graph_based_eval_opsd_4gpu.sh"):
        assert (ROOT / "graph_based_scripts" / name).is_file()
    assert (ROOT / "eval/evaluate_math.py").is_file()
