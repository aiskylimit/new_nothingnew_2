import os
from pathlib import Path
import tempfile
import uuid

from adaptive_graph_ftrl_redesign.evaluation import prepare_eval_tempdir


def test_overlong_inherited_tempdir_is_replaced(tmp_path, monkeypatch):
    old = tmp_path / ("long-run-name-" * 8)
    old.mkdir()
    for name in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(name, str(old))
    with tempfile.TemporaryDirectory(prefix="l2f-test-", dir="/tmp") as short:
        monkeypatch.setattr("adaptive_graph_ftrl_redesign.evaluation.tempfile.mkdtemp",
                            lambda **kwargs: short)
        result = prepare_eval_tempdir()
        assert result == short
        assert all(os.environ[name] == short for name in ("TMPDIR", "TMP", "TEMP"))
        assert len(os.fsencode(str(Path(result).resolve() / str(uuid.uuid4())))) <= 107


def test_short_existing_tempdir_is_preserved(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="l2f-test-", dir="/tmp") as short:
        for name in ("TMPDIR", "TMP", "TEMP"):
            monkeypatch.setenv(name, short)
        assert prepare_eval_tempdir() == short
