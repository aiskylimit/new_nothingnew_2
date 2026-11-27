"""`load_dataset` that still works when the `datasets` package cannot be imported.

Some offline venvs have a broken pandas ("C extension: None not built"), and `datasets` imports pandas, so
`import datasets` fails and with it every loader of this package. This module imports `datasets` lazily and
uses it whenever it imports; only when it does not, and the path is a local snapshot directory of a Hub
dataset (what the offline download step writes), it reads the snapshot's data files with pyarrow instead and
gives back a small Dataset with exactly the operations the loaders use - column_names, filter, shuffle(seed),
select, len, iteration - with the same semantics as `datasets`:

  * data files: the README.md `configs:` entry of the requested config ("default" when none is given),
    resolved like the Hub does (split -> glob patterns, files sorted); without `configs:`, files whose
    name/folder carries the split name (train/training, test/testing/eval, validation/valid/dev), and all
    data files as "train" when none does. Hidden files and folders (.cache, .gitattributes) are skipped.
  * shuffle(seed): np.random.default_rng(seed).permutation(n), as datasets.Dataset.shuffle does.
  * rows: plain dicts, in file order.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path

_DATA_EXT = (".parquet", ".jsonl", ".json", ".csv", ".arrow")
_SPLIT_WORDS = {"train": ("train", "training"), "test": ("test", "testing", "eval", "evaluation"),
                "validation": ("validation", "valid", "val", "dev")}
_warned = set()


def load_dataset(path, name=None, split=None, **kwargs):
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as e:
        if not os.path.isdir(str(path)):
            raise
        if str(path) not in _warned:
            _warned.add(str(path))
            print(f"[local_dataset] `datasets` does not import ({type(e).__name__}: {str(e)[:120]}); "
                  f"reading {path} with pyarrow", flush=True)
        return LocalDataset.from_snapshot(str(path), name, split or "train")
    return hf_load_dataset(path, name, split=split, **kwargs)


class LocalDataset:
    def __init__(self, rows, columns):
        self._rows = rows
        self.column_names = list(columns)

    # ---- the datasets.Dataset operations the loaders use ----
    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    def filter(self, fn):
        return LocalDataset([r for r in self._rows if fn(r)], self.column_names)

    def select(self, indices):
        return LocalDataset([self._rows[i] for i in indices], self.column_names)

    def shuffle(self, seed=None):
        import numpy as np
        perm = np.random.default_rng(seed).permutation(len(self._rows))
        return LocalDataset([self._rows[int(i)] for i in perm], self.column_names)

    # ---- reading a snapshot ----
    @classmethod
    def from_snapshot(cls, root, name, split):
        files = data_files(root, name, split)
        if not files:
            raise FileNotFoundError(f"no data files for split {split!r} in {root}")
        rows, columns = [], []
        for f in files:
            r, c = _read(f)
            rows.extend(r)
            columns += [x for x in c if x not in columns]
        return cls(rows, columns)


def _visible(rel):
    return not any(part.startswith((".", "__")) for part in Path(rel).parts)


def _all_files(root):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not d.startswith((".", "__"))]
        for fn in fns:
            rel = os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/")
            if _visible(rel) and rel.lower().endswith(_DATA_EXT):
                out.append(rel)
    return sorted(out)


def _readme_configs(root):
    readme = Path(root) / "README.md"
    if not readme.exists():
        return None
    text = readme.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\s*\n(.*?)\n---\s*(\n|$)", text, re.S)
    if not m:
        return None
    import yaml
    meta = yaml.safe_load(m.group(1)) or {}
    return meta.get("configs")


def _glob(root, patterns, files):
    out = []
    for pat in patterns:
        pat = pat.lstrip("./")
        hits = [f for f in files if fnmatch.fnmatchcase(f, pat) or fnmatch.fnmatchcase(f, pat + "/*")
                or f.startswith(pat.rstrip("/") + "/")]
        out += [h for h in sorted(hits) if h not in out]
    return out


def data_files(root, name, split):
    files = _all_files(root)
    configs = _readme_configs(root)
    if configs:
        want = name or "default"
        cfg = next((c for c in configs if c.get("config_name", "default") == want), None)
        if cfg is None and name is None:
            cfg = next((c for c in configs if c.get("default")), None) or configs[0]
        if cfg is None:
            raise ValueError(f"config {want!r} not in {root}/README.md")
        df = cfg.get("data_files")
        if isinstance(df, str):
            df = [{"split": "train", "path": df}]
        elif isinstance(df, list) and df and not isinstance(df[0], dict):
            df = [{"split": "train", "path": df}]
        elif isinstance(df, dict):
            df = [{"split": k, "path": v} for k, v in df.items()]
        pats = [e["path"] for e in (df or []) if e.get("split", "train") == split]
        pats = [p for x in pats for p in (x if isinstance(x, list) else [x])]
        return [os.path.join(root, f) for f in _glob(root, pats, files)]
    # no configs: the Hub's split-name conventions
    def has(word, f):
        return re.search(rf"(^|[/\-._ 0-9]){word}([/\-._ 0-9]|$)", f.lower()) is not None
    by_split = {s: [f for f in files if any(has(w, f) for w in ws)] for s, ws in _SPLIT_WORDS.items()}
    if not any(by_split.values()):
        chosen = files if split == "train" else []
    else:
        chosen = by_split.get(split, [])
    if chosen:  # one format, the most common one (as the Hub infers the builder from the extensions)
        exts = [os.path.splitext(f)[1].lower() for f in chosen]
        top = max(set(exts), key=exts.count)
        chosen = [f for f in chosen if f.lower().endswith(top)]
    return [os.path.join(root, f) for f in chosen]


def _read(path):
    p = path.lower()
    if p.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        return t.to_pylist(), t.column_names
    if p.endswith(".arrow"):
        import pyarrow as pa
        with pa.memory_map(path) as src:
            try:
                t = pa.ipc.open_stream(src).read_all()
            except pa.ArrowInvalid:
                t = pa.ipc.open_file(src).read_all()
        return t.to_pylist(), t.column_names
    if p.endswith(".csv"):
        import pyarrow.csv as pcsv
        t = pcsv.read_csv(path)
        return t.to_pylist(), t.column_names
    with open(path, encoding="utf-8") as fh:  # .jsonl / .json
        text = fh.read()
    try:
        obj = json.loads(text)
        rows = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    return rows, cols
