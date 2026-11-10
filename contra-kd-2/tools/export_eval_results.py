import argparse
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description="Copy evaluation results without generated responses")
    parser.add_argument("eval_main", type=Path, help="Directory containing the saved evaluations")
    args = parser.parse_args()
    source = args.eval_main.resolve()
    if not source.is_dir():
        parser.error(f"Evaluation directory does not exist: {source}")

    destination = source / "results"
    if destination.exists() or destination.is_symlink():
        parser.error(f"Export already exists: {destination}. Move it before exporting again.")

    def ignore(directory, names):
        excluded = {"preds.txt", "answers.jsonl"}
        if Path(directory) == source:
            excluded.add(destination.name)
        return excluded.intersection(names)

    with TemporaryDirectory(prefix=".eval-export-", dir=source.parent) as temporary:
        staged = Path(temporary) / destination.name
        copytree(source, staged, symlinks=True, ignore=ignore)
        size = sum(path.stat().st_size for path in staged.rglob("*") if not path.is_symlink() and path.is_file())
        staged.rename(destination)

    print(f"Exported {size / 1_000_000:.2f} MB ({size:,} bytes) to {destination}")


if __name__ == "__main__":
    main()
