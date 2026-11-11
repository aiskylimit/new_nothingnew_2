import argparse
import os
from fcntl import LOCK_EX, flock
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

    def ignore(directory, names):
        excluded = {"preds.txt", "answers.jsonl"}
        if Path(directory) == source:
            excluded.add(destination.name)
        return excluded.intersection(names)

    with TemporaryDirectory(prefix=".eval-export-", dir=source.parent) as temporary:
        staged = Path(temporary) / destination.name
        copytree(source, staged, symlinks=True, ignore=ignore)
        size = sum(path.stat().st_size for path in staged.rglob("*") if not path.is_symlink() and path.is_file())
        previous = Path(temporary) / "previous"
        directory_fd = os.open(source, os.O_RDONLY)
        try:
            flock(directory_fd, LOCK_EX)
            if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                parser.error(f"Export path must be a directory: {destination}")
            try:
                if destination.exists():
                    destination.rename(previous)
                staged.rename(destination)
            except BaseException:
                if previous.exists() and not destination.exists():
                    previous.rename(destination)
                raise
        finally:
            os.close(directory_fd)

    print(f"Exported {size / 1_000_000:.2f} MB ({size:,} bytes) to {destination}")


if __name__ == "__main__":
    main()
