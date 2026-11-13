import argparse
import os
from fcntl import LOCK_EX, flock
from itertools import chain
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description="Copy aggregate evaluation results and run logs")
    parser.add_argument("eval_dir", type=Path, help="Directory containing the saved evaluations")
    args = parser.parse_args()
    source = args.eval_dir.resolve()
    if not source.is_dir():
        parser.error(f"Evaluation directory does not exist: {source}")

    destination = source.with_name(f"{source.name}_export")
    with TemporaryDirectory(prefix=".eval-export-", dir=source.parent) as temporary:
        staged = Path(temporary) / destination.name
        files = chain(source.rglob("results_*.json"), source.rglob("eval.log"))
        size = 0
        for path in files:
            if path.is_symlink() or not path.is_file():
                continue
            target = staged / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(path, target)
            size += target.stat().st_size
        if not staged.exists():
            parser.error(f"No aggregate results or evaluation logs found in {source}")

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
