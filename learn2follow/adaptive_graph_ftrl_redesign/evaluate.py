"""Question-only benchmark evaluation through the existing SE-OPSD launcher."""

from .evaluation import main as evaluate_checkpoints


def main():
    evaluate_checkpoints(default_launcher="se-opsd")


if __name__ == "__main__":
    main()
