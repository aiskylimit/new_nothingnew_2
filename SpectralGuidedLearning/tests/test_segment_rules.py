import json
from pathlib import Path

import pytest

from sgl.segment import SEGMENTERS, split

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"
SAMPLE = json.loads((ASSETS / "palign_sample.jsonl").open().readline())["output"]
TEXTS = [
    SAMPLE,
    "\n\nleading break\n\n\n\nfour newlines\n\nWait, a cue.\n\nAlternatively x\n\nHowever y",
    "no breaks at all",
    "Wait\n\nWait\n\n\n\nBut wait, no.\n\n",
    "",
]


@pytest.mark.parametrize("mode", sorted(SEGMENTERS))
@pytest.mark.parametrize("text", TEXTS)
def test_lossless_segmenters_reproduce_the_text(mode, text):
    pieces = split(text, mode)
    assert "".join(pieces) == text
    assert all(pieces)


def test_unknown_segmenter():
    with pytest.raises(ValueError):
        split("x", "words")
