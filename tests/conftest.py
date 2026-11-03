import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """A very small synthetic dataset (scale 0.02 = 10.8K orders) in a temp folder."""
    from cottonil.synth import generate
    d = tmp_path_factory.mktemp("tiny") / "COTTONIL_Data"
    generate(d, scale=0.02, seed=3, verbose=False)
    old = os.environ.get("COTTONIL_DATA_DIR")
    os.environ["COTTONIL_DATA_DIR"] = str(d)
    yield d
    if old is None:
        os.environ.pop("COTTONIL_DATA_DIR", None)
    else:
        os.environ["COTTONIL_DATA_DIR"] = old
