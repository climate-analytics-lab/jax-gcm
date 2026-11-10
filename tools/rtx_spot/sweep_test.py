"""Check that the strict comparison rejects scientifically silent changes."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from flax.serialization import msgpack_serialize

os.environ.setdefault("RESULT_DIR", "/tmp/jcm-rtx-test")
spec = importlib.util.spec_from_file_location("rtx_sweep", Path(__file__).with_name("sweep.py"))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def fixture(directory, value=0.0, dtype="float32"):
    directory.mkdir()
    a = np.array([value], dtype=dtype)
    payload = {"dycore": {"temperature": a}, "physics": {"humidity": a},
               "clock": {"step": np.array(3600)}, "elapsed_days": 30.0}
    (directory / "final.ckpt").write_bytes(msgpack_serialize(payload))
    xr.Dataset({"temperature": (("level",), a)}, coords={"level": [47]}).to_netcdf(directory / "state_day30.nc")
    return sweep.fingerprints(directory)


@pytest.mark.parametrize("value,dtype", [(0.0, "float32"), (-0.0, "float32"),
                                            (1e-7, "float32"), (0.0, "float64"),
                                            (float("nan"), "float32")])
def test_strict_comparison(tmp_path, value, dtype):
    reference = fixture(tmp_path / "reference")
    candidate = fixture(tmp_path / "candidate", value, dtype)
    expected = value == 0.0 and not np.signbit(value) and dtype == "float32"
    assert sweep.compare(reference, candidate)["bitwise_equal"] == expected


def test_missing_array_rejected(tmp_path):
    reference = fixture(tmp_path / "reference")
    candidate = dict(reference)
    candidate.pop("checkpoint/physics/humidity")
    assert not sweep.compare(reference, candidate)["bitwise_equal"]
