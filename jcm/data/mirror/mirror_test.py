"""Unit tests for the mirror builders' pure-math pieces.

The builders themselves read multi-GB Glade sources; these tests cover
the statistics/registry machinery on synthetic inputs. The shared
regridding helpers are tested in ``jcm/data/regridding_test.py``, the
downloader in ``jcm/data/remote_test.py``.
"""

import unittest

import numpy as np

from jcm.data.mirror.sso import finalize
from jcm.data.mirror.registry import build_registry, write_registry


class SsoFinalizeTest(unittest.TestCase):
    def _acc(self, **over):
        n = np.full(4, 10.0)
        acc = {"n": n, "land": n.copy(), "sh": np.zeros(4),
               "sh2": np.zeros(4), "shx2": np.zeros(4),
               "shy2": np.zeros(4), "shxy": np.zeros(4),
               "pic": np.zeros(4), "val": np.zeros(4)}
        acc.update(over)
        return acc

    def test_isotropic_slope_has_zero_anisotropy(self):
        # equal x/y gradient variance, no correlation -> gamma = 1 (round)
        acc = self._acc(shx2=np.full(4, 10.0), shy2=np.full(4, 10.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orogam"], 1.0)
        np.testing.assert_allclose(out["orosig"], 1.0)  # sqrt(K+L=1)

    def test_pure_xslope_is_fully_anisotropic(self):
        # gradient variance only in x -> gamma = 0, theta = 0 deg
        acc = self._acc(shx2=np.full(4, 20.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orogam"], 0.0)
        np.testing.assert_allclose(out["orothe"], 0.0)
        np.testing.assert_allclose(out["orosig"], np.sqrt(2.0))

    def test_ocean_cells_are_zeroed(self):
        acc = self._acc(land=np.array([10.0, 0.0, 10.0, 0.0]),
                        sh=np.full(4, 50.0), shx2=np.full(4, 4.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orog"], [5.0, 0.0, 5.0, 0.0])
        np.testing.assert_allclose(out["lsm"], [1.0, 0.0, 1.0, 0.0])

    def test_variance_from_sufficient_statistics(self):
        h = np.array([1.0, 3.0, 5.0, 7.0])
        acc = self._acc(n=np.full(4, 2.0), land=np.full(4, 2.0),
                        sh=2 * h, sh2=2 * h ** 2 + 2.0)
        # per-cell: two samples h +/- 1 -> mean h, std 1
        out = finalize(acc)
        np.testing.assert_allclose(out["orog"], h)
        np.testing.assert_allclose(out["orostd"], 1.0)


class RegistryTest(unittest.TestCase):
    def test_registry_hashes_files(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sub").mkdir()
            (Path(d) / "sub" / "a.nc").write_bytes(b"hello")
            path = write_registry(d)
            reg = json.loads(Path(path).read_text())
            self.assertIn("sub/a.nc", reg["files"])
            self.assertEqual(reg["files"]["sub/a.nc"]["size"], 5)
            # registry.json itself is excluded
            reg2 = build_registry(d)
            self.assertNotIn("registry.json", reg2["files"])


if __name__ == "__main__":
    unittest.main()
