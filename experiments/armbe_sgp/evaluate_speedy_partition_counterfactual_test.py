import unittest
from types import SimpleNamespace

import numpy as np

from evaluate_speedy_partition_counterfactual import repartition_cloud_cover


class SpeedyPartitionCounterfactualTest(unittest.TestCase):
    def test_repartition_preserves_nested_total(self):
        nested = SimpleNamespace(
            cloudc=np.asarray([0.6, 0.2]),
            cloudstr=np.asarray([0.0, 0.0]),
            copy=lambda **values: SimpleNamespace(**values),
        )
        baseline = SimpleNamespace(
            cloudc=np.asarray([0.3, 0.0]),
            cloudstr=np.asarray([0.3, 0.0]),
        )

        actual = repartition_cloud_cover(nested, baseline)

        np.testing.assert_allclose(actual.cloudc + actual.cloudstr, [0.6, 0.2])
        np.testing.assert_allclose(actual.cloudstr, [0.3, 0.0])


if __name__ == "__main__":
    unittest.main()
