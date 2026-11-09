"""Keep timing trials distinct from checkpointed recovery/stability tests."""

import unittest

import background


class ContinuousProtocolTest(unittest.TestCase):
    def test_continuous_protocol_is_a_single_month_with_endpoint_output(self):
        args = background.simulation_arguments(True)
        options = dict(zip(args[::2], args[1::2]))
        self.assertEqual(options, {"--mode": "benchmark", "--total-days": "30",
                                   "--save-days": "30", "--repeats": "1", "--truncation": "63"})

    def test_daily_checkpoint_protocol_remains_separate(self):
        args = background.simulation_arguments(False)
        options = dict(zip(args[::2], args[1::2]))
        self.assertEqual(options["--mode"], "month")
        self.assertEqual(options["--save-days"], "1")


if __name__ == "__main__":
    unittest.main()
