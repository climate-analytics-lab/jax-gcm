"""Guard against accidentally restarting each daily chunk from its initial state."""

import unittest
from unittest.mock import Mock

import month


class MonthContinuationTest(unittest.TestCase):
    def test_recovery_preserves_clock_and_preceding_days(self):
        report = dict(simulated_days=15, total_days=30, spectral_truncation=63,
                      layers=47, time_step_minutes=15,
                      chunks=[dict(day=day, finite=True) for day in range(1, 16)])
        self.assertEqual(month.recovery_start(report, 15.0, 63, 1440), 16)
        with self.assertRaises(ValueError):
            month.recovery_start(report, 15.0, 63, 0)
        report['chunks'][4]['finite'] = False
        with self.assertRaises(ValueError):
            month.recovery_start(report, 15.0, 63, 1440)

    def test_one_reset_then_29_continuations(self):
        model = Mock()
        initial, forcing = object(), object()
        for day in range(1, 31):
            month.integrate_day(model, initial, forcing, day)
        model.run.assert_called_once_with(initial_state=initial, forcing=forcing,
                                         total_time=1.0, save_interval=1.0,
                                         output_averages=False)
        self.assertEqual(model.resume.call_count, 29)
        for call in model.resume.call_args_list:
            self.assertEqual(call.kwargs, dict(forcing=forcing, total_time=1.0,
                                              save_interval=1.0, output_averages=False))


if __name__ == "__main__":
    unittest.main()
