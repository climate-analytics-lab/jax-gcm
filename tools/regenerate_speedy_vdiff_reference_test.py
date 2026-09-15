"""The parts of the reference regenerator that need no gfortran or network."""

import ast
import unittest

from tools.regenerate_speedy_vdiff_reference import _format_array


class TestFormatArray(unittest.TestCase):
    """The emitted literal is pasted into the test verbatim, so it has to
    parse back to exactly the values that went in -- a lost digit would move
    the reference without moving anything visible.
    """

    VALUES = [0.0, 0.0, 0.0, 0.0, 1.5418215701168233e-06,
              1.8719998350134365e-05, 2.5021758360846339e-05,
              -6.7127537049716855e-05]

    def test_round_trips_exactly(self):
        text = _format_array(self.VALUES, 16)
        self.assertEqual(ast.literal_eval(f"[{text}]"), self.VALUES)

    def test_zeros_are_written_plainly(self):
        self.assertTrue(_format_array([0.0, 0.0], 16).startswith("0.0, 0.0"))

    def test_wraps_to_the_indented_line_width(self):
        lines = _format_array(self.VALUES, 16).split("\n")
        self.assertGreater(len(lines), 1, "a full row should wrap")
        for line in lines:
            self.assertLessEqual(len(line) + 16, 79)

    def test_a_single_value_stays_on_one_line(self):
        self.assertEqual(_format_array([1.25], 16), "1.2500000000000000e+00")


if __name__ == "__main__":
    unittest.main()
