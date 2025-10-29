import unittest
from jcm.date import fraction_of_year_elapsed, DateData
import jax_datetime as jdt

class TestDateUnit(unittest.TestCase):

    def test_fraction_of_year(self):
        # Test the fraction of the year elapsed function

        # Test leap year
        # Note, the below test incorrectly loops back to the beginning of the year, this doesn't matter for the fraction of the year
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2000-01-01')), 1.0, places=2)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2000-07-02')), 0.5, places=2)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2000-12-31')), 365/366, places=2)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2000-02-29')), (31+28)/366, places=2)

        # Test non-leap year
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2001-01-01')), 0.0, places=2)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2001-07-02 12:00:00')), 0.5, places=2)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2001-12-31')), 364/365, places=2)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2001-02-28')), (31+27)/365, places=2)

    def test_date_data(self):
        # Test the DateData class

        # Test with no input
        d = DateData.zeros()
        self.assertEqual(d.tyear, 0.0)

        # Test with input
        d = DateData.set_date(jdt.to_datetime('2000-07-02'))
        self.assertAlmostEqual(d.tyear, 0.5, places=2)

        # Test copy
        d2 = d.copy()
        self.assertAlmostEqual(d2.tyear, 0.5, places=2)

        # Test copy with input
        d3 = d.copy(0.25)
        self.assertAlmostEqual(d3.tyear, 0.25, places=2)