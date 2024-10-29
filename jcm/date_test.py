import unittest
from jcm import date
from datetime import datetime

class TestDateUnit(unittest.TestCase):

    def test_fraction_of_year(self):
        # Test the fraction of the year elapsed function

        # Test leap year
        self.assertAlmostEqual(date.fraction_of_year_elapsed(datetime(2000, 1, 1)), 0.0)
        self.assertAlmostEqual(date.fraction_of_year_elapsed(datetime(2000, 7, 2)), 0.5)
        self.assertAlmostEqual(date.fraction_of_year_elapsed(datetime(2000, 12, 31)), 365/366)
        self.assertAlmostEqual(date.fraction_of_year_elapsed(datetime(2000, 2, 29)), (31+28)/366)

        # Test non-leap year    
        self.assertAlmostEqual(date.fraction_of_year_elapsed(datetime(2001, 1, 1)), 0.0)
        self.assertAlmostEqual(date.fraction_of_year_elapsed(datetime(2001, 7, 2, 12)), 0.5)
        self.assertAlmostEqual(date.fraction_of_year_elapsed(datetime(2001, 12, 31)), 364/365)
        self.assertAlmostEqual(date.fraction_of_year_elapsed(datetime(2001, 2, 28)), (31+27)/365)
    
    def test_date_data(self):
        # Test the DateData class

        # Test with no input
        d = date.DateData()
        self.assertEqual(d.tyear, 0.0)

        # Test with input
        d = date.DateData(datetime(2000, 7, 2))
        self.assertAlmostEqual(d.tyear, 0.5)

        # Test copy
        d2 = d.copy()
        self.assertAlmostEqual(d2.tyear, 0.5)

        # Test copy with input
        d3 = d.copy(0.25)
        self.assertAlmostEqual(d3.tyear, 0.25)