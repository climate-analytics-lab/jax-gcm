import unittest
from jcm.date import fraction_of_year_elapsed, DateData, parse_duration_days
from jcm.model import Model
import jax_datetime as jdt
import jax.numpy as jnp
from jcm.physics.speedy.speedy_coords import get_speedy_coords

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

    def test_overflow(self):
        # Use gregorian calendar so the 365.2425-day-per-year arithmetic in
        # this test matches the model's internal accounting; SPEEDY's 365-day
        # default would accumulate the 0.2425-day mismatch across decades.
        model = Model(
            coords=get_speedy_coords(),
            start_date=jdt.to_datetime('1970-01-01'),
            calendar='gregorian',
        )
        for i in range(6):
            year = 10**i
            date = model._date_from_sim_time((year+.5) * 365.2425 * 86400)
            self.assertEqual(date.model_year, jnp.round(1970 + year))
            self.assertTrue(jnp.isclose(date.tyear, 0.5, atol=1e-2))


class TestParseDurationDays(unittest.TestCase):

    def test_numeric_passthrough(self):
        self.assertEqual(parse_duration_days(10), 10.0)
        self.assertEqual(parse_duration_days(2.5), 2.5)

    def test_fixed_units(self):
        self.assertAlmostEqual(parse_duration_days('30 minutes'), 30 / 1440)
        self.assertAlmostEqual(parse_duration_days('12 hours'), 0.5)
        self.assertAlmostEqual(parse_duration_days('5 days'), 5.0)
        self.assertAlmostEqual(parse_duration_days('2 weeks'), 14.0)

    def test_unit_aliases(self):
        # The same physical duration spelled differently should agree.
        self.assertEqual(parse_duration_days('1 d'), parse_duration_days('1 day'))
        self.assertEqual(parse_duration_days('1 day'), parse_duration_days('1 days'))
        self.assertEqual(parse_duration_days('3 hr'), parse_duration_days('3 hours'))
        self.assertEqual(parse_duration_days('1 mo'), parse_duration_days('1 month'))
        self.assertEqual(parse_duration_days('1 yr'), parse_duration_days('1 year'))

    def test_calendar_year(self):
        self.assertAlmostEqual(parse_duration_days('1 year', calendar='365_day'), 365.0)
        self.assertAlmostEqual(parse_duration_days('1 year', calendar='gregorian'), 365.2425)
        self.assertAlmostEqual(parse_duration_days('5 years', calendar='365_day'), 1825.0)

    def test_calendar_month(self):
        self.assertAlmostEqual(parse_duration_days('1 month', calendar='365_day'), 365.0 / 12)
        self.assertAlmostEqual(parse_duration_days('12 months', calendar='365_day'), 365.0)

    def test_unknown_unit_rejected(self):
        with self.assertRaises(ValueError):
            parse_duration_days('1 fortnight')
        with self.assertRaises(ValueError):
            parse_duration_days('not a duration')
