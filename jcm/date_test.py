import unittest
from jcm.date import fraction_of_year_elapsed, DateData, parse_duration_days
from jcm.model import Model
import jax_datetime as jdt
import jax.numpy as jnp
from jcm.physics.speedy.speedy_coords import get_speedy_coords

class TestDateUnit(unittest.TestCase):

    def test_fraction_of_year_gregorian_leap_year(self):
        # Under gregorian, fraction-of-year is exactly day-of-year / actual
        # year length (366 in 2000, a leap year). #410.
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2000-01-01')), 0.0, places=4)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2000-07-02')), 183/366, places=4)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2000-12-31')), 365/366, places=4)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2000-02-29')), (31+28)/366, places=4)

    def test_fraction_of_year_gregorian_non_leap(self):
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2001-01-01')), 0.0, places=4)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2001-07-02 12:00:00')), (182 + 0.5)/365, places=4)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2001-12-31')), 364/365, places=4)
        self.assertAlmostEqual(fraction_of_year_elapsed(jdt.to_datetime('2001-02-28')), (31+27)/365, places=4)

    def test_fraction_of_year_365_day_wraps_at_365(self):
        # Under '365_day' the year is a fixed-length 365-day chunk indexed
        # against the gregorian `delta.days` mod 365, so two dates exactly
        # 365 days apart agree on tyear.
        a = fraction_of_year_elapsed(jdt.to_datetime('2001-01-01'), calendar='365_day')
        b = fraction_of_year_elapsed(jdt.to_datetime('2002-01-01'), calendar='365_day')
        self.assertAlmostEqual(float(a), float(b), places=6)
        # 365 days after Jan 1 2001 lands at the same tyear.
        c = fraction_of_year_elapsed(
            jdt.Datetime.from_pydatetime(jdt.to_datetime('2001-01-01'))
            + jdt.Timedelta(days=jnp.int32(365)),
            calendar='365_day',
        )
        self.assertAlmostEqual(float(a), float(c), places=6)

    def test_date_data(self):
        # Test the DateData class — `tyear`/`model_year` are now methods
        # derived from `dt`, so `dt` is the only state.

        # Default: 1950-01-01.
        d = DateData.zeros()
        self.assertAlmostEqual(float(d.tyear()), 0.0, places=4)

        # set_date with an explicit Datetime.
        d = DateData.set_date(jdt.to_datetime('2000-07-02'))
        self.assertAlmostEqual(float(d.tyear()), 183/366, places=4)

        # copy preserves dt.
        d2 = d.copy()
        self.assertAlmostEqual(float(d2.tyear()), 183/366, places=4)

        # copy with a new dt overrides.
        d3 = d.copy(dt=jdt.to_datetime('2001-04-01'))
        self.assertAlmostEqual(float(d3.tyear()), 90/365, places=4)

    def test_overflow(self):
        model = Model(
            coords=get_speedy_coords(),
            start_date=jdt.to_datetime('1970-01-01'),
            calendar='gregorian',
        )
        for i in range(6):
            year = 10**i
            date = model._date_from_sim_time((year+.5) * 365.2425 * 86400)
            self.assertEqual(date.model_year('gregorian'), jnp.round(1970 + year))
            self.assertTrue(jnp.isclose(date.tyear('gregorian'), 0.5, atol=2e-2))


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
