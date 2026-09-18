import unittest
from jcm.date import fraction_of_year_elapsed, DateData
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
        model = Model(coords=get_speedy_coords(), start_date=jdt.to_datetime('1970-01-01'))
        for i in range(6):
            year = 10**i
            date = model._date_from_sim_time((year+.5) * 365.2425 * 86400)
            self.assertEqual(date.model_year, jnp.round(1970 + year))
            self.assertTrue(jnp.isclose(date.tyear, 0.5, atol=1e-2))

    def test_float32_sim_time_is_not_a_long_clock(self):
        """State.sim_time is float32 in the integrator. Adding dt=1800 s to it is exact
        below 2**24 s, rounds to +1792 s in [2**27, 2**32) and to +2048 s past 2**32 s:
        a 366.87-day and then a 321.0-day year. A long run must therefore keep sim_time
        short and carry the date exactly (see test_traced_start_date_origin)."""
        for start, inc in ((0.0, 1800.0), (2.0**30, 1792.0), (2.0**32, 2048.0)):
            t = jnp.float32(start)
            t_new = (t + jnp.float32(1800.0)).astype(jnp.float32)
            self.assertEqual(float(t_new) - float(t), inc)

    def test_traced_start_date_origin(self):
        """_date_from_sim_time / run_from_state accept a (traced) start_date origin, so a
        chunked caller can re-zero sim_time every chunk and carry the date in exact
        jax_datetime integer arithmetic; the result matches the int-based calendar 400 yr
        out, where a float32 sim_time could not."""
        import jax
        model = Model(coords=get_speedy_coords(), start_date=jdt.to_datetime('2015-01-18'))
        for years in (0, 136, 400):
            origin_np = jdt.to_datetime('2015-01-18') + jdt.to_timedelta(int(round(years * 365.2425)), 'day')
            origin = jax.tree.map(jnp.asarray, origin_np)
            tyear_of = jax.jit(lambda t, o: model._date_from_sim_time(t, o).tyear)
            for sim_time in (0.0, 1800.0, 86400.0, 3 * 86400.0 + 1800.0, 5 * 86400.0 - 1800.0):
                expect = fraction_of_year_elapsed(origin_np + jdt.to_timedelta(int(sim_time), 'second'))
                got = tyear_of(jnp.float32(sim_time), origin)
                self.assertTrue(jnp.isclose(got, expect, atol=2e-6), (years, sim_time, got, expect))
        # default origin unchanged
        self.assertTrue(jnp.isclose(model._date_from_sim_time(jnp.float32(86400.0)).tyear,
                                    fraction_of_year_elapsed(jdt.to_datetime('2015-01-19')), atol=2e-6))
