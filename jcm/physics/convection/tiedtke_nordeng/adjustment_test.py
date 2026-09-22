"""Unit tests for convective adjustment module

Date: 2025-01-10
"""

import jax.numpy as jnp
from .adjustment import cuadjtq, _qsat_and_dqsat_dt


class TestCuadjtq:
    """The linearised Newton-step saturation adjustment (port of
    ``mo_cuadjust.f90`` ``cuadjtq``).
    """

    def test_no_adjustment_for_subsaturated_input(self):
        """``kcall=1`` (condensation only) must leave subsaturated air
        unchanged: ``q < q_sat`` → ``cond = 0``.
        """
        T = jnp.array(280.0)
        p = jnp.array(90000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 0.5 * qs  # 50 % RH
        T_adj, q_adj, cond = cuadjtq(T, q, p, kcall=1)
        assert jnp.allclose(T_adj, T)
        assert jnp.allclose(q_adj, q)
        assert jnp.allclose(cond, 0.0)

    def test_modest_supersat_lands_close_to_saturation(self):
        """The Newton step is first-order accurate: at modest (10 %)
        supersaturation the two-pass ``cuadjtq`` lands within 1 % of
        saturation. (At very high supersaturation the linearisation
        leaves more residual — that's expected ECHAM behaviour and
        covered by ``test_strong_supersat_lands_subsaturated_not_super``.)
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.10 * qs
        T_adj, q_adj, _ = cuadjtq(T, q, p, kcall=1)
        qs_adj, _ = _qsat_and_dqsat_dt(T_adj, p)
        rh = float(q_adj / qs_adj)
        assert 0.99 <= rh <= 1.01, f"post-cuadjtq RH = {rh*100:.2f} %"

    def test_strong_supersat_lands_subsaturated_not_super(self):
        """At 50 % supersaturation the Newton step over-warms in the
        first pass; the ``kcall=1`` clip then prevents the second pass
        from re-evaporating. Final state must be subsaturated (the
        scheme is conservative — it never leaves the column above
        ``q_sat`` for the caller's downstream physics).
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.5 * qs
        T_adj, q_adj, _ = cuadjtq(T, q, p, kcall=1)
        qs_adj, _ = _qsat_and_dqsat_dt(T_adj, p)
        rh = float(q_adj / qs_adj)
        assert rh <= 1.0, f"left supersaturated at RH = {rh*100:.2f} %"

    def test_kcall_1_only_condenses(self):
        """``kcall=1`` (cubase / cuasc) must never produce negative
        condensate (no evaporation in the updraft branch).
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        # Subsaturated input: simple form would evaporate (cond < 0)
        q = 0.7 * qs
        _T_adj, _q_adj, cond = cuadjtq(T, q, p, kcall=1)
        assert float(cond) >= 0.0

    def test_kcall_2_only_evaporates(self):
        """``kcall=2`` (cudlfs / cuddraf) must never produce positive
        condensate (no condensation in the downdraft branch).
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.3 * qs
        _T_adj, _q_adj, cond = cuadjtq(T, q, p, kcall=2)
        assert float(cond) <= 0.0

    def test_kcall_0_allows_both_directions(self):
        """``kcall=0`` (cuini env q_sat) lets the Newton step go either
        way — used to align environmental q with q_sat at half levels.
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        # Subsaturated → negative cond (evaporation of imaginary liquid).
        q = 0.7 * qs
        _T1, _q1, cond_dry = cuadjtq(T, q, p, kcall=0)
        assert float(cond_dry) < 0.0
        # Supersaturated → positive cond.
        q = 1.3 * qs
        _T2, _q2, cond_wet = cuadjtq(T, q, p, kcall=0)
        assert float(cond_wet) > 0.0

    def test_moist_static_energy_conserved_per_step(self):
        """``cuadjtq`` releases L·Δq of latent heat per kg of water
        condensed. ``cp·ΔT + L·Δq`` must be conserved up to the
        linearisation residual (sub-1 % at 50 % supersat).
        """
        from jcm.constants import cpd, alhc
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.5 * qs
        T_adj, q_adj, _cond = cuadjtq(T, q, p, kcall=1)
        h_before = cpd * T + alhc * q
        h_after = cpd * T_adj + alhc * q_adj
        rel_imbalance = float(jnp.abs(h_after - h_before) / h_before)
        assert rel_imbalance < 1e-3, (
            f"moist static energy drift = {rel_imbalance*100:.4f} %"
        )

    def test_refine_pass_reduces_residual(self):
        """The ``refine=True`` second iteration must leave the column
        closer to saturation than the single Newton pass.
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 2.0 * qs  # strong supersat — exposes the linearisation residual
        T1, q1, _ = cuadjtq(T, q, p, kcall=1, refine=False)
        T2, q2, _ = cuadjtq(T, q, p, kcall=1, refine=True)
        qs1, _ = _qsat_and_dqsat_dt(T1, p)
        qs2, _ = _qsat_and_dqsat_dt(T2, p)
        rh1 = float(q1 / qs1)
        rh2 = float(q2 / qs2)
        assert abs(rh2 - 1.0) <= abs(rh1 - 1.0)


if __name__ == "__main__":
    t = TestCuadjtq()
    t.test_no_adjustment_for_subsaturated_input()
    t.test_modest_supersat_lands_close_to_saturation()
    t.test_strong_supersat_lands_subsaturated_not_super()
    t.test_kcall_1_only_condenses()
    t.test_kcall_2_only_evaporates()
    t.test_kcall_0_allows_both_directions()
    t.test_moist_static_energy_conserved_per_step()
    t.test_refine_pass_reduces_residual()
