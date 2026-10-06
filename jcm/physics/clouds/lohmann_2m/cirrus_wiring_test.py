"""Kaercher-Lohmann cirrus wired into the 2M column sweep (#552, jax-gcm#1017
task 2 part 2b).

Complements ``cirrus_reference_test.py`` (the bare ``xfrzmstr`` port) and
``lohmann_2m_test.py``'s ``TestUpdateInCloudWater_2M`` (the ``pnicex``/
aerosol-number-cap unit tests) with the thing neither isolates: that
``cloud_microphysics_2m`` -- the full per-column sweep -- actually produces a
non-zero ICNC from genuine homogeneous nucleation when `nic_cirrus=2` and a
population publishes `cirrus_aerosol_number`, not merely a floor value.
"""
from __future__ import annotations

import jax.numpy as jnp

from jcm.physics import thermodynamics
from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
from jcm.physics.clouds.lohmann_2m_params import CloudParams2M


def _cold_supersaturated_column(nlev=8, cirrus_level=3):
    """Return a column with one cold, ice-supersaturated, thinly-iced level.

    The other levels are warm and dry (inert): only ``cirrus_level`` is
    designed to exercise the cirrus path. A little pre-existing cloud ice
    and cover (well above ``cqtmin``/``clc_min``) are seeded there so the
    test exercises ``update_in_cloud_water``'s ``ll2_ic`` candidate-update
    branch under realistic column-sweep conditions, not a hand-built
    single-call ``update_in_cloud_water`` input (that is already covered,
    in isolation, by ``lohmann_2m_test.py``).
    """
    t = jnp.linspace(260.0, 290.0, nlev)
    p = jnp.linspace(2.5e4, 1.0e5, nlev)
    t = t.at[cirrus_level].set(210.0)
    p = p.at[cirrus_level].set(2.5e4)
    rho = p / (287.0 * t)
    qsat_ice, _ = thermodynamics.saturation_specific_humidity_and_derivative(
        t, p, phase="auto")
    q = 0.3 * qsat_ice
    # SCRHOM(210 K) = 2.418 - 210/245.68 ~= 1.563, i.e. susati ~= 0.563 is
    # the homogeneous threshold at this level's temperature; 1.8x clears it
    # with margin (susati = 0.8).
    q = q.at[cirrus_level].set(1.8 * qsat_ice[cirrus_level])  # supersaturated
    qc = jnp.zeros(nlev)
    qi = jnp.zeros(nlev).at[cirrus_level].set(5e-6)
    cf = jnp.zeros(nlev).at[cirrus_level].set(0.3)
    qnc = jnp.zeros(nlev)
    qni = jnp.zeros(nlev).at[cirrus_level].set(10.0)  # tiny -- icnc <= icemin
    dz = jnp.full(nlev, 500.0)
    tke = jnp.full(nlev, 0.3)
    return dict(t=t, q=q, p=p, qc=qc, qi=qi, qnc=qnc, qni=qni, cf=cf,
                rho=rho, dz=dz, tke=tke)


def test_cirrus_homogeneous_nucleation_lifts_icnc_above_the_floor():
    """nic_cirrus=2 with a non-zero aerosol source nucleates real ICNC.

    The #552 regression this guards: before the fix, ``pnicex`` was always
    zero (no producer wired the slot) and the nic_cirrus=2 candidate capped
    at zero, pinning ICNC at the ``icemin`` floor in every pure-ice cloud
    regardless of supersaturation or aerosol loading.
    """
    nlev, k = 8, 3
    col = _cold_supersaturated_column(nlev, k)
    params = CloudParams2M.default().replace(nic_cirrus=2)

    tend, *_ = cloud_microphysics_2m(
        col["t"], col["q"], col["p"], col["qc"], col["qi"], col["qnc"],
        col["qni"], col["cf"], col["rho"], col["dz"], col["tke"],
        jnp.zeros(nlev), jnp.zeros(nlev), jnp.zeros(nlev), 1200.0, params,
        cirrus_aerosol_number=jnp.full(nlev, 1.0e9),
    )
    icnc = col["qni"][k] + tend.dqnidt[k] * 1200.0
    assert jnp.isfinite(icnc)
    assert float(icnc) > float(params.icemin) * 10.0, (
        "cirrus nucleation should lift ICNC well above the floor, not pin it there")

    # A thousandfold LESS available aerosol gives a smaller candidate (CI's
    # own CTOT cap, XICEHOM's ``CI = MIN(CI, CTOT)``, binds in this regime
    # -- the harness's own designed cells confirm 1e6 m-3 is below where it
    # saturates, see hamcirrus_README.md): isolates the sensitivity to the
    # aerosol-number channel specifically, distinguishing it from this cold
    # column's OTHER ice sources (DeMott/contact freezing, the <238 K
    # freeze-all path), which fire regardless of ``cirrus_aerosol_number``
    # and would otherwise make a bare "with vs. zero aerosol" comparison
    # noisy or, past the cap's saturation point, insensitive to a further
    # increase.
    tend_less, *_ = cloud_microphysics_2m(
        col["t"], col["q"], col["p"], col["qc"], col["qi"], col["qnc"],
        col["qni"], col["cf"], col["rho"], col["dz"], col["tke"],
        jnp.zeros(nlev), jnp.zeros(nlev), jnp.zeros(nlev), 1200.0, params,
        cirrus_aerosol_number=jnp.full(nlev, 1.0e6),
    )
    icnc_less = col["qni"][k] + tend_less.dqnidt[k] * 1200.0
    assert float(icnc_less) < float(icnc)


def test_nic_cirrus_1_default_is_bit_identical_regardless_of_cirrus_aerosol_number():
    """MAM4-neutral: today's default (nic_cirrus=1) never reads the new input.

    Every existing preset uses nic_cirrus=1; this is the test's own evidence
    for that (not just an assertion elsewhere) -- the bitid probe is the
    repo-wide version of this same claim.
    """
    nlev, k = 8, 3
    col = _cold_supersaturated_column(nlev, k)
    params = CloudParams2M.default()
    assert params.nic_cirrus == 1

    args = (col["t"], col["q"], col["p"], col["qc"], col["qi"], col["qnc"],
            col["qni"], col["cf"], col["rho"], col["dz"], col["tke"],
            jnp.zeros(nlev), jnp.zeros(nlev), jnp.zeros(nlev), 1200.0, params)
    tend_a, *_ = cloud_microphysics_2m(*args)
    tend_b, *_ = cloud_microphysics_2m(
        *args, cirrus_aerosol_number=jnp.full(nlev, 1.0e10))
    assert jnp.array_equal(tend_a.dqnidt, tend_b.dqnidt)
    assert jnp.array_equal(tend_a.dqidt, tend_b.dqidt)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
