"""The deep plume's condensate ledger against a literal ``cuasc`` transliteration.

``cuasc_port_test`` pins the ascent's rules one at a time. This test pins
their composition: a deep (``ktype = 1``) plume on a tropical L47 column is
run through :func:`calculate_updraft` and through a line-by-line numpy
transliteration of ECHAM's ``cuasc`` section 3.5-5 and ``cuentr``
(mo_cuascent.f90:243-565, 719-788; Fortran 1-based indexing kept), and the
per-layer mass flux ``pmfu``, condensate flux ``pmful``, precipitation
formation ``pdmfup`` and detrained condensate ``plude`` must agree.

What the transliteration takes as given, because each has its own reference
test: the ``cuini`` half-level environment, the ``cubase`` parcel at
``kcbot`` with its sub-cloud ``zbuoy``, ``cumastr``'s ``ictop0``/``khmin``/
``zhhatt``/``zhcbase``, and ``cuadjtq`` (kcall = 1). The ledger these
compose is where the plume's condensate goes (in-plume precipitation via
``cprcon``, turbulent/organized/cloud-top detrainment), which sets the
convective ice supply of the upper troposphere.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.half_level_ledger_test import (
    _l47_tropical_column,
)
from jcm.physics.convection.tiedtke_nordeng.half_levels import (
    cubase_seed_static_energy,
)
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    find_cloud_base,
)
from jcm.physics.convection.tiedtke_nordeng.types import ConvectionParameters
from jcm.physics.convection.tiedtke_nordeng.updraft import (
    calculate_updraft,
    cloud_base_mse,
    column_environment,
    cubase_parcel,
    estimate_cloud_top,
    mse_minimum_level,
    saturated_mse_hat,
    saturation_adjustment,
)
from jcm.physics.thermodynamics import moist_isobaric_heat_capacity


def _cuasc_deep(klev, *, pten, pgeo, pgeoh, paphp1, ptenh, pqenh, pqsenh,
                pcpcu, kcbot, kctop0, khmin, klwmin, phhatt, phcbase, pmfub,
                tu_cb, qu_cb, lu_cb, zbuoy_sub, cuadjtq, k):
    """``cuasc`` + ``cuentr`` for one ocean deep plume; arrays 1-based."""
    grav, rd, rv, vtmpc1 = c.grav, c.rd, c.rv, c.vtmpc1
    zcons2 = 1.0 / (grav * k["dt"])
    ptu, pqu = ptenh.copy(), pqenh.copy()
    z = lambda: np.zeros(klev + 2)  # noqa: E731
    plu, pmfu, pmfus, pmfuq, pmful = z(), z(), z(), z(), z()
    plude, pdmfup, zoentr, zodetr = z(), z(), z(), z()
    # The cubase parcel at kcbot and its fluxes (section 3.0).
    ptu[kcbot], pqu[kcbot], plu[kcbot] = tu_cb, qu_cb, lu_cb
    pmfu[kcbot] = pmfub
    pmfus[kcbot] = pmfub * (pcpcu[kcbot] * ptu[kcbot] + pgeoh[kcbot])
    pmfuq[kcbot] = pmfub * pqu[kcbot]
    pmful[kcbot] = pmfub * plu[kcbot]
    zpbase = paphp1[kcbot]
    # Section 3.5 plus the sub-cloud and kcbot passes of l.512-526.
    zbuoy = (grav * (ptu[kcbot] - ptenh[kcbot]) / ptenh[kcbot]
             + grav * vtmpc1 * (pqu[kcbot] - pqenh[kcbot]) + zbuoy_sub)

    def organized_entrainment(jk, zbuoy):
        zbuoyz = max(grav * (ptu[jk] - ptenh[jk]) / ptenh[jk]
                     + grav * vtmpc1 * (pqu[jk] - pqenh[jk])
                     - grav * plu[jk], 0.0)
        zdz = (pgeo[jk - 1] - pgeo[jk]) / grav
        zdrodz = (-np.log(pten[jk - 1] / pten[jk]) / zdz
                  - grav / (rd * ptenh[jk] * (1.0 + vtmpc1 * pqenh[jk])))
        zbuoy = zbuoy + zbuoyz * zdz
        zoentr[jk - 1] = min(max(zbuoyz * 0.5 / (1.0 + zbuoy) + zdrodz, 0.0),
                             k["centrmax"])
        return zbuoy

    zbuoy = organized_entrainment(kcbot, zbuoy)
    kctop = kcbot
    alive = True
    for jk in range(kcbot - 1, 1, -1):
        if not alive:
            break
        # cuentr (kk = jk), deep branch.
        zrrho = (rd * ptenh[jk + 1] * (1.0 + vtmpc1 * pqenh[jk + 1])
                 / paphp1[jk + 1])
        zdprho = (paphp1[jk + 1] - paphp1[jk]) / grav
        zpmid = 0.5 * (zpbase + paphp1[kctop0])
        icond1 = (zpmid - paphp1[jk]) < 0.0
        zentr = k["entrpen"] * pmfu[jk + 1] * zdprho * zrrho
        zdmfde = zentr
        zdmfen = zentr if (jk >= max(klwmin, kctop0 + 2) or icond1) else 0.0
        if kctop0 <= jk <= khmin and khmin > kctop0:
            zzmzk = -(pgeoh[khmin] - pgeoh[jk]) / grav
            ztmzk = -(pgeoh[khmin] - pgeoh[kctop0]) / grav
            zorgde = (np.tan(3.1415 * (zzmzk / ztmzk) * 0.5)
                      * 3.1415 * 0.5 / ztmzk)
            zodetr[jk] = min(zorgde, k["centrmax"]) * pmfu[jk + 1] * (
                (paphp1[jk + 1] - paphp1[jk]) / grav * zrrho)
        # cuasc l.351-421.
        zmfmax_k = (paphp1[jk] - paphp1[jk - 1]) * zcons2
        zmftest = pmfu[jk + 1] + zdmfen - zdmfde
        zdmfen = max(zdmfen - max(zmftest - min(zmftest, zmfmax_k), 0.0), 0.0)
        zdmfde = min(zdmfde, 0.75 * pmfu[jk + 1])
        pmfu[jk] = pmfu[jk + 1] + zdmfen - zdmfde
        zdz_g = (pgeoh[jk] - pgeoh[jk + 1]) / grav
        zoentr[jk] = zoentr[jk] * zdz_g * pmfu[jk + 1]
        zmftest = pmfu[jk] + zoentr[jk] - zodetr[jk]
        zoentr[jk] = max(
            zoentr[jk] - max(zmftest - min(zmftest, zmfmax_k), 0.0), 0.0)
        if jk <= khmin:
            zalvs = c.alhc if ptu[jk + 1] > c.tmelt else c.alhs
            zmse = (pcpcu[jk + 1] * ptu[jk + 1] + zalvs * pqu[jk + 1]
                    + pgeoh[jk + 1])
            znevn = (pgeoh[kctop0] - pgeoh[jk + 1]) * (
                zmse - phhatt[jk + 1]) / grav
            znevn = 1.0 if znevn <= 0.0 else znevn
            zodetr[jk] = min(zodetr[jk], max(
                (phcbase - zmse) / znevn * zdz_g * pmfu[jk + 1], 0.0))
        zodetr[jk] = min(zodetr[jk], 0.75 * pmfu[jk])
        pmfu[jk] = pmfu[jk] + zoentr[jk] - zodetr[jk]
        ent = zdmfen + zoentr[jk]
        s_env = pcpcu[jk + 1] * ptenh[jk + 1] + pgeoh[jk + 1]
        zalvs = c.alhc if ptenh[jk + 1] > c.tmelt else c.alhs
        zga = zalvs * pqsenh[jk + 1] / (rv * ptenh[jk + 1] ** 2)
        zdt = ((plu[jk + 1] - vtmpc1 * (pqsenh[jk + 1] - pqenh[jk + 1]))
               / (1.0 / ptenh[jk + 1] + vtmpc1 * zga))
        zscod = max(s_env + pcpcu[jk + 1] * zdt, 0.0)
        zqcod = max(pqsenh[jk + 1] + zga * zdt, 0.0)
        zscde = ((pcpcu[jk + 1] * ptu[jk + 1] + pgeoh[jk + 1]) * zdmfde
                 + zodetr[jk] * zscod)
        zqude = pqu[jk + 1] * zdmfde + zodetr[jk] * zqcod
        plude[jk] = plu[jk + 1] * (zdmfde + zodetr[jk])
        mf = max(k["cmfcmin"], pmfu[jk])
        plu[jk] = (pmful[jk + 1] - plude[jk]) / mf
        pqu[jk] = (pmfuq[jk + 1] + pqenh[jk + 1] * ent - zqude) / mf
        ptu[jk] = min(400.0, max(100.0, (
            (pmfus[jk + 1] + s_env * ent - zscde) / mf - pgeoh[jk])
            / pcpcu[jk]))
        zqold = pqu[jk]
        ptu[jk], pqu[jk] = cuadjtq(ptu[jk], pqu[jk], paphp1[jk])
        alive = False
        if pqu[jk] < zqold:
            plu[jk] += zqold - pqu[jk]
            zbuo = (ptu[jk] * (1.0 + vtmpc1 * pqu[jk] - plu[jk])
                    - ptenh[jk] * (1.0 + vtmpc1 * pqenh[jk]))
            if zbuo > 0.0 and pmfu[jk] >= 0.01 * pmfub and jk >= kctop0:
                alive = True
                kctop = jk
                zprcon = (0.0 if zpbase - paphp1[jk] < k["zdnoprc"]
                          else k["cprcon"])
                zlnew = plu[jk] / (1.0 + zprcon * (pgeoh[jk] - pgeoh[jk + 1]))
                pdmfup[jk] = max(0.0, (plu[jk] - zlnew) * pmfu[jk])
                plu[jk] = zlnew
        if not alive:
            pmfu[jk] = 0.0
        pmful[jk] = plu[jk] * pmfu[jk]
        pmfus[jk] = (pcpcu[jk] * ptu[jk] + pgeoh[jk]) * pmfu[jk]
        pmfuq[jk] = pqu[jk] * pmfu[jk]
        if alive:
            zbuoy = organized_entrainment(jk, zbuoy)
    # Section 5: the cloud-top overshoot.
    jk = kctop - 1
    zdmfde = (1.0 - k["cmfctop"]) * pmfu[jk + 1]
    plude[jk] = zdmfde * plu[jk + 1]
    pmfu[jk] = pmfu[jk + 1] - zdmfde
    pdmfup[jk] = 0.0
    pmful[jk] = plu[jk] * pmfu[jk]
    if jk >= 2:
        plude[jk - 1] = pmful[jk]
    return dict(pmfu=pmfu, pmful=pmful, pdmfup=pdmfup, plude=plude,
                kctop=kctop)


class TestDeepPlumeLedgerAgainstCuasc(unittest.TestCase):

    def test_per_layer_ledger_matches_the_transliteration(self):
        with jax.enable_x64(True):
            T, q, p, ph, _, _ = _l47_tropical_column(rh=0.8)
            T, q, p, ph = (jnp.asarray(a, jnp.float64) for a in (T, q, p, ph))
            nlev = T.shape[0]
            cfg = ConvectionParameters.default()
            cp = moist_isobaric_heat_capacity(q)
            env = column_environment(T, q, p, cp, pressure_half=ph)
            kb, has = find_cloud_base(T, q, p, cfg, None, None,
                                      cp_moist=cp, env=env)
            self.assertTrue(bool(has))
            kb = int(kb)
            tu_cb, qu_cb, lu_cb = cubase_parcel(env, kb)
            hhatt = saturated_mse_hat(env)
            hcb = cloud_base_mse(env, kb, tu_cb, qu_cb)
            ictop0 = int(estimate_cloud_top(hhatt, hcb, kb))
            khmin = int(mse_minimum_level(env, T, q, cp, kb, ictop0))
            klwmin, mfub, dt = nlev - 1, 0.02, 720.0
            up = calculate_updraft(
                T, q, p, None, None, kb, ictop0, 1, mfub, cfg, cp_moist=cp,
                env=env, dt=dt, klwmin=jnp.asarray(klwmin),
                khmin=jnp.asarray(khmin))

            # The sub-cloud Nordeng buoyancy the plume leaves kcbot with
            # (cuasc l.512-523 over klevm1..kcbot+1).
            geo, geoh = np.asarray(env.geo), np.asarray(env.geoh)
            tenh, qenh = np.asarray(env.tenh), np.asarray(env.qenh)
            t_sub = ((np.asarray(cubase_seed_static_energy(env)) - geoh)
                     / np.asarray(env.cpcu))
            zbz = np.maximum(c.grav * (t_sub - tenh) / tenh
                             + c.grav * c.vtmpc1 * (qenh[-1] - qenh), 0.0)
            zdz = (np.concatenate([geo[:1], geo[:-1]]) - geo) / c.grav
            lv = np.arange(nlev)
            zbuoy_sub = float(np.sum(
                np.where((lv > kb) & (lv <= nlev - 2), zbz * zdz, 0.0)))

            def cuadjtq(t, qv, pp):
                tn, qn, _ = saturation_adjustment(
                    jnp.asarray(t), jnp.asarray(qv), jnp.asarray(pp))
                return float(tn), float(qn)

            def one(a):
                return np.concatenate([[np.nan], np.asarray(a, np.float64),
                                       [np.nan]])

            ref = _cuasc_deep(
                nlev, pten=one(T), pgeo=one(geo), pgeoh=one(geoh),
                paphp1=np.concatenate([[np.nan], np.asarray(env.paph)]),
                ptenh=one(tenh), pqenh=one(qenh), pqsenh=one(env.qsenh),
                pcpcu=one(env.cpcu), kcbot=kb + 1, kctop0=ictop0 + 1,
                khmin=khmin + 1, klwmin=klwmin + 1, phhatt=one(hhatt),
                phcbase=float(hcb), pmfub=mfub, tu_cb=float(tu_cb),
                qu_cb=float(qu_cb), lu_cb=float(lu_cb), zbuoy_sub=zbuoy_sub,
                cuadjtq=cuadjtq,
                k=dict(dt=dt, entrpen=float(cfg.entrpen),
                       centrmax=float(cfg.cu_centrmax),
                       cmfcmin=float(cfg.cmfcmin),
                       cmfctop=float(cfg.cu_cmfctop),
                       cprcon=float(cfg.cprcon),
                       zdnoprc=float(cfg.cu_dnoprc_ocean)))

        kctop = int(up.kctop)
        self.assertEqual(kctop, ref["kctop"] - 1)
        # A genuinely deep plume: it precipitates and reaches the upper
        # troposphere, so every branch of the ledger is exercised.
        self.assertLess(float(env.paph[kctop]), 3.0e4)
        sl = slice(kctop - 2, kb + 1)
        f = lambda name: ref[name][1:nlev + 1][sl]  # noqa: E731
        mfu, lu = np.asarray(up.mfu)[sl], np.asarray(up.lu)[sl]
        scale = mfub * 1e-3
        np.testing.assert_allclose(mfu, f("pmfu"), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(mfu * lu, f("pmful"), rtol=1e-9,
                                   atol=1e-9 * scale)
        np.testing.assert_allclose(np.asarray(up.pdmfup)[sl], f("pdmfup"),
                                   rtol=1e-9, atol=1e-9 * scale)
        np.testing.assert_allclose(np.asarray(up.plude)[sl], f("plude"),
                                   rtol=1e-9, atol=1e-9 * scale)
        self.assertGreater(float(np.sum(np.asarray(up.pdmfup))), 0.0)


if __name__ == "__main__":
    unittest.main()
