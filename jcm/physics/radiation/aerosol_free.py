"""Vocabulary and validation for the aerosol-free (``*noa``) diagnostic.

Split out of :mod:`jcm.physics.radiation.rrtmgp` so that
:func:`jcm.physics.echam.echam_terms.echam_physics` can validate the
setting for *every* radiation scheme. Importing ``rrtmgp`` eagerly loads
the RRTMGP gas-optics tables and requires the optional ``jax-rrtmgp``
dependency, so a grey-radiation config must not have to pay for it just to
be told it passed a nonsensical flag — which is precisely how the
``aerosol_free_interval`` argument used to be accepted and silently
ignored on the grey and emulated paths.
"""

#: How the aerosol-free (``*noa``) TOA fluxes for AeroCom ERFari are
#: produced. ONE name per behaviour. The previous API spelled this as
#: three independent flags (an on/off bool, an ``alternate`` bool and an
#: integer interval), which made two thirds of the combinations illegal
#: and needed explicit guards to reject them — a closed vocabulary makes
#: those states unrepresentable instead of merely rejected.
#:
#:   ``off``          no ``*noa`` fluxes, no ERFari, no extra cost
#:   ``exact``        companion solve every radiation step — the reference
#:   ``paired``       companion every ``aerosol_free_interval`` steps
#:   ``alternating``  no companion; alternate the single solve instead
AEROSOL_FREE_MODES = ("off", "exact", "paired", "alternating")

#: What each approximation costs you. Measured on a 155-day ERA5-nudged
#: T63L47 semi-Lagrangian run (jax-gcm#630) against a reference ERFari of
#: -0.766 W/m2, with a floor of 0.0023 W/m2 from a second run of ``exact``
#: itself differing only in node and A100 variant. NOTE that the `paired`
#: fidelity figure is stale — see below and jax-gcm#648. Full context and
#: caveats: ``docs/source/design/aerocom_erfari_sampling.md``.
MODE_EVIDENCE = {
    "paired": (
        "+17 % runtime. The SIMULATION stays bit-identical to 'exact' — only "
        "the diagnostic is approximated, by holding the aerosol effect "
        "between companion solves. Its ERFari error is currently "
        "UNQUANTIFIED: the 0.095 W/m2 measured at N=4 predates three fixes "
        "to the hold (a dark companion erased the held fraction, a twilight "
        "ratio could fabricate a ~150 W/m2 effect, and the division NaN'd "
        "gradients), all of which pushed the error in that direction, so "
        "treat 0.095 as a stale upper bound until jax-gcm#648 re-measures"),
    "alternating": (
        "+0 % runtime; ERFari off by 0.067 W/m2 (9 %, ~30x the floor) AND "
        "the model feels aerosol-free heating half the time, so the "
        "SIMULATION itself differs — every output is affected, not just the "
        "*noa fluxes"),
}


def resolve_aerosol_free(mode: str, interval: int | None) -> int:
    """Validate a mode/interval pair and return the companion spacing.

    Returns the interval the compute path should use (1 for every mode
    except ``paired``). Raises :class:`ValueError` with a message that
    names the alternatives, since the mode string is the whole API now and
    a typo must not fall back to a silent default.
    """
    if mode not in AEROSOL_FREE_MODES:
        raise ValueError(
            f"aerosol_free={mode!r} is not one of {AEROSOL_FREE_MODES}. Use "
            "'exact' for the reference ERFari, 'paired' + "
            "aerosol_free_interval=N to trade fidelity for runtime, or "
            "'off' for no *noa fluxes.")
    # The interval is subordinate to the mode: it is the parameter of
    # 'paired' and meaningless elsewhere. Requiring it exactly when it
    # applies means neither "paired but I forgot N" nor "N set on a mode
    # that ignores it" can reach a run silently.
    if mode != "paired":
        if interval is not None:
            raise ValueError(
                f"aerosol_free_interval only applies to aerosol_free="
                f"'paired', not {mode!r}.")
        return 1
    if interval is None:
        raise ValueError(
            "aerosol_free='paired' needs aerosol_free_interval=N (radiation "
            "steps between companion solves). For N=1 use "
            "aerosol_free='exact' — that is the same scheme, named for what "
            "it is.")
    if int(interval) < 2:
        raise ValueError(
            "aerosol_free_interval must be >= 2 for 'paired'; interval 1 IS "
            "aerosol_free='exact'.")
    return int(interval)


#: Smallest all-sky TOA flux [W/m2] from which the held aerosol-effect
#: FRACTION may be derived. Below this the ratio is dominated by the
#: near-terminator geometry that produced it and is meaningless when
#: re-applied to a sunlit flux, so the previous fraction is kept instead.
#: Longwave fluxes (~150-350 W/m2) never approach it; this only ever gates
#: the shortwave.
NOA_FRAC_MIN_FLUX = 1.0


def update_effect_fraction(allsky, noa, prev_frac):
    """Aerosol effect as a fraction of the all-sky flux, from a companion.

    Pure arithmetic, kept out of the RRTMGP compute path so it can be
    tested against dark and near-terminator inputs directly — driving those
    conditions through a full solve requires controlling solar geometry,
    which is exactly how an earlier version of this logic shipped with
    three latent bugs.

    Returns ``prev_frac`` wherever ``allsky`` is too small for the ratio to
    mean anything, so a companion that lands on a dark column leaves the
    stored fraction intact instead of overwriting it with a meaningless
    value.
    """
    import jax.numpy as jnp

    usable = jnp.abs(allsky) > NOA_FRAC_MIN_FLUX
    # Double `where`: the masked branch must not evaluate the division even
    # under grad. A single `where` leaves a NaN primal whose VJP returns
    # 0/0 regardless of the incoming cotangent (jax-gcm#558/#559).
    denom = jnp.where(usable, allsky, 1.0)
    ratio = (allsky - noa) / denom
    # The clamp is a guard, not physics: an aerosol effect exceeding the
    # whole all-sky flux means the ratio came from a flux too small to
    # trust, and applying it to a sunlit column would fabricate a huge
    # forcing.
    return jnp.where(usable, jnp.clip(ratio, -1.0, 1.0), prev_frac)


def apply_effect_fraction(fresh, frac):
    """Re-apply a held effect fraction to a fresh all-sky flux.

    Scale-free, so a dark column reconstructs to exactly zero rather than
    subtracting a stale daytime effect from a zero flux.
    """
    return fresh * (1.0 - frac)


def hold_all(fresh, frac):
    """Apply each key's own held fraction to that key's fresh flux.

    ``fresh`` is one value per ``*noa`` key in the canonical order
    (toa_sw_up, toa_lw_up, toa_sw_up_clear, toa_lw_up_clear); ``frac`` is
    the matching leading axis of ``RadiationData.noa_effect_frac``.

    Trivial, but it exists as a named function so the key-to-fraction
    pairing is testable on its own: an adversarial review showed that
    making every slot hold the *longwave* fraction — which would corrupt
    rsutnoa, rsutcsnoa and rlutcsnoa in every `paired` run — passed the
    entire integration suite, because the shortwave slots are identically
    zero in the test fixture's solar geometry.
    """
    return tuple(apply_effect_fraction(f, frac[i])
                 for i, f in enumerate(fresh))
