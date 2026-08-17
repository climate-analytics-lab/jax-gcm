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

#: What each approximation actually costs you, measured on a 155-day
#: ERA5-nudged T63L47 semi-Lagrangian run (jax-gcm#630) against a
#: reference ERFari of -0.766 W/m2. The yardstick is a floor of
#: 0.0023 W/m2 — a second run of ``exact`` itself, differing only in node
#: and A100 variant — so both approximations sit ~1.5 orders of magnitude
#: above run-to-run reproducibility. See
#: ``docs/source/design/aerocom_erfari_sampling.md``.
MODE_EVIDENCE = {
    "paired": (
        "+17 % runtime; ERFari off by 0.095 W/m2 (12 % of the signal, ~40x "
        "the reproducibility floor) at N=4, essentially all of it shortwave. "
        "The SIMULATION stays bit-identical to 'exact' — only the diagnostic "
        "is approximated, by holding the aerosol effect between companion "
        "solves"),
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
