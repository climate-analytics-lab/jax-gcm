"""The aerosol-free (``*noa``) diagnostic: spacing, validation, hold.

Split out of :mod:`jcm.physics.radiation.rrtmgp` so that
:func:`jcm.physics.echam.echam_terms.echam_physics` can validate the
setting for *every* radiation scheme. Importing ``rrtmgp`` eagerly loads
the RRTMGP gas-optics tables and requires the optional ``jax-rrtmgp``
dependency, so a grey-radiation config must not have to pay for it just to
be told it passed a nonsensical value.

The whole API is ONE integer, ``aerosol_free_interval``:

===========  =========================================================
``None``     no ``*noa`` fluxes, no ERFari, no extra cost (default)
``1``        companion solve every radiation step — exact ERFari
``N > 1``    companion every Nth step; the aerosol effect is held in
             between. A monotonic cost/fidelity dial: cost falls and
             error grows with N.
===========  =========================================================
"""


def resolve_aerosol_free_interval(interval: int | None) -> int | None:
    """Validate ``aerosol_free_interval`` and return it normalised.

    ``None`` means the diagnostic is off. Anything else must be a positive
    integer; ``1`` is the exact reference and ``N > 1`` trades fidelity for
    runtime.
    """
    if interval is None:
        return None
    # A bool is an int in Python, and the PREVIOUS API spelled this switch
    # as `aerosol_free_radiation: true` — so `aerosol_free_interval: true`
    # is a live migration typo that would silently mean N=1.
    if isinstance(interval, bool):
        raise ValueError(
            "aerosol_free_interval is a radiation-step count, not a flag "
            f"(got {interval!r}). Use 1 for the exact ERFari reference, "
            "N > 1 to subsample, or leave it unset for no *noa fluxes.")
    value = int(interval)
    # Reject a fractional cadence rather than truncating it: this value
    # decides which radiation calls run the companion solve, so silently
    # turning 2.9 into 2 would change both the cost and the diagnostic
    # without saying so (Codex review on PR #652).
    if value != interval:
        raise ValueError(
            f"aerosol_free_interval must be a whole number of radiation "
            f"steps (got {interval!r}); it selects which calls run the "
            "aerosol-free companion, so a fractional value has no meaning.")
    if value < 1:
        raise ValueError(
            f"aerosol_free_interval must be >= 1 (got {interval!r}). Use 1 "
            "for the exact ERFari reference, N > 1 to run the aerosol-free "
            "companion every Nth radiation step, or leave it unset for no "
            "*noa fluxes at all.")
    return value


#: Seconds in a solar day, for the cadence check below.
_SECONDS_PER_DAY = 86400.0


def check_cadence_precesses(interval, radiation_interval_seconds):
    """Reject a companion cadence locked to the solar day.

    The companion gate is GLOBAL — the same radiation calls run the
    aerosol-free solve for every column — so a column is only ever sampled
    at the local solar times the cadence happens to land on. When
    ``interval x radiation_interval`` is a whole number of days, that is a
    single local time, forever: columns dark at it never produce a usable
    aerosol-effect fraction, their stored fraction stays at its zero
    initial value, and every daylight held step then reports
    ``toa_sw_up_noa == toa_sw_up``. The result is a silent, indefinite
    zero shortwave ERFari over a band of longitudes — valid-looking output
    that is simply wrong.

    Raising is deliberate: the failure is invisible in the output, and the
    remedy is to pick a neighbouring N. Found by Codex review on PR #652;
    the underlying limitation (a global gate cannot guarantee per-column
    daylight sampling) is jax-gcm#653.
    """
    if interval is None or interval <= 1:
        return
    seconds = float(radiation_interval_seconds)
    if seconds <= 0:
        return
    days = interval * seconds / _SECONDS_PER_DAY
    if abs(days - round(days)) > 1e-6 or round(days) < 1:
        return
    hours = interval * seconds / 3600.0
    raise ValueError(
        f"aerosol_free_interval={interval} with radiation_interval="
        f"{seconds:g} s puts the aerosol-free companion every {hours:g} h — "
        f"exactly {round(days)} solar day(s). The companion would then "
        "always land at the same local solar time, so columns dark at that "
        "time never get an aerosol-free sample and would report zero "
        "shortwave ERFari indefinitely. Pick a cadence that is not a whole "
        f"number of days, e.g. {interval - 1} or {interval + 1}.")


#: What subsampling costs you, for the startup warning at N > 1. Measured
#: on a 155-day ERA5-nudged T63L47 semi-Lagrangian run (jax-gcm#630)
#: against a reference ERFari of -0.766 W/m2, with a floor of 0.0023 W/m2
#: from a second run of N=1 differing only in node and A100 variant.
#:
#: STALE: the 0.095 W/m2 was measured before three bugs in the hold were
#: fixed (a dark companion erased the held fraction, a twilight ratio could
#: fabricate a ~150 W/m2 effect, and the division NaN'd gradients), all of
#: which pushed the error in that direction. Treat it as an upper bound
#: until jax-gcm#648 re-measures. Full context:
#: ``docs/source/design/aerocom_erfari_sampling.md``.
SUBSAMPLING_CAVEAT = (
    "the aerosol effect is held between companion solves, so ERFari is "
    "approximate. At N=4 the error measured 0.095 W/m2 (12 % of the "
    "signal, ~40x the run-to-run reproducibility floor) — though that "
    "figure predates three fixes to the hold and is a stale upper bound "
    "(jax-gcm#648). The SIMULATION is bit-identical to N=1 either way: "
    "only the diagnostic is approximated"
)


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


#: The four aerosol-free keys, in the canonical order used by every tuple
#: in this module and by ``RadiationData.noa_frac_*``.
NOA_KEYS = ("toa_sw_up", "toa_lw_up", "toa_sw_up_clear", "toa_lw_up_clear")


def hold_all(fresh, frac):
    """Apply each key's own held fraction to that key's fresh flux.

    ``fresh`` and ``frac`` are both one value per ``*noa`` key in the
    canonical order :data:`NOA_KEYS`.

    Trivial, but it exists as a named function so the key-to-fraction
    pairing is testable on its own: an adversarial review showed that
    making every slot hold the *longwave* fraction — which would corrupt
    rsutnoa, rsutcsnoa and rlutcsnoa in every `paired` run — passed the
    entire integration suite, because the shortwave slots are identically
    zero in the test fixture's solar geometry.
    """
    return tuple(apply_effect_fraction(f, frac[i])
                 for i, f in enumerate(fresh))
