"""Physical constants used by the model regardless of physics package.

These are general physical constants shared across SPEEDY, ICON/ECHAM, and any
future physics packages. Package-specific tunables live with that package.

Design
------
* **Single source of truth, one name per quantity.** Each *independent*
  physical quantity is a field of :class:`PhysicalConstants` exactly once, under
  a single canonical name (no aliases — e.g. dry-air specific heat is ``cpd``,
  the dry-air gas constant ``rd``, the melting point ``tmelt``). Quantities that
  are algebraically derived from others (``rd = akap·cpd``, ``cvd = cpd - rd``,
  ``rgrav = 1/grav``, the ``vtmpc*`` moisture coefficients) are exposed as
  :class:`property` objects. Because they recompute on access, they can never
  drift out of sync with the bases — even after an override.

* **Process-global override.** The module owns a single live singleton,
  :data:`physical_constants`. Call :func:`set_constants` to rebind it (e.g. for
  a different planet or a sensitivity study). A module-level ``__getattr__``
  forwards bare-name access (``jcm.constants.grav``) to that singleton, so any
  consumer that reads constants by *attribute access* — ``import jcm.constants
  as c; ... c.grav`` — sees overrides applied with :func:`set_constants`.

  Important: ``from jcm.constants import grav`` binds the *value* at import time
  and will NOT track later overrides. Consumers that need to honour overrides
  must use attribute access on the module (or read fields off a
  :class:`PhysicalConstants` instance threaded in explicitly). Override before
  constructing the model.
"""

from typing import NamedTuple


class PhysicalConstants(NamedTuple):
    """Physical constants used across atmospheric physics packages.

    Only *independent* quantities are fields (and therefore pytree leaves);
    derived quantities and aliases are properties, so they always reflect the
    current base values — including after ``_replace``/override.
    """

    # --- Fundamental --------------------------------------------------------
    rearth: float = 6.371e+6           # Radius of Earth (m)
    omega: float = 7.292e-05           # Rotation rate of Earth (rad/s)
    grav: float = 9.81                 # Gravitational acceleration (m/s²)
    karman_const: float = 0.4          # von Kármán constant (dimensionless)
    ak: float = 1.3806504e-23          # Boltzmann constant (J/K)

    # --- Heat capacities (J/K/kg) & kappa -----------------------------------
    # cpd is the higher-precision ECHAM-6.3 dry-air value, shared with SPEEDY.
    cpd: float = 1004.64               # Dry air, constant pressure
    cpv: float = 1869.46               # Water vapor, constant pressure
    akap: float = 2.0 / 7.0            # kappa = R/cp for a diatomic gas

    # --- Gas constant for water vapor (J/K/kg) ------------------------------
    rv: float = 461.0
    eps: float = 0.622                 # Ratio of molecular weights (Md/Mv)

    # --- Reference pressures (Pa) -------------------------------------------
    # Two distinct quantities, not a duplicate:
    #   p0      thermodynamic reference pressure (Exner / potential temperature)
    #   p0s1_bg standard mean sea-level pressure (ICAO), used e.g. as the base
    #           pressure in saturation-vapor formulas and hydrostatic init.
    p0: float = 1.0e+5
    p0s1_bg: float = 101325.0

    # --- Latent heats (J/kg) ------------------------------------------------
    alhc: float = 2.501e6              # Condensation
    alhs: float = 2.834e6              # Sublimation
    alhf: float = 3.34e5               # Fusion

    # --- Radiation ----------------------------------------------------------
    sbc: float = 5.67e-8               # Stefan-Boltzmann constant (W/m²/K⁴)
    solc: float = 1361.0               # Solar constant (W/m²)

    # --- Reference temperature (K) ------------------------------------------
    tmelt: float = 273.15              # Melting point of ice

    # --- Cloud microphysics densities (kg/m³) -------------------------------
    rhow: float = 1000.0               # Liquid water
    rhoi: float = 917.0                # Ice

    # --- Numerical ----------------------------------------------------------
    epsilon: float = 1e-12             # Small number to prevent division by zero

    # --- Derived quantities (recompute from the fields above) ---------------
    @property
    def rgrav(self) -> float:
        """Reciprocal of gravity (s²/m)."""
        return 1.0 / self.grav

    @property
    def rd(self) -> float:
        """Dry-air gas constant (J/K/kg). R = akap·cpd, so R/cp == akap exactly.

        With the defaults this is 287.04 — the ECHAM-6.3 value.
        """
        return self.akap * self.cpd

    @property
    def cvd(self) -> float:
        """Dry air, constant volume (J/K/kg)."""
        return self.cpd - self.rd

    @property
    def cvv(self) -> float:
        """Water vapor, constant volume (J/K/kg)."""
        return self.cpv - self.rv

    @property
    def vtmpc1(self) -> float:
        """Virtual-temperature / buoyancy coefficient (rv/rd - 1)."""
        return self.rv / self.rd - 1.0

    @property
    def vtmpc2(self) -> float:
        """Moist heat-capacity coefficient (cpv/cpd - 1)."""
        return self.cpv / self.cpd - 1.0

    @classmethod
    def default(cls) -> 'PhysicalConstants':
        """Return the default physical constants."""
        return cls()


# The single live singleton. Read it (or use module attribute access, which
# forwards here) rather than caching individual values, so overrides are seen.
physical_constants = PhysicalConstants.default()


def set_constants(constants: PhysicalConstants | None = None, **overrides) -> PhysicalConstants:
    """Override the process-global physical constants and return the new set.

    Call this *before* constructing a model so the dynamical core (which reads
    the live singleton at construction) and any attribute-access consumers pick
    up the override. Only *base* fields may be overridden by keyword — derived
    quantities (rd, cvd, rgrav, the vtmpc*, and the aliases) recompute
    automatically.

    Examples
    --------
    >>> import jcm.constants as c
    >>> c.set_constants(grav=9.80665)           # tweak one base value
    >>> c.set_constants(PhysicalConstants(...))  # or replace wholesale

    Note: ``from jcm.constants import grav`` captures a value at import time and
    will not track this. Use attribute access (``c.grav``) instead.

    """
    global physical_constants
    if constants is not None:
        if overrides:
            raise ValueError("Pass either a PhysicalConstants instance or keyword overrides, not both.")
        physical_constants = constants
    elif overrides:
        physical_constants = physical_constants._replace(**overrides)
    return physical_constants


def __getattr__(name: str):
    """Forward bare-name access (e.g. ``jcm.constants.grav``) to the singleton.

    Invoked only for names not defined at module level (PEP 562), so it serves
    every field/derived/alias on :data:`physical_constants` while letting
    ``PhysicalConstants``/``physical_constants``/``set_constants`` resolve
    normally.
    """
    try:
        return getattr(physical_constants, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
