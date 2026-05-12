"""Cloud diagnostic sub-struct shared across cloud schemes.

The diagnostic dict key ``"clouds"`` carries a :class:`CloudData` typed
sub-struct that is read and written by the cloud-fraction scheme
(``SundqvistCloudFraction``) and the microphysics schemes
(``Echam1MMicrophysics``, ``Lohmann2MMicrophysics``). It lives next to
those schemes so a downstream user adding a new cloud scheme can extend
or replace it without reaching into the ECHAM tree.
"""

from __future__ import annotations

import jax.numpy as jnp
import tree_math


@tree_math.struct
class CloudData:
    """Cloud-fraction, condensate, and surface-precip diagnostics."""

    # Cloud fraction
    cloud_fraction: jnp.ndarray      # Cloud fraction [1] (nlev, ncols)

    # Cloud condensate (updated by condensation within the cloud scheme)
    qc: jnp.ndarray                  # Cloud water [kg/kg] (nlev, ncols)
    qi: jnp.ndarray                  # Cloud ice [kg/kg] (nlev, ncols)

    # Surface precipitation (from microphysics autoconversion)
    precip_rain: jnp.ndarray         # Rain precipitation [kg/m²/s] (ncols,)
    precip_snow: jnp.ndarray         # Snow precipitation [kg/m²/s] (ncols,)

    # Cloud properties
    droplet_number: jnp.ndarray  # Droplet number concentration [1/m³] (nlev, ncols)

    # Previous-timestep (t-dt) 2M number concentrations carried across
    # steps so the 2M ``update_tendencies_and_important_vars`` step has
    # the tm1 state it needs. Stored per kg of air (matching the
    # qnc/qni tracer convention).
    qnc_prev: jnp.ndarray            # Previous-step cloud droplet number [1/kg] (nlev, ncols)
    qni_prev: jnp.ndarray            # Previous-step ice crystal number    [1/kg] (nlev, ncols)

    # All-sky and clear-sky outgoing TOA fluxes from the radiation term's
    # cloudy + clear beam-split, written so users can compute the cloud
    # radiative effect (CRE) directly from a single output dataset:
    #
    #     CRE_SW = toa_sw_up_clear - toa_sw_up_all   (negative; cooling)
    #     CRE_LW = toa_lw_up_clear - toa_lw_up_all   (positive; warming)
    #     CRE    = CRE_SW + CRE_LW
    #
    # The all-sky values mirror ``RadiationData.toa_{sw,lw}_up`` — they
    # are duplicated here so the CRE consumer can read everything from
    # the ``"clouds"`` diagnostic key without having to cross-reference
    # the radiation key.
    toa_sw_up_all: jnp.ndarray       # All-sky outgoing SW at TOA [W/m²] (ncols,)
    toa_sw_up_clear: jnp.ndarray     # Clear-sky outgoing SW at TOA [W/m²] (ncols,)
    toa_lw_up_all: jnp.ndarray       # All-sky OLR at TOA [W/m²] (ncols,)
    toa_lw_up_clear: jnp.ndarray     # Clear-sky OLR at TOA [W/m²] (ncols,)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            cloud_fraction=jnp.zeros((nlev,) + nodal_shape),
            qc=jnp.zeros((nlev,) + nodal_shape),
            qi=jnp.zeros((nlev,) + nodal_shape),
            precip_rain=jnp.zeros(nodal_shape),
            precip_snow=jnp.zeros(nodal_shape),
            droplet_number=jnp.zeros((nlev,) + nodal_shape),
            qnc_prev=jnp.zeros((nlev,) + nodal_shape),
            qni_prev=jnp.zeros((nlev,) + nodal_shape),
            toa_sw_up_all=jnp.zeros(nodal_shape),
            toa_sw_up_clear=jnp.zeros(nodal_shape),
            toa_lw_up_all=jnp.zeros(nodal_shape),
            toa_lw_up_clear=jnp.zeros(nodal_shape),
        )

    def copy(self, **kwargs):
        new_data = {
            'cloud_fraction': self.cloud_fraction,
            'qc': self.qc,
            'qi': self.qi,
            'precip_rain': self.precip_rain,
            'precip_snow': self.precip_snow,
            'droplet_number': self.droplet_number,
            'qnc_prev': self.qnc_prev,
            'qni_prev': self.qni_prev,
            'toa_sw_up_all': self.toa_sw_up_all,
            'toa_sw_up_clear': self.toa_sw_up_clear,
            'toa_lw_up_all': self.toa_lw_up_all,
            'toa_lw_up_clear': self.toa_lw_up_clear,
        }
        new_data.update(kwargs)
        return CloudData(**new_data)


def radiation_cloud_fields(state, diagnostics):
    """Return ECHAM-ordered cloud fields for radiation.

    ECHAM ``physc`` calls ``cover`` before radiation, then passes the
    diagnosed cloud fraction plus the pre-cloud-step ``xlm1`` / ``xim1``
    condensate fields into radiation. Large-scale cloud microphysics runs
    later. Mirror that here: fresh cloud fraction comes from
    ``diagnostics["clouds"]``, while condensate comes from state tracers.
    """
    clouds = diagnostics["clouds"]
    cloud_water = state.tracers.get("qc", jnp.zeros_like(state.temperature))
    cloud_ice = state.tracers.get("qi", jnp.zeros_like(state.temperature))
    return cloud_water, cloud_ice, clouds.cloud_fraction
