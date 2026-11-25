# ERA5 nudging: what is relaxed, and where

`nudging=era5` adds a `jcm.nudging.NudgingTerm` that relaxes the model toward an
ERA5 reference, `dX/dt = inv_tau * (X_ref - X)` (Krishnamurti et al. 1991). The
term is an ordinary composable `PhysicsTerm`: it works on the physics-facing
gridpoint state, so any dycore gets it, and it is layout-agnostic (the same code
runs on the whole-grid `(nlev, nlon, nlat)` and the column-vectorised
`(nlev, ncols)` layouts).

## The vertical profile of the relaxation

The inverse timescale `inv_tau` is a per-level array. Three things shape it:

- `tau_hours` sets its magnitude (6 h by default).
- `pbl_levels` sets it to zero in the bottom levels, so the boundary layer
  responds freely.
- `min_pressure_hpa` sets it to zero for every level whose *reference* pressure
  is below it. The reference pressure of a level is its mid-level pressure at a
  standard surface pressure of `constants.p0s1_bg`, 101325 Pa by default
  (`jcm.physics.coords_util.reference_pressure_pa`, which the Hines launch
  pressure and the hyperdiffusion mapping onto a user level table also use),
  for hybrid and sigma coordinates alike. The WeatherBench2 13-level ERA5 stores
  stop at 50 hPa and clamp above it, so relaxing the stratosphere toward them
  would drag it to the 50 hPa values; this is the hard floor.

`nudge_temperature` additionally switches on the same profile for temperature
(winds-only is the default).

### Why a latitude-dependent top

A single `min_pressure_hpa` is one step for the whole globe, but the tropopause
is near 100 hPa in the tropics and near 300 hPa at the poles. A step at 150 hPa
nudges the extratropical lowermost stratosphere; a step at 400 hPa stops
nudging the tropical upper troposphere. For a stratospheric-transport
configuration neither is acceptable, so the top of the nudged region can be made
a function of latitude with an optional `nudging.lat_top` block
(`jcm.nudging.LatTop`). With `lat_top: null` (the default) the per-level step
above is the whole profile.

The top pressure is

- `p_tropical_hpa` for `|lat| <= lat_tropical_deg`,
- `p_polar_hpa` for `|lat| >= lat_polar_deg`,
- between them a blend in `ln p` with a `cos^2` weight in latitude:

  `w_trop = cos^2( (pi/2) (|lat| - lat_tropical_deg) / (lat_polar_deg - lat_tropical_deg) )`

  `ln p_top = w_trop ln p_tropical + (1 - w_trop) ln p_polar`

and is symmetric about the equator. The nudging weight of a level with reference
pressure `p` in a column at latitude `lat` is

`w(p, lat) = clip( log10(p / p_top(lat)) / taper_decades, 0, 1 )`,

which is 0 above the top and rises linearly in `log10 p` to 1 a distance
`taper_decades` below it, instead of a step (`taper_decades: 0` recovers the
step). Every nudged variable's `inv_tau` is multiplied by `w`.

How it composes:

- `min_pressure_hpa` stays a hard floor on top of `w`: levels above it are never
  nudged, whatever `w` says. A floor that lies above either `lat_top` pressure
  (a smaller pressure number than the top) would silently become the binding cut
  and override the profile, so it is rejected at construction. The default
  60 hPa is compatible with any realistic tropopause-following top.
- `pbl_levels`, `tau_hours` and `nudge_temperature` act exactly as without
  `lat_top`; the weight is a multiplicative factor on the per-level array.

The weight depends on the level's *reference* pressure and the column's
latitude only. It is a fixed climatological proxy for the tropopause and does
**not** follow the instantaneous tropopause of the simulated column; a
column-by-column diagnosed tropopause would make the nudging depend on the state
it is relaxing.

The weight is evaluated at each call as an `(nlev, ncols)` array from the
per-column latitudes every horizontally-aware term caches
(`jcm.physics.coords_util.column_lat_lon`) and reshaped row-major to the state's
layout, so no code branches on the horizontal rank. The profile parameters are
differentiable leaves of a `flax.struct.dataclass`; evaluating the weight per
call, not once at `cache_coords`, is what lets a gradient with respect to them
reach the tendency.

### Provenance

The shape of the profile (tropical and polar tops, a `cos^2` latitude blend in
`log p`, a taper in `log10 p`) follows the tropopause-following nudging top used
by the downstream stratospheric-transport configuration
(reflective-org/jcm-strat), which uses `p_tropical = 150 hPa`,
`p_polar = 400 hPa`, blend between 20 and 35 degrees and a 0.1-decade taper;
those are the defaults of `LatTop` and the values commented in
`jcm/config/nudging/era5.yaml`.
