# What ECHAM's convective trigger requires of a sounding

`TiedtkeConvection`'s cloud-base search is a faithful port of ECHAM `cubase`
(`mo_cuinitialize.f90:276-320`). It has a property that surprises anyone
constructing an idealised column by hand, and it has already cost one
release-blocking regression (#773), so it is worth stating plainly.

## The trigger

A parcel starts at the lowest model level with the environment's temperature
and humidity and is walked upward one level at a time conserving dry static
energy. At each level, in order:

1. **Dry buoyancy gate** — `zbuo = Tv_u − Tv_e + zlift`. If this is not
   positive, `klab` falls to 0 and the walk stops: the column gets **no
   convection at all**, because the next iteration is gated on
   `IF(klab(jk+1).EQ.1)`.
2. **Condensation** via the damped `cuadjtq` Newton step. The first level that
   condenses is the LCL; `klab` becomes 2 and the walk stops there.
3. **Cloud-base test** at that LCL only, with condensate loading. Cloud base
   exists iff the moist buoyancy is positive.

`zlift` is the sub-grid thermal excess of the warmest boundary-layer plumes
(`MIN(MAX(cminbuoy, MIN(cmaxbuoy, thvsig·cbfac)), 1.0)`), so it is **at most
1 K**. CAPE never enters: `has_cloud_base` gates everything upstream of the
trigger weight.

## The consequence

A dry-lifted parcel loses buoyancy against its environment at
`(Γ_d − Γ_env)` per unit height — about 3.3 K/km against a 6.5 K/km lapse
rate. So a sounding running at a free-tropospheric lapse rate **right down to
the surface** exhausts `zlift` within a couple of hundred metres and never
reaches its own LCL, however large its CAPE. Such a column convects in no
ECHAM configuration.

Real tropical sub-cloud layers are near-neutral, and so are the ones jcm's own
vdiff produces in a coupled run: through a well-mixed layer the dry static
energy is constant, `zbuo = zlift > 0`, and the parcel reaches the LCL. ECHAM
leans on this twice over — its environment half levels are the DSE *upper
envelope* of the adjacent full levels
(`ptenh(jk) = (MAX(s(jk−1), s(jk)) − geoh(jk))/cpm`, then monotonized upward in
`cuini`), which flattens any dry-neutral or dry-unstable layer before the
parcel is ever compared against it.

**So: any idealised, prescribed or hand-built column handed to Tiedtke needs a
dry-adiabatic sub-cloud layer.** `jcm.rce.rce_initial_state` builds one by
default (`mixed_layer_top_m`, 800 m); the convection tests' `_sounding`
helpers take a `bl_top_m`; `updraft_test.py` pins both sides of the
discriminator (`test_unmixed_boundary_layer_gets_no_convection` /
`test_well_mixed_layer_convects_at_the_echam_minimum_lift`).

## Why this is not a place to add a guard

Relaxing the gate — searching upward for the LFC instead of stopping at the
LCL, or inflating `zlift` — lets a plume start above a layer the parcel could
never have crossed, which is the defect #684/#690 removed. The mid-level
(`cubasmc`) trigger is ECHAM's designed answer for elevated convection with no
surface connection, but it needs resolved ascent (`omega < 0`) and a nearly
saturated environment (`RH > 0.90`), so it is dormant in a single-column driver
with no dycore. If an idealised column will not convect, the sounding is what
should change.

## The failure mode it produces

The symptom is not an error. Convection simply never fires, and everything
downstream of it goes quietly to zero — convective tracer transport, in-plume
scavenging (`_conv_scav_flux`), convective below-cloud washout — while
turbulent mixing keeps lofting tracers, so profiles still look alive. Unit
tests of the downstream terms do not see it either, because they feed
themselves a synthetic `ConvectionData`. The assertion that catches it is a
**composed-column state check**: run the real stack and require a live updraft,
non-empty condensate and precip flux, and a soluble tracer depleted against an
insoluble twin (`tracer_transport_test.ComposedColumnScavengingTest`).
