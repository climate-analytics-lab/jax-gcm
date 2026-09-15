"""Regenerate the Fortran reference tendencies for SPEEDY's vertical diffusion.

``jcm/physics/vertical_diffusion/speedy_vdiff_test.py`` pins the port against
tendencies produced by the original Fortran. Those are frozen literals, and
they are *not* constants-independent: they scale with ``jcm.constants.cpd``
through ``fshcse``/``fvdise`` and with ``alhc`` through ``dmse``. A revision to
either -- ``cpd`` has moved once already, from 1004.0 to 1004.64 -- puts them
outside the test's tolerance. Regenerating them from the JAX port would turn an
independent cross-implementation check into a self-consistency one that can no
longer see a port bug, so this script exists to regenerate them from the
Fortran instead::

    python tools/regenerate_speedy_vdiff_reference.py

It prints a ``FORTRAN_REFERENCE`` block to paste back into the test.

The reference is SPEEDY's ``source/vertical_diffusion.f90``, which is
**downloaded at run time and compiled unmodified**. It is not vendored here:
speedy.f90 is licensed for non-commercial use only and requires its copyright
notice on substantial portions, neither of which fits an Apache-2.0 tree. Only
the four stub modules it imports (``types``, ``params``,
``physical_constants``, ``geometry``) and the driver are written out below, and
they supply jcm's ``cpd``/``alhc`` rather than SPEEDY's own ``cp = 1004.0``, so
what the test compares is the *formulation* and not the choice of constants.

Requires ``gfortran`` and network access.
"""

import argparse
import hashlib
import pathlib
import subprocess
import sys
import tempfile
import urllib.request

# Pinned to a commit, not a branch: the whole point of this reference is that it
# is independent of the JAX port, and a regeneration that silently picked up an
# upstream edit would re-pin the test to a *different* scheme while looking like
# a constants refresh. The digest makes that failure loud rather than silent.
SOURCE_COMMIT = "f0a358e9914a4de32836c1e126a37b37bd454fda"
SOURCE_URL = ("https://raw.githubusercontent.com/samhatfield/speedy.f90/"
              f"{SOURCE_COMMIT}/source/vertical_diffusion.f90")
SOURCE_SHA256 = "5992f6a73bd3bade11c374cec5bd8078a99277d40b30c35122f7a622e87aa809"

# The modules vertical_diffusion.f90 imports, cut down to the handful of names
# it actually uses. cp and alhc are jcm's values (jcm.constants.cpd and
# jcm.physics.speedy.physical_constants.alhc), not SPEEDY's.
STUBS = """\
module types
    implicit none
    integer, parameter :: p = kind(1.0d0)
end module

module params
    implicit none
    integer, parameter :: ix = 1, il = 1, kx = {kx}
end module

module physical_constants
    use types, only: p
    use params, only: kx
    implicit none
    real(p), parameter :: cp = {cpd!r}_p
    real(p), parameter :: alhc = {alhc!r}_p
    real(p) :: sigh(0:kx)
end module

module geometry
    use types, only: p
    use params, only: kx
    implicit none
    real(p) :: fsg(kx), dhs(kx)
end module
"""

# Reads the cases this script writes out, one per record, and prints the
# tendencies. The unlimited-repeat edit descriptor keeps each tendency on one
# line whatever kx is; a fixed count would revert the format past its width and
# split the record, which the parser below would read as a new tendency. fsg and dhs are derived from the half levels exactly as
# compute_speedy_vertical_coords does.
DRIVER = """\
program vdif_reference
    use types, only: p
    use params, only: ix, il, kx
    use physical_constants, only: sigh
    use geometry, only: fsg, dhs
    use vertical_diffusion, only: get_vertical_diffusion_tend
    implicit none

    real(p) :: se(ix,il,kx), rh(ix,il,kx), qa(ix,il,kx), qsat(ix,il,kx), phi(ix,il,kx)
    real(p) :: utenvd(ix,il,kx), vtenvd(ix,il,kx), ttenvd(ix,il,kx), qtenvd(ix,il,kx)
    integer :: icnv(ix,il), k, n, icase, iu

    open(newunit=iu, file='cases.txt', status='old', action='read')
    read(iu,*) (sigh(k), k = 0, kx)
    do k = 1, kx
        fsg(k) = 0.5_p*(sigh(k) + sigh(k-1))
        dhs(k) = sigh(k) - sigh(k-1)
    end do
    read(iu,*) n
    do icase = 1, n
        read(iu,*) icnv(1,1)
        read(iu,*) (se(1,1,k),   k = 1, kx)
        read(iu,*) (rh(1,1,k),   k = 1, kx)
        read(iu,*) (qa(1,1,k),   k = 1, kx)
        read(iu,*) (qsat(1,1,k), k = 1, kx)
        read(iu,*) (phi(1,1,k),  k = 1, kx)
        call get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv, &
            & utenvd, vtenvd, ttenvd, qtenvd)
        write(*,'(a,*(es24.16))') 'ttenvd ', (ttenvd(1,1,k), k = 1, kx)
        write(*,'(a,*(es24.16))') 'qtenvd ', (qtenvd(1,1,k), k = 1, kx)
    end do
    close(iu)
end program
"""


def _format_array(values, indent):
    """Wrap a float list to the repo's line width, printing zeros as ``0.0``."""
    out, line = [], ""
    for value in values:
        text = "0.0" if value == 0.0 else f"{value:.16e}"
        if len(line) + len(text) + 2 > 74 - indent:
            out.append(line.rstrip())
            line = ""
        line += text + ", "
    out.append(line.rstrip().rstrip(","))
    return ("\n" + " " * indent).join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_URL,
                        help="vertical_diffusion.f90 to compile (URL or path)")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="skip the digest check, for trying a local edit")
    args = parser.parse_args(argv)

    # Imported rather than duplicated: the test module is where the soundings,
    # the case list and the override rule live.
    import numpy as np

    import jcm.constants as c
    from jcm.physics.speedy.physical_constants import alhc
    from jcm.physics.speedy.speedy_coords import SpeedyCoords
    from jcm.physics.vertical_diffusion.speedy_vdiff_test import (
        MOISTURE_GATE_CASES, _with_rh_overrides,
    )

    kx = 8
    # SIGMA_LAYER_BOUNDARIES declares the table as exact decimals but stores it
    # float32, so widen through the shortest decimal that round-trips: the
    # Fortran then runs SPEEDY's own table rather than its float32 rounding.
    # (Either choice is well inside the test's tolerance -- they differ by ~3e-7
    # relative -- but the reference should be the table as written.)
    hsg = [float(np.format_float_positional(v, unique=True, trim="0"))
           for v in np.asarray(
               SpeedyCoords.single_column_coords(num_levels=kx).hsg)]

    with tempfile.TemporaryDirectory() as tmp:
        build = pathlib.Path(tmp)
        if args.source.startswith(("http://", "https://")):
            with urllib.request.urlopen(args.source) as response:
                scheme = response.read().decode()
        else:
            scheme = pathlib.Path(args.source).read_text()
        digest = hashlib.sha256(scheme.encode()).hexdigest()
        if digest != SOURCE_SHA256 and not args.allow_unpinned:
            raise SystemExit(
                f"{args.source} hashes to {digest}, not the pinned "
                f"{SOURCE_SHA256}. If upstream has genuinely changed the "
                "scheme, update SOURCE_COMMIT and SOURCE_SHA256 together and "
                "say so where the new reference is committed -- it is then a "
                "different scheme, not a refreshed constant.")
        (build / "vertical_diffusion.f90").write_text(scheme)
        (build / "stubs.f90").write_text(
            STUBS.format(kx=kx, cpd=float(c.cpd), alhc=float(alhc)))
        (build / "driver.f90").write_text(DRIVER)

        with open(build / "cases.txt", "w") as handle:
            handle.write(" ".join(f"{v:.17e}" for v in hsg) + "\n")
            handle.write(f"{len(MOISTURE_GATE_CASES)}\n")
            for _, sounding, overrides, deep in MOISTURE_GATE_CASES:
                rh, qa = _with_rh_overrides(sounding, overrides)
                handle.write("1\n" if deep else "0\n")
                for field in (sounding["se"], rh, qa, sounding["qsat"],
                              sounding["phi"]):
                    handle.write(" ".join(f"{v:.17e}" for v in field) + "\n")

        subprocess.run(
            ["gfortran", "-O0", "-o", "vdif_reference",
             "stubs.f90", "vertical_diffusion.f90", "driver.f90"],
            cwd=build, check=True)
        output = subprocess.run(["./vdif_reference"], cwd=build, check=True,
                                capture_output=True, text=True).stdout

    lines = output.strip().split("\n")
    print("FORTRAN_REFERENCE = {")
    for index, (name, _, _, _) in enumerate(MOISTURE_GATE_CASES):
        tendencies = {}
        for line in lines[2 * index:2 * index + 2]:
            key, *values = line.split()
            tendencies[key] = [float(v) for v in values]
        print(f'    "{name}": dict(')
        print(f'        ttenvd=[{_format_array(tendencies["ttenvd"], 16)}],')
        print(f'        qtenvd=[{_format_array(tendencies["qtenvd"], 16)}],')
        print("    ),")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
