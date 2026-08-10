"""Build the CEDS + BB4CMIP7 zarr mirror stores."""
import sys

sys.path.insert(0, "/glade/derecho/scratch/duncanwp/tmp/jam-fixes-dev")
from jcm.data.mirror.emissions import (SPECIES, build_store,
                                       load_bb_species, load_ceds_species)

OUT = "/glade/derecho/scratch/duncanwp/hf_mirror/build"

print("CEDS anthro:", flush=True)
build_store(load_ceds_species, SPECIES, f"{OUT}/ceds_anthro.zarr",
            "CEDS-CMIP-2025-04-18 (input4MIPs CMIP7), sector-summed, 0.5 deg")
print("BB4CMIP7:", flush=True)
build_store(load_bb_species, SPECIES, f"{OUT}/bb4cmip7.zarr",
            "DRES-CMIP-BB4CMIP7-2-0 (input4MIPs CMIP7), 0.25 deg")
print("DONE", flush=True)
