"""SLIM, the Simple Land Interface Model (Laguë, Bonan & Swann 2019).

``land_model: slim`` on the ECHAM hosts: a soil-water bucket, a snow bucket
and a ten-layer soil temperature column with prescribed monthly surface
properties (ESCOMP/SimpleLand ``mml_main.F90``). See
``docs/source/science/surface.md`` ("The SLIM land").
"""

from jcm.physics.surface.slim.slim_land import NSOIL, SlimParameters
from jcm.physics.surface.slim.slim_term import SLIM_LAND_KEY, SlimLand, SlimLandData

__all__ = ["NSOIL", "SLIM_LAND_KEY", "SlimLand", "SlimLandData", "SlimParameters"]
