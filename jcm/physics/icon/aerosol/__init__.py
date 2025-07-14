"""Aerosol module for ICON physics."""

from .aerosol_params import AerosolParameters
from .simple_aerosol import get_simple_aerosol

__all__ = ["AerosolParameters", "get_simple_aerosol"]