"""
Radiation parameterization for ICON physics

This module implements shortwave and longwave radiation calculations
including gas absorption, cloud optics, and radiative transfer.

The implementation follows a modular design:
- Solar calculations using jax-solar
- Gas optics for absorption
- Cloud optical properties
- Two-stream radiative transfer solver
- Heating rate calculations

Date: 2025-01-10
"""

# Module will be populated as we implement components
__all__ = []