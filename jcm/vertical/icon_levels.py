"""
ICON vertical hybrid sigma-pressure coordinate definitions.

This module provides ICON's vertical coordinate tables for various numbers
of atmospheric levels, parsed from the official ICON vertical_coord_tables.
"""

import dataclasses
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Optional


@dataclasses.dataclass(frozen=True)
class HybridLevels:
    """Hybrid sigma-pressure coordinate definition.
    
    Pressure at interfaces: p = a + b * p_surface
    """
    nlevels: int
    a_boundaries: jnp.ndarray  # Pressure coefficient at interfaces (Pa)
    b_boundaries: jnp.ndarray  # Sigma coefficient at interfaces (dimensionless)
    
    @property
    def a_centers(self) -> jnp.ndarray:
        """Pressure coefficients at level centers."""
        return (self.a_boundaries[1:] + self.a_boundaries[:-1]) / 2
    
    @property
    def b_centers(self) -> jnp.ndarray:
        """Sigma coefficients at level centers."""
        return (self.b_boundaries[1:] + self.b_boundaries[:-1]) / 2
    
    def get_pressure_levels(self, surface_pressure: jnp.ndarray) -> jnp.ndarray:
        """Calculate pressure at level centers.
        
        Args:
            surface_pressure: Surface pressure field (Pa)
            
        Returns:
            Pressure at level centers (nlevels, *surface_pressure.shape)
        """
        # Handle different dimensions of surface_pressure
        if surface_pressure.ndim == 0:
            # Scalar
            return self.a_centers + self.b_centers * surface_pressure
        elif surface_pressure.ndim == 1:
            # 1D array
            return self.a_centers[:, None] + self.b_centers[:, None] * surface_pressure[None, :]
        else:
            # Multi-dimensional array - reshape for broadcasting
            # a_centers and b_centers are 1D (nlevels,)
            # surface_pressure is (..., nlon, nlat)
            # Result should be (nlevels, ..., nlon, nlat)
            extra_dims = (1,) * surface_pressure.ndim
            a_broadcast = self.a_centers.reshape(self.nlevels, *extra_dims)
            b_broadcast = self.b_centers.reshape(self.nlevels, *extra_dims)
            return a_broadcast + b_broadcast * surface_pressure[None, ...]
    
    def get_pressure_interfaces(self, surface_pressure: jnp.ndarray) -> jnp.ndarray:
        """Calculate pressure at level interfaces.
        
        Args:
            surface_pressure: Surface pressure field (Pa)
            
        Returns:
            Pressure at interfaces (nlevels+1, *surface_pressure.shape)
        """
        # Handle different dimensions of surface_pressure
        if surface_pressure.ndim == 0:
            # Scalar
            return self.a_boundaries + self.b_boundaries * surface_pressure
        elif surface_pressure.ndim == 1:
            # 1D array
            return self.a_boundaries[:, None] + self.b_boundaries[:, None] * surface_pressure[None, :]
        else:
            # Multi-dimensional array - reshape for broadcasting
            # a_boundaries and b_boundaries are 1D (nlevels+1,)
            # surface_pressure is (..., nlon, nlat)
            # Result should be (nlevels+1, ..., nlon, nlat)
            extra_dims = (1,) * surface_pressure.ndim
            a_broadcast = self.a_boundaries.reshape(self.nlevels + 1, *extra_dims)
            b_broadcast = self.b_boundaries.reshape(self.nlevels + 1, *extra_dims)
            return a_broadcast + b_broadcast * surface_pressure[None, ...]


class ICONLevels:
    """Factory for ICON vertical coordinate definitions."""
    
    # Cache for loaded level definitions
    _levels_cache: Dict[int, HybridLevels] = {}
    
    @classmethod
    def get_levels(cls, nlevels: int) -> HybridLevels:
        """Get ICON hybrid levels for specified number of levels.
        
        Args:
            nlevels: Number of vertical levels (must be available in ICON tables)
            
        Returns:
            HybridLevels object with coefficients
        """
        if nlevels not in cls._levels_cache:
            cls._levels_cache[nlevels] = cls._load_levels(nlevels)
        return cls._levels_cache[nlevels]
    
    @classmethod
    def _load_levels(cls, nlevels: int) -> HybridLevels:
        """Load ICON level coefficients from table files."""
        # Try to find the vertical coord table file
        base_paths = [
            Path("../icon_plumeworld/vertical_coord_tables"),
            Path("./vertical_coord_tables"),
        ]
        
        filename = f"atm_hyb_sp_{nlevels}"
        table_file = None
        
        for base_path in base_paths:
            candidate = base_path / filename
            if candidate.exists():
                table_file = candidate
                break
        
        if table_file is None:
            # If file not found, use built-in definitions for common levels
            return cls._get_builtin_levels(nlevels)
        
        # Parse the table file
        a_values = []
        b_values = []
        
        with open(table_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        k = int(parts[0])
                        a = float(parts[1])
                        b = float(parts[2])
                        a_values.append(a)
                        b_values.append(b)
                    except ValueError:
                        continue
        
        if len(a_values) != nlevels + 1:
            raise ValueError(f"Expected {nlevels+1} boundary values, got {len(a_values)}")
        
        return HybridLevels(
            nlevels=nlevels,
            a_boundaries=jnp.array(a_values),
            b_boundaries=jnp.array(b_values)
        )
    
    @classmethod
    def _get_builtin_levels(cls, nlevels: int) -> HybridLevels:
        """Get built-in level definitions for common configurations."""
        
        if nlevels == 47:
            # ICON 47-level standard configuration
            a_boundaries = jnp.array([
                0.00000000000, 1.98918528294, 6.57208964360, 15.67390258170,
                30.62427876410, 54.54572041260, 92.55883043370, 150.50469698200,
                235.32745773100, 356.10025910400, 523.91952428200, 751.04294180400,
                1051.13722461000, 1438.98841128000, 1930.17735994000, 2540.69700000000,
                3286.55300000000, 4199.57400000000, 5303.95700000000, 6624.70400000000,
                8187.18500000000, 9976.13700000000, 11820.54000000000, 13431.39000000000,
                14736.36000000000, 15689.21000000000, 16266.61000000000, 16465.00000000000,
                16297.62000000000, 15791.60000000000, 14985.27000000000, 13925.52000000000,
                12665.29000000000, 11261.23000000000, 9771.40600000000, 8253.21100000000,
                6761.34000000000, 5345.91400000000, 4050.71800000000, 2911.56900000000,
                1954.80500000000, 1195.89000000000, 638.14890000000, 271.62650000000,
                72.06360000000, 0.00000000000, 0.00000000000, 0.00000000000
            ])
            
            b_boundaries = jnp.array([
                0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
                0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
                0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
                0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
                0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
                0.00000000000, 0.00040000000, 0.00290000000, 0.00920000000,
                0.02030000000, 0.03700000000, 0.05950000000, 0.08790000000,
                0.12200000000, 0.16140000000, 0.20570000000, 0.25420000000,
                0.30620000000, 0.36110000000, 0.41820000000, 0.47670000000,
                0.53590000000, 0.59510000000, 0.65360000000, 0.71060000000,
                0.76540000000, 0.81720000000, 0.86500000000, 0.90770000000,
                0.94420000000, 0.97300000000, 0.99230000000, 1.00000000000
            ])
            
            return HybridLevels(
                nlevels=47,
                a_boundaries=a_boundaries,
                b_boundaries=b_boundaries
            )
            
        elif nlevels == 40:
            # ICON 40-level configuration
            a_boundaries = jnp.array([
                27381.9054049070, 26991.9442204250, 26590.5390359760, 26177.4403857780,
                25752.3948782790, 25315.1451483130, 24865.4298087160, 24402.9834013950,
                23927.5363478650, 23438.8148992220, 22936.5410855720, 22420.4326648850,
                21890.2030712940, 21345.5613628160, 20786.2121684930, 20211.8556349540,
                19622.1873723850, 19016.8983998990, 18395.6750903180, 17758.1991143310,
                17104.1473840520, 16433.1919959550, 15745.0001731790, 15039.2342072100,
                14315.5513989190, 13573.6039989610, 12813.0391475160, 12033.4988133860,
                11234.6197324160, 10416.0333452530, 9577.3657344247, 8718.2375607417,
                7838.2639990006, 6937.0546729974, 6014.2135898353, 5069.3390735220,
                4102.0236978492, 3111.8542185484, 2098.4115047143, 1061.2704694889,
                0.0000000000
            ])
            
            b_boundaries = jnp.array([
                0.0000000000, 0.0142415650, 0.0289010701, 0.0439876262,
                0.0595104870, 0.0754790518, 0.0919028665, 0.1087916257,
                0.1261551746, 0.1440035106, 0.1623467854, 0.1811953064,
                0.2005595393, 0.2204501094, 0.2408778037, 0.2618535732,
                0.2833885341, 0.3054939706, 0.3281813366, 0.3514622576,
                0.3753485329, 0.3998521376, 0.4249852251, 0.4507601285,
                0.4771893633, 0.5042856296, 0.5320618139, 0.5605309917,
                0.5897064296, 0.6196015876, 0.6502301212, 0.6816058842,
                0.7137429305, 0.7466555168, 0.7803581051, 0.8148653646,
                0.8501921748, 0.8863536276, 0.9233650298, 0.9612419058,
                1.0000000000
            ])
            
            return HybridLevels(
                nlevels=40,
                a_boundaries=a_boundaries,
                b_boundaries=b_boundaries
            )
            
        else:
            raise ValueError(f"No built-in level definition for {nlevels} levels. "
                           f"Available: 40, 47")
    
    @classmethod
    def available_levels(cls) -> list[int]:
        """Return list of available level configurations."""
        # Check available files
        base_path = Path("/Users/watson-parris/Code/icon_plumeworld/vertical_coord_tables")
        available = []
        
        if base_path.exists():
            for file in base_path.glob("atm_hyb_sp_*"):
                try:
                    nlevels = int(file.stem.split('_')[-1])
                    available.append(nlevels)
                except ValueError:
                    continue
        
        # Add built-in levels
        available.extend([40, 47])
        
        return sorted(list(set(available)))