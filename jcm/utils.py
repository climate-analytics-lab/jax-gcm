import jax.numpy as jnp
from jax import jit
import jax.tree_util as jtu
from dinosaur.coordinate_systems import HorizontalGridTypes

# Function to take a field in grid space and truncate it to a given wavenumber
def spectral_truncation(grid: HorizontalGridTypes, grid_field, truncation_number=None):
    """
        grid_field: field in grid space
        trunc: truncation level, # of wavenumbers to keep
    """
    spectral_field = grid.to_modal(grid_field)
    nx,mx = spectral_field.shape
    n_indices, m_indices = jnp.meshgrid(jnp.arange(nx), jnp.arange(mx), indexing='ij')
    total_wavenumber = m_indices + n_indices

    # truncate to grid truncation if no truncation number is given
    truncation_number = truncation_number or (grid.total_wavenumbers - 2)

    spectral_field = jnp.where(total_wavenumber > truncation_number, 0.0, spectral_field)

    truncated_grid_field = grid.to_nodal(spectral_field)

    return truncated_grid_field

@jit
def pass_fn(operand):
    return operand

def ones_like(x):
    return jtu.tree_map(lambda y: jnp.ones_like(y), x)

def bools_like(x):
    return jtu.tree_map(lambda y: jnp.zeros_like(y, dtype=bool), x)

def bool_version(cls):
    """
    Makes a version of the given tree_math.struct where all fields are bools.
    """
    from dataclasses import make_dataclass, fields, field
    import tree_math

    @classmethod
    def all_false(cls_):
        return cls_(**{f.name: False for f in fields(cls_)})
    
    @classmethod
    def all_true(cls_):
        return cls_(**{f.name: True for f in fields(cls_)})

    return tree_math.struct(
        make_dataclass(
            cls.__name__ + "Config",
            [(f.name, bool, field(metadata=f.metadata)) for f in fields(cls)],
            namespace={"all_false": all_false, "all_true": all_true}
        )
    )

def mask_leaves(data, config):
    leaves_data, treedef_data = jtu.tree_flatten(data)
    leaves_config, _ = jtu.tree_flatten(config)

    return jtu.tree_unflatten(
        treedef_data,
        (d if c else None for d, c in zip(leaves_data, leaves_config))
    )