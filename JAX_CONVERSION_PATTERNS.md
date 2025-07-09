# JAX Conversion Patterns for Fortran Climate Code

This document outlines standard patterns for converting Fortran conditional logic to JAX-compatible operations. These patterns should be applied consistently across all physics modules.

## Core Principles

1. **No Python Control Flow**: Replace `if/else`, `for`, `while` with JAX operations
2. **Static Shapes**: All arrays must have compile-time known shapes
3. **Pure Functions**: No side effects, no mutations
4. **Vectorization Ready**: Design for `vmap` over spatial dimensions

## Pattern 1: Simple Conditional Assignment

### Before (Python/Fortran):
```python
if condition:
    result = value_a
else:
    result = value_b
```

### After (JAX):
```python
result = jnp.where(condition, value_a, value_b)
```

## Pattern 2: Conditional Computation

### Before (Python/Fortran):
```python
if condition:
    result = expensive_computation(x)
else:
    result = default_value
```

### After (JAX):
```python
result = lax.cond(
    condition,
    lambda: expensive_computation(x),
    lambda: default_value
)
```

## Pattern 3: Range-based Loops

### Before (Python/Fortran):
```python
for k in range(start, end):
    if some_condition(k):
        process(k)
```

### After (JAX):
```python
def process_level(k):
    return lax.cond(
        some_condition(k),
        lambda: process(k),
        lambda: default_result
    )

k_levels = jnp.arange(start, end)
results = jax.vmap(process_level)(k_levels)
```

## Pattern 4: Early Exit Loops (Find First)

### Before (Python/Fortran):
```python
found = False
result = default
for k in range(n):
    if condition(k):
        result = k
        found = True
        break
```

### After (JAX):
```python
conditions = jax.vmap(condition)(jnp.arange(n))
found = jnp.any(conditions)
result = jnp.where(found, jnp.argmax(conditions), default)
```

## Pattern 5: Accumulating Scan

### Before (Python/Fortran):
```python
state = initial_state
for k in range(n):
    state = update_state(state, inputs[k])
```

### After (JAX):
```python
def step_function(state, inputs_k):
    new_state = update_state(state, inputs_k)
    return new_state, new_state

final_state, all_states = lax.scan(
    step_function,
    initial_state,
    inputs
)
```

## Pattern 6: Conditional Array Updates

### Before (Python/Fortran):
```python
if condition:
    array[i] = new_value
```

### After (JAX):
```python
array = array.at[i].set(jnp.where(condition, new_value, array[i]))
# or more simply:
array = jnp.where(condition, array.at[i].set(new_value), array)
```

## Pattern 7: Masked Operations

### Before (Python/Fortran):
```python
for i in range(n):
    if mask[i]:
        result[i] = computation(data[i])
    else:
        result[i] = default_value
```

### After (JAX):
```python
computed = jax.vmap(computation)(data)
result = jnp.where(mask, computed, default_value)
```

## Pattern 8: Nested Conditionals

### Before (Python/Fortran):
```python
if condition1:
    if condition2:
        result = value_a
    else:
        result = value_b
else:
    result = value_c
```

### After (JAX):
```python
result = lax.cond(
    condition1,
    lambda: lax.cond(condition2, lambda: value_a, lambda: value_b),
    lambda: value_c
)
```

## Pattern 9: Multi-branch Selection

### Before (Python/Fortran):
```python
if type == 1:
    result = computation_a()
elif type == 2:
    result = computation_b()
else:
    result = computation_c()
```

### After (JAX):
```python
branches = [
    lambda: computation_a(),
    lambda: computation_b(),
    lambda: computation_c()
]
result = lax.switch(type, branches)
```

## Downdraft Module Conversion Example

Let's apply these patterns to the downdraft module:

### Problem: Python `if` in `calculate_downdraft`

```python
# Before:
if has_lfs:
    # Initialize downdraft
    td_init = td_init.at[lfs].set(0.5 * (updraft_state.tu[lfs] + twb))
    # ...
```

### Solution: Use `lax.cond` with lambda functions

```python
# After:
def initialize_downdraft():
    twb, qwb = wetbulb_temperature(
        temperature[lfs], humidity[lfs], pressure[lfs]
    )
    td_new = td_init.at[lfs].set(0.5 * (updraft_state.tu[lfs] + twb))
    qd_new = qd_init.at[lfs].set(0.5 * (updraft_state.qu[lfs] + qwb))
    mfd_new = mfd_init.at[lfs].set(-config.cmfctop * updraft_state.mfu[kbase])
    return td_new, qd_new, mfd_new

def no_downdraft():
    return td_init, qd_init, mfd_init

td_final, qd_final, mfd_final = lax.cond(
    has_lfs,
    initialize_downdraft,
    no_downdraft
)
```

## Key Guidelines

1. **Always use lambda functions** in `lax.cond` to avoid eager evaluation
2. **Vectorize first**, then apply conditionals
3. **Use `jnp.where`** for simple value selection
4. **Use `lax.cond`** for expensive computations
5. **Use `lax.scan`** for sequential dependencies
6. **Use `jax.vmap`** for parallel operations
7. **Test with `jax.jit`** to ensure no control flow issues

## Common Pitfalls

1. **Eager evaluation**: Using `lax.cond(condition, func(), default)` instead of `lax.cond(condition, lambda: func(), lambda: default)`
2. **Shape inconsistency**: Branches returning different shapes
3. **Tracer leakage**: Using Python conditionals on traced values
4. **Dynamic shapes**: Array sizes depending on runtime values

## Pattern 10: Centralized Vectorization in Physics

### Problem:
All physics modules need to process atmospheric columns independently, requiring the same reshape/vmap/reshape pattern.

### Solution:
Handle vectorization at the `compute_tendencies` level, so individual physics modules work with 2D arrays.

### Implementation:
```python
# In IconPhysics.compute_tendencies():
def compute_tendencies(self, state, boundaries, geometry, date):
    # Get array dimensions
    nlev, nlat, nlon = state.temperature.shape
    ncols = nlat * nlon
    
    # Reshape to 2D for column-wise processing
    vectorized_state = PhysicsState(
        temperature=state.temperature.reshape(nlev, ncols),
        # ... other fields
    )
    
    # Apply physics terms (each handles 2D arrays internally)
    for term in self.terms:
        term_tendency, physics_data = term(vectorized_state, ...)
        
        # Reshape back to 3D and accumulate
        reshaped_tendency = PhysicsTendency(
            temperature=term_tendency.temperature.reshape(nlev, nlat, nlon),
            # ... other fields
        )
        tendencies = tendencies + reshaped_tendency
```

### Benefits:
- **Reusable**: All physics modules benefit from centralized vectorization
- **Cleaner code**: Individual modules focus on single-column physics
- **Consistent**: Same pattern across all modules
- **Efficient**: Automatic vectorization via `jax.vmap`

### Individual Module Pattern:
```python
def _apply_physics_module(self, state, physics_data, boundaries, geometry):
    # state is already 2D: [nlev, ncols]
    
    def single_column_physics(temp_col, humid_col, ...):
        # Apply physics to single column
        return dtedt, dqdt, dudt, dvdt
    
    # Vectorize over all columns
    tendencies = jax.vmap(
        single_column_physics,
        in_axes=(1, 1, ...),  # vmap over column dimension
        out_axes=(1, 1, 1, 1)
    )(state.temperature, state.specific_humidity, ...)
    
    # Return 2D tendencies
    return PhysicsTendency(temperature=tendencies[0], ...), physics_data
```

## Performance Tips

1. **Minimize `lax.cond`**: Use `jnp.where` when possible
2. **Batch operations**: Group similar conditionals
3. **Avoid nested scans**: Restructure algorithms when possible
4. **Use static arguments**: Mark configuration as static in JIT
5. **Centralize vectorization**: Handle reshape/vmap at the framework level
6. **Return simple arrays**: Avoid complex NamedTuples in vmap when possible

## Module Conversion Checklist

When converting a new physics module:

1. ✅ **Replace Python conditionals** with JAX patterns (1-9)
2. ✅ **Design for 2D input** (use centralized vectorization)
3. ✅ **Use single-column functions** with `jax.vmap`
4. ✅ **Test with `jax.jit`** to ensure no control flow issues
5. ✅ **Validate physics** against reference implementation
6. ✅ **Benchmark performance** at different resolutions

This pattern guide should be applied to all physics modules as we convert them from Fortran to JAX-compatible Python.