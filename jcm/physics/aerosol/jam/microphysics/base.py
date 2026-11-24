"""Microphysics-core contract and shared execution helpers for JAM.

The microphysics term is the single swap point between the harness and an
interchangeable aerosol core. It publishes a static ``ModalAerosolSpec``
(the population shape), declares the prognostic tracers that population
needs, and each step writes the ``_jam_state`` diagnostic the downstream
harness terms read.

Concrete cores (:class:`PlaceholderMicrophysics`, :class:`Mam4JaxMicrophysics`,
:class:`M7JaxMicrophysics`) share precision scoping, targeted operator-split
tracer extraction, cell-minor ``JamAerosolState`` construction, and optional
``jax.lax.scan`` + ``jax.checkpoint`` cell chunking through
:class:`ModalMicrophysicsTerm` so accelerator throughput and peak-HBM controls
behave identically across aerosol schemes.
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import ClassVar, Sequence

import jax
import jax.numpy as jnp

from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import tracer_specs
from jcm.physics.physics_term import PhysicsTerm, TracerSpec

_TINY_VOL_DEFAULT = 1.0e-35


def _parse_chunk_size_env(*var_names: str) -> int | None:
    """Read an optional positive integer cell chunk size from environment variables."""
    for name in var_names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        lowered = raw.strip().lower()
        if lowered in ("0", "none", "off", "false"):
            return None
        return int(raw)
    return None


class ModalMicrophysicsTerm(PhysicsTerm):
    """Base class for an interchangeable modal aerosol microphysics core.

    Subclasses set :attr:`spec` (a :class:`ModalAerosolSpec`) and implement
    :meth:`__call__` to write the ``_jam_state`` diagnostic. The tracer set
    is derived from the population, so the harness seeds exactly the tracers
    the active core needs.
    """

    category: ClassVar[str] = "aerosol_microphysics"
    provides: ClassVar[tuple[str, ...]] = ("_jam_state",)

    #: Set by subclasses — the population this core advances.
    spec: ClassVar[ModalAerosolSpec]

    def required_tracers(self) -> tuple[TracerSpec, ...]:  # type: ignore[override]
        """Interstitial + cloud-borne tracers for this population."""
        return tracer_specs(self.spec)

    # ------------------------------------------------------------------
    # Shared precision, tracer-extraction, chunking, and JamAerosolState
    # ------------------------------------------------------------------

    def configure_core_execution(
        self,
        *,
        enable_x64: bool | None = None,
        core_dtype: str | None = None,
        chunk_size: int | None = None,
        env_prefix: str = "JAM_MICROPHYSICS",
    ) -> None:
        """Configure global ``jax_enable_x64``, per-core dtype, and cell chunk size.

        Mirrors the established JAM core precision contract:
          * ``enable_x64`` sets global precision at construction (defaulting to
            ``<env_prefix>_ENABLE_X64``, default ``"1"``).
          * ``core_dtype`` selects ``"float32"`` or ``"float64"`` for this core
            (defaulting to ``<env_prefix>_CORE_DTYPE``, default ``"float64"``).
          * ``chunk_size`` optionally tiles grid cells into ``(n_chunks, chunk_size)``
            via ``jax.lax.scan`` (defaulting to ``<env_prefix>_CHUNK_SIZE`` or
            ``JAM_MICROPHYSICS_CHUNK_SIZE`` when unset).
        """
        if enable_x64 is None:
            want_x64 = os.environ.get(f"{env_prefix}_ENABLE_X64", "1") != "0"
        else:
            want_x64 = bool(enable_x64)
        jax.config.update("jax_enable_x64", want_x64)
        self._enable_x64 = want_x64

        if core_dtype is None:
            core_dtype = os.environ.get(f"{env_prefix}_CORE_DTYPE", "float64")
        if core_dtype not in ("float32", "float64"):
            raise ValueError(
                f"core_dtype must be 'float32' or 'float64', got {core_dtype!r}"
            )
        self._core_f32 = core_dtype == "float32" or not want_x64

        if chunk_size is None:
            chunk_size = _parse_chunk_size_env(
                f"{env_prefix}_CHUNK_SIZE", "JAM_MICROPHYSICS_CHUNK_SIZE"
            )
        if chunk_size is not None and int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        self._chunk_size = int(chunk_size) if chunk_size is not None else None

    @property
    def working_dtype(self):
        """JAX floating dtype used inside this core's ``_step``."""
        return jnp.float32 if self._core_f32 else jnp.float64

    def core_precision_scope(self):
        """Scoped ``jax.enable_x64(False)`` context when running a float32 core."""
        return jax.enable_x64(False) if self._core_f32 else contextlib.nullcontext()

    @staticmethod
    def cast_tree_to_dtype(tree, dtype):
        """Cast floating-point array leaves of ``tree`` to ``dtype`` (call outside x64=False scope)."""
        return jax.tree_util.tree_map(
            lambda x: x.astype(dtype)
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
            else x,
            tree,
        )

    def fetch_working_tracers(
        self,
        state,
        diagnostics: dict,
        names: Sequence[str],
        dtype,
        *,
        cell_shape: tuple[int, ...] | None = None,
    ) -> tuple[jnp.ndarray, ...]:
        """Fetch and non-negative-floor only ``names`` from the operator-split tracer view."""
        if not names:
            return ()
        from jcm.physics.aerosol.jam.removal_split import split_view

        shape = state.temperature.shape if cell_shape is None else cell_shape
        view = split_view(self.spec, state, diagnostics, names=names)
        zeros = None
        out = []
        for name in names:
            raw = view.get(name)
            if raw is None:
                if zeros is None:
                    zeros = jnp.zeros(shape, dtype)
                out.append(zeros)
            else:
                out.append(jnp.maximum(jnp.asarray(raw, dtype), 0.0))
        return tuple(out)

    def map_cells(
        self,
        fn,
        cell_inputs,
        cell_shape: tuple[int, ...],
        *,
        chunk_size: int | None = None,
        remat: bool = True,
    ):
        """Apply ``fn(batch_inputs)`` across flattened cells, optionally tiled via ``lax.scan``.

        Every array leaf in ``cell_inputs`` must have leading dimensions ``cell_shape``
        (shape ``cell_shape + extra_dims``). Inside ``fn``, every leaf has shape
        ``(batch_size,) + extra_dims``. Every array leaf returned by ``fn`` must
        likewise have leading dimension ``batch_size`` and is reshaped back to
        ``cell_shape + out_extra_dims``.

        When ``n_cells > chunk_size``, tail cells in the final partial chunk are
        filled with ``stop_gradient`` copies of the last valid cell so padded
        lanes remain in-domain without contaminating gradients on real cells.
        """
        eff_chunk = self._chunk_size if chunk_size is None else chunk_size
        rank = len(cell_shape)
        n_cells = math.prod(cell_shape) if rank else 1

        def _flatten(arr):
            arr = jnp.asarray(arr)
            return arr.reshape((n_cells,) + arr.shape[rank:])

        def _unflatten(arr):
            return arr.reshape(cell_shape + arr.shape[1:])

        flat_inputs = jax.tree_util.tree_map(_flatten, cell_inputs)

        if eff_chunk is None or n_cells == 0 or n_cells <= eff_chunk:
            eval_fn = jax.checkpoint(fn) if (eff_chunk is not None and remat and n_cells > 0) else fn
            flat_out = eval_fn(flat_inputs)
            return jax.tree_util.tree_map(_unflatten, flat_out)

        rem = n_cells % eff_chunk
        pad = (eff_chunk - rem) if rem else 0
        n_chunks = (n_cells + pad) // eff_chunk

        def _to_chunks(arr):
            if pad:
                tail = jnp.repeat(jax.lax.stop_gradient(arr[-1:]), pad, axis=0)
                arr = jnp.concatenate((arr, tail), axis=0)
            return arr.reshape((n_chunks, eff_chunk) + arr.shape[1:])

        chunked_inputs = jax.tree_util.tree_map(_to_chunks, flat_inputs)
        step_fn = jax.checkpoint(fn) if remat else fn

        def _scan_body(carry, chunk_in):
            return carry, step_fn(chunk_in)

        _, chunked_out = jax.lax.scan(_scan_body, None, chunked_inputs)

        def _from_chunks(arr):
            flat = arr.reshape((n_chunks * eff_chunk,) + arr.shape[2:])
            if pad:
                flat = flat[:n_cells]
            return flat.reshape(cell_shape + arr.shape[2:])

        return jax.tree_util.tree_map(_from_chunks, chunked_out)

    @staticmethod
    def mode_kappa_and_mass(
        species_masses_by_mode: Sequence[Sequence[tuple[jnp.ndarray, float, float]]],
        *,
        tiny_vol: float = _TINY_VOL_DEFAULT,
    ) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
        """Compute volume-weighted hygroscopicity ``kappa`` and total ``mass`` per mode."""
        kappas = []
        masses = []
        for sp_entries in species_masses_by_mode:
            tot_mass = None
            tot_vol = None
            vol_kappa = None
            for m, density, hyg in sp_entries:
                v = m / density
                vk = v * hyg
                if tot_mass is None:
                    tot_mass = m
                    tot_vol = v
                    vol_kappa = vk
                else:
                    tot_mass = tot_mass + m
                    tot_vol = tot_vol + v
                    vol_kappa = vol_kappa + vk
            safe_vol = jnp.maximum(tot_vol, tiny_vol)
            kappas.append(jnp.where(tot_vol > tiny_vol, vol_kappa / safe_vol, 0.0))
            masses.append(tot_mass)
        return tuple(kappas), tuple(masses)

    @staticmethod
    def build_jam_state(
        r_dry_modes: Sequence[jnp.ndarray],
        r_wet_modes: Sequence[jnp.ndarray],
        rho_modes: Sequence[jnp.ndarray],
        kappa_modes: Sequence[jnp.ndarray],
        mass_modes: Sequence[jnp.ndarray],
        number_modes: Sequence[jnp.ndarray],
    ) -> JamAerosolState:
        """Assemble ``JamAerosolState`` directly in ``(n_modes, nlev, *horiz)`` cell-minor layout."""
        return JamAerosolState(
            r_dry=jnp.stack(tuple(r_dry_modes), axis=0),
            r_wet=jnp.stack(tuple(r_wet_modes), axis=0),
            rho=jnp.stack(tuple(rho_modes), axis=0),
            kappa=jnp.stack(tuple(kappa_modes), axis=0),
            mass=jnp.stack(tuple(mass_modes), axis=0),
            number=jnp.stack(tuple(number_modes), axis=0),
        )
