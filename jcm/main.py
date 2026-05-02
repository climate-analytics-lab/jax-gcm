#!/usr/bin/env python
"""Hydra-based CLI entry point for JAX-GCM.

Examples
--------
Default 10-day SPEEDY aquaplanet run::

    python -m jcm.main      # via the module path
    ./jcm/main.py           # or directly as an executable

Switch physics package via Hydra config groups::

    python -m jcm.main physics=echam grid=echam_t85_l47_hybrid run=longrun
    python -m jcm.main physics=held_suarez grid=held_suarez_t31_l8

Override individual options::

    python -m jcm.main run.total_time=30 run.save_interval=1 run.time_step=20
    python -m jcm.main physics=echam physics.radiation=rrtmgp init=jw

Multi-run sweep::

    python -m jcm.main -m run.time_step=10,20,30

"""


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from jcm.runners import resolve_output_path, run, save_predictions


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run a JAX-GCM simulation configured via Hydra."""
    print(OmegaConf.to_yaml(cfg))
    predictions = run(cfg)
    output_path = resolve_output_path(cfg, HydraConfig.get())
    save_predictions(predictions, output_path)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
