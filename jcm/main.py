import hydra
from omegaconf import DictConfig
import os
from held_suarez_model import HeldSuarezModel
import argparse
from dinosaur import primitive_equations_states
from dataclasses import asdict
    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_model(cfg: DictConfig) -> None:
    os.makedirs(cfg.output.directory, exist_ok=True)
    model = HeldSuarezModel(
        time_step=cfg.model.time_step,
        save_interval=cfg.model.save_interval,
        total_time=cfg.model.total_time,
        layers=cfg.model.layers
    )

    if hasattr(cfg, 'radiation'):
        model.configure_radiation(cfg.radiation)
    if hasattr(cfg, 'convection'):
        modle.configure_convection(cfg.convection)
    if hasattr(cfg, 'diffusion'):
        model.configure_diffusion(cfg.diffusion)

    state = model.get_initial_state()
    state.tracers = {
        'specific_humidity': primitive_equations_states.gaussian_scalar(
            model.coords, model.physics_specs
        )
    }

    final_state, predictions = model.unroll(state)
    output_path = os.path.join(cfg.output.directory, cfg.output.filename)
    ds = model.data_to_xarray(asdict(predictions))
    ds.to_netcdf(output_path)

if __name__ == "__main__":
    run_model()