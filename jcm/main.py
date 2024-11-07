import hydra
from omegaconf import DictConfig
import os
from held_suarez_model import HeldSuarezModel
import argparse
from dinosaur import primitive_equations_states
from dataclasses import asdict

# def parse_args():
#     # optional arguments
#     parser = argparse.ArgumentParser(description="Instantiate and run SpeedyModel.")
#     parser.add_argument('--time_step', type=int, default=10, help="Time step")
#     parser.add_argument('--save_interval', type=int, default=10, help="Save checkpoint after given interval")
#     parser.add_argument('--total_time', type=int, default=10, help="Total time")
#     parser.add_argument('--layers', type=int, default=8, help="Number of layers")
   
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()

#     model = HeldSuarezModel(
#         time_step=args.time_step,
#         save_interval=args.save_interval,
#         total_time=args.total_time,
#         layers=args.layers
#     )
    
#     # Get the initial state
#     state = model.get_initial_state()
#     state.tracers = {
#             'specific_humidity': primitive_equations_states.gaussian_scalar(
#                 model.coords, model.physics_specs)}
#     # Use the initial state to call unroll
#     final_state, predictions = model.unroll(state)
    
#     ds = model.data_to_xarray(asdict(predictions))
#     ds.to_netcdf("model_state.nc")
    
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