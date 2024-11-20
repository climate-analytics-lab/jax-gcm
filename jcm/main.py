import hydra
from omegaconf import DictConfig
from jcm.model import SpeedyModel
from dinosaur import primitive_equations_states
from dataclasses import asdict

"""
Now when running main, you can pass in arguments through the command line:

python main.py
python main.py model.time_step=20

By default, values are saved as:
time_step: 10
save_interval: 10
total_time: 10
layers: 8

"""
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    model = SpeedyModel(
        time_step=cfg.model.time_step,
        save_interval=cfg.model.save_interval,
        total_time=cfg.model.total_time,
        layers=cfg.model.layers
    )
    
    state = model.get_initial_state()
    state.tracers = {
        'specific_humidity': primitive_equations_states.gaussian_scalar(
            model.coords, model.physics_specs)}
            
    final_state, predictions = model.unroll(state)
    ds = model.data_to_xarray(asdict(predictions))
    ds.to_netcdf("model_state.nc")

if __name__ == "__main__":
    main()