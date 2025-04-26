import hydra
from omegaconf import DictConfig
from jcm.model import Model
from dinosaur import primitive_equations_states
from dataclasses import asdict
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Allows you to run Speedy Model with adjustable parameters
    """
    model = Model(
        time_step=cfg.model.time_step,
        save_interval=cfg.model.save_interval,
        total_time=cfg.model.total_time,
        layers=cfg.model.layers
    )
    
    state = model.get_initial_state()
    
    final_state, predictions = model.unroll(state)
    ds = model.data_to_xarray(asdict(predictions))
    hydra_cfg = HydraConfig.get()
    print(hydra_cfg.mode)
    base_dir = Path('outputs') / hydra_cfg.run.dir.split('outputs/')[-1]
    
    if str(hydra_cfg.mode) == "RunMode.MULTIRUN":
        output_dir = base_dir / 'multirun' / str(hydra_cfg.job.num)
    else:
        output_dir = base_dir
    
    
    filename = f"model_state.nc"
    output_path = output_dir / filename
    
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(output_path))

if __name__ == "__main__":
    main()