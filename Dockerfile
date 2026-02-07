FROM python:3.11-slim

WORKDIR /app

# Install jcm and dependencies
RUN pip install --upgrade pip && \
    pip install jcm hydra-core omegaconf

# Create config directory and config file
RUN mkdir -p /app/config && \
    cat > /app/config/config.yaml << 'EOF'
model:
  time_step: 10
  save_interval: 10
  total_time: 10
  start_date: "2026-02-01"

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
EOF

# Create custom main script
RUN cat > /app/run_model.py << 'EOF'
import hydra
from omegaconf import DictConfig, OmegaConf
from jcm.model import Model
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import jax_datetime as jdt

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Run JCM Model with adjustable parameters"""
    
    # Allow command-line overrides
    OmegaConf.set_struct(cfg, False)
    
    # Convert string date to jax_datetime.Datetime
    start_date = jdt.to_datetime(cfg.model.start_date)
    
    print(f"🚀 Starting JCM simulation with:")
    print(f"   Time step: {cfg.model.time_step} minutes")
    print(f"   Save interval: {cfg.model.save_interval} days")
    print(f"   Total time: {cfg.model.total_time} days")
    print(f"   Start date: {cfg.model.start_date}")
    print()
    
    model = Model(
        time_step=cfg.model.time_step,
        start_date=start_date
    )
    
    predictions = model.run(
        save_interval=cfg.model.save_interval,
        total_time=cfg.model.total_time
    )
    
    ds = predictions.to_xarray()
    hydra_cfg = HydraConfig.get()
    base_dir = Path('outputs') / hydra_cfg.run.dir.split('outputs/')[-1]
    
    if str(hydra_cfg.mode) == "RunMode.MULTIRUN":
        output_dir = base_dir / 'multirun' / str(hydra_cfg.job.num)
    else:
        output_dir = base_dir
    
    filename = "model_state.nc"
    output_path = output_dir / filename
    
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(output_path))
    
    print()
    print(f"✅ Simulation complete!")
    print(f"📁 Output saved to: {output_path}")

if __name__ == "__main__":
    main()
EOF

# Set the custom script as the entry point
ENTRYPOINT ["python", "/app/run_model.py"]
CMD []