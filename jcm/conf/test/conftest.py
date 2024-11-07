import pytest
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig
from held_suarez_model import HeldSuarezModel
import os

@pytest.fixture
def test_config() -> DictConfig:
    with initialize(version_base=None, config_path="../conf/test"):
        cfg = compose(config_name="config")
        return cfg

@pytest.fixture
def model(test_config) -> HeldSuarezModel:
    os.makedirs(test_config.output.directory, exist_ok=True)
    return HeldSuarezModel(
        time_step=test_config.model.time_step,
        save_interval=test_config.model.save_interval,
        total_time=test_config.model.total_time,
        layers=test_config.model.layers
    )
