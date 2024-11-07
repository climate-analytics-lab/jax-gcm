def test_radiation_config(model, test_config):
    """Test radiation configuration"""
    assert test_config.radiation.shortwave.enabled
    assert test_config.radiation.shortwave.scheme == "simplified"
    assert test_config.radiation.longwave.enabled
    assert test_config.radiation.longwave.scheme == "simplified"

def test_convection_config(model, test_config):
    """Test convection configuration"""
    assert hasattr(test_config.convection, 'scheme')
    assert hasattr(test_config.convection, 'timestep_ratio')

def test_diffusion_config(model, test_config):
    """Test diffusion configuration"""
    assert test_config.diffusion.vertical.enabled
    assert test_config.diffusion.horizontal.enabled
    assert isinstance(test_config.diffusion.vertical.coefficient, float)