# Physical constants used by the model regardless of Physics module
 
p0 = 1e5 # Pressure normalization factor for PhysicsState (Pa)
grav = 9.81 # Gravitational acceleration (m/s/s)
cp = 1004.0 # Specific heat at constant pressure (J/K/kg)
akap = 2.0/7.0 # 1 - 1/gamma where gamma is the heat capacity ratio of a perfect diatomic gas (7/5)
rgas = akap * cp # Gas constant per unit mass for dry air (J/K/kg)