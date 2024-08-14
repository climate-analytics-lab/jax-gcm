solc    = 342.0 # Solar constant (area averaged) in W/m^2
rhcl1   = 0.30  # Relative humidity threshold corresponding to
                                        # cloud cover = 0
rhcl2   = 1.00  # Relative humidity correponding to cloud cover = 1
qacl    = 0.20  # Specific humidity threshold for cloud cover
wpcl    = 0.2   # Cloud cover weight for the square-root of precipitation
                                        # (for p = 1 mm/day)
pmaxcl  = 10.0  # Maximum value of precipitation (mm/day) contributing to
                                        # cloud cover
clsmax  = 0.60  # Maximum stratiform cloud cover
clsminl = 0.15  # Minimum stratiform cloud cover over land (for RH = 1)
gse_s0  = 0.25  # Gradient of dry static energy corresponding to
                                        # stratiform cloud cover = 0
gse_s1  = 0.40  # Gradient of dry static energy corresponding to
                                        # stratiform cloud cover = 1
albcl   = 0.43  # Cloud albedo (for cloud cover = 1)
albcls  = 0.50  # Stratiform cloud albedo (for st. cloud cover = 1)
epssw   = 0.020 # Fraction of incoming solar radiation absorbed by ozone

# Shortwave absorptivities (for dp = 10^5 Pa)
absdry =  0.033 # Absorptivity of dry air (visible band)
absaer =  0.033 # Absorptivity of aerosols (visible band)
abswv1 =  0.022 # Absorptivity of water vapour
                                        # (visible band, for dq = 1 g/kg)
abswv2 = 15.000 # Absorptivity of water vapour
                                        # (near IR band, for dq = 1 g/kg)
abscl1 =  0.015 # Absorptivity of clouds (visible band, maximum value)
abscl2 =  0.15  # Absorptivity of clouds
                                        # (visible band, for dq_base = 1 g/kg)