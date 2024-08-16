# these are all just for clouds - none are shared. clouds function doesn't need any shared variables (i dont think)
# these should get moved to shortwave_radiation_clouds and this file can be deleted

rhcl1   = 0.30  # Relative humidity threshold corresponding to cloud cover = 0
rhcl2   = 1.00  # Relative humidity correponding to cloud cover = 1
qacl    = 0.20  # Specific humidity threshold for cloud cover
wpcl    = 0.2   # Cloud cover weight for the square-root of precipitation (for p = 1 mm/day)
pmaxcl  = 10.0  # Maximum value of precipitation (mm/day) contributing to cloud cover
clsmax  = 0.60  # Maximum stratiform cloud cover
lsminl = 0.15  # Minimum stratiform cloud cover over land (for RH = 1)
gse_s0  = 0.25  # Gradient of dry static energy corresponding to stratiform cloud cover = 0
gse_s1  = 0.40  # Gradient of dry static energy corresponding to stratiform cloud cover = 1
