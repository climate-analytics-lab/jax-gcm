import xarray as xr
import numpy as np
import jax.numpy as jnp
import jcm.slabocean_model as som
from pathlib import Path

class Recorder:

    def __init__(
        self,
        count_per_avg : int, 
        model         : som.Model,
        varnames      : list | tuple = ["sst", "sic", "d_o"],
        output_style: str = "netcdf",
    ) -> None:


        self.count_per_avg = count_per_avg
        self.model = model
        self.varnames = varnames
        self.output_style = output_style

        if output_style not in ["netcdf", "zarr"]:
            raise Exception("Error: Unknown output_style %s " % (output_style,))

        self.clear()


    def clear(self) -> "Recorder":
        
        self.count     = 0
        self.data_hot  = {}
        self.data      = {}
        for varname in self.varnames:
            self.data[varname] = []

        return self

    def avgAndMoveOn(
        self,
    ) -> "Recorder":

        for varname in self.varnames:
            
            if self.count > 0:
                self.data[varname].append( self.data_hot[varname] / self.count )

            self.data_hot[varname][:] = 0.0
    
        self.count = 0

        return self
    
    def record(
        self,
    ) -> "Recorder":
        
        state = self.model.st 
        for varname in self.varnames:
            if varname not in self.data_hot:
                self.data_hot[varname] = np.zeros( getattr(state, varname).shape )

            self.data_hot[varname] += np.asarray(getattr(state, varname))

            #if varname == "sst":
            #    print("[Recorder] Mean of sst = ", np.nanmean(state.sst))
            
        self.count += 1

        if self.count == self.count_per_avg:
            print("Count = %d. Average and move on." % (self.count,))
            self.avgAndMoveOn()

        return self

    def output(
        self,
        filepath  : str | Path,
        force_avg : bool = False,
    ) -> "Recorder":
        
        if self.count != 0:

            if force_avg:
                print("Warning: now count is %d / %d. Force it to avg." % (self.count, self.count_per_avg,))
                self.avgAndMoveOn()
            else:
                raise Exception("Warning: now count is %d / %d." % (self.count, self.count_per_avg,))
        
        # Prepare xarray
        data_vars = {
            varname : ( ["time", "lat", "lon"], np.stack(self.data[varname], axis=0) )
            for varname in self.varnames
        }

        ds = xr.Dataset(
            data_vars = data_vars
        )

        print("Output file: ", str(filepath))
        print("Output style:: ", self.output_style)
        if self.output_style == "netcdf":
            ds.to_netcdf(filepath, unlimited_dims = "time")
        elif self.output_style == "zarr":
            ds.to_zarr(filepath, mode="w")

        self.clear()

        return self
