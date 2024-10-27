# date: 08/15/2024
# The top-level program. Here we initialize the model and run the main loop
# until the (continually updated) model datetime (`model_datetime`) equals the
# final datetime (`end_datetime`).

import jax.numpy as jnp 

# Import necessary modules
from jcm.params import nsteps, delt, nsteps_out, nstrad
from date import model_datetime, end_datetime, newdate, datetime_equal
from jcm.shortwave_radiation import compute_shortwave
from input_output import output
from coupler import couple_sea_land
from initialization import initialize
from time_stepping import step
from diagnostics import check_diagnostics
from prognostics import vor, div, t, ps, tr, phi
from forcing import set_forcing

def speedy():
    # Time step counter
    model_step = 1

    # Initialization
    initialize()

    # Model main loop
    while not datetime_equal(model_datetime, end_datetime):
        # Daily tasks
        if (model_step - 1) % nsteps == 0:
            # Set forcing terms according to date
            set_forcing(1)

        # Determine whether to compute shortwave radiation on this time step
        compute_shortwave = (model_step % nstrad) == 1

        # Perform one leapfrog time step
        step(2, 2, 2 * delt)

        # Check model diagnostics
        check_diagnostics(vor[:, :, :, 2], div[:, :, :, 2], t[:, :, :, 2], model_step)

        # Increment time step counter
        model_step += 1

        # Increment model datetime
        newdate()

        # Output
        if (model_step - 1) % nsteps_out == 0:
            output(model_step - 1, vor, div, t, ps, tr, phi)

        # Exchange data with coupler
        couple_sea_land(1 + model_step // nsteps)

if __name__ == "__main__":
    speedy()