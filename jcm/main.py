from jcm.held_suarez_model import HeldSuarezModel
import argparse
from dinosaur import primitive_equations_states
from dataclasses import asdict

def parse_args():
    # optional arguments
    parser = argparse.ArgumentParser(description="Instantiate and run SpeedyModel.")
    parser.add_argument('--time_step', type=int, default=10, help="Time step")
    parser.add_argument('--save_interval', type=int, default=10, help="Save checkpoint after given interval")
    parser.add_argument('--total_time', type=int, default=10, help="Total time")
    parser.add_argument('--layers', type=int, default=8, help="Number of layers")
   
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = HeldSuarezModel(
        time_step=args.time_step,
        save_interval=args.save_interval,
        total_time=args.total_time,
        layers=args.layers
    )
    
    # Get the initial state
    state = model.get_initial_state()
    state.tracers = {
            'specific_humidity': primitive_equations_states.gaussian_scalar(
                model.coords, model.physics_specs)}
    # Use the initial state to call unroll
    final_state, predictions = model.unroll(state)
    
    ds = model.data_to_xarray(asdict(predictions))
    ds.to_netcdf("model_state.nc")
    