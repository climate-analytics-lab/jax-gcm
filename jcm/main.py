from model import SpeedyModel
import argparse
import dinosaur
from dinosaur.xarray_utils import data_to_xarray

def parse_args():
    parser = argparse.ArgumentParser(description="Instantiate and run SpeedyModel.")
    parser.add_argument('--time_step', type=int, default=10, help="Time step")
    parser.add_argument('--save_interval', type=int, default=10, help="Save checkpoint after given interval")
    parser.add_argument('--total_time', type=int, default=1200, help="Total time")
    parser.add_argument('--layers', type=int, default=8, help="Number of layers")
   
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = SpeedyModel(
        time_step=args.time_step,
        save_interval=args.save_interval,
        total_time=args.total_time,
        layers=args.layers
    )
    state = model.get_initial_state()
    res = model.unroll(state)
    data_to_xarray(res).to_netcdf("model_state.nc")
