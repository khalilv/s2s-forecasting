# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from dask.distributed import Client
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
import logging
from s2s.utils.data_utils import DEFAULT_PRESSURE_LEVELS

def process_climatology(path, variables, save_dir, logger):
    zarr_ds = xr.open_zarr(path)
    climatology_ds = xr.Dataset()
    for var in tqdm(variables):
        if len(zarr_ds[var].shape) == 4:  # surface level variables
            surf_data = zarr_ds[var]
            if surf_data.dims[2:] == ("longitude", "latitude"):
                surf_data = surf_data.transpose("hour", "dayofyear", "latitude", "longitude")
            climatology_ds[var] = surf_data
        else:
            assert len(zarr_ds[var].shape) == 5
            all_levels = zarr_ds["level"][:].compute().to_numpy()
            all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
            for level in all_levels:
                atm_data = zarr_ds.sel(level=[level])[var].squeeze()
                if atm_data.dims[2:] == ("longitude", "latitude"):
                    atm_data = atm_data.transpose("hour", "dayofyear", "latitude", "longitude")               
                climatology_ds[f"{var}_{int(level)}"] = atm_data
    climatology_ds.chunk('auto').to_zarr(save_dir, mode="w")

@click.command()
@click.option("--root_dir", type=click.Path(exists=True))
@click.option("--save_dir", type=str)
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_precipitation_6hr",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ],
)
def main(
    root_dir,
    save_dir,
    variables,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    client = Client(dashboard_address=':0', local_directory='/glade/derecho/scratch/kvirji/tmp') 
    logger.info(client)

    os.makedirs(save_dir, exist_ok=True)

    process_climatology(root_dir, variables, save_dir, logger)

if __name__ == "__main__":
    main()
