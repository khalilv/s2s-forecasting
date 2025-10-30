# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
ERA5 Data Preprocessing Script

This script preprocesses ERA5 data from WeatherBench2 zarr format into sharded zarr files
suitable for training machine learning models. It handles both surface and atmospheric
variables, computes normalization statistics, and splits data into train/val/test partitions.

The script performs the following steps:
1. Processes static variables (orography, land-sea mask, etc.)
2. Processes surface and atmospheric variables for train/val/test years
3. Computes mean and standard deviation statistics for normalization
4. Saves data as sharded zarr files for efficient loading

Usage:
    python preprocess_era5.py \\
        --root_dir /path/to/era5.zarr \\
        --save_dir /path/to/output \\
        --start_train_year 1979 \\
        --start_val_year 2016 \\
        --start_test_year 2017 \\
        --end_year 2019 \\
        --num_shards 8

Requirements:
    - Input data must be in zarr format from WeatherBench2
    - Sufficient disk space for output (can be several TB)
    - Dask distributed client for parallel processing

Adapted from: ClimaX: https://github.com/microsoft/ClimaX
"""

import os
from dask.distributed import Client
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
import logging
from s2s.utils.data_utils import DEFAULT_PRESSURE_LEVELS, HRS_PER_LEAP_YEAR, is_leap_year

# Physical constants
GRAVITY = 9.80665  # Standard gravity in m/s^2 for geopotential conversion


def process_static(path, static_variables, save_dir):
    """
    Process static/time-invariant variables and compute their statistics.

    Static variables include orography, land-sea mask, soil type, etc. These are
    constant across time and are used as conditioning inputs to the model.

    Args:
        path (str): Path to the input zarr dataset
        static_variables (list): List of static variable names to process
        save_dir (str): Directory where processed static data will be saved

    Raises:
        ValueError: If static variable not found in input dataset
        IOError: If unable to write output zarr files

    Output:
        - {save_dir}/static/static.zarr: Static fields
        - {save_dir}/static/statistics.zarr: Mean and std of each static field
    """
    os.makedirs(os.path.join(save_dir, "static"), exist_ok=True)

    try:
        zarr_ds = xr.open_zarr(path)
    except Exception as e:
        raise IOError(f"Failed to open zarr dataset at {path}: {e}")

    static_ds = xr.Dataset()
    statistics_ds = xr.Dataset()

    for s in static_variables:
        # Handle special case: convert geopotential to orography (height in meters)
        if s == 'orography':
            if 'geopotential_at_surface' not in zarr_ds:
                raise ValueError(f"Variable 'geopotential_at_surface' not found in dataset. "
                               f"Available: {list(zarr_ds.data_vars)}")
            static_field = zarr_ds['geopotential_at_surface'] / GRAVITY
        else:
            if s not in zarr_ds:
                raise ValueError(f"Variable '{s}' not found in dataset. "
                               f"Available: {list(zarr_ds.data_vars)}")
            static_field = zarr_ds[s]

        # Ensure consistent dimension ordering (latitude, longitude)
        if static_field.dims == ("longitude", "latitude"):
            static_field = static_field.transpose("latitude", "longitude")

        static_ds[s] = static_field

        # Compute statistics for normalization
        statistics_ds[f"{s}_mean"] = xr.DataArray(static_field.mean().compute().item())
        statistics_ds[f"{s}_std"] = xr.DataArray(static_field.std().compute().item())

    # Save static fields
    data_output_path = os.path.join(save_dir, "static", "static.zarr")
    static_ds.chunk('auto').to_zarr(data_output_path, mode="w")

    # Save statistics
    statistics_output_path = os.path.join(save_dir, "static", "statistics.zarr")
    statistics_ds.to_zarr(statistics_output_path, mode="w")


def process_surf_atm(path, variables, years, save_dir, partition, num_shards_per_year, hrs_per_step):
    """
    Process surface and atmospheric variables and shard them by year.

    This function handles both surface variables (2D: time x lat x lon) and atmospheric
    variables (3D: time x level x lat x lon). Data is split into shards for memory
    efficiency, and normalization statistics are computed for the training partition.

    Args:
        path (str): Path to the input zarr dataset
        variables (list): List of variable names to process
        years (range): Range of years to process
        save_dir (str): Base directory for saving output
        partition (str): Data partition name ('train', 'val', or 'test')
        num_shards_per_year (int): Number of shards to split each year into
        hrs_per_step (int): Hours per timestep (typically 1 or 6)

    Raises:
        ValueError: If variable not found or dimensions don't match expected format
        AssertionError: If timestep doesn't divide evenly into year

    Output:
        - {save_dir}/{partition}/{year}_{shard_id}.zarr: Sharded data files
        - {save_dir}/statistics.zarr: Normalization statistics (train partition only)

    Notes:
        - Handles leap years correctly (366 days)
        - Computes weighted mean/std across all shards for training data
        - Pressure levels are filtered by DEFAULT_PRESSURE_LEVELS from data_utils
    """
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    try:
        zarr_ds = xr.open_zarr(path)
    except Exception as e:
        raise IOError(f"Failed to open zarr dataset at {path}: {e}")

    if partition == "train":
        statistics = {"mean": {}, "std": {}}

    for year in tqdm(years):

        if is_leap_year(year):
            assert HRS_PER_LEAP_YEAR % hrs_per_step == 0
            total_steps = HRS_PER_LEAP_YEAR // hrs_per_step
        else:
            assert (HRS_PER_LEAP_YEAR - 24) % hrs_per_step == 0
            total_steps = (HRS_PER_LEAP_YEAR - 24) // hrs_per_step

        num_steps_per_shard = total_steps // num_shards_per_year

        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_steps_per_shard
            end_id = total_steps if shard_id == num_shards_per_year - 1 else start_id + num_steps_per_shard
            shard_ds = xr.Dataset()

            for var in variables:

                if len(zarr_ds[var].shape) == 3:  # surface level variables (T,H,W)
                    surf_shard = zarr_ds[var].sel(time=str(year))[start_id:end_id]
                    if surf_shard.dims[1:] == ("longitude", "latitude"):
                        surf_shard = surf_shard.transpose("time", "latitude", "longitude") #transpose to T x H x W
                    shard_ds[var] = surf_shard
                    n_surf = surf_shard.time.size

                    if partition == "train":  # compute mean and std of each var in each shard
                        surf_var_mean = surf_shard.mean().compute().item()
                        surf_var_std = surf_shard.std().compute().item()
                        if var not in statistics["mean"]:
                            statistics["mean"][var] = [[surf_var_mean, n_surf]]
                            statistics["std"][var] = [[surf_var_std, n_surf]]
                        else:
                            statistics["mean"][var].append([surf_var_mean, n_surf])
                            statistics["std"][var].append([surf_var_std, n_surf])

                else:  # multiple-level variables
                    assert len(zarr_ds[var].shape) == 4
                    all_levels = zarr_ds["level"][:].compute().to_numpy()
                    all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                    for level in all_levels:
                        ds_level = zarr_ds.sel(level=[level], time=str(year))
                        level = int(level)
                        atm_shard = ds_level[var][start_id:end_id].squeeze()
                        if atm_shard.dims[1:] == ("longitude", "latitude"):
                            atm_shard = atm_shard.transpose("time", "latitude", "longitude")
                        shard_ds[f"{var}_{level}"] = atm_shard
                        n_atm = atm_shard.time.size

                        if partition == "train":  # compute mean and std of each var in each year
                            atm_var_mean = atm_shard.mean().compute().item()
                            atm_var_std = atm_shard.std().compute().item()
                            if f"{var}_{level}" not in statistics["mean"]:
                                statistics["mean"][f"{var}_{level}"] = [[atm_var_mean, n_atm]]
                                statistics["std"][f"{var}_{level}"] = [[atm_var_std, n_atm]]
                            else:
                                statistics["mean"][f"{var}_{level}"].append([atm_var_mean, n_atm])
                                statistics["std"][f"{var}_{level}"].append([atm_var_std, n_atm])
            
            shard_output_path = os.path.join(save_dir, partition, f"{year}_{shard_id}.zarr")
            shard_ds.chunk('auto').to_zarr(shard_output_path, mode="w")
               

    if partition == "train":
        statistics_ds = xr.Dataset()
        for var in statistics["mean"].keys():
            mean = np.stack(statistics["mean"][var], axis=0)
            std = np.stack(statistics["std"][var], axis=0)
            agg_mean = np.sum(mean[:,1] * mean[:,0]) / np.sum(mean[:,1])
            sum_w_var = np.sum((std[:,1] - 1) * (std[:,0]**2))
            sum_group_var = np.sum((std[:,1]) * (mean[:,0] - agg_mean)**2)
            agg_std = np.sqrt((sum_w_var  + sum_group_var)/(np.sum(std[:,1]) - 1))
            statistics_ds[f"{var}_mean"] = xr.DataArray(agg_mean)
            statistics_ds[f"{var}_std"] = xr.DataArray(agg_std)

        statistics_output_path = os.path.join(save_dir, "statistics.zarr")
        statistics_ds.to_zarr(statistics_output_path, mode="w")


@click.command()
@click.option("--root_dir", type=click.Path(exists=True), required=True,
              help="Path to input ERA5 zarr dataset (e.g., /path/to/era5.zarr)")
@click.option("--save_dir", type=str, required=True,
              help="Directory where processed data will be saved")
@click.option("--dask_tmp_dir", type=str, default=None,
              help="Temporary directory for Dask workers (default: system tmp dir)")
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
    help="Variables to process"
)
@click.option(
    "--static_variables",
    "-s",
    type=click.STRING,
    multiple=True,
    default=[
        "land_sea_mask",
        "orography",
        "soil_type",
        "geopotential_at_surface",
    ],
    help="Static variables to process"
)
@click.option("--start_train_year", type=int, default=1979,
              help="First year of training data (inclusive)")
@click.option("--start_val_year", type=int, default=2016,
              help="First year of validation data (inclusive)")
@click.option("--start_test_year", type=int, default=2017,
              help="First year of test data (inclusive)")
@click.option("--end_year", type=int, default=2019,
              help="Last year of test data (exclusive)")
@click.option("--num_shards", type=int, default=8,
              help="Number of shards to split each year into")
@click.option("--hrs_per_step", type=int, default=1,
              help="Hours per timestep (1 for hourly, 6 for 6-hourly)")
def main(
    root_dir,
    save_dir,
    dask_tmp_dir,
    variables,
    static_variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
    hrs_per_step,
):
    """
    Preprocess ERA5 data from WeatherBench2 format.

    This script converts raw ERA5 zarr data into processed, sharded zarr files
    suitable for training ML models. It computes normalization statistics and
    splits data into train/val/test partitions.

    Example:
        python preprocess_era5.py \\
            --root_dir /glade/derecho/scratch/user/DATA/era5_daily/era5.zarr \\
            --save_dir /glade/derecho/scratch/user/DATA/era5_processed \\
            --start_train_year 1979 \\
            --start_val_year 2016 \\
            --start_test_year 2017 \\
            --end_year 2019 \\
            --num_shards 8
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Validate year ranges
    if not (start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year):
        raise ValueError(
            f"Year ranges must be strictly increasing: "
            f"train({start_train_year}-{start_val_year}) < "
            f"val({start_val_year}-{start_test_year}) < "
            f"test({start_test_year}-{end_year})"
        )

    # Initialize Dask client
    client_kwargs = {'dashboard_address': ':0'}
    if dask_tmp_dir:
        client_kwargs['local_directory'] = dask_tmp_dir

    client = Client(**client_kwargs)
    logger.info(f"Dask client initialized: {client}")
    logger.info(f"Dashboard available at: {client.dashboard_link}")
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    logger.info(f"Processing years - Train: {start_train_year}-{start_val_year-1}, "
               f"Val: {start_val_year}-{start_test_year-1}, "
               f"Test: {start_test_year}-{end_year-1}")
    logger.info(f"Variables: {len(variables)} surface/atmospheric, {len(static_variables)} static")
    logger.info(f"Output directory: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    # Process static variables
    logger.info("Processing static variables...")
    process_static(root_dir, static_variables, save_dir)
    logger.info("Static variables complete")

    # Process train/val/test partitions
    logger.info(f"Processing training data ({len(train_years)} years)...")
    process_surf_atm(root_dir, variables, train_years, save_dir, "train", num_shards, hrs_per_step)
    logger.info("Training data complete")

    logger.info(f"Processing validation data ({len(val_years)} years)...")
    process_surf_atm(root_dir, variables, val_years, save_dir, "val", num_shards, hrs_per_step)
    logger.info("Validation data complete")

    logger.info(f"Processing test data ({len(test_years)} years)...")
    process_surf_atm(root_dir, variables, test_years, save_dir, "test", num_shards, hrs_per_step)
    logger.info("Test data complete")

    # Save lat/lon coordinates
    logger.info("Saving latitude/longitude coordinates...")
    zarr_ds = xr.open_zarr(root_dir)
    lat = zarr_ds["latitude"].compute().to_numpy()
    lon = zarr_ds["longitude"].compute().to_numpy()
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)
    logger.info(f"Coordinates saved: lat {lat.shape}, lon {lon.shape}")

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"Output saved to: {save_dir}")
    logger.info("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
