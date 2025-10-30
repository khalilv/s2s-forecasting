# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
ERA5 Climatology Data Preprocessing Script

This script preprocesses ERA5 climatology data from WeatherBench2 zarr format.
Climatology data contains long-term means computed over a reference period
(e.g., 1990-2017), used for computing anomalies and baseline comparisons.

Supports multiple temporal frequencies:
    - **Hourly climatology**: (hour, dayofyear, lat, lon) - 24 hours per day
    - **6-hourly climatology**: (hour, dayofyear, lat, lon) - 4 synoptic times (0, 6, 12, 18)
    - **Daily climatology**: (dayofyear, lat, lon) - daily means (no hour dimension)

The script automatically detects the temporal frequency from the input data structure.

Usage:
    python preprocess_era5_climatology.py \\
        --root_dir /path/to/era5_climatology.zarr \\
        --save_dir /path/to/output/climatology.zarr

Examples:
    # Process hourly climatology (24 hours/day)
    python preprocess_era5_climatology.py \\
        --root_dir /path/to/hourly_climatology.zarr \\
        --save_dir /path/to/output/hourly_clim.zarr

    # Process 6-hourly climatology (4 times/day: 0, 6, 12, 18 UTC)
    python preprocess_era5_climatology.py \\
        --root_dir /path/to/6hourly_climatology.zarr \\
        --save_dir /path/to/output/6hourly_clim.zarr

    # Process daily climatology (no hour dimension)
    python preprocess_era5_climatology.py \\
        --root_dir /path/to/daily_climatology.zarr \\
        --save_dir /path/to/output/daily_clim.zarr

Requirements:
    - Input data must be ERA5 climatology in zarr format from WeatherBench2
    - Sufficient disk space for output
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
from s2s.utils.data_utils import DEFAULT_PRESSURE_LEVELS


def process_climatology(path, variables, save_dir, logger):
    """
    Process ERA5 climatology data and save to zarr format.

    Climatology data contains long-term means for each day of year, computed over
    a reference period (typically 1990-2017). This is used for computing anomalies
    relative to climatological means.

    Supports multiple temporal frequencies:
    - **Hourly**: (hour, dayofyear, lat, lon) - 24 hours per day
    - **6-hourly**: (hour, dayofyear, lat, lon) - 4 synoptic times (0, 6, 12, 18)
    - **Daily**: (dayofyear, lat, lon) - no hour dimension

    Args:
        path (str): Path to input climatology zarr dataset
        variables (list): List of variable names to process
        save_dir (str): Output zarr file path
        logger (logging.Logger): Logger instance for progress messages

    Raises:
        IOError: If unable to read input or write output
        ValueError: If variable not found in dataset

    Output:
        - {save_dir}: Zarr dataset with consistent dimension ordering

    Notes:
        - Automatically detects temporal frequency from data structure
        - Handles both surface and atmospheric variables
        - Pressure levels filtered by DEFAULT_PRESSURE_LEVELS from data_utils
        - Ensures consistent dimension ordering
    """
    try:
        zarr_ds = xr.open_zarr(path)
    except Exception as e:
        raise IOError(f"Failed to open climatology zarr dataset at {path}: {e}")

    # Detect temporal frequency by checking for hour dimension
    has_hour_dim = 'hour' in zarr_ds.dims
    if has_hour_dim:
        n_hours = zarr_ds.dims['hour']
        if n_hours == 24:
            freq_type = "hourly"
        elif n_hours == 4:
            freq_type = "6-hourly"
        else:
            freq_type = f"{n_hours}-hour"
        logger.info(f"Detected {freq_type} climatology (hour dimension with {n_hours} timesteps)")
    else:
        freq_type = "daily"
        logger.info("Detected daily climatology (no hour dimension)")

    climatology_ds = xr.Dataset()
    logger.info(f"Processing {len(variables)} variables...")

    for var in tqdm(variables, desc="Processing variables"):
        if var not in zarr_ds:
            logger.warning(f"Variable '{var}' not found in dataset, skipping...")
            continue

        var_dims = zarr_ds[var].dims
        var_shape = zarr_ds[var].shape

        # Determine if this is a surface or atmospheric variable
        has_level_dim = 'level' in var_dims

        if has_level_dim:
            # Atmospheric variable with pressure levels
            all_levels = zarr_ds["level"][:].compute().to_numpy()
            all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
            logger.debug(f"  Processing atmospheric variable '{var}' at {len(all_levels)} pressure levels")

            for level in all_levels:
                atm_data = zarr_ds.sel(level=[level])[var].squeeze()

                # Ensure consistent dimension ordering
                if has_hour_dim:
                    # Expected: (hour, dayofyear, level, lat, lon) -> remove level, get (hour, dayofyear, lat, lon)
                    expected_dims = ("hour", "dayofyear", "latitude", "longitude")
                    if atm_data.dims[-2:] == ("longitude", "latitude"):
                        atm_data = atm_data.transpose("hour", "dayofyear", "latitude", "longitude")
                    elif atm_data.dims != expected_dims:
                        logger.warning(f"  Unexpected dimension order for {var}_{int(level)}: {atm_data.dims}")
                else:
                    # Expected: (dayofyear, level, lat, lon) -> remove level, get (dayofyear, lat, lon)
                    expected_dims = ("dayofyear", "latitude", "longitude")
                    if atm_data.dims[-2:] == ("longitude", "latitude"):
                        atm_data = atm_data.transpose("dayofyear", "latitude", "longitude")
                    elif atm_data.dims != expected_dims:
                        logger.warning(f"  Unexpected dimension order for {var}_{int(level)}: {atm_data.dims}")

                climatology_ds[f"{var}_{int(level)}"] = atm_data

        else:
            # Surface variable
            surf_data = zarr_ds[var]

            # Ensure consistent dimension ordering
            if has_hour_dim:
                # Expected: (hour, dayofyear, lat, lon)
                expected_dims = ("hour", "dayofyear", "latitude", "longitude")
                if surf_data.dims[-2:] == ("longitude", "latitude"):
                    surf_data = surf_data.transpose("hour", "dayofyear", "latitude", "longitude")
                elif surf_data.dims != expected_dims:
                    logger.warning(f"  Unexpected dimension order for {var}: {surf_data.dims}")
            else:
                # Expected: (dayofyear, lat, lon)
                expected_dims = ("dayofyear", "latitude", "longitude")
                if surf_data.dims[-2:] == ("longitude", "latitude"):
                    surf_data = surf_data.transpose("dayofyear", "latitude", "longitude")
                elif surf_data.dims != expected_dims:
                    logger.warning(f"  Unexpected dimension order for {var}: {surf_data.dims}")

            climatology_ds[var] = surf_data
            logger.debug(f"  Processed surface variable: {var}, shape: {surf_data.shape}")

    # Save processed climatology
    logger.info(f"Saving climatology to {save_dir}...")
    os.makedirs(os.path.dirname(save_dir) if os.path.dirname(save_dir) else ".", exist_ok=True)
    climatology_ds.chunk('auto').to_zarr(save_dir, mode="w")
    logger.info(f"Climatology saved with {len(climatology_ds.data_vars)} variables")
    logger.info(f"Output dimensions: {dict(climatology_ds.dims)}")

@click.command()
@click.option("--root_dir", type=click.Path(exists=True), required=True,
              help="Path to input ERA5 climatology zarr dataset")
@click.option("--save_dir", type=str, required=True,
              help="Path where processed climatology will be saved (e.g., /path/to/output.zarr)")
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
    help="Variables to process (can specify multiple times)"
)
def main(
    root_dir,
    save_dir,
    dask_tmp_dir,
    variables,
):
    """
    Preprocess ERA5 climatology data from WeatherBench2 format.

    This script converts raw ERA5 climatology zarr data into processed format
    with consistent dimension ordering and filtered pressure levels.

    Example:
        python preprocess_era5_climatology.py \\
            --root_dir /glade/derecho/scratch/user/DATA/era5_climatology/climatology.zarr \\
            --save_dir /glade/derecho/scratch/user/DATA/era5_climatology_processed/climatology.zarr
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ERA5 Climatology Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Input: {root_dir}")
    logger.info(f"Output: {save_dir}")
    logger.info(f"Variables: {len(variables)}")

    # Initialize Dask client
    client_kwargs = {'dashboard_address': ':0'}
    if dask_tmp_dir:
        client_kwargs['local_directory'] = dask_tmp_dir

    client = Client(**client_kwargs)
    logger.info(f"Dask client initialized: {client}")
    logger.info(f"Dashboard available at: {client.dashboard_link}")

    # Ensure output directory exists
    output_dir = os.path.dirname(save_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process climatology
    process_climatology(root_dir, variables, save_dir, logger)

    logger.info("=" * 60)
    logger.info("Climatology preprocessing complete!")
    logger.info(f"Output saved to: {save_dir}")
    logger.info("=" * 60)

    client.close()

if __name__ == "__main__":
    main()
