# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from dask.distributed import Client
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
import logging
from s2s.utils.data_utils import DEFAULT_PRESSURE_LEVELS, HRS_PER_LEAP_YEAR, is_leap_year

def process_static(path, static_variables, save_dir):
    os.makedirs(os.path.join(save_dir, "static"), exist_ok=True)
    zarr_ds = xr.open_zarr(path)
    static_ds = xr.Dataset()
    statistics_ds = xr.Dataset()
    for s in static_variables:
        if s == 'orography':
            static_field = zarr_ds['geopotential_at_surface']/9.80665
        else:
            static_field = zarr_ds[s]
        
        if static_field.dims == ("longitude", "latitude"):
            static_field = static_field.transpose("latitude", "longitude") #transpose to T x H x W
        
        static_ds[s] = static_field

        statistics_ds[f"{s}_mean"] = xr.DataArray(static_field.mean().compute().item())
        statistics_ds[f"{s}_std"] = xr.DataArray(static_field.std().compute().item())
    
    data_output_path = os.path.join(save_dir, "static", "static.zarr")
    static_ds.chunk('auto').to_zarr(data_output_path, mode="w")
    
    statistics_output_path = os.path.join(save_dir, "static", "statistics.zarr")
    statistics_ds.to_zarr(statistics_output_path, mode="w")
    
def process_surf_atm(path, variables, years, save_dir, partition, num_shards_per_year, hrs_per_step):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)
    zarr_ds = xr.open_zarr(path)

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

        statistics_output_path = os.path.join(save_dir, partition, "statistics.zarr")
        statistics_ds.to_zarr(statistics_output_path, mode="w")


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
@click.option(
    "--static_variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "land_sea_mask", 
        "orography", 
        "soil_type",
        "geopotential_at_surface",
    ],
)
@click.option("--start_train_year", type=int, default=1979)
@click.option("--start_val_year", type=int, default=2016)
@click.option("--start_test_year", type=int, default=2017)
@click.option("--end_year", type=int, default=2019)
@click.option("--num_shards", type=int, default=8)
@click.option("--hrs_per_step", type=int, default=1)
def main(
    root_dir,
    save_dir,
    variables,
    static_variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
    hrs_per_step,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    client = Client(dashboard_address=':0', local_directory='/glade/derecho/scratch/kvirji/tmp') 
    logger.info(client)

    assert start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    os.makedirs(save_dir, exist_ok=True)

    process_static(root_dir, static_variables, save_dir)

    process_surf_atm(root_dir, variables, train_years, save_dir, "train", num_shards, hrs_per_step)
    process_surf_atm(root_dir, variables, val_years, save_dir, "val", num_shards, hrs_per_step)
    process_surf_atm(root_dir, variables, test_years, save_dir, "test", num_shards, hrs_per_step)

    # save lat and lon data
    zarr_ds = xr.open_zarr(root_dir)
    lat = zarr_ds["latitude"].compute().to_numpy()
    lon = zarr_ds["longitude"].compute().to_numpy()
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)


if __name__ == "__main__":
    main()
