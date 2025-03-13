import xarray as xr
import numpy as np
from s2s.utils.metrics import lat_weighted_rmse
import torch
from tqdm import tqdm

def calculate_climatology_rmse(years, clim_path, era5_path, output_path):
    """Calculate RMSE between ERA5 data and climatology for specified years.
    
    Args:
        years (list): List of years to process
        clim_path (str): Path to climatology zarr file
        era5_path (str): Path to ERA5 zarr file 
        output_path (str): Path to save output npz file
    """
    clim = xr.open_zarr(clim_path).load()
    era5 = xr.open_zarr(era5_path)
    clim_rmse = {}
    
    for var in tqdm(clim.data_vars, desc='Processing variables'):
        if var in era5.data_vars:

            var_labels = [var] if len(era5[var].shape) == 3 else [f'{var}_{l}' for l in era5.level.values]
            rmse = lat_weighted_rmse(vars=var_labels, lat=era5.latitude.values)
            
            for year in years:
                era5_yearly = era5[var].sel(time=era5.time.dt.year == year).load()

                doys = era5_yearly.time.dt.dayofyear
                tods = era5_yearly.time.dt.hour
                if not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)): #match doys with leap year doys
                    doys = doys.where(doys <= 59, doys + 1)

                for i, (t, d) in enumerate(zip(tods.values, doys.values)):
                    clim_pt = clim[var].sel(hour=t, dayofyear=d)
                    era5_pt = era5[var][i]
                    if era5_pt.dims[-2:] == ("longitude", "latitude"):
                        era5_pt = era5_pt.transpose("latitude", "longitude") if len(era5[var].shape) == 3 else era5_pt.transpose("level", "latitude", "longitude")
                    if clim_pt.dims[-2:] == ("longitude", "latitude"):
                        clim_pt = clim_pt.transpose("latitude", "longitude") if len(era5[var].shape) == 3 else clim_pt.transpose("level", "latitude", "longitude")
                    clim_pt = torch.from_numpy(clim_pt.values)
                    era5_pt = torch.from_numpy(era5_pt.values)
                 
                    if len(era5[var].shape) == 3: #unsqueeze variable dimension
                        clim_pt = clim_pt.unsqueeze(0)
                        era5_pt = era5_pt.unsqueeze(0)
                    
                    #unsqueeze batch dimension
                    clim_pt = clim_pt.unsqueeze(0)
                    era5_pt = era5_pt.unsqueeze(0)

                    rmse.update(clim_pt, era5_pt)

            clim_rmse.update(rmse.compute())
            rmse.reset()

    np.savez(output_path, **clim_rmse)

def main():
    years = [2021, 2022]
    clim_path = '/glade/derecho/scratch/kvirji/DATA/era5_climatology/1990-2019_6h_64x32_equiangular_conservative.zarr'
    era5_path = '/glade/derecho/scratch/kvirji/DATA/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
    output_path = '/glade/derecho/scratch/kvirji/s2s-forecasting/src/s2s/utils/climatology_rmse_2021_2022.npz'
    calculate_climatology_rmse(years, clim_path, era5_path, output_path)


if __name__ == "__main__":
    main()