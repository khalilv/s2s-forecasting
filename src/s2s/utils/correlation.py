import glob
import os 
import xarray as xr
import numpy as np
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

output_path = '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/correlation.npz'
dataset_path = '/glade/derecho/scratch/kvirji/DATA/preprocessed/era5/1959-2023_01_10-6h-64x32_equiangular_conservative_n_shards_1/train'
statistics_path = '/glade/derecho/scratch/kvirji/DATA/preprocessed/era5/1959-2023_01_10-6h-64x32_equiangular_conservative_n_shards_1/statistics.zarr'
variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"] + [f'{atm_var}_{level}' for atm_var in ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'relative_humidity', 'specific_humidity'] for level in [50,250,500,600,700,850,925]]
years = [2019]
hrs_each_step = 6

file_list = sorted(glob.glob(os.path.join(dataset_path, "*")))
files = sorted([f for f in file_list if any(f.split('/')[-1].startswith(f"{year}_") for year in years)])
data = xr.open_mfdataset(files, engine="zarr", concat_dim="time", combine="nested",parallel=True, chunks='auto')
data = data[variables].to_array().transpose('time', 'variable', 'latitude', 'longitude')
data = data.to_numpy() #T, V, H, W

var_means = data.mean(axis=(0,2,3))
var_stds = data.std(axis=(0,2,3))
data_norm = (data - var_means.reshape(1, -1, 1, 1)) / var_stds.reshape(1, -1, 1, 1)
data_norm = data_norm.reshape(data.shape[0], -1)

means = []
stds = []
timestep_hours = []
for step in tqdm(range(20)):
    lags = data_norm[:-step] if step > 0 else data_norm
    targets = data_norm[step:]
    vals = []
    for t in range(lags.shape[0]):    
        v = np.corrcoef(targets[t], lags[t])[0, 1]
        # v = normalized_mutual_info_score(targets[t], lags[t])
        vals.append(v)
    vals = np.array(vals)
    means.append(vals.mean())
    stds.append(vals.std())
    timestep_hours.append(step*hrs_each_step)

np.savez(output_path, timestep_hours=timestep_hours, means=means, stds=stds)
