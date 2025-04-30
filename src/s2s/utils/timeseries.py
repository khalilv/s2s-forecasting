import numpy as np
import os
import xarray as xr
from s2s.utils.transforms import NormalizeDenormalize
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

variable_list = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential_50", "geopotential_250", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "u_component_of_wind_50", "u_component_of_wind_250", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "v_component_of_wind_50", "v_component_of_wind_250", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "temperature_50", "temperature_250", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "relative_humidity_50", "relative_humidity_250", "relative_humidity_500", "relative_humidity_600", "relative_humidity_700", "relative_humidity_850", "relative_humidity_925", "specific_humidity_50", "specific_humidity_250", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925"]
predictions_filename = '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/7d_finetune_8_steps/eval_with_save/logs/version_0/outputs.npz'
ground_truth_dir = '/glade/derecho/scratch/kvirji/DATA/preprocessed/era5/1959-2023_01_10-6h-64x32_equiangular_conservative_n_shards_1/test'
statistics_filename = '/glade/derecho/scratch/kvirji/DATA/preprocessed/era5/1959-2023_01_10-6h-64x32_equiangular_conservative_n_shards_1/statistics.zarr'
start_dates = ['2021-01-16T00:00:00.000000000', '2021-06-16T00:00:00.000000000', '2021-09-16T00:00:00.000000000']
h, w = 22, 53
history_steps = 3
variable = '2m_temperature'
output_filename = f'/glade/derecho/scratch/kvirji/s2s-forecasting/plots/timeseries_{variable}_gridpoint_{h}_{w}_7_day_lead_time.png'


########################################################

predictions = np.load(predictions_filename, allow_pickle=True)

ground_truth_pattern = os.path.join(ground_truth_dir, "*.zarr")
ground_truth_zarr = xr.open_mfdataset(ground_truth_pattern, engine="zarr", concat_dim="time", combine="nested")

statistics = xr.open_zarr(statistics_filename, chunks='auto')
mean = np.array([statistics[f"{var}_mean"] for var in variable_list])
std = np.array([statistics[f"{var}_std"] for var in variable_list])
denormalize = NormalizeDenormalize(mean, std)

# Create a larger figure
plt.figure(figsize=(12, 8))

for start_date in start_dates:
    predictions_dict = predictions[start_date].item()
    target_dates, preds = predictions_dict['target_date'], predictions_dict['preds']
    target_dates = np.array([np.datetime64(date) for date in target_dates])
    

    time_resolution = target_dates[1] - target_dates[0]
    # Get 7 days before the first target date
    history = np.array([target_dates[0] - (i+1)*time_resolution for i in range(history_steps)][::-1])
    history_plus_target_dates = np.concatenate([history, target_dates])
    
    # Get ground truth for all dates (including previous days)
    ground_truth = ground_truth_zarr[variable].sel(time=history_plus_target_dates).load()
    preds = denormalize.denormalize(preds)[:, variable_list.index(variable)]
    
    color = plt.cm.tab10(start_dates.index(start_date) % 10) 
    
    # Plot ground truth for all dates
    plt.plot(range(1, len(target_dates) + 1), preds[:, h, w], color=color, label=start_date)
    plt.plot(range(-history_steps + 1, len(target_dates) + 1), ground_truth[:, h, w], color=color, linestyle='--')


ticks = range(-history_steps + 1, len(target_dates) + 1)
plt.xticks(ticks, list(ticks))

# Set labels and title
plt.xlabel('Lead Time (weeks)')
plt.ylabel(f'{variable}')
plt.title(f'Grid Point ({h}, {w})')

legend_elements = []

for i, date in enumerate(start_dates):
    color = plt.cm.tab10(i % 10)
    legend_elements.append(Line2D([0], [0], color=color, label=date))

legend_elements.append(Line2D([0], [0], color='black', linestyle='--', label='Ground Truth'))

plt.legend(handles=legend_elements, loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure to a file
plt.savefig(output_filename, dpi=300, bbox_inches='tight')




