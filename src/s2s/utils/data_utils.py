# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch 
import calendar
from matplotlib import pyplot as plt 

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
    "land_sea_mask",
    "orography",
    "lattitude",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]
DEFAULT_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for l in DEFAULT_PRESSURE_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

BOUNDARIES = {
    'NorthAmerica': { # 8x14
        'lat_range': (15, 65),
        'lon_range': (220, 300)
    },
    'SouthAmerica': { # 14x10
        'lat_range': (-55, 20),
        'lon_range': (270, 330)
    },
    'Europe': { # 6x8
        'lat_range': (30, 65),
        'lon_range': (0, 40)
    },
    'SouthAsia': { # 10, 14
        'lat_range': (-15, 45),
        'lon_range': (25, 110)
    },
    'EastAsia': { # 10, 12
        'lat_range': (5, 65),
        'lon_range': (70, 150)
    },
    'Australia': { # 10x14
        'lat_range': (-50, 10),
        'lon_range': (100, 180)
    },
    'Global': { # 32, 64
        'lat_range': (-90, 90),
        'lon_range': (0, 360)
    }
}
HRS_PER_LEAP_YEAR = 8784

def get_region_info(region, lat, lon, patch_size):
    region = BOUNDARIES[region]
    lat_range = region['lat_range']
    lon_range = region['lon_range']
    lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        'patch_ids': valid_patch_ids,
        'min_h': min_h,
        'max_h': max_h,
        'min_w': min_w,
        'max_w': max_w
    }

def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    input_timestamps = [batch[i][5] for i in range(len(batch))]
    output_timestamps = [batch[i][6] for i in range(len(batch))]
    return (
        inp,
        out,
        lead_times,
        [v for v in variables],
        [v for v in out_variables],
        input_timestamps,
        output_timestamps
    )

def leap_year_data_adjustment(data, hrs_per_step):
    leap_year_steps = HRS_PER_LEAP_YEAR // hrs_per_step
    if data.shape[0] < leap_year_steps:
        feb29_start = 59 * 24//hrs_per_step #feb 29th would be the 59th day in a leap year
        data_with_nan = np.insert(data, [feb29_start]*(24//hrs_per_step), np.nan, axis=0)
        return data_with_nan
    else:
        return data

def leap_year_time_adjustment(time, hrs_per_step):
    leap_year_steps = HRS_PER_LEAP_YEAR // hrs_per_step
    if time.shape[0] < leap_year_steps:
        feb29_start = 59 * 24//hrs_per_step #feb 29th would be the 59th day in a leap year
        feb29_vals = [f'02-29T{hh:02d}:00' for hh in range(0, 24, hrs_per_step)]
        adjusted_time = np.insert(time, feb29_start, feb29_vals, axis=0)
        return adjusted_time
    else:
        return time

def is_leap_year(year):
    return calendar.isleap(year)

def plot_spatial_map(data, title=None, filename=None):
    """Plot a spatial map of 2D data with latitude and longitude axes.
    
    Args:
        data (np.ndarray): 2D array of values to plot
        title (str, optional): Title for the plot
        filename (str, optional): Filename to save the plot as PNG
    """    
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return

def encode_timestamp(timestamp_str):
    """
    Encodes a timestamp string of the form 'MM-DDTHH:MM' into a unique number.
    For example, '01-01T00:00' -> 1010000 (MMDDHHMM).
    """
    month = int(timestamp_str[0:2])
    day = int(timestamp_str[3:5])
    hour = int(timestamp_str[6:8])
    minute = int(timestamp_str[9:11])
    encoded = (month * 1000000) + (day * 10000) + (hour * 100) + minute
    return encoded

def decode_timestamp(encoded):
    """
    Decodes the unique number back into the timestamp string 'MM-DDTHH:MM'.
    For example, 1010000 -> '01-01T00:00'.
    """
    minute = encoded % 100
    encoded = encoded // 100
    hour = encoded % 100
    encoded = encoded // 100
    day = encoded % 100
    month = encoded // 100
    return f"{month:02d}-{day:02d}T{hour:02d}:{minute:02d}"

