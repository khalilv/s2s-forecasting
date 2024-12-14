# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch 
import calendar
from matplotlib import pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.patches as patches

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
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "geopotential_at_surface": "z",
    "soil_type": "slt",
    "mean_sea_level_pressure": "msl",
}

#Aurora and WB2 use different short_names for some surface variables
AURORA_NAME_TO_VAR = NAME_TO_VAR.copy()
AURORA_NAME_TO_VAR["2m_temperature"] = "2t"
AURORA_NAME_TO_VAR["10m_u_component_of_wind"] = "10u"
AURORA_NAME_TO_VAR["10m_v_component_of_wind"] = "10v"


STATIC_VARS = [
    "land_sea_mask",
    "orography",
    "geopotential_at_surface",
    "soil_type",
]

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
]

ATMOSPHERIC_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]

NAME_TO_WEIGHT = {
    "mean_sea_level_pressure": 1.6,
    "10m_u_component_of_wind": 0.77,
    "10m_v_component_of_wind": 0.66,
    "2m_temperature": 3.5,
    "geopotential": 3.5,
    "specific_humidity": 0.8,
    "temperature": 1.7,
    "u_component_of_wind": 0.87,
    "v_component_of_wind": 0.6
}

DEFAULT_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

HRS_PER_LEAP_YEAR = 8784

def split_surface_atmospheric(variables: list):
    """Split input variables into surface, and atmospheric variables.
    
    Args:
        variables (list): List of variable names
        
    Returns:
        tuple: Lists of (surface_vars, atmospheric_vars)
    """
    surface_vars = []
    atmospheric_vars = []
    
    for var in variables:
        if var in SURFACE_VARS:
            surface_vars.append(var)
        else:   
            atm_var = '_'.join(var.split('_')[:-1])
            pressure_level = var.split('_')[-1]
            assert pressure_level.isdigit(), f"Found invalid pressure level in {var}"
            if atm_var in ATMOSPHERIC_VARS: 
                if atm_var not in atmospheric_vars:
                    atmospheric_vars.append(atm_var)
            else:
                raise ValueError(f"{var} could not be identified as a surface or atmospheric variable")                
      
    return surface_vars, atmospheric_vars

def collate_fn(batch):
    batch = list(zip(*batch)) 
    inp = torch.stack(batch[0])
    static = torch.stack(batch[1])
    out = torch.stack(batch[2])
    clim = torch.stack(batch[3]) if batch[3][0] is not None else None
    lead_times = torch.stack(batch[4])
    variables = batch[5][0]
    static_variables = batch[6][0]
    out_variables = batch[7][0]
    input_timestamps = np.array(batch[8])
    output_timestamps = np.array(batch[9])
    remaining_predict_steps = np.array(batch[10])
    worker_ids = np.array(batch[11])
    return (
        inp,
        static,
        out,
        clim,
        lead_times,
        variables,
        static_variables,
        out_variables,
        input_timestamps,
        output_timestamps,
        remaining_predict_steps,
        worker_ids
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

def encode_timestamp(timestamp):
    """
    Encodes a timestamp string of the form 'YYYY-MM-DDTHH:MM' into a unique number.
    For example, '2020-01-01T00:00' -> 202001010000 (YYYYMMDDHHMM).
    """
    year = int(timestamp[0:4])
    month = int(timestamp[5:7])
    day = int(timestamp[8:10])
    hour = int(timestamp[11:13])
    minute = int(timestamp[14:16])
    encoded = (year * 100000000) + (month * 1000000) + (day * 10000) + (hour * 100) + minute
    return encoded

def decode_timestamp(encoded):
    """
    Decodes the unique number back into the timestamp string 'YYYY-MM-DDTHH:MM'.
    For example, 202001010000 -> '2020-01-01T00:00'.
    """
    minute = encoded % 100
    encoded = encoded // 100
    hour = encoded % 100
    encoded = encoded // 100
    day = encoded % 100
    encoded = encoded // 100
    month = encoded % 100
    year = encoded // 100
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}"

def remove_year(timestamp):
    """
    Removes the year field from a timestamp to access just the doy and tod.
    For example, '2020-01-01T00:00' -> '01-01T00:00' .
    """
    return timestamp[5:]


def plot_spatial_map_with_basemap(data, lon, lat, title=None, filename=None, zlabel="", cMap='viridis'):
    """
    Plot a spatial map of 2D data with latitude and longitude axes using Basemap, without normalization and with automatic color intervals.
    
    Args:
        data (np.ndarray): 2D array of values to plot
        lon (np.ndarray): 1D array of longitude values
        lat (np.ndarray): 1D array of latitude values
        title (str, optional): Title for the plot
        filename (str, optional): Filename to save the plot as PNG
        zlabel (str, optional): Label for colorbar
        cMap (str, optional): Colormap to use
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    m = Basemap(projection='cyl', resolution='c', 
                llcrnrlat=-90, urcrnrlat=90, 
                llcrnrlon=-180, urcrnrlon=180, ax=ax)
    m.drawcoastlines()

    lon = np.where(lon >= 180, lon - 360, lon)
    lon = np.roll(lon, int(len(lon) / 2))
    data = np.roll(data, int(len(lon) / 2), axis=1)
    
    x, y = np.meshgrid(lon, lat)
    im = m.pcolormesh(x, y, data, cmap=cMap, shading='auto', latlon=True)
    
    cbar = m.colorbar(im, location='bottom', pad=0.03, fraction=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(zlabel, fontsize=12)
    
    top_left_lon, top_left_lat = -126, 50
    bottom_right_lon, bottom_right_lat = -112, 30
    width = bottom_right_lon - top_left_lon
    height = top_left_lat - bottom_right_lat
    
    rect = patches.Rectangle((top_left_lon, bottom_right_lat), width, height, linewidth=2, edgecolor='pink', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    if title:
        plt.title(title)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.close()

    return


def zero_pad(input, pad_rows=1, pad_dim=1):
    pad_shape = list(input.shape)
    pad_shape[pad_dim] = pad_rows
    if isinstance(input, torch.Tensor):
        pad_tensor = torch.zeros(
            pad_shape,
            dtype=input.dtype,
            device=input.device
        )
        return torch.cat((input, pad_tensor), dim=pad_dim)
    elif isinstance(input, np.ndarray):
        pad_array = np.zeros(
            pad_shape,
            dtype=input.dtype
        )
        return np.concatenate((input, pad_array), axis=pad_dim)
    else:
        raise TypeError("Unsupported tensor type. Must be torch.Tensor or np.ndarray.")
