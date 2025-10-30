"""
Compute ACC (Anomaly Correlation Coefficient) spatial maps for FuXi forecasts.

This module provides functions to:
- Load FuXi forecast ensembles from disk
- Load ERA5 truth data and climatology
- Compute ACC spatial maps comparing forecasts to observations
- Plot and save ACC spatial maps with latitude-weighted global means

Processes forecasts for 2017-2022 across all lead times (1-215 days) and ensemble members.
"""

import os
import glob
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import torch
import matplotlib.colors as mcolors
from s2s.utils.metrics import acc_spatial_map
from s2s.utils.plot import plot_spatial_map_with_basemap


# Mapping from FuXi channel names to ERA5 variable names
FUXI_TO_ERA5_CHANNEL_MAP = {
    't2m': '2m_temperature',
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'msl': 'mean_sea_level_pressure',
    'sp': 'surface_pressure',
    'tp': 'total_precipitation_24hr',
    'tcwv': 'total_column_water_vapour',
    'z500': 'geopotential',
    'u850': 'u_component_of_wind',
    'v850': 'v_component_of_wind',
    'q': 'specific_humidity',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    't': 'temperature',
    'w': 'vertical_velocity'
}


def load_fuxi_forecasts(
    base_dir: str = '/glade/derecho/scratch/kvirji/DATA/MJO/FuXi/generated',
    year_range: tuple = (2017, 2017),
    members: Optional[List[int]] = None,
    lead_times: Optional[List[int]] = None,
    channels: Optional[List[str]] = None,
    init_dates: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    verbose: bool = True
) -> xr.DataArray:
    """
    Lazily load FuXi forecast ensembles into a structured xarray DataArray.

    Parameters
    ----------
    base_dir : str
        Base directory containing FuXi forecasts
    year_range : tuple
        (start_year, end_year) inclusive range of years to load. Default is (2017, 2017).
    members : List[int], optional
        List of ensemble members to load (0-50). If None, loads all 51 members.
    lead_times : List[int], optional
        List of lead times to load (1-215). If None, loads all 215 lead times.
    channels : List[str], optional
        List of channel names to load (e.g., ['t2m', 'u850']). If None, loads only 't2m' by default.
    init_dates : List[str], optionald
        Specific initialization dates to load (format: 'YYYYMMDD').
        If None, loads all dates in year_range.
    chunks : dict, optional
        Dask chunking specification. Default is {'init_time': 1, 'members': 1, 'lead_times': -1}.
    verbose : bool
        Whether to print progress information

    Returns
    -------
    xr.DataArray
        DataArray with dimensions (init_time, lead_times, members, channels, lat, lon)
    """

    # Get list of initialization dates
    if init_dates is None:
        all_dirs = sorted(glob.glob(os.path.join(base_dir, '*')))
        init_dates = []
        for d in all_dirs:
            date_str = os.path.basename(d)
            if len(date_str) == 8 and date_str.isdigit():
                year = int(date_str[:4])
                if year_range[0] <= year <= year_range[1]:
                    init_dates.append(date_str)

    if verbose:
        print(f"Found {len(init_dates)} initialization dates from {init_dates[0]} to {init_dates[-1]}")

    # Set default members, lead times, and channels
    if members is None:
        members = list(range(51))
    if lead_times is None:
        lead_times = list(range(1, 216))
    if channels is None:
        channels = ['t2m']  # Default to only 2m temperature for faster loading

    if verbose:
        print(f"Loading {len(members)} members, {len(lead_times)} lead times, {len(channels)} channels: {channels}")

    # Set default chunking
    if chunks is None:
        chunks = {'init_time': 1, 'members': 1, 'lead_times': -1}

    # Build file paths and load lazily
    datasets = []

    if verbose:
        print("Building dataset structure...")
        init_iter = tqdm(init_dates, desc="Init dates")
    else:
        init_iter = init_dates

    for init_date in init_iter:
        member_datasets = []

        for member in members:
            lead_time_datasets = []

            for lead_time in lead_times:
                file_path = os.path.join(
                    base_dir,
                    init_date,
                    'member',
                    f'{member:02d}',
                    f'{lead_time:02d}.nc'
                )

                if os.path.exists(file_path):
                    # Open with dask for lazy loading
                    ds = xr.open_dataarray(file_path, chunks='auto')

                    # Remove the singleton time dimension and assign lead_time coordinate
                    ds = ds.squeeze('time', drop=True).squeeze('lead_time', drop=True)
                    lead_time_datasets.append(ds)
                else:
                    if verbose:
                        print(f"Warning: Missing file {file_path}")

            if lead_time_datasets:
                # Concatenate along lead_times dimension
                member_da = xr.concat(
                    lead_time_datasets,
                    dim=pd.Index(lead_times[:len(lead_time_datasets)], name='lead_times')
                )
                member_datasets.append(member_da)

        if member_datasets:
            # Concatenate along members dimension
            init_da = xr.concat(
                member_datasets,
                dim=pd.Index(members[:len(member_datasets)], name='members')
            )
            datasets.append(init_da)

    if verbose:
        print("Concatenating all initialization times...")

    # Convert init_dates to datetime objects
    init_times = pd.to_datetime(init_dates, format='%Y%m%d')

    # Concatenate along init_time dimension
    full_dataset = xr.concat(
        datasets,
        dim=pd.Index(init_times, name='init_time')
    )

    # Filter channels if specified
    if channels is not None:
        if verbose:
            print(f"Filtering to {len(channels)} channels: {channels}")
        full_dataset = full_dataset.sel(channel=channels)

    # Apply chunking for dask
    if chunks:
        full_dataset = full_dataset.chunk(chunks)

    if verbose:
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {full_dataset.shape}")
        print(f"Dimensions: {full_dataset.dims}")
        print(f"Size in memory (if computed): {full_dataset.nbytes / 1e9:.2f} GB")

    return full_dataset


def load_sample_forecast(
    base_dir: str = '/glade/derecho/scratch/kvirji/DATA/MJO/FuXi/generated',
    init_date: str = '20170101',
    member: int = 0,
    lead_time: int = 1
) -> xr.DataArray:
    """
    Load a single forecast file for inspection.

    Parameters
    ----------
    base_dir : str
        Base directory containing FuXi forecasts
    init_date : str
        Initialization date (format: 'YYYYMMDD')
    member : int
        Ensemble member (0-50)
    lead_time : int
        Lead time step (1-215)

    Returns
    -------
    xr.DataArray
        Single forecast with dimensions (time, lead_time, channel, lat, lon)
    """
    file_path = os.path.join(
        base_dir,
        init_date,
        'member',
        f'{member:02d}',
        f'{lead_time:02d}.nc'
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return xr.open_dataarray(file_path)


def get_available_init_dates(
    base_dir: str = '/glade/derecho/scratch/kvirji/DATA/MJO/FuXi/generated',
    year_range: Optional[tuple] = None
) -> List[str]:
    """
    Get list of available initialization dates.

    Parameters
    ----------
    base_dir : str
        Base directory containing FuXi forecasts
    year_range : tuple, optional
        (start_year, end_year) to filter dates

    Returns
    -------
    List[str]
        List of initialization dates in 'YYYYMMDD' format
    """
    all_dirs = sorted(glob.glob(os.path.join(base_dir, '*')))
    init_dates = []

    for d in all_dirs:
        date_str = os.path.basename(d)
        if len(date_str) == 8 and date_str.isdigit():
            if year_range:
                year = int(date_str[:4])
                if year_range[0] <= year <= year_range[1]:
                    init_dates.append(date_str)
            else:
                init_dates.append(date_str)

    return init_dates


def filter_by_valid_time_month(
    ds: xr.DataArray,
    lead_time_days: int,
    months: List[int],
    verbose: bool = True
) -> xr.DataArray:
    """
    Filter forecasts to those where valid time (init_time + lead_time) falls in specified months.

    Parameters
    ----------
    ds : xr.DataArray
        DataArray with dimensions (init_time, lead_times, members, channel, lat, lon)
    lead_time_days : int
        Lead time in days to select
    months : List[int]
        List of month numbers (1-12) to filter by. E.g., [12, 1, 2] for DJF.
    verbose : bool
        Whether to print progress information

    Returns
    -------
    xr.DataArray
        Filtered DataArray with dimensions (init_time, members, channel, lat, lon)
        where init_time values are those whose valid time falls in the specified months.
    """
    # Select the specific lead time
    if 'lead_times' in ds.dims:
        if verbose:
            print(f"Selecting lead time: {lead_time_days} days")
        ds_lead = ds.sel(lead_times=lead_time_days)
    else:
        ds_lead = ds

    # Compute valid times (init_time + lead_time)
    valid_times = ds_lead.init_time + pd.Timedelta(days=lead_time_days)

    # Get months for each valid time
    valid_months = valid_times.dt.month

    # Create mask for desired months
    mask = valid_months.isin(months)

    # Filter to keep only init_times where valid time is in desired months
    ds_filtered = ds_lead.isel(init_time=mask)

    if verbose:
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        month_str = ', '.join([month_names[m] for m in sorted(months)])
        print(f"Filtered to {len(ds_filtered.init_time)} forecasts with valid time in {month_str}")
        print(f"Output shape: {ds_filtered.shape}")
        print(f"Output dimensions: {ds_filtered.dims}")

    return ds_filtered


def compute_acc_spatial_maps(
    forecasts: xr.DataArray,
    truth: xr.DataArray,
    climatology: xr.DataArray,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute ACC spatial maps for forecasts averaged over init_time and members.

    Uses the existing acc_spatial_map metric from metrics.py for consistency.

    Parameters
    ----------
    forecasts : xr.DataArray
        Forecast data with dimensions (init_time, members, channel, lat, lon)
    truth : xr.DataArray
        Ground truth data with dimensions (time, channel, lat, lon)
        where time corresponds to forecast valid times
    climatology : xr.DataArray
        Climatology data with dimensions (dayofyear, channel, lat, lon)
        where dayofyear corresponds to valid day-of-year for each forecast
    verbose : bool
        Whether to print progress information

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with channel names as keys and ACC spatial maps as values
    """
    if verbose:
        print("Computing ACC spatial maps using acc_spatial_map from metrics.py...")
        print(f"Forecasts shape: {forecasts.shape}")
        print(f"Truth shape: {truth.shape}")
        print(f"Climatology shape: {climatology.shape}")

    # Get channels
    channels = list(forecasts.channel.values) if 'channel' in forecasts.dims else [str(forecasts.channel.values)]

    # Get resolution
    resolution = (forecasts.sizes['lat'], forecasts.sizes['lon'])

    # Initialize the acc_spatial_map metric
    metric = acc_spatial_map(vars=channels, resolution=resolution)

    # Align truth and climatology with forecasts
    # Truth has 'time' dimension, rename to 'init_time' for alignment
    # Climatology has 'dayofyear' dimension, rename to 'init_time' for alignment
    truth_aligned = truth.rename({'time': 'init_time'})
    clim_aligned = climatology.rename({'dayofyear': 'init_time'})

    # Expand members dimension for truth and climatology (they are constant across members)
    if 'members' not in truth_aligned.dims and 'members' in forecasts.dims:
        truth_aligned = truth_aligned.expand_dims({'members': forecasts.members}, axis=1)
    if 'members' not in clim_aligned.dims and 'members' in forecasts.dims:
        clim_aligned = clim_aligned.expand_dims({'members': forecasts.members}, axis=1)

    # Stack init_time and members into batch dimension
    # The metric expects (B, V, H, W) - batch, variables, height, width
    pred_stacked = forecasts.stack(batch=('init_time', 'members'))
    target_stacked = truth_aligned.stack(batch=('init_time', 'members'))
    clim_stacked = clim_aligned.stack(batch=('init_time', 'members'))

    # Transpose to get (batch, channel, lat, lon)
    pred_stacked = pred_stacked.transpose('batch', 'channel', 'lat', 'lon')
    target_stacked = target_stacked.transpose('batch', 'channel', 'lat', 'lon')
    clim_stacked = clim_stacked.transpose('batch', 'channel', 'lat', 'lon')

    # Convert to torch tensors
    pred_tensor = torch.from_numpy(pred_stacked.values).float()
    target_tensor = torch.from_numpy(target_stacked.values).float()
    clim_tensor = torch.from_numpy(clim_stacked.values).float()

    if verbose:
        print(f"Tensor shapes: pred={pred_tensor.shape}, target={target_tensor.shape}, clim={clim_tensor.shape}")

    # Update the metric with all data at once
    metric.update(pred_tensor, target_tensor, clim_tensor)

    # Compute the ACC spatial maps
    acc_dict = metric.compute()

    # Convert to numpy arrays and extract results
    acc_maps = {}
    for channel in channels:
        key = f"acc_spatial_{channel}"
        if key in acc_dict:
            acc_map = acc_dict[key].cpu().numpy()
            acc_maps[channel] = acc_map

            if verbose:
                print(f"  Channel {channel}:")
                print(f"    ACC range: [{np.nanmin(acc_map):.3f}, {np.nanmax(acc_map):.3f}]")
                print(f"    Mean ACC: {np.nanmean(acc_map):.3f}")

    return acc_maps


def save_acc_spatial_maps(
    acc_maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    lead_time_days: int,
    output_dir: str,
    init_times: Optional[pd.DatetimeIndex] = None,
    year_range: Optional[tuple] = None,
    months: Optional[List[int]] = None,
    verbose: bool = True
) -> str:
    """
    Save ACC spatial maps to a netCDF file for later analysis.

    Parameters
    ----------
    acc_maps : Dict[str, np.ndarray]
        Dictionary with channel names as keys and ACC spatial maps (lat, lon) as values
    lat : np.ndarray
        Latitude values
    lon : np.ndarray
        Longitude values
    lead_time_days : int
        Lead time in days for the forecast
    output_dir : str
        Directory to save the netCDF file
    init_times : pd.DatetimeIndex, optional
        Initialization times that were used to compute the ACC maps
    year_range : tuple, optional
        (start_year, end_year) for metadata
    months : List[int], optional
        List of months used for filtering (for metadata)
    verbose : bool
        Whether to print progress information

    Returns
    -------
    str
        Path to the saved netCDF file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Convert acc_maps dict to xarray Dataset
    data_vars = {}
    for channel, acc_map in acc_maps.items():
        data_vars[f'acc_{channel}'] = (['lat', 'lon'], acc_map)

    # Add initialization times as a data variable if provided
    if init_times is not None:
        data_vars['init_times'] = (['init_time'], init_times.values)

    # Create the dataset
    coords = {
        'lat': lat,
        'lon': lon,
    }

    # Add init_time as a coordinate if provided
    if init_times is not None:
        coords['init_time'] = init_times

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords
    )

    # Add metadata attributes
    ds.attrs['lead_time_days'] = lead_time_days
    ds.attrs['description'] = 'Anomaly Correlation Coefficient (ACC) spatial maps for FuXi forecasts'
    ds.attrs['created'] = pd.Timestamp.now().isoformat()

    if year_range:
        ds.attrs['year_range'] = f"{year_range[0]}-{year_range[1]}"

    if months:
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        month_str = ','.join([month_names[m] for m in sorted(months)])
        ds.attrs['months'] = month_str
        ds.attrs['months_filter'] = str(months)

    if init_times is not None:
        ds.attrs['n_init_times'] = len(init_times)

    # Build filename
    month_suffix = ''
    if months:
        if months == [12, 1, 2]:
            month_suffix = '_DJF'
        elif months == [3, 4, 5]:
            month_suffix = '_MAM'
        elif months == [6, 7, 8]:
            month_suffix = '_JJA'
        elif months == [9, 10, 11]:
            month_suffix = '_SON'
        else:
            month_suffix = f"_months{''.join(map(str, sorted(months)))}"

    filename = f"acc_spatial_lead{lead_time_days:03d}{month_suffix}.nc"
    filepath = os.path.join(output_dir, filename)

    # Save to netCDF
    ds.to_netcdf(filepath)

    if verbose:
        print(f"  Saved ACC maps to: {filepath}")
        print(f"    Variables: {list(data_vars.keys())}")
        print(f"    Lead time: {lead_time_days} days")
        if months:
            print(f"    Months: {month_str}")
        if init_times is not None:
            print(f"    Number of init times: {len(init_times)}")

    return filepath


def plot_acc_spatial_maps(
    acc_maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    lead_time_days: Optional[int] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True
):
    """
    Plot ACC spatial maps for each channel using basemap.

    Parameters
    ----------
    acc_maps : Dict[str, np.ndarray]
        Dictionary with channel names as keys and ACC spatial maps (lat, lon) as values
    lat : np.ndarray
        Latitude values
    lon : np.ndarray
        Longitude values
    lead_time_days : int, optional
        Lead time in days for the forecast (for title)
    output_dir : str, optional
        Directory to save plots. If None, displays plots instead.
    verbose : bool
        Whether to print progress information
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Create custom discrete colormap for ACC
    acc_colors = ['#FFFECB', '#f5ff63', '#beff31', '#2ae400', '#1c9800',
                  '#136600', '#f74212', '#cf0724', '#9e051b', '#6c0412']
    acc_cmap = mcolors.ListedColormap(acc_colors)
    acc_bounds = np.arange(0.0, 1.1, 0.1)  # [0.0, 0.1, 0.2, ..., 1.0]
    acc_norm = mcolors.BoundaryNorm(acc_bounds, acc_cmap.N)

    # Compute latitude weights (cos(lat) normalized)
    lat_rad = np.deg2rad(lat)
    lat_weights = np.cos(lat_rad)
    lat_weights = lat_weights / lat_weights.sum()

    for channel, acc_map in acc_maps.items():
        if verbose:
            print(f"Plotting ACC map for {channel}...")

        # Compute latitude-weighted mean ACC
        # acc_map has shape (lat, lon), weight by latitude
        weighted_acc = np.nansum(acc_map * lat_weights[:, np.newaxis], axis=0)
        lat_weighted_mean_acc = np.nanmean(weighted_acc)

        # Build title with lead time and weighted mean ACC
        title_parts = ['ACC Spatial Map']
        if lead_time_days is not None:
            title_parts.append(f'Lead: {lead_time_days}d')
        title_parts.append(channel)
        title_parts.append(f'(Lat-weighted mean: {lat_weighted_mean_acc:.3f})')
        title = ' - '.join(title_parts)

        filename = None
        if output_dir:
            filename = os.path.join(output_dir, f'acc_spatial_{channel}_{lead_time_days}.png')

        plot_spatial_map_with_basemap(
            data=acc_map,
            lon=lon,
            lat=lat,
            title=title,
            filename=filename,
            zlabel='Anomaly Correlation Coefficient',
            cMap=acc_cmap,
            norm=acc_norm,
            vmin=0.0,
            vmax=1.0
        )

        if verbose and filename:
            print(f"  Saved to: {filename}")
            print(f"  Latitude-weighted mean ACC: {lat_weighted_mean_acc:.3f}")


def load_era5_truth(
    era5_path: str,
    valid_times: pd.DatetimeIndex,
    channels: List[str],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    verbose: bool = True
) -> xr.DataArray:
    """
    Load ERA5 truth data for specific valid times and interpolate to target grid.

    Parameters
    ----------
    era5_path : str
        Path to ERA5 zarr dataset
    valid_times : pd.DatetimeIndex
        Times to extract from ERA5 (forecast valid times)
    channels : List[str]
        List of channel/variable names to load
    target_lat : np.ndarray
        Target latitude grid
    target_lon : np.ndarray
        Target longitude grid
    verbose : bool
        Whether to print progress information

    Returns
    -------
    xr.DataArray
        Truth data with dimensions (time, channel, lat, lon)
    """
    if verbose:
        print(f"Loading ERA5 truth data from {era5_path}")
        print(f"  Valid times: {len(valid_times)} times from {valid_times[0]} to {valid_times[-1]}")
        print(f"  Channels: {channels}")

    # Open ERA5 dataset
    era5 = xr.open_zarr(era5_path)

    if verbose:
        print(f"  ERA5 dataset loaded with shape: {era5.dims}")

    # Select the times and variables
    # ERA5 time coordinate name might be 'time' or 'datetime'
    time_coord = 'time' if 'time' in era5.dims else 'datetime'

    # Select valid times (with method='nearest' for small mismatches)
    era5_selected = era5.sel({time_coord: valid_times}, method='nearest')

    # Select channels - map FuXi names to ERA5 names
    channel_data = []
    for channel in channels:
        # Get ERA5 variable name from mapping
        era5_var = FUXI_TO_ERA5_CHANNEL_MAP.get(channel, channel)

        if era5_var in era5_selected:
            var_data = era5_selected[era5_var]
            channel_data.append(var_data)
        else:
            raise ValueError(f"Channel {channel} (ERA5: {era5_var}) not found in ERA5 dataset. Available: {list(era5_selected.data_vars)}")

    # Stack channels
    truth = xr.concat(channel_data, dim=pd.Index(channels, name='channel'))

    # Rename coordinates to standard names (lat/lon)
    coord_mapping = {}
    if 'latitude' in truth.dims:
        coord_mapping['latitude'] = 'lat'
    if 'longitude' in truth.dims:
        coord_mapping['longitude'] = 'lon'
    if coord_mapping:
        truth = truth.rename(coord_mapping)

    # Check if grids match (handling reversed coordinates)
    lat_match = (np.allclose(truth.lat.values, target_lat) or
                 np.allclose(truth.lat.values[::-1], target_lat))
    lon_match = (np.allclose(truth.lon.values, target_lon) or
                 np.allclose(truth.lon.values[::-1], target_lon))

    if not (lat_match and lon_match):
        # Grids don't match - need to interpolate
        if verbose:
            print(f"  Interpolating from {truth.lat.size}x{truth.lon.size} to {target_lat.size}x{target_lon.size}")
        truth = truth.interp(lat=target_lat, lon=target_lon, method='linear')
    elif not (np.allclose(truth.lat.values, target_lat) and np.allclose(truth.lon.values, target_lon)):
        # Grids match but are reversed - reindex without interpolation
        if verbose:
            print(f"  Reordering coordinates to match target grid")
        truth = truth.reindex(lat=target_lat, lon=target_lon)

    # Keep time dimension as 'time' (not init_time - this is valid time, not init time!)
    if time_coord != 'time' and time_coord in truth.dims:
        truth = truth.rename({time_coord: 'time'})

    # Transpose to match expected dimension order: (time, channel, lat, lon)
    truth = truth.transpose('time', 'channel', 'lat', 'lon')

    if verbose:
        print(f"  Truth data loaded with shape: {truth.shape}")
        print(f"  Dimensions: {truth.dims}")

    return truth


def load_era5_climatology(
    clim_path: str,
    day_of_year: np.ndarray,
    channels: List[str],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    verbose: bool = True
) -> xr.DataArray:
    """
    Load ERA5 daily climatology for specific day-of-year values.

    Parameters
    ----------
    clim_path : str
        Path to ERA5 climatology zarr dataset
    day_of_year : np.ndarray
        Day-of-year values (1-366) for each forecast
    channels : List[str]
        List of channel/variable names to load
    target_lat : np.ndarray
        Target latitude grid
    target_lon : np.ndarray
        Target longitude grid
    verbose : bool
        Whether to print progress information

    Returns
    -------
    xr.DataArray
        Climatology data with dimensions (time, channel, lat, lon)
    """
    if verbose:
        print(f"Loading ERA5 climatology from {clim_path}")
        print(f"  Day-of-year range: {day_of_year.min()} to {day_of_year.max()}")
        print(f"  Channels: {channels}")

    # Open climatology dataset
    clim = xr.open_zarr(clim_path)

    if verbose:
        print(f"  Climatology dataset loaded with shape: {clim.dims}")

    # The climatology has a dayofyear dimension
    # Select the appropriate days
    clim_selected = clim.sel(dayofyear=day_of_year, method='nearest')

    # Select channels - map FuXi names to ERA5 names
    channel_data = []
    for channel in channels:
        # Get ERA5 variable name from mapping
        era5_var = FUXI_TO_ERA5_CHANNEL_MAP.get(channel, channel)

        if era5_var in clim_selected:
            var_data = clim_selected[era5_var]
            channel_data.append(var_data)
        else:
            raise ValueError(f"Channel {channel} (ERA5: {era5_var}) not found in climatology. Available: {list(clim_selected.data_vars)}")

    # Stack channels
    climatology = xr.concat(channel_data, dim=pd.Index(channels, name='channel'))

    # Rename coordinates to standard names (lat/lon)
    coord_mapping = {}
    if 'latitude' in climatology.dims:
        coord_mapping['latitude'] = 'lat'
    if 'longitude' in climatology.dims:
        coord_mapping['longitude'] = 'lon'
    if coord_mapping:
        climatology = climatology.rename(coord_mapping)

    # Check if grids match (handling reversed coordinates)
    lat_match = (np.allclose(climatology.lat.values, target_lat) or
                 np.allclose(climatology.lat.values[::-1], target_lat))
    lon_match = (np.allclose(climatology.lon.values, target_lon) or
                 np.allclose(climatology.lon.values[::-1], target_lon))

    if not (lat_match and lon_match):
        # Grids don't match - need to interpolate
        if verbose:
            print(f"  Interpolating from {climatology.lat.size}x{climatology.lon.size} to {target_lat.size}x{target_lon.size}")
        climatology = climatology.interp(lat=target_lat, lon=target_lon, method='linear')
    elif not (np.allclose(climatology.lat.values, target_lat) and np.allclose(climatology.lon.values, target_lon)):
        # Grids match but are reversed - reindex without interpolation
        if verbose:
            print(f"  Reordering coordinates to match target grid")
        climatology = climatology.reindex(lat=target_lat, lon=target_lon)

    # Transpose to match expected dimension order: (dayofyear, channel, lat, lon)
    climatology = climatology.transpose('dayofyear', 'channel', 'lat', 'lon')

    if verbose:
        print(f"  Climatology loaded with shape: {climatology.shape}")
        print(f"  Dimensions: {climatology.dims}")

    return climatology


if __name__ == "__main__":
    # Example usage: Compute ACC spatial maps for FuXi forecasts
    print("="*80)
    print("Computing ACC Spatial Maps for FuXi Forecasts")
    print("="*80)

    # Paths
    ERA5_DAILY_PATH = '/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr'
    ERA5_CLIM_PATH = '/glade/derecho/scratch/kvirji/DATA/era5_climatology/1990-2017-daily_clim_daily_mean_61_dw_240x121_equiangular_with_poles_conservative.zarr'

    # Output directories
    BASE_OUTPUT_DIR = '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/fuxi_acc_spatial_maps_DJF'
    ACC_MAPS_DIR = os.path.join(BASE_OUTPUT_DIR, 'acc_maps')  # For intermediate .nc files
    PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'plots')  # For plots

    # Parameters
    YEAR_RANGE = (2017, 2023)  # All years 2017-2022
    MEMBERS = None  # All 51 ensemble members (set to None for all)
    LEAD_TIMES = list(range(1, 216))  # Lead times 1-215 days
    CHANNELS = ['t2m']  # 2-meter temperature
    MONTHS = [12, 1, 2]  # All months (set to None to use all, or [12, 1, 2] for DJF)

    # Memory estimate
    print("\n" + "="*80)
    print("Memory Estimate:")
    print("="*80)
    n_years = YEAR_RANGE[1] - YEAR_RANGE[0] + 1
    approx_init_dates = 12 * n_years  # Approximate
    n_members = 51
    n_lead_times = len(LEAD_TIMES)
    mem_per_lead = approx_init_dates * n_members * 1 * 1 * 121 * 240 * 8 / (1024**3)
    print(f"Approximate init dates: {approx_init_dates}")
    print(f"Members: {n_members}")
    print(f"Lead times: {n_lead_times}")
    print(f"Peak memory per lead time: ~{mem_per_lead:.2f} GB")
    print(f"Processing {n_lead_times} lead times sequentially...")
    print("="*80)

    # Loop through each lead time
    for lead_idx, lead_time_days in enumerate(LEAD_TIMES, 1):
        print(f"\n{'='*80}")
        print(f"Processing Lead Time {lead_time_days}/{LEAD_TIMES[-1]} ({lead_idx}/{n_lead_times})")
        print(f"{'='*80}")

        # Load forecasts for this lead time
        print(f"Loading FuXi forecasts for lead time {lead_time_days}...")
        forecasts = load_fuxi_forecasts(
            year_range=YEAR_RANGE,
            members=MEMBERS,
            lead_times=[lead_time_days],  # Single lead time
            channels=CHANNELS,
            chunks={'init_time': 1, 'members': 1, 'lead_times': -1},
            verbose=False  # Reduce verbosity in loop
        )

        # Filter to specific season if specified
        if MONTHS is not None:
            forecasts_filtered = filter_by_valid_time_month(
                forecasts,
                lead_time_days=lead_time_days,
                months=MONTHS,
                verbose=False
            )
        else:
            # Just remove the lead_times dimension
            forecasts_filtered = forecasts.squeeze('lead_times', drop=True)

        print(f"  Forecasts shape: {forecasts_filtered.shape}")

        # Compute valid times for loading truth data
        valid_times = pd.DatetimeIndex((forecasts_filtered.init_time + pd.Timedelta(days=lead_time_days)).values)

        # Load truth data
        print(f"  Loading ERA5 truth...")
        truth = load_era5_truth(
            era5_path=ERA5_DAILY_PATH,
            valid_times=valid_times,
            channels=CHANNELS,
            target_lat=forecasts_filtered.lat.values,
            target_lon=forecasts_filtered.lon.values,
            verbose=False
        )

        # Load climatology
        print(f"  Loading ERA5 climatology...")
        day_of_year = valid_times.dayofyear.values
        climatology = load_era5_climatology(
            clim_path=ERA5_CLIM_PATH,
            day_of_year=day_of_year,
            channels=CHANNELS,
            target_lat=forecasts_filtered.lat.values,
            target_lon=forecasts_filtered.lon.values,
            verbose=False
        )

        # Compute ACC spatial maps
        print(f"  Computing ACC spatial maps...")
        acc_maps = compute_acc_spatial_maps(
            forecasts=forecasts_filtered,
            truth=truth,
            climatology=climatology,
            verbose=False
        )

        # Save ACC spatial maps to netCDF files
        print(f"  Saving ACC spatial maps to netCDF...")
        save_acc_spatial_maps(
            acc_maps=acc_maps,
            lat=forecasts_filtered.lat.values,
            lon=forecasts_filtered.lon.values,
            lead_time_days=lead_time_days,
            output_dir=ACC_MAPS_DIR,
            init_times=pd.DatetimeIndex(forecasts_filtered.init_time.values),
            year_range=YEAR_RANGE,
            months=MONTHS,
            verbose=False
        )

        # Plot ACC spatial maps
        print(f"  Plotting and saving...")
        # Save all plots in the same directory (filename already includes lead time)
        plot_acc_spatial_maps(
            acc_maps=acc_maps,
            lat=forecasts_filtered.lat.values,
            lon=forecasts_filtered.lon.values,
            lead_time_days=lead_time_days,
            output_dir=PLOTS_DIR,
            verbose=False
        )

        # Print summary for this lead time
        for channel, acc_map in acc_maps.items():
            # Compute latitude-weighted mean
            lat_rad = np.deg2rad(forecasts_filtered.lat.values)
            lat_weights = np.cos(lat_rad)
            lat_weights = lat_weights / lat_weights.sum()
            weighted_acc = np.nansum(acc_map * lat_weights[:, np.newaxis], axis=0)
            lat_weighted_mean_acc = np.nanmean(weighted_acc)
            print(f"  {channel}: Lat-weighted mean ACC = {lat_weighted_mean_acc:.3f}")

    print("\n" + "="*80)
    print("Done! Results saved to:")
    print(f"  ACC maps (netCDF): {ACC_MAPS_DIR}/")
    print(f"    Files: acc_spatial_lead*.nc")
    print(f"  Plots (PNG): {PLOTS_DIR}/")
    print(f"    Files: acc_spatial_*.png")
    print("="*80)