"""
Plot ACC spatial maps from saved netCDF files.

This script loads pre-computed ACC spatial maps from netCDF files and creates
visualizations using various plotting options. This allows for flexible
re-plotting without recomputing the expensive ACC calculations.
"""

import os
import glob
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import xarray as xr
import matplotlib.colors as mcolors
from s2s.utils.plot import plot_spatial_map_with_basemap


def load_acc_spatial_map(filepath: str, verbose: bool = True) -> Dict:
    """
    Load ACC spatial map from a netCDF file.

    Parameters
    ----------
    filepath : str
        Path to the netCDF file containing ACC spatial maps
    verbose : bool
        Whether to print information about the loaded data

    Returns
    -------
    Dict
        Dictionary containing:
        - 'acc_maps': Dict with channel names as keys and ACC spatial maps as values
        - 'lat': Latitude values
        - 'lon': Longitude values
        - 'lead_time': Lead time in days
        - 'metadata': Dictionary of metadata attributes
    """
    # Load the dataset
    ds = xr.open_dataset(filepath)

    if verbose:
        print(f"Loaded: {os.path.basename(filepath)}")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Lead time: {ds.attrs.get('lead_time_days', 'N/A')} days")
        if 'months' in ds.attrs:
            print(f"  Months: {ds.attrs['months']}")
        if 'year_range' in ds.attrs:
            print(f"  Year range: {ds.attrs['year_range']}")

    # Extract ACC maps
    acc_maps = {}
    for var_name in ds.data_vars:
        if var_name.startswith('acc_'):
            # Extract channel name (e.g., 'acc_t2m' -> 't2m')
            channel = var_name[4:]
            acc_maps[channel] = ds[var_name].values

    # Extract coordinates
    lat = ds.lat.values
    lon = ds.lon.values

    # Extract metadata
    lead_time = ds.attrs.get('lead_time_days', None)
    metadata = dict(ds.attrs)

    ds.close()

    return {
        'acc_maps': acc_maps,
        'lat': lat,
        'lon': lon,
        'lead_time': lead_time,
        'metadata': metadata
    }


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


def plot_acc_maps_from_files(
    input_dir: str,
    output_dir: str,
    lead_times: Optional[List[int]] = None,
    pattern: str = '*.nc',
    verbose: bool = True
):
    """
    Load ACC maps from netCDF files and create plots.

    Parameters
    ----------
    input_dir : str
        Directory containing the ACC map netCDF files
    output_dir : str
        Directory to save the plots
    lead_times : List[int], optional
        List of specific lead times to plot. If None, plots all available files.
    pattern : str
        Glob pattern to match netCDF files. Default is '*.nc'.
    verbose : bool
        Whether to print progress information
    """
    # Find all netCDF files
    all_files = sorted(glob.glob(os.path.join(input_dir, pattern)))

    if not all_files:
        print(f"No files found matching {pattern} in {input_dir}")
        return

    if verbose:
        print(f"Found {len(all_files)} ACC map files in {input_dir}")

    # Filter by lead times if specified
    if lead_times is not None:
        # Extract lead times from filenames and filter
        filtered_files = []
        for filepath in all_files:
            filename = os.path.basename(filepath)
            # Extract lead time from filename (e.g., 'acc_spatial_lead010_DJF.nc')
            try:
                # Find 'lead' in filename and extract the number after it
                if 'lead' in filename:
                    lead_str = filename.split('lead')[1].split('_')[0].split('.')[0]
                    file_lead_time = int(lead_str)
                    if file_lead_time in lead_times:
                        filtered_files.append(filepath)
            except (IndexError, ValueError):
                if verbose:
                    print(f"Warning: Could not extract lead time from {filename}")
                continue
        files_to_plot = filtered_files
    else:
        files_to_plot = all_files

    if verbose:
        print(f"Plotting {len(files_to_plot)} files...")

    # Process each file
    for i, filepath in enumerate(files_to_plot, 1):
        if verbose:
            print(f"\n[{i}/{len(files_to_plot)}] Processing {os.path.basename(filepath)}")

        # Load the ACC map
        data = load_acc_spatial_map(filepath, verbose=False)

        # Plot the ACC map
        plot_acc_spatial_maps(
            acc_maps=data['acc_maps'],
            lat=data['lat'],
            lon=data['lon'],
            lead_time_days=data['lead_time'],
            output_dir=output_dir,
            verbose=False
        )

        if verbose:
            for channel, acc_map in data['acc_maps'].items():
                # Compute latitude-weighted mean
                lat_rad = np.deg2rad(data['lat'])
                lat_weights = np.cos(lat_rad)
                lat_weights = lat_weights / lat_weights.sum()
                weighted_acc = np.nansum(acc_map * lat_weights[:, np.newaxis], axis=0)
                lat_weighted_mean_acc = np.nanmean(weighted_acc)
                print(f"  {channel}: Lead {data['lead_time']}d, Lat-weighted mean ACC = {lat_weighted_mean_acc:.3f}")

    if verbose:
        print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    # Example usage: Plot ACC maps from saved netCDF files
    print("="*80)
    print("Plotting ACC Spatial Maps from Saved Files")
    print("="*80)

    # Paths
    BASE_INPUT_DIR = '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/fuxi_acc_spatial_maps_DJF'
    ACC_MAPS_DIR = os.path.join(BASE_INPUT_DIR, 'acc_maps')
    OUTPUT_DIR = os.path.join(BASE_INPUT_DIR, 'plots_replot')

    # Parameters
    LEAD_TIMES = None  # None for all, or specify list like [1, 5, 10, 30, 60]
    PATTERN = 'acc_spatial_lead*.nc'  # Pattern to match files

    print(f"\nInput directory: {ACC_MAPS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    if LEAD_TIMES:
        print(f"Lead times to plot: {LEAD_TIMES}")
    else:
        print("Plotting all available lead times")
    print("="*80)

    # Plot the ACC maps
    plot_acc_maps_from_files(
        input_dir=ACC_MAPS_DIR,
        output_dir=OUTPUT_DIR,
        lead_times=LEAD_TIMES,
        pattern=PATTERN,
        verbose=True
    )

    print("\n" + "="*80)
    print(f"Done! Plots saved to {OUTPUT_DIR}")
    print("="*80)
