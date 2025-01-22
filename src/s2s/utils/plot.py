import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.patches as patches

def plot_aggregated_variable(npz_files: list, labels: list, variable: str, x_key: str, x_label: str = None, y_label: str = None, title: str = None, output_filename: str = None):
    """Plot variable from multiple npz files with corresponding labels.
    
    Args:
        npz_files (list): List of paths to npz files
        labels (list): List of labels corresponding to each npz file
        variable (str): Name of variable to plot
        x_key (str): Key for x-axis values in npz files
        x_label (str, optional): Label for x-axis. If None, uses x_key
        y_label (str, optional): Label for y-axis. If None, uses variable name
        title (str, optional): Plot title. If None, generates default title
        output_filename (str, optional): Absolute filename to save the plot as PNG
    """
    assert len(npz_files) == len(labels), "Error: number of files and labels must match"
    
    plt.figure(figsize=(10, 6))

    for npz_file, label in zip(npz_files, labels):
        data = np.load(npz_file)
        
        assert variable in data, f"Error: variable {variable} not found in {npz_file}"
        assert x_key in data, f"Error: x-axis key {x_key} not found in {npz_file}"
        assert len(data[variable].shape) == 1, f'Error: variable {variable} with shape {data[variable].shape} must be 1 dimensional'
        assert len(data[x_key].shape) == 1, f'Error: x-axis key {x_key} with shape {data[x_key].shape} must be 1 dimensional'
        
        plt.plot(data[x_key], data[variable], label=label, marker='o')
        
    plt.xlabel(x_label if x_label else x_key)
    plt.ylabel(y_label if y_label else variable)
    plt.title(title if title else f'{y_label if y_label else variable} vs {x_label if x_label else x_key}')
    plt.grid(True)
    plt.legend()
        
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_spatial_map_with_basemap(data: np.ndarray, lon: np.ndarray, lat: np.ndarray, title: str = None, filename: str = None, zlabel: str = "", cMap: str = 'viridis'):
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
    _, ax = plt.subplots(figsize=(16, 6))
    
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

def main():
    
    npz_files = [
        '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/aurora/0.25-pretrained/temporal_attention/phase1-6h-5.625_1step/eval/logs/version_0/results.npz',
        '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/aurora/0.25-pretrained/finetuned/phase1-6h-5.625_1step/eval/logs/version_0/results.npz'
    ]
    labels = [
        'Temporal',
        'Baseline'
    ]
    plot_aggregated_variable(
        npz_files=npz_files,
        labels=labels,
        variable = 'w_rmse_2m_temperature',
        x_key='lead_time_hrs',
        x_label='Lead time (hrs)',
        y_label='RMSE',
        title='T2M RMSE vs Lead Time (hrs)',
        output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/rmse_t2m_12h_hist.png',
    )

if __name__ == "__main__":
    main()
