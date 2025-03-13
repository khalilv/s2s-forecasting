import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.patches as patches

def plot_aggregated_variables(npz_files: list, labels: list, variables: list, x_key: str, x_label: str = None, y_label: str = None, titles: list = None, output_filename: str = None, clim_filename = None):
    """Plot multiple variables from multiple npz files with corresponding labels.
    
    Args:
        npz_files (list): List of paths to npz files
        labels (list): List of labels corresponding to each npz file
        variables (list): List of variable names to plot
        x_key (str): Key for x-axis values in npz files
        x_label (str, optional): Label for x-axis. If None, uses x_key
        y_label (str, optional): Label for y-axis. If None, uses variable names
        titles (list, optional): List of plot titles. If None, generates default titles
        output_filename (str, optional): Absolute filename to save the plot as PNG
        clim_filename (str, optional): If provided include climatology errors on plots

    """
    assert len(npz_files) == len(labels), "Error: number of files and labels must match"
    assert len(variables) <= 8, "Error: maximum number of variables is 8"
    
    # Define subplot layout based on number of variables
    layout_map = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (3, 3),
        7: (3, 4),
        8: (4, 4)
    }
    
    nrows, ncols = layout_map[len(variables)]
    fig = plt.figure(figsize=(8*ncols, 6*nrows))
    
    # Generate colors for each npz file that will be consistent across subplots
    colors = plt.cm.rainbow(np.linspace(0, 1, len(npz_files)))

    if clim_filename:
        clim = np.load(clim_filename)
    
    for idx, variable in enumerate(variables):
        ax = plt.subplot(nrows, ncols, idx + 1)
        
        for file_idx, (npz_file, label) in enumerate(zip(npz_files, labels)):
            data = np.load(npz_file)
            
            assert variable in data, f"Error: variable {variable} not found in {npz_file}"
            assert x_key in data, f"Error: x-axis key {x_key} not found in {npz_file}"
            assert len(data[variable].shape) == 1, f'Error: variable {variable} with shape {data[variable].shape} must be 1 dimensional'
            assert len(data[x_key].shape) == 1, f'Error: x-axis key {x_key} with shape {data[x_key].shape} must be 1 dimensional'
            
            ax.plot(data[x_key], data[variable], label=label, marker='.', color=colors[file_idx])

        if clim_filename and variable in clim:
            ax.axhline(y=clim[variable], color='black', linestyle='--', label='Climatology')

        ax.set_xlabel(x_label if x_label else x_key)
        ax.set_ylabel(y_label if y_label else variable)
        ax.set_title(titles[idx] if titles else f'{y_label if y_label else variable} vs {x_label if x_label else x_key}')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout(h_pad=2.0)  # Increase vertical spacing between rows
        
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def plot_attn_weights(npz_file: str, labels: list, variables: list, title: str, x_ticks: list, output_filename: str = None):
    """Plot attention weights for multiple variables across history timesteps.

    Args:
        npz_file (str): Path to .npz file containing attention weights
        labels (list): List of labels for each variable in the plot legend
        variables (list): List of variable names to plot attention weights for
        title (str): Title for the plot
        x_ticks (list): List of x-axis tick labels showing history timesteps
        output_filename (str, optional): If provided, save plot to this file. Defaults to None.
    """
    data = np.load(npz_file)
    history_attn_weights_mean = []
    history_attn_weights_std = []
    for i, variable in enumerate(variables):
        assert variable in data, f"Error: variable {variable} not found in {npz_file}"
        attn_weights = data[variable]
        attn_weights_mean = attn_weights.mean(axis=(0,1,2,3))
        attn_weights_std = attn_weights.std(axis=(0,1,2,3))
        for t in range(len(attn_weights_mean)):
            if i == 0:
                history_attn_weights_mean.append([attn_weights_mean[t]])
                history_attn_weights_std.append([attn_weights_std[t]])
            else:
                history_attn_weights_mean[t].append(attn_weights_mean[t])
                history_attn_weights_std[t].append(attn_weights_std[t])

    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(history_attn_weights_mean)) 
    width = 0.8 / len(variables) 
    
    for i in range(len(variables)):
        values = [weights[i] for weights in history_attn_weights_mean]
        yerr = [weights[i] for weights in history_attn_weights_std]
        x_pos = x - 0.4 + (i + 0.5) * width
        ax.bar(x_pos, values, width, label=labels[i], yerr=yerr, ecolor='black', capsize=3)
    
    ax.set_xlabel('History (hrs)')
    ax.set_ylabel('Weight')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.legend()
    
    plt.tight_layout()
    
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
        variable = 'w_acc',
        x_key='lead_time_hrs',
        x_label='Lead time (hrs)',
        y_label='ACC',
        title='Average ACC vs Lead Time (hrs)',
        output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/acc_average.png',
    )
    plot_aggregated_variable(
        npz_files=npz_files,
        labels=labels,
        variable = 'w_rmse_2m_temperature',
        x_key='lead_time_hrs',
        x_label='Lead time (hrs)',
        y_label='RMSE',
        title='T2M RMSE vs Lead Time (hrs)',
        output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/rmse_t2m.png',
    )
    plot_aggregated_variable(
        npz_files=npz_files,
        labels=labels,
        variable = 'w_acc_2m_temperature',
        x_key='lead_time_hrs',
        x_label='Lead time (hrs)',
        y_label='ACC',
        title='T2M ACC vs Lead Time (hrs)',
        output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/acc_t2m.png',
    )

if __name__ == "__main__":
    main()
