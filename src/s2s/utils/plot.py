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
        ax.legend(loc='upper right')
    
    plt.tight_layout(h_pad=2.0)  # Increase vertical spacing between rows
        
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_ensemble_members(npz_file: str, members: list, labels: list, variables: list, x_key: str, x_label: str = None, y_label: str = None, titles: list = None, output_filename: str = None, clim_filename = None):
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
    assert len(members) == len(labels), "Error: members list and labels must match in length"
    assert len(variables) <= 8, "Error: maximum number of variablesand is 8"
    
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
    colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)-1))
    colors = np.vstack([colors, [0, 0, 0, 1]])  # Add black as the last color

    if clim_filename:
        clim = np.load(clim_filename)
    
    data = np.load(npz_file)

    for idx, variable in enumerate(variables):
        ax = plt.subplot(nrows, ncols, idx + 1)
        
        for color_idx, (member, label) in enumerate(zip(members, labels)):
            
            member_variable = f'{variable}_{member}' if member else variable
            assert member_variable in data, f"Error: variable {member_variable} not found in {npz_file}"
            assert x_key in data, f"Error: x-axis key {x_key} not found in {npz_file}"
            assert len(data[member_variable].shape) == 1, f'Error: variable {member_variable} with shape {data[member_variable].shape} must be 1 dimensional'
            assert len(data[x_key].shape) == 1, f'Error: x-axis key {x_key} with shape {data[x_key].shape} must be 1 dimensional'
            
            ax.plot(data[x_key], data[member_variable], label=label, marker='.', color=colors[color_idx])

        if clim_filename and variable in clim:
            ax.axhline(y=clim[variable], color='black', linestyle='--', label='Climatology')

        ax.set_xlabel(x_label if x_label else x_key)
        ax.set_ylabel(y_label if y_label else variable)
        ax.set_title(titles[idx] if titles else f'{y_label if y_label else variable} vs {x_label if x_label else x_key}')
        ax.grid(True)
        # ax.legend()
    
    plt.tight_layout(h_pad=2.0)  # Increase vertical spacing between rows
        
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def plot_attn_weights(npz_file: str, key: str, title: str, x_ticks: list, x_label: str, output_filename: str = None):
    """Plot attention weights for multiple variables across history timesteps.

    Args:
        npz_file (str): Path to .npz file containing attention weights
        key (str): Key of weights in data file
        title (str): Title for the plot
        x_ticks (list): List of x-axis tick labels showing history timesteps
        x_label (str): Label for x-axis
        output_filename (str, optional): If provided, save plot to this file. Defaults to None.
    """
    data = np.load(npz_file)
    
    attn_weights = data[key]
    attn_weights_mean = attn_weights.mean(axis=(0,1,2))
    attn_weights_std = attn_weights.std(axis=(0,1,2))
   
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(attn_weights_mean)) 
    width = 0.8
    
    x_pos = x - 0.4 + (0.5) * width
    ax.bar(x_pos, attn_weights_mean, width, yerr=attn_weights_std, ecolor='black', capsize=3)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Weight')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks, rotation=90)
    
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_spatial_map_with_basemap(data: np.ndarray, lon: np.ndarray, lat: np.ndarray, title: str = None, filename: str = None, zlabel: str = "", cMap: str = 'viridis', vmin: float = None, vmax: float = None):
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
        vmin (float, optional): Minimum value for colorbar
        vmax (float, optional): Maximum value for colorbar
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
    im = m.pcolormesh(x, y, data, cmap=cMap, shading='auto', latlon=True, vmin=vmin, vmax=vmax)
    
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

def plot_correlation(npz_file: str, x_label: str, y_label: str, title: str, freq: int, start: int = 0, end: int = None, output_filename: str = None):
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
    means = data['correlation_means'][start:end:freq]
    stds = data['correlation_stds'][start:end:freq] 
    timesteps = data['timestep_hours'][start:end:freq] / 24
    
    plt.plot(timesteps, means, linestyle='-', color = 'g')
    plt.fill_between(timesteps, np.subtract(means, stds), np.add(means, stds), color='lightgreen')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    
    npz_files = [
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/aurora/0.25-pretrained/finetuned/phase1-6h-5.625_1step_patch_size_2/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/irvine/5.625/6hr_lead_time/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/6h_finetune/eval/logs/version_2/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/6h_finetune_4_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/6h_finetune_8_steps/eval/logs/version_2/results.npz',
        '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/1d_finetune/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/1d_finetune_2_steps/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/1d_finetune_4_steps/eval/logs/version_1/results.npz',
        '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/1d_finetune_8_steps/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/3d_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/3d_finetune_2_steps/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/3d_finetune_4_steps/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/3d_finetune_8_steps/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/5d_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/5d_finetune_4_steps/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/5d_finetune_8_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/7d_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/7d_finetune_2_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/7d_finetune_4_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/7d_finetune_8_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/10d_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/10d_finetune_2_step/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/10d_finetune_4_step/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/10d_finetune_8_step/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/14d_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/14d_finetune_2_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/14d_finetune_4_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_1_7d_step_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_1_7d_step_hist_finetune_2_step/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_1_7d_step_hist_finetune_4_step/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_1_7d_step_hist_finetune_8_step/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_4_6h_step_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_[-1432]_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_[-7272,-5812,-4352,-2892,-1432]_hist_finetune/eval_rollout/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_[-7272,-5812,-4352,-2892,-1432]_hist_finetune_2_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/7d_[-7272,-5812,-4352,-2892,-1432]_hist_finetune_4_steps/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/temporal/14d_[-7244,-5784,-4324,-2864,-1404]_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/7d_finetune_4_steps_weighted/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/7d_[-28]_hist_finetune_from_baseline/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/7d_[-28]_hist_finetune_from_baseline_same_lt/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/1d_[-12,-8,-4]_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/mid_temporal_fusion/7d_[-28]_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/mid_temporal_fusion/7d_[-28]_hist_finetune_2_step/eval/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/6h_[-7,-6,-5,-4,-3,-2,-1]_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/6h_[-7,-6,-5,-4,-3,-2,-1]_hist_finetune_2_step/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/6h_[-7,-6,-5,-4,-3,-2,-1]_hist_finetune_4_step/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/6h_[-7,-6,-5,-4,-3,-2,-1]_hist_finetune_nofreeze/eval/logs/version_0/results.npz',
        #'/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h, 12h, 18h, 24h, 30h, 36h, 42h, 48h]/eval/logs/version_0/results.npz',
        #'/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h, 12h, 18h, 24h, 30h, 36h, 42h, 48h]/mean_eval/logs/version_0/results.npz',
        #'/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/6h_[-7,-6,-5,-4,-3,-2,-1]_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[1d,2d,3d,4d]/eval_temporal_fusion/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[1d,2d,3d,4d]/eval_32_member_mean/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[1d,2d,3d,4d]/eval_64_member_mean/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[1d,2d,3d,4d]/eval_homogeneous/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h-7d]/eval_32_member_random_ensemble/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h-7d]/eval_32_member_shortest_path_ensemble/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h-7d]/eval_64_member_random_ensemble/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h-7d]/eval_homogeneous/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h-7d]/eval_32_member_random_ensemble/logs/version_1/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/perceiver/6h_[-4,-3,-2,-1]_hist_finetune/eval/logs/version_0/results.npz',
        # '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/perceiver/6h_[-4,-3,-2,-1]_hist_finetune/eval/logs/version_7/results.npz',
        '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/perceiver/1d_[-32,-28,-24,-20,-16,-12,-8,-4]_hist_finetune/eval/logs/version_0/results.npz',
        '/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/perceiver/1d_full_attn_[-4d,-3d,-2d,-1d]_hist_finetune/eval/logs/version_2/results.npz'

        ]
    labels = [
        '1 day (1 step)',
        '1 day (8 step)',
        '1 day (1 step) with [-8d, -7d, -6d, -5d, -4d, -3d, -2d, -1d, 0] history',
        'Full Attn'

        # '1 day',
        # '3 day',
        # '5 day',
        # '7 day', 
        # '10 day', 
        # '14 day',
        # 'Random 32 member ensemble',       
        # 'Shortest path 32 member ensemble',
        # 'Randomized lead time 32 member ensemble',
        # 'Homogeneous ensemble',
    ]
    acc_variables=['w_acc_2m_temperature', 'w_acc_u_component_of_wind_500', 'w_acc_temperature_850', 'w_acc_geopotential_500', 'w_acc_10m_u_component_of_wind', 'w_acc_10m_v_component_of_wind']
    rmse_variables=['w_rmse_2m_temperature', 'w_rmse_u_component_of_wind_500', 'w_rmse_temperature_850', 'w_rmse_geopotential_500', 'w_rmse_10m_u_component_of_wind', 'w_rmse_10m_v_component_of_wind']
    titles=['T2M', 'U500', 'T850', 'Z500', 'U10', 'V10']
    plot_aggregated_variables(
        npz_files=npz_files,
        labels=labels,
        variables = acc_variables,
        x_key='lead_time_hrs',
        x_label='Lead time (hrs)',
        y_label='ACC',
        titles=titles,
        output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/1d_acc_temporal.png',
    )
    plot_aggregated_variables(
        npz_files=npz_files,
        labels=labels,
        variables = rmse_variables,
        x_key='lead_time_hrs',
        x_label='Lead time (hrs)',
        y_label='RMSE',
        titles=titles,
        output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/1d_rmse_temporal.png',
        clim_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climatology/climatology_rmse_2021_2022.npz'
    )
    # plot_ensemble_members(
    #     npz_file='/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h-7d]/eval_32_member_random_ensemble/logs/version_1/results.npz',
    #     members=[f'member_{i}' for i in range(32)] + [''],
    #     labels = [f'Member {i}' for i in range(32)] + ['Ensemble Mean'],
    #     variables=acc_variables, 
    #     x_key='lead_time_hrs',
    #     x_label='Lead time (hrs)',
    #     y_label='ACC',
    #     titles=titles,
    #     output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/ensemble_members_acc_pres.png',
    # )
    # plot_ensemble_members(
    #     npz_file='/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/baseline_[6h-7d]/eval_32_member_random_ensemble/logs/version_1/results.npz',
    #     members=[f'member_{i}' for i in range(32)] + [''],
    #     labels = [f'Member {i}' for i in range(32)] + ['Ensemble Mean'],
    #     variables=rmse_variables, 
    #     x_key='lead_time_hrs',
    #     x_label='Lead time (hrs)',
    #     y_label='RMSE',
    #     titles=titles,
    #     output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/ensemble_members_rmse_pres.png',
    #     clim_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climatology/climatology_rmse_2021_2022.npz'
    # )

    out_variables = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential_50', 'geopotential_250', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925', 'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'relative_humidity_50', 'relative_humidity_250', 'relative_humidity_500', 'relative_humidity_600', 'relative_humidity_700', 'relative_humidity_850', 'relative_humidity_925', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925']
    # plot_attn_weights(
    #     npz_file='/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/6h_[-7,-6,-5,-4,-3,-2,-1]_hist_finetune/eval/logs/version_0/results.npz',
    #     key='attn_weights_time',
    #     title='Temporal aggregation attention weights',
    #     x_ticks=[t/4 for t in [-7,-6,-5,-4,-3,-2,-1, 0]],
    #     x_label='History (days)',
    #     output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/attn_weights/late_fusion_6h_attn_weights_[-7,-6,-5,-4,-3,-2,-1]_hist_time.png',
    # )
    # plot_attn_weights(
    #     npz_file='/glade/derecho/scratch/kvirji/s2s-forecasting/exps/climaX/all_vars/late_temporal_fusion/6h_[-7,-6,-5,-4,-3,-2,-1]_hist_finetune/eval/logs/version_0/results.npz',
    #     key='attn_weights_var',
    #     title='Variable aggregation attention weights',
    #     x_ticks=out_variables,
    #     x_label='Variable',
    #     output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/attn_weights/late_fusion_6h_attn_weights_[-7,-6,-5,-4,-3,-2,-1]_hist_var.png',
    # )
    # plot_correlation(
    #     npz_file='/glade/derecho/scratch/kvirji/s2s-forecasting/exps/correlation.npz',
    #     x_label='Lag (days)',
    #     y_label='Pearson correlation coeff',
    #     title='Correlation Analysis',
    #     freq=1,
    #     start=0,
    #     end=None,
    #     output_filename='/glade/derecho/scratch/kvirji/s2s-forecasting/plots/correlation.png',

    # )

if __name__ == "__main__":
    main()
