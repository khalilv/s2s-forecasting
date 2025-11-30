# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchmetrics import Metric
from s2s.utils.data_utils import SURFACE_VARS, ATMOSPHERIC_VARS, NAME_TO_WEIGHT

class mse(Metric):
    """Mean Squared Error metric with optional denormalization.

    Computes MSE averaged over spatial dimensions for each variable independently.

    Args:
        vars (list): Variable names to compute MSE for.
        transforms (callable, optional): Transform to denormalize predictions/targets. Defaults to None.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, vars, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("mse_over_hw_sum", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.vars = vars
        self.transforms = transforms
        self.suffix = suffix

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets)
        
        B = preds.shape[0]          
        error = (preds - targets) ** 2
        mse_over_hw = error.mean(dim=(2,3))

        self.mse_over_hw_sum += mse_over_hw.sum(dim=0)
        self.count += B

    def compute(self):
        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f'mse_{var}_{self.suffix}' if self.suffix else f'mse_{var}'
            loss_dict[var_loss_name] = self.mse_over_hw_sum[i] / self.count
        
        loss_name = f'mse_{self.suffix}' if self.suffix else f'mse'
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))

        return loss_dict

class rmse(Metric):
    """Root Mean Squared Error metric with optional denormalization.

    Computes RMSE averaged over spatial dimensions for each variable independently.

    Args:
        vars (list): Variable names to compute RMSE for.
        transforms (callable, optional): Transform to denormalize predictions/targets. Defaults to None.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, vars, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("mse_over_hw_sum", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.vars = vars
        self.transforms = transforms
        self.suffix = suffix

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets)
        
        B = preds.shape[0]          
        error = (preds - targets) ** 2
        mse_over_hw = error.mean(dim=(2,3))

        self.mse_over_hw_sum += mse_over_hw.sum(dim=0)
        self.count += B

    def compute(self):
        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f'rmse_{var}_{self.suffix}' if self.suffix else f'rmse_{var}'
            loss_dict[var_loss_name] = torch.sqrt(self.mse_over_hw_sum[i] / self.count)

        loss_name = f'rmse_{self.suffix}' if self.suffix else f'rmse'
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))
        
        return loss_dict

class lat_weighted_mse(Metric):
    """Latitude-weighted MSE metric accounting for grid cell area differences.

    Weights errors by cos(latitude) to account for grid cell area variation.
    Optionally applies per-variable weights for multi-variable aggregation.

    Args:
        vars (list): Variable names to compute weighted MSE for.
        lat (np.ndarray): Latitude coordinates in degrees.
        var_weights (list, optional): Per-variable weights. Defaults to uniform weighting.
        transforms (callable, optional): Transform to denormalize predictions/targets. Defaults to None.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, vars, lat, var_weights=None, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("w_mse_over_hw_sum", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        self.warning_printed = False

        #latitude weights
        w_lat = np.cos(np.deg2rad(lat))
        w_lat = w_lat / w_lat.mean()
        self.register_buffer("w_lat", torch.from_numpy(w_lat).view(1, 1, -1, 1))  # Shape (1, 1, H, 1)

        if var_weights:
            w_var = torch.tensor(var_weights)
        else:
            w_var = torch.ones(len(vars))
        self.register_buffer("w_var", w_var.view(1, -1, 1, 1)) # Shape (1, V, 1, 1)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.w_lat.shape[-2] != preds.shape[-2]:
            if not self.warning_printed:
                print(f'Warning: Found mismatch in resolutions w_lat: {self.w_lat.shape[-2]}, prediction: {preds.shape[-2]}. Subsetting w_lat to match preds.')
                self.warning_printed = True
                self.w_lat = self.w_lat[..., :preds.shape[-2], :]
            
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 
        
        B = preds.shape[0]          
        error = (preds - targets) ** 2
        w_mse_over_hw = (error * self.w_lat * self.w_var).mean(dim=(2,3))
        
        self.w_mse_over_hw_sum += w_mse_over_hw.sum(dim=0)
        self.count += B


    def compute(self):
        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f"w_mse_{var}_{self.suffix}" if self.suffix else f"w_mse_{var}"
            loss_dict[var_loss_name] = self.w_mse_over_hw_sum[i] / self.count

        loss_name = f"w_mse_{self.suffix}" if self.suffix else f"w_mse"
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))

        return loss_dict

class variable_weighted_mae(Metric):
    """Variable-weighted MAE with separate weights for surface and atmospheric variables.

    Applies different weighting schemes to surface vs atmospheric variables,
    useful for balancing multi-level forecast errors.

    Args:
        vars (list): Variable names (must be identifiable as surface or atmospheric).
        alpha (float): Weight for surface variables.
        beta (float): Weight for atmospheric variables.
        gamma (float): Global scaling factor.
        transforms (callable, optional): Transform to denormalize predictions/targets. Defaults to None.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, vars, alpha, beta, gamma, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("weighted_mae_over_vhw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.vars = vars
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.transforms = transforms
        self.suffix = suffix
        self.surf_var_idxs = []
        self.atm_var_idxs = []
        surf_var_weights = []
        atm_var_weights = []
        for idx,v in enumerate(self.vars):
            if v in SURFACE_VARS:
                self.surf_var_idxs.append(idx)
                if v in NAME_TO_WEIGHT:
                    surf_var_weights.append(NAME_TO_WEIGHT[v])
                else:
                    surf_var_weights.append(1)
            else:   
                atm_var = '_'.join(v.split('_')[:-1])
                pressure_level = v.split('_')[-1]
                assert pressure_level.isdigit(), f"Found invalid pressure level in {v}"
                if atm_var in ATMOSPHERIC_VARS: 
                    self.atm_var_idxs.append(idx)
                    if atm_var in NAME_TO_WEIGHT:
                        atm_var_weights.append(NAME_TO_WEIGHT[atm_var])
                    else:
                        atm_var_weights.append(1)
                else:
                    raise ValueError(f"{v} could not be identified as a surface or atmospheric variable")                
    
        self.register_buffer("surf_var_weights", torch.tensor(surf_var_weights))
        self.register_buffer("atm_var_weights", torch.tensor(atm_var_weights))

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 

        surf_preds, atm_preds = preds[:,self.surf_var_idxs,:,:], preds[:,self.atm_var_idxs,:,:]
        surf_targets, atm_targets = targets[:,self.surf_var_idxs,:,:], targets[:,self.atm_var_idxs,:,:]
        B, Vs, Va = preds.shape[0], surf_preds.shape[1], atm_preds.shape[1]

        surf_ae = torch.abs(surf_preds - surf_targets) #(T, Vs, H, W)
        surf_mae_over_hw_weighted_sum_over_v = (surf_ae.mean(dim=(-1, -2)) * self.surf_var_weights).sum(dim=1)  #(T,)
        
        atm_ae = torch.abs(atm_preds - atm_targets) #(T, Va, H, W)
        atm_mae_over_hw_weighted_sum_over_v = (atm_ae.mean(dim=(-1, -2)) * self.atm_var_weights).sum(dim=1)  #(T,)
        
        mae_over_hw_weighted_sum_over_v = (self.alpha * surf_mae_over_hw_weighted_sum_over_v) + (self.beta * atm_mae_over_hw_weighted_sum_over_v) 
        self.weighted_mae_over_vhw += ((self.gamma * mae_over_hw_weighted_sum_over_v) / (Vs + Va)).sum(dim=0)
        self.count += B

    def compute(self):
        loss_dict = {}
        loss_name = f"var_w_mae_{self.suffix}" if self.suffix else f"var_w_mae"
        loss_dict[loss_name] = self.weighted_mae_over_vhw / self.count
        return loss_dict

class lat_weighted_rmse(Metric):
    """Latitude-weighted RMSE metric accounting for grid cell area differences.

    Weights errors by cos(latitude) to account for grid cell area variation.

    Args:
        vars (list): Variable names to compute weighted RMSE for.
        lat (np.ndarray): Latitude coordinates in degrees.
        transforms (callable, optional): Transform to denormalize predictions/targets. Defaults to None.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, vars, lat, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("w_mse_over_hw_sum", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        self.warning_printed = False

        #latitude weights
        w_lat = np.cos(np.deg2rad(lat))
        w_lat = w_lat / w_lat.mean()
        self.register_buffer("w_lat", torch.from_numpy(w_lat).view(1, 1, -1, 1))  # Shape (1, 1, H, 1)


    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.w_lat.shape[-2] != preds.shape[-2]:
            if not self.warning_printed:
                print(f'Warning: Found mismatch in resolutions w_lat: {self.w_lat.shape[-2]}, prediction: {preds.shape[-2]}. Subsetting w_lat to match preds.')
                self.warning_printed = True
                self.w_lat = self.w_lat[..., :preds.shape[-2], :]

        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 
        
        B = preds.shape[0]          
        error = (preds - targets) ** 2
        w_mse_over_hw = (error * self.w_lat).mean(dim=(2,3))
        
        self.w_mse_over_hw_sum += w_mse_over_hw.sum(dim=0)
        self.count += B

    def compute(self):
        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f"w_rmse_{var}_{self.suffix}" if self.suffix else f"w_rmse_{var}"
            loss_dict[var_loss_name] = torch.sqrt(self.w_mse_over_hw_sum[i] / self.count)

        loss_name = f"w_rmse_{self.suffix}" if self.suffix else f"w_rmse"
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))
        
        return loss_dict

class rmse_spatial_map(Metric):
    """RMSE computed at each grid point to produce spatial error maps.

    Useful for visualizing where model predictions are most/least accurate.

    Args:
        vars (list): Variable names to compute spatial RMSE for.
        resolution (tuple): Grid resolution (height, width).
        transforms (callable, optional): Transform to denormalize predictions/targets. Defaults to None.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, vars, resolution, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)

        self.add_state("mse_spatial_map_sum", default=torch.zeros(len(vars),resolution[0], resolution[1]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.vars = vars
        self.transforms = transforms
        self.suffix = suffix
        self.warning_printed = False


    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.mse_spatial_map_sum.shape[-2] != preds.shape[-2]:
            if not self.warning_printed:
                print(f'Warning: Found mismatch in resolutions mse_spatial_map: {self.mse_spatial_map_sum.shape[-2]}, prediction: {preds.shape[-2]}. Subsetting mse_spatial_map_sum to match preds.')
                self.warning_printed = True
                self.mse_spatial_map_sum = self.mse_spatial_map_sum[..., :preds.shape[-2], :]

        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 
        
        B = preds.shape[0]          
        error = (preds - targets) ** 2
        self.mse_spatial_map_sum += error.sum(dim=0)
        self.count += B
    
    def compute(self):
        spatial_map_dict = {}

        for i, var in enumerate(self.vars):
            spatial_map_name = f"rmse_spatial_{var}_{self.suffix}" if self.suffix else f"rmse_spatial_{var}"
            spatial_map_dict[spatial_map_name] = torch.sqrt(self.mse_spatial_map_sum[i] / self.count)

        spatial_map_name = f"rmse_spatial_{self.suffix}" if self.suffix else f"rmse"
        spatial_map_dict[spatial_map_name] = torch.mean(torch.stack(list(spatial_map_dict.values())), dim=(0))
        
        return spatial_map_dict


class lat_weighted_acc(Metric):
    """Latitude-weighted Anomaly Correlation Coefficient (ACC) relative to climatology.

    Measures pattern correlation between forecast anomalies and observed anomalies,
    weighted by grid cell area. ACC is a standard metric for weather forecast skill.

    Args:
        vars (list): Variable names to compute ACC for.
        lat (np.ndarray): Latitude coordinates in degrees.
        transforms (callable, optional): Transform to denormalize predictions/targets. Defaults to None.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, vars, lat, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("w_acc_over_hw_sum", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        self.warning_printed = False

        #latitude weights
        w_lat = np.cos(np.deg2rad(lat))
        w_lat = w_lat / w_lat.mean()
        self.register_buffer("w_lat", torch.from_numpy(w_lat).view(1, 1, -1, 1))  # Shape (1, 1, H, 1)

    def update(self, preds: torch.Tensor, targets: torch.Tensor, climatology: torch.Tensor):
        if self.w_lat.shape[-2] != preds.shape[-2]:
            if not self.warning_printed:
                print(f'Warning: Found mismatch in resolutions w_lat: {self.w_lat.shape[-2]}, prediction: {preds.shape[-2]}. Subsetting w_lat to match preds.')
                self.warning_printed = True
                self.w_lat = self.w_lat[..., :preds.shape[-2], :]

        assert preds.shape == climatology.shape
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 
        
        B = preds.shape[0]          
        preds = preds - climatology
        targets = targets - climatology
        
        w_acc_over_hw = torch.sum((self.w_lat * preds * targets), dim=(2,3)) / torch.sqrt(
            torch.sum((self.w_lat * preds**2), dim=(2,3)) * torch.sum((self.w_lat * targets**2), dim=(2,3)))
        
        self.w_acc_over_hw_sum += w_acc_over_hw.sum(dim=0)
        self.count += B

    def compute(self):
        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f"w_acc_{var}_{self.suffix}" if self.suffix else f"w_acc_{var}"
            loss_dict[var_loss_name] = self.w_acc_over_hw_sum[i] / self.count

        loss_name = f"w_acc_{self.suffix}" if self.suffix else f"w_acc"
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))
        
        return loss_dict


class acc_spatial_map(Metric):
    """Anomaly Correlation Coefficient computed at each grid point to produce spatial correlation maps.

    Visualizes spatial patterns of forecast skill relative to climatology.

    Args:
        vars (list): Variable names to compute spatial ACC for.
        resolution (tuple): Grid resolution (height, width).
        transforms (callable, optional): Transform to denormalize predictions/targets. Defaults to None.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, vars, resolution, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)

        self.add_state("acc_spatial_map_sum_tp", default=torch.zeros(len(vars),resolution[0], resolution[1]), dist_reduce_fx="sum")
        self.add_state("acc_spatial_map_sum_pp", default=torch.zeros(len(vars),resolution[0], resolution[1]), dist_reduce_fx="sum")
        self.add_state("acc_spatial_map_sum_tt", default=torch.zeros(len(vars),resolution[0], resolution[1]), dist_reduce_fx="sum")
        
        self.vars = vars
        self.transforms = transforms
        self.suffix = suffix
        self.warning_printed = False

    def update(self, preds: torch.Tensor, targets: torch.Tensor, climatology: torch.Tensor):
        if self.acc_spatial_map_sum_tp.shape[-2] != preds.shape[-2]:
            if not self.warning_printed:
                print(f'Warning: Found mismatch in resolutions acc_spatial_map: {self.acc_spatial_map_sum_tp.shape[-2]}, prediction: {preds.shape[-2]}. Subsetting acc_spatial_map to match preds.')
                self.warning_printed = True
                self.acc_spatial_map_sum_tp = self.acc_spatial_map_sum_tp[..., :preds.shape[-2], :]
                self.acc_spatial_map_sum_pp = self.acc_spatial_map_sum_pp[..., :preds.shape[-2], :]
                self.acc_spatial_map_sum_tt = self.acc_spatial_map_sum_tt[..., :preds.shape[-2], :]


        assert preds.shape == climatology.shape
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 
        
        preds = preds - climatology
        targets = targets - climatology
        
        self.acc_spatial_map_sum_tp += torch.sum((preds * targets), dim=(0)) 
        self.acc_spatial_map_sum_pp += torch.sum((preds **2), dim=(0)) 
        self.acc_spatial_map_sum_tt += torch.sum((targets **2), dim=(0)) 
    
    def compute(self):
        spatial_map_dict = {}

        for i, var in enumerate(self.vars):
            spatial_map_name = f"acc_spatial_{var}_{self.suffix}" if self.suffix else f"acc_spatial_{var}"
            spatial_map_dict[spatial_map_name] = self.acc_spatial_map_sum_tp[i] / torch.sqrt(self.acc_spatial_map_sum_pp[i] * self.acc_spatial_map_sum_tt[i])

        spatial_map_name = f"acc_spatial_{self.suffix}" if self.suffix else f"acc"
        spatial_map_dict[spatial_map_name] = torch.mean(torch.stack(list(spatial_map_dict.values())), dim=(0))
        
        return spatial_map_dict

class aggregate_attn_weights(Metric):
    """Aggregate attention weights across batches for analysis.

    Accumulates attention weights from transformer-based models to understand
    which features the model attends to during forecasting.

    Args:
        L (int): Number of layers.
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        suffix (str, optional): Suffix for metric names in output dictionary. Defaults to None.
    """
    def __init__(self, L, C_in, C_out, suffix=None, **kwargs):
        super().__init__(**kwargs)

        self.add_state("attn_weights_sum", default=torch.zeros(L, C_in, C_out), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.suffix = suffix

    def update(self, weights: torch.Tensor):
        assert len(weights.shape) == 4, f'Expected weight tensor with shape [B, L, C_in, C_out] but received tensor with shape {weights.shape}'

        self.attn_weights_sum += torch.sum(weights, dim=0)
        self.count += weights.shape[0]
    
    def compute(self):
        weights_dict = {}

        weight_name = f"attn_weights_{self.suffix}" if self.suffix else f"attn_weights"
        weights_dict[weight_name] = self.attn_weights_sum / self.count

        return weights_dict


if __name__ == '__main__':
    """Example demonstrating forecast metrics computation."""
    import torch

    print("Forecast Metrics Examples")
    print("=" * 70)

    # Create dummy data
    batch_size = 4
    num_vars = 3
    nlat, nlon = 16, 32
    vars = ['2m_temperature', 'geopotential_500', 'u_component_of_wind_850']

    # Generate random predictions, targets, and climatology
    preds = torch.randn(batch_size, num_vars, nlat, nlon) * 10 + 273.15
    targets = preds + torch.randn_like(preds) * 2  # Add some error
    climatology = torch.ones_like(preds) * 273.15  # Simple constant climatology

    # Latitude coordinates
    lat = np.linspace(90, -90, nlat)

    print(f"\nData shapes:")
    print(f"  Predictions: {preds.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Climatology: {climatology.shape}")
    print(f"  Variables: {vars}")

    # 1. Basic MSE
    print("\n1. Mean Squared Error (MSE):")
    print("-" * 70)
    mse_metric = mse(vars=vars, suffix='test')
    mse_metric.update(preds, targets)
    mse_results = mse_metric.compute()
    for key, value in mse_results.items():
        print(f"  {key}: {value:.4f}")

    # 2. RMSE
    print("\n2. Root Mean Squared Error (RMSE):")
    print("-" * 70)
    rmse_metric = rmse(vars=vars, suffix='test')
    rmse_metric.update(preds, targets)
    rmse_results = rmse_metric.compute()
    for key, value in rmse_results.items():
        print(f"  {key}: {value:.4f}")

    # 3. Latitude-weighted MSE
    print("\n3. Latitude-Weighted MSE:")
    print("-" * 70)
    lat_mse_metric = lat_weighted_mse(vars=vars, lat=lat, suffix='test')
    lat_mse_metric.update(preds, targets)
    lat_mse_results = lat_mse_metric.compute()
    for key, value in lat_mse_results.items():
        print(f"  {key}: {value:.4f}")

    # 4. Latitude-weighted RMSE
    print("\n4. Latitude-Weighted RMSE:")
    print("-" * 70)
    lat_rmse_metric = lat_weighted_rmse(vars=vars, lat=lat, suffix='test')
    lat_rmse_metric.update(preds, targets)
    lat_rmse_results = lat_rmse_metric.compute()
    for key, value in lat_rmse_results.items():
        print(f"  {key}: {value:.4f}")

    # 5. Latitude-weighted ACC (requires climatology)
    print("\n5. Latitude-Weighted Anomaly Correlation Coefficient (ACC):")
    print("-" * 70)
    acc_metric = lat_weighted_acc(vars=vars, lat=lat, suffix='test')
    acc_metric.update(preds, targets, climatology)
    acc_results = acc_metric.compute()
    for key, value in acc_results.items():
        print(f"  {key}: {value:.4f}")

    # 6. RMSE Spatial Map
    print("\n6. RMSE Spatial Map:")
    print("-" * 70)
    rmse_spatial_metric = rmse_spatial_map(vars=vars, resolution=(nlat, nlon), suffix='test')
    rmse_spatial_metric.update(preds, targets)
    rmse_spatial_results = rmse_spatial_metric.compute()
    for key, value in rmse_spatial_results.items():
        if 'spatial' in key and len(value.shape) == 2:
            print(f"  {key}: shape={value.shape}, mean={value.mean():.4f}, std={value.std():.4f}")

    # 7. ACC Spatial Map
    print("\n7. ACC Spatial Map:")
    print("-" * 70)
    acc_spatial_metric = acc_spatial_map(vars=vars, resolution=(nlat, nlon), suffix='test')
    acc_spatial_metric.update(preds, targets, climatology)
    acc_spatial_results = acc_spatial_metric.compute()
    for key, value in acc_spatial_results.items():
        if 'spatial' in key and len(value.shape) == 2:
            print(f"  {key}: shape={value.shape}, mean={value.mean():.4f}, std={value.std():.4f}")

    # 8. Variable-weighted MAE
    print("\n8. Variable-Weighted MAE (for mixed surface/atmospheric variables):")
    print("-" * 70)
    var_mae_metric = variable_weighted_mae(
        vars=vars,
        alpha=1.0,  # Surface variable weight
        beta=0.8,   # Atmospheric variable weight
        gamma=1.0,  # Global scale
        suffix='test'
    )
    var_mae_metric.update(preds, targets)
    var_mae_results = var_mae_metric.compute()
    for key, value in var_mae_results.items():
        print(f"  {key}: {value:.4f}")

    # 9. Multiple updates (simulating multiple batches)
    print("\n9. Multi-batch aggregation example:")
    print("-" * 70)
    multi_rmse = rmse(vars=vars)
    for i in range(5):
        batch_preds = torch.randn(batch_size, num_vars, nlat, nlon) * 10 + 273.15
        batch_targets = batch_preds + torch.randn_like(batch_preds) * 2
        multi_rmse.update(batch_preds, batch_targets)
    multi_rmse_results = multi_rmse.compute()
    print(f"  Aggregated over 5 batches ({5*batch_size} total samples):")
    for key, value in multi_rmse_results.items():
        print(f"    {key}: {value:.4f}")

    print("\n" + "=" * 70)
    print("These metrics can be used in PyTorch Lightning modules for")
    print("tracking forecast accuracy during training and validation.")
