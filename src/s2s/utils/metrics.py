# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchmetrics import Metric
from s2s.utils.data_utils import SURFACE_VARS, ATMOSPHERIC_VARS, NAME_TO_WEIGHT, PRESSURE_LEVEL_WEIGHTS_DICT

class mse(Metric):
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
            var_loss_name = f"w_mse_{var}_{self.suffix}" if self.suffix else f"w_mse_{var}"
            loss_dict[var_loss_name] = self.w_mse_over_hw_sum[i] / self.count

        loss_name = f"w_mse_{self.suffix}" if self.suffix else f"w_mse"
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))

        return loss_dict

class pressure_level_lat_weighted_mse(Metric):
    def __init__(self, vars, lat, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("w_mse_over_hw_sum", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        self.warning_printed = False
        level_weights = []
        for v in self.vars:
            if v in PRESSURE_LEVEL_WEIGHTS_DICT:
                level_weights.append(PRESSURE_LEVEL_WEIGHTS_DICT[v])
            else:
                raise ValueError(f"{v} does not have a pressure level weight assigned")                    
        self.register_buffer("level_weights", torch.tensor(level_weights).view(1, -1, 1, 1)) # Shape (1, 1, H, 1)

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
        w_mse_over_hw = (error * self.w_lat * self.level_weights).mean(dim=(2,3))
        
        self.w_mse_over_hw_sum += w_mse_over_hw.sum(dim=0)
        self.count += B


    def compute(self):
        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f"level_w_mse_{var}_{self.suffix}" if self.suffix else f"level_w_mse_{var}"
            loss_dict[var_loss_name] = self.w_mse_over_hw_sum[i] / self.count

        loss_name = f"level_w_mse_{self.suffix}" if self.suffix else f"level_w_mse"
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))

        return loss_dict

class variable_weighted_mae(Metric):
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