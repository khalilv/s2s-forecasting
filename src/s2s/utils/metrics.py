# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchmetrics import Metric

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
        self.add_state("warning_printed", default=torch.tensor(False), dist_reduce_fx=None)

        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        
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

class lat_weighted_rmse(Metric):
    def __init__(self, vars, lat, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("w_mse_over_hw_sum", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("warning_printed", default=torch.tensor(False), dist_reduce_fx=None)

        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix

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

class lat_weighted_rmse_spatial_map(Metric):
    def __init__(self, vars, lat, resolution, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)

        assert(len(lat) == resolution[0]), f"Found mismatch in size between input latitudes {len(lat)} and specified resolution {resolution}"

        self.add_state("w_mse_spatial_map_sum", default=torch.zeros(len(vars),resolution[0], resolution[1]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("warning_printed", default=torch.tensor(False), dist_reduce_fx=None)

        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix

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
                self.w_mse_spatial_map_sum = self.w_mse_spatial_map_sum[..., :preds.shape[-2], :]

        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 
        
        B = preds.shape[0]          
        error = (preds - targets) ** 2
        self.w_mse_spatial_map_sum += (error * self.w_lat).sum(dim=0)
        self.count += B
    
    def compute(self):
        spatial_map_dict = {}

        for i, var in enumerate(self.vars):
            spatial_map_name = f"w_rmse_spatial_{var}_{self.suffix}" if self.suffix else f"w_rmse_spatial_map_{var}"
            spatial_map_dict[spatial_map_name] = torch.sqrt(self.w_mse_spatial_map_sum[i] / self.count)

        spatial_map_name = f"w_rmse_spatial_{self.suffix}" if self.suffix else f"w_rmse"
        spatial_map_dict[spatial_map_name] = torch.mean(torch.stack(list(spatial_map_dict.values())), dim=(0))
        
        return spatial_map_dict


class lat_weighted_acc(Metric):
    def __init__(self, vars, lat, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("w_acc_over_hw_sum", default=torch.zeros(len(vars)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("warning_printed", default=torch.tensor(False), dist_reduce_fx=None)

        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        
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

    
class lat_weighted_acc_spatial_map(Metric):
    def __init__(self, vars, lat, resolution, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        assert(len(lat) == resolution[0]), f"Found mismatch in size between input latitudes {len(lat)} and specified resolution {resolution}"

        self.add_state("w_acc_spatial_map_sum_tp", default=torch.zeros(len(vars),resolution[0], resolution[1]), dist_reduce_fx="sum")
        self.add_state("w_acc_spatial_map_sum_pp", default=torch.zeros(len(vars),resolution[0], resolution[1]), dist_reduce_fx="sum")
        self.add_state("w_acc_spatial_map_sum_tt", default=torch.zeros(len(vars),resolution[0], resolution[1]), dist_reduce_fx="sum")
        self.add_state("warning_printed", default=torch.tensor(False), dist_reduce_fx=None)
        
        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        
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
                self.w_acc_spatial_map_sum_tp = self.w_acc_spatial_map_sum_tp[..., :preds.shape[-2], :]
                self.w_acc_spatial_map_sum_pp = self.w_acc_spatial_map_sum_pp[..., :preds.shape[-2], :]
                self.w_acc_spatial_map_sum_tt = self.w_acc_spatial_map_sum_tt[..., :preds.shape[-2], :]


        assert preds.shape == climatology.shape
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 
        
        preds = preds - climatology
        targets = targets - climatology
        
        self.w_acc_spatial_map_sum_tp += torch.sum((self.w_lat * preds * targets), dim=(0)) 
        self.w_acc_spatial_map_sum_pp += torch.sum((self.w_lat * preds **2), dim=(0)) 
        self.w_acc_spatial_map_sum_tt += torch.sum((self.w_lat * targets **2), dim=(0)) 
    
    def compute(self):
        spatial_map_dict = {}

        for i, var in enumerate(self.vars):
            spatial_map_name = f"w_acc_spatial_{var}_{self.suffix}" if self.suffix else f"w_acc_spatial_{var}"
            spatial_map_dict[spatial_map_name] = self.w_acc_spatial_map_sum_tp[i] / torch.sqrt(self.w_acc_spatial_map_sum_pp[i] * self.w_acc_spatial_map_sum_tt[i])

        spatial_map_name = f"w_acc_spatial_{self.suffix}" if self.suffix else f"w_acc"
        spatial_map_dict[spatial_map_name] = torch.mean(torch.stack(list(spatial_map_dict.values())), dim=(0))
        
        return spatial_map_dict