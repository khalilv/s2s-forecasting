# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from s2s.utils.data_utils import encode_timestamp, decode_timestamp, remove_year

class mse(Metric):
    def __init__(self, vars, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.vars = vars
        self.transforms = transforms
        self.suffix = suffix

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 

        error = (preds - targets) ** 2

        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f'mse_{var}_{self.suffix}' if self.suffix else f'mse_{var}'
            loss_dict[var_loss_name] = error[:, i].mean()
        
        loss_name = f'mse_{self.suffix}' if self.suffix else f'mse'
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))

        return loss_dict

class rmse(Metric):
    def __init__(self, vars, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.vars = vars
        self.transforms = transforms
        self.suffix = suffix

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        
        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 

        error = (preds - targets) ** 2

        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f'rmse_{var}_{self.suffix}' if self.suffix else f'rmse_{var}'
            loss_dict[var_loss_name] = torch.sqrt(error[:, i].mean())

        loss_name = f'rmse_{self.suffix}' if self.suffix else f'rmse'
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))
        
        return loss_dict

class lat_weighted_mse(Metric):
    def __init__(self, vars, lat, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)

        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 

        error = (preds - targets) ** 2  # [T, V, H, W]

        # lattitude weights
        w_lat = np.cos(np.deg2rad(self.lat))
        w_lat = w_lat / w_lat.mean()  # (H, )
        w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f"w_mse_{var}_{self.suffix}" if self.suffix else f"w_mse_{var}"
            loss_dict[var_loss_name] = (error[:, i] * w_lat).mean()

        loss_name = f"w_mse_{self.suffix}" if self.suffix else f"w_mse"
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))

        return loss_dict

class lat_weighted_rmse(Metric):
    def __init__(self, vars, lat, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)

        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 

        error = (preds - targets) ** 2  # [T, V, H, W]

        # lattitude weights
        w_lat = np.cos(np.deg2rad(self.lat))
        w_lat = w_lat / w_lat.mean()  # (H, )
        w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss_name = f"w_rmse_{var}_{self.suffix}" if self.suffix else f"w_rmse_{var}"
            loss_dict[var_loss_name] = torch.sqrt((error[:, i] * w_lat).mean())

        loss_name = f"w_rmse_{self.suffix}" if self.suffix else f"w_rmse"
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))
        
        return loss_dict

class lat_weighted_rmse_spatial_map(Metric):
    def __init__(self, vars, lat, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)

        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 

        error = (preds - targets) ** 2  # [T, V, H, W]

        # lattitude weights
        w_lat = np.cos(np.deg2rad(self.lat))
        w_lat = w_lat / w_lat.mean()  # (H, )
        w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

        spatial_map_dict = {}

        for i, var in enumerate(self.vars):
            spatial_map_name = f"w_rmse_spatial_map_{var}_{self.suffix}" if self.suffix else f"w_rmse_spatial_map_{var}"
            spatial_map_dict[spatial_map_name] = torch.sqrt((error[:, i] * w_lat).mean(dim=(0)))

        spatial_map_name = f"w_rmse_{self.suffix}" if self.suffix else f"w_rmse"
        spatial_map_dict[spatial_map_name] = torch.mean(torch.stack(list(spatial_map_dict.values())), dim=(0))
        
        return spatial_map_dict


class lat_weighted_acc(Metric):
    def __init__(self, vars, lat, clim, clim_timestamps, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("encoded_timestamps", default=[], dist_reduce_fx="cat")
        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        self.clim = clim
        self.clim_timestamps = clim_timestamps

    def update(self, preds: torch.Tensor, targets: torch.Tensor, output_timestamps: list):
        self.preds.append(preds)
        self.targets.append(targets)
        encoded_timestamps = torch.tensor([encode_timestamp(ts) for ts in output_timestamps], device=preds.device)
        self.encoded_timestamps.append(encoded_timestamps)
    

    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        encoded_timestamps = dim_zero_cat(self.encoded_timestamps)  
        decoded_timestamps = [decode_timestamp(ts.item()) for ts in encoded_timestamps]
        decoded_timestamps_no_year = [remove_year(ts) for ts in decoded_timestamps]

        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 

        # lattitude weights
        w_lat = np.cos(np.deg2rad(self.lat))
        w_lat = w_lat / w_lat.mean()  # (H, )
        w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=targets.dtype, device=targets.device)  # (1, H, 1)

        clim_timestamp_to_index = {timestamp: idx for idx, timestamp in enumerate(self.clim_timestamps)}
        clim_subset_indices = [clim_timestamp_to_index[timestamp] for timestamp in decoded_timestamps_no_year]
        clim_subset = self.clim[clim_subset_indices]
        clim_subset = clim_subset.to(device=targets.device)
        preds = preds - clim_subset
        targets = targets - clim_subset
        loss_dict = {}

        for i, var in enumerate(self.vars):
            pred_prime = preds[:, i]
            targ_prime = targets[:, i]
            acc = torch.sum((w_lat * pred_prime * targ_prime), dim=(1,2)) / torch.sqrt(
                torch.sum((w_lat * pred_prime**2), dim=(1,2)) * torch.sum((w_lat * targ_prime**2), dim=(1,2)))
            var_loss_name = f"w_acc_{var}_{self.suffix}" if self.suffix else f"w_acc_{var}"
            loss_dict[var_loss_name] = acc.mean()

        loss_name = f"w_acc_{self.suffix}" if self.suffix else f"w_acc"
        loss_dict[loss_name] = torch.mean(torch.stack(list(loss_dict.values())))
        
        return loss_dict
    
class lat_weighted_acc_spatial_map(Metric):
    def __init__(self, vars, lat, clim, clim_timestamps, transforms=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("encoded_timestamps", default=[], dist_reduce_fx="cat")
        self.vars = vars
        self.lat = lat
        self.transforms = transforms
        self.suffix = suffix
        self.clim = clim
        self.clim_timestamps = clim_timestamps

    def update(self, preds: torch.Tensor, targets: torch.Tensor, output_timestamps: list):
        self.preds.append(preds)
        self.targets.append(targets)
        encoded_timestamps = torch.tensor([encode_timestamp(ts) for ts in output_timestamps], device=preds.device)
        self.encoded_timestamps.append(encoded_timestamps)
    
    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        encoded_timestamps = dim_zero_cat(self.encoded_timestamps)  
        decoded_timestamps = [decode_timestamp(ts.item()) for ts in encoded_timestamps]
        decoded_timestamps_no_year = [remove_year(ts) for ts in decoded_timestamps]

        if self.transforms is not None:
            preds = self.transforms(preds)
            targets = self.transforms(targets) 

        # lattitude weights
        w_lat = np.cos(np.deg2rad(self.lat))
        w_lat = w_lat / w_lat.mean()  # (H, )
        w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=targets.dtype, device=targets.device)  # (1, H, 1)

        clim_timestamp_to_index = {timestamp: idx for idx, timestamp in enumerate(self.clim_timestamps)}
        clim_subset_indices = [clim_timestamp_to_index[timestamp] for timestamp in decoded_timestamps_no_year]
        clim_subset = self.clim[clim_subset_indices]
        clim_subset = clim_subset.to(device=targets.device)
        preds = preds - clim_subset
        targets = targets - clim_subset
        spatial_map_dict = {}

        for i, var in enumerate(self.vars):
            pred_prime = preds[:, i]
            targ_prime = targets[:, i]
            acc_spatial_map = torch.sum((w_lat * pred_prime * targ_prime), dim=(0)) / torch.sqrt(
                torch.sum((w_lat * pred_prime**2), dim=(0)) * torch.sum((w_lat * targ_prime**2), dim=(0)))
            spatial_map_name = f"w_acc_spatial_{var}_{self.suffix}" if self.suffix else f"w_acc_spatial_{var}"
            spatial_map_dict[spatial_map_name] = acc_spatial_map

        spatial_map_name = f"w_acc_spatial_{self.suffix}" if self.suffix else f"w_acc"
        spatial_map_dict[spatial_map_name] = torch.mean(torch.stack(list(spatial_map_dict.values())), dim=(0))
        
        return spatial_map_dict