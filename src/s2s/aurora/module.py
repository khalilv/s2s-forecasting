# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any
import numpy as np
import torch
from pytorch_lightning import LightningModule

from s2s.aurora.model.aurora import Aurora, AuroraSmall, AuroraHighRes
from s2s.aurora.batch import Batch
from s2s.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from s2s.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_rmse,
    lat_weighted_acc_spatial_map,
    lat_weighted_rmse_spatial_map
)
from s2s.utils.data_utils import plot_spatial_map, split_surface_atmospheric, AURORA_NAME_TO_VAR, SURFACE_VARS, ATMOSPHERIC_VARS, STATIC_VARS
#3) Global forecast module - abstraction for training/validation/testing steps. setup for the module including hyperparameters is included here

class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting with the Aurora model.

    Args:
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        pretrained_path: str = "",
        version: int = 0,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        img_size: list = [32, 64],
        patch_size: int = 16,
        embed_dim: int = 64,
        depth: int = 8,
        decoder_depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        parallel_patch_embed: bool = False
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.version = version
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.drop_rate = drop_rate
        self.parallel_patch_embed = parallel_patch_embed
        self.surf_stats = {}
        self.save_hyperparameters(logger=False, ignore=["net"])      

    
    def init_metrics(self):
        self.train_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat)
        self.val_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, None)
        self.val_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, None)
        self.val_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, self.val_clim, self.val_clim_timestamps, None)
        self.test_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, None)        
        self.test_lat_weighted_rmse_spatial_map = lat_weighted_rmse_spatial_map(self.out_variables, self.lat, None)
        self.test_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, None)
        self.test_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, self.test_clim, self.test_clim_timestamps, None)
        self.test_lat_weighted_acc_spatial_map = lat_weighted_acc_spatial_map(self.out_variables, self.lat, self.test_clim, self.test_clim_timestamps, None)

    def init_network(self):
        surf_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.in_surface_variables])
        static_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.static_variables])
        atmos_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.in_atmospheric_variables])
        if self.version == 0:
            self.net = Aurora(
                surf_vars=surf_vars,
                static_vars=static_vars,
                atmos_vars=atmos_vars,
                surf_stats=self.surf_stats
            )
        elif self.version == 1:
            self.net = AuroraSmall(
                surf_vars=surf_vars,
                static_vars=static_vars,
                atmos_vars=atmos_vars,
                surf_stats=self.surf_stats
            )
        elif self.version == 2:
            self.net = AuroraHighRes(
                surf_vars=surf_vars,
                static_vars=static_vars,
                surf_stats=self.surf_stats
            )
        else:
            raise ValueError(f"Invalid version number: {self.version}. Must be 0: Aurora, 1: AuroraSmall, or 2: AuroraHighRes.")
        
        if len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)
    
    def load_pretrained_weights(self, path):
        self.net.load_checkpoint_local(path)

    def update_normalization_stats(self, variables, mean, std):
        assert len(variables) == len(mean) and len(mean) == len(std)
        for i,v in enumerate(variables):
            if v in SURFACE_VARS:
                self.surf_stats[AURORA_NAME_TO_VAR[v]] = (mean[i], std[i])
            elif v in STATIC_VARS:
                self.surf_stats[AURORA_NAME_TO_VAR[v]] = (mean[i], std[i])
            else:
                atm_var = '_'.join(v.split('_')[:-1])
                pressure_level = v[-3:] if len(v.split('_')[-1]) == 3 else v[-2:]
                assert pressure_level.isdigit(), f"Found invalid pressure level in {v}"
                if atm_var in ATMOSPHERIC_VARS:
                    self.surf_stats[f"{AURORA_NAME_TO_VAR[atm_var]}_{pressure_level}"] = (mean[i], std[i])
                else:
                    raise ValueError(f"{v} could not be identified as a surface, static, or atmospheric variable")                
    
    def create_aurora_batch(self, x, static, variables, static_variables, input_timestamps):
        surf_vars = {}
        atmos_vars = {}

        atmos_data_per_var = {}
        atmos_levels_per_var = {}

        for i, v in enumerate(variables):
            if v in SURFACE_VARS:
                surf_vars[AURORA_NAME_TO_VAR[v]] = x[:,:,i,:,:]
            else:
                atm_var = '_'.join(v.split('_')[:-1])
                pressure_level = v[-3:] if len(v.split('_')[-1]) == 3 else v[-2:]
                assert pressure_level.isdigit(), f"Found invalid pressure level in {v}"
                if atm_var in ATMOSPHERIC_VARS:
                    if AURORA_NAME_TO_VAR[atm_var] not in atmos_data_per_var:
                        atmos_data_per_var[AURORA_NAME_TO_VAR[atm_var]] = [(int(pressure_level), x[:,:,i,:,:])]
                        atmos_levels_per_var[AURORA_NAME_TO_VAR[atm_var]] = [int(pressure_level)]
                    else:
                        atmos_data_per_var[AURORA_NAME_TO_VAR[atm_var]].append((int(pressure_level), x[:,:,i,:,:]))
                        atmos_levels_per_var[AURORA_NAME_TO_VAR[atm_var]].append(int(pressure_level))
                else:
                    raise ValueError(f"{v} could not be identified as a surface or atmospheric variable")  
        
        #sort pressure levels for each variable 
        for v in atmos_data_per_var.keys():
            atmos_data_per_var[v].sort(key=lambda x: x[0])
            atmos_levels_per_var[v].sort()

        #assert all pressure levels are equal and ordered for all atmospheric variables
        for v1 in atmos_levels_per_var.keys():
            for v2 in atmos_levels_per_var.keys():
                assert atmos_levels_per_var[v1] == atmos_levels_per_var[v2], f"Pressure levels don't match: {v1}: {atmos_levels_per_var[v1]}, {v2}: {atmos_levels_per_var[v2]}"
        
        atmos_levels = tuple(next(iter(atmos_levels_per_var.values()))) #get pressure levels from first key
        
        #stack pressure levels for each variable to form tensors of (B, T, C, H, W)
        for v, data in atmos_data_per_var.items():
            channel_list = [c for _, c in data]
            atmos_vars[v] = torch.stack(channel_list, dim=2)
        
        print('here')


    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_variables(self, in_variables: list, static_variables: list, out_variables: list):
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.static_variables = static_variables
        self.in_surface_variables, self.in_atmospheric_variables = split_surface_atmospheric(in_variables)
        self.out_surface_variables, self.out_atmospheric_variables = split_surface_atmospheric(out_variables)

    def set_val_clim(self, clim, timestamps):
        self.val_clim = clim
        self.val_clim_timestamps = timestamps

    def set_test_clim(self, clim, timestamps):
        self.test_clim = clim
        self.test_clim_timestamps = timestamps
    
    def training_step(self, batch: Any, batch_idx: int):
        x, static, y, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps = batch #spread batch data 
        
        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)

        #prepend static variables to input variables
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        inputs = torch.cat((static, x), dim=2).to(torch.float32)

        in_variables = static_variables + ["lattitude"] + variables

        if inputs.shape[1] > 1:
            raise NotImplementedError("history_range > 1 is not supported yet.")
        inputs = inputs.squeeze() #squeeze history dimension
   
        preds = self.net.forward(inputs, lead_times, in_variables, out_variables)
        
        #cast to float
        preds = preds.float()
        y = y.float()
        
        batch_loss = self.train_lat_weighted_mse(preds, y)
        for var in batch_loss.keys():
            self.log(
                "train/" + var,
                batch_loss[var],
                prog_bar=True,
            )
        self.train_lat_weighted_mse.reset()
        return batch_loss['w_mse']
    
    def validation_step(self, batch: Any, batch_idx: int):
        x, static, y, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps = batch
        
        aurora_batch = self.create_aurora_batch(x, static, variables, static_variables, input_timestamps)
        
        preds = self.net.forward(batch)
        
        #cast to float
        preds = preds.float()
        y = y.float()

        self.val_lat_weighted_mse.update(preds, y)
        self.val_lat_weighted_rmse.update(preds, y)
        self.val_lat_weighted_acc.update(preds, y, output_timestamps)
        
    def on_validation_epoch_end(self):
        w_mse = self.val_lat_weighted_mse.compute()
        w_rmse = self.val_lat_weighted_rmse.compute()
        w_acc = self.val_lat_weighted_acc.compute()
       
        #scalar metrics
        loss_dict = {**w_mse, **w_rmse, **w_acc}
        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                prog_bar=False,
                sync_dist=True
            )
        self.val_lat_weighted_mse.reset()
        self.val_lat_weighted_rmse.reset()
        self.val_lat_weighted_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        x, static, y, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps = batch

        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)

        #prepend static variables to input variables
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        inputs = torch.cat((static, x), dim=2).to(torch.float32)

        in_variables = static_variables + ["lattitude"] + variables

        if inputs.shape[1] > 1:
            raise NotImplementedError("history_range > 1 is not supported yet.")
        inputs = inputs.squeeze() #squeeze history dimension

        preds = self.net.forward(inputs, lead_times, in_variables, out_variables)

        #cast to float
        preds = preds.float()
        y = y.float()
        
        self.test_lat_weighted_mse.update(preds, y)
        self.test_lat_weighted_rmse.update(preds, y)
        self.test_lat_weighted_rmse_spatial_map.update(preds, y)
        self.test_lat_weighted_acc.update(preds, y, output_timestamps)
        self.test_lat_weighted_acc_spatial_map.update(preds, y, output_timestamps)

    def on_test_epoch_end(self):
        w_mse = self.test_lat_weighted_mse.compute()
        w_rmse = self.test_lat_weighted_rmse.compute()
        w_acc = self.test_lat_weighted_acc.compute()
        w_rmse_spatial_maps = self.test_lat_weighted_rmse_spatial_map.compute()
        w_acc_spatial_maps = self.test_lat_weighted_acc_spatial_map.compute()

        #scalar metrics
        loss_dict = {**w_mse, **w_rmse, **w_acc}
        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                prog_bar=False,
                sync_dist=True
            )

        #spatial maps
        for var in w_rmse_spatial_maps.keys():
            plot_spatial_map(np.flipud(w_rmse_spatial_maps[var].cpu().numpy()), title=var, filename=f"{self.logger.log_dir}/test_{var}.png")
        for var in w_acc_spatial_maps.keys():
            plot_spatial_map(np.flipud(w_acc_spatial_maps[var].cpu().numpy()), title=var, filename=f"{self.logger.log_dir}/test_{var}.png")

        self.test_lat_weighted_mse.reset()
        self.test_lat_weighted_rmse.reset()
        self.test_lat_weighted_acc.reset()
        self.test_lat_weighted_acc_spatial_map.reset()
        self.test_lat_weighted_rmse_spatial_map.reset()

    #optimizer definition - will be used to optimize the network based
    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.lr,
                    "betas": (self.beta_1, self.beta_2),
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.lr,
                    "betas": (self.beta_1, self.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.warmup_epochs,
            self.max_epochs,
            self.warmup_start_lr,
            self.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
