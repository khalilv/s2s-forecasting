# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
import torch
import numpy as np
from typing import Any, Callable
from datetime import datetime, timezone
from pytorch_lightning import LightningModule
from tqdm import tqdm
from s2s.aurora.model.aurora import Aurora, AuroraSmall, AuroraHighRes
from s2s.aurora.batch import Batch, Metadata
from s2s.aurora.rollout import rollout
from s2s.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from s2s.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_rmse,
    lat_weighted_acc_spatial_map,
    lat_weighted_rmse_spatial_map
)
from s2s.utils.data_utils import plot_spatial_map_with_basemap, split_surface_atmospheric, AURORA_NAME_TO_VAR, SURFACE_VARS, ATMOSPHERIC_VARS, STATIC_VARS
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
        self.atmos_stats = {}
        self.delta_time = None
        self.plot_variables = []
        self.flip_lat = False
        self.lat = None
        self.lon = None
        self.test_resolution_warning_printed = False
        self.val_resolution_warning_printed = False
        self.train_resolution_warning_printed = False
        self.save_hyperparameters(logger=False, ignore=["net"])      

    
    def init_metrics(self):
        assert self.lat is not None, 'Latitude values not initialized yet.'
        assert self.lon is not None, 'Longitude values not initialized yet.'
        self.train_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat)
        self.val_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, None)
        self.val_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, None)
        self.val_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, None)
        self.test_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, None)        
        self.test_lat_weighted_rmse_spatial_map = lat_weighted_rmse_spatial_map(self.out_variables, self.lat, (len(self.lat), len(self.lon)), None)
        self.test_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, None)
        self.test_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, None)
        self.test_lat_weighted_acc_spatial_map = lat_weighted_acc_spatial_map(self.out_variables, self.lat, (len(self.lat), len(self.lon)), None)

    def init_network(self):
        assert self.delta_time is not None, 'delta_time hyperparameter must be set before initializing model'
        surf_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.in_surface_variables])
        static_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.static_variables])
        atmos_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.in_atmospheric_variables])
        if self.version == 0:
            self.net = Aurora(
                surf_vars=surf_vars,
                static_vars=static_vars,
                atmos_vars=atmos_vars,
                surf_stats=self.surf_stats,
                atmos_stats=self.atmos_stats,
                delta_time=self.delta_time
            )
        elif self.version == 1:
            self.net = AuroraSmall(
                surf_vars=surf_vars,
                static_vars=static_vars,
                atmos_vars=atmos_vars,
                surf_stats=self.surf_stats,
                atmos_stats=self.atmos_stats,
                delta_time=self.delta_time

            )
        elif self.version == 2:
            self.net = AuroraHighRes(
                surf_vars=surf_vars,
                static_vars=static_vars,
                surf_stats=self.surf_stats,
                atmos_stats=self.atmos_stats,
                delta_time=self.delta_time
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
                pressure_level = v.split('_')[-1]
                assert pressure_level.isdigit(), f"Found invalid pressure level in {v}"
                if atm_var in ATMOSPHERIC_VARS:
                    self.atmos_stats[f"{AURORA_NAME_TO_VAR[atm_var]}_{pressure_level}"] = (mean[i], std[i])
                else:
                    raise ValueError(f"{v} could not be identified as a surface, static, or atmospheric variable")                
    
    def construct_aurora_batch(self, x, static, variables, static_variables, input_timestamps):
        surf_vars = {}
        atmos_vars = {}
        static_vars = {}
        atmos_data = {}
        atmos_levels = {}
        
        #split surface and atmospheric data
        for i, v in enumerate(variables):
            if v in SURFACE_VARS:
                surf_vars[AURORA_NAME_TO_VAR[v]] = torch.flip(x[:,:,i,:,:], dims=[-2]) if self.flip_lat else x[:,:,i,:,:]
            else:
                atm_var = '_'.join(v.split('_')[:-1])
                pressure_level = v.split('_')[-1]
                assert pressure_level.isdigit(), f"Found invalid pressure level in {v}"
                if atm_var in ATMOSPHERIC_VARS:
                    if AURORA_NAME_TO_VAR[atm_var] not in atmos_data:
                        atmos_data[AURORA_NAME_TO_VAR[atm_var]] = [(int(pressure_level), torch.flip(x[:,:,i,:,:], dims=[-2]) if self.flip_lat else x[:,:,i,:,:])]
                        atmos_levels[AURORA_NAME_TO_VAR[atm_var]] = [int(pressure_level)]
                    else:
                        atmos_data[AURORA_NAME_TO_VAR[atm_var]].append((int(pressure_level), torch.flip(x[:,:,i,:,:], dims=[-2]) if self.flip_lat else x[:,:,i,:,:]))
                        atmos_levels[AURORA_NAME_TO_VAR[atm_var]].append(int(pressure_level))
                else:
                    raise ValueError(f"{v} could not be identified as a surface or atmospheric variable")  
        
        #sort pressure levels for each variable 
        for v in atmos_data.keys():
            atmos_data[v].sort(key=lambda x: x[0])
            atmos_levels[v].sort()

        #assert all pressure levels are equal and ordered for all atmospheric variables
        for v1 in atmos_levels.keys():
            for v2 in atmos_levels.keys():
                assert atmos_levels[v1] == atmos_levels[v2], f"Pressure levels don't match: {v1}: {atmos_levels[v1]}, {v2}: {atmos_levels[v2]}"
        
        atmos_levels = tuple(next(iter(atmos_levels.values()))) #get pressure levels from first key
        
        #stack pressure levels for each atmospheric variable to form tensors of (B, T, C, H, W)
        for v, data in atmos_data.items():
            channel_list = [c for _, c in data]
            atmos_vars[v] = torch.stack(channel_list, dim=2)
        
        #format static variables
        for i, v in enumerate(static_variables):
            if v in STATIC_VARS:
                static_vars[AURORA_NAME_TO_VAR[v]] = torch.flip(static[0,i,:,:], dims=[-2]) if self.flip_lat else static[0,i,:,:]
 
        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=torch.from_numpy(self.lat).flip(dims=[0]) if self.flip_lat else torch.from_numpy(self.lat),
                lon=torch.from_numpy(self.lon),
                time=tuple([datetime.fromtimestamp(ts[-1].astype(int), tz=timezone.utc) for ts in input_timestamps.astype('datetime64[s]')]),
                atmos_levels=atmos_levels,
            )
        )

    def deconstruct_aurora_batch(self, batch: Batch, variables):
        preds = []
        timestamps = [np.datetime64(dt) for dt in batch.metadata.time]
        for v in variables:
            if v in SURFACE_VARS:
                surf_data = batch.surf_vars[AURORA_NAME_TO_VAR[v]].squeeze(1)
                preds.append(torch.flip(surf_data, dims=[-2]) if self.flip_lat else surf_data)
            else: 
                atm_var = '_'.join(v.split('_')[:-1])
                pressure_level = v.split('_')[-1]
                assert pressure_level.isdigit(), f"Found invalid pressure level in {v}"
                if atm_var in ATMOSPHERIC_VARS:
                    atm_data = batch.atmos_vars[AURORA_NAME_TO_VAR[atm_var]].squeeze(1)[:,batch.metadata.atmos_levels.index(int(pressure_level)),:,:]
                    preds.append(torch.flip(atm_data, dims=[-2]) if self.flip_lat else surf_data)
                else:
                    raise ValueError(f"{v} could not be identified as a surface or atmospheric variable")  
        preds = torch.stack(preds, dim=1) #(T, V, H, W)
        return preds, timestamps

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
        if self.lat[0] < self.lat[1]: #aurora expects latitudes in decreasing order
            self.flip_lat = True
            print('Found increasing latitudes. Aurora expects latitudes in decreasing order. Input data will be flipped along H axis, and output will be flipped back before computing metrics')

    def set_plot_variables(self, plot_variables: list):
        self.plot_variables = plot_variables
        print('Set variables to plot spatial maps for during evaluation: ', plot_variables)

    def set_delta_time(self, predict_step_size, hrs_each_step):
        self.delta_time = int(predict_step_size * hrs_each_step)

    def set_variables(self, in_variables: list, static_variables: list, out_variables: list):
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.static_variables = static_variables
        self.in_surface_variables, self.in_atmospheric_variables = split_surface_atmospheric(in_variables)
        self.out_surface_variables, self.out_atmospheric_variables = split_surface_atmospheric(out_variables)
    
    def training_step(self, batch: Any, batch_idx: int):
        x, static, y, _, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps = batch #spread batch data 
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")

        input_batch = self.construct_aurora_batch(x, static, variables, static_variables, input_timestamps)
        
        rollout_steps = y.shape[1]
        rollout_batches = [rollout_batch.to("cpu") for rollout_batch in rollout(self.net, input_batch, steps=rollout_steps)]
        output_batch = rollout_batches[-1].to(self.device)
        preds, pred_timestamps = self.deconstruct_aurora_batch(output_batch, out_variables)        
        
        assert (pred_timestamps == output_timestamps[:,-1]).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,-1]}'

        target = y[:, -1]

        if target.shape[-2:] != preds.shape[-2:]:
            if not self.train_resolution_warning_printed:
                print(f'Warning: Found mismatch in resolutions target: {target.shape}, prediction: {preds.shape}. Subsetting target to match preds.')
                self.train_resolution_warning_printed = True
            target = target[..., :preds.shape[-2], :preds.shape[-1]]
        
        batch_loss = self.train_lat_weighted_mse(preds, target)
        for var in batch_loss.keys():
            self.log(
                "train/" + var,
                batch_loss[var],
                prog_bar=True,
            )
        self.train_lat_weighted_mse.reset()
        return batch_loss['w_mse']
    
    def validation_step(self, batch: Any, batch_idx: int):
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps = batch
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")
        
        input_batch = self.construct_aurora_batch(x, static, variables, static_variables, input_timestamps)
        
        rollout_steps = y.shape[1]
        with torch.inference_mode():
            rollout_batches = [rollout_batch.to("cpu") for rollout_batch in rollout(self.net, input_batch, steps=rollout_steps)]
        output_batch = rollout_batches[-1].to(self.device)
        preds, pred_timestamps = self.deconstruct_aurora_batch(output_batch, out_variables)        
        
        assert (pred_timestamps == output_timestamps[:,-1]).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,-1]}'

        target = y[:, -1]
        clim = climatology[:,-1]

        if target.shape[-2:] != preds.shape[-2:]:
            if not self.val_resolution_warning_printed:
                print(f'Warning: Found mismatch in resolutions target: {target.shape}, prediction: {preds.shape}. Subsetting target to match preds.')
                self.val_resolution_warning_printed = True
            target = target[..., :preds.shape[-2], :preds.shape[-1]]
            clim = clim[..., :preds.shape[-2], :preds.shape[-1]]

        self.val_lat_weighted_mse.update(preds, target)
        self.val_lat_weighted_rmse.update(preds, target)
        
        self.val_lat_weighted_acc.update(preds, target, clim)
        
    def on_validation_epoch_end(self):
        self.val_resolution_warning_printed = False
        self.train_resolution_warning_printed = False
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
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps = batch
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")
        
        input_batch = self.construct_aurora_batch(x, static, variables, static_variables, input_timestamps)
        
        rollout_steps = y.shape[1]
        with torch.inference_mode():
            rollout_batches = [rollout_batch.to("cpu") for rollout_batch in rollout(self.net, input_batch, steps=rollout_steps)]
        output_batch = rollout_batches[-1].to(self.device)
        preds, pred_timestamps = self.deconstruct_aurora_batch(output_batch, out_variables)        
        
        assert (pred_timestamps == output_timestamps[:,-1]).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,-1]}'

        target = y[:,-1] 
        clim = climatology[:,-1] 

        if target.shape[-2:] != preds.shape[-2:]:
            if not self.test_resolution_warning_printed:
                print(f'Warning: Found mismatch in resolutions target: {target.shape}, prediction: {preds.shape}. Subsetting target to match preds.')
                self.test_resolution_warning_printed = True
            target = target[..., :preds.shape[-2], :preds.shape[-1]]
            clim = clim[..., :preds.shape[-2], :preds.shape[-1]]
                      
        self.test_lat_weighted_mse.update(preds, target)
        self.test_lat_weighted_rmse.update(preds, target)
        self.test_lat_weighted_rmse_spatial_map.update(preds, target)

        self.test_lat_weighted_acc.update(preds, target, clim)
        self.test_lat_weighted_acc_spatial_map.update(preds, target, clim)

    def on_test_epoch_end(self):
        self.test_resolution_warning_printed = False
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
        for plot_var in tqdm(self.plot_variables, desc="Plotting RMSE spatial maps"):
            for var in w_rmse_spatial_maps.keys():
                if plot_var in var:
                    plot_spatial_map_with_basemap(data=w_rmse_spatial_maps[var].float().cpu(), lat=self.lat, lon=self.lon, title=var, filename=f"{self.logger.log_dir}/test_{var}.png")
        for plot_var in tqdm(self.plot_variables, desc="Plotting ACC spatial maps"):
            for var in w_acc_spatial_maps.keys():
                if plot_var in var:
                    plot_spatial_map_with_basemap(data=w_acc_spatial_maps[var].float().cpu(), lat=self.lat, lon=self.lon, title=var, filename=f"{self.logger.log_dir}/test_{var}.png")

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
