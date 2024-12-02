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
from s2s.utils.lr_scheduler import LinearWarmupCosineAnnealingLR, LinearWarmupConstantLR
from s2s.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_rmse,
    acc_spatial_map,
    rmse_spatial_map,
    variable_weighted_mae
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
        update_statistics: bool = False,
        delta_time: int = 6,
        use_activation_checkpointing: bool = False,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        optim_lr: float = 5e-5,
        optim_beta_1: float = 0.9,
        optim_beta_2: float = 0.999,
        optim_weight_decay: float = 5e-6,
        optim_warmup_steps: int = 10000,
        optim_max_steps: int = 200000,
        optim_warmup_start_lr: float = 1e-8,
        mae_alpha: float = 0.25, 
        mae_beta: float = 1.0,
        mae_gamma: float = 2.0,
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.version = version
        self.update_statistics = update_statistics
        self.delta_time = delta_time
        self.use_activation_checkpointing = use_activation_checkpointing
        self.drop_path = drop_path
        self.drop_rate = drop_rate
        self.optim_lr = optim_lr
        self.optim_beta_1 = optim_beta_1
        self.optim_beta_2 = optim_beta_2
        self.optim_weight_decay = optim_weight_decay
        self.optim_warmup_steps = optim_warmup_steps
        self.optim_max_steps = optim_max_steps
        self.optim_warmup_start_lr = optim_warmup_start_lr
        self.mae_alpha = mae_alpha
        self.mae_beta = mae_beta
        self.mae_gamma = mae_gamma
        self.surf_stats = {}
        self.atmos_stats = {}
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
        self.train_variable_weighted_mae = variable_weighted_mae(self.out_variables, self.mae_alpha, self.mae_beta, self.mae_gamma)
        self.val_variable_weighted_mae = variable_weighted_mae(self.out_variables, self.mae_alpha, self.mae_beta, self.mae_gamma)
        self.val_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, suffix='12h')
        self.val_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, suffix='12h')
        self.test_rmse_spatial_map = rmse_spatial_map(self.out_variables, (len(self.lat), len(self.lon)))
        self.test_variable_weighted_mae = variable_weighted_mae(self.out_variables, self.mae_alpha, self.mae_beta, self.mae_gamma)
        self.test_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat)
        self.test_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat)
        self.test_acc_spatial_map = acc_spatial_map(self.out_variables, (len(self.lat), len(self.lon)))

    def init_network(self):
        surf_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.in_surface_variables])
        static_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.static_variables])
        atmos_vars=tuple([AURORA_NAME_TO_VAR[v] for v in self.in_atmospheric_variables])
        surf_stats = self.surf_stats if self.update_statistics else None
        atmos_stats = self.atmos_stats if self.update_statistics else None
        if self.version == 0:
            self.net = Aurora(
                surf_vars=surf_vars,
                static_vars=static_vars,
                atmos_vars=atmos_vars,
                surf_stats=surf_stats,
                atmos_stats=atmos_stats,
                delta_time=self.delta_time,
                drop_path=self.drop_path,
                drop_rate=self.drop_rate
            )
        elif self.version == 1:
            self.net = AuroraSmall(
                surf_vars=surf_vars,
                static_vars=static_vars,
                atmos_vars=atmos_vars,
                surf_stats=surf_stats,
                atmos_stats=atmos_stats,
                delta_time=self.delta_time,
                drop_path=self.drop_path,
                drop_rate=self.drop_rate
            )
        elif self.version == 2:
            self.net = AuroraHighRes(
                surf_vars=surf_vars,
                static_vars=static_vars,
                surf_stats=surf_stats,
                atmos_stats=atmos_stats,
                delta_time=self.delta_time,
                drop_path=self.drop_path,
                drop_rate=self.drop_rate
            )
        else:
            raise ValueError(f"Invalid version number: {self.version}. Must be 0: Aurora, 1: AuroraSmall, or 2: AuroraHighRes.")
        
        if len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)
        
        if self.use_activation_checkpointing:
            self.net.configure_activation_checkpointing()
    
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
                    preds.append(torch.flip(atm_data, dims=[-2]) if self.flip_lat else atm_data)
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
        
        rollout_steps = int(lead_times[0][-1] // self.delta_time)
        yield_steps = (lead_times[0] // self.delta_time) - 1
        batch_loss = []
        for idx, output_batch in enumerate(rollout(self.net, input_batch, steps=rollout_steps, yield_intermediate=True, yield_steps=yield_steps)):
            preds, pred_timestamps = self.deconstruct_aurora_batch(output_batch, out_variables)        
            assert (pred_timestamps == output_timestamps[:,idx]).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,idx]}'
            target = y[:, idx]
            if target.shape[-2:] != preds.shape[-2:]:
                if not self.train_resolution_warning_printed:
                    print(f'Warning: Found mismatch in resolutions target: {target.shape}, prediction: {preds.shape}. Subsetting target to match preds.')
                    self.train_resolution_warning_printed = True
                target = target[..., :preds.shape[-2], :preds.shape[-1]]
        
            loss = self.train_variable_weighted_mae(preds, target)
            batch_loss.append(loss['var_w_mae'])
            self.train_variable_weighted_mae.reset()

        average_loss = torch.mean(torch.stack(batch_loss))
        self.log(
            "train/var_w_mae",
            average_loss,
            prog_bar=True,
        )            
        return average_loss
    
    def validation_step(self, batch: Any, batch_idx: int):
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps = batch
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")
        
        input_batch = self.construct_aurora_batch(x, static, variables, static_variables, input_timestamps)
        
        rollout_steps = int(lead_times[0][-1] // self.delta_time)
        yield_steps = (lead_times[0] // self.delta_time) - 1
        for idx, output_batch in enumerate(rollout(self.net, input_batch, steps=rollout_steps, yield_intermediate=True, yield_steps=yield_steps)):
            preds, pred_timestamps = self.deconstruct_aurora_batch(output_batch, out_variables)        
            assert (pred_timestamps == output_timestamps[:,idx]).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,idx]}'
            target = y[:, idx]
            clim = climatology[:,idx]
            if target.shape[-2:] != preds.shape[-2:]:
                if not self.train_resolution_warning_printed:
                    print(f'Warning: Found mismatch in resolutions target: {target.shape}, prediction: {preds.shape}. Subsetting target to match preds.')
                    self.train_resolution_warning_printed = True
                target = target[..., :preds.shape[-2], :preds.shape[-1]]
                clim = clim[..., :preds.shape[-2], :preds.shape[-1]]

            self.val_variable_weighted_mae.update(preds, target)
        self.val_lat_weighted_rmse.update(preds, target)
        self.val_lat_weighted_acc.update(preds, target, clim)
        
    def on_validation_epoch_end(self):
        self.val_resolution_warning_printed = False
        self.train_resolution_warning_printed = False
        w_rmse = self.val_lat_weighted_rmse.compute()
        w_acc = self.val_lat_weighted_acc.compute()
        var_w_mae = self.val_variable_weighted_mae.compute()

        #scalar metrics
        loss_dict = {**var_w_mae, **w_rmse, **w_acc}
        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                prog_bar=False,
                sync_dist=True
            )
        self.val_variable_weighted_mae.reset()
        self.val_lat_weighted_rmse.reset()
        self.val_lat_weighted_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps = batch
        
        print(input_timestamps, output_timestamps)
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")
        
        input_batch = self.construct_aurora_batch(x, static, variables, static_variables, input_timestamps)
        
        rollout_steps = int(lead_times[0][-1] // self.delta_time)
        output_batch = rollout(self.net, input_batch, steps=rollout_steps, yield_intermediate=False)
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

        self.test_variable_weighted_mae.update(preds, target)              
        self.test_lat_weighted_rmse.update(preds, target)
        self.test_rmse_spatial_map.update(preds, target)

        self.test_lat_weighted_acc.update(preds, target, clim)
        self.test_acc_spatial_map.update(preds, target, clim)

    def on_test_epoch_end(self):
        var_w_mae = self.test_variable_weighted_mae.compute()
        w_rmse = self.test_lat_weighted_rmse.compute()
        w_acc = self.test_lat_weighted_acc.compute()
        rmse_spatial_maps = self.test_rmse_spatial_map.compute()
        acc_spatial_maps = self.test_acc_spatial_map.compute()

        #scalar metrics
        loss_dict = {**var_w_mae, **w_rmse, **w_acc}
        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                prog_bar=False,
                sync_dist=True
            )

        if self.global_rank == 0:
            latitudes, longitudes = self.lat.copy(), self.lon.copy()
            for plot_var in tqdm(self.plot_variables, desc="Plotting RMSE spatial maps"):
                for var in rmse_spatial_maps.keys():
                    if plot_var in var:
                        map = rmse_spatial_maps[var].float().cpu()
                        if map.shape[0] != len(latitudes) or map.shape[1] != len(longitudes):
                            print(f'Warning: Found mismatch in resolutions rmse_spatial_map for {var}: {map.shape}, latitude: {len(latitudes)}, longitude: {len(longitudes)}. Subsetting latitude and/or longitude values to match spatial_map resolution')
                            plot_spatial_map_with_basemap(data=map, lat=latitudes[:map.shape[0]], lon=longitudes[:map.shape[1]], title=var, filename=f"{self.logger.log_dir}/test_{var}.png")
                        else:
                            plot_spatial_map_with_basemap(data=map, lat=latitudes, lon=longitudes, title=var, filename=f"{self.logger.log_dir}/test_{var}.png")

            for plot_var in tqdm(self.plot_variables, desc="Plotting ACC spatial maps"):
                for var in acc_spatial_maps.keys():
                    if plot_var in var:
                        map = acc_spatial_maps[var].float().cpu()
                        if map.shape[0] != len(latitudes) or map.shape[1] != len(longitudes):
                            print(f'Warning: Found mismatch in resolutions acc_spatial_map for {var}: {map.shape}, latitude: {len(latitudes)}, longitude: {len(longitudes)}. Subsetting latitude and/or longitude values to match spatial_map resolution')
                            plot_spatial_map_with_basemap(data=map, lat=latitudes[:map.shape[0]], lon=longitudes[:map.shape[1]], title=var, filename=f"{self.logger.log_dir}/test_{var}.png")
                        else:
                            plot_spatial_map_with_basemap(data=map, lat=latitudes, lon=longitudes, title=var, filename=f"{self.logger.log_dir}/test_{var}.png")
            
        self.test_variable_weighted_mae.reset()
        self.test_lat_weighted_rmse.reset()
        self.test_lat_weighted_acc.reset()
        self.test_acc_spatial_map.reset()
        self.test_rmse_spatial_map.reset()
        self.test_resolution_warning_printed = False

    #optimizer definition - will be used to optimize the network based
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.optim_lr,
            betas=(self.optim_beta_1, self.optim_beta_2),
            weight_decay=self.optim_weight_decay
        )

        # #pretraining
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=self.optim_warmup_steps,
            max_steps=self.optim_max_steps,
            warmup_start_lr=self.optim_warmup_start_lr,
            eta_min=self.optim_lr / 10,
        )

        # #finetuning
        # lr_scheduler = LinearWarmupConstantLR(
        #     optimizer, 
        #     warmup_steps=self.optim_warmup_steps
        # )

        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
