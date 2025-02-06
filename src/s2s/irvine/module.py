# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
import torch
import numpy as np
from typing import Any
from datetime import datetime, timezone
from pytorch_lightning import LightningModule
from tqdm import tqdm
from s2s.irvine.model.network import WeatherForecast
from s2s.irvine.model.batch import Batch, Metadata
from s2s.irvine.rollout import rollout
from s2s.utils.lr_scheduler import LinearWarmupCosineAnnealingLR, LinearWarmupConstantLR
from s2s.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_rmse,
    acc_spatial_map,
    rmse_spatial_map,
    lat_weighted_mse,
)
from s2s.utils.data_utils import split_surface_atmospheric, NAME_TO_VAR, SURFACE_VARS, ATMOSPHERIC_VARS, STATIC_VARS
from s2s.utils.plot import plot_spatial_map_with_basemap

class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting. """

    def __init__(
        self,
        pretrained_path: str = "",
        temporal_attention: bool = False,
        load_strict: bool = False,
        delta_time: int = 6,
        history_size: int = 2,
        patch_size: int = 2,
        latent_levels: int = 4,
        embed_dim: int = 512,
        use_activation_checkpointing: bool = False,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        autocast: bool = False,
        optim_lr: float = 5e-5,
        optim_beta_1: float = 0.9,
        optim_beta_2: float = 0.999,
        optim_weight_decay: float = 5e-6,
        optim_warmup_steps: int = 10000,
        optim_max_steps: int = 200000,
        optim_warmup_start_lr: float = 1e-8,
        monitor_val_step: int = 1,
        monitor_test_steps: list = [1],
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.temporal_attention = temporal_attention
        self.load_strict = load_strict
        self.delta_time = delta_time
        self.history_size = history_size
        self.patch_size = patch_size
        self.latent_levels = latent_levels
        self.embed_dim = embed_dim
        self.use_activation_checkpointing = use_activation_checkpointing
        self.drop_path = drop_path
        self.drop_rate = drop_rate
        self.autocast = autocast
        self.optim_lr = optim_lr
        self.optim_beta_1 = optim_beta_1
        self.optim_beta_2 = optim_beta_2
        self.optim_weight_decay = optim_weight_decay
        self.optim_warmup_steps = optim_warmup_steps
        self.optim_max_steps = optim_max_steps
        self.optim_warmup_start_lr = optim_warmup_start_lr
        self.monitor_val_step = monitor_val_step
        self.monitor_test_steps = monitor_test_steps
        self.plot_variables = []
        self.flip_lat = False
        self.lat = None
        self.lon = None
        self.denormalization = None
        self.test_resolution_warning_printed = False
        self.val_resolution_warning_printed = False
        self.train_resolution_warning_printed = False
        assert self.monitor_val_step > 0, 'Validation step to monitor must be > 0'
        assert all(step > 0 for step in self.monitor_test_steps), 'All test steps to monitor must be > 0'
        self.save_hyperparameters(logger=False, ignore=["net"])      

    def init_metrics(self):
        assert self.lat is not None, 'Latitude values not initialized yet.'
        assert self.lon is not None, 'Longitude values not initialized yet.'
        denormalize = self.denormalization.denormalize if self.denormalization else None

        #train metrics
        self.train_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, suffix='norm')

        #validation metrics
        val_suffix = f'{int(self.monitor_val_step*self.delta_time)}hrs'
        self.val_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, suffix='norm')
        self.val_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, denormalize, suffix=val_suffix)
        self.val_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, denormalize, suffix=val_suffix)
        
        #test metrics
        self.test_lat_weighted_mse, self.test_lat_weighted_rmse, self.test_lat_weighted_acc, self.test_acc_spatial_map, self.test_rmse_spatial_map = {}, {}, {}, {}, {}
        for step in self.monitor_test_steps:
            self.test_lat_weighted_mse[step] = lat_weighted_mse(self.out_variables, self.lat) 
            self.test_lat_weighted_rmse[step] = lat_weighted_rmse(self.out_variables, self.lat, denormalize) 
            self.test_lat_weighted_acc[step] = lat_weighted_acc(self.out_variables, self.lat, denormalize) 
            self.test_acc_spatial_map[step] = acc_spatial_map(self.out_variables, (len(self.lat), len(self.lon)), denormalize) 
            self.test_rmse_spatial_map[step] = rmse_spatial_map(self.out_variables, (len(self.lat), len(self.lon)), denormalize) 

    def set_denormalization(self, denormalization):
        self.denormalization = denormalization

    def init_network(self):
        surf_vars=tuple([NAME_TO_VAR[v] for v in self.in_surface_variables])
        static_vars=tuple([NAME_TO_VAR[v] for v in self.static_variables])
        atmos_vars=tuple([NAME_TO_VAR[v] for v in self.in_atmospheric_variables])
        self.net = WeatherForecast(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            delta_time=self.delta_time,
            drop_path=self.drop_path,
            drop_rate=self.drop_rate,
            autocast=self.autocast,
            history_size=self.history_size,
            temporal_attention=self.temporal_attention,
            patch_size=self.patch_size,
            latent_levels=self.latent_levels,
            embed_dim=self.embed_dim,
        )

        if len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)
        
        if self.use_activation_checkpointing:
            self.net.configure_activation_checkpointing()
           
    def load_pretrained_weights(self, path):
        self.net.load_checkpoint_local(path, strict=self.load_strict)
    
    def setup(self, stage: str):
        self.denormalization.to(device=self.device, dtype=self.dtype)
        for step in self.monitor_test_steps:
            self.test_lat_weighted_mse[step].to(self.device)
            self.test_lat_weighted_acc[step].to(self.device)
            self.test_lat_weighted_rmse[step].to(self.device)
            self.test_rmse_spatial_map[step].to(self.device)
            self.test_acc_spatial_map[step].to(self.device)
       
    def construct_batch(self, x, static, variables, static_variables, input_timestamps):
        surf_vars = {}
        atmos_vars = {}
        static_vars = {}
        atmos_data = {}
        atmos_levels = {}
        
        #split surface and atmospheric data
        for i, v in enumerate(variables):
            if v in SURFACE_VARS:
                surf_vars[NAME_TO_VAR[v]] = torch.flip(x[:,:,i,:,:], dims=[-2]) if self.flip_lat else x[:,:,i,:,:]
            else:
                atm_var = '_'.join(v.split('_')[:-1])
                pressure_level = v.split('_')[-1]
                assert pressure_level.isdigit(), f"Found invalid pressure level in {v}"
                if atm_var in ATMOSPHERIC_VARS:
                    if NAME_TO_VAR[atm_var] not in atmos_data:
                        atmos_data[NAME_TO_VAR[atm_var]] = [(int(pressure_level), torch.flip(x[:,:,i,:,:], dims=[-2]) if self.flip_lat else x[:,:,i,:,:])]
                        atmos_levels[NAME_TO_VAR[atm_var]] = [int(pressure_level)]
                    else:
                        atmos_data[NAME_TO_VAR[atm_var]].append((int(pressure_level), torch.flip(x[:,:,i,:,:], dims=[-2]) if self.flip_lat else x[:,:,i,:,:]))
                        atmos_levels[NAME_TO_VAR[atm_var]].append(int(pressure_level))
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
                static_vars[NAME_TO_VAR[v]] = torch.flip(static[0,i,:,:], dims=[-2]) if self.flip_lat else static[0,i,:,:]

        batch_timestamps = tuple(
            tuple(datetime.fromtimestamp(t.astype(int), tz=timezone.utc) for t in ts)
            for ts in input_timestamps.astype('datetime64[s]')
        )
                                       
        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=torch.from_numpy(self.lat).flip(dims=[0]) if self.flip_lat else torch.from_numpy(self.lat),
                lon=torch.from_numpy(self.lon),
                time=batch_timestamps,
                atmos_levels=atmos_levels,
            )
        )

    def deconstruct_batch(self, batch: Batch, variables, preserve_history = False):
        preds = []
        timestamps = np.array([np.datetime64(dt[-1]) for dt in batch.metadata.time], dtype='datetime64[ns]')
        for v in variables:
            if v in SURFACE_VARS:
                if preserve_history:
                    surf_data = batch.surf_vars[NAME_TO_VAR[v]]
                else:
                    surf_data = batch.surf_vars[NAME_TO_VAR[v]].squeeze(1)
                preds.append(torch.flip(surf_data, dims=[-2]) if self.flip_lat else surf_data)
            else: 
                atm_var = '_'.join(v.split('_')[:-1])
                pressure_level = v.split('_')[-1]
                assert pressure_level.isdigit(), f"Found invalid pressure level in {v}"
                if atm_var in ATMOSPHERIC_VARS:
                    if preserve_history:
                        atm_data = batch.atmos_vars[NAME_TO_VAR[atm_var]][:,:,batch.metadata.atmos_levels.index(int(pressure_level)),:,:]
                    else:
                        atm_data = batch.atmos_vars[NAME_TO_VAR[atm_var]].squeeze(1)[:,batch.metadata.atmos_levels.index(int(pressure_level)),:,:]
                    preds.append(torch.flip(atm_data, dims=[-2]) if self.flip_lat else atm_data)
                else:
                    raise ValueError(f"{v} could not be identified as a surface or atmospheric variable")  
        if preserve_history:
            preds = torch.stack(preds, dim=2) #(B, T, V, H, W)
        else:
            preds = torch.stack(preds, dim=1) #(B, V, H, W)
        return preds, timestamps

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
        if self.lat[0] < self.lat[1]: #model expects latitudes in decreasing order
            self.flip_lat = True
            print('Found increasing latitudes. Model expects latitudes in decreasing order. Input data will be flipped along H axis, and output will be flipped back before computing metrics')

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
        x, static, y, _, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch

        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.") 
        
        input_batch = self.construct_batch(x, static, variables, static_variables, input_timestamps)
    
        rollout_steps = int(lead_times[0][-1] // self.delta_time)
        yield_steps = (lead_times[0] // self.delta_time) - 1
        total_loss = 0

        for idx, output_batch in enumerate(rollout(self.net, input_batch, steps=rollout_steps, yield_steps=yield_steps)):
            preds, pred_timestamps = self.deconstruct_batch(output_batch, out_variables)        
            assert (pred_timestamps == output_timestamps[:,idx]).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,idx]}'
            target = y[:, idx]
            if target.shape[-2:] != preds.shape[-2:]:
                if not self.train_resolution_warning_printed:
                    print(f'Warning: Found mismatch in resolutions target: {target.shape}, prediction: {preds.shape}. Subsetting target to match preds.')
                    self.train_resolution_warning_printed = True
                target = target[..., :preds.shape[-2], :preds.shape[-1]]
        
            loss = self.train_lat_weighted_mse(preds, target)
            total_loss += loss['w_mse_norm']
            self.train_lat_weighted_mse.reset()

        average_loss = total_loss / rollout_steps
        self.log(
            "train/w_mse_norm",
            average_loss,
            prog_bar=True,
        )

        return average_loss
    
    def validation_step(self, batch: Any, batch_idx: int):
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch

        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")
        
        assert self.monitor_val_step <= y.shape[1], f'Unable to monitor predictions at step {self.monitor_val_step} with a prediction size of {y.shape[1]}'
        
        input_batch = self.construct_batch(x, static, variables, static_variables, input_timestamps)
        monitor_val_step_idx = self.monitor_val_step - 1
        target = y[:, monitor_val_step_idx]
        clim = climatology[:, monitor_val_step_idx]
        output_timestamps = output_timestamps[:, monitor_val_step_idx]

        lead_times_subset = lead_times[0][:self.monitor_val_step]
        rollout_steps = int(lead_times_subset[-1] // self.delta_time)
        yield_steps = (lead_times_subset // self.delta_time) - 1
        rollout_batches = [rollout_batch for rollout_batch in rollout(self.net, input_batch, steps=rollout_steps, yield_steps=yield_steps[-1:])]
        output_batch = rollout_batches[-1]
        
        preds, pred_timestamps = self.deconstruct_batch(output_batch, out_variables)
        assert (pred_timestamps == output_timestamps).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps}'

        if target.shape[-2:] != preds.shape[-2:]:
            if not self.train_resolution_warning_printed:
                print(f'Warning: Found mismatch in resolutions target: {target.shape}, prediction: {preds.shape}. Subsetting target to match preds.')
                self.train_resolution_warning_printed = True
            target = target[..., :preds.shape[-2], :preds.shape[-1]]
            clim = clim[..., :preds.shape[-2], :preds.shape[-1]]

        self.val_lat_weighted_mse.update(preds, target)
        self.val_lat_weighted_rmse.update(preds, target)
        self.val_lat_weighted_acc.update(preds, target, clim)
        
    def on_validation_epoch_end(self):
        self.val_resolution_warning_printed = False
        self.train_resolution_warning_printed = False
        w_rmse = self.val_lat_weighted_rmse.compute()
        w_acc = self.val_lat_weighted_acc.compute()
        w_mse_norm = self.val_lat_weighted_mse.compute()

        #scalar metrics
        loss_dict = {**w_mse_norm, **w_rmse, **w_acc}
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
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")
        
        input_batch = self.construct_batch(x, static, variables, static_variables, input_timestamps)
        
        rollout_steps = int(lead_times[0][-1] // self.delta_time)
        yield_steps = [step - 1 for step in self.monitor_test_steps]
        assert max(yield_steps) < rollout_steps, f'Invalid test steps {self.monitor_test_steps} to monitor for {rollout_steps} rollout steps'
        
        yield_step_idx = 0
        for rollout_batch in rollout(self.net, input_batch, steps=rollout_steps, yield_steps=yield_steps):
            step = yield_steps[yield_step_idx]
            preds, pred_timestamps = self.deconstruct_batch(rollout_batch, out_variables)        
            
            assert (pred_timestamps == output_timestamps[:,step]).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,step]}'

            target = y[:,step] 
            clim = climatology[:,step] 

            if target.shape[-2:] != preds.shape[-2:]:
                if not self.test_resolution_warning_printed:
                    print(f'Warning: Found mismatch in resolutions target: {target.shape}, prediction: {preds.shape}. Subsetting target to match preds.')
                    self.test_resolution_warning_printed = True
                target = target[..., :preds.shape[-2], :preds.shape[-1]]
                clim = clim[..., :preds.shape[-2], :preds.shape[-1]]

            self.test_lat_weighted_mse[step + 1].update(preds, target)              
            self.test_lat_weighted_rmse[step + 1].update(preds, target)
            self.test_rmse_spatial_map[step + 1].update(preds, target)

            self.test_lat_weighted_acc[step + 1].update(preds, target, clim)
            self.test_acc_spatial_map[step + 1].update(preds, target, clim)
            yield_step_idx += 1

    def on_test_epoch_end(self):
        results_dict = {'lead_time_hrs': []}
        for step in self.monitor_test_steps:
            lead_time = int(step*self.delta_time) #current forecast lead time in hours
            results_dict['lead_time_hrs'].append(lead_time)
            suffix = f'{lead_time}hrs'

            w_mse_norm = self.test_lat_weighted_mse[step].compute()
            w_rmse = self.test_lat_weighted_rmse[step].compute()
            w_acc = self.test_lat_weighted_acc[step].compute()
            rmse_spatial_maps = self.test_rmse_spatial_map[step].compute()
            acc_spatial_maps = self.test_acc_spatial_map[step].compute()
            
            #scalar metrics
            loss_dict = {**w_mse_norm, **w_rmse, **w_acc}
            for var in loss_dict.keys():
                if var not in results_dict:
                    results_dict[var] = []  
                results_dict[var].append(loss_dict[var].item())

                self.log(
                    f'test/{var}_{suffix}',
                    loss_dict[var],
                    prog_bar=False,
                    sync_dist=True
                )
            
            maps_dict = {**rmse_spatial_maps, **acc_spatial_maps}
            for var in maps_dict.keys():
                if var not in results_dict:
                    results_dict[var] = []  
                results_dict[var].append(maps_dict[var].float().cpu().numpy())
            
            if self.global_rank == 0:
                latitudes, longitudes = self.lat.copy(), self.lon.copy()
                for plot_var in tqdm(self.plot_variables, desc="Plotting spatial maps"):
                    for var in maps_dict.keys():
                        if plot_var in var:
                            map = maps_dict[var].float().cpu()
                            if map.shape[0] != len(latitudes) or map.shape[1] != len(longitudes):
                                print(f'Warning: Found mismatch in spatial map resolutions for {var}: {map.shape}, latitude: {len(latitudes)}, longitude: {len(longitudes)}. Subsetting latitude and/or longitude values to match spatial map resolution')
                                plot_spatial_map_with_basemap(data=map, lat=latitudes[:map.shape[0]], lon=longitudes[:map.shape[1]], title=f'{var}_{suffix}', filename=f"{self.logger.log_dir}/test_{var}_{suffix}.png")
                            else:
                                plot_spatial_map_with_basemap(data=map, lat=latitudes, lon=longitudes, title=f'{var}_{suffix}', filename=f"{self.logger.log_dir}/test_{var}_{suffix}.png")                

            self.test_lat_weighted_mse[step].reset()
            self.test_lat_weighted_rmse[step].reset()
            self.test_lat_weighted_acc[step].reset()
            self.test_acc_spatial_map[step].reset()
            self.test_rmse_spatial_map[step].reset()
        
        np.savez(f'{self.logger.log_dir}/results.npz', **results_dict)
        self.test_resolution_warning_printed = False

    #optimizer definition - will be used to optimize the network based
    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params=trainable_params,
            lr=self.optim_lr,
            betas=(self.optim_beta_1, self.optim_beta_2),
            weight_decay=self.optim_weight_decay
        )

        #learning rate with decay and warmup
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=self.optim_warmup_steps,
            max_steps=self.optim_max_steps,
            warmup_start_lr=self.optim_warmup_start_lr,
            eta_min=self.optim_lr / 10,
        )

        #constant learning rate with warmup
        # lr_scheduler = LinearWarmupConstantLR(
        #     optimizer, 
        #     warmup_steps=self.optim_warmup_steps
        # )

        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
