# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms
from tqdm import tqdm
from s2s.climaX.arch import ClimaX
from s2s.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from s2s.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_rmse,
    acc_spatial_map,
    rmse_spatial_map
)
from s2s.climaX.pos_embed import interpolate_pos_embed
from s2s.utils.plot import plot_spatial_map_with_basemap
from s2s.utils.data_utils import WEIGHTS_DICT

class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting with the ClimaX model.

    Args:
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_steps (int, optional): Number of warmup steps.
        max_steps (int, optional): Number of total steps.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        pretrained_path: str = "",
        delta_time: int = 6,
        temporal_attention: bool = False,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_steps: int = 10000,
        max_steps: int = 200000,
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
        monitor_val_step: int = None,
        monitor_test_steps: list = [1],
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.delta_time = delta_time
        self.temporal_attention = temporal_attention
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
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
        self.monitor_val_step = monitor_val_step
        self.monitor_test_steps = monitor_test_steps
        self.denormalization = None
        self.lat = None
        self.lon = None
        self.plot_variables = []
        if self.monitor_val_step:
            assert self.monitor_val_step > 0, 'Validation step to monitor must be > 0'
        else:
            print('Info: Validation step to monitor not provided. Will monitor the average across all validation steps')
        assert all(step > 0 for step in self.monitor_test_steps), 'All test steps to monitor must be > 0'
        self.save_hyperparameters(logger=False, ignore=["net"])      

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location=torch.device("cpu"), weights_only=True)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"), weights_only=True)
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]

        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys():
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
            elif checkpoint_model[k].shape != state_dict[k].shape:
                if 'token_embeds' in k and all([state_dict[k].shape[i] == checkpoint_model[k].shape[i] for i in [0, 2, 3]]) and state_dict[k].shape[1] > checkpoint_model[k].shape[1]:
                    print(f'Adapting initial history weights for {k}')
                    w = torch.zeros(state_dict[k].shape)
                    w[:, -checkpoint_model[k].shape[1]:] = checkpoint_model[k]
                    checkpoint_model[k] = w
                else:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    
    def init_metrics(self):
        assert self.lat is not None, 'Latitude values not initialized yet.'
        assert self.lon is not None, 'Longitude values not initialized yet.'
        denormalize = self.denormalization.denormalize if self.denormalization else None

        var_weights = []
        for var in self.out_variables:
            print(f'Info: setting weight for {var} to {WEIGHTS_DICT[var]}')
            var_weights.append(WEIGHTS_DICT[var])

        self.train_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, var_weights, suffix='norm')
        
        if self.monitor_val_step:
            val_suffix = f'{int(self.delta_time*self.monitor_val_step)}hrs'
        else:
            val_suffix = 'average'
        self.val_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, var_weights, suffix=f'norm_{val_suffix}')
        self.val_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, denormalize, suffix=val_suffix)
        self.val_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, denormalize, suffix=val_suffix)
        
        self.test_lat_weighted_mse, self.test_lat_weighted_rmse, self.test_lat_weighted_acc, self.test_acc_spatial_map, self.test_rmse_spatial_map = {}, {}, {}, {}, {}
        for step in self.monitor_test_steps:
            self.test_lat_weighted_mse[step] = lat_weighted_mse(self.out_variables, self.lat, var_weights, suffix='norm')        
            self.test_rmse_spatial_map[step] = rmse_spatial_map(self.out_variables, (len(self.lat), len(self.lon)), denormalize)
            self.test_lat_weighted_rmse[step] = lat_weighted_rmse(self.out_variables, self.lat, denormalize)
            self.test_lat_weighted_acc[step] = lat_weighted_acc(self.out_variables, self.lat, denormalize)
            self.test_acc_spatial_map[step] = acc_spatial_map(self.out_variables, (len(self.lat), len(self.lon)), denormalize)

    def init_network(self):
        variables = self.static_variables + ["latitude"] + self.in_variables #climaX includes 2d latitude as an input field
        self.net = ClimaX(in_vars=variables, 
                          img_size=self.img_size, 
                          patch_size=self.patch_size, 
                          embed_dim=self.embed_dim, 
                          depth=self.depth, 
                          decoder_depth=self.decoder_depth, 
                          num_heads=self.num_heads, 
                          mlp_ratio=self.mlp_ratio, 
                          drop_path=self.drop_path, 
                          drop_rate=self.drop_rate, 
                          history_size=self.history_size,
                          history_step=self.history_step,
                          temporal_attention=self.temporal_attention)
        if len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)

    def set_denormalization(self, denormalization):
        self.denormalization = denormalization
    
    def set_history_size_and_step(self, history_size, history_step):
        self.history_size = history_size + 1 # +1 because the model considers the current timestep as history
        self.history_step = history_step
      
    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def setup(self, stage: str):
        self.denormalization.to(device=self.device, dtype=self.dtype)
        for step in self.monitor_test_steps:
            self.test_lat_weighted_mse[step].to(self.device)
            self.test_lat_weighted_acc[step].to(self.device)
            self.test_lat_weighted_rmse[step].to(self.device)
            self.test_rmse_spatial_map[step].to(self.device)
            self.test_acc_spatial_map[step].to(self.device)
       
    def set_lat2d(self, normalize: bool):
        self.lat2d = torch.from_numpy(np.tile(self.lat, (self.img_size[1], 1)).T).unsqueeze(0) #climaX includes 2d latitude as an input field
        if normalize:
            normalization = transforms.Normalize(self.lat2d.mean(), self.lat2d.std())
            self.lat2d = normalization(self.lat2d) #normalized lat2d with shape [1, H, W]

    def set_variables(self, in_variables: list, static_variables: list, out_variables: list):
        self.in_variables = in_variables
        self.static_variables = static_variables
        self.out_variables = out_variables

    def set_plot_variables(self, plot_variables: list):
        self.plot_variables = plot_variables
        print('Set variables to plot spatial maps for during evaluation: ', plot_variables)

    def training_step(self, batch: Any, batch_idx: int):
        x, static, y, _, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch #spread batch data 

        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")
        
        assert self.delta_time == lead_times[0][0], f'Found mismatch between configured delta time {self.delta_time} and input lead time {lead_times[0][0]}'
        
        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        current_timestamps = input_timestamps[:, -1]
        in_variables = static_variables + ["latitude"] + variables
        delta_times = torch.zeros(x.shape[0]) + self.delta_time 
        dtype = x.dtype

        rollout_steps = int(lead_times[0][-1] // self.delta_time)
        batch_loss = {}
        for step in range(rollout_steps):
            #prepend static variables to input variables
            inputs = torch.cat((static, x), dim=2).to(dtype)

            dts = delta_times / 100 #divide deltas_times by 100 following climaX 
            preds = self.net.forward(inputs, dts.to(self.device), in_variables, out_variables)

            pred_timestamps = current_timestamps + delta_times.numpy().astype('timedelta64[h]')
            assert (output_timestamps[:, step] == pred_timestamps).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,step]}'

            #set y and preds to float32 for metric calculations
            preds = preds.float()
            targets = y[:, step].float().squeeze(1)
            
            step_loss = self.train_lat_weighted_mse(preds, targets)
            for var in step_loss.keys():
                if var in batch_loss:
                    batch_loss[var] += step_loss[var]
                else:
                    batch_loss[var] = step_loss[var]
            
            self.train_lat_weighted_mse.reset()
            x = torch.cat([x[:,1:], preds.unsqueeze(1)], axis=1)
            current_timestamps = pred_timestamps
       
        for var in batch_loss.keys():
            self.log(
                "train/" + var,
                batch_loss[var] / rollout_steps,
                prog_bar=False,
            )
        return batch_loss['w_mse_norm'] / rollout_steps
    
    def validation_step(self, batch: Any, batch_idx: int):
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times not implemented yet.")

        assert self.delta_time == lead_times[0][0], f'Found mismatch between configured delta time {self.delta_time} and input lead time {lead_times[0][0]}'

        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        current_timestamps = input_timestamps[:, -1]
        in_variables = static_variables + ["latitude"] + variables
        delta_times = torch.zeros(x.shape[0]) + self.delta_time 
        dtype = x.dtype

        rollout_steps = int(lead_times[0][-1] // self.delta_time)
        if self.monitor_val_step:
            assert self.monitor_val_step - 1 <= y.shape[1], f'Invalid test steps {self.monitor_test_steps} to monitor for {y.shape[1]} rollout steps'

        for step in range(rollout_steps):
            #prepend static variables to input variables
            inputs = torch.cat((static, x), dim=2).to(dtype)

            dts = delta_times / 100 #divide deltas_times by 100 following climaX 
            preds = self.net.forward(inputs, dts.to(self.device), in_variables, out_variables)

            pred_timestamps = current_timestamps + delta_times.numpy().astype('timedelta64[h]')
            
            if not self.monitor_val_step or step + 1 == self.monitor_val_step:
                assert (output_timestamps[:, step] == pred_timestamps).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,step]}'

                #set y and preds to float32 for metric calculations
                preds = preds.float()
                targets = y[:, step].float().squeeze(1)
                clim = climatology[:, step].float().squeeze(1)
                
                self.val_lat_weighted_mse.update(preds, targets)
                self.val_lat_weighted_rmse.update(preds, targets)

                self.val_lat_weighted_acc.update(preds, targets, clim)

            x = torch.cat([x[:,1:], preds.unsqueeze(1)], axis=1)
            current_timestamps = pred_timestamps
        
    def on_validation_epoch_end(self):
        w_mse_norm = self.val_lat_weighted_mse.compute()
        w_rmse = self.val_lat_weighted_rmse.compute()
        w_acc = self.val_lat_weighted_acc.compute()
       
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

        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1)
        current_timestamps = input_timestamps[:, -1]
        in_variables = static_variables + ["latitude"] + variables
        delta_times = torch.zeros(x.shape[0]) + self.delta_time 
        dtype = x.dtype
        
        rollout_steps = int(lead_times[0][-1] // self.delta_time)
        assert max(self.monitor_test_steps) - 1 <= y.shape[1], f'Invalid test steps {self.monitor_test_steps} to monitor for {y.shape[1]} rollout steps'
        for step in range(rollout_steps):
            #prepend static variables to input variables
            inputs = torch.cat((static, x), dim=2).to(dtype)

            dts = delta_times / 100 #divide deltas_times by 100 following climaX 
            preds = self.net.forward(inputs, dts.to(self.device), in_variables, out_variables)

            pred_timestamps = current_timestamps + delta_times.numpy().astype('timedelta64[h]')

            if step + 1 in self.monitor_test_steps:
                assert (output_timestamps[:, step] == pred_timestamps).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,step]}'
                
                #set y and preds to float32 for metric calculations
                preds = preds.float()
                targets = y[:, step].float().squeeze(1)
                clim = climatology[:, step].float().squeeze(1)
                
                self.test_lat_weighted_mse[step + 1].update(preds, targets)
                self.test_lat_weighted_rmse[step + 1].update(preds, targets)
                self.test_rmse_spatial_map[step + 1].update(preds, targets)
                self.test_lat_weighted_acc[step + 1].update(preds, targets, clim)
                self.test_acc_spatial_map[step + 1].update(preds, targets, clim)
            
            x = torch.cat([x[:,1:], preds.unsqueeze(1)], axis=1)
            current_timestamps = pred_timestamps

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
            self.warmup_steps,
            self.max_steps,
            self.warmup_start_lr,
            self.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
