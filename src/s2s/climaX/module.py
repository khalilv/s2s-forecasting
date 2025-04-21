# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any
import random
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
    rmse_spatial_map,
    aggregate_attn_weights,
)
from s2s.climaX.pos_embed import interpolate_pos_embed
from s2s.utils.plot import plot_spatial_map_with_basemap
from s2s.utils.data_utils import WEIGHTS_DICT
from s2s.utils.trajectory import random_trajectories, build_trajectories, homogeneous_trajectories, load_trajectories_from_file

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
        randomize_lead_time: bool = False,
        rollout_steps: int = 1,
        lead_time_candidates: list = [],
        ensemble_inference: bool = False,
        ensemble_size: int = 32,
        trajecory_path: str = '',
        monitor_val_step: int = None,
        monitor_test_steps: list = [1],
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
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
        self.randomize_lead_time = randomize_lead_time
        self.lead_time_candidates = lead_time_candidates
        self.rollout_steps = rollout_steps
        self.ensemble_inference = ensemble_inference
        self.ensemble_size = ensemble_size
        self.trajectory_path = trajecory_path
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
            elif 'var_query' in k:
                checkpoint_model[k] = checkpoint_model[k].expand(state_dict[k].shape)
            elif checkpoint_model[k].shape != state_dict[k].shape:
                if 'token_embeds' in k and all([state_dict[k].shape[i] == checkpoint_model[k].shape[i] for i in [0, 2, 3]]):
                    w = torch.zeros(state_dict[k].shape)
                    if state_dict[k].shape[1] > checkpoint_model[k].shape[1]:
                        w[:, -checkpoint_model[k].shape[1]:] = checkpoint_model[k]
                    else:
                        w = checkpoint_model[k][:, -state_dict[k].shape[1]:]
                    print(f'Adapting initial history weights for {k}')
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
        
        val_suffix = f'rollout_step_{self.monitor_val_step}' if self.monitor_val_step else 'average'
        self.val_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, var_weights, suffix=f'norm_{val_suffix}')
        self.val_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, denormalize, suffix=val_suffix)
        self.val_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, denormalize, suffix=val_suffix)
        
        self.test_lat_weighted_mse, self.test_lat_weighted_rmse, self.test_lat_weighted_acc, self.test_acc_spatial_map, self.test_rmse_spatial_map, self.test_time_agg_weights, self.test_var_agg_weights = {}, {}, {}, {}, {}, {}, {}
        for step in self.monitor_test_steps:
            self.test_lat_weighted_mse[step] = lat_weighted_mse(self.out_variables, self.lat, var_weights, suffix='norm')        
            self.test_rmse_spatial_map[step] = rmse_spatial_map(self.out_variables, (len(self.lat), len(self.lon)), denormalize)
            self.test_lat_weighted_rmse[step] = lat_weighted_rmse(self.out_variables, self.lat, denormalize)
            self.test_lat_weighted_acc[step] = lat_weighted_acc(self.out_variables, self.lat, denormalize)
            self.test_acc_spatial_map[step] = acc_spatial_map(self.out_variables, (len(self.lat), len(self.lon)), denormalize)
            self.test_time_agg_weights[step] = aggregate_attn_weights(len(self.lat) * len(self.lon) // self.patch_size**2, 1, len(self.history), suffix='time')
            self.test_var_agg_weights[step] = aggregate_attn_weights(len(self.lat) * len(self.lon) // self.patch_size**2, 1, len(self.out_variables), suffix='var')

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
                          history=self.history,
                          hrs_each_step=self.hrs_each_step,
                          temporal_attention=self.temporal_attention)
        if len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)
            
    def freeze(self):
        trainable_params = ['time_query', 'time_agg']
        for name, param in self.net.named_parameters():
            if any([t in name for t in trainable_params]) and 'lead_time' not in name:
                print(f'Training param {name}')
                param.requires_grad = True 
            else:
                param.requires_grad = False

    def set_predict_step_size(self, step_size: int):
        self.predict_step_size = step_size

    def set_denormalization(self, denormalization):
        self.denormalization = denormalization
    
    def set_history(self, history: list):
        self.history = history + [0] # + [0] because the model considers the current timestamp as history
        # self.history = history[::8] + [0] # + [0] because the model considers the current timestamp as history
    
    def set_hrs_each_step(self, hrs_each_step: int):
        self.hrs_each_step = hrs_each_step
      
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
            self.test_time_agg_weights[step].to(self.device)
            self.test_var_agg_weights[step].to(self.device)
       
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
    
    def randomize_batch_lead_time(self, batch: Any):
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, remaining_predict_steps, worker_id = batch #spread batch data
        random_lead_time = random.choice(self.lead_time_candidates) #select random lead time from candidates
        
        mask = (lead_times % random_lead_time == 0)[0] #lead times should be the same across batch elements so this is ok
        divisible_lead_times = torch.where(mask)[0]

        #only consider lead times up to max_rollout_steps
        if len(divisible_lead_times) > self.rollout_steps:
            divisible_lead_times = divisible_lead_times[:self.rollout_steps]
        
        divisible_lead_times = divisible_lead_times.cpu() #send to cpu

        #subset data
        y = y[:, divisible_lead_times]
        lead_times = lead_times[:, divisible_lead_times]
        output_timestamps = output_timestamps[:, divisible_lead_times]
        if len(output_timestamps.shape) == 1:
            output_timestamps = output_timestamps[:, np.newaxis]
        if climatology is not None:
            climatology = climatology[:, divisible_lead_times]
            
        return x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, remaining_predict_steps, worker_id
     
    def training_step(self, batch: Any, batch_idx: int):
        if self.randomize_lead_time:
            batch = self.randomize_batch_lead_time(batch)

        x, static, y, _, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch #spread batch data 

        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times within batch not supported.")
       
        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        current_timestamps = input_timestamps[:, -1]
        in_variables = static_variables + ["latitude"] + variables
        dtype = x.dtype

        assert self.rollout_steps <= y.shape[1], f'Invalid rollout steps {self.rollout_steps} for target with {y.shape[1]} steps'

        preds = []
        for step in range(self.rollout_steps):
            inputs = torch.cat((static, x), dim=2).to(dtype)

            delta_time = lead_times[:,step] - lead_times[:,step - 1] if step > 0 else lead_times[:,step]
            dts = delta_time / 100 #divide deltas_times by 100 following climaX 
            step_preds, _, _ = self.net.forward(inputs, dts, step, in_variables, out_variables)

            pred_timestamps = current_timestamps + delta_time.cpu().numpy().astype('timedelta64[h]')
            assert (output_timestamps[:, step] == pred_timestamps).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,step]}'

            preds.append(step_preds)
            # x_latest = step_preds.unsqueeze(1)
            x = torch.cat([x[:,1:], step_preds.unsqueeze(1)], axis=1)
            current_timestamps = pred_timestamps
        
        #set y and preds to float32 for metric calculations
        preds = torch.stack(preds, dim=1).float()
        targets = y.float()

        #flatten batch and step dimensions
        preds = preds.flatten(0,1)
        targets = targets.flatten(0,1)
        
        batch_loss = self.train_lat_weighted_mse(preds, targets) #retains gradients
       
        for var in batch_loss.keys():
            self.log(
                "train/" + var,
                batch_loss[var],
                prog_bar=False,
            )

        self.train_lat_weighted_mse.reset()
        return batch_loss['w_mse_norm']
    
    def validation_step(self, batch: Any, batch_idx: int):
        if self.randomize_lead_time:
            batch = self.randomize_batch_lead_time(batch)
            
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times within batch not supported.")

        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        current_timestamps = input_timestamps[:, -1]
        in_variables = static_variables + ["latitude"] + variables
        dtype = x.dtype

        if self.monitor_val_step:
            assert self.monitor_val_step - 1 <= y.shape[1], f'Invalid test steps {self.monitor_test_steps} to monitor for {self.rollout_steps} rollout steps'

        for step in range(self.rollout_steps):
            inputs = torch.cat((static, x), dim=2).to(dtype)

            delta_time = lead_times[:,step] - lead_times[:,step - 1] if step > 0 else lead_times[:,step]
            dts = delta_time / 100 #divide deltas_times by 100 following climaX 
            preds, _, _ = self.net.forward(inputs, dts, step, in_variables, out_variables)

            pred_timestamps = current_timestamps + delta_time.cpu().numpy().astype('timedelta64[h]')
            
            if not self.monitor_val_step or step + 1 == self.monitor_val_step:
                assert (output_timestamps[:, step] == pred_timestamps).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,step]}'

                #set y and preds to float32 for metric calculations
                preds = preds.float()
                targets = y[:, step].float().squeeze(1)
                clim = climatology[:, step].float().squeeze(1)
                
                self.val_lat_weighted_mse.update(preds, targets)
                self.val_lat_weighted_rmse.update(preds, targets)

                self.val_lat_weighted_acc.update(preds, targets, clim)

            # x_latest = preds.unsqueeze(1)
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
        if self.ensemble_inference:
            self.ensemble_test_step(batch, batch_idx)
        else:
            x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch
        
            if not torch.all(lead_times == lead_times[0]):
                raise NotImplementedError("Variable lead times within batch not supported.")

            #append 2d latitude to static variables
            lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
            static = torch.cat((static,lat2d_expanded), dim=1)
            static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1)
            current_timestamps = input_timestamps[:, -1]
            in_variables = static_variables + ["latitude"] + variables
            dtype = x.dtype
            
            steps = lead_times.shape[1]
            assert max(self.monitor_test_steps) - 1 <= steps, f'Invalid test steps {self.monitor_test_steps} to monitor for {steps} rollout steps'
                    
            for step in range(steps):

                inputs = torch.cat((static, x), dim=2).to(dtype)

                delta_time = lead_times[:,step] - lead_times[:,step - 1] if step > 0 else lead_times[:,step]
                dts = delta_time / 100 #divide deltas_times by 100 following climaX 
                preds, var_agg_weights, time_agg_weights = self.net.forward(inputs, dts, step, in_variables, out_variables, need_weights=True)

                pred_timestamps = current_timestamps + delta_time.numpy().astype('timedelta64[h]')

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

                    if self.temporal_attention:
                        self.test_time_agg_weights[step + 1].update(time_agg_weights)
                    self.test_var_agg_weights[step + 1].update(var_agg_weights)
                    
                # x_latest = preds.unsqueeze(1)
                x = torch.cat([x[:,1:], preds.unsqueeze(1)], axis=1)
                current_timestamps = pred_timestamps
    
    def ensemble_test_step(self, batch: Any, batch_idx: int):
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch
        
        if not torch.all(lead_times == lead_times[0]):
            raise NotImplementedError("Variable lead times within batch not supported.")

        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1)
        current_timestamps = input_timestamps[:, -1]
        in_variables = static_variables + ["latitude"] + variables
        dtype = x.dtype
        
        steps = lead_times.shape[1]
        assert max(self.monitor_test_steps) - 1 <= steps, f'Invalid test steps {self.monitor_test_steps} to monitor for {steps} rollout steps'

        if batch_idx == 0:
            if self.trajectory_path:
                self.trajectories = load_trajectories_from_file(self.trajectory_path)
            else:
                self.trajectories = build_trajectories(self.ensemble_size, lead_times[0].cpu().numpy(), self.lead_time_candidates)
            
        ensembles = {}
        for step in range(steps):
            lead_time = lead_times[0][step].item()
            ensembles[lead_time] = {}
            for traj in self.trajectories[lead_time]:

                traj_tensor = torch.tensor(traj, device=x.device).expand(x.shape[0], -1)
                xt = x
                current_timestamps = input_timestamps[:, -1]

                intermediate_idx = len(traj) - 1
                while intermediate_idx > 0:
                    intermediate_lead_time = sum(traj[:intermediate_idx])
                    key = tuple(traj[:intermediate_idx])
                    if intermediate_lead_time in ensembles and key in ensembles[intermediate_lead_time]:
                        xt = ensembles[intermediate_lead_time][key]
                        time_elapsed = traj_tensor[:,:intermediate_idx].sum(dim = 1)
                        current_timestamps = current_timestamps + time_elapsed.cpu().numpy().astype('timedelta64[h]')
                        break
                    intermediate_idx -= 1

                for i in range(intermediate_idx, len(traj)):
                    inputs = torch.cat((static, xt), dim=2).to(dtype)

                    delta_time = traj_tensor[:, i]
                    dts = delta_time / 100 #divide deltas_times by 100 following climaX 
                    preds, var_agg_weights, time_agg_weights = self.net.forward(inputs, dts, step, in_variables, out_variables, need_weights=True)
                    xt = torch.cat([xt[:,1:], preds.unsqueeze(1)], axis=1)
                    pred_timestamps = current_timestamps + delta_time.cpu().numpy().astype('timedelta64[h]')
                    current_timestamps = pred_timestamps
                

                assert (output_timestamps[:, step] == pred_timestamps).all(), f'Prediction timestamps {pred_timestamps} do not match target timestamps {output_timestamps[:,step]}'
                
                ensembles[lead_time][tuple(traj)] = torch.cat([xt[:,1:], preds.unsqueeze(1)], axis=1)

            members = torch.stack(list(ensembles[lead_time].values()), dim = 1).squeeze(2)
            ensemble_mean = torch.mean(members, dim=1)

            if step + 1 in self.monitor_test_steps:
                
                #set y and preds to float32 for metric calculations
                ensemble_mean = ensemble_mean.float()
                targets = y[:, step].float().squeeze(1)
                clim = climatology[:, step].float().squeeze(1)
                
                self.test_lat_weighted_mse[step + 1].update(ensemble_mean, targets)
                self.test_lat_weighted_rmse[step + 1].update(ensemble_mean, targets)
                self.test_rmse_spatial_map[step + 1].update(ensemble_mean, targets)
                self.test_lat_weighted_acc[step + 1].update(ensemble_mean, targets, clim)
                self.test_acc_spatial_map[step + 1].update(ensemble_mean, targets, clim)

                if self.temporal_attention:
                    self.test_time_agg_weights[step + 1].update(time_agg_weights)
                self.test_var_agg_weights[step + 1].update(var_agg_weights)


    def on_test_epoch_end(self):
        results_dict = {'lead_time_hrs': [], 'out_variables': self.out_variables}
        for step in self.monitor_test_steps:
            lead_time = int(step*self.predict_step_size*self.hrs_each_step) #current forecast lead time in hours
            results_dict['lead_time_hrs'].append(lead_time)
            suffix = f'{lead_time}hrs'
            w_mse_norm = self.test_lat_weighted_mse[step].compute()
            w_rmse = self.test_lat_weighted_rmse[step].compute()
            w_acc = self.test_lat_weighted_acc[step].compute()
            rmse_spatial_maps = self.test_rmse_spatial_map[step].compute()
            acc_spatial_maps = self.test_acc_spatial_map[step].compute()
            var_agg_attn_weights = self.test_var_agg_weights[step].compute()
            time_attn_weights = self.test_time_agg_weights[step].compute() if self.temporal_attention else {}

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
            
            attn_weights_dict = {**var_agg_attn_weights, **time_attn_weights}
            for k in attn_weights_dict.keys():
                if k not in results_dict:
                    results_dict[k] = []
                results_dict[k].append(attn_weights_dict[k].float().cpu().numpy())

            self.test_lat_weighted_mse[step].reset()
            self.test_lat_weighted_rmse[step].reset()
            self.test_lat_weighted_acc[step].reset()
            self.test_acc_spatial_map[step].reset()
            self.test_rmse_spatial_map[step].reset()
            self.test_time_agg_weights[step].reset()
            self.test_var_agg_weights[step].reset()

        
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
