# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, Callable
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
from s2s.utils.data_utils import plot_spatial_map_with_basemap
#3) Global forecast module - abstraction for training/validation/testing steps. setup for the module including hyperparameters is included here

class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting with the ClimaX model.

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
        self.denormalization = None
        self.lat = None
        self.lon = None
        self.plot_variables = []
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
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    
    def init_metrics(self):
        assert self.lat is not None, 'Latitude values not initialized yet.'
        assert self.lon is not None, 'Longitude values not initialized yet.'
        denormalize = self.denormalization.denormalize if self.denormalization else None
        self.train_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat)
        self.val_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, denormalize)
        self.val_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, denormalize)
        self.val_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, denormalize)
        self.test_lat_weighted_mse = lat_weighted_mse(self.out_variables, self.lat, denormalize)        
        self.test_rmse_spatial_map = rmse_spatial_map(self.out_variables, (len(self.lat), len(self.lon)), denormalize)
        self.test_lat_weighted_rmse = lat_weighted_rmse(self.out_variables, self.lat, denormalize)
        self.test_lat_weighted_acc = lat_weighted_acc(self.out_variables, self.lat, denormalize)
        self.test_acc_spatial_map = acc_spatial_map(self.out_variables, (len(self.lat), len(self.lon)), denormalize)

    def init_network(self):
        variables = self.static_variables + ["lattitude"] + self.in_variables #climaX includes 2d latitude as an input field
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
                          parallel_patch_embed=self.parallel_patch_embed)
        if len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)

    def set_denormalization(self, denormalization):
        self.denormalization = denormalization
      
    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
       
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
        x, static, y, _, lead_times, variables, static_variables, out_variables, _, _, _, _ = batch #spread batch data 
        
        if batch_idx == 0:
            self.denormalization.to(device=y.device, dtype=y.dtype)

        if y.shape[1] > 1:
            raise NotImplementedError('Multiple prediction steps is not supported yet.')
        
        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)

        #prepend static variables to input variables
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        inputs = torch.cat((static, x), dim=2).to(x.dtype)

        in_variables = static_variables + ["lattitude"] + variables

        if inputs.shape[1] > 1:
            raise NotImplementedError("History_range > 1 is not supported yet.")
        inputs = inputs.squeeze(1) #squeeze history dimension
   
        #divide lead_times by 100 following climaX
        lead_times = lead_times / 100
        lead_times = lead_times.squeeze(1)

        preds = self.net.forward(inputs, lead_times, in_variables, out_variables)
        
        #set y and preds to float32 for metric calculations
        preds = preds.float()
        y = y.to(dtype=preds.dtype, device=preds.device).squeeze(1)
        
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
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, _, _, _, _ = batch
        
        if batch_idx == 0:
            self.denormalization.to(device=y.device, dtype=y.dtype)

        if y.shape[1] > 1:
            raise NotImplementedError('Multiple prediction steps is not supported yet.')
        
        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)

        #prepend static variables to input variables
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        inputs = torch.cat((static, x), dim=2).to(x.dtype)

        in_variables = static_variables + ["lattitude"] + variables

        if inputs.shape[1] > 1:
            raise NotImplementedError("History_range > 1 is not supported yet.")
        inputs = inputs.squeeze(1) #squeeze history dimension

        #divide lead_times by 100 following climaX
        lead_times = lead_times / 100
        lead_times = lead_times.squeeze(1)

        preds = self.net.forward(inputs, lead_times, in_variables, out_variables)

        #set y and preds to float32 for metric calculations
        preds = preds.float()
        y = y.float().squeeze(1)
        climatology = climatology.float().squeeze(1)

        self.val_lat_weighted_mse.update(preds, y)
        self.val_lat_weighted_rmse.update(preds, y)

        self.val_lat_weighted_acc.update(preds, y, climatology)
        
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
        x, static, y, climatology, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, _, _ = batch
        
        if batch_idx == 0:
            self.denormalization.to(device=y.device, dtype=y.dtype)

        if y.shape[1] > 1:
            raise NotImplementedError('Multiple prediction steps is not supported yet.')

        #append 2d latitude to static variables
        lat2d_expanded = self.lat2d.unsqueeze(0).expand(static.shape[0], -1, -1, -1).to(device=static.device)
        static = torch.cat((static,lat2d_expanded), dim=1)

        #prepend static variables to input variables
        static = static.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1) 
        inputs = torch.cat((static, x), dim=2).to(x.dtype)

        in_variables = static_variables + ["lattitude"] + variables

        if inputs.shape[1] > 1:
            raise NotImplementedError("History_range > 1 is not supported yet.")
        inputs = inputs.squeeze(1) #squeeze history dimension

        #divide lead_times by 100 following climaX
        lead_times = lead_times / 100
        lead_times = lead_times.squeeze(1)

        preds = self.net.forward(inputs, lead_times, in_variables, out_variables)

        #set y and preds to float32 for metric calculations
        preds = preds.float()
        y = y.float().squeeze(1)
        climatology = climatology.float().squeeze(1)
        
        self.test_lat_weighted_mse.update(preds, y)
        self.test_lat_weighted_rmse.update(preds, y)
        self.test_rmse_spatial_map.update(preds, y)

        self.test_lat_weighted_acc.update(preds, y, climatology)
        self.test_acc_spatial_map.update(preds, y, climatology)

    def on_test_epoch_end(self):
        w_mse = self.test_lat_weighted_mse.compute()
        w_rmse = self.test_lat_weighted_rmse.compute()
        w_acc = self.test_lat_weighted_acc.compute()
        rmse_spatial_maps = self.test_rmse_spatial_map.compute()
        acc_spatial_maps = self.test_acc_spatial_map.compute()

        #scalar metrics
        loss_dict = {**w_mse, **w_rmse, **w_acc}
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
                        plot_spatial_map_with_basemap(data=rmse_spatial_maps[var].float().cpu(), lat=latitudes, lon=longitudes, title=var, filename=f"{self.logger.log_dir}/test_{var}.png")
            for plot_var in tqdm(self.plot_variables, desc="Plotting ACC spatial maps"):
                for var in acc_spatial_maps.keys():
                    if plot_var in var:
                        plot_spatial_map_with_basemap(data=acc_spatial_maps[var].float().cpu(), lat=latitudes, lon=longitudes, title=var, filename=f"{self.logger.log_dir}/test_{var}.png")

        self.test_lat_weighted_mse.reset()
        self.test_lat_weighted_rmse.reset()
        self.test_lat_weighted_acc.reset()
        self.test_acc_spatial_map.reset()
        self.test_rmse_spatial_map.reset()

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
