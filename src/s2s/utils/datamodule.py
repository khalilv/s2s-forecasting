# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import xarray as xr
from typing import Optional
import glob 
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from s2s.utils.data_utils import collate_fn
from s2s.utils.transforms import NormalizeDenormalize
from s2s.utils.dataset import (
    Forecast,
    ZarrReader,
    ShuffleIterableDataset
)

class GlobalForecastDataModule(LightningDataModule):
    """DataModule for global forecast data.

    Args:
        root_dir (str): Root directory for sharded data.
        climatology_val (str): Path to zarr file for validation climatology.
        climatology_test (str): Path to zarr file for test climatology.
        in_variables (list): List of input variables.
        static_variables (list): List of static variables.
        max_buffer_size (int): Maximum buffer size for shuffling.
        out_variables (list, optional): List of output variables.
        plot_variables (list, optional): List of variable to plot.
        predict_size (int, optional): Length of outputs. Defaults to 1.
        predict_step (int, optional): Step size between outputs. Defaults to 1.
        history (list, optional): List of history elements to include in input. 
            If provided must all be negative integers. Defaults to [] (current timestamp).
        hrs_each_step (int, optional): Hours between consecutive steps.
        batch_size (int, optional): Batch size.
        mem_load (float, optional): Between 0 and 1 inclusive. Percentage of data to load at a time for each worker. 
            1 = load entire data slice (fastest). 0 = load one sample at a time (lowest memory usage).
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
        normalize_data (bool, optional): Flag to normalize data.
    """

    def __init__(
        self,
        root_dir,
        climatology_val,
        climatology_test,
        in_variables,
        static_variables,
        max_buffer_size: int = 100,
        out_variables = None,
        plot_variables = None,
        predict_size: int = 1,
        predict_step: int = 1,
        history: list = [],
        hrs_each_step: int = 1,
        batch_size: int = 64,
        mem_load: float = 0.0,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalize_data: bool = False,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        if out_variables is None:
            self.out_variables = in_variables #set output variables equal to input variables if not specified 
        elif isinstance(out_variables, str):
            out_variables = [out_variables]
            self.out_variables = out_variables
        else:
            self.out_variables = out_variables
        
        if plot_variables is None:
            self.plot_variables = [] #set plot variables to an empty list if not specified
        elif isinstance(out_variables, str):
            plot_variables = [plot_variables]
            self.plot_variables = plot_variables
        else:
            self.plot_variables = plot_variables

        self.root_dir = root_dir
        self.climatology_val = climatology_val
        self.climatology_test = climatology_test
        self.in_variables = in_variables
        self.static_variables = static_variables
        self.max_buffer_size = max_buffer_size
        self.predict_size = predict_size
        self.predict_step = predict_step       
        self.history = history
        self.hrs_each_step = hrs_each_step
        self.batch_size = batch_size
        self.mem_load = mem_load
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize_data = normalize_data
        
        self.lister_train = sorted(glob.glob(os.path.join(self.root_dir, "train", "*")))
        self.lister_val = sorted(glob.glob(os.path.join(self.root_dir, "val", "*")))
        self.lister_test = sorted(glob.glob(os.path.join(self.root_dir, "test", "*")))
        self.static_variable_file = os.path.join(self.root_dir, "static", "static.zarr")

        in_mean, in_std = self.get_normalization_stats(self.in_variables)
        out_mean, out_std = self.get_normalization_stats(self.out_variables)
        static_mean, static_std = self.get_normalization_stats(self.static_variables, "static")
        self.in_transforms = NormalizeDenormalize(in_mean, in_std)
        self.output_transforms = NormalizeDenormalize(out_mean, out_std)
        self.static_transforms = NormalizeDenormalize(static_mean, static_std)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

        if not self.normalize_data:
            print('Warning: Both input and output data will not be normalized. Model will be run on unnormalized data')

    def get_normalization_stats(self, variables, partition = ""):
        statistics = xr.open_zarr(os.path.join(self.root_dir, partition, "statistics.zarr"), chunks='auto')
        normalize_mean = np.array([statistics[f"{var}_mean"] for var in variables])
        normalize_std = np.array([statistics[f"{var}_std"] for var in variables])
        return normalize_mean, normalize_std
    
    def get_history(self):
        return self.history

    def get_hrs_each_step(self):
        return self.hrs_each_step

    def get_predict_step_size(self):
        return self.predict_step
    
    def get_transforms(self, group: str):
        if group == 'in':
            return copy.deepcopy(self.in_transforms)
        elif group == 'out':
            return copy.deepcopy(self.output_transforms)
        elif group == 'static':
            return copy.deepcopy(self.static_transforms)
        else:
            raise ValueError(f"Invalid normalization group name: {group}")

    def update_normalization_stats(self, mean, std, group: str):
        if not self.normalize_data:
            print(f"Warning: Updating normalization statistics for normalization group: {group} when normalize_data is False. This will likely have no effect.")
        else:
            print(f"Info: Updating normalization statistics for normalization group: {group}")
        if group == 'in':
            self.in_transforms.update(mean, std)
        elif group == 'out':
            self.output_transforms.update(mean, std)
        elif group == 'static':
            self.static_transforms.update(mean, std)
        else:
            raise ValueError(f"Invalid normalization group name: {group}")

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.root_dir, "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.data_train = ShuffleIterableDataset(
                Forecast(
                    ZarrReader(
                        file_list=self.lister_train,
                        static_variable_file=self.static_variable_file,
                        in_variables=self.in_variables,
                        static_variables=self.static_variables,
                        out_variables=self.out_variables,
                        shuffle=True,
                        predict_size=self.predict_size,
                        predict_step=self.predict_step,
                        history=self.history,
                        hrs_each_step=self.hrs_each_step,
                    ),
                    normalize_data = self.normalize_data,
                    in_transforms=self.in_transforms,
                    static_transforms=self.static_transforms,
                    output_transforms=self.output_transforms,
                    mem_load=self.mem_load
                ),
                max_buffer_size=self.max_buffer_size,
            )
            self.data_val = Forecast(
                ZarrReader(
                    file_list=self.lister_val,
                    climatology_file=self.climatology_val,
                    static_variable_file=self.static_variable_file,
                    in_variables=self.in_variables,
                    static_variables=self.static_variables,
                    out_variables=self.out_variables,
                    shuffle=False,
                    predict_size=self.predict_size,
                    predict_step=self.predict_step,
                    history=self.history,
                    hrs_each_step=self.hrs_each_step,
                ),
                normalize_data = self.normalize_data,
                in_transforms=self.in_transforms,
                static_transforms=self.static_transforms,
                output_transforms=self.output_transforms,
                mem_load=self.mem_load
            )
                
        if stage == 'test':
            self.data_test = Forecast(
                ZarrReader(
                    file_list=self.lister_test,
                    climatology_file=self.climatology_test,
                    static_variable_file=self.static_variable_file,
                    in_variables=self.in_variables,
                    static_variables=self.static_variables,
                    out_variables=self.out_variables,
                    shuffle=False,
                    predict_size=self.predict_size,
                    predict_step=self.predict_step,
                    history=self.history,
                    hrs_each_step=self.hrs_each_step,
                ),
                normalize_data = self.normalize_data,
                in_transforms=self.in_transforms,
                static_transforms=self.static_transforms,
                output_transforms=self.output_transforms,
                mem_load=self.mem_load
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
