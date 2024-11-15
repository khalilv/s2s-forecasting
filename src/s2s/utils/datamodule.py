# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional
import glob 
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from s2s.utils.data_utils import collate_fn, remove_year
from s2s.utils.dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)

class GlobalForecastDataModule(LightningDataModule):
    """DataModule for global forecast data.

    Args:
        root_dir (str): Root directory for sharded data.
        in_variables (list): List of input variables.
        static_variables (list): List of static variables.
        buffer_size (int): Buffer size for shuffling.
        out_variables (list, optional): List of output variables.
        plot_variables (list, optional): List of variable to plot.
        predict_size (int, optional): Length of outputs. Defaults to 1.
        predict_step (int, optional): Step size between outputs. Defaults to 1.
        history_size (int, optional): Length of history. Defaults to 1. Set to 0 to include only the current timestamp
        history_step (int, optional): Step size between history elements. Defaults to 1.
        hrs_each_step (int, optional): Hours between consecutive steps.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
        normalize_data (bool, optional): Flag to normalize data.
    """

    def __init__(
        self,
        root_dir,
        in_variables,
        static_variables,
        buffer_size = 10000,
        out_variables = None,
        plot_variables = None,
        predict_size: int = 1,
        predict_step: int = 1,
        history_size: int = 1,
        history_step: int = 1,
        hrs_each_step: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalize_data: bool = False,
    ):
        super().__init__()
        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )

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
        self.in_variables = in_variables
        self.static_variables = static_variables
        self.buffer_size = buffer_size
        self.predict_size = predict_size
        self.predict_step = predict_step       
        self.history_size = history_size
        self.history_step = history_step
        self.hrs_each_step = hrs_each_step
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize_data = normalize_data
        
        self.lister_train = sorted(glob.glob(os.path.join(self.root_dir, "train", "*")))
        self.lister_val = sorted(glob.glob(os.path.join(self.root_dir, "val", "*")))
        self.lister_test = sorted(glob.glob(os.path.join(self.root_dir, "test", "*")))
        self.static_variable_file = os.path.join(self.root_dir, "static", "static.npz")

        in_mean, in_std = self.get_normalization_stats(self.in_variables)
        out_mean, out_std = self.get_normalization_stats(self.out_variables)
        static_mean, static_std = self.get_normalization_stats(self.static_variables, "static")
        self.in_transforms = transforms.Normalize(in_mean,in_std)
        self.output_transforms = transforms.Normalize(out_mean, out_std)
        self.static_transforms = transforms.Normalize(static_mean, static_std)

        self.val_climatology_shards, self.val_climatology_timestamp_map = self.load_climatology("val")
        self.test_climatology_shards, self.test_climatology_timestamp_map = self.load_climatology("test")

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None
        
    def get_normalization_stats(self, variables, partition = ""):
        normalize_mean = dict(np.load(os.path.join(self.root_dir, partition, "normalize_mean.npz")))
        normalize_mean = np.array([normalize_mean[var] for var in variables])
        normalize_std = dict(np.load(os.path.join(self.root_dir, partition, "normalize_std.npz")))
        normalize_std = np.array([normalize_std[var] for var in variables])
        return normalize_mean, normalize_std

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.root_dir, "lon.npy"))
        return lat, lon

    def load_climatology(self, partition = ""):
        path = os.path.join(self.root_dir, partition, "*climatology*.npz")
        files = sorted(glob.glob(path))
        assert len(files) > 0, f"No climatology files found in {path}"
        climatology_shards = {}
        timestamp_map = {}
        for file_id, file in enumerate(files):
            shard = np.load(file, mmap_mode='r')
            climatology_shards[file_id] = shard
            timestamps = shard['timestamps']
            for t_id, t in enumerate(timestamps):
                timestamp_map[t] = (file_id, t_id)
        return climatology_shards, timestamp_map
    
    def get_climatology(self, variables, timestamps, partition = ""):
            if partition == 'val':
                climatology_shards = self.val_climatology_shards
                timestamp_map = self.val_climatology_timestamp_map
            elif partition == 'test':
                climatology_shards = self.test_climatology_shards
                timestamp_map = self.test_climatology_timestamp_map
            else:
                raise ValueError(f"Invalid partition '{partition}' for climatology. Must be either 'val' or 'test'.")
            
            climatology = []
            climatology_timestamps = []
            t_ids_per_file = {}
            timestamps_year_removed = [remove_year(t) for t in timestamps]
            for timestamp in timestamps_year_removed:
                if timestamp in timestamp_map:
                    file_id, t_id = timestamp_map[timestamp]
                    if file_id in t_ids_per_file:
                        t_ids_per_file[file_id].append(t_id)
                    else:
                        t_ids_per_file[file_id] = [t_id]
                else:
                    raise KeyError(f"Timestamp {timestamp} not found in climatology data.")
            
            for file_id in t_ids_per_file.keys():
                shard = climatology_shards[file_id]
                shard_data = np.concatenate([shard[var][t_ids_per_file[file_id]] for var in variables], axis=1)
                climatology_timestamps.extend(shard['timestamps'][t_ids_per_file[file_id]])
                climatology.append(shard_data)
                
            climatology = np.concatenate(climatology, axis=0)
            timestamp_index_map = {t: i for i, t in enumerate(climatology_timestamps)}
            ordered_indices = [timestamp_index_map[t] for t in timestamps_year_removed]
            climatology = climatology[ordered_indices] 
            return climatology

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_train,
                            static_variable_file=self.static_variable_file,
                            start_idx=0,
                            end_idx=1,
                            in_variables=self.in_variables,
                            static_variables=self.static_variables,
                            out_variables=self.out_variables,
                            shuffle=False,
                            multi_dataset_training=False,
                            predict_size=self.predict_size,
                            predict_step=self.predict_step,
                            history_size=self.history_size,
                            history_step=self.history_step,
                            hrs_each_step=self.hrs_each_step,
                        ),
                        random_lead_time=False,
                    ),
                    normalize_data = self.normalize_data,
                    in_transforms=self.in_transforms,
                    static_transforms=self.static_transforms,
                    output_transforms=self.output_transforms,
                ),
                buffer_size=self.buffer_size,
            )

            self.data_val = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_val,
                        static_variable_file=self.static_variable_file,
                        start_idx=0,
                        end_idx=1,
                        in_variables=self.in_variables,
                        static_variables=self.static_variables,
                        out_variables=self.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                        predict_size=self.predict_size,
                        predict_step=self.predict_step,
                        history_size=self.history_size,
                        history_step=self.history_step,
                        hrs_each_step=self.hrs_each_step,
                    ),
                    random_lead_time=False,
                ),
                normalize_data = self.normalize_data,
                in_transforms=self.in_transforms,
                static_transforms=self.static_transforms,
                output_transforms=self.output_transforms,
            )

            self.data_test = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_test,
                        static_variable_file=self.static_variable_file,
                        start_idx=0,
                        end_idx=1,
                        in_variables=self.in_variables,
                        static_variables=self.static_variables,
                        out_variables=self.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                        predict_size=self.predict_size,
                        predict_step=self.predict_step,
                        history_size=self.history_size,
                        history_step=self.history_step,
                        hrs_each_step=self.hrs_each_step,
                    ),
                    random_lead_time=False,
                ),
                normalize_data = self.normalize_data,
                in_transforms=self.in_transforms,
                static_transforms=self.static_transforms,
                output_transforms=self.output_transforms,
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
