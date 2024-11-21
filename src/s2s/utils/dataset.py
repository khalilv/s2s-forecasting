# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random
import xarray as xr
import numpy as np
import torch
from torch.utils.data import IterableDataset


class ZarrReader(IterableDataset):
    """Dataset for loading numpy files.

    Args:
        file_list (list): List of numpy file paths. Assumed to be sorted.
        static_variable_file (str): Path to static variable file.
        start_idx (float): Start index as a fraction of the total file list.
        end_idx (float): End index as a fraction of the total file list.
        in_variables (list): List of input variables.
        static_variables (list): List of static variables.
        out_variables (list): List of output variables.
        climatology_file (str): Filepath to climatology (optional)
        predict_size (int, optional): Length of outputs. Defaults to 1.
        predict_step (int, optional): Step size between outputs. Defaults to 1.
        history_size (int, optional): Length of history. Defaults to 1. Set to 0 to include only the current timestamp
        history_step (int, optional): Step size between history elements. Defaults to 1.
        hrs_each_step (int, optional): Hours between consecutive steps. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the file list. Defaults to False.
        multi_dataset_training (bool, optional): Flag for multi-dataset training. Defaults to False.
    """
    def __init__(
        self,
        file_list,
        static_variable_file,
        start_idx,
        end_idx,
        in_variables,
        static_variables,
        out_variables,
        climatology_file: str = None,
        predict_size: int = 1,
        predict_step: int = 1,  
        history_size: int = 1,
        history_step: int = 1,
        hrs_each_step: int = 1,
        shuffle: bool = False,
        multi_dataset_training=False,
    ) -> None:
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.climatology_file = climatology_file
        self.static_variable_file = static_variable_file
        self.in_variables = in_variables
        self.static_variables = static_variables
        self.out_variables = out_variables
        self.shuffle = shuffle
        self.multi_dataset_training = multi_dataset_training
        self.predict_size = predict_size
        self.predict_step = predict_step
        self.history_size = history_size
        self.history_step = history_step
        self.hrs_each_step = hrs_each_step
        self.history_range = self.history_size * self.history_step 
        self.predict_range = self.predict_size * self.predict_step
        if self.history_size > 0:
            assert self.history_step > 0, "History step must be greater than 0 when including history"
        if self.predict_size > 0:
            assert self.predict_step > 0, "Predict step must be greater than 0 when forecasting"

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            if self.multi_dataset_training:
                num_nodes = int(os.environ.get("NODES", None))
                num_gpus_per_node = int(world_size / num_nodes)
                num_shards = num_workers_per_ddp * num_gpus_per_node
                rank = rank % num_gpus_per_node
            else:
                num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
        
        static_data = xr.open_zarr(self.static_variable_file, chunks='auto')
        climatology_data = xr.open_zarr(self.climatology_file, chunks='auto') if self.climatology_file else None

        #carry_over_data prevents needlessly throwing out data samples. 
        #it will only be used if files have temporal ordering (i.e. shuffle=false)
        self.carry_over_data = None 
        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = xr.open_zarr(path, chunks='auto')
            if self.carry_over_data is not None and not self.shuffle:
                data = xr.concat([self.carry_over_data, data], dim="time")
            
            assert len(data.time) > (self.predict_range + self.history_range), f"Data shard with size {len(data.time)} is not large enough for a history size of {self.history_size} with step {self.history_step} and a prediction size of {self.predict_size} with step {self.predict_step}"

            if not self.shuffle and idx < iter_end - 1 and (self.predict_range + self.history_range > 0):
                self.carry_over_data = data.isel(time=slice(-(self.predict_range + self.history_range),None))
            
            yield data, static_data, climatology_data, self.in_variables, self.static_variables, self.out_variables, self.predict_range, self.predict_step, self.history_range, self.history_step, self.hrs_each_step


class Forecast(IterableDataset):
    def __init__(
        self, dataset: ZarrReader, 
        normalize_data: bool, 
        in_transforms: torch.nn.Module, 
        static_transforms: torch.nn.Module, 
        output_transforms: torch.nn.Module, 
        region_info = None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.normalize_data = normalize_data
        self.in_transforms = in_transforms
        self.static_transforms = static_transforms
        self.output_transforms = output_transforms
        self.region_info = region_info
        if region_info is not None:
            raise NotImplementedError("Regional forecast is not supported yet.")
        
    def __iter__(self):
        for data, static_data, climatology_data, in_variables, static_variables, out_variables, predict_range, predict_step, history_range, history_step, hrs_each_step in self.dataset:
            static = static_data[static_variables].to_array().transpose('variable','latitude','longitude')
            static = torch.tensor(static.values, dtype=torch.float32)
            for t in range(history_range, len(data.time) - predict_range):
                x = data[in_variables].isel(time=slice(t-history_range,t+1,history_step if history_range > 0 else 1)).to_array().transpose('time', 'variable', 'latitude', 'longitude')               
                input = torch.tensor(x.values, dtype=torch.float32)
                input_timestamps = np.array(x['time'].values)
                if predict_range == 0:
                    y = data[out_variables].isel(time=[t]).to_array().transpose('time', 'variable', 'latitude', 'longitude')               
                    lead_times = torch.tensor([0], dtype=torch.float32)
                else:
                    y = data[out_variables].isel(time=slice(t + predict_step, t + predict_step * (predict_range + 1), predict_step)).to_array().transpose('time', 'variable', 'latitude', 'longitude')
                    lead_times = torch.tensor([hrs_each_step*step for step in range(predict_step, predict_step * (predict_range + 1), predict_step)], dtype=torch.float32)
                output = torch.tensor(y.values, dtype=torch.float32)
                output_timestamps = np.array(y['time'].values)
                if climatology_data is not None:
                    tod = np.array([ts.astype('datetime64[h]').astype(int) % 24 // hrs_each_step for ts in output_timestamps])
                    doy = np.array([(np.datetime64(ts, 'D') - np.datetime64(f"{ts.astype('datetime64[Y]')}-01-01", 'D')).astype(int) for ts in output_timestamps])
                    xr.DataArray(data=np.arange(len(tod)),dims=["pairs"])
                    climatology = climatology_data[out_variables].isel(hour=("pairs", tod), dayofyear=("pairs",doy)).to_array().transpose('pairs', 'variable', 'latitude', 'longitude')
                    climatology = torch.tensor(climatology.values, dtype=torch.float32)
                else:
                    climatology = None
                    
                if self.normalize_data:
                    yield self.in_transforms(input), self.static_transforms(static), self.output_transforms(output), climatology, lead_times, in_variables, static_variables, out_variables, input_timestamps, output_timestamps
                else:
                    yield input, static, output, climatology, lead_times, in_variables, static_variables, out_variables, input_timestamps, output_timestamps

class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()
