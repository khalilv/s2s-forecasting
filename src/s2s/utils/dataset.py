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
        iter_start = 0
        iter_end = len(self.file_list)
        
        static_data = xr.open_zarr(self.static_variable_file, chunks='auto')
        climatology_data = xr.open_zarr(self.climatology_file, chunks='auto') if self.climatology_file else None

        #carry_over_data prevents needlessly throwing out data samples. 
        #it will only be used if files have temporal ordering (i.e. shuffle=false)
        carry_over_data = None 
        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = xr.open_zarr(path, chunks='auto')
            if carry_over_data is not None and not self.shuffle:
                data = xr.concat([carry_over_data, data], dim="time")
            
            assert len(data.time) > (self.predict_range + self.history_range), f"Data shard with size {len(data.time)} is not large enough for a history size of {self.history_size} with step {self.history_step} and a prediction size of {self.predict_size} with step {self.predict_step}"

            if not self.shuffle and idx < iter_end - 1 and (self.predict_range + self.history_range > 0):
                carry_over_data = data.isel(time=slice(-(self.predict_range + self.history_range),None))
            
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                num_workers, worker_id = 1, 0
            else:
                num_workers, worker_id = worker_info.num_workers, worker_info.id

            # split data across workers
            timesteps_per_worker = len(data.time) // num_workers
            assert timesteps_per_worker > (self.predict_range + self.history_range), f"Data shard per worker with size {timesteps_per_worker} is not large enough for a history size of {self.history_size} with step {self.history_step} and a prediction size of {self.predict_size} with step {self.predict_step}. Decrease num_workers."
            start_idx = worker_id * timesteps_per_worker
            end_idx = len(data.time) if worker_id == num_workers - 1 else (worker_id + 1) * timesteps_per_worker
            start_idx_plus_carry_over = max(start_idx - (self.predict_range + self.history_range), 0)
            data_per_worker = data.isel(time=slice(start_idx_plus_carry_over, end_idx))
            yield data_per_worker, static_data, climatology_data, self.in_variables, self.static_variables, self.out_variables, self.predict_range, self.predict_step, self.history_range, self.history_step, self.hrs_each_step


class Forecast(IterableDataset):
    def __init__(
        self, dataset: ZarrReader, 
        normalize_data: bool, 
        in_transforms: torch.nn.Module, 
        static_transforms: torch.nn.Module, 
        output_transforms: torch.nn.Module, 
        mem_load: float = 0.0,
        region_info = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.normalize_data = normalize_data
        self.in_transforms = in_transforms
        self.static_transforms = static_transforms
        self.output_transforms = output_transforms
        self.region_info = region_info
        self.mem_load = mem_load
        if region_info is not None:
            raise NotImplementedError("Regional forecast is not supported yet.")
        
    def __iter__(self):
        for data, static_data, climatology_data, in_variables, static_variables, out_variables, predict_range, predict_step, history_range, history_step, hrs_each_step in self.dataset:
            static = static_data[static_variables].to_array().transpose('variable','latitude','longitude').load()
            static = torch.tensor(static.values, dtype=torch.float32)
            if self.mem_load == 0:
                shard_length = 1
            elif self.mem_load == 1:
                shard_length = len(data.time)
            else:
                partitions = int(1 / self.mem_load)
                shard_length = len(data.time) // partitions
            shard_length = max(shard_length, history_range + predict_range + 1)

            carry_over_data = None
            for shard_start in range(0, len(data.time), shard_length):
                shard_end = min(shard_start + shard_length, len(data.time))
                
                data_shard = data.isel(time=slice(shard_start, shard_end))
                if carry_over_data is not None :
                    data_shard = xr.concat([carry_over_data, data_shard], dim="time")
                assert len(data_shard.time) > (predict_range + history_range), f"Data shard with size {len(data_shard.time)} is not large enough for a history size of {history_range // history_step} with step {history_step} and a prediction size of {predict_range // predict_step} with step {predict_step}"
                
                if climatology_data is not None:
                    doys = np.array([(np.datetime64(ts, 'D') - np.datetime64(f"{ts.astype('datetime64[Y]')}-01-01", 'D')).astype(int) + 1 for ts in data_shard['time'].values])
                    climatology_shard = climatology_data[out_variables].sel(dayofyear=np.unique(doys))

                if (predict_range + history_range > 0):
                    carry_over_data = data_shard.isel(time=slice(-(predict_range + history_range),None))
            
                data_shard = data_shard.load()
                climatology_shard = climatology_shard.load()
                for t in range(history_range, len(data_shard.time) - predict_range):
                    x = data_shard[in_variables].isel(time=slice(t-history_range,t+1,history_step if history_range > 0 else 1)).to_array().transpose('time', 'variable', 'latitude', 'longitude')               
                    input = torch.tensor(x.values, dtype=torch.float32)
                    input_timestamps = np.array(x['time'].values)
                    if predict_range == 0:
                        y = data_shard[out_variables].isel(time=[t]).to_array().transpose('time', 'variable', 'latitude', 'longitude')               
                        lead_times = torch.tensor([0], dtype=torch.float32)
                    else:
                        y = data_shard[out_variables].isel(time=slice(t + predict_step, t + predict_step * (predict_range + 1), predict_step)).to_array().transpose('time', 'variable', 'latitude', 'longitude')
                        lead_times = torch.tensor([hrs_each_step*step for step in range(predict_step, predict_step * (predict_range + 1), predict_step)], dtype=torch.float32)
                    output = torch.tensor(y.values, dtype=torch.float32)
                    output_timestamps = np.array(y['time'].values)
                    if climatology_data is not None:
                        tod = np.array([ts.astype('datetime64[h]').astype(int) % 24 for ts in output_timestamps])
                        doy = np.array([(np.datetime64(ts, 'D') - np.datetime64(f"{ts.astype('datetime64[Y]')}-01-01", 'D')).astype(int) + 1 for ts in output_timestamps])
                        tod_da = xr.DataArray(tod, dims=["pairs"])
                        doy_da = xr.DataArray(doy, dims=["pairs"])
                        climatology = climatology_shard[out_variables].sel(hour=tod_da, dayofyear=doy_da).to_array().transpose('pairs', 'variable', 'latitude', 'longitude')
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
