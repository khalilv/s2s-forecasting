# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import xarray as xr
import numpy as np
import torch
from torch.utils.data import IterableDataset


class ZarrReader(IterableDataset):
    """Dataset for loading zarr files.

    Args:
        file_list (list): List of zarr file paths.
        static_variable_file (str): Path to static variable file.
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
        self.years = sorted(set(f.split("/")[-1].split("_")[0] for f in self.file_list))

    def __iter__(self):
        static_data = xr.open_zarr(self.static_variable_file, chunks='auto')
        climatology_data = xr.open_zarr(self.climatology_file, chunks='auto') if self.climatology_file else None
        if self.shuffle:
            random.shuffle(self.years)

        #carry_over_data prevents needlessly throwing out data samples. 
        #it will only be used if files have temporal ordering (i.e. shuffle=false)
        yearly_carry_over_data = None 
        for year in self.years:
            year_files = sorted([f for f in self.file_list if f.split('/')[-1].startswith(f"{year}_")])
    
            yearly_data = xr.open_mfdataset(
                year_files,
                engine="zarr",
                concat_dim="time",  
                combine="nested",    
                parallel=True,
                chunks='auto'
            )

            if yearly_carry_over_data is not None and not self.shuffle:
                yearly_data = xr.concat([yearly_carry_over_data, yearly_data], dim="time")
            
            assert len(yearly_data.time) > (self.predict_range + self.history_range), f"Yearly data with size {len(yearly_data.time)} is not large enough for a history size of {self.history_size} with step {self.history_step} and a prediction size of {self.predict_size} with step {self.predict_step}"

            if not self.shuffle and (self.predict_range + self.history_range > 0):
                yearly_carry_over_data = yearly_data.isel(time=slice(-(self.predict_range + self.history_range),None))

            if not torch.distributed.is_initialized():
                global_rank = 0
                world_size = 1
            else:
                global_rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            
            # split data across ranks. each rank will get an identical slice to avoid synchronization issues. if data cannot be split evenly, remainder will be discarded 
            timesteps_per_rank = (len(yearly_data.time) + ((world_size - 1)*(self.predict_range + self.history_range))) // world_size
            assert timesteps_per_rank > (self.predict_range + self.history_range), f"Data per rank with size {timesteps_per_rank} is not large enough for a history size of {self.history_size} with step {self.history_step} and a prediction size of {self.predict_size} with step {self.predict_step}. Decrease devices."
            rank_start_idx = global_rank * (timesteps_per_rank - (self.predict_range + self.history_range))
            rank_end_idx = rank_start_idx + timesteps_per_rank
            data_per_rank = yearly_data.isel(time=slice(rank_start_idx, rank_end_idx))


            #within each rank split data across workers
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

            # split data across workers. each worker will get an identical slice to avoid synchronization issues. if data cannot be split evenly, remainder will be discarded 
            timesteps_per_worker = (len(data_per_rank.time) + ((num_workers - 1)*(self.predict_range + self.history_range))) // num_workers
            assert timesteps_per_worker > (self.predict_range + self.history_range), f"Data per worker with size {timesteps_per_worker} is not large enough for a history size of {self.history_size} with step {self.history_step} and a prediction size of {self.predict_size} with step {self.predict_step}. Decrease num_workers."
            worker_start_idx = worker_id * (timesteps_per_worker - (self.predict_range + self.history_range))
            worker_end_idx = worker_start_idx + timesteps_per_worker
            data_per_worker = data_per_rank.isel(time=slice(worker_start_idx, worker_end_idx))
            
            if climatology_data is not None:
                assert (data_per_worker.latitude.values == climatology_data.latitude.values).all(), f'Mismatch found between climatology latitudes [{climatology_data.latitude.values[0]},...{climatology_data.latitude.values[-1]}] and data latitudes [{data_per_worker.latitude.values[0]},...{data_per_worker.latitude.values[-1]}]. This will cause the wrong climatology values to be used when calculating ACC'
            
            print(f'Info: Year {year}. Rank: {global_rank + 1}/{world_size} gets {rank_start_idx} to {rank_end_idx}. Worker {worker_id + 1}/{num_workers} in rank {global_rank + 1} gets {worker_start_idx} to {worker_end_idx}')
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
                shard_length = history_range + predict_range + 1
            elif self.mem_load == 1:
                shard_length = len(data.time)
            else:
                partitions = int(1 / self.mem_load)
                shard_length = max(len(data.time) // partitions, history_range + predict_range + 1)

            carry_over_data = None
            for shard_start in range(0, len(data.time), shard_length):
                shard_end = min(shard_start + shard_length, len(data.time))
                
                data_shard = data.isel(time=slice(shard_start, shard_end))
                if carry_over_data is not None:
                    data_shard = xr.concat([carry_over_data, data_shard], dim="time")
                assert len(data_shard.time) > (predict_range + history_range), f"Data shard with size {len(data_shard.time)} is not large enough for a history size of {history_range // history_step} with step {history_step} and a prediction size of {predict_range // predict_step} with step {predict_step}"
                
                if climatology_data is not None:
                    doys = np.array([(np.datetime64(ts, 'D') - np.datetime64(f"{ts.astype('datetime64[Y]')}-01-01", 'D')).astype(int) + 1 for ts in data_shard['time'].values])
                    climatology_shard = climatology_data[out_variables].sel(dayofyear=np.unique(doys))
                    climatology_shard = climatology_shard.load()

                if (predict_range + history_range > 0):
                    carry_over_data = data_shard.isel(time=slice(-(predict_range + history_range),None))
            
                data_shard = data_shard.load()
                for t in range(history_range, len(data_shard.time) - predict_range):
                    x = data_shard[in_variables].isel(time=slice(t-history_range,t+1,history_step if history_range > 0 else 1)).to_array().transpose('time', 'variable', 'latitude', 'longitude')               
                    input = torch.tensor(x.values, dtype=torch.float32)
                    input_timestamps = np.array(x['time'].values)
                    if predict_range == 0:
                        y = data_shard[out_variables].isel(time=[t]).to_array().transpose('time', 'variable', 'latitude', 'longitude')               
                        lead_times = torch.tensor([0], dtype=torch.float32)
                    else:
                        y = data_shard[out_variables].isel(time=slice(t + predict_step, (t + predict_step) + predict_range, predict_step)).to_array().transpose('time', 'variable', 'latitude', 'longitude')
                        lead_times = torch.tensor([hrs_each_step*step for step in range(predict_step, predict_step + predict_range, predict_step)], dtype=torch.float32)
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
