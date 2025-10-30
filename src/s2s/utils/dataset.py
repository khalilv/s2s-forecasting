# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import xarray as xr
import numpy as np
import torch
from torch.utils.data import IterableDataset

class ZarrReader(IterableDataset):
    """IterableDataset for loading preprocessed zarr files with distributed training support.

    Handles automatic data splitting across distributed ranks and DataLoader workers,
    year grouping for efficiency, and carryover mechanism to preserve temporal continuity.
    Validates spatial coordinate alignment when using climatology.

    Args:
        file_list (list): List of zarr file paths to load.
        static_variable_file (str): Path to static variables zarr file.
        in_variables (list): Input variable names.
        static_variables (list): Static variable names.
        out_variables (list): Output variable names.
        climatology_file (str, optional): Path to climatology zarr. Spatial grids must match data.
        predict_size (int, optional): Number of forecast timesteps. Defaults to 1.
        predict_step (int, optional): Spacing between forecast timesteps (e.g., 6 = every 6 hours). Defaults to 1.
        history (list, optional): Negative integers for historical timesteps (e.g., [-3, -2, -1]). Defaults to [].
        hrs_each_step (int, optional): Hours between consecutive data timesteps. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle year groups. Defaults to False.

    Attributes:
        predict_range (int): Total forecast duration = predict_size * predict_step.
        history_range (int): Total history duration in timesteps.
        years (list): Sorted unique years in dataset.

    Yields:
        Tuples of (data_per_worker, static_data, climatology_data, variables, config, worker_id).
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
        history: list = [],
        hrs_each_step: int = 1,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.climatology_file = climatology_file
        self.static_variable_file = static_variable_file
        self.in_variables = in_variables
        self.static_variables = static_variables
        self.out_variables = out_variables
        self.shuffle = shuffle
        self.predict_size = predict_size
        self.predict_step = predict_step
        self.history = history
        self.hrs_each_step = hrs_each_step
        self.predict_range = self.predict_size * self.predict_step
        if self.history:
            assert all(h < 0 and isinstance(h, int) for h in self.history), "All history elements must be negative integers"
        self.history_range = min(self.history) * -1 if self.history else 0
        if self.predict_size > 0:
            assert self.predict_step > 0, "Predict step must be greater than 0 when forecasting"
        self.years = sorted(set(f.split("/")[-1].split("_")[0] for f in self.file_list))

    def __iter__(self):
        static_data = xr.open_zarr(self.static_variable_file, chunks='auto')
        climatology_data = xr.open_zarr(self.climatology_file, chunks='auto') if self.climatology_file else None
        nyears = ((self.predict_range + self.history_range) * self.hrs_each_step // 8760) + 1
        if self.shuffle:
            groups = [self.years[i:i + nyears] for i in range(len(self.years)- nyears + 1)]
            generator = torch.Generator()
            indices = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[i] for i in indices]
        else:
            groups = [self.years[i:i + nyears] for i in range(0, len(self.years), nyears)]
            
        #carry_over_data prevents needlessly throwing out data samples. 
        #it will only be used if files have temporal ordering (i.e. shuffle=false)
        carry_over_data = None 
        for group in groups:
            files = sorted([f for f in self.file_list if any(f.split('/')[-1].startswith(f"{year}_") for year in group)])

            data = xr.open_mfdataset(
                files,
                engine="zarr",
                concat_dim="time",  
                combine="nested",    
                parallel=True,
                chunks='auto'
            )

            if carry_over_data is not None and not self.shuffle:
                # Validate temporal contiguity between carryover and new data
                time_gap = data.time.values[0] - carry_over_data.time.values[-1]
                expected_gap = np.timedelta64(self.hrs_each_step, 'h')
                assert time_gap == expected_gap, f"Non-contiguous time detected: gap between carryover ({carry_over_data.time.values[-1]}) and new data ({data.time.values[0]}) is {time_gap}, expected {expected_gap}"
                data = xr.concat([carry_over_data, data], dim="time")

            assert len(data.time) > (self.predict_range + self.history_range), f"Data slice with size {len(data.time)} is not large enough for history timestamps {self.history} and a prediction size of {self.predict_size} with step {self.predict_step}"

            if not self.shuffle and (self.predict_range + self.history_range > 0):
                carry_over_data = data.isel(time=slice(-(self.predict_range + self.history_range),None))

            if not torch.distributed.is_initialized():
                global_rank = 0
                world_size = 1
            else:
                global_rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            
            # split data across ranks. each rank will get an identical slice to avoid synchronization issues. if data cannot be split evenly, remainder will be discarded 
            timesteps_per_rank = (len(data.time) + ((world_size - 1)*(self.predict_range + self.history_range))) // world_size
            assert timesteps_per_rank > (self.predict_range + self.history_range), f"Data per rank with size {timesteps_per_rank} is not large enough for history timestamps {self.history} and a prediction size of {self.predict_size} with step {self.predict_step}. Decrease devices."
            rank_start_idx = global_rank * (timesteps_per_rank - (self.predict_range + self.history_range))
            rank_end_idx = rank_start_idx + timesteps_per_rank
            data_per_rank = data.isel(time=slice(rank_start_idx, rank_end_idx))

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
            assert timesteps_per_worker > (self.predict_range + self.history_range), f"Data per worker with size {timesteps_per_worker} is not large enough for history timestamps {self.history} and a prediction size of {self.predict_size} with step {self.predict_step}. Decrease num_workers."
            worker_start_idx = worker_id * (timesteps_per_worker - (self.predict_range + self.history_range))
            worker_end_idx = worker_start_idx + timesteps_per_worker
            data_per_worker = data_per_rank.isel(time=slice(worker_start_idx, worker_end_idx))
            
            if climatology_data is not None:
                assert (data_per_worker.latitude.values == climatology_data.latitude.values).all(), f'Mismatch found between climatology latitudes [{climatology_data.latitude.values[0]},...{climatology_data.latitude.values[-1]}] and data latitudes [{data_per_worker.latitude.values[0]},...{data_per_worker.latitude.values[-1]}]. This will cause the wrong climatology values to be used when calculating ACC'
                assert (data_per_worker.longitude.values == climatology_data.longitude.values).all(), f'Mismatch found between climatology longitudes [{climatology_data.longitude.values[0]},...{climatology_data.longitude.values[-1]}] and data longitudes [{data_per_worker.longitude.values[0]},...{data_per_worker.longitude.values[-1]}]. This will cause the wrong climatology values to be used when calculating ACC'

            print(f'Info: Processing {group}. Rank: {global_rank + 1}/{world_size} gets {rank_start_idx} to {rank_end_idx}. Worker {worker_id + 1}/{num_workers} in rank {global_rank + 1} gets {worker_start_idx} to {worker_end_idx}')
            yield data_per_worker, static_data, climatology_data, self.in_variables, self.static_variables, self.out_variables, self.predict_range, self.predict_step, self.history, self.history_range, self.hrs_each_step, worker_id


class Forecast(IterableDataset):
    """Wraps ZarrReader to generate input/output pairs with optional normalization and climatology.

    Handles memory-efficient data loading via sharding, automatic climatology indexing detection
    and correction (0-indexed vs 1-indexed dayofyear), and alignment of climatology to forecast timestamps.

    Args:
        dataset (ZarrReader): Underlying ZarrReader instance.
        normalize_data (bool): Whether to apply transforms to data.
        in_transforms (torch.nn.Module): Transform for input variables (e.g., NormalizeDenormalize).
        static_transforms (torch.nn.Module): Transform for static variables.
        output_transforms (torch.nn.Module): Transform for output variables.
        mem_load (float, optional): Memory loading strategy. 0.0=minimal (history+forecast+1 timesteps),
            0.5=half dataset, 1.0=entire dataset. Defaults to 0.0.
        region_info (optional): Regional subsetting. Not yet implemented.

    Yields:
        Tuples of (input, static, output, climatology, lead_times, variable_names,
                   timestamps, num_forecast_steps, worker_id).
        - input: Tensor (time, variables, lat, lon) - input features
        - static: Tensor (variables, lat, lon) - static features
        - output: Tensor (time, variables, lat, lon) - forecast targets
        - climatology: Tensor (time, variables, lat, lon) or None
        - lead_times: Tensor (time,) - forecast lead times in hours
    """
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
        for data, static_data, climatology_data, in_variables, static_variables, out_variables, predict_range, predict_step, history, history_range, hrs_each_step, worker_id in self.dataset:
            static = static_data[static_variables].to_array().transpose('variable','latitude','longitude').load()
            static = torch.tensor(static.values)
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
                assert len(data_shard.time) > (predict_range + history_range), f"Data shard with size {len(data_shard.time)} is not large enough for is not large enough for history timestamps {self.history} and a prediction size of {predict_range // predict_step} with step {predict_step}"
                
                if climatology_data is not None:
                    doys = np.array([(np.datetime64(ts, 'D') - np.datetime64(f"{ts.astype('datetime64[Y]')}-01-01", 'D')).astype(int) + 1 for ts in data_shard['time'].values])
                    # Check if climatology is 0-indexed or 1-indexed
                    clim_doy_min = climatology_data.dayofyear.values.min()
                    if clim_doy_min == 0:
                        # Climatology is 0-indexed (0-365), shift to match 1-indexed doys
                        doys_adjusted = doys - 1
                    else:
                        assert clim_doy_min == 1, f"Unexpected climatology dayofyear indexing. Expected min=1 or 0, got {clim_doy_min}"
                        doys_adjusted = doys
                    climatology_shard = climatology_data[out_variables].sel(dayofyear=np.unique(doys_adjusted))
                    climatology_shard = climatology_shard.load()

                if (predict_range + history_range > 0):
                    carry_over_data = data_shard.isel(time=slice(-(predict_range + history_range),None))
            
                data_shard = data_shard.load()
                for t in range(history_range, len(data_shard.time) - predict_range):
                    x = data_shard[in_variables].isel(time=[t + h for h in history + [0]]).to_array().transpose('time', 'variable', 'latitude', 'longitude')               
                    input = torch.tensor(x.values)
                    input_timestamps = np.array(x['time'].values)
                    if predict_range == 0:
                        y = data_shard[out_variables].isel(time=[t]).to_array().transpose('time', 'variable', 'latitude', 'longitude')               
                        lead_times = torch.tensor([0])
                    else:
                        y = data_shard[out_variables].isel(time=slice(t + predict_step, (t + predict_step) + predict_range, predict_step)).to_array().transpose('time', 'variable', 'latitude', 'longitude')
                        lead_times = torch.tensor([hrs_each_step*step for step in range(predict_step, predict_step + predict_range, predict_step)])
                    output = torch.tensor(y.values)
                    output_timestamps = np.array(y['time'].values)
                    if climatology_data is not None:
                        tod = np.array([ts.astype('datetime64[h]').astype(int) % 24 for ts in output_timestamps])
                        doy = np.array([(np.datetime64(ts, 'D') - np.datetime64(f"{ts.astype('datetime64[Y]')}-01-01", 'D')).astype(int) + 1 for ts in output_timestamps])
                        # Adjust doy based on climatology indexing (same logic as above)
                        if clim_doy_min == 0:
                            doy_adjusted = doy - 1
                        else:
                            doy_adjusted = doy
                        try:
                            tod_da = xr.DataArray(tod, dims=["pairs"])
                            doy_da = xr.DataArray(doy_adjusted, dims=["pairs"])
                            climatology = climatology_shard[out_variables].sel(hour=tod_da, dayofyear=doy_da).to_array().transpose('pairs', 'variable', 'latitude', 'longitude')
                            climatology = torch.tensor(climatology.values)
                        except KeyError as e:
                            raise ValueError(
                                f"Failed to select climatology for hour={tod} and dayofyear={doy_adjusted}. "
                                f"Available hours: {climatology_shard.hour.values if 'hour' in climatology_shard else 'N/A'}, "
                                f"Available dayofyear: {climatology_shard.dayofyear.values.min()}-{climatology_shard.dayofyear.values.max()}. "
                                f"Original error: {e}"
                            )
                    else:
                        climatology = None
                        
                    if self.normalize_data:
                        yield self.in_transforms.normalize(input), self.static_transforms.normalize(static), self.output_transforms.normalize(output), climatology, lead_times, in_variables, static_variables, out_variables, input_timestamps, output_timestamps, len(lead_times), worker_id
                    else:
                        yield input, static, output, climatology, lead_times, in_variables, static_variables, out_variables, input_timestamps, output_timestamps, len(lead_times), worker_id

class ShuffleIterableDataset(IterableDataset):
    """Provides buffered shuffling for IterableDatasets.

    Maintains a buffer of samples and randomly yields items from the buffer,
    refilling as items are consumed. Provides better randomization than no shuffling
    while being memory-efficient.

    Args:
        dataset (IterableDataset): Dataset to wrap with shuffling.
        max_buffer_size (int): Size of shuffle buffer. Larger values provide
            better randomization but use more memory. Must be > 0.
    """
    def __init__(self, dataset, max_buffer_size: int) -> None:
        super().__init__()
        assert max_buffer_size > 0
        self.dataset = dataset
        self.max_buffer_size = max_buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.max_buffer_size:
                idx = random.randint(0, self.max_buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()