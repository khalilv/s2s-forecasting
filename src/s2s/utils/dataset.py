# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class NpyReader(IterableDataset):
    """Dataset for loading numpy files.

    Args:
        file_list (list): List of numpy file paths. Assumed to be sorted.
        static_variable_file (str): Path to static variable file.
        start_idx (float): Start index as a fraction of the total file list.
        end_idx (float): End index as a fraction of the total file list.
        in_variables (list): List of input variables.
        static_variables (list): List of static variables.
        out_variables (list): List of output variables.
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
        
        static_data = np.load(self.static_variable_file)
        static_data_dict = {k: static_data[k] for k in self.static_variables}

        #carry_over_data prevents needlessly throwing out data samples. 
        #it will only be used if files have temporal ordering (i.e. shuffle=false)
        self.carry_over_data = None 
        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = np.load(path)
            data_dict  = {k: data[k] for k in self.in_variables}
            timestamps = data['timestamps']
            if self.carry_over_data is not None and not self.shuffle:
                for k in self.in_variables:
                    data_dict[k] = np.concatenate([self.carry_over_data[k], data_dict[k]], axis=0)
                timestamps = np.concatenate([self.carry_over_data['timestamps'], timestamps], axis=0)

            for k in self.in_variables:
                assert data_dict[k].shape[0] > (self.predict_range + self.history_range), f"Data shard with size {data_dict[k].shape[0]} is not large enough for a history size of {self.history_size} with step {self.history_step} and a prediction size of {self.predict_size} with step {self.predict_step}"

            if not self.shuffle and idx < iter_end - 1 and (self.predict_range + self.history_range > 0):
                self.carry_over_data = {k: data_dict[k][-(self.predict_range + self.history_range):] for k in self.in_variables}
                self.carry_over_data['timestamps'] = timestamps[-(self.predict_range + self.history_range):]
            
            yield data_dict, static_data_dict, self.in_variables, self.static_variables, self.out_variables, timestamps, self.predict_range, self.predict_step, self.history_range, self.history_step, self.hrs_each_step


class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, random_lead_time: bool = False
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.random_lead_time = random_lead_time

    def __iter__(self):
        for data, static_data, in_variables, static_variables, out_variables, timestamps, predict_range, predict_step, history_range, history_step, hrs_each_step in self.dataset:
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1) # T, V_in, H, W
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1) # T, V_out, H, W
            y = torch.from_numpy(y)
            static = np.stack([static_data[k].astype(np.float32) for k in static_variables], axis=0) #V_static, H, W
            static = torch.from_numpy(static)
            
            inputs = torch.empty((x.shape[0] - history_range  - predict_range, 
                                  history_range // history_step + 1 if history_range > 0 else 1, x.shape[1], x.shape[2], x.shape[3])) #T, R, V, H, W where R = history size
            input_timestamps = []

            for t in range(history_range, x.shape[0] - predict_range):
                inputs[t-history_range] = x[t-history_range:t+1:history_step if history_range > 0 else 1]
                input_timestamps.append(timestamps[t-history_range:t+1:history_step if history_range > 0 else 1])
            input_timestamps = np.array(input_timestamps)
            
            output_step = predict_step
            lead_times = []
            output_ids = []
            if predict_range == 0:
                lead_times.append(0.0)
                output_ids.append(history_range)
            else:
                while output_step <= predict_range:
                    lead_times.append(output_step * hrs_each_step)
                    output_ids.append(output_step + history_range)
                    output_step += predict_step
            lead_times = torch.tensor(lead_times).unsqueeze(0).repeat(inputs.shape[0], 1)
            output_ids = torch.tensor(output_ids).repeat(inputs.shape[0], 1) + torch.arange(inputs.shape[0]).unsqueeze(1)
            outputs = y[output_ids]
            output_timestamps = timestamps[output_ids]
            yield inputs, static, outputs, lead_times, in_variables, static_variables, out_variables, input_timestamps, output_timestamps


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset, normalize_data: bool, in_transforms: torch.nn.Module, static_transforms: torch.nn.Module, output_transforms: torch.nn.Module, region_info = None):
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
        for (inp, static, out, lead_times, in_variables, static_variables, out_variables, input_timestamps, output_timestamps) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                if self.normalize_data:
                    yield self.in_transforms(inp[i]), self.static_transforms(static), self.output_transforms(out[i]), lead_times[i], in_variables, static_variables, out_variables, input_timestamps[i], output_timestamps[i]
                else:
                    yield inp[i], static, out[i], lead_times[i], in_variables, static_variables, out_variables, input_timestamps[i], output_timestamps[i]


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
