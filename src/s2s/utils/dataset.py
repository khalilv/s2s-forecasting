# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        in_variables,
        out_variables,
        max_predict_range: int = 6,
        history_range: int = 1,
        hrs_each_step: int = 1,
        shuffle: bool = False,
        multi_dataset_training=False,
    ) -> None:
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.in_variables = in_variables
        self.out_variables = out_variables if out_variables is not None else in_variables
        self.shuffle = shuffle
        self.multi_dataset_training = multi_dataset_training
        self.max_predict_range = max_predict_range
        self.history_range = history_range
        self.hrs_each_step = hrs_each_step

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

            if not self.shuffle and idx < iter_end - 1:
                self.carry_over_data = {k: data_dict[k][-(self.max_predict_range + self.history_range - 1):] for k in self.in_variables}
                self.carry_over_data['timestamps'] = timestamps[-(self.max_predict_range + self.history_range - 1):]
            
            yield data_dict, self.in_variables, self.out_variables, timestamps, self.max_predict_range, self.history_range, self.hrs_each_step


class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, random_lead_time: bool = False
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.random_lead_time = random_lead_time

    def __iter__(self):
        for data, in_variables, out_variables, timestamps, max_predict_range, history_range, hrs_each_step in self.dataset:
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1) # T, V_in, H, W
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1) # T, V_out, H, W
            y = torch.from_numpy(y)

            
            inputs = torch.empty((x.shape[0] - history_range - max_predict_range + 1, 
                                  history_range, x.shape[1], x.shape[2], x.shape[3])) #T,R,V,H,W where R = history size
            input_timestamps = []

            for t in range(history_range, x.shape[0]-max_predict_range + 1):
                inputs[t-history_range] = x[t-history_range:t]
                input_timestamps.append(timestamps[t-history_range:t])
            input_timestamps = np.array(input_timestamps)

            if self.random_lead_time:
                predict_ranges = torch.randint(low=1, high=max_predict_range, size=(inputs.shape[0],))
            else:
                predict_ranges = torch.ones(inputs.shape[0]).to(torch.long) * max_predict_range
            lead_times = hrs_each_step * predict_ranges / 100
            lead_times = lead_times.to(inputs.dtype)
            output_ids = torch.arange(inputs.shape[0]) + predict_ranges + history_range - 1
            outputs = y[output_ids]
            output_timestamps = timestamps[output_ids]
            
            yield inputs, outputs, lead_times, in_variables, out_variables, input_timestamps, output_timestamps


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset, in_transforms: torch.nn.Module, output_transforms: torch.nn.Module, region_info = None):
        super().__init__()
        self.dataset = dataset
        self.in_transforms = in_transforms
        self.output_transforms = output_transforms
        self.region_info = region_info
        if region_info is not None:
            raise NotImplementedError("Regional forecast is not supported yet.")

    def __iter__(self):
        for (inp, out, lead_times, in_variables, out_variables, input_timestamps, output_timestamps) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                if self.region_info is not None:
                    yield self.in_transforms(inp[i]), self.output_transforms(out[i]), lead_times[i], in_variables, out_variables, self.region_info, input_timestamps[i], output_timestamps[i]
                else:
                    yield self.in_transforms(inp[i]), self.output_transforms(out[i]), lead_times[i], in_variables, out_variables, input_timestamps[i], output_timestamps[i]


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
