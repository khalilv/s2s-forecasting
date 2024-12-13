import random
from s2s.utils.data_utils import collate_fn

class ReplayBuffer():
    def __init__(self, to_cpu: bool = False):
        self.buffer = []
        self.to_cpu = to_cpu

    def __len__(self):
        return len(self.buffer)

    def add(self, batch):
        x, static, y, _, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, remaining_predict_steps, worker_ids = batch
        if self.to_cpu:
            x, static, y, lead_times = x.to('cpu'), static.to('cpu'), y.to('cpu'), lead_times.to('cpu')
        for i in range(x.shape[0]):
            self.buffer.append((x[i], static[i], y[i], None, lead_times[i], variables, static_variables, out_variables, input_timestamps[i], output_timestamps[i], remaining_predict_steps[i], worker_ids[i]))

    def sample(self, batch_size):
        batch_size = min(self.__len__(), batch_size)
        idxs = random.sample(range(self.__len__()), batch_size)
        samples = [self.buffer[i] for i in idxs]
        for i, idx in enumerate(idxs):
            self.buffer[idx], self.buffer[-(i + 1)] = self.buffer[-(i + 1)], self.buffer[idx]
        self.buffer = self.buffer[:-batch_size]
        return collate_fn(samples)

    def reset(self):
        self.buffer = []