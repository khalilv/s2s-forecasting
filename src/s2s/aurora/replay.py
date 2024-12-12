import random
from s2s.utils.data_utils import collate_fn

class ReplayBuffer():
    def __init__(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add_batch(self, batch):
        x, static, y, _, lead_times, variables, static_variables, out_variables, input_timestamps, output_timestamps, remaining_predict_steps, worker_ids = batch
        for i in range(x.shape[0]):
            self.add_sample((x[i], static[i], y[i], None, lead_times[i], variables, static_variables, out_variables, input_timestamps[i], output_timestamps[i], remaining_predict_steps[i], worker_ids[i]))

    def add_sample(self, sample):
        self.buffer.append(sample)

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