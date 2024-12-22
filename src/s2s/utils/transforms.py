import torch

class NormalizeDenormalize:
    def __init__(self, mean, std, device=torch.device('cpu'), dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype) 
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype)

    def to(self, device=None, dtype=None):
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        self.mean = self.mean.to(device=self.device, dtype=self.dtype)
        self.std = self.std.to(device=self.device, dtype=self.dtype)
        return

    def normalize(self, x: torch.Tensor):
        return (x - self.mean[..., None, None]) / self.std[..., None, None]
    
    def update(self, mean, std):
        assert len(mean) == len(self.mean), f"Length of mean values ({len(mean)}) must match existing mean length ({len(self.mean)})"
        assert len(std) == len(self.std), f"Length of std values ({len(std)}) must match existing std length ({len(self.std)})"
        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype)
    
    def denormalize(self, x: torch.Tensor):
        return x * self.std[..., None, None] + self.mean[..., None, None]
