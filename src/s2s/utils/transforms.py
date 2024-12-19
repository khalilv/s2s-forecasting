import torch

class normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean) 
        self.std = torch.tensor(std)

    def __call__(self, x: torch.Tensor):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean[..., None, None]) / std[..., None, None]
    
    def update(self, mean, std):
        assert len(mean) == len(self.mean), f"Length of mean values ({len(mean)}) must match existing mean length ({len(self.mean)})"
        assert len(std) == len(self.std), f"Length of std values ({len(std)}) must match existing std length ({len(self.std)})"

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def denormalize(self, x: torch.Tensor):
        mean = self.mean.to(device=x.device)
        std = self.std.to(device=x.device)
        return x * std[..., None, None] + mean[..., None, None]
