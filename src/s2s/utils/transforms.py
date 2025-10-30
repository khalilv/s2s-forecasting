import torch

class NormalizeDenormalize:
    """Normalization and denormalization transform for atmospheric data using z-score standardization.

    Applies (x - mean) / std for normalization and reverses it for denormalization.
    Broadcasting via [..., None, None] assumes data dimensions are (..., variables, lat, lon).

    Args:
        mean (array-like): Mean value for each variable.
        std (array-like): Standard deviation for each variable.
        device (torch.device, optional): Device for tensors. Defaults to CPU.
        dtype (torch.dtype, optional): Data type. Defaults to float32.

    Note:
        The to() method modifies in-place and returns None (not self), preventing method chaining.
    """
    def __init__(self, mean, std, device=torch.device('cpu'), dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype)

    def to(self, device=None, dtype=None):
        """Move mean and std tensors to specified device and/or dtype. Modifies in-place."""
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        self.mean = self.mean.to(device=self.device, dtype=self.dtype)
        self.std = self.std.to(device=self.device, dtype=self.dtype)
        return

    def normalize(self, x: torch.Tensor):
        """Apply z-score normalization: (x - mean) / std.

        Args:
            x (torch.Tensor): Input tensor with shape (..., variables, lat, lon).

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return (x - self.mean[..., None, None]) / self.std[..., None, None]

    def update(self, mean, std):
        """Update mean and std values. Validates that lengths match existing tensors.

        Args:
            mean (array-like): New mean values.
            std (array-like): New standard deviation values.
        """
        assert len(mean) == len(self.mean), f"Length of mean values ({len(mean)}) must match existing mean length ({len(self.mean)})"
        assert len(std) == len(self.std), f"Length of std values ({len(std)}) must match existing std length ({len(self.std)})"
        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype)

    def denormalize(self, x: torch.Tensor):
        """Reverse z-score normalization: x * std + mean.

        Args:
            x (torch.Tensor): Normalized tensor with shape (..., variables, lat, lon).

        Returns:
            torch.Tensor: Denormalized tensor with same shape as input.
        """
        return x * self.std[..., None, None] + self.mean[..., None, None]


if __name__ == '__main__':
    """Simple example demonstrating normalization and denormalization."""
    print("Testing NormalizeDenormalize...")

    # Create dummy data: 2 variables (temperature, geopotential) with 8x16 spatial grid
    nvars, nlat, nlon = 2, 8, 16
    temperature_mean, temperature_std = 273.5, 10.0
    geopotential_mean, geopotential_std = 50000.0, 1000.0

    temperature = torch.randn(nlat, nlon) * temperature_std + temperature_mean
    geopotential = torch.randn(nlat, nlon) * geopotential_std + geopotential_mean
    data = torch.stack([temperature, geopotential], dim=0)  # Shape: (2, 8, 16)

    print(f"\nOriginal data shape: {data.shape}")
    print(f"Temperature: mean={temperature.mean():.2f}, std={temperature.std():.2f}")
    print(f"Geopotential: mean={geopotential.mean():.2f}, std={geopotential.std():.2f}")

    # Create transform
    mean = [temperature_mean, geopotential_mean]
    std = [temperature_std, geopotential_std]
    transform = NormalizeDenormalize(mean=mean, std=std)

    # Normalize
    normalized = transform.normalize(data)
    print(f"\nNormalized data shape: {normalized.shape}")
    print(f"Normalized mean: {normalized.mean():.4f} (should be ~0)")
    print(f"Normalized std: {normalized.std():.4f} (should be ~1)")

    # Denormalize and verify round-trip
    denormalized = transform.denormalize(normalized)
    print(f"\nDenormalized data shape: {denormalized.shape}")
    matches = torch.allclose(data, denormalized, atol=1e-5)
    print(f"Normalize -> Denormalize produces original data: {matches}")
    if matches:
        print(f"  Max absolute error: {(data - denormalized).abs().max():.2e}")
    else:
        print(f"  ERROR: Round-trip failed!")

    # Test update
    print(f"\nTesting update method...")
    new_mean = [270.0, 48000.0]
    new_std = [12.0, 1200.0]
    transform.update(mean=new_mean, std=new_std)
    print(f"Updated mean: {transform.mean.tolist()}")
    print(f"Updated std: {transform.std.tolist()}")