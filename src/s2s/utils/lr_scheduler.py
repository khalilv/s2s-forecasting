# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between
    warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and
    eta_min."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Maximum number of iterations for linear warmup
            max_steps (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_steps:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_steps:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_steps - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_steps) % (2 * (self.max_steps - self.warmup_steps)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_steps - self.warmup_steps))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_steps - 1) / (self.max_steps - self.warmup_steps)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_steps:
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / max(1, self.warmup_steps - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            for base_lr in self.base_lrs
        ]

class LinearWarmupConstantLR(_LRScheduler):
    """Linear warmup followed by constant learning rate.

    Linearly increases learning rate from 0 to base_lr over warmup_steps,
    then maintains constant base_lr thereafter.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for linear warmup.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self) -> List[float]:
        """Compute learning rate for a given epoch index."""
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]


if __name__ == '__main__':
    """Example demonstrating both learning rate schedulers."""
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    # Create simple model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Test LinearWarmupCosineAnnealingLR
    print("LinearWarmupCosineAnnealingLR:")
    print("=" * 50)
    scheduler1 = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=100,
        max_steps=1000,
        warmup_start_lr=0.0,
        eta_min=1e-6
    )

    lrs1 = []
    for step in range(1000):
        lrs1.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler1.step()

    print(f"  Initial LR (step 0): {lrs1[0]:.6f}")
    print(f"  LR after warmup (step 100): {lrs1[100]:.6f}")
    print(f"  LR at halfway (step 500): {lrs1[500]:.6f}")
    print(f"  Final LR (step 999): {lrs1[-1]:.6f}")

    # Reset optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Test LinearWarmupConstantLR
    print("\nLinearWarmupConstantLR:")
    print("=" * 50)
    scheduler2 = LinearWarmupConstantLR(
        optimizer,
        warmup_steps=100
    )

    lrs2 = []
    for step in range(1000):
        lrs2.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler2.step()

    print(f"  Initial LR (step 0): {lrs2[0]:.6f}")
    print(f"  LR after warmup (step 100): {lrs2[100]:.6f}")
    print(f"  LR at halfway (step 500): {lrs2[500]:.6f}")
    print(f"  Final LR (step 999): {lrs2[-1]:.6f}")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(lrs1, label='LinearWarmupCosineAnnealing')
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.5, label='Warmup end')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('LinearWarmupCosineAnnealingLR')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(lrs2, label='LinearWarmupConstant', color='orange')
    plt.axvline(x=100, color='r', linestyle='--', alpha=0.5, label='Warmup end')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('LinearWarmupConstantLR')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
