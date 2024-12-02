"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from typing import Generator, Union

import torch

from s2s.aurora.batch import Batch
from s2s.aurora.model.aurora import Aurora

__all__ = ["rollout"]


def rollout(model: Aurora, batch: Batch, steps: int, yield_steps: list = []) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.
        yield_intermediate (bool): If true, yield intermediate predictions.
        yield_steps (list): List of steps to yield intermediate predictions on.

    Yields:
        :class:`aurora.batch.Batch`: The prediction at each yield step.

    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    for step in range(steps):
        pred = model.forward(batch)

        if step in yield_steps:
            yield pred

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )