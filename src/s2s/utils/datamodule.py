# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import xarray as xr
from typing import Optional
import glob 
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from s2s.utils.data_utils import collate_fn
from s2s.utils.transforms import NormalizeDenormalize
from s2s.utils.dataset import (
    Forecast,
    ZarrReader,
    ShuffleIterableDataset
)

class GlobalForecastDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for global forecasting.

    High-level interface for loading train/val/test datasets with automatic normalization,
    climatology integration, and dataloader creation. This is the main entry point for
    users working with preprocessed ERA5 data.

    Automatically:
    - Loads normalization statistics from preprocessing
    - Creates ZarrReader → Forecast → DataLoader pipeline
    - Applies shuffling to training data
    - Integrates climatology for validation and test sets
    - Manages transforms for input, output, and static variables

    Args:
        root_dir (str): Root directory containing train/val/test subdirectories with zarr shards.
        climatology_val (str): Path to validation climatology zarr file.
        climatology_test (str): Path to test climatology zarr file.
        in_variables (list): Input variable names (e.g., ['2m_temperature', 'geopotential']).
        static_variables (list): Static variable names (e.g., ['orography', 'land_sea_mask']).
        max_buffer_size (int, optional): Buffer size for training data shuffling. Defaults to 100.
        out_variables (list, optional): Output variables. If None, uses in_variables. Defaults to None.
        plot_variables (list, optional): Variables to plot during evaluation. Defaults to None.
        predict_size (int, optional): Number of forecast timesteps. Defaults to 1.
        predict_step (int, optional): Spacing between forecast timesteps. Defaults to 1.
        history (list, optional): Negative integers for historical timesteps (e.g., [-6, -12]). Defaults to [].
        hrs_each_step (int, optional): Hours between consecutive data timesteps. Defaults to 1.
        batch_size (int, optional): Batch size for dataloaders. Defaults to 64.
        mem_load (float, optional): Memory loading strategy (0.0-1.0). 0.0=minimal, 1.0=load all. Defaults to 0.0.
        num_workers (int, optional): Number of dataloader workers. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory for faster GPU transfer. Defaults to False.
        normalize_data (bool, optional): Whether to apply normalization. Defaults to False.
    """

    def __init__(
        self,
        root_dir,
        climatology_val,
        climatology_test,
        in_variables,
        static_variables,
        max_buffer_size: int = 100,
        out_variables = None,
        plot_variables = None,
        predict_size: int = 1,
        predict_step: int = 1,
        history: list = [],
        hrs_each_step: int = 1,
        batch_size: int = 64,
        mem_load: float = 0.0,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalize_data: bool = False,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        if out_variables is None:
            self.out_variables = in_variables #set output variables equal to input variables if not specified 
        elif isinstance(out_variables, str):
            out_variables = [out_variables]
            self.out_variables = out_variables
        else:
            self.out_variables = out_variables
        
        if plot_variables is None:
            self.plot_variables = [] #set plot variables to an empty list if not specified
        elif isinstance(out_variables, str):
            plot_variables = [plot_variables]
            self.plot_variables = plot_variables
        else:
            self.plot_variables = plot_variables

        self.root_dir = root_dir
        self.climatology_val = climatology_val
        self.climatology_test = climatology_test
        self.in_variables = in_variables
        self.static_variables = static_variables
        self.max_buffer_size = max_buffer_size
        self.predict_size = predict_size
        self.predict_step = predict_step       
        self.history = history
        self.hrs_each_step = hrs_each_step
        self.batch_size = batch_size
        self.mem_load = mem_load
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize_data = normalize_data
        
        self.lister_train = sorted(glob.glob(os.path.join(self.root_dir, "train", "*")))
        self.lister_val = sorted(glob.glob(os.path.join(self.root_dir, "val", "*")))
        self.lister_test = sorted(glob.glob(os.path.join(self.root_dir, "test", "*")))
        self.static_variable_file = os.path.join(self.root_dir, "static", "static.zarr")

        in_mean, in_std = self.get_normalization_stats(self.in_variables)
        out_mean, out_std = self.get_normalization_stats(self.out_variables)
        static_mean, static_std = self.get_normalization_stats(self.static_variables, "static")
        self.in_transforms = NormalizeDenormalize(in_mean, in_std)
        self.output_transforms = NormalizeDenormalize(out_mean, out_std)
        self.static_transforms = NormalizeDenormalize(static_mean, static_std)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

        if not self.normalize_data:
            print('Warning: Both input and output data will not be normalized. Model will be run on unnormalized data')

    def get_normalization_stats(self, variables, partition = ""):
        """Load normalization statistics (mean and std) for specified variables.

        Args:
            variables (list): Variable names to load statistics for.
            partition (str, optional): Subdirectory ('static' or ''). Defaults to ''.

        Returns:
            tuple: (mean array, std array) for the variables.
        """
        statistics = xr.open_zarr(os.path.join(self.root_dir, partition, "statistics.zarr"), chunks='auto')
        normalize_mean = np.array([statistics[f"{var}_mean"] for var in variables])
        normalize_std = np.array([statistics[f"{var}_std"] for var in variables])
        return normalize_mean, normalize_std

    def get_history(self):
        """Get the history configuration (list of negative timestep offsets)."""
        return self.history

    def get_hrs_each_step(self):
        """Get the hours between consecutive timesteps in the data."""
        return self.hrs_each_step

    def get_predict_step_size(self):
        """Get the forecast step size (spacing between output timesteps)."""
        return self.predict_step

    def get_transforms(self, group: str):
        """Get a deep copy of transforms for specified variable group.

        Args:
            group (str): Variable group ('in', 'out', or 'static').

        Returns:
            NormalizeDenormalize: Copy of the transform for the specified group.
        """
        if group == 'in':
            return copy.deepcopy(self.in_transforms)
        elif group == 'out':
            return copy.deepcopy(self.output_transforms)
        elif group == 'static':
            return copy.deepcopy(self.static_transforms)
        else:
            raise ValueError(f"Invalid normalization group name: {group}")

    def update_normalization_stats(self, mean, std, group: str):
        """Update normalization statistics for specified variable group.

        Args:
            mean (array-like): New mean values.
            std (array-like): New standard deviation values.
            group (str): Variable group to update ('in', 'out', or 'static').
        """
        if not self.normalize_data:
            print(f"Warning: Updating normalization statistics for normalization group: {group} when normalize_data is False. This will likely have no effect.")
        else:
            print(f"Info: Updating normalization statistics for normalization group: {group}")
        if group == 'in':
            self.in_transforms.update(mean, std)
        elif group == 'out':
            self.output_transforms.update(mean, std)
        elif group == 'static':
            self.static_transforms.update(mean, std)
        else:
            raise ValueError(f"Invalid normalization group name: {group}")

    def get_lat_lon(self):
        """Load and return latitude and longitude coordinates.

        Returns:
            tuple: (latitude array, longitude array).
        """
        lat = np.load(os.path.join(self.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.root_dir, "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for specified stage ('fit' for train/val, 'test' for test).

        Called automatically by PyTorch Lightning trainer.

        Args:
            stage (str, optional): 'fit' or 'test'. Defaults to None.
        """
        if stage == 'fit':
            self.data_train = ShuffleIterableDataset(
                Forecast(
                    ZarrReader(
                        file_list=self.lister_train,
                        static_variable_file=self.static_variable_file,
                        in_variables=self.in_variables,
                        static_variables=self.static_variables,
                        out_variables=self.out_variables,
                        shuffle=True,
                        predict_size=self.predict_size,
                        predict_step=self.predict_step,
                        history=self.history,
                        hrs_each_step=self.hrs_each_step,
                    ),
                    normalize_data = self.normalize_data,
                    in_transforms=self.in_transforms,
                    static_transforms=self.static_transforms,
                    output_transforms=self.output_transforms,
                    mem_load=self.mem_load
                ),
                max_buffer_size=self.max_buffer_size,
            )
            self.data_val = Forecast(
                ZarrReader(
                    file_list=self.lister_val,
                    climatology_file=self.climatology_val,
                    static_variable_file=self.static_variable_file,
                    in_variables=self.in_variables,
                    static_variables=self.static_variables,
                    out_variables=self.out_variables,
                    shuffle=False,
                    predict_size=self.predict_size,
                    predict_step=self.predict_step,
                    history=self.history,
                    hrs_each_step=self.hrs_each_step,
                ),
                normalize_data = self.normalize_data,
                in_transforms=self.in_transforms,
                static_transforms=self.static_transforms,
                output_transforms=self.output_transforms,
                mem_load=self.mem_load
            )
                
        if stage == 'test':
            self.data_test = Forecast(
                ZarrReader(
                    file_list=self.lister_test,
                    climatology_file=self.climatology_test,
                    static_variable_file=self.static_variable_file,
                    in_variables=self.in_variables,
                    static_variables=self.static_variables,
                    out_variables=self.out_variables,
                    shuffle=False,
                    predict_size=self.predict_size,
                    predict_step=self.predict_step,
                    history=self.history,
                    hrs_each_step=self.hrs_each_step,
                ),
                normalize_data = self.normalize_data,
                in_transforms=self.in_transforms,
                static_transforms=self.static_transforms,
                output_transforms=self.output_transforms,
                mem_load=self.mem_load
            )

    def train_dataloader(self):
        """Create training dataloader with shuffling enabled.

        Returns:
            DataLoader: Training data loader.
        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Create validation dataloader with climatology integration.

        Returns:
            DataLoader: Validation data loader.
        """
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """Create test dataloader with climatology integration.

        Returns:
            DataLoader: Test data loader.
        """
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )


if __name__ == '__main__':
    """Example demonstrating the complete datamodule workflow with dummy preprocessed data."""
    import tempfile
    import shutil

    print("Creating dummy preprocessed data structure...")

    tmp_dir = tempfile.mkdtemp(prefix='datamodule_example_')

    try:
        # Create directory structure matching preprocessed ERA5 output
        os.makedirs(os.path.join(tmp_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, 'static'), exist_ok=True)

        # Small grid for demonstration
        nlat, nlon = 8, 16
        lat = np.linspace(90, -90, nlat)
        lon = np.linspace(0, 360, nlon, endpoint=False)

        # Save coordinates
        np.save(os.path.join(tmp_dir, 'lat.npy'), lat)
        np.save(os.path.join(tmp_dir, 'lon.npy'), lon)

        # Create static variables
        static_file = os.path.join(tmp_dir, 'static', 'static.zarr')
        xr.Dataset(
            {'orography': (['latitude', 'longitude'], np.random.randn(nlat, nlon) * 1000)},
            coords={'latitude': lat, 'longitude': lon}
        ).to_zarr(static_file, mode='w')

        # Create static statistics
        xr.Dataset({
            'orography_mean': 500.0,
            'orography_std': 300.0
        }).to_zarr(os.path.join(tmp_dir, 'static', 'statistics.zarr'), mode='w')

        # Create main statistics
        xr.Dataset({
            'temperature_mean': 273.15,
            'temperature_std': 10.0,
            'geopotential_mean': 50000.0,
            'geopotential_std': 1000.0
        }).to_zarr(os.path.join(tmp_dir, 'statistics.zarr'), mode='w')

        # Create dummy train/val/test shards (6-hourly data)
        for split, num_shards in [('train', 2), ('val', 1), ('test', 1)]:
            for i in range(num_shards):
                shard_file = os.path.join(tmp_dir, split, f'2020_{i}.zarr')
                time = np.array([
                    np.datetime64('2020-01-01T00:00:00') + np.timedelta64(h*6, 'h')
                    for h in range(40)
                ])
                xr.Dataset(
                    {
                        'temperature': (['time', 'latitude', 'longitude'],
                                      np.random.randn(len(time), nlat, nlon) * 10 + 273.15),
                        'geopotential': (['time', 'latitude', 'longitude'],
                                       np.random.randn(len(time), nlat, nlon) * 1000 + 50000),
                    },
                    coords={'time': time, 'latitude': lat, 'longitude': lon}
                ).to_zarr(shard_file, mode='w')

        # Create climatology files
        clim_val = os.path.join(tmp_dir, 'climatology_val.zarr')
        clim_test = os.path.join(tmp_dir, 'climatology_test.zarr')
        for clim_file in [clim_val, clim_test]:
            xr.Dataset(
                {
                    'temperature': (['hour', 'dayofyear', 'latitude', 'longitude'],
                                  np.random.randn(4, 15, nlat, nlon) * 10 + 273.15),
                    'geopotential': (['hour', 'dayofyear', 'latitude', 'longitude'],
                                   np.random.randn(4, 15, nlat, nlon) * 1000 + 50000),
                },
                coords={'hour': [0, 6, 12, 18], 'dayofyear': np.arange(1, 16),
                       'latitude': lat, 'longitude': lon}
            ).to_zarr(clim_file, mode='w')

        print(f"Dummy data created in: {tmp_dir}\n")

        # Initialize datamodule
        print("Initializing GlobalForecastDataModule...")
        datamodule = GlobalForecastDataModule(
            root_dir=tmp_dir,
            climatology_val=clim_val,
            climatology_test=clim_test,
            in_variables=['temperature', 'geopotential'],
            static_variables=['orography'],
            out_variables=['temperature', 'geopotential'],
            predict_size=4,
            predict_step=1,
            history=[-1],
            hrs_each_step=6,
            batch_size=2,
            mem_load=1.0,
            num_workers=0,
            normalize_data=True,
            max_buffer_size=10
        )

        print(f"  History: {datamodule.get_history()}")
        print(f"  Hours per step: {datamodule.get_hrs_each_step()}")
        print(f"  Predict step size: {datamodule.get_predict_step_size()}")

        # Setup for training
        print("\nSetting up for training...")
        datamodule.setup(stage='fit')

        # Get dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Test training data
        print("\nIterating through training data...")
        for i, batch in enumerate(train_loader):
            input, static, output, clim, lead_times, in_vars, static_vars, out_vars, *_ = batch
            print(f"  Train batch {i+1}:")
            print(f"    input shape: {input.shape}")
            print(f"    static shape: {static.shape}")
            print(f"    output shape: {output.shape}")
            print(f"    climatology shape: {clim.shape if clim is not None else None}")
            print(f"    lead_times: {lead_times[0].tolist()}")
            if i >= 1:
                break

        # Test validation data
        print("\nIterating through validation data...")
        for i, batch in enumerate(val_loader):
            input, static, output, clim, lead_times, in_vars, static_vars, out_vars, *_ = batch
            print(f"  Val batch {i+1}:")
            print(f"    input shape: {input.shape}")
            print(f"    static shape: {static.shape}")
            print(f"    output shape: {output.shape}")
            print(f"    climatology shape: {clim.shape if clim is not None else None}")
            print(f"    lead_times: {lead_times[0].tolist()}")
            if i >= 1:
                break

        # Get coordinates
        lat, lon = datamodule.get_lat_lon()
        print(f"\nCoordinates:")
        print(f"  Latitude: {lat.shape} [{lat[0]:.1f} to {lat[-1]:.1f}]")
        print(f"  Longitude: {lon.shape} [{lon[0]:.1f} to {lon[-1]:.1f}]")

    finally:
        shutil.rmtree(tmp_dir)
        print(f"\nCleaned up: {tmp_dir}")
