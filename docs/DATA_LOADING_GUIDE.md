# Data Loading Guide

Guide for loading preprocessed ERA5 data for training and evaluation.

## Example Usage

The `GlobalForecastDataModule` is the main interface for loading data:

```python
from s2s.utils.datamodule import GlobalForecastDataModule

# Initialize datamodule
datamodule = GlobalForecastDataModule(
    root_dir='/path/to/era5_processed',
    climatology_val='/path/to/climatology_val.zarr',
    climatology_test='/path/to/climatology_test.zarr',
    in_variables=['2m_temperature', 'geopotential', 'u_component_of_wind'],
    static_variables=['orography', 'land_sea_mask'],
    out_variables=['2m_temperature', 'geopotential'],
    predict_size=28*4,      # 28 days * 4 timesteps/day
    predict_step=6,         # Forecast every 6 hours
    history=[-6, -12],      # Include 6 and 12 hours ago
    hrs_each_step=1,        # Data is hourly
    batch_size=32,
    num_workers=4,
    normalize_data=True,
    mem_load=0.5            # Load 50% of data at once
)

# Setup for training
datamodule.setup(stage='fit')

# Get dataloaders
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# Iterate through batches
for batch in train_loader:
    input, static, output, climatology, lead_times, *metadata = batch
    # input: (batch, time, variables, lat, lon)
    # output: (batch, time, variables, lat, lon)
    # climatology: (batch, time, variables, lat, lon) or None
```

## Architecture

```
Data Preprocessing → GlobalForecastDataModule → DataLoaders -> Model
                                ↓
                    ZarrReader → Forecast → Transforms
```

**GlobalForecastDataModule**: High-level interface for data loading
- Loads normalization statistics
- Creates train/val/test datasets
- Handles climatology integration for validation/test sets
- Returns PyTorch DataLoaders

**ZarrReader**: Low-level zarr file reader with distributed training support
- Splits data across GPUs and workers automatically

**Forecast**: Generates input/output pairs with optional normalization
- Memory-efficient loading via sharding
- Aligns climatology with forecast timestamps

## Key Parameters

### Forecast Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `predict_size` | Number of forecast timesteps | `28*4` (28 days at 6-hourly) |
| `predict_step` | Spacing between outputs | `6` (every 6 hours) |
| `history` | Past timesteps to include | `[-6, -12]` (6 and 12 hours ago) |
| `hrs_each_step` | Hours between data timesteps | `1` (hourly data) |

### Performance

| Parameter | Description | Example |
|-----------|-------------|----------------|
| `batch_size` | Samples per batch | `32` (Adjust for memory constraints) |
| `num_workers` | DataLoader workers | `1` (Adjust for compute constraints and dataset size) |
| `mem_load` | Percentage of worker data slice to load into memory at a time (0.0-1.0) | `0.0` (smallest memory footprint, slowest), `1.0` (largest memory footprint, fastest) |
| `pin_memory` | Speed up CPU -> GPU transfer | `True` (Adjust for memory constraints) |

### Example Variables

```python
# Surface variables
in_variables = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
    ...
]

# Atmospheric variables (with pressure levels)
in_variables = [
    'geopotential_500',
    'temperature_850',
    'u_component_of_wind_500',
    'v_component_of_wind_850',
    ...
]

# Static variables
static_variables = [
    'orography',
    'land_sea_mask',
    ...
]

# Output variables (can differ from input but running the model autoregressively will not be possible)
out_variables = ['2m_temperature', 'geopotential_500', ...]
```

## See Also

- [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) - Data preprocessing pipeline
- [src/s2s/utils/datamodule.py](../src/s2s/utils/datamodule.py) - Main entry point with working example
- [src/s2s/utils/dataset.py](../src/s2s/utils/dataset.py) - Lower-level dataset components
- [src/s2s/utils/transforms.py](../src/s2s/utils/transforms.py) - Normalization utilities with working example
