# Data Loading Guide

Guide for loading preprocessed ERA5 data for training and evaluation.

## Quick Start

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
preprocessed zarr files → GlobalForecastDataModule → DataLoader → Model
                              ↓
                    ZarrReader → Forecast → transforms
```

**GlobalForecastDataModule**: High-level interface that manages everything
- Automatically loads normalization statistics
- Creates train/val/test datasets
- Handles climatology integration for validation/test
- Returns PyTorch DataLoaders

**ZarrReader**: Low-level zarr file reader with distributed training support
- Splits data across GPUs and workers automatically
- Handles year grouping and temporal continuity

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

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `batch_size` | Samples per batch | Start with 32, adjust based on GPU memory |
| `num_workers` | DataLoader workers | 2-4 per GPU |
| `mem_load` | Memory loading (0.0-1.0) | 0.0 (minimal), 0.5 (balanced), 1.0 (all) |
| `pin_memory` | Pin memory for GPU | `True` when using GPU |

### Variables

```python
# Surface variables
in_variables = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure'
]

# Atmospheric variables (with pressure levels)
in_variables = [
    'geopotential_500',
    'temperature_850',
    'u_component_of_wind_500',
    'v_component_of_wind_850'
]

# Static variables
static_variables = [
    'orography',
    'land_sea_mask'
]

# Output variables (can differ from input)
out_variables = ['2m_temperature', 'geopotential_500']
```

## Common Patterns

### Training Setup

```python
# Training with shuffling, no climatology
datamodule = GlobalForecastDataModule(
    root_dir=data_dir,
    climatology_val=clim_file,
    climatology_test=clim_file,
    in_variables=in_vars,
    static_variables=static_vars,
    out_variables=out_vars,
    predict_size=112,       # 28 days at 6-hourly
    predict_step=6,
    batch_size=32,
    num_workers=4,
    normalize_data=True,
    max_buffer_size=1000    # Shuffle buffer for training
)
```

### Distributed Training

The datamodule automatically handles distributed training. Just initialize PyTorch Lightning:

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    devices=4,              # 4 GPUs
    strategy='ddp',
    max_epochs=100
)

trainer.fit(model, datamodule)
```

Data is automatically split across GPUs - no additional configuration needed.

### Denormalization

```python
# Get transforms for denormalization
out_transform = datamodule.get_transforms('out')

# During inference
predictions_normalized = model(input)
predictions = out_transform.denormalize(predictions_normalized)
```

### Accessing Coordinates

```python
lat, lon = datamodule.get_lat_lon()
# lat: array (121,) from 90 to -90
# lon: array (240,) from 0 to 360
```

## Data Directory Structure

After preprocessing, your data should be organized as:

```
era5_processed/
├── train/
│   ├── 1979_0.zarr
│   ├── 1979_1.zarr
│   └── ...
├── val/
│   ├── 2016_0.zarr
│   └── ...
├── test/
│   ├── 2017_0.zarr
│   └── ...
├── static/
│   ├── static.zarr
│   └── statistics.zarr
├── statistics.zarr
├── lat.npy
└── lon.npy
```

## Troubleshooting

### "Data slice not large enough"

**Problem**: Not enough timesteps for forecast + history window.

**Solution**:
- Reduce `num_workers` in DataLoader
- Increase `mem_load` to load larger chunks
- Reduce `predict_size` or shorten `history`

### "Mismatch between climatology and data coordinates"

**Problem**: Spatial grids don't align.

**Solution**: Re-preprocess climatology with same spatial resolution as data.

### Out of memory

**Problem**: GPU or RAM exhausted.

**Solution**:
- **GPU**: Reduce `batch_size`
- **CPU**: Reduce `mem_load` (try 0.2 or 0.0)
- **CPU**: Reduce `num_workers`

### Slow data loading

**Problem**: Long wait between batches.

**Solution**:
- Increase `mem_load` to load larger chunks
- Increase `num_workers` (typically 2-4 per GPU)
- Increase `batch_size` to amortize loading overhead

## Advanced: Lower-Level Components

For custom use cases, you can use the lower-level components directly:

```python
from s2s.utils.dataset import ZarrReader, Forecast
from s2s.utils.transforms import NormalizeDenormalize
from torch.utils.data import DataLoader
from s2s.utils.data_utils import collate_fn

# Create reader
reader = ZarrReader(
    file_list=train_files,
    static_variable_file=static_file,
    in_variables=in_vars,
    static_variables=static_vars,
    out_variables=out_vars,
    predict_size=112,
    predict_step=6,
    shuffle=True
)

# Create transforms
transforms = NormalizeDenormalize(mean=mean_values, std=std_values)

# Create dataset
dataset = Forecast(
    dataset=reader,
    normalize_data=True,
    in_transforms=transforms,
    static_transforms=static_transforms,
    output_transforms=transforms,
    mem_load=0.5
)

# Create dataloader
loader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)
```

See `datamodule.py` example (`python -m src.s2s.utils.datamodule`) for a working demonstration.

## See Also

- [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) - Data preprocessing pipeline
- [src/s2s/utils/datamodule.py](../src/s2s/utils/datamodule.py) - Main entry point with working example
- [src/s2s/utils/dataset.py](../src/s2s/utils/dataset.py) - Lower-level dataset components
- [src/s2s/utils/transforms.py](../src/s2s/utils/transforms.py) - Normalization utilities
