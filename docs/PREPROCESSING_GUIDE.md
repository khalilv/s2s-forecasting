# ERA5 Preprocessing Guide

Guide for downloading and preprocessing ERA5 data from WeatherBench2 for forecasting.

## Prerequisites

```bash
# Install Google Cloud SDK (for gsutil)
gsutil --version

# Activate conda environment for dependencies
conda activate s2s
```

Ensure sufficient disk space for raw and processed data.

## Step 1: Download ERA5 Daily Data

Edit output directory and required variables in `src/scripts/download_wb2.sh`. Then run:

```bash
bash src/scripts/download_wb2.sh
```

## Step 2: Download Climatology

Edit output directory and required variables in `src/scripts/download_wb2_climatology.sh`. Then run:

```bash
bash src/scripts/download_wb2_climatology.sh
```

## Step 3: Preprocess ERA5 Data

Converts raw ERA5 into sharded zarr files for efficient training.

```bash
python src/s2s/utils/preprocess_era5.py \
    --root_dir /path/to/era5_daily/dataset.zarr \
    --save_dir /path/to/era5_processed \
    --start_train_year 1979 \
    --start_val_year 2016 \
    --start_test_year 2017 \
    --end_year 2023 \
    --num_shards 8 \
    --hrs_per_step 1
```
### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--root_dir` | Input zarr dataset | *required* |
| `--save_dir` | Output directory | *required* |
| `--start_train_year` | First training year | 1979 |
| `--start_val_year` | First validation year | 2016 |
| `--start_test_year` | First test year | 2017 |
| `--end_year` | Last year (exclusive) | 2019 |
| `--num_shards` | Shards per year | 8 |
| `--hrs_per_step` | Hours per timestep | 1 |

### What It Does

1. Processes static variables (orography, land-sea mask, soil type)
2. Processes time-varying variables with train/val/test splits
3. Shards data by year for memory efficient loading
4. Computes normalization statistics (over training data only)
5. Saves coordinates (lat, lon)

### Output Structure

```
era5_processed/
├── static/
│   ├── static.zarr
│   └── statistics.zarr
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
├── statistics.zarr       # Normalization stats
├── lat.npy               # Latitude array
└── lon.npy               # Longitude array
```

## Step 4: Preprocess Climatology

Processes raw climatology file.

```bash
python src/s2s/utils/preprocess_era5_climatology.py \
    --root_dir /path/to/era5_climatology/dataset.zarr \
    --save_dir /path/to/climatology.zarr
```

### Supported Frequencies

| Type | Dimensions | Detection |
|------|-----------|-----------|
| Daily | (dayofyear, lat, lon) | No hour dimension |
| 6-hourly | (hour, dayofyear, lat, lon) | 4 hours (0, 6, 12, 18) |
| Hourly | (hour, dayofyear, lat, lon) | 24 hours |

### Output

```
climatology_processed/
└── climatology.zarr      # Processed climatology
```

Output dimensions:
- **Daily**: (dayofyear: 366, lat, lon)
- **6-hourly**: (hour: 4, dayofyear: 366, lat, lon)
- **Hourly**: (hour: 24, dayofyear: 366, lat, lon)

## Complete Directory Structure

After preprocessing:

```
DATA/
├── era5_daily/                    # Raw daily data
│   └── *.zarr/
├── era5_climatology/              # Raw climatology
│   └── climatology_*.zarr/
├── era5_processed/                # Processed data (use for training)
│   ├── static/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── statistics.zarr
│   ├── lat.npy
│   └── lon.npy
└── climatology_processed/         # Processed climatology
    └── climatology.zarr
```

## Next Steps

After preprocessing, use the data with `GlobalForecastDataModule`:

See [DATA_LOADING_GUIDE.md](DATA_LOADING_GUIDE.md) for detailed usage.

## See Also

- [src/scripts/download_wb2.sh](../src/scripts/download_wb2.sh) - ERA5 download script
- [src/scripts/download_wb2_climatology.sh](../src/scripts/download_wb2_climatology.sh) - Climatology download script
- [src/s2s/utils/preprocess_era5.py](../src/s2s/utils/preprocess_era5.py) - Main preprocessing script
- [src/s2s/utils/preprocess_era5_climatology.py](../src/s2s/utils/preprocess_era5_climatology.py) - Climatology preprocessing
