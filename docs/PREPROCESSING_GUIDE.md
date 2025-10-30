# ERA5 Data Preprocessing Guide

This guide explains how to download and preprocess ERA5 data from WeatherBench2 for use in the s2s-forecasting project.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Step-by-Step Instructions](#step-by-step-instructions)
5. [Output Structure](#output-structure)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Overview

The preprocessing pipeline consists of four main steps:

```
1. Download ERA5 daily data         → download_wb2.sh
2. Download ERA5 climatology        → download_wb2_climatology.sh
3. Preprocess ERA5 data             → preprocess_era5.py
4. Preprocess climatology           → preprocess_era5_climatology.py
```

## Prerequisites

### 1. Install Google Cloud SDK (for gsutil)

```bash
# Verify with:
gsutil --version
```

### 2. Python Environment

Ensure you have the required conda environment activated:
```bash
conda activate s2s
```

### 3. Disk Space

Ensure you have sufficient disk space for:
- Raw ERA5 daily data
- Raw ERA5 climatology
- Processed datasets

## Quick Start

```bash
# 1. Download ERA5 daily data
bash src/scripts/download_wb2.sh

# 2. Download ERA5 climatology
bash src/scripts/download_wb2_climatology.sh

# 3. Preprocess ERA5 data
python src/s2s/utils/preprocess_era5.py \
    --root_dir /path/to/era5_daily/dataset.zarr \
    --save_dir /path/to/output/era5_processed \
    --start_train_year 1979 \
    --start_val_year 2016 \
    --start_test_year 2017 \
    --end_year 2023 \
    --num_shards 8 \
    --dask_tmp_dir /path/to/tmp

# 4. Preprocess climatology
python src/s2s/utils/preprocess_era5_climatology.py \
    --root_dir /path/to/era5_climatology/dataset.zarr \
    --save_dir /path/to/output/climatology.zarr \
    --dask_tmp_dir /path/to/tmp
```

## Step-by-Step Instructions

### Step 1: Download ERA5 Daily Data

The ERA5 daily data contains hourly atmospheric and surface variables from 1959-2023.

#### 1.1 Configure Output Directory

Edit `src/scripts/download_wb2.sh`:
```bash
# Change this line to your desired location:
ROOT="/path/to/your/data/era5_daily"
```

#### 1.2 Run Download Script

```bash
bash src/scripts/download_wb2.sh
```

**What it downloads:**
- **Surface variables:** 2m temperature, 10m winds, pressure, precipitation, etc.
- **Atmospheric variables:** Geopotential, winds, temperature, humidity at pressure levels
- **Dimensions:** Latitude, longitude, time, level
- **Metadata:** Zarr metadata files

**Progress:** The script will show progress for each variable. If interrupted, simply re-run it - it will resume from where it left off.

**Expected output location:**
```
/path/to/your/data/era5_daily/
└── 1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr/
    ├── .zattrs
    ├── .zgroup
    ├── .zmetadata
    ├── 2m_temperature/
    ├── 10m_u_component_of_wind/
    ├── ...
    └── latitude/
```

### Step 2: Download ERA5 Climatology

The climatology contains long-term daily means (1990-2017) used for computing anomalies.

#### 2.1 Configure Output Directory

Edit `src/scripts/download_wb2_climatology.sh`:
```bash
# Change this line to your desired location:
ROOT="/path/to/your/data/era5_climatology"
```

#### 2.2 Run Download Script

```bash
bash src/scripts/download_wb2_climatology.sh
```

**Expected output location:**
```
/path/to/your/data/era5_climatology/
└── 1990-2017-daily_clim_daily_mean_61_dw_240x121_equiangular_with_poles_conservative.zarr/
    ├── .zattrs
    ├── 2m_temperature/
    ├── geopotential/
    └── ...
```

### Step 3: Preprocess ERA5 Data

The preprocessing script converts raw ERA5 data into sharded zarr files suitable for training ML models.

#### 3.1 Basic Usage

```bash
python src/s2s/utils/preprocess_era5.py \
    --root_dir /path/to/era5_daily/dataset.zarr \
    --save_dir /path/to/output/era5_processed \
    --start_train_year 1979 \
    --start_val_year 2016 \
    --start_test_year 2017 \
    --end_year 2023 \
    --num_shards 8
```

#### 3.2 Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--root_dir` | Input zarr dataset path | *required* |
| `--save_dir` | Output directory | *required* |
| `--start_train_year` | First training year | 1979 |
| `--start_val_year` | First validation year | 2016 |
| `--start_test_year` | First test year | 2017 |
| `--end_year` | Last year (exclusive) | 2019 |
| `--num_shards` | Shards per year | 8 |
| `--hrs_per_step` | Hours per timestep | 1 |
| `--dask_tmp_dir` | Dask temp directory | system tmp |

#### 3.3 What It Does

1. **Processes static variables:** Orography, land-sea mask, soil type
2. **Processes time-varying variables:** Surface and atmospheric variables
3. **Shards data by year:** Splits each year into multiple shards for memory efficiency
4. **Computes statistics:** Mean and standard deviation for normalization (training data only)
5. **Saves coordinates:** Latitude and longitude arrays

#### 3.4 Expected Output

```
/path/to/output/era5_processed/
├── static/
│   ├── static.zarr              # Static fields
│   └── statistics.zarr          # Static field statistics
├── train/
│   ├── 1979_0.zarr
│   ├── 1979_1.zarr
│   ├── ...
│   └── 2015_7.zarr
├── val/
│   ├── 2016_0.zarr
│   └── ...
├── test/
│   ├── 2017_0.zarr
│   └── ...
├── statistics.zarr              # Normalization statistics
├── lat.npy                      # Latitude coordinates
└── lon.npy                      # Longitude coordinates
```

### Step 4: Preprocess Climatology

The climatology preprocessing script **automatically detects** the temporal frequency of your climatology data and handles it appropriately.

#### 4.1 Supported Temporal Frequencies

The script supports three types of climatology:

| Type | Dimensions | Use Case |
|------|-----------|----------|
| **Daily** | (dayofyear, lat, lon) | Daily climatological means |
| **6-hourly** | (hour, dayofyear, lat, lon) | 4 synoptic times per day (0, 6, 12, 18 UTC) |
| **Hourly** | (hour, dayofyear, lat, lon) | 24 hours per day |

The script will automatically detect which type you have based on the presence and size of the `hour` dimension.

#### 4.2 Basic Usage

```bash
python src/s2s/utils/preprocess_era5_climatology.py \
    --root_dir /path/to/era5_climatology/dataset.zarr \
    --save_dir /path/to/output/climatology_processed/climatology.zarr
```

**Example with daily climatology:**
```bash
python src/s2s/utils/preprocess_era5_climatology.py \
    --root_dir /path/to/data/era5_climatology/daily_clim.zarr \
    --save_dir /path/to/output/climatology_processed/daily_clim.zarr
```

**Example with 6-hourly climatology:**
```bash
python src/s2s/utils/preprocess_era5_climatology.py \
    --root_dir /path/to/data/era5_climatology/6hourly_clim.zarr \
    --save_dir /path/to/output/climatology_processed/6hourly_clim.zarr
```

#### 4.3 Detection Output

When you run the script, it will automatically detect and log the temporal frequency:

```
INFO - Detected daily climatology (no hour dimension)
```

or

```
INFO - Detected 6-hourly climatology (hour dimension with 4 timesteps)
```

or

```
INFO - Detected hourly climatology (hour dimension with 24 timesteps)
```

#### 4.4 Expected Output

```
/path/to/output/climatology_processed/
└── climatology.zarr             # Processed climatology with consistent dimensions
```

The output will have dimensions appropriate to your input:
- **Daily**: `(dayofyear: 366, lat: 121, lon: 240)`
- **6-hourly**: `(hour: 4, dayofyear: 366, lat: 121, lon: 240)`
- **Hourly**: `(hour: 24, dayofyear: 366, lat: 121, lon: 240)`

## Output Structure

After preprocessing, your data will be organized as follows:

```
DATA/
├── era5_daily/                                    # Raw ERA5 daily data
│   └── 1959-2023_01_10-1h-240x121_...zarr/
│
├── era5_climatology/                              # Raw ERA5 climatology
│   └── 1990-2017-daily_clim_...zarr/
│
├── era5_processed/                                # Processed ERA5 data
│   ├── static/
│   │   ├── static.zarr
│   │   └── statistics.zarr
│   ├── train/
│   │   ├── {year}_{shard}.zarr  (1979-2015)
│   ├── val/
│   │   ├── {year}_{shard}.zarr  (2016)
│   ├── test/
│   │   ├── {year}_{shard}.zarr  (2017-2022)
│   ├── statistics.zarr
│   ├── lat.npy
│   └── lon.npy
│
└── era5_climatology_processed/                    # Processed climatology
    └── climatology.zarr
```