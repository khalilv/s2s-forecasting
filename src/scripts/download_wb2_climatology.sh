#!/bin/bash
#
# WeatherBench2 ERA5 Daily Climatology Download Script
#
# This script downloads ERA5 climatology from the WeatherBench2 Google Cloud Storage bucket.
# Climatology contains long-term means computed over some reference period
#
# Requirements:
#   - gsutil command-line tool (part of Google Cloud SDK)
#   - Sufficient disk space (~100s of GB)
#
# Usage:
#   1. Edit the configuration below to set your desired parameters
#   2. Run: bash download_wb2_climatology.sh
#
# The script will:
#   - Skip files that already exist (resume capability)
#   - Download metadata files first (.zattrs, .zgroup, .zmetadata)
#   - Download all surface and atmospheric climatology variables
#   - Exit with error if any download fails
#
# Reference: https://weatherbench2.readthedocs.io/
#
# =============================================================================

# Configuration
DATASET="era5-hourly-climatology/1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr"
SURFACE_VARIABLES=(2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure total_precipitation_6hr)
ATMOSPHERIC_VARIABLES=(geopotential u_component_of_wind v_component_of_wind temperature specific_humidity)
DIMENSIONS=(dayofyear latitude longitude level hour)
METADATA=(.zattrs .zgroup .zmetadata)

# IMPORTANT: Update this path to your desired output directory
ROOT="/glade/derecho/scratch/kvirji/DATA/"

# Validate gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "ERROR: gsutil not found. Please install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

DATA=("${SURFACE_VARIABLES[@]}" "${ATMOSPHERIC_VARIABLES[@]}" "${DIMENSIONS[@]}")

echo "============================================================================="
echo "WeatherBench2 ERA5 Daily Climatology Download"
echo "============================================================================="
echo "Dataset: ${DATASET}"
echo "Reference period: 1990-2017"
echo "Output directory: ${ROOT}/${DATASET}"
echo "Variables to download:"
echo "  Surface (${#SURFACE_VARIABLES[@]}): ${SURFACE_VARIABLES[*]}"
echo "  Atmospheric (${#ATMOSPHERIC_VARIABLES[@]}): ${ATMOSPHERIC_VARIABLES[*]}"
echo "  Dimensions (${#DIMENSIONS[@]}): ${DIMENSIONS[*]}"
echo "============================================================================="

# Create output directory
if [ ! -d "${ROOT}/${DATASET}" ]; then
    echo "Creating output directory: ${ROOT}/${DATASET}"
    mkdir -p "${ROOT}/${DATASET}"
fi

# Download metadata files
echo ""
echo "Step 1/2: Downloading metadata files..."
for file in "${METADATA[@]}"; do
    if [ ! -f "${ROOT}/${DATASET}/${file}" ]; then
        echo "  Downloading ${file}..."
        if ! gsutil -m cp "gs://weatherbench2/datasets/${DATASET}/${file}" "${ROOT}/${DATASET}/${file}"; then
            echo "ERROR: Failed to download ${file}"
            exit 1
        fi
    else
        echo "  ${file} already exists, skipping"
    fi
done

# Download data variables
echo ""
echo "Step 2/2: Downloading data variables and dimensions..."
total_vars=${#DATA[@]}
current=0
for variable in "${DATA[@]}"; do
    current=$((current + 1))
    echo "  [${current}/${total_vars}] Downloading ${variable}..."
    if ! gsutil -m cp -r -n "gs://weatherbench2/datasets/${DATASET}/${variable}/" "${ROOT}/${DATASET}/"; then
        echo "ERROR: Failed to download ${variable}"
        exit 1
    fi
done

echo ""
echo "============================================================================="
echo "Download complete!"
echo "Data saved to: ${ROOT}/${DATASET}"
echo ""
echo "Next steps:"
echo "  1. Verify the data: Check ${ROOT}/${DATASET} directory"
echo "  2. Preprocess the data: Run src/s2s/utils/preprocess_era5_climatology.py"
echo "============================================================================="
