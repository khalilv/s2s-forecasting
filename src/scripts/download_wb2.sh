STATIC_VARIABLES=(land_sea_mask soil_type geopotential_at_surface)
SURFACE_VARIABLES=(2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure total_precipitation_6hr)
ATMOSPHERIC_VARIABLES=(geopotential u_component_of_wind v_component_of_wind temperature relative_humidity specific_humidity)
DIMENSIONS=(latitude longitude time level)
METADATA=(.zattrs .zgroup .zmetadata)
DATASET="1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
ROOT="/glade/derecho/scratch/kvirji/DATA/era5"

DATA=("${STATIC_VARIABLES[@]}" "${SURFACE_VARIABLES[@]}" "${ATMOSPHERIC_VARIABLES[@]}" "${DIMENSIONS[@]}")

if [ ! -d "${ROOT}/${DATASET}" ]; then
    mkdir -p "${ROOT}/${DATASET}"
fi

# Add error handling for gsutil commands
for file in "${METADATA[@]}"; do
    if [ ! -f "${ROOT}/${DATASET}/${file}" ]; then
        if ! gsutil -m cp "gs://weatherbench2/datasets/era5/${DATASET}/${file}" "${ROOT}/${DATASET}/${file}"; then
            echo "Error downloading ${file}"
            exit 1
        fi
    else
        echo "${file} already exists."
    fi
done

for variable in "${DATA[@]}"; do
    if ! gsutil -m cp -r -n "gs://weatherbench2/datasets/era5/${DATASET}/${variable}/" "${ROOT}/${DATASET}/"; then
        echo "Error downloading ${variable}"
        exit 1
    fi
done
