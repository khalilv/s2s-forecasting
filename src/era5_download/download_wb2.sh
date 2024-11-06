VARIABLES=(2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure total_precipitation_6hr geopotential u_component_of_wind v_component_of_wind temperature relative_humidity specific_humidity land_sea_mask soil_type geopotential_at_surface)
DATASET="1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
FILES=(.zattrs .zgroup .zmetadata)
ROOT="/glade/derecho/scratch/kvirji"

if [ ! -d "${ROOT}/${DATASET}" ]; then
    mkdir -p "${ROOT}/${DATASET}"
fi
for file in ${FILES[@]}; do
    if [ ! -f "${ROOT}/${DATASET}/${file}" ]; then
        gsutil -m cp "gs://weatherbench2/datasets/era5/${DATASET}/${file}" "${ROOT}/${DATASET}/${file}"
    else
        echo "${file} already exists."
    fi
done
for variable in ${VARIABLES[@]}; do
    if [ ! -d "${ROOT}/${DATASET}/${variable}" ]; then
        mkdir -p "${ROOT}/${DATASET}/${variable}"
    fi
    # list files in the remote directory
    remote_files=$(gsutil ls "gs://weatherbench2/datasets/era5/${DATASET}/${variable}/")
    
    #copy each file if they do not exist
    for remote_file in $remote_files; do
        local_file="${ROOT}/${DATASET}/${variable}/$(basename $remote_file)"
        
        if [ ! -f "$local_file" ]; then
            gsutil -m cp "$remote_file" "$local_file"
        else
            echo "${remote_file} already exists."
        fi
    done
done
