#!/usr/bin/env python
'''
2m_temperature
total_precipitation
10m_u_component_of_wind
10m_v_component_of_wind
'''        
import os
import cdsapi
import logging

# Initialize the client
client = cdsapi.Client()

# Error handling and logging function
def download_data(year, variable, prefix):
    try:
        filename = f'{prefix}_{year}.nc'
        if not os.path.exists(filename):
            dataset = "reanalysis-era5-single-levels"
            request = {
                'product_type': ['reanalysis'],
                'variable': [variable],
                'year': [str(year)],
                'month': [str(m).zfill(2) for m in range(1,13)],
                'day': [str(d).zfill(2) for d in range(1,32)],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
                '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00',
                '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                'data_format': 'netcdf',
                'download_format': 'unarchived'
            }
            client.retrieve(dataset, request, filename)
            logging.info(f"Downloaded: {filename}")
        else:
            logging.info(f"File already exists: {filename}")
    except Exception as e:
        logging.error(f"Failed to download data for {year}: {e}")

# ----------- Specify parameters here -----------------
variable_to_download = '2m_temperature'
prefix = 't2m'
data_dir = f'/glade/derecho/scratch/kvirji/s2s.dir/era5/{prefix}/'
# -----------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)

#Main loop
for year in range(1979, 2024):
    download_data(year, variable_to_download, prefix)
