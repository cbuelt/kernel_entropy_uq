# Script to download the ERA5 dataset from WeatherBench2. The script filters the data to the Europe grid and the specified date range.

#import apache_beam 
import weatherbench2
import xarray as xr
import gcsfs
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
    # Set parameters
    date_range = pd.date_range(f"2010-01-01", f"2022-12-31T18", freq = "6h")
    path = "/home/groups/ai/datasets/weather_forecasting/"
    # # Europe grid
    lat_range = np.arange(35, 75, 0.25)
    lon_range = np.append(np.arange(347.5,360, 0.25),np.arange(0, 42.5,0.25))

    # ERA 5
    # era5_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
    # variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "temperature", "geopotential", "land_sea_mask", "geopotential_at_surface"]
    # filename = "era5"

    # # Open file from weatherbench and filter
    # data = xr.open_zarr(era5_path)
    # data_reduced = data[variables].sel(latitude = lat_range, longitude = lon_range, method = "nearest").sel(time = date_range, method = "nearest")
    # data_reduced["geopotential"] = data_reduced["geopotential"].sel(level = 500)
    # data_reduced["temperature"] = data_reduced["temperature"].sel(level = 850)
    # # Rechunk and save
    # data_reduced.chunk("auto").to_zarr(path + filename+".zarr", zarr_format = 2, consolidated = False)

    # # Save statistics
    # statistics = np.zeros((len(variables), 2))
    # for i,var in enumerate(variables):
    #     if var == "land_sea_mask":
    #         statistics[i,0] = 0
    #         statistics[i,1] = 1
    #     else:
    #         statistics[i,0] = data_reduced[var].mean().compute()
    #         statistics[i,1] = data_reduced[var].std().compute()

    # np.save(path + "era5_statistics.npy", statistics)

    # data_reduced.close()
    # data.close()


    # HRES
    ifs_ens_path = 'gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr'
    date_range = pd.date_range(f"2018-01-01", f"2022-12-31T06", freq = "24h")
    variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "temperature", "geopotential"]
    filename = "ifs_ens"

    # Open file from weatherbench and filter (12h prediction horizon)
    data = xr.open_zarr(ifs_ens_path)
    data_reduced = data[variables].sel(latitude = lat_range, longitude = lon_range, method = "nearest").isel(prediction_timedelta = 2)
    data_reduced["geopotential"] = data_reduced["geopotential"].sel(level = 500)
    data_reduced["temperature"] = data_reduced["temperature"].sel(level = 850)

    # Rechunk and save
    data_reduced.drop_vars(["level", "prediction_timedelta"]).chunk("auto").to_zarr(path + filename+".zarr", zarr_format = 2, consolidated = False)

    # Save statistics
    statistics = np.zeros((len(variables), 2))
    for i,var in enumerate(variables):
        statistics[i,0] = data_reduced[var].mean().compute()
        statistics[i,1] = data_reduced[var].std().compute()

    np.save(path + "ifs_ens_statistics.npy", statistics)

    data_reduced.close()
    data.close()
