import os
from typing import Tuple

import torch
import xarray as xr
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import lightning as L

WB_INPUT = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "temperature",
    "geopotential",
]

WB_TARGET = ["2m_temperature"]


class WeatherBench(Dataset):
    """
    A class used to handle the WeatherBench dataset, downloaded using the corresponding script.
    """

    def __init__(
        self,
        data_path: str = "/home/groups/ai/datasets/weather_forecasting/",
        var: str = "train",
        normalize=True,
    ):
        self.var = var
        self.normalize = normalize

        era5_slice, ifs_slice = self.get_split(self.var)
        era5_path = os.path.join(data_path, "era5.zarr")
        ifs_path = os.path.join(data_path, "ifs_ens.zarr")
        era5 = xr.open_zarr(era5_path, consolidated=False)
        self.land_sea_mask = era5["land_sea_mask"].to_numpy()
        self.land_sea_mask = np.where(self.land_sea_mask > 0.5, 1, 0)
        self.era5 = era5[WB_TARGET].sel(time=era5_slice)
        self.dataset = xr.open_zarr(ifs_path, consolidated=False)[WB_INPUT].sel(
            time=ifs_slice
        )
        # Create orography and discrete lat/lon grid
        lat_vals = self.dataset.latitude.values  # shape (160,)
        lon_vals = self.dataset.longitude.values  # shape (220,)
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)  # both shape
        # Map latitudes to row indices
        lat_indices = np.searchsorted(lat_vals, lat2d[:, 0])[:, None]  # shape (160, 1), broadcast later
        # Map longitudes to column indices
        lon_sort_idx = np.argsort(lon_vals)
        lon_vals_sorted = lon_vals[lon_sort_idx]
        lon2d_sorted = lon2d[:, lon_sort_idx]
        lon_indices = np.searchsorted(lon_vals_sorted, lon2d_sorted[0, :])[None, :]  # shape (1, 220), broadcast later
        # Broadcast to full 2D index grids
        lat_idx_grid = np.broadcast_to(lat_indices, lon2d.shape)  # shape (160, 220)
        lon_idx_grid = np.broadcast_to(lon_indices, lon2d.shape) 
        self.dataset["lat"] = xr.DataArray(lat_idx_grid, dims=["latitude", "longitude"])
        self.dataset["lon"] = xr.DataArray(lon_idx_grid, dims=["latitude", "longitude"])

        self.target_mean, self.target_std = self.get_statistics(data_path, data="era5")
        self.input_mean, self.input_std = self.get_statistics(data_path, data="ifs_ens")

        # Filter and prepare input data
        if self.var == "test":
            self.input_array = (
                self.dataset.to_array()
                .stack(pos=("latitude", "longitude"))
                .rename({"time": "sample"})
                .load()
            )
            self.target_array = (
                self.era5.to_array()
                .stack(pos=("latitude", "longitude"))
                .rename({"time": "sample"})
                .load()
            )
        else:
            self.input_array = (
                self.dataset.to_array()
                .where(self.land_sea_mask == 1)
                .stack(sample=("time", "latitude", "longitude"))
                .dropna(dim="sample")
                .load()
            )
            self.target_array = (
                self.era5.to_array()
                .where(self.land_sea_mask == 1)
                .stack(sample=("time", "latitude", "longitude"))
                .dropna(dim="sample")
                .load()
            )


        self.n_vars = len(WB_INPUT)

    @staticmethod
    def get_split(var: str) -> slice:
        if var == "train":
            ifs_range = pd.date_range(f"2018-01-01", f"2020-12-31T00", freq = "24h")
            era5_range = pd.date_range(f"2018-01-01T12", f"2020-12-31T12", freq = "24h")
        elif var == "val":
            ifs_range = pd.date_range(f"2021-01-01", f"2021-12-31T00", freq = "24h")
            era5_range = pd.date_range(f"2021-01-01T12", f"2021-12-31T12", freq = "24h")
        elif var == "test":
            ifs_range = pd.date_range(f"2022-01-01", f"2022-12-31T00", freq = "24h")
            era5_range = pd.date_range(f"2022-01-01T12", f"2022-12-31T12", freq = "24h")
        else:
            raise AssertionError(f"{var} is not in [train, val, test]")
        return era5_range, ifs_range

    @staticmethod
    def get_statistics(data_path: str, data="era5") -> Tuple:
        statistics = np.load(os.path.join(data_path, f"{data}_statistics.npy"))
        mean, std = np.split(statistics, 2, axis=-1)
        return mean.squeeze(), std.squeeze()

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return self.input_array.sizes["sample"]

    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """
        input = self.input_array[:-2, idx].values.copy()
        target = self.target_array[:,idx].values.copy()
        embedding = self.input_array[-2:, idx].values.copy()

        if self.normalize:
            if self.var == "test":
                input[:self.n_vars] = (input[:self.n_vars] - self.input_mean[:,None]) / self.input_std[:,None]
                target = (target - self.target_mean[0]) / self.target_std[0]
                input = input.T
                target = target.T
                embedding = embedding.T
            else:
                input[:self.n_vars] = (input[:self.n_vars] - self.input_mean) / self.input_std
                target = (target - self.target_mean[0]) / self.target_std[0]

        input_tensor = torch.from_numpy(input).float()
        target_tensor = torch.from_numpy(target).float()
        embedding_tensor = torch.from_numpy(embedding).to(torch.int32)

        return input_tensor, embedding_tensor, target_tensor
    

class WeatherBenchModule(L.LightningDataModule):
    def __init__(self, train_batch_size:int = 1024, num_workers:int = 0, data_path: str = "/home/groups/ai/datasets/weather_forecasting/", normalize=True):
        super().__init__()
        self.data_path = data_path
        self.normalize = normalize
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.val_batch_size = 4 * 160 * 220

        # Get datasets
        self.train_dataset = WeatherBench(data_path=self.data_path, var="train", normalize=self.normalize)
        self.val_dataset = WeatherBench(data_path=self.data_path, var="val", normalize=self.normalize)
        self.test_dataset = WeatherBench(data_path=self.data_path, var="test", normalize=self.normalize)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    dataset = WeatherBench(var="train")
    print(len(dataset))
    input, embedding, target = dataset.__getitem__(0)
    print(input.shape, embedding.shape, target.shape)
    print(embedding)