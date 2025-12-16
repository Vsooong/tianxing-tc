import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import torch
import pandas as pd

WeatherModel = None  # not open-sourced
ordering = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "r50",
    "r100",
    "r150",
    "r200",
    "r250",
    "r300",
    "r400",
    "r500",
    "r600",
    "r700",
    "r850",
    "r925",
    "r1000",
    # "sst",
]


def extract_field_tx(
    ds=None,
    filename=".../tx_2021-07-18 18:00:00.nc",
    initial_lon=132.1,
    initial_lat=23.7,
    forecast_steps=28,
):

    data = xr.open_dataarray(filename)
    data = data.sel(step=slice(1, forecast_steps))

    if initial_lon < 0:
        initial_lon = 360 + initial_lon
    track_lons = np.array([initial_lon])
    track_lats = np.array([initial_lat])

    lons = data.lon
    lats = data.lat
    search_radius = 4  # 4 degree
    last_lon = initial_lon
    last_lat = initial_lat
    print(
        "Tracking for cyclone:", data.coords["time"].values[0], initial_lon, initial_lat
    )

    for step in range(1, len(data.step) + 1):
        # print("Step:", step)
        sub_msl = data.sel(
            variable="msl",
            lon=slice(last_lon - search_radius, last_lon + search_radius),
            lat=slice(last_lat + search_radius, last_lat - search_radius),
            step=step,
        )

        min_msl = sub_msl.min().values
        min_msl_lon = sub_msl.where(sub_msl == min_msl, drop=True).lon.values[0]
        min_msl_lat = sub_msl.where(sub_msl == min_msl, drop=True).lat.values[0]
        track_lons = np.append(track_lons, min_msl_lon)
        track_lats = np.append(track_lats, min_msl_lat)
        print(min_msl_lon, min_msl_lat)
        last_lon = min_msl_lon
        last_lat = min_msl_lat

        # subtract the field center (size 51x51) at track location
        lon_idx = int(abs(lons - min_msl_lon).argmin())
        lat_idx = int(abs(lats - min_msl_lat).argmin())
        lon_slice = slice(max(0, lon_idx - 25), min(len(lons), lon_idx + 25 + 1))
        lat_slice = slice(max(0, lat_idx - 25), min(len(lats), lat_idx + 25 + 1))

        data_tensor = (
            data.isel(lon=lon_slice, lat=lat_slice).sel(step=step).squeeze(dim="time")
        )
        # padding the last dim of  field to 51 if necessary
        if data_tensor.shape[-1] != 51:
            pad = 51 - data_tensor.shape[-1]
            data_tensor = np.pad(data_tensor, ((0, 0), (0, 0), (0, pad)), "constant")
        if data_tensor.shape[-2] != 51:
            pad = 51 - data_tensor.shape[-2]
            data_tensor = np.pad(data_tensor, ((0, 0), (0, pad), (0, 0)), "constant")

        # track_field[step-1] = data_tensor

    track_lons[0] = initial_lon
    track_lats[0] = initial_lat

    return track_lons, track_lats


def plot_track(lons, lats):
    # 创建绘图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(
        [min(lons) - 5, max(lons) + 5, min(lats) - 5, max(lats) + 5],
        crs=ccrs.PlateCarree(),
    )

    # 添加地图特征
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="bisque")
    ax.add_feature(cfeature.OCEAN, edgecolor="none", facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # 绘制预报台风路径
    ax.plot(
        lons,
        lats,
        marker="o",
        color="b",
        linestyle="-",
        markersize=5,
        label="Forecast Track",
    )
    ax.scatter(
        lons[0], lats[0], color="red", zorder=5, label="Start Position"
    )  # 起始点

    # 添加网格线和标签
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    plt.legend(loc="upper left")
    plt.show()


def save_pred(
    preds, time, steps: list, variable, lat, lon, model_name, num_steps, save_dir
):
    os.makedirs(save_dir, exist_ok=True)
    assert len(preds) == len(steps)
    init_time = pd.to_datetime(time)
    ds = xr.DataArray(
        preds[None],
        dims=["time", "step", "variable", "lat", "lon"],
        coords=dict(
            time=[init_time],
            step=steps,
            variable=variable,
            lat=lat,
            lon=lon,
        ),
    ).astype(np.float32)
    save_name = os.path.join(save_dir, f"{model_name}_{init_time}_{num_steps}.nc")
    ds.to_netcdf(save_name, mode="w")


def generate_prediction(time="2021-07-18 18:00:00"):
    if isinstance(time, str):
        time = pd.to_datetime(time)

    # dataset = Fuxi_dataset_month_nc()
    dataset = weather_dataset  # not open-sourced
    idx = dataset.get_idx(time)
    ds, timestamp = dataset[idx]
    print(timestamp, ds.shape)

    ds = ds[np.newaxis, ...]
    device = torch.device(f"cuda:5")
    fuxi = WeatherModel(device=device)
    outputs = fuxi.run_steps(ds, timestamp, num_steps=28)
    file_path = f"/data3/WangGuanSong/TianXing/tropical cyclone/pictures"
    save_pred(
        outputs,
        timestamp,
        steps=range(1, 29),
        variable=ordering,
        lat=np.linspace(90, -90, 721),
        lon=np.linspace(0, 359.75, 1440),
        model_name="fuxi",
        save_dir=file_path,
    )