import sys
from generate_per import SphericalGaussian, Brown
from math import ceil
import torch
import gc
import pandas as pd
import numpy as np
import datetime as dt
import xarray as xr
from track_field import WeatherModel, ordering
import h5py


def add_perturbation(
    input_x,
    method="SphericalGaussian",
    noise_amplitude=0.15,
    tau=3,
    en_num=14,
    batch_size=7,
):
    if en_num == 1:
        return input_x
    nensemble = en_num - 1
    assert isinstance(input_x, torch.Tensor)
    perturbation_dict = {
        "SphericalGaussian": SphericalGaussian(
            noise_amplitude=noise_amplitude, alpha=2, tau=tau, sigma=None
        ),
        "Brown": Brown(noise_amplitude=noise_amplitude, reddening=1.0),
    }
    perturbation = perturbation_dict[method]
    if batch_size is None:
        batch_size = nensemble
    number_of_batches = ceil(nensemble / batch_size)
    ensemble_index = 0
    all_perts = [input_x]
    channel_mask = torch.ones(size=(1, 73, 1, 1))
    channel_mask[
        :, [0, 1, 2, 3, 5, 7, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72], :, :
    ] = 0
    for i in range(number_of_batches):
        ensemble_index = i * batch_size
        mini_batch_size = min(batch_size, nensemble - ensemble_index)
        batch_input = input_x.repeat(mini_batch_size, 1, 1, 1)
        perturbed_field = perturbation(batch_input, channel_mask)
        all_perts.append(perturbed_field)
        gc.collect()

    perts = torch.cat(all_perts, dim=0)
    return perts


def Running_Ensemble_Inference(
    tc_id="Sally",
    timestamp=None,
    input_path=None,
    label_lon=[],
    label_lat=[],
    device_id=1,
    nensemble=15,
):

    model_wrapper = WeatherModel
    if input_path is not None:
        all_fields = model_wrapper.load_input_from_given_nc(input_path)
        all_fields = model_wrapper.normalise(all_fields)
    elif timestamp is not None:
        all_fields = model_wrapper.load_input_from_arxiv(timestamp, return_all=False)
        all_fields = model_wrapper.normalise(all_fields)

    else:
        raise ValueError("timestamp or input_path should be provided")

    input_field = all_fields.to(f"cuda:{device_id}")

    perts = add_perturbation(
        input_field,
        method="SphericalGaussian",
        noise_amplitude=0.15,
        tau=50,
        en_num=nensemble,
        batch_size=5,
    )
    # perts = perts.to("cpu")
    # print(perts.shape)
    times = 16
    save_steps = range(1, times + 1)
    lats = np.linspace(90, -90, 721).astype(np.float32)
    lons = np.linspace(0, 359.75, 1440).astype(np.float32)
    steps = np.arange(0, 17)

    start_lon = label_lon[0]
    start_lat = label_lat[0]

    preds = model_wrapper.forward_trajectory_new(
        perts, times=16, save_steps=save_steps, sub_batch_size=2
    )
    print(preds.shape)  # (nensemble, 16, 73, 721, 1440)
    # return
    input_field = input_field.cpu().numpy()
    # print(outputs.shape)
    for i in range(nensemble):
        output = np.concatenate([input_field, preds[i]], axis=0)
        pred_global = xr.DataArray(
            output[None],
            dims=["time", "step", "variable", "lat", "lon"],
            coords=dict(
                time=[timestamp],
                step=steps,
                variable=ordering,
                lat=lats,
                lon=lons,
            ),
        ).astype(np.float32)
        track_field, track_lons, track_lats = extract_field(
            pred_global, initial_lon=start_lon, initial_lat=start_lat
        )
        save_path = f".../{tc_id}_{i}_{timestamp}.h5"
        with h5py.File(save_path, "w") as f:
            f.create_dataset("track_field", data=track_field)
            f.create_dataset("track_lons", data=track_lons)
            f.create_dataset("track_lats", data=track_lats)
            f.create_dataset("init_time", data=str(timestamp))
            f.create_dataset("ID", data=tc_id)

    return


def plot_perturbation():
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    input_x = torch.randn(1, 73, 721, 1440).to(device)
    perts = add_perturbation(
        input_x, method="SphericalGaussian", noise_amplitude=0.15, en_num=2, tau=49
    )
    perts = perts - input_x
    perts = perts.to("cpu")
    print(perts.shape)
    lats = np.linspace(90, -90, 721).astype(np.float32)
    lons = np.linspace(0, 359.75, 1440).astype(np.float32)
    # for i in range(20,22):
    plt.figure(figsize=(10, 10))
    ax = plt.axes(
        projection=ccrs.Orthographic(central_longitude=104, central_latitude=24)
    )
    i = 9
    data = perts[1, i].numpy()
    data = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords=dict(lat=lats, lon=lons),
    )
    ax.imshow(
        data, cmap="coolwarm", transform=ccrs.PlateCarree(), interpolation="nearest"
    )
    plt.tight_layout()
    plt.savefig(
        f"/data3/WangGuanSong/TianXing/tropical cyclone/pictures/other/perturbation_{i}.png"
    )
    plt.close()
    # channel_mask[:, [0,1,2,3,5,7,60,61,62,63,64,65,66,67,68,69,70,71,72], :, :] = 0


if __name__ == "__main__":
    plot_perturbation()
