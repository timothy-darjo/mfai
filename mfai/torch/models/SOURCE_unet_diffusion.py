
import xarray as xr
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from datetime import datetime
from torch.utils.data import Dataset
import pandas as pd
import json
from netCDF4 import Dataset as cdfDataset


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv


# Directory containing the yearly NetCDF files (weatherbench)
#data_dir_temp = "/scratch/globc/villon/weatherbench/5.625deg/temperature/"
save_dir= "/scratch/globc/villon/weatherbench_test"
checkpoint_dir = "/scratch/globc/villon/weatherbench_test/models/checkpoints/"
#number_of_pressure_levels=3
batch_size_global = 32 #64 avec lstm et 3 variables, 128 raw cnn #entre 32 et 64*nb GPU
# 13 pressure value =  5 fois plus de temps de calcul


def get_input_channels(ds_slice, variables_with_lvls, variables_without_lvls):
    all_channels = []
    channel_names = []  # we'll return this too
    for var in variables_with_lvls:
        levels = variables_with_lvls[var]["levels"]
        for level in levels:
            if level not in ds_slice[var].level.values:
                continue
            arr = ds_slice[var].sel(level=level).values
            all_channels.append(arr)
            channel_names.append(f"{var}@{level}")

    for var in variables_without_lvls:
        arr = ds_slice[var].sel(level=1001).values
        all_channels.append(arr)
        channel_names.append(f"{var}@1001")

    data = np.stack(all_channels, axis=1)  # (time, channels, lat, lon)
    return data, channel_names


def data_creation_unet(
    config,
    remove_years_from_training=4,  # Number of years to remove from training to ease training
    validation_buffer_months=6,  # Number of months to exclude before validation
    testing_buffer_months=6,  # Number of months to exclude before testing
):
    """
    Args:
        config (dict): Dictionary containing data directory, variable levels, and time ranges.
        remove_years_from_training (int): Number of years to exclude from training to ease training.
        validation_buffer_months (int): Buffer period (in months) before validation set.
        testing_buffer_months (int): Buffer period (in months) before testing set.
    """
    # Extract configuration
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    variables_with_levels = config.get("variables_with_levels", {})
    variables_without_levels = config.get("variables_without_levels", {})

    train_years = config["train_years"]
    val_years = config["val_years"]
    test_years = config["test_years"]

    # Initialize an empty list to hold individual datasets
    datasets = []

    # Load and concatenate data for each variable
    for var, details in variables_with_levels.items():
        subdir = details["subdir"]
        levels = details["levels"]
        var_dir = os.path.join(data_dir, subdir)
        nc_files = sorted([os.path.join(var_dir, f) for f in os.listdir(var_dir) if f.endswith(".nc")])
        one_nc_file = nc_files[0]
        dataset = cdfDataset(one_nc_file, mode='r')
        variables = dataset.variables.keys()
        print("Variables:", list(variables))

        # Open multiple files as a single dataset
        var_ds = xr.open_mfdataset(nc_files, combine='by_coords')
        for v in var_ds.data_vars:
            nans = var_ds[v].isnull().sum().compute()

            print(f"  → {v}: {nans} NaNs, shape: {var_ds[v].shape}")
        # Select required levels
        var_ds = var_ds.sel(level=levels)
        for level in levels:
            for v in var_ds.data_vars:
                if "level" in var_ds[v].dims:
                    subset = var_ds[v].sel(level=level)
                    print(f"  → {v} @ level {level}: NaNs = {subset.isnull().sum().compute().item()}")
        # Add the variable dataset to the list
        datasets.append(var_ds)

    #Load and concatenate data for variables without levels
    for var, details in variables_without_levels.items():
        subdir = details["subdir"]
        var_dir = os.path.join(data_dir, subdir)
        nc_files = sorted([os.path.join(var_dir, f) for f in os.listdir(var_dir) if f.endswith(".nc")])
        dataset = cdfDataset(nc_files[0], mode='r')
        variables = dataset.variables.keys()

        # Open multiple files as a single dataset
        var_ds = xr.open_mfdataset(nc_files, combine='by_coords')
        for v in var_ds.data_vars:
            nans = var_ds[v].isnull().sum().compute()
            print(f"  → {v}: {nans} NaNs, shape: {var_ds[v].shape}")
        # Add a pseudo 'level' dimension with a distinct value, e.g., 1001
        if 'level' not in var_ds.dims:
            var_ds = var_ds.expand_dims({'level': [1001]})
        else:
            print(subdir)
            print(var_ds.dims)
            level_values = var_ds.coords['level'].values
            print(f"Levels in {subdir}: {level_values}")

        # Add the variable dataset to the list
        datasets.append(var_ds)
    for i, ds_test in enumerate(datasets):
        print(f"Dataset {i} dims: {ds_test.dims}")
    for i, d in enumerate(datasets):
        print(f"Dataset before merging {i} - NaN check:")
        for v in d.data_vars:
            print(f"  {v}: NaNs = {d[v].isnull().sum().compute().item()}, shape: {d[v].shape}")
    # Convert list to xarray Dataset (Merging all variables)
    ds = xr.merge(datasets)
    # Normalize variables
    normalization_stats = {}

    # Normalize variables with levels
    for var in variables_with_levels.keys():
        normalization_stats[var] = {}

        # Levels expected for this variable (as configured)
        expected_levels = variables_with_levels[var]["levels"]
        available_levels = ds[var].level.values

        for level in expected_levels:
            if level not in available_levels:
                print(f"⚠️ Skipping {var}@{level}: not present in available levels {available_levels}")
                continue

            print(f"New Before normalization {var} level {level}: "
                  f"min = {ds[var].sel(level=level).min().compute().item()}, "
                  f"max = {ds[var].sel(level=level).max().compute().item()}, "
                  f"NaNs = {ds[var].sel(level=level).isnull().sum().compute().item()}")

            mean = ds[var].sel(level=level).mean().values
            std = ds[var].sel(level=level).std().values

            if np.isnan(mean) or np.isnan(std) or std == 0:
                print(f"⚠️ Skipping normalization for {var} @ level {level}: mean={mean}, std={std}")
                continue

            normalization_stats[var][str(level)] = {"mean": float(mean), "std": float(std)}
            ds[var].loc[dict(level=level)] = (ds[var].sel(level=level) - mean) / std

    # Normalize variables without levels (only at level 1001)
    for var in variables_without_levels.keys():
        normalization_stats[var] = {}
        level = 1001

        if level not in ds[var].level.values:
            print(f"⚠️ Skipping normalization for {var}: dummy level {level} not found in dataset.")
            continue

        print(f"Before normalization {var} level {level}: "
              f"min = {ds[var].sel(level=level).min().compute().item()}, "
              f"max = {ds[var].sel(level=level).max().compute().item()}, "
              f"NaNs = {ds[var].sel(level=level).isnull().sum().compute().item()}")

        mean = ds[var].sel(level=level).mean().values
        std = ds[var].sel(level=level).std().values

        if np.isnan(mean) or np.isnan(std) or std == 0:
            print(f"⚠️ Skipping normalization for {var} @ level {level}: mean={mean}, std={std}")
            continue

        normalization_stats[var][str(level)] = {"mean": float(mean), "std": float(std)}
        ds[var].loc[dict(level=level)] = (ds[var].sel(level=level) - mean) / std


    # Save normalization stats
    with open(os.path.join(output_dir, "normalization_stats.json"), "w") as f:
        json.dump(normalization_stats, f)
    print(f"After normalization {var} level {level}: min = {ds[var].sel(level=level).min().compute().item()}, max = {ds[var].sel(level=level).max().compute().item()}, NaNs = {ds[var].sel(level=level).isnull().sum().compute().item()}")

    # Define buffer periods
    val_start = pd.to_datetime(val_years[0]) - pd.DateOffset(months=validation_buffer_months)
    val_end = pd.to_datetime(val_years[1])
    test_start = pd.to_datetime(test_years[0]) - pd.DateOffset(months=testing_buffer_months)
    test_end = pd.to_datetime(test_years[1])

    # Buffer slices
    validation_buffer = slice(val_start.strftime("%Y-%m-%d"), (pd.to_datetime(val_years[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    testing_buffer = slice(test_start.strftime("%Y-%m-%d"), (pd.to_datetime(test_years[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

    # Remove training years for simplification
    train_end = pd.to_datetime(train_years[1]) - pd.DateOffset(years=remove_years_from_training)
    training_exclusion = slice(train_end.strftime("%Y-%m-%d"), train_years[1])

    # Exclude these periods from the dataset
    to_remove = xr.concat([
        ds.sel(time=validation_buffer),
        ds.sel(time=testing_buffer),
        ds.sel(time=training_exclusion)
    ], dim="time")

    # Define datasets
    ds_train = ds.drop_sel(time=to_remove["time"])
    ds_val = ds.sel(time=slice(*val_years))
    ds_test = ds.sel(time=slice(*test_years))

    # Prepare input and target data
    X_train,channel_names = get_input_channels(ds_train.isel(time=slice(0, -1)), variables_with_levels,variables_without_levels)
    Y_train,channel_names = get_input_channels(ds_train.isel(time=slice(1, None)), variables_with_levels,variables_without_levels)

    X_val,channel_names = get_input_channels(ds_val.isel(time=slice(0, -1)), variables_with_levels,variables_without_levels)
    Y_val,channel_names = get_input_channels(ds_val.isel(time=slice(1, None)), variables_with_levels,variables_without_levels)

    X_test,channel_names = get_input_channels(ds_test.isel(time=slice(0, -1)), variables_with_levels,variables_without_levels)
    Y_test,channel_names = get_input_channels(ds_test.isel(time=slice(1, None)), variables_with_levels,variables_without_levels) # Assertions to ensure time ordering
    assert all(ds_train["time"].values[i] <= ds_train["time"].values[i + 1] for i in range(len(ds_train["time"].values) - 1)), "Training time is not ordered!"
    assert all(ds_val["time"].values[i] <= ds_val["time"].values[i + 1] for i in range(len(ds_val["time"].values) - 1)), "Validation time is not ordered!"
    assert all(ds_test["time"].values[i] <= ds_test["time"].values[i + 1] for i in range(len(ds_test["time"].values) - 1)), "Testing time is not ordered!"

    #in case of bug on data: print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    #Save the arrays and the channel_names
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "channel_names.json"), "w") as f:
        json.dump(channel_names, f) 
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "Y_val.npy"), Y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "Y_test.npy"), Y_test)

    print(f"Datasets saved in {output_dir}")   





def data_loading ():
    X_train = np.load(save_dir+"/"+"X_train.npy")
    Y_train = np.load(save_dir+"/"+"Y_train.npy")
    X_val = np.load(save_dir+"/"+"X_val.npy")
    Y_val = np.load(save_dir+"/"+"Y_val.npy")
    X_test = np.load(save_dir+"/"+"X_test.npy")
    Y_test = np.load(save_dir+"/"+"Y_test.npy")
    for name, arr in zip(
        ['X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test'],
        [X_train, Y_train, X_val, Y_val, X_test, Y_test]):
        n_nans = np.isnan(arr).sum()
        print(f"{name} — shape: {arr.shape}, NaNs: {n_nans}")
        if n_nans > 0:
            print(f"⚠️ Warning: {name} contains {n_nans} NaNs!")
    number_of_input_for_model=X_train.shape[1]
    #print("Number of inputs:", number_of_input_for_model)



    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    # Debug: Check first sample
    x0 = X_test_tensor[0]
    y0 = Y_test_tensor[0]
    print("Input sample 0: min =", torch.min(X_test_tensor), "max =", torch.max(X_test_tensor), "NaNs =", torch.isnan(x0).any())
    print("Target sample 0: min =", torch.min(Y_test_tensor), "max =", torch.max(Y_test_tensor), "NaNs =", torch.isnan(y0).any())
    
        # Create datasets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, prefetch_factor=2, batch_size=batch_size_global, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset,prefetch_factor=2, batch_size=batch_size_global,num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset,prefetch_factor=2, batch_size=batch_size_global,num_workers=4, drop_last=True)

    return train_loader,val_loader,test_loader,number_of_input_for_model






def diffusion_loss(predicted_noise, true_noise):
    #Mean Squared Error loss for predicted vs true noise.
    #peut être a revtrailler mais pour le moment a l'air de fonctionner parfaitement
    return torch.mean((predicted_noise - true_noise) ** 2)

class NoiseScheduler:
    def __init__(self, timesteps=25):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])

    def get_noise_level(self, t):
        """
        Get the noise level for the given time step(s).
        Args:
            t (torch.Tensor): A tensor of time steps of shape [batch_size].
        Returns:
            torch.Tensor: Noise level tensor reshaped to [batch_size, 1, 1, 1].
        """
        if isinstance(t, torch.Tensor):
            t = t.clamp(0, self.timesteps - 1).long()  # Ensure valid range
        noise_level = self.alpha_cumprod[t]  # Shape: [batch_size]
        return noise_level.view(-1, 1, 1, 1)  # Reshape for broadcasting

class DiffusionModel(nn.Module):
    def __init__(self, unet_model, noise_scheduler):
        super(DiffusionModel, self).__init__()
        self.unet = unet_model
        self.noise_scheduler = noise_scheduler
    def forward(self, y, x, t):
        noise_level = self.noise_scheduler.get_noise_level(t).to(x.device)
        noise = torch.randn_like(y)
        noisy_y = y + noise_level * noise

        predicted_noise = self.unet(noisy_y, x)
        return predicted_noise,noise


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, base_channels=64):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(input_channels * 2, base_channels, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        
        # Decoder
        self.dec3 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(base_channels * 2, output_channels, kernel_size=3, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()
    
    def forward(self, noisy_y, x):
        # Encoder
        input = torch.cat([noisy_y, x], dim=1)  # (B, 2C, H, W) if x and y same shape

        e1 = self.relu(self.enc1(input))  
        e2 = self.relu(self.enc2(self.maxpool(e1)))
        e3 = self.relu(self.enc3(self.maxpool(e2)))
        
        # Bottleneck
        b = self.relu(self.bottleneck(self.maxpool(e3)))
        
        # Decoder
        d3 = self.relu(self.dec3(self.upsample(b)))
        d2 = self.relu(self.dec2(self.upsample(d3 + e3)))
        d1 = self.dec1(self.upsample(d2 + e2))
        
        return d1        




def plot_predictions(predicted, ground_truth, time_points, unique_id, string_test_variable):
    # Ensure time points match predicted
    time_points = time_points[:len(predicted)]

    # Create a figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    # Plot both predictions and ground truth
    axs[0].plot(time_points, predicted, label="Predictions", color="blue", alpha=0.7)
    axs[0].plot(time_points, ground_truth, label="Ground Truth", color="orange", alpha=0.7)
    axs[0].set_title("Predicted vs Ground Truth")
    axs[0].set_ylabel(string_test_variable)
    axs[0].legend()
    axs[0].grid(True)

    # Plot only predictions
    axs[1].plot(time_points, predicted, label="Predictions", color="blue", alpha=0.7)
    axs[1].set_title("Predictions")
    axs[1].set_ylabel(string_test_variable)
    axs[1].grid(True)

    # Plot only ground truth
    axs[2].plot(time_points, ground_truth, label="Ground Truth", color="orange", alpha=0.7)
    axs[2].set_title("Ground Truth")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel(string_test_variable)
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plots
    plt.savefig(f"results/predictions_three_part.png")
    print(f"Saved predictions plot to results/predictions_three_part.png")

    # Show the plot
    plt.show()

def plot_weather_maps(ground_truth, predictions, lats, lons, string_test_variable, title_prefix="", save_path=None):
    """
    Generates a single image with three side-by-side heatmaps:
    1. Average Ground Truth
    2. Average Model Predictions
    3. Error Map (Ground Truth - Prediction)

    Parameters:
    - ground_truth (numpy array): Shape (lat, lon) - True temperature data
    - predictions (numpy array): Shape (lat, lon) - Model predictions
    - lats (numpy array): 1D array of latitude coordinates
    - lons (numpy array): 1D array of longitude coordinates
    - title_prefix (str): Optional prefix for titles.
    """

    # Compute difference map
    diff = ground_truth - predictions  # Shape: (lat, lon)
    #
    #print("ground_truth")
    #print(ground_truth)
    #print("predictions")
    #print(predictions)
    #print("diff")
    #print(diff)
    # Create latitude and longitude edges (for pcolormesh)
    lats_edges = np.linspace(-90, 90, ground_truth.shape[0] + 1)  # 33 for 32 lat points
    lons_edges = np.linspace(0, 360, ground_truth.shape[1] + 1)  # 65 for 64 lon points

    # Set up the figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    # Titles for each panel
    titles = [
        f"{title_prefix} Average Ground Truth "+string_test_variable,
        f"{title_prefix} Average Predicted "+string_test_variable,
        f"{title_prefix} Prediction Error (GT - Prediction)"
    ]

    vmin = np.min(diff)
    vmax = np.max(diff)
    # Data for each panel
    data_list = [ground_truth, predictions, diff]
    cmaps = ["coolwarm", "coolwarm", "RdBu_r"]
    vmin_list = [None, None, None]  # Only fix vmin/vmax for error map
    vmax_list = [None, None, None]

    # Loop over the 3 panels
    for i, ax in enumerate(axes):
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        img = ax.pcolormesh(lons_edges, lats_edges, data_list[i], transform=ccrs.PlateCarree(),
                            cmap=cmaps[i], vmin=vmin_list[i], vmax=vmax_list[i])

        plt.colorbar(img, ax=ax, orientation="horizontal", pad=0.05, label="")
        ax.set_title(titles[i])

    # Adjust layout for better spacing
    plt.tight_layout()

        # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show the figure
    plt.show()

def initialize_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_variable_channel_index(variables_with_levels, variables_without_levels, variable, level):
    """
    Get the channel index for a specific variable and level.
    Args:
        variables_with_levels (dict): Dictionary of variables and levels.
        variable (str): The variable to retrieve (e.g., 't' or 'u').
        level (int): The pressure level.

    Returns:
        int: The channel index corresponding to the variable and level.
    """
    level=int(level)
    print(f"Requested Variable: {variable}, Level: {level}")

    channel_index = 0
    for var, details in variables_with_levels.items():
        print (details)
        levels = details["levels"]
        print(f"Processing Variable: {var}, Levels: {levels}, Current Channel Index: {channel_index}")
        if var == variable:
            level_index = levels.index(level)  # Check where this level exists
            print(f"Found {variable} at Level {level} → Level Index: {level_index}, Final Channel Index: {channel_index + level_index}")
            return channel_index + levels.index(level)
        channel_index += len(levels)
    for var in variables_without_levels.keys():
        if var == variable:
            if level == 1001:  # The artificial level assigned to single-level variables
                return channel_index
            else:
                raise ValueE    
    raise ValueError(f"Variable {variable} or level {level} not found.")

#train like testing
def test_model_2(
    model, loader, criterion, noise_scheduler, device,
    variable, level,channel_to_index, desired_lat=10, desired_lon=15, 
    variables_with_levels=None, variables_without_levels=None, normalization_stats=None, output_csv="predictions_vs_groundtruth.csv"
):
    """
    Test the model for a specific variable and level, and collect predictions for plotting. Save results in a CSV.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the test on.
        variable (str): The variable to evaluate (e.g., 't' for temperature, 'u' for wind speed).
        level (int): The pressure level.
        desired_lat (int): Index for the latitude.
        desired_lon (int): Index for the longitude.
        variables_with_levels (dict): Dictionary of variables and levels for indexing.
        normalization_stats (dict): Dictionary of normalization stats for all variables.
        output_csv (str): Path to save the predictions and ground truth.

    Returns:
        tuple: Average test loss, a dictionary of metrics (MAE, RMSE, R²), and lists of predictions and ground truths.
    """
    '''if variable in variables_with_levels:

        if variables_with_levels is None:
            raise ValueError("variables_with_levels must be provided.")
        # Get the channel index for the desired variable and level
        channel_index = get_variable_channel_index(variables_with_levels,variables_without_levels, variable, level)
    #print(f"Compute Channel Index: {channel_index}")
    elif variable in variables_without_levels:
        if level is not None:
            channel_index = get_variable_channel_index(variables_with_levels, variables_without_levels, variable, level)
    else:
        raise ValueError(f"Variable '{variable}' not found in provided variable dictionaries.")'''
    
    channel_key = f"{variable}@{level}"
    if channel_key not in channel_to_index:
        raise ValueError(f"Channel {channel_key} not found in channel_names!")

    channel_index = channel_to_index[channel_key]

    print("normalization stats ",normalization_stats)
    # Retrieve normalization stats
    if normalization_stats is None or variable not in normalization_stats or str(level) not in normalization_stats[variable]:
        raise ValueError(f"Normalization stats for variable '{variable}' at level {level} not found.")
   
    if variable in variables_with_levels:
        if str(level) not in normalization_stats[variable]:
            raise ValueError(f"Normalization stats for variable '{variable}' at level {level} not found.")
        mean = normalization_stats[variable][str(level)]["mean"]
        std = normalization_stats[variable][str(level)]["std"]
    else:  # Variable without levels
        if str(level) not in normalization_stats[variable]:
            raise ValueError(f"Normalization stats for variable '{variable}' at level {level} not found.")
        mean = normalization_stats[variable]["1001"]["mean"]
        std = normalization_stats[variable]["1001"]["std"]


    print("mean and std",mean,std)
    model.eval()
    total_loss, total_mae, total_rmse, ss_res, ss_tot = 0.0, 0.0, 0.0, 0.0, 0.0
    predicted_values = []
    ground_truth_values = []

    # Storage for full spatial maps over time
    pred_maps = []
    gt_maps = []

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prediction (normalized)", "Ground Truth (normalized)", "Prediction (real)", "Ground Truth (real)"])

        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                #print("NaNs in input:", torch.isnan(X_batch).any().item())
                #print("NaNs in real:", torch.isnan(Y_batch).any().item())

                # Forward pass
                predicted_noise,_ = model(X_batch, torch.tensor([0] * X_batch.size(0), device=device))
                predicted_output = X_batch - predicted_noise  # Reconstruct clean signal
                # Compute loss
                loss = criterion(predicted_output, Y_batch.unsqueeze(1))
                total_loss += loss.item()
                # Extract predictions and ground truth for the specific channel
                #print("predicted_output shape:", predicted_output.shape)

                pred_norm = predicted_output[:, channel_index, desired_lat, desired_lon].cpu().numpy()
                gt_norm = Y_batch[:, channel_index, desired_lat, desired_lon].cpu().numpy()
                
                # Denormalize predictions and ground truth
                pred_real = (pred_norm * std) + mean
                gt_real = (gt_norm * std) + mean

                for pn, gn, pr, gr in zip(pred_norm, gt_norm, pred_real, gt_real):
                    writer.writerow([pn, gn, pr, gr])
                    predicted_values.append(pr)
                    ground_truth_values.append(gr)

                # Store full maps

                pred_norm_for_maps = predicted_output[:, channel_index, :, :].cpu().numpy()  # Full (lat, lon)
                gt_norm_for_maps = Y_batch[:, channel_index, :, :].cpu().numpy()


                # Convert to real values (denormalization)
                pred_real_for_maps = (pred_norm_for_maps * std) + mean  # Shape: (batch, lat, lon)
                gt_real_for_maps = (gt_norm_for_maps * std) + mean  # Shape: (batch, lat, lon)
                if pred_real_for_maps.shape[0] == 32:  # Only append if batch size matches expected
                    pred_maps.append(pred_real_for_maps)
                    gt_maps.append(gt_real_for_maps)
                else:
                    print(f"Skipping batch with shape {pred_real_for_maps.shape}")

                # Metrics
                mae = torch.abs(predicted_output[:, channel_index] - Y_batch[:, channel_index]).mean().item()
                rmse = torch.sqrt(((predicted_output[:, channel_index] - Y_batch[:, channel_index]) ** 2).mean()).item()
                total_mae += mae
                total_rmse += rmse

                y_mean = np.mean(gt_norm)
                ss_res = np.sum((gt_norm - pred_norm) ** 2)
                ss_tot = np.sum((gt_norm - y_mean) ** 2)

    # Convert lists to numpy arrays: (time, lat, lon)
    pred_maps = np.stack(pred_maps, axis=0)  # Ensures consistent dimensions
    gt_maps = np.stack(gt_maps, axis=0)

    # Compute time-averaged maps: (lat, lon)
    pred_map_avg = np.mean(pred_maps, axis=(0,1))
    gt_map_avg = np.mean(gt_maps, axis=(0,1))

    #print("pred_map")
    #print(pred_maps[0])
    #print("gt_maps")
    #print(gt_maps[0])
    #print("predicted_values")
    #print(predicted_values[0])

    r_squared = 1 - (ss_res / ss_tot)
    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)
    avg_rmse = total_rmse / len(loader)
    print(f"Results saved in {output_csv}")
    return avg_loss, {"mae": avg_mae, "rmse": avg_rmse, "r2": r_squared}, predicted_values, ground_truth_values, pred_map_avg, gt_map_avg


#inference_like_testing
def test_model(
    model, loader, criterion, noise_scheduler, device,
    variable, level, channel_to_index, desired_lat=10, desired_lon=15,
    variables_with_levels=None, variables_without_levels=None, normalization_stats=None,
    output_csv="predictions_vs_groundtruth.csv", num_timesteps=25
):
    channel_key = f"{variable}@{level}"
    if channel_key not in channel_to_index:
        raise ValueError(f"Channel {channel_key} not found in channel_names!")
    channel_index = channel_to_index[channel_key]

    # Get normalization stats
    if normalization_stats is None or variable not in normalization_stats or str(level) not in normalization_stats[variable]:
        raise ValueError(f"Normalization stats for variable '{variable}' at level {level} not found.")
    mean = normalization_stats[variable][str(level)]["mean"]
    std = normalization_stats[variable][str(level)]["std"]
    print("mean used for denorm:", mean)
    print("std used for denorm:", std)
    model.eval()
    predicted_values = []
    ground_truth_values = []
    pred_maps, gt_maps = [], []
    ss_res, ss_tot = 0.0, 0.0
    total_mae, total_rmse = 0.0, 0.0

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prediction (normalized)", "Ground Truth (normalized)", "Prediction (real)", "Ground Truth (real)"])

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Start from random noise
                y_pred = torch.randn_like(y_batch)

                # Run reverse diffusion
                for t in reversed(range(num_timesteps)):
                    #print(t)
                    t_scalar = int(t)  # int
                    t_tensor = torch.full((x_batch.size(0),), t_scalar, device=device, dtype=torch.long)  # for model

                    predicted_noise,_ = model(y_pred, x_batch, t_tensor)

                    # Convert scalar t to tensor values (on device)
                    alpha_t = noise_scheduler.alpha_cumprod[t_scalar].to(device)
                    beta_t = noise_scheduler.betas[t_scalar].to(device)

                    # scalars (shape: []), need to unsqueeze them for broadcasting
                    alpha_t = alpha_t.view(1, 1, 1, 1)
                    beta_t = beta_t.view(1, 1, 1, 1)
                    one = torch.ones_like(alpha_t)
                 

                    #print("one:", one)
                    #print("alpha_t:", alpha_t)
                    #print("beta_t:", beta_t)
                    #print("predicted_noise:", predicted_noise.shape)
                    #print("y_pred:", y_pred.shape)

                    # Break the math down step-by-step
                    sqrt_alpha_t = torch.sqrt(alpha_t)
                    sqrt_one_minus_alpha_t = torch.sqrt(one - alpha_t)


                    y_pred = (one / torch.sqrt(alpha_t)) * (y_pred - (beta_t / torch.sqrt(one - alpha_t)) * predicted_noise)
                    # Clamp to avoid numerical blow-up

                    if t_scalar > 950:
                        y_pred = y_pred.clamp(-10, 10)
                    else:
                        y_pred = y_pred.clamp(-10, 10)
                    if t_scalar > 0:
                        noise = torch.randn_like(y_pred)
                        y_pred += torch.sqrt(beta_t) * noise

                # Extract values at location
                pred_norm = y_pred[:, channel_index, desired_lat, desired_lon].cpu().numpy()
                gt_norm = y_batch[:, channel_index, desired_lat, desired_lon].cpu().numpy()
                print("y_pred normalized min/max:", y_pred.min().item(), y_pred.max().item())

                pred_real = (pred_norm * std) + mean
                gt_real = (gt_norm * std) + mean
                print (pred_real)

                for pn, gn, pr, gr in zip(pred_norm, gt_norm, pred_real, gt_real):
                    writer.writerow([pn, gn, pr, gr])
                    predicted_values.append(pr)
                    ground_truth_values.append(gr)

                # Full maps
                pred_norm_maps = y_pred[:, channel_index, :, :].cpu().numpy()
                gt_norm_maps = y_batch[:, channel_index, :, :].cpu().numpy()

                pred_real_maps = (pred_norm_maps * std) + mean
                gt_real_maps = (gt_norm_maps * std) + mean

                if pred_real_maps.shape[0] == 32:
                    pred_maps.append(pred_real_maps)
                    gt_maps.append(gt_real_maps)

                # Metrics
                mae = torch.abs(y_pred[:, channel_index] - y_batch[:, channel_index]).mean().item()
                rmse = torch.sqrt(((y_pred[:, channel_index] - y_batch[:, channel_index]) ** 2).mean()).item()
                total_mae += mae
                total_rmse += rmse

                y_mean = np.mean(gt_norm)
                ss_res += np.sum((gt_norm - pred_norm) ** 2)
                ss_tot += np.sum((gt_norm - y_mean) ** 2)

    # Stack and average maps
    pred_maps = np.stack(pred_maps, axis=0)
    gt_maps = np.stack(gt_maps, axis=0)
    pred_map_avg = np.mean(pred_maps, axis=(0, 1))
    gt_map_avg = np.mean(gt_maps, axis=(0, 1))

    # Final metrics
    predicted_values = np.array(predicted_values)
    ground_truth_values = np.array(ground_truth_values)

    y_mean_global = np.mean(ground_truth_values)
    ss_res = np.sum((ground_truth_values - predicted_values) ** 2)
    ss_tot = np.sum((ground_truth_values - y_mean_global) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    avg_mae = np.mean(np.abs(predicted_values - ground_truth_values))
    avg_rmse = np.sqrt(np.mean((predicted_values - ground_truth_values) ** 2))
    print(f"Results saved in {output_csv}")

    return 0.0, {"mae": avg_mae, "rmse": avg_rmse, "r2": r_squared}, predicted_values, ground_truth_values, pred_map_avg, gt_map_avg




def train_one_epoch_unet(model, train_loader, optimizer, criterion, noise_scheduler, device, epoch):
    model.train()
    epoch_loss = 0

    for batch_idx, (x, y)  in enumerate(train_loader):
        x = x.to(device)
        y= y.to(device)
        t = torch.randint(0, noise_scheduler.timesteps, (x.size(0),), device=device)  # Random time steps

        optimizer.zero_grad()
        predicted_noise, true_noise = model(y, x, t)  # Forward pass
        loss = criterion(predicted_noise, true_noise)  # Compute loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print("Loss:", loss.item())
    print("Predicted noise mean/std:", predicted_noise.mean().item(), predicted_noise.std().item())
    print("True noise mean/std:", true_noise.mean().item(), true_noise.std().item())
    return epoch_loss / len(train_loader)

def validate_one_epoch_unet(model, val_loader, criterion, noise_scheduler, device):
    model.eval()
    val_loss = 0
    all_predictions = []
    all_ground_truth = []
    with torch.no_grad():
        for batch_idx, (x, y)  in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            t = torch.randint(0, noise_scheduler.timesteps, (x.size(0),), device=device)

            predicted_noise, true_noise = model(y,x, t)  # Forward pass
            loss = criterion(predicted_noise, true_noise)  # Compute loss
            val_loss += loss.item()

            # Collect predictions and ground truth for metrics
            all_predictions.append(predicted_noise.cpu().numpy())
            all_ground_truth.append(true_noise.cpu().numpy())

    # Flatten predictions and ground truth
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Compute metrics
    metrics = calculate_metrics(all_ground_truth, all_predictions)

    return val_loss / len(val_loader), metrics

def calculate_metrics(ground_truth, predictions):
    """
    Calculate MAE, RMSE, and R² metrics.

    Args:
        ground_truth (np.ndarray): Ground truth values, shape (num_samples, ...).
        predictions (np.ndarray): Predicted values, shape (num_samples, ...).

    Returns:
        dict: Dictionary containing 'mae', 'rmse', and 'r2' metrics.
    """
    # Flatten arrays to simplify calculations
    ground_truth = ground_truth.flatten()
    predictions = predictions.flatten()

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(ground_truth - predictions))

    # Root Mean Squared Error (RMSE)
    mse = np.mean((ground_truth - predictions) ** 2)
    rmse = np.sqrt(mse)

    # R² Score
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {'mae': mae, 'rmse': rmse, 'r2': r2}
'''
'''

def unnormalize_predictions(ground_truth, predicted, mean_X_train, std_X_train):
    """
    Unnormalize ground truth and predicted temperature values.

    Args:
        ground_truth (list or np.ndarray): Normalized ground truth values.
        predicted (list or np.ndarray): Normalized predicted values.
        mean_X_train (float): Mean of the training data used for normalization.
        std_X_train (float): Standard deviation of the training data used for normalization.

    Returns:
        tuple: Unnormalized ground truth and predicted values.
    """
    # Convert to numpy arrays if not already
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)


    # Unnormalize
    ground_truth_unnormalized = (ground_truth * std_X_train) + mean_X_train
    predicted_unnormalized = (predicted * std_X_train) + mean_X_train

    return ground_truth_unnormalized, predicted_unnormalized

def run_model_single_gpu(config, train_loader,val_loader,test_loader,number_of_input_for_model, num_epochs=20, learning_rate=0.01, save_plots=True, unique_id=None):
    """
    Train, validate, and test the WeatherPredictor model on a single GPU.

    Args:
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        save_plots (bool): Whether to save prediction plots.
        unique_id (str): Unique identifier for saving results (optional).
    """
    #/scratch/globc/villon/weatherbench/5.625deg/temperature/
    num_epochs=50
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    test_years = config["test_years"]
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Device setup
    print(f"Using device: {device}")
    #testing_years = slice("2003-01-01", "2005-12-31")  # Adjust as per your test range
    # Load normalization stats
    with open(os.path.join(output_dir, "normalization_stats.json"), "r") as f:
        normalization_stats = json.load(f)
    input_channels=number_of_input_for_model
    # Instantiate components for UNET
    unet = UNet(input_channels=input_channels, output_channels=input_channels, base_channels=64)
    noise_scheduler = NoiseScheduler(timesteps=25)
    diffusion_model = DiffusionModel(unet, noise_scheduler).to(device)
    #print(diffusion_model)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)
    criterion = diffusion_loss  # MSE loss
    # Unique identifier for filenames
    if not unique_id:
        unique_id = datetime.now().strftime("%d_%m_%Y_%H_%M")
    # CSV and checkpoint paths
    csv_file = f"results/testing_results_{unique_id}.csv"
    checkpoint_file = f"{checkpoint_dir}model_epoch_{unique_id}.pth"
    # Training and validation loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch_unet(diffusion_model, train_loader, optimizer, criterion, noise_scheduler, device, epoch)
        val_loss, metrics = validate_one_epoch_unet(diffusion_model, val_loader, criterion, noise_scheduler, device)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print()
        #print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_file)
            print(f"Model checkpoint saved at {checkpoint_file}")





    ###testing loop
    with open(os.path.join(output_dir, "channel_names.json"), "r") as f:
        channel_names = json.load(f)

    channel_to_index = {name: i for i, name in enumerate(channel_names)}
    #channel_index = channel_to_index["t2m@1001"]
    variables_with_levels = config.get("variables_with_levels", {})
    variables_without_levels = config.get("variables_without_levels", {})
    
    # Test model and collect predictions
    #liste des variable et levels a tester
    variables_list=["t"]
    levels_list=["850"]
    for test_variable in variables_list:
        for level in levels_list :

            if test_variable in variables_with_levels :
                variable_sub_folder=variables_with_levels[test_variable]["subdir"]
                variable_path=os.path.join(data_dir,variable_sub_folder )

            elif test_variable in variables_without_levels :
                variable_sub_folder=variables_without_levels[test_variable]["subdir"]
                variable_path=os.path.join(data_dir,variable_sub_folder )
            print(variable_path)
            nc_files = sorted([os.path.join(variable_path, f) for f in os.listdir(variable_path) if f.endswith(".nc")])
            ds = xr.open_mfdataset(nc_files, combine='by_coords')
            test_ds = ds.sel(time=slice(*test_years))
            test_time_points = test_ds['time'].values  # Extract time points as a NumPy array
            # Select the levels 1000, 925, and 850
            #ds_filtered = ds.sel(level=[1000,925,850])
            #ds= ds_filtered
            #print(normalization_stats)
            if test_variable == "t":
                unit ="K"
            elif test_variable == "u" or test_variable =="v":
                unit = "m/s"
            elif test_variable == "t2m" :
                unit = "K"
            test_variable_string= test_variable+" ("+unit+") "+ level +" hPa"

            test_variable_string_to_save= test_variable+"_"+ level +"_hPa"



            test_loss, metrics, predictions, ground_truth, pred_map, gt_map = test_model(
                model=diffusion_model,
                loader=test_loader,
                noise_scheduler=noise_scheduler,
                criterion=criterion,
                device=device,
                variable=test_variable,  # Temperature, precip, etc
                level=level,  # Pressure level in hPa
                channel_to_index=channel_to_index,
                desired_lat=10,  # Latitude index
                desired_lon=15,  # Longitude index
                variables_with_levels=config["variables_with_levels"],
                variables_without_levels=config["variables_without_levels"],
                normalization_stats=normalization_stats,
                output_csv="results_"+test_variable_string_to_save+".csv"
            )

            #test_loss, test_metrics, predicted_temperatures, ground_truth_temperatures = test_model(diffusion_model, test_loader, criterion, device, desired_level=0, desired_lat=10, desired_lon=15, output_csv="predictions_vs_groundtruth_real.csv",    mean=mean, std=std)   

            print(f"\nTest Results - Loss: {test_loss:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

            #unnormalized_ground_truth, unnormalized_predicted = unnormalize_predictions(ground_truth, predictions, mean, std)
            #no need as we get unnormalized from the test_model function
            # Save prediction plots
            #if save_plots:
                #plot_predictions(predicted_temperatures, ground_truth_temperatures, test_time_points, unique_id)
            if save_plots:
                plot_predictions(predictions, ground_truth, test_time_points, unique_id,test_variable_string)

            test_variable_string_to_save= test_variable+"_"+ level +"_hPa"

            # Generate lat/lon grids based on dataset resolution
            lats = np.linspace(-90, 90, pred_map.shape[0])  # Adjust based on dataset
            lons = np.linspace(0, 360, pred_map.shape[1])  # Adjust range
            image_compar_path=output_dir+"/"+test_variable_string_to_save+"comparison.png"
            #print("Latitude grid:", lats)
            #print("Longitude grid:", lons)
            # Call the visualization function
            #print("gt_map avant visualisation")
            #print(gt_map)
            plot_weather_maps(gt_map, pred_map, lats, lons, test_variable_string, title_prefix="Test Period "+ test_variable_string,save_path=image_compar_path)



config = {
    "data_dir": "/scratch/globc/villon/weatherbench/5.625deg/",
    "output_dir": "/scratch/globc/villon/weatherbench_test",
    "variables_with_levels": {
        "t": {
            "levels": [1000, 925, 850],
            "subdir": "temperature"
        },
        "u": {
            "levels": [1000, 925, 850],
            "subdir": "u_component_of_wind"
        }
    },
    "variables_without_levels": {  # Surface variables (without pressure levels)
        "tp": {"subdir": "total_precipitation"},
        "t2m" : {"subdir": "2m_temperature"}
        #"z": {"subdir": "geopotential"}
    },

    "train_years": ("1979-01-01", "1979-12-31"), #    "train_years": ("1979-01-01", "1999-12-31"),

    "val_years": ("1982-01-01", "1982-12-31"), #"val_years": ("1982-01-01", "1985-12-31")
    "test_years": ("2010-01-01", "2010-06-30"),#"test_years": ("2010-01-01", "2012-12-31"),
}


#data_creation_unet(config)
train_loader,val_loader,test_loader, number_of_input_for_model = data_loading()
run_model_single_gpu(config,train_loader,val_loader,test_loader,number_of_input_for_model, num_epochs=1, learning_rate=0.001)


#total years: 1979-2020
'''

config = {
    "data_dir": "/scratch/globc/villon/weatherbench/5.625deg/",
    "output_dir": "/scratch/globc/villon/weatherbench_test",
    
    "variables_with_levels": {  # Variables with multiple pressure levels
        "t": {
            "levels": [1000, 925, 850],
            "subdir": "temperature"
        },
        "u": {
            "levels": [1000, 925, 850],
            "subdir": "u_component_of_wind"
        }
    },

    "variables_without_levels": {  # Surface variables (without pressure levels)
        "tp": {"subdir": "total_precipitation"},
    #    "z": {"subdir": "geopotential"}
    },

    "train_years": ("1979-01-01", "1999-12-31"),
    "val_years": ("1982-01-01", "1985-12-31"),
    "test_years": ("2010-01-01", "2012-12-31"),
}

data_creation_unet(config)

train_loader,val_loader,test_loader, number_of_input_for_model = data_loading()
run_model_single_gpu(config,train_loader,val_loader,test_loader,number_of_input_for_model, num_epochs=1, learning_rate=0.001)
'''

#total years: 1979-2020

##to read info from nc file :

'''from netCDF4 import Dataset
nc_file = 'total_precipitation_1999_5.625deg.nc'
dataset = Dataset(nc_file, mode='r')
variables = dataset.variables.keys()
print("Variables:", list(variables))'''
