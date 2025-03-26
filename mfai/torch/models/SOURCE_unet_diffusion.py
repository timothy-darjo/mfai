
#MUST REMOVE BEFORE PR
#example code to be integrated into torch/models
#code from Sébastien VILLON, Cerfacs

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


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv


# Directory containing the yearly NetCDF files (weatherbench)
data_dir_temp = "/scratch/globc/villon/weatherbench/5.625deg/temperature/"
save_dir= "/scratch/globc/villon/weatherbench_test"
checkpoint_dir = "/scratch/globc/villon/weatherbench_test/models/checkpoints/"
#number_of_pressure_levels=3
batch_size_global = 32 #64 avec lstm et 3 variables, 128 raw cnn #entre 32 et 64*nb GPU
# 13 pressure value =  5 fois plus de temps de calcul

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
    variables_with_levels = config["variables_with_levels"]
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

        # Open multiple files as a single dataset
        var_ds = xr.open_mfdataset(nc_files, combine='by_coords')

        # Select required levels if applicable
        if levels:
            var_ds = var_ds.sel(level=levels)

        # Add the variable dataset to the main dataset
        datasets.append(var_ds)


    # Convert list to xarray Dataset (Merging all variables)
    ds = xr.merge(datasets)
    # Normalize variables
    normalization_stats = {}
    for var in variables_with_levels.keys():
        normalization_stats[var] = {}
        if 'level' in ds[var].dims:
            for level in ds[var].level.values:
                mean = ds[var].sel(level=level).mean().values
                std = ds[var].sel(level=level).std().values
                normalization_stats[var][str(level)] = {"mean": float(mean), "std": float(std)}
                ds[var].loc[dict(level=level)] = (ds[var].sel(level=level) - mean) / std
                #ds[var] = ds[var].assign_coords(level=ds[var].level).where(ds[var].level == level, (ds[var] - mean) / std)
        else:
            mean = ds[var].mean().values
            std = ds[var].std().values
            normalization_stats[var]["global"] = {"mean": float(mean), "std": float(std)}
            ds[var] = (ds[var] - mean) / std

    # Save normalization stats
    with open(os.path.join(output_dir, "normalization_stats.json"), "w") as f:
        json.dump(normalization_stats, f)

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
    X_train = np.concatenate([ds_train[var].values[:-1] for var in variables_with_levels.keys()], axis=1)
    Y_train = np.concatenate([ds_train[var].values[1:] for var in variables_with_levels.keys()], axis=1)

    X_val = np.concatenate([ds_val[var].values[:-1] for var in variables_with_levels.keys()], axis=1)
    Y_val = np.concatenate([ds_val[var].values[1:] for var in variables_with_levels.keys()], axis=1)

    X_test = np.concatenate([ds_test[var].values[:-1] for var in variables_with_levels.keys()], axis=1)
    Y_test = np.concatenate([ds_test[var].values[1:] for var in variables_with_levels.keys()], axis=1)
    # Assertions to ensure time ordering
    assert all(ds_train["time"].values[i] <= ds_train["time"].values[i + 1] for i in range(len(ds_train["time"].values) - 1)), "Training time is not ordered!"
    assert all(ds_val["time"].values[i] <= ds_val["time"].values[i + 1] for i in range(len(ds_val["time"].values) - 1)), "Validation time is not ordered!"
    assert all(ds_test["time"].values[i] <= ds_test["time"].values[i + 1] for i in range(len(ds_test["time"].values) - 1)), "Testing time is not ordered!"

    #in case of bug on data: print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    #Save the arrays
    os.makedirs(output_dir, exist_ok=True)
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
    number_of_input_for_model=X_train.shape[1]
    #print("Number of inputs:", number_of_input_for_model)



    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, prefetch_factor=2, batch_size=batch_size_global, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset,prefetch_factor=2, batch_size=batch_size_global)
    test_loader = DataLoader(test_dataset,prefetch_factor=2, batch_size=batch_size_global)

    return train_loader,val_loader,test_loader,number_of_input_for_model






def diffusion_loss(predicted_noise, true_noise):
    #Mean Squared Error loss for predicted vs true noise.
    #peut Ãªtre a revtrailler mais pour le moment a l'air de fonctionner parfaitement
    return torch.mean((predicted_noise - true_noise) ** 2)

class NoiseScheduler:
    def __init__(self, timesteps=1000):
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
    def forward(self, x, t):
        noise_level = self.noise_scheduler.get_noise_level(t).to(x.device)
        noisy_x = x + noise_level * torch.randn_like(x)
        predicted_noise = self.unet(noisy_x)
        return predicted_noise



class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, base_channels=64):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
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
    
    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.maxpool(e1)))
        e3 = self.relu(self.enc3(self.maxpool(e2)))
        
        # Bottleneck
        b = self.relu(self.bottleneck(self.maxpool(e3)))
        
        # Decoder
        d3 = self.relu(self.dec3(self.upsample(b)))
        d2 = self.relu(self.dec2(self.upsample(d3 + e3)))
        d1 = self.dec1(self.upsample(d2 + e2))
        
        return d1        




def plot_predictions(predicted, ground_truth, time_points, unique_id):
    # Ensure time points match predicted
    time_points = time_points[:len(predicted)]

    # Create a figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    # Plot both predictions and ground truth
    axs[0].plot(time_points, predicted, label="Predictions", color="blue", alpha=0.7)
    axs[0].plot(time_points, ground_truth, label="Ground Truth", color="orange", alpha=0.7)
    axs[0].set_title("Predicted vs Ground Truth")
    axs[0].set_ylabel("Temperature (Â°C)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot only predictions
    axs[1].plot(time_points, predicted, label="Predictions", color="blue", alpha=0.7)
    axs[1].set_title("Predictions")
    axs[1].set_ylabel("Temperature (Â°C)")
    axs[1].grid(True)

    # Plot only ground truth
    axs[2].plot(time_points, ground_truth, label="Ground Truth", color="orange", alpha=0.7)
    axs[2].set_title("Ground Truth")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Temperature (Â°C)")
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plots
    changer les path si besoin
    plt.savefig(f"results/predictions_three_part.png")
    print(f"Saved predictions plot to results/predictions_three_part.png")

    # Show the plot
    plt.show()

def plot_weather_maps(ground_truth, predictions, lats, lons, title_prefix="", save_path=None):
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
    print(ground_truth)
    print(predictions)
    print(diff)
    # Create latitude and longitude edges (for pcolormesh)
    lats_edges = np.linspace(-90, 90, ground_truth.shape[0] + 1)  # 33 for 32 lat points
    lons_edges = np.linspace(0, 360, ground_truth.shape[1] + 1)  # 65 for 64 lon points

    # Set up the figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    # Titles for each panel
    titles = [
        f"{title_prefix} Average Ground Truth Temperature",
        f"{title_prefix} Average Predicted Temperature",
        f"{title_prefix} Prediction Error (GT - Prediction)"
    ]

    # Data for each panel
    data_list = [ground_truth, predictions, diff]
    cmaps = ["coolwarm", "coolwarm", "RdBu_r"]
    vmin_list = [None, None, -5]  # Only fix vmin/vmax for error map
    vmax_list = [None, None, 5]

    # Loop over the 3 panels
    for i, ax in enumerate(axes):
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        img = ax.pcolormesh(lons_edges, lats_edges, data_list[i], transform=ccrs.PlateCarree(),
                            cmap=cmaps[i], vmin=vmin_list[i], vmax=vmax_list[i])

        plt.colorbar(img, ax=ax, orientation="horizontal", pad=0.05, label="Temperature (K)")
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


def get_variable_channel_index(variables_with_levels, variable, level):
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

        levels = details["levels"]
        print(f"Processing Variable: {var}, Levels: {levels}, Current Channel Index: {channel_index}")

        if var == variable:

            level_index = levels.index(level)  # Check where this level exists
            print(f"Found {variable} at Level {level} â†’ Level Index: {level_index}, Final Channel Index: {channel_index + level_index}")
            return channel_index + levels.index(level)
        channel_index += len(levels)
    raise ValueError(f"Variable {variable} or level {level} not found.")


def test_model(
    model, loader, criterion, device,
    variable, level, desired_lat=10, desired_lon=15, 
    variables_with_levels=None, normalization_stats=None, output_csv="predictions_vs_groundtruth.csv"
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
        tuple: Average test loss, a dictionary of metrics (MAE, RMSE, RÂ²), and lists of predictions and ground truths.
    """
    if variables_with_levels is None:
        raise ValueError("variables_with_levels must be provided.")

    # Get the channel index for the desired variable and level
    channel_index = get_variable_channel_index(variables_with_levels, variable, level)
    #print(f"Compute Channel Index: {channel_index}")

    # Retrieve normalization stats
    if normalization_stats is None or variable not in normalization_stats or str(level) not in normalization_stats[variable]:
        raise ValueError(f"Normalization stats for variable '{variable}' at level {level} not found.")
    mean = normalization_stats[variable][str(level)]["mean"]
    std = normalization_stats[variable][str(level)]["std"]
    print(mean,std)

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


                # Forward pass
                predicted_noise = model(X_batch, torch.tensor([0] * X_batch.size(0), device=device))
                predicted_output = X_batch - predicted_noise  # Reconstruct clean signal

                # Compute loss
                loss = criterion(predicted_output, Y_batch.unsqueeze(1))
                total_loss += loss.item()

                # Extract predictions and ground truth for the specific channel
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

    r_squared = 1 - (ss_res / ss_tot)
    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)
    avg_rmse = total_rmse / len(loader)
    print(f"Results saved in {output_csv}")
    return avg_loss, {"mae": avg_mae, "rmse": avg_rmse, "r2": r_squared}, predicted_values, ground_truth_values, pred_map_avg, gt_map_avg

def train_one_epoch_unet(model, train_loader, optimizer, criterion, noise_scheduler, device, epoch):
    model.train()
    epoch_loss = 0

    for batch_idx, (x, _)  in enumerate(train_loader):
        x = x.to(device)
        true_noise = torch.randn_like(x).to(device)  # True noise
        t = torch.randint(0, noise_scheduler.timesteps, (x.size(0),), device=device)  # Random time steps

        optimizer.zero_grad()
        predicted_noise = model(x, t)  # Forward pass
        loss = criterion(predicted_noise, true_noise)  # Compute loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        '''if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")
'''
    return epoch_loss / len(train_loader)

def validate_one_epoch_unet(model, val_loader, criterion, noise_scheduler, device):
    model.eval()
    val_loss = 0
    all_predictions = []
    all_ground_truth = []
    with torch.no_grad():
        for batch_idx, (x, _)  in enumerate(val_loader):
            x = x.to(device)
            true_noise = torch.randn_like(x).to(device)
            t = torch.randint(0, noise_scheduler.timesteps, (x.size(0),), device=device)

            predicted_noise = model(x, t)  # Forward pass
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
    Calculate MAE, RMSE, and RÂ² metrics.

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

    # RÂ² Score
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {'mae': mae, 'rmse': rmse, 'r2': r2}

class DiffusionModel(nn.Module):
    def __init__(self, unet_model, noise_scheduler):
        super(DiffusionModel, self).__init__()
        self.unet = unet_model
        self.noise_scheduler = noise_scheduler
    
    def forward(self, x, t):

        noise_level = self.noise_scheduler.get_noise_level(t)
        noise_level = noise_level.to(x.device)

        noisy_x = x + noise_level * torch.randn_like(x)
        predicted_noise = self.unet(noisy_x)
        return predicted_noise

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

def run_model_single_gpu(config, train_loader,val_loader,test_loader,number_of_input_for_model, num_epochs=1, learning_rate=0.01, save_plots=True, unique_id=None):
    """
    Train, validate, and test the WeatherPredictor model on a single GPU.

    Args:
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        save_plots (bool): Whether to save prediction plots.
        unique_id (str): Unique identifier for saving results (optional).
    """
    #data_dir = config["data_dir"]
    data_dir = data_dir_temp
    output_dir = config["output_dir"]
    test_years = config["test_years"]
    nc_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")])
    ds = xr.open_mfdataset(nc_files, combine='by_coords')
    test_ds = ds.sel(time=slice(*test_years))
    test_time_points = test_ds['time'].values  # Extract time points as a NumPy array

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Select the levels 1000, 925, and 850
    ds_filtered = ds.sel(level=[1000,925,850])
    ds= ds_filtered
    # Device setup
    print(f"Using device: {device}")
    #testing_years = slice("2003-01-01", "2005-12-31")  # Adjust as per your test range
    # Load normalization stats
    with open(os.path.join(output_dir, "normalization_stats.json"), "r") as f:
        normalization_stats = json.load(f)

    test_variable="t" # Specify the variable ('t' for temperature or 'wind_speed')
    level= "1000"
    #print(normalization_stats)

    mean = normalization_stats[test_variable][level]["mean"]
    std = normalization_stats[test_variable][level]["std"]
    print(mean)
    print(std)
    input_channels=number_of_input_for_model


    # Instantiate components for UNET
    unet = UNet(input_channels=input_channels, output_channels=input_channels, base_channels=64)
    noise_scheduler = NoiseScheduler(timesteps=1000)
    diffusion_model = DiffusionModel(unet, noise_scheduler).to(device)
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



     ####to de-nomalized the output:




    # Test model and collect predictions

    test_loss, metrics, predictions, ground_truth, pred_map, gt_map = test_model(
        model=diffusion_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        variable=test_variable,  # Temperature
        level=level,  # Pressure level in hPa
        desired_lat=10,  # Latitude index
        desired_lon=15,  # Longitude index
        variables_with_levels=config["variables_with_levels"],
        normalization_stats=normalization_stats,
        output_csv="results_temperature_1000hPa.csv"
    )

    #test_loss, test_metrics, predicted_temperatures, ground_truth_temperatures = test_model(diffusion_model, test_loader, criterion, device, desired_level=0, desired_lat=10, desired_lon=15, output_csv="predictions_vs_groundtruth_real.csv",    mean=mean, std=std)   

    print(f"\nTest Results - Loss: {test_loss:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, RÂ²: {metrics['r2']:.4f}")

    #unnormalized_ground_truth, unnormalized_predicted = unnormalize_predictions(ground_truth, predictions, mean, std)
    #no need as we get unnormalized from the test_model function
    # Save prediction plots
    #if save_plots:
        #plot_predictions(predicted_temperatures, ground_truth_temperatures, test_time_points, unique_id)
    if save_plots:
        plot_predictions(predictions, ground_truth, test_time_points, unique_id)


    # Generate lat/lon grids based on dataset resolution
    lats = np.linspace(-90, 90, pred_map.shape[0])  # Adjust based on dataset
    lons = np.linspace(0, 360, pred_map.shape[1])  # Adjust range
    image_compar_path=output_dir+"/"+"comparison.png"
    print("Latitude grid:", lats)
    print("Longitude grid:", lons)
    # Call the visualization function
    plot_weather_maps(gt_map, pred_map, lats, lons, title_prefix="Test Period (T 1000 hPa)",save_path=image_compar_path)


changer les path

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
    "train_years": ("1979-01-01", "1999-12-31"),
    "val_years": ("1982-01-01", "1985-12-31"),
    "test_years": ("2010-01-01", "2010-06-30"),#"test_years": ("2010-01-01", "2012-12-31"),
}


data_creation_unet(config)
train_loader,val_loader,test_loader, number_of_input_for_model = data_loading()
run_model_single_gpu(config,train_loader,val_loader,test_loader,number_of_input_for_model, num_epochs=1, learning_rate=0.001)


#total years: 1979-2020
