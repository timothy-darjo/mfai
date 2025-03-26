"""
Implementation of Villon S.'s diffusion-based model with UNet
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from mfai.torch.models.base import ModelABC, ModelType
from torch import nn

from mfai.torch.models.unet import UNet #importing base unet from unet.py. Replaces UNet class in original code

def diffusion_loss(predicted_noise, true_noise):
    #Mean Squared Error loss for predicted vs true noise
    return torch.mean((predicted_noise - true_noise) ** 2)

class UNet(nn.Module): #copied from original code; maybe redundant with unet.py class
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

class NoiseScheduler: #copied from original code
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

@dataclass_json
@dataclass(slots=True)
class UnetDiffusionSettings:
    num_filters: int = 64
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Identity"

    base_channels: int = 64

class UnetDiffusion(ModelABC, nn.Module):
    settings_kls = UnetDiffusionSettings
    onnx_supported: bool = True
    supported_num_spatial_dims = (2,)
    num_spatial_dims: int = 2
    features_last: bool = False
    model_type: int = ModelType.CONVOLUTIONAL
    register: bool = True
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: Union[None, Tuple[int, int]] = None,
        settings: UnetDiffusionSettings,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings

        self.unet = UNet(input_channels=in_channels, output_channels=out_channels, base_channels=settings.base_channels)
        self.noise_scheduler = NoiseScheduler(timesteps=1000)