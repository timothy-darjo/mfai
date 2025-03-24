"""
Implementation of Villon S.'s diffusion-based model with UNet
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from mfai.torch.models.base import ModelABC, ModelType
from torch import nn

@dataclass_json
@dataclass(slots=True)
class UnetDiffusionSettings:
    num_filters: int = 64
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Identity"

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