import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from models.attention import SelfAttention  # Import Self-Attention

class DiscriminatorBlock(nn.Module):
    """Basic Block for Discriminator with Spectral Normalization"""
    def __init__(self, in_channels, out_channels, stride, use_norm=True):
        super().__init__()
        layers = [
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1))
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))  # Add normalization except first layer
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    """PatchGAN Discriminator with Spectral Normalization and Self-Attention"""
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], use_attention=True):
        super().__init__()

        # Initial Convolution (No Instance Norm for first layer)
        self.initial_layer = DiscriminatorBlock(in_channels, features[0], stride=2, use_norm=False)

        layers = []
        in_channels = features[0]

        # Apply Self-Attention in the middle layers for better feature learning
        self.use_attention = use_attention
        for i, feature in enumerate(features[1:]):
            layers.append(DiscriminatorBlock(in_channels, feature, stride=2 if feature != features[-1] else 1, use_norm=(feature != 1)))
            if self.use_attention and feature == 256:  # Apply attention at 256 channels (adjustable)
                layers.append(SelfAttention(feature))  # No need to apply spectral_norm here anymore
            in_channels = feature

        self.feature_layers = nn.Sequential(*layers)

        # Final Classification Layer (PatchGAN Output)
        self.final_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)),
            nn.Sigmoid()  # Add activation for probability output
        )

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.feature_layers(x)
        return self.final_layer(x)
