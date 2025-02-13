import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from models.attention import SelfAttention  # Import self-attention

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect")),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(spectral_norm(nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            if feature == 256:
                layers.append(SelfAttention(feature))  # Add self-attention at 256 channels

            in_channels = feature

        layers.append(spectral_norm(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(self.initial_layer(x))
