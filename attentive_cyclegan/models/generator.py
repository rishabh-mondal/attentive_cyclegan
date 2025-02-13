import torch
import torch.nn as nn
from models.attention import SelfAttention, SEBlock  # Import attention layers

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=6):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.downsampling_layers = nn.ModuleList([
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
        ])
        self.attention = SelfAttention(num_features * 4)  # Apply self-attention

        self.residual_layers = nn.Sequential(*[nn.Conv2d(num_features * 4, num_features * 4, 3, 1, 1) for _ in range(num_residuals)])
        self.channel_attention = SEBlock(num_features * 4)  # SE Block for better feature learning

        self.upsampling_layers = nn.ModuleList([
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(num_features * 2, num_features, 3, stride=2, padding=1, output_padding=1),
        ])
        self.last_layer = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.attention(x)
        x = self.residual_layers(x)
        x = self.channel_attention(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        return torch.tanh(self.last_layer(x))
