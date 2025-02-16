import torch
import torch.nn as nn
from models.attention import SelfAttention, SEBlock, CBAM  # Import attention layers
from models.swin import SwinTransformerBlock  # Import Swin Transformer block

class ResidualBlock(nn.Module):
    """Residual Block with Instance Normalization"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection

class AdaIN(nn.Module):
    """Adaptive Instance Normalization (AdaIN)"""
    def __init__(self, num_features):
        super().__init__()
        self.scale = nn.Linear(512, num_features)
        self.bias = nn.Linear(512, num_features)

    def forward(self, x, style):
        gamma = self.scale(style).unsqueeze(2).unsqueeze(3)
        beta = self.bias(style).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

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

        self.attention = SelfAttention(num_features * 4)  # Apply Self-Attention

        self.residual_layers = nn.Sequential(*[ResidualBlock(num_features * 4) for _ in range(num_residuals)])

        self.channel_attention = CBAM(num_features * 4)  # CBAM for better feature selection

        self.swin_transformer = SwinTransformerBlock(dim=num_features * 4, 
                                                     num_heads=4, 
                                                     window_size=4,
                                                     use_checkpoint=True,
                                                     fp16=True)  # Swin Transformer Block

        self.adain = AdaIN(num_features * 4)  # Adaptive Instance Normalization

        self.upsampling_layers = nn.ModuleList([
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(num_features * 2, num_features, 3, stride=2, padding=1, output_padding=1),
        ])

        self.last_layer = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x, style=None):
        x = self.initial_layer(x)

        for layer in self.downsampling_layers:
            x = layer(x)

        x = self.attention(x)  # Self-Attention Layer
        x = self.residual_layers(x)  # Residual Blocks
        x = self.channel_attention(x)  # Channel Attention
        x = self.swin_transformer(x)  # Swin Transformer Block

        if style is not None:
            x = self.adain(x, style)  # Apply AdaIN if style is provided

        for layer in self.upsampling_layers:
            x = layer(x)

        return torch.tanh(self.last_layer(x))

# ✅ Enable Multi-GPU Training
def build_generator(img_channels, num_features=64, num_residuals=6):
    model = Generator(img_channels, num_features, num_residuals)
    if torch.cuda.device_count() > 1:
        print(f"✅ Using {torch.cuda.device_count()} GPUs for Generator")
        model = nn.DataParallel(model)  # Wrap in DataParallel
    return model
