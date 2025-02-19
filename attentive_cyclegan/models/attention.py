import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class SelfAttention(nn.Module):
    """Self-Attention Mechanism for capturing global dependencies."""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        # Apply spectral normalization to Conv2d layers directly
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        
        self.softmax = nn.Softmax(dim=-1)  # Softmax on the last dimension (H*W)

    def forward(self, x):
        batch, C, H, W = x.size()

        # Reshape for multi-head attention compatibility: [B, H*W, C//8]
        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)  # [B, H*W, C//8]
        key = self.key(x).view(batch, -1, H * W)  # [B, C//8, H*W]
        attention = self.softmax(torch.bmm(query, key))  # Compute attention map (B, H*W, H*W)

        value = self.value(x).view(batch, -1, H * W)  # [B, C, H*W]
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, C, H, W)  # Apply attention to values

        return x + out  # Residual connection

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention."""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, C, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, C)  # Global pooling
        y = self.fc(y).view(batch, C, 1, 1)  # Fully connected layers
        return x * y.expand_as(x)  # Scale input by attention weights

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM) - Channel + Spatial Attention."""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention
        self.se = SEBlock(in_channels, reduction=reduction)

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply SE block (channel attention)
        x_se = self.se(x)

        # Compute Spatial Attention
        max_pool = torch.max(x_se, dim=1, keepdim=True)[0]  # Max pooling
        avg_pool = torch.mean(x_se, dim=1, keepdim=True)  # Avg pooling
        spatial_attention = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))

        return x_se * spatial_attention  # Apply spatial attention to channel-refined features
