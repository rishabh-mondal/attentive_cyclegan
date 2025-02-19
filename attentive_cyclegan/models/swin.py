import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, use_checkpoint=False, fp16=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  
        self.use_checkpoint = use_checkpoint
        self.fp16 = fp16
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape 
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # Reshaping for attention
        
        # Function for the main forward pass
        def _forward(x):
            residual = x
            x = self.norm1(x)

            # Apply Window-based Attention with cyclic shift if necessary
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))  # Cyclic shift
            
            # Compute attention
            x, _ = self.attn(x, x, x)  # Attention mechanism
            x = x + residual  # Add residual

            residual = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = x + residual  # Add residual after MLP
            return x

        # Use activation checkpointing if enabled
        if self.use_checkpoint:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                x = checkpoint.checkpoint(_forward, x)
        else:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                x = _forward(x)

        # Convert back to image format (B, C, H, W)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x
