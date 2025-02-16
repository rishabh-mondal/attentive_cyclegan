import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, use_checkpoint=False, fp16=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint  # ✅ Enable checkpointing
        self.fp16 = fp16  # ✅ Enable mixed precision

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Convert to sequence format for multihead attention
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch, dim)

        # ✅ Define _forward inside forward so it has access to x
        def _forward(x):
            residual = x
            x = self.norm1(x)
            x, _ = self.attn(x, x, x)
            x = x + residual

            residual = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = x + residual
            return x

        # ✅ Use activation checkpointing correctly
        if self.use_checkpoint:
            with torch.cuda.amp.autocast(enabled=self.fp16):  # Enable FP16 if needed
                x = checkpoint.checkpoint(_forward, x)
        else:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                x = _forward(x)

        # Convert back to image format
        x = x.permute(1, 2, 0).view(B, C, H, W)  # Restore original shape
        return x
