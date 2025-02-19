import torch
print(f"PyTorch sees {torch.cuda.device_count()} GPUs")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated")

import torch
print(torch.__version__)  # PyTorch version
print(torch.version.cuda)  # CUDA version PyTorch was built with
print(torch.cuda.is_available())  # Check if CUDA is available
import torch
print(torch.version.cuda)  # PyTorch CUDA version
print(torch.backends.cudnn.version())  # cuDNN version


import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! Using GPU: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU instead.")

# Check CUDA device properties
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
