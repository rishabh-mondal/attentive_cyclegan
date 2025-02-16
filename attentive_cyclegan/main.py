import torch
import os
import datetime
from utils.dataset import load_images
from utils.train import train

# Define configurable hyperparameters
BATCH_SIZE = 1
EPOCHS = 160

# Automatically detect available GPUs
NUM_GPUS = torch.cuda.device_count()
DEVICE = "cuda" if torch.cuda.is_available() and NUM_GPUS > 0 else "cpu"

# Print GPU information
# if DEVICE == "cuda":
#     print(f"🚀 Detected {NUM_GPUS} GPUs:")
#     os.system("nvidia-smi")  # Show GPU details

# Define dataset directories
SOURCE_DIR = "/home/umang.shikarvar/distance_exp/west_bengal_same_class_count_10_120_1000/images"
TARGET_DIR = "/home/umang.shikarvar/distance_exp/haryana_same_class_count_10_120_1000/images"

# Load datasets
try:
    source_dataloader = load_images(SOURCE_DIR, BATCH_SIZE)
    target_dataloader = load_images(TARGET_DIR, BATCH_SIZE)
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    exit(1)

# Generate unique log directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = f"tb_logs_directory/{timestamp}"
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists

if __name__ == "__main__":
    print("=" * 60)
    print(f"🚀 Starting CycleGAN Training")
    print(f"🔹 Using {DEVICE} ({NUM_GPUS} GPUs available)")
    print(f"🔹 Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"🔹 Logging to: {LOG_DIR}")
    print("=" * 60)

    # Start training with multiple GPUs using DataParallel
    train(
        source=source_dataloader,
        target=target_dataloader,
        batch_size=BATCH_SIZE,
        source_state="west_bengal",
        target_state="haryana",
        num_epochs=EPOCHS,
        device=DEVICE,
        log_dir=LOG_DIR
    )
