import torch
from utils.dataset import load_images
from utils.train import train

# Define configurable hyperparameters
BATCH_SIZE = 4
EPOCHS = 160

# Automatically detect available GPUs
NUM_GPUS = torch.cuda.device_count()
DEVICE = "cuda" if NUM_GPUS > 0 else "cpu"

# Define dataset directories
SOURCE_DIR = "/home/umang.shikarvar/distance_exp/west_bengal_same_class_count_10_120_1000/images"
TARGET_DIR = "/home/umang.shikarvar/distance_exp/haryana_same_class_count_10_120_1000/images"

# Load datasets with batch size from main.py
source_dataloader = load_images(SOURCE_DIR, BATCH_SIZE)
target_dataloader = load_images(TARGET_DIR, BATCH_SIZE)

if __name__ == "__main__":
    print(f"Using {NUM_GPUS} GPUs for training on {DEVICE} with batch size {BATCH_SIZE} and {EPOCHS} epochs...")

    # Start training with multiple GPUs using DataParallel
    train(
        source=source_dataloader,
        target=target_dataloader,
        batch_size=BATCH_SIZE,
        source_state="west_bengal",
        target_state="haryana",
        num_epochs=EPOCHS,
        device=DEVICE
    )
