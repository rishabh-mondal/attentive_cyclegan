import torch
import os
import datetime
from utils.dataset import load_images
from utils.train import train

BATCH_SIZE = 2
EPOCHS = 160

# Automatically detect available GPUs
NUM_GPUS = torch.cuda.device_count()
DEVICE = "cuda" if torch.cuda.is_available() and NUM_GPUS > 0 else "cpu"

# Print GPU information
# if DEVICE == "cuda":
#     print(f"🚀 Detected {NUM_GPUS} GPUs:")
#     for i in range(NUM_GPUS):
#         print(f"  {i}: {torch.cuda.get_device_name(i)}")
    # os.system("nvidia-smi")  # Show detailed GPU status

# Define dataset directories
SOURCE_DIR = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/region_performance/west_bengal_same_class_count_10_120_1000/images"
TARGET_DIR = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/region_performance/haryana_same_class_count_10_120_1000/images"

source_dataloader = load_images(SOURCE_DIR, BATCH_SIZE)
target_dataloader = load_images(TARGET_DIR, BATCH_SIZE)


# Generate unique log directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = f"tb_logs_directory/{timestamp}"
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists

# Clear CUDA cache (helpful if running on GPUs with large models)
if DEVICE == "cuda":
    torch.cuda.empty_cache()

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
    print("✅ Training completed successfully!")

