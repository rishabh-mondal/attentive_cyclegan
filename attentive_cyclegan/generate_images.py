import torch
import os
import torchvision.utils as vutils
from utils.dataset import load_images
from models.generator import Generator  # Ensure correct import



source_state_name = "west_bengal"
target_state_name = "haryana"




# Set paths manually
WEIGHTS_PATH = f"/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/attentive_cyclegan/weights/generator_{source_state_name}_to_{target_state_name}_40.pth"
SOURCE_DIR = f"/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/region_performance/{source_state_name}_same_class_count_10_120_1000/images"
OUTPUT_DIR = f"/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/synthetic_data/{source_state_name}_to_{target_state_name}/images"

# Define hyperparameters manually
BATCH_SIZE = 1
GPU_ID = 0  # Set GPU index manually

# Set device (use specified GPU if available)
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

# Function to denormalize images (assuming normalization to [-1, 1])
def denormalize(tensor):
    return (tensor * 0.5) + 0.5  # Convert back to [0,1] range

# Function to save generated images
def save_image(tensor, path):
    """Saves a PyTorch tensor as an image file."""
    tensor = denormalize(tensor).clamp(0, 1)  # Denormalize and clamp values
    vutils.save_image(tensor, path)

# Function to generate and save images
def generate_and_save_images(generator, dataloader, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

    for example_input, img_path in dataloader:  # Get both image tensor & path
        example_input = example_input.to(device)  # Move to device

        with torch.no_grad():
            generated_image = generator(example_input)  # Generate image

        # Extract original filename (without extension)
        original_filename = os.path.basename(img_path[0])  # First item from batch
        filename_without_ext = os.path.splitext(original_filename)[0]

        # Save generated image with same filename
        save_path = os.path.join(output_dir, f"{filename_without_ext}.png")
        save_image(generated_image, save_path)
        print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    print(f"🚀 Using GPU: {GPU_ID} | Loading model from: {WEIGHTS_PATH}")

    # Load dataset
    source_dataloader = load_images(SOURCE_DIR, batch_size=BATCH_SIZE)

    # Load Generator model
    generator_g = Generator(img_channels=3).to(device)
    # generator_g.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        # Load model weights, handling DataParallel-trained models
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)

    if "module." in list(state_dict.keys())[0]:  # Check if DataParallel wrapper exists
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict

    generator_g.load_state_dict(state_dict)

    generator_g.eval()

    # Generate images
    print(f"🚀 Generating images and saving to {OUTPUT_DIR}...")
    generate_and_save_images(generator_g, source_dataloader, OUTPUT_DIR, device)
    print("✅ Image generation complete.")
