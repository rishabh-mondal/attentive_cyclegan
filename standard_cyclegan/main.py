import torch
from utils.dataset import load_images
from utils.train import train_cyclegan

# Set device to GPU or CPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Load datasets
source = load_images('/home/umang.shikarvar/distance_exp/west_bengal_same_class_count_10_120_1000/images')
target = load_images('/home/umang.shikarvar/distance_exp/haryana_same_class_count_10_120_1000/images')

# Train CycleGAN
train_cyclegan(source, target, device, epochs=200)