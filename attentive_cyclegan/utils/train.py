import torch
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import torch.nn as nn
import torch.optim as optim
import time
import warnings
from itertools import zip_longest
from torch.utils.tensorboard import SummaryWriter
import os
# Custom imports
from models.generator import Generator
from models.discriminator import Discriminator
from utils.loss import (
    cycle_consistency_loss, 
    identity_loss, 
    hinge_loss_discriminator, 
    hinge_loss_generator, 
    perceptual_loss
)

# Suppress warnings
warnings.filterwarnings("ignore")
# ✅ Enable PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def train(source, target, batch_size, source_state, target_state, num_epochs, device="cuda", log_dir="tb_logs"):
    """
    Trains the CycleGAN model using DataParallel for multi-GPU support.

    Args:
        source (DataLoader): Dataloader for domain X (source images).
        target (DataLoader): Dataloader for domain Y (target images).
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        device (str): Device to run training on ("cuda" or "cpu").
        log_dir (str): Directory to store TensorBoard logs.
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Using {num_gpus} GPUs for training.")

    # Instantiate models
    generator_g = Generator(img_channels=3).to(device)  # G: X → Y
    generator_f = Generator(img_channels=3).to(device)  # F: Y → X
    discriminator_x = Discriminator().to(device)  # D_X
    discriminator_y = Discriminator().to(device)  # D_Y

    # Enable multi-GPU training
    if num_gpus > 1:
        generator_g = nn.DataParallel(generator_g)
        generator_f = nn.DataParallel(generator_f)
        discriminator_x = nn.DataParallel(discriminator_x)
        discriminator_y = nn.DataParallel(discriminator_y)

    # Optimizers
    lr = 1e-4
    betas = (0.5, 0.999)
    optimizer_G = optim.Adam(list(generator_g.parameters()) + list(generator_f.parameters()), lr=lr, betas=betas)
    optimizer_D = optim.Adam(list(discriminator_x.parameters()) + list(discriminator_y.parameters()), lr=lr, betas=betas)

    # Learning Rate Scheduler (Linear Decay after 30 epochs)
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - 30) / 130 

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda)

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📜 Training started for {num_epochs} epochs on {device}...")

    # Training Loop
    for epoch in range(num_epochs):
        start_time = time.time()
        g_loss_total, d_loss_total = 0, 0
        cycle_loss_total, identity_loss_total, perceptual_loss_total = 0, 0, 0

        for real_x_batch, real_y_batch in zip_longest(source, target, fillvalue=None):
            if real_x_batch is None or real_y_batch is None:
                continue

            real_x, real_y = real_x_batch[0].to(device), real_y_batch[0].to(device)

            # ------------------------
            # Train Generators G and F
            # ------------------------

            fake_y = generator_g(real_x)  # G(X)
            fake_x = generator_f(real_y)  # F(Y)

            # Identity loss (G(Y) ≈ Y and F(X) ≈ X)
            id_loss_x = identity_loss(real_x, generator_f(real_x))
            id_loss_y = identity_loss(real_y, generator_g(real_y))

            # Adversarial loss with hinge loss
            adv_loss_g = hinge_loss_generator(discriminator_y(fake_y))
            adv_loss_f = hinge_loss_generator(discriminator_x(fake_x))

            # Cycle-consistency loss
            cycle_x = generator_f(fake_y)  # F(G(X)) ≈ X
            cycle_y = generator_g(fake_x)  # G(F(Y)) ≈ Y
            cycle_loss_x = cycle_consistency_loss(real_x, cycle_x)
            cycle_loss_y = cycle_consistency_loss(real_y, cycle_y)

            # Perceptual loss
            perceptual_loss_x = perceptual_loss(fake_x, real_x)
            perceptual_loss_y = perceptual_loss(fake_y, real_y)

            # Total generator loss
            total_g_loss = adv_loss_g + adv_loss_f + cycle_loss_x + cycle_loss_y + id_loss_x + id_loss_y + perceptual_loss_x + perceptual_loss_y

            optimizer_G.zero_grad()
            total_g_loss.backward()
            optimizer_G.step()

            # -------------------------
            # Train Discriminators X, Y
            # -------------------------

            real_pred_x = discriminator_x(real_x)
            fake_pred_x = discriminator_x(fake_x.detach())
            dx_loss = hinge_loss_discriminator(real_pred_x, fake_pred_x)

            real_pred_y = discriminator_y(real_y)
            fake_pred_y = discriminator_y(fake_y.detach())
            dy_loss = hinge_loss_discriminator(real_pred_y, fake_pred_y)

            total_d_loss = dx_loss + dy_loss

            optimizer_D.zero_grad()
            total_d_loss.backward()
            optimizer_D.step()

            # Accumulate losses
            g_loss_total += total_g_loss.item()
            d_loss_total += total_d_loss.item()
            cycle_loss_total += cycle_loss_x.item() + cycle_loss_y.item()
            identity_loss_total += id_loss_x.item() + id_loss_y.item()
            perceptual_loss_total += perceptual_loss_x.item() + perceptual_loss_y.item()

        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()

        epoch_time = time.time() - start_time

        # Log losses to TensorBoard
        writer.add_scalar('Loss/Generators', g_loss_total, epoch + 1)
        writer.add_scalar('Loss/Discriminators', d_loss_total, epoch + 1)
        writer.add_scalar('Loss/Cycle_Consistency', cycle_loss_total, epoch + 1)
        writer.add_scalar('Loss/Identity', identity_loss_total, epoch + 1)
        writer.add_scalar('Loss/Perceptual', perceptual_loss_total, epoch + 1)

        # Print all losses per epoch
        print("-" * 60)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time/60:.2f} minutes.")
        print(f"  Generator Loss: {g_loss_total:.4f}")
        print(f"  Discriminator Loss: {d_loss_total:.4f}")
        print(f"  Cycle Consistency Loss: {cycle_loss_total:.4f}")
        print(f"  Identity Loss: {identity_loss_total:.4f}")
        print(f"  Perceptual Loss: {perceptual_loss_total:.4f}")
        print("-" * 60)

        # Save model checkpoints every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(generator_g.state_dict(), f'weights/generator_{source_state}_to_{target_state}_{epoch+1}.pth')
            torch.save(generator_f.state_dict(), f'weights/generator_{target_state}_to_{source_state}_{epoch+1}.pth')

    print("✅ Training Complete!")