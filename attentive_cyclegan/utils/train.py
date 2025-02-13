import torch
import torch.nn as nn
import torch.optim as optim
from itertools import zip_longest
# from torch.utils.tensorboard import SummaryWriter
from models.generator import Generator
from models.discriminator import Discriminator

from utils.loss import adversarial_loss, cycle_consistency_loss, identity_loss
import warnings
import time
# Suppress all warnings
warnings.filterwarnings("ignore")

def train(source, target, batch_size, num_epochs, device="cuda", log_dir="tb_logs"):
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
    
    print(f"Using {num_gpus} GPUs for training.")

    # Instantiate models
    generator_g = Generator(img_channels=3).to(device)  # G: X → Y
    generator_f = Generator(img_channels=3).to(device)  # F: Y → X
    discriminator_x = Discriminator().to(device)  # Discriminator for domain X
    discriminator_y = Discriminator().to(device)  # Discriminator for domain Y

    # Use DataParallel if multiple GPUs are available
    if num_gpus > 1:
        generator_g = nn.DataParallel(generator_g)
        generator_f = nn.DataParallel(generator_f)
        discriminator_x = nn.DataParallel(discriminator_x)
        discriminator_y = nn.DataParallel(discriminator_y)

    # Optimizers
    lr = 2e-4
    betas = (0.5, 0.999)
    generator_g_optimizer = optim.Adam(generator_g.parameters(), lr=lr, betas=betas)
    generator_f_optimizer = optim.Adam(generator_f.parameters(), lr=lr, betas=betas)
    discriminator_x_optimizer = optim.Adam(discriminator_x.parameters(), lr=lr, betas=betas)
    discriminator_y_optimizer = optim.Adam(discriminator_y.parameters(), lr=lr, betas=betas)

    # Learning rate schedulers
    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - 100) / 100  

    scheduler_g = optim.lr_scheduler.LambdaLR(generator_g_optimizer, lr_lambda=lr_lambda)
    scheduler_f = optim.lr_scheduler.LambdaLR(generator_f_optimizer, lr_lambda=lr_lambda)
    scheduler_dx = optim.lr_scheduler.LambdaLR(discriminator_x_optimizer, lr_lambda=lr_lambda)
    scheduler_dy = optim.lr_scheduler.LambdaLR(discriminator_y_optimizer, lr_lambda=lr_lambda)

    # Initialize TensorBoard
    # writer = SummaryWriter(log_dir=log_dir)

    print(f"Training started for {num_epochs} epochs on {device}...")

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        g_loss_total, f_loss_total, dx_loss_total, dy_loss_total = 0, 0, 0, 0
        cycle_loss_total, identity_loss_total = 0, 0

        for real_x_batch, real_y_batch in zip_longest(source, target, fillvalue=None):
            if real_x_batch is None or real_y_batch is None:
                continue

            real_x, real_y = real_x_batch[0].to(device), real_y_batch[0].to(device)

            # ------------------------
            # Train Generators G and F
            # ------------------------

            # Identity loss (G(Y) ≈ Y and F(X) ≈ X)
            identity_x = generator_f(real_x)
            identity_y = generator_g(real_y)
            id_loss_x = identity_loss(real_x, identity_x)
            id_loss_y = identity_loss(real_y, identity_y)

            # Adversarial loss
            fake_y = generator_g(real_x)  # G(X)
            fake_x = generator_f(real_y)  # F(Y)

            adv_loss_g = adversarial_loss(discriminator_y(fake_y), torch.ones_like(discriminator_y(fake_y)))
            adv_loss_f = adversarial_loss(discriminator_x(fake_x), torch.ones_like(discriminator_x(fake_x)))

            # Cycle-consistency loss
            cycle_x = generator_f(fake_y)  # F(G(X)) ≈ X
            cycle_y = generator_g(fake_x)  # G(F(Y)) ≈ Y
            cycle_loss_x = cycle_consistency_loss(real_x, cycle_x)
            cycle_loss_y = cycle_consistency_loss(real_y, cycle_y)

            # Total generator loss
            total_g_loss = adv_loss_g + cycle_loss_x + id_loss_y
            total_f_loss = adv_loss_f + cycle_loss_y + id_loss_x

            generator_g_optimizer.zero_grad()
            generator_f_optimizer.zero_grad()
            total_g_loss.backward(retain_graph=True)
            total_f_loss.backward()
            generator_g_optimizer.step()
            generator_f_optimizer.step()

            # -------------------------
            # Train Discriminators X, Y
            # -------------------------

            # Discriminator X loss
            real_loss_x = adversarial_loss(discriminator_x(real_x), torch.ones_like(discriminator_x(real_x)))
            fake_loss_x = adversarial_loss(discriminator_x(fake_x.detach()), torch.zeros_like(discriminator_x(fake_x)))
            dx_loss = (real_loss_x + fake_loss_x) * 0.5

            discriminator_x_optimizer.zero_grad()
            dx_loss.backward()
            discriminator_x_optimizer.step()

            # Discriminator Y loss
            real_loss_y = adversarial_loss(discriminator_y(real_y), torch.ones_like(discriminator_y(real_y)))
            fake_loss_y = adversarial_loss(discriminator_y(fake_y.detach()), torch.zeros_like(discriminator_y(fake_y)))
            dy_loss = (real_loss_y + fake_loss_y) * 0.5

            discriminator_y_optimizer.zero_grad()
            dy_loss.backward()
            discriminator_y_optimizer.step()

            # Accumulate losses
            g_loss_total += total_g_loss.item()
            f_loss_total += total_f_loss.item()
            dx_loss_total += dx_loss.item()
            dy_loss_total += dy_loss.item()
            cycle_loss_total += cycle_loss_x.item() + cycle_loss_y.item()
            identity_loss_total += id_loss_x.item() + id_loss_y.item()

        scheduler_g.step()
        scheduler_f.step()
        scheduler_dx.step()
        scheduler_dy.step()
        epoch_time = time.time() - start_time


        # # Log losses to TensorBoard
        # writer.add_scalar('Loss/Generator_G', g_loss_total, epoch + 1)
        # writer.add_scalar('Loss/Generator_F', f_loss_total, epoch + 1)
        # writer.add_scalar('Loss/Discriminator_X', dx_loss_total, epoch + 1)
        # writer.add_scalar('Loss/Discriminator_Y', dy_loss_total, epoch + 1)
        # writer.add_scalar('Loss/Cycle_Consistency', cycle_loss_total, epoch + 1)
        # writer.add_scalar('Loss/Identity', identity_loss_total, epoch + 1)

        # Print all losses per epoch
        print("-" * 60)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time/60:.2f} minutes.")
        print(f"  Generator_G Loss: {g_loss_total:.4f}")
        print(f"  Generator_F Loss: {f_loss_total:.4f}")
        print(f"  Discriminator_X Loss: {dx_loss_total:.4f}")
        print(f"  Discriminator_Y Loss: {dy_loss_total:.4f}")
        print(f"  Cycle Consistency Loss: {cycle_loss_total:.4f}")
        print(f"  Identity Loss: {identity_loss_total:.4f}")
        print("-" * 60)

            # Save model checkpoints every 10 epochs
    if (epoch + 1) % 20 == 0:
        torch.save(generator_g.state_dict(), f'/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/attentive_cyclegan/weights/generator_WB_to_Haryana_{epoch+1}.pth')
        torch.save(generator_f.state_dict(), f'/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/attentive_cyclegan/weights/generator_Haryana_to_WB_{epoch+1}.pth')

    print("Training Complete!")
