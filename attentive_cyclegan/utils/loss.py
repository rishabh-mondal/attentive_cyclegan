import torch
import torch.nn as nn
import lpips  
import torchvision.models as models
import torch.nn.functional as F

# Define loss functions
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

# Initialize LPIPS loss (only once, globally)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_loss = lpips.LPIPS(net='vgg').to(device)

def hinge_loss_discriminator(real_pred, fake_pred):
    """
    Computes Hinge Loss for the Discriminator.

    Args:
        real_pred (torch.Tensor): Discriminator output for real images.
        fake_pred (torch.Tensor): Discriminator output for fake images.

    Returns:
        torch.Tensor: Hinge loss for the discriminator.
    """
    real_loss = F.relu(1 - real_pred).mean()  # Max(0, 1 - D(real))
    fake_loss = F.relu(1 + fake_pred).mean()  # Max(0, 1 + D(fake))
    return (real_loss + fake_loss) / 2  # Average loss

def hinge_loss_generator(fake_pred):
    """
    Computes Hinge Loss for the Generator.

    Args:
        fake_pred (torch.Tensor): Discriminator output for fake images.

    Returns:
        torch.Tensor: Hinge loss for the generator.
    """
    return -fake_pred.mean()  # Maximizing D(fake)

def cycle_consistency_loss(real, cycled, lambda_cycle=5.0):
    """
    Computes cycle consistency loss (L1 loss between real and cycled images).

    Args:
        real (torch.Tensor): Original image.
        cycled (torch.Tensor): Reconstructed image.
        lambda_cycle (float): Weight factor for cycle loss.

    Returns:
        torch.Tensor: Cycle consistency loss.
    """
    return l1_loss(real, cycled) * lambda_cycle

def identity_loss(real, same, lambda_identity=0.3):
    """
    Computes identity loss (L1 loss between real and identity-mapped image).

    Args:
        real (torch.Tensor): Real image.
        same (torch.Tensor): Identity image (G(Y) ≈ Y or F(X) ≈ X).
        lambda_identity (float): Weight factor for identity loss.

    Returns:
        torch.Tensor: Identity loss.
    """
    return l1_loss(real, same) * lambda_identity

class VGGPerceptualLoss(nn.Module):
    """
    Computes Perceptual Loss using VGG-19 feature maps.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # Initialize VGG model (only once)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16]  # Use first 16 layers
        vgg.eval()  # Set VGG to eval mode
        vgg.to(self.device)  # Move VGG model to device

        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters

        self.vgg = vgg
        self.l1_loss = nn.L1Loss()

    def forward(self, fake, real):
        """
        Args:
            fake (torch.Tensor): Generated image.
            real (torch.Tensor): Ground truth image.

        Returns:
            torch.Tensor: Perceptual loss.
        """
        fake, real = fake.to(self.device), real.to(self.device)  # Ensure inputs are on the correct device
        return self.l1_loss(self.vgg(fake), self.vgg(real))

def perceptual_loss(fake, real, lambda_perceptual=0.1):
    """
    Computes VGG Perceptual Loss and LPIPS loss.

    Args:
        fake (torch.Tensor): Generated image.
        real (torch.Tensor): Ground truth image.
        lambda_perceptual (float): Weight factor for perceptual loss.

    Returns:
        torch.Tensor: Combined perceptual loss.
    """
    # Use pre-initialized LPIPS and VGG loss
    vgg_loss_fn = VGGPerceptualLoss(device=fake.device)  # Instantiate VGG loss on the correct device

    # Combined perceptual loss
    return lambda_perceptual * (vgg_loss_fn(fake, real) + lpips_loss(fake, real).mean())
