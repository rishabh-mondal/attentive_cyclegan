import torch
import torch.nn as nn

# Define loss functions
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

def adversarial_loss(pred, target):
    """
    Computes adversarial loss using MSE.

    Args:
        pred (torch.Tensor): Discriminator output.
        target (torch.Tensor): Target labels (real or fake).

    Returns:
        torch.Tensor: Loss value.
    """
    return mse_loss(pred, target)


def cycle_consistency_loss(real, cycled, lambda_cycle=7.0):
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


def identity_loss(real, same, lambda_identity=1.0):
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
