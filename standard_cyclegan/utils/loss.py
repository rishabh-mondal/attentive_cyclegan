import torch
import torch.nn as nn

# Adversarial loss (for GANs)
adversarial_loss = nn.MSELoss()

# Cycle-consistency loss
def cycle_consistency_loss(real, reconstructed, lambda_cycle=10):
    return lambda_cycle * torch.mean(torch.abs(real - reconstructed))

# Identity loss
def identity_loss(real, same, lambda_identity=5):
    return lambda_identity * torch.mean(torch.abs(real - same))