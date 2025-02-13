import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Conv2d(in_channels, feature, kernel_size=4, stride=2 if feature != features[-1] else 1, padding=1, padding_mode="reflect"),
            )
            layers.append(nn.InstanceNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        return self.model(x)