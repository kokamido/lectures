from torch import nn


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=8),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x
