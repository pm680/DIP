import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        ### FILL: add more CONV Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=6, stride=4, padding=1),  # Input channels: 8, Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=6, stride=4, padding=1),  # Input channels: 32, Output channels: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=6, stride=4, padding=1),  # Input channels: 128, Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=6, stride=4, padding=1),  # Input channels: 32, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 3
            nn.BatchNorm2d(3),
            nn.Tanh()  # Output activation function
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Decoder forward pass
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        ### FILL: encoder-decoder forward pass

        output = x
        
        return output
    