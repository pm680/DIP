import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(  # input (256, 512), output (128, 256)
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.conv2 = nn.Sequential(  # input (128, 256), output (32, 64)
            nn.Conv2d(8, 32, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.conv3 = nn.Sequential(  # input (32, 64), output (8, 16)
            nn.Conv2d(32, 128, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        # Decoder (Deconvolutional Layers)
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Decoder forward pass
        x = self.convT1(x)
        x = self.convT2(x)
        output = self.convT3(x)

        return output


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(UpSampleLayer, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DownSampleLayer, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down_sample_layer_1 = DownSampleLayer(3, 64)
        self.down_sample_layer_2 = DownSampleLayer(64, 128)
        self.down_sample_layer_3 = DownSampleLayer(128, 256)
        self.down_sample_layer_4 = DownSampleLayer(256, 256)
        self.down_sample_layer_5 = DownSampleLayer(256, 256)
        self.down_sample_layer_6 = DownSampleLayer(256, 256)
        self.bottle_neck = DownSampleLayer(256, 256)
        self.up_sample_layer_1 = UpSampleLayer(256, 256)
        self.up_sample_layer_2 = UpSampleLayer(512, 256)
        self.up_sample_layer_3 = UpSampleLayer(512, 256)
        self.up_sample_layer_4 = UpSampleLayer(512, 128)
        self.up_sample_layer_5 = UpSampleLayer(384, 64)
        self.up_sample_layer_6 = UpSampleLayer(192, 64)
        self.final_layer = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x1 = self.down_sample_layer_1(x)
        x2 = self.down_sample_layer_2(x1)
        x3 = self.down_sample_layer_3(x2)
        x4 = self.down_sample_layer_4(x3)
        x5 = self.down_sample_layer_5(x4)
        x6 = self.down_sample_layer_6(x5)
        bottleneck = self.bottle_neck(x6)
        u1 = self.up_sample_layer_1(bottleneck)
        u2 = self.up_sample_layer_2(torch.cat([u1, x6], dim=1))
        u3 = self.up_sample_layer_3(torch.cat([u2, x5], dim=1))
        u4 = self.up_sample_layer_4(torch.cat([u3, x4], dim=1))
        u5 = self.up_sample_layer_5(torch.cat([u4, x3], dim=1))
        u6 = self.up_sample_layer_6(torch.cat([u5, x2], dim=1))
        out = self.final_layer(torch.cat([u6, x1], dim=1))
        return torch.tanh(out)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downSampleLayers = nn.Sequential(
            DownSampleLayer(6, 32),
            DownSampleLayer(32, 64),
        )
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn = nn.BatchNorm2d(128)
        self.last = nn.Conv2d(128, 1, kernel_size=3)

    def forward(self, anno, img):
        x = torch.cat([anno, img], dim=1)
        x = self.downSampleLayers(x)
        x = F.dropout2d(self.bn(F.leaky_relu_(self.conv1(x))))
        x = self.last(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return torch.sigmoid(x)