import torch
from torch import nn

# 构建网络——UNet
# 下采样
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.con_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, if_maxpool=True):
        if if_maxpool:
            x = self.maxpool(x)
        x = self.con_relu(x)
        return x


# 上采样
class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.con_relu = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.con_relu(x)
        x = self.upconv_relu(x)
        return x


# 组建Unet
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = Encoder(3, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)
        self.encoder5 = Encoder(512, 1024)
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = Decoder(512)
        self.decoder2 = Decoder(256)
        self.decoder3 = Decoder(128)

        self.conDouble = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        x1 = self.encoder1(x, if_maxpool=False)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        x5 = self.upconv_relu(x5)

        x5 = torch.cat((x4, x5), dim=1)
        x5 = self.decoder1(x5)
        x5 = torch.cat((x3, x5), dim=1)
        x5 = self.decoder2(x5)
        x5 = torch.cat((x2, x5), dim=1)
        x5 = self.decoder3(x5)
        x5 = torch.cat((x1, x5), dim=1)

        x5 = self.conDouble(x5)
        x5 = self.last(x5)
        return x5