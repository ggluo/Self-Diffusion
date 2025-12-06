import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SelfAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Projections
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        HW = H * W

        # Projections
        query = self.query_conv(x).view(B, -1, HW).permute(0, 2, 1)  # (B, HW, C//r)
        key = self.key_conv(x).view(B, -1, HW)                       # (B, C//r, HW)
        value = self.value_conv(x).view(B, -1, HW).permute(0, 2, 1)  # (B, HW, C)

        # Attention map
        attention = torch.bmm(query, key)                            # (B, HW, HW)
        attention = self.softmax(attention)

        # Attention output
        out = torch.bmm(attention, value)                            # (B, HW, C)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)     # (B, C, H, W)

        out = self.out_conv(out)
        out = self.gamma * out + x
        return out

class UNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        """
        U-Net bottleneck with optional attention block.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_attention (bool): Whether to include the attention block.
        """
        super(UNetBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = SelfAttentionBlock(out_channels)
    
    def forward(self, x):
        # Double convolution in bottleneck
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        return x

class ExtraDeepUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bottleneck=True, use_attention=False):
        super(ExtraDeepUNet, self).__init__()
        self.type(torch.complex64)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.bottleneck = bottleneck

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048 // factor)
        if bottleneck:
            self.bottleneck = UNetBottleneck(2048 // factor, 2048//factor, use_attention=use_attention)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128 // factor, bilinear)
        self.up5 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        if self.bottleneck:
            x6 = self.bottleneck(x6)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits
