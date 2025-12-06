import torch.nn as nn
import torch
import math

norm = nn.BatchNorm1d
class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filter_number=16):
        """
        Args:
            in_channels (int): Number of input channels (default: 1).
            out_channels (int): Number of output channels (default: 1).
            filter_number (int): Base number of filters; scales as filter_number, 2*filter_number, 4*filter_number (default: 16).
        """
        super(UNet1D, self).__init__()
        
        # Define filter sizes based on filter_number
        enc1_filters = filter_number
        enc2_filters = 2 * filter_number
        bottleneck_filters = 4 * filter_number
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, enc1_filters, kernel_size=3, padding=1),
            norm(enc1_filters),
            nn.ReLU(),
            nn.Conv1d(enc1_filters, enc1_filters, kernel_size=3, padding=1),
            norm(enc1_filters),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(enc1_filters, enc2_filters, kernel_size=3, padding=1),
            norm(enc2_filters),
            nn.ReLU(),
            nn.Conv1d(enc2_filters, enc2_filters, kernel_size=3, padding=1),
            norm(enc2_filters),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(enc2_filters, bottleneck_filters, kernel_size=3, padding=1),
            norm(bottleneck_filters),
            nn.ReLU(),
            nn.Conv1d(bottleneck_filters, bottleneck_filters, kernel_size=3, padding=1),
            norm(bottleneck_filters),
            nn.ReLU()
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose1d(bottleneck_filters, enc2_filters, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(enc2_filters * 2, enc2_filters, kernel_size=3, padding=1),  # enc2_filters * 2 due to concatenation
            nn.ReLU(),
            nn.Conv1d(enc2_filters, enc2_filters, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose1d(enc2_filters, enc1_filters, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(enc1_filters * 2, enc1_filters, kernel_size=3, padding=1),  # enc1_filters * 2 due to concatenation
            nn.ReLU(),
            nn.Conv1d(enc1_filters, enc1_filters, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.final = nn.Conv1d(enc1_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u1 = self.up1(b)
        u1 = torch.cat([u1, e2], dim=1)  # Concatenate with enc2 output
        d1 = self.dec1(u1)
        u2 = self.up2(d1)
        u2 = torch.cat([u2, e1], dim=1)  # Concatenate with enc1 output
        d2 = self.dec2(u2)
        out = self.final(d2)
        return out

class SimpleNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filter_number=16):
        """
        Args:
            in_channels (int): Number of input channels (default: 1).
            out_channels (int): Number of output channels (default: 1).
            filter_number (int): Base number of filters for the convolutional layers (default: 16).
        """
        super(SimpleNet1D, self).__init__()
        
        # First convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, filter_number, kernel_size=3, padding=1),
            norm(filter_number),
            nn.ReLU()
        )
        
        # Second convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv1d(filter_number, filter_number, kernel_size=3, padding=1),
            norm(filter_number),
            nn.ReLU()
        )
        
        # Final convolution to match output channels
        self.final = nn.Conv1d(filter_number, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.final(x)
        return out
# Example usage
if __name__ == "__main__":
    # Default filter_number
    model = UNet1D(in_channels=1, out_channels=1, filter_number=16)
    x = torch.randn(1, 1, 64)  # Batch size 1, 1 channel, 64 length
    y = model(x)
    print(f"Output shape with filter_number=16: {y.shape}")
    print(f"Filters: enc1={16}, enc2={32}, bottleneck={64}")

    # Custom filter_number
    model_custom = UNet1D(in_channels=1, out_channels=1, filter_number=8)
    y_custom = model_custom(x)
    print(f"Output shape with filter_number=8: {y_custom.shape}")
    print(f"Filters: enc1={8}, enc2={16}, bottleneck={32}")