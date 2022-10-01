import torch.nn as nn


# Convolution, batch norm, ReLu with typical parameters defaulted
class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, ks=3, stride=1, pad=1, bn=True, activ=True):
        super().__init__()
        # The main convolution
        # Use nn.Conv2d to define a convolution layer with the parameters specified
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=ks, stride=stride, pad=pad)

        # Optionally include activations
        # If BatchNorm is True, use nn.BatchNorm2d to include a Batch Normalisation
        self.batchnorm = nn.BatchNorm2d(out_planes) if bn else nn.Identity()
        
        # If activ is True, use nn.ReLU to include a ReLU
        self.relu = nn.ReLU() if activ else nn.Identity()

    def forward(self, x):
        # Run through convolution then BatchNorm then ReLU
        x = self.conv(x)
        x = self.batchnorm(x) # Will use identity function if bn was false in constructor
        x = self.relu(x)  # Will use identity function if activ was false in constructor
        return x


# ResNet Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, downsample=False):
        super(BasicBlock, self).__init__()

        # If downsample is True, we will downsample with a stride 2 convolution
        self.downsample = downsample
        if self.downsample:
            # Use the Conv class to downsample, changing the number of filters to out_planes
            self.conv1 = Conv(in_planes, out_planes, stride=2, pad=2)
            # We will also need a convolution to downsample on the skip connection to 'fix' the dimensions
            # Use the Conv class to downsample without ReLU
            self.conv_skip1 = Conv(in_planes, out_planes, stride=2, pad=2, activ=False)
        # If downsample is False, we can just use the default settings of our Conv class
        else:
            # Use the Conv class, changing the number of filters to out_planes
            self.conv1 = Conv(in_planes, out_planes)
        # The second convolution doesn't use ReLU since this is applied after the skip connection
        # Use the Conv class without ReLU
        self.conv2 = Conv(in_planes, out_planes, activ=False)
        # Define a ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Save original or downsample
        residual = self.conv_skip1(x) if self.downsample else x
        # Double convolution
        out = self.conv(x)
        out = self.conv2(x)
        # Apply skip connection and final activation
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, depths, actions_size):
        super(ResNet, self).__init__()
        self.block = block
        self.depths = depths

        # Input transition
        self.in1 = Conv(3, 64, ks=7, stride=2, pad=3)
        self.in2 = nn.MaxPool2d(2)

        # Downsample path
        self.down1 = self._make_layer(12, 24, depths[0], downsample=False)
        self.down2 = self._make_layer(24, 36, depths[1])
        self.down3 = self._make_layer(36, 48, depths[2])
        self.down4 = self._make_layer(48, actions_size, depths[3])

        # Output transition
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.flatten = nn.Flatten(start_dim=1)
        # self.fc = nn.Linear(in_features=512, out_features=num_classes)

    # Create a middle layer
    def _make_layer(self, in_channels, out_channels, depth, downsample=True):
        # Increase the number of filters
        # Change the filters and downsample with the first block
        layers = [self.block(in_channels, out_channels, downsample=downsample)]
        
        # Add a repeat the block depth - 1 times
        for _ in range(depth - 1):
            # Repeat the block without changing the filters
            layers.append(self.block(out_channels, out_channels))
        
        # Convert the layers list into a Sequential
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input
        x = self.in1(x)
        x = self.in2(x)

        # Downsample
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # Output
        # x = self.avg_pool(x)
        # x = self.flatten(x)
        # x = self.fc(x)

        return x    


def build_ResNet(action_size):
    return ResNet(BasicBlock, [2, 2, 2, 2], action_size)
