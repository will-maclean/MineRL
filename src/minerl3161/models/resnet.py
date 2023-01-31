import torch as th
import torch.nn as nn


# Convolution, batch norm, ReLu with typical parameters defaulted
class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, ks=3, stride=1, pad=1, bn=True, activ=True):
        super().__init__()
        # The main convolution
        # Use nn.Conv2d to define a convolution layer with the parameters specified
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=ks, stride=stride, padding=pad)

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
        out = self.conv1(x)
        out = self.conv2(x)
        # Apply skip connection and final activation
        out += residual
        out = self.relu(out)
        return out

# ResNet Bottleneck Block
class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, downsample=False):
        super(Bottleneck, self).__init__()
        
        # Reduce planes in the centre of the bottleneck by a factor of 4
        reduced_planes = out_planes // 4
        
        # Reduce planes with a 1x1 convolution
        self.conv1 = Conv(in_planes, reduced_planes, ks=1, pad=0)
        
        # Downsampling uses a stride 2 conv
        self.downsample = downsample
        
        # If downsample is True, we will downsample with a stride 2 convolution
        if self.downsample:
            self.conv2 = Conv(reduced_planes, reduced_planes, stride=2, pad=2)
            
            # We will also need a convolution to downsample on the skip connection
            self.downsample = Conv(in_planes, out_planes, ks=1, stride=2, activ=None)
        
        # If downsample is False, we can just use the default settings of our Conv class
        else:
            self.conv2 = Conv(reduced_planes, reduced_planes)
        
        # Increase planes with a 1x1 convolution
        self.conv3 = Conv(reduced_planes, out_planes, ks=1, pad=0, activ=None)

        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Save original or downsample
        identity = self.downsample(x) if self.downsample else x

        # Triple convolution
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        # Apply skip connection and final activation
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, depths, actions_size, sample_input=None):
        super(ResNet, self).__init__()
        self.block = block
        self.depths = depths

        in_channels =sample_input.shape[1] if sample_input is not None else 4

        # Input transition
        self.in1 = Conv(in_channels, 12, ks=7, stride=2, pad=4)
        self.in2 = nn.MaxPool2d(2)

        # Downsample path
        self.down1 = self._make_layer(12, 12, depths[0], downsample=False)
        self.down2 = self._make_layer(12, 36, depths[1])
        self.down3 = self._make_layer(36, 48, depths[2])
        self.down4 = self._make_layer(48, 48, depths[3])

        # Output transition
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)

        if sample_input is not None:
            fc_input = self._calc_fc_input_size(sample_input)
        else:
            fc_input = 512
            

        self.fc = nn.Linear(in_features=fc_input, out_features=actions_size)
    
    def _calc_fc_input_size(self, x):
        # Input
        x = self.in1(x)
        x = self.in2(x)

        # Downsample
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # Output
        x = self.avg_pool(x)
        x = self.flatten(x)

        return x.shape[1]

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
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x    


def build_ResNet(sample_input: th.tensor, n_output):
    return ResNet(Bottleneck, [3, 3, 3, 3], n_output, sample_input=sample_input)
