import torch as th
from torch import nn


class NothingNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class CNN(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)

        return x


class MLP(nn.Module):
    def __init__(self, input_size, output_size, layers=(), softmax=True) -> None:
        super().__init__()

        if len(layers) == 0:
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Softmax(output_size) if softmax else nn.Sigmoid(),
            )

        else:
            layers = []
            layers.append(nn.Linear(input_size, layers[0]), nn.ReLU())

            for i in range(1, len(layers) - 1):
                layers.append(nn.Linear(layers[i - 1], layers[i]))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(layers[-1], output_size))
            layers.append(
                nn.Softmax(output_size) if softmax else nn.Sigmoid(),
            )

            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MineRLFeatureExtraction(nn.Module):
    def __init__(self, observation_space, feature_names=None) -> None:
        super().__init__()

        self.layers = {}

        if feature_names == None:
            # set some useful defaults
            feature_names = [
                "pov",
            ]

        for feature in feature_names:
            if feature == "pov":
                # add the CNN
                self.layers[feature] = CNN(observation_space[feature].shape)
            else:
                # assume this needs a MLP
                self.layers[feature]

        self.layers = nn.ModuleDict(self.layers)

    def forward(self, x):
        outputs = []

        for feature_name in self.layers:
            processed_feature = self.layers[feature_name](x[feature_name])
            outputs.append(processed_feature)

        output = th.concat(outputs)

        return output
