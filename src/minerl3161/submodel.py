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

        # TODO: modify for minerl pov
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
    def __init__(self, input_size, output_size, layers_size=(), softmax=True) -> None:
        super().__init__()

        if len(layers_size) == 0:
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Softmax(-1) if softmax else nn.ReLU(),
            )

        else:
            layers = []
            layers.append(nn.Linear(input_size, layers_size[0]))
            layers.append(nn.ReLU())

            for i in range(1, len(layers_size) - 1):
                layers.append(nn.Linear(layers_size[i - 1], layers_size[i]))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(layers_size[-1], output_size))
            layers.append(
                nn.Softmax(-1) if softmax else nn.ReLU(),
            )

            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MineRLFeatureExtraction(nn.Module):
    def __init__(
        self, observation_space, feature_names=None, mlp_hidden_size=64
    ) -> None:
        super().__init__()

        self.layers = {}

        for feature in feature_names:
            if feature == "pov":
                # add the CNN
                self.layers[feature] = CNN(observation_space[feature].shape)
            else:
                try:
                    # assume this needs a MLP
                    self.layers[feature] = MLP(
                        observation_space[feature].shape[0], mlp_hidden_size, softmax=False, layers_size=(mlp_hidden_size, mlp_hidden_size)
                    )
                except TypeError as e:
                    # we aren't equipped to handle dictionary observation spaces
                    # therefore, we'll skip this observation
                    # if you want to be able to use this observation, convert it to 
                    # an array with an ObservationWrapper
                    continue

        self.layers = nn.ModuleDict(self.layers)

    def forward(self, x):
        outputs = []

        for feature_name in self.layers:
            processed_feature = self.layers[feature_name](x[feature_name])
            outputs.append(processed_feature)

        output = th.concat(outputs, dim=1)

        return output
