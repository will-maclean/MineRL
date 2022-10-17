from typing import Tuple

import torch as th
from torch import nn

from .resnet import build_ResNet


class NothingNet(nn.Module):
    """
    A module which can be used a placeholder of a Network is required, but we want nothing to occur in the forward pass.
    """

    def __init__(self) -> None:
        """
        Initialiser for NothingNet
        """
        super().__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Defines the forward pass of the model

        Args:
            x (th.Tensor): state to pass forward

        Returns:
            th.Tensor: model output
        """
        return x


class CNN(nn.Module):
    """
    Contains the implementation for the Convolutional Nerual Network being used to extract the pov features
    from the state space.
    """

    def __init__(self, input_shape: Tuple[int]) -> None:
        """
        Initialiser for CNN

        Args:
            input_shape (Dict[str, Tuple[int]]): state shape to be used
        """
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
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Defines the forward pass of the model

        Args:
            x (th.Tensor): state to pass forward

        Returns:
            th.Tensor: model output
        """
        x = self.cnn(x)
        x = x.flatten(1)

        return x


class MLP(nn.Module):
    """
    This class implements a configurable Multi Layer Perceptron, which is used as a building block for other networks.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        layers_size: Tuple[int] = (), 
        softmax: bool = True
    ) -> None:
        """
        Initialiser for MLP

        Args:
            input_size (int): the input size of the MLP
            output_size (int): the output size of the MLP
            layers_size (Tuple[int]): the layers size of the MLP
            softmax (bool): determines the activation function that should be used (either softmax or ReLU)
        """
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

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Defines the forward pass of the model

        Args:
            x (th.Tensor): state to pass forward

        Returns:
            th.Tensor: model output
        """
        return self.layers(x)


class MineRLFeatureExtraction(nn.Module):
    """
    TODO: description and init docstring, arg types
    """

    def __init__(
        self, 
        observation_space, 
        feature_names: bool = None, 
        mlp_hidden_size: int = 64
    ) -> None:
        """
        Initialiser for MineRLFeatureExtraction

        Args:
            observation_space (): 
            feature_names (bool):  the hyperparameters being used internally in this class
            mlp_hidden_size (int): the size of the network's hidden layer/s
        """
        super().__init__()

        self.layers = {}

        for feature in feature_names:
            if feature == "pov":
                # add the Resnet/CNN
                sample_input = th.rand((1, *(observation_space[feature].shape)))
                # self.layers[feature] = build_ResNet(sample_input=sample_input, n_output=mlp_hidden_size)
                self.layers[feature] = CNN(observation_space[feature].shape)

            elif feature == "compass":
                # we don't want to do any processing on the compass observation
                self.layers[feature] = NothingNet()
            
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

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Defines the forward pass of the model

        Args:
            x (th.Tensor): state to pass forward

        Returns:
            th.Tensor: model output
        """
        outputs = []

        for feature_name in self.layers:
            processed_feature = self.layers[feature_name](x[feature_name])
            outputs.append(processed_feature)

        output = th.concat(outputs, dim=1)

        return output
