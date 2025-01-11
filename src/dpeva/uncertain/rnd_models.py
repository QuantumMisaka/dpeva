"""Models for the Random Network Distillation (RND) algorithm."""

from __future__ import annotations

import torch
import torch.nn as nn
import unittest


# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        """
        Initialize the residual block.
        :param hidden_dim: Dimension of the hidden layer
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second fully connected layer
        self.relu = nn.LeakyReLU()  # Activation function

    def forward(self, x):
        """
        Forward pass.
        :param x: Input data
        :return: Output data
        """
        residual = x  # Save the input as residual
        out = self.relu(self.fc1(x))  # Pass through the first fully connected layer and activation function
        out = self.fc2(out)  # Pass through the second fully connected layer
        out += residual  # Add the residual connection
        out = self.relu(out)  # Pass through the activation function again
        return out
    
    
# Define the target and predictor networks with residual connections
class RNDNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=240, num_residual_blocks=1):
        """
        Initialize the architecture of the target and predictor networks.
        :param input_dim: Dimension of the input data
        :param output_dim: Dimension of the output data
        :param hidden_dim: Dimension of the hidden layer, default is 240
        :param num_residual_blocks: Number of residual blocks, default is 1
        """
        super(RNDNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input layer
        self.residual_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)])  # Multiple residual blocks
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.relu = nn.LeakyReLU()  # Activation function

    def forward(self, x):
        """
        Forward pass.
        :param x: Input data
        :return: Output data
        """
        x = self.fc1(x)  # Pass through the input layer
        x = self.relu(x)  # Pass through the activation function
        x = self.residual_blocks(x)  # Pass through the residual blocks (including activation function)
        x = self.fc2(x)  # Pass through the output layer, no activation function
        return x

# Unit tests
class TestRNDNetwork(unittest.TestCase):
    def test_network_output_shape(self):
        input_dim = 64
        output_dim = 32
        batch_size = 10
        model = RNDNetwork(input_dim, output_dim, hidden_dim=240, num_residual_blocks=2)
        input_tensor = torch.randn(batch_size, input_dim)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([batch_size, output_dim]))


if __name__ == "__main__":
    unittest.main()