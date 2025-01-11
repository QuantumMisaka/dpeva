"""Main module for Random Network Distillation (RND)"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import unittest
import logging
import time
from .rnd_models import RNDNetwork

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the RND module
class RandomNetworkDistillation:
    def __init__(self, input_dim, output_dim, 
                 hidden_dim=240, num_residual_blocks=1,
                 distance_metric="cossim", 
                 device='cpu'):
        """
        Initialize the RND module.
        :param input_dim: Dimension of the input data
        :param output_dim: Dimension of the output data
        :param hidden_dim: Dimension of the hidden layer, default is 240
        :param num_residual_blocks: Number of residual blocks, default is 1
        :param distance_metric: Distance metric ("mse", "kld", "cossim", "ce"), default is "cossim"
        :param use_normalization: Whether to enable dynamic normalization, default is False (may result in negative intrinsic rewards)
        :param device: Device to use for computation ('cpu' or 'cuda'), default is 'cpu'
        """
        self.device = torch.device(device)
        
        # Initialize target network, predictor network, and optimizer
        self.target_network = RNDNetwork(input_dim, output_dim, hidden_dim, num_residual_blocks).to(self.device)  # 目标网络
        for param in self.target_network.parameters():
            param.requires_grad = False # Freeze the parameters of the target network
        self.predictor_network = RNDNetwork(input_dim, output_dim, hidden_dim, num_residual_blocks).to(self.device)  # Predictor network
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=1e-3)  # Optimizer
        self.distance_metric = distance_metric # Distance metric
        self.loss_fn = self._get_loss_fn(distance_metric)  # Loss function

    def _get_loss_fn(self, distance_metric):
        """
        Select the loss function based on the distance metric.
        :param distance_metric: Distance metric
        :return: Corresponding loss function
        """
        if distance_metric == "kld":
            softmax = nn.Softmax(dim=-1)
            log_softmax = nn.LogSoftmax(dim=-1)
            kld_novelty_func = lambda target, pred: torch.sum(softmax(target) * (log_softmax(target) - log_softmax(pred)), dim=-1)
            return kld_novelty_func
        elif distance_metric == "mse":
            return nn.MSELoss(reduction='none')
        elif distance_metric == "cossim":
            return lambda target, pred: -(nn.CosineSimilarity(dim=-1)(target, pred) - 1) 
        elif distance_metric == "ce":
            # seems something wrong here
            softmax = nn.Softmax(dim=-1)
            log_softmax = nn.LogSoftmax(dim=-1)
            return lambda target, pred: -torch.sum(softmax(target) * log_softmax(pred), dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    def get_intrinsic_reward(self, state):
        """
        Calculate the intrinsic reward.
        :param state: Input state
        :return: Intrinsic reward value
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert input data to tensor and add batch dimension
        target_output = self.target_network(state).detach()  # Output of the target network
        predictor_output = self.predictor_network(state)  # Output of the predictor network
        if self.distance_metric == "mse":
            intrinsic_reward = torch.sum(self.loss_fn(predictor_output, target_output), dim=-1).item()
        else:
            intrinsic_reward = torch.mean(self.loss_fn(target_output, predictor_output)).item()
        return intrinsic_reward
    
    def eval_intrinsic_rewards(self, target_vector, batch_size=4096):
        """
        Calculate the intrinsic rewards for the target vector.
        :param target_vector: Target vector, shape (num_samples, input_dim)
        :param batch_size: Batch size, default is 4096
        """
        logger.info(f"Calculating intrinsic rewards for target vector of size {len(target_vector)} with batch size {batch_size}")
        intrinsic_rewards = []
        for i in range(0, len(target_vector), batch_size):
            logger.info(f"Calculating intrinsic rewards for batch {i // batch_size + 1}/{len(target_vector) // batch_size + 1}")
            batch = target_vector[i:i + batch_size]  
            batch_rewards = [self.get_intrinsic_reward(state) for state in batch]  
            intrinsic_rewards.extend(batch_rewards)
        intrinsic_rewards = np.array(intrinsic_rewards)
        logger.info(f"Intrinsic rewards calculation done")
        return intrinsic_rewards

    def update_predictor(self, state):
        """
        Update the predictor network.
        :param state: Input state
        :return: Loss value of the current batch
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert input data to tensor and add batch dimension
        target_output = self.target_network(state).detach()  # Output of the target network
        predictor_output = self.predictor_network(state)  # Output of the predictor network
        if self.distance_metric == "mse":
            loss = torch.mean(self.loss_fn(predictor_output, target_output))
        else:
            loss = torch.mean(self.loss_fn(target_output, predictor_output))
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update parameters
        return loss.item()  # Return the loss value of the current batch
    
    def train(self, train_data, num_epochs=40, batch_size=2048, initial_lr=1e-3, 
              gamma=0.90, loss_down_step=5,):
        """
        Train the RND model.
        :param train_data: Training data, shape (num_samples, input_dim)
        :param num_epochs: Number of training epochs, default is 200
        :param batch_size: Batch size, default is 2048
        :param initial_lr: Initial learning rate, default is 1e-3
        :param gamma: Learning rate decay factor, default is 0.90
        :param loss_down_ratio: Learning rate decay step, default is 10
        """
        
        # Set the initial learning rate for the optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr

        # Set the initial learning rate for the optimizer
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

         # Start training
        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # Record the start time of the epoch
            epoch_loss = 0.0  # Record the total loss of the current epoch

            logger.info(f"Epoch {epoch + 1}/{num_epochs} started, learning rate: {scheduler.get_last_lr()[0]:.6f}")

            # Shuffle the data
            np.random.shuffle(train_data)

             # Train in batches
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]  # Get the current batch
                batch_loss = self.update_predictor(batch)  # Update the predictor network and return the loss
                epoch_loss += batch_loss  # Accumulate batch loss

            # Calculate the average loss for the current epoch
            epoch_loss /= (len(train_data) / batch_size)
            epoch_time = time.time() - epoch_start_time  # Calculate the time taken for the epoch

            # Log the results
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed, "
                         f"Time: {epoch_time:.2f}s, "
                         f"Loss: {epoch_loss:.6f}")

            # Update the learning rate
            if epoch % loss_down_step == 0:
                scheduler.step()

    def save_predictor_network(self, path):
        """
        Save the predictor network's model parameters.
        :param path: Path to save the model
        """
        torch.save(self.predictor_network.state_dict(), path)
        logger.info(f"Predictor network saved to {path}")

    def load_predictor_network(self, path):
        """
        Load the predictor network's model parameters.
        :param path: Path to load the model
        """
        self.predictor_network.load_state_dict(torch.load(path, map_location=self.device))
        self.predictor_network.eval()
        logger.info(f"Predictor network loaded from {path}")

    def save_target_network(self, path):
        """
        Save the target network's model parameters.
        :param path: Path to save the model
        """
        torch.save(self.target_network.state_dict(), path)
        logger.info(f"Target network saved to {path}")

    def load_target_network(self, path):
        """
        Load the target network's model parameters.
        :param path: Path to load the model
        """
        self.target_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.eval()
        logger.info(f"Target network loaded from {path}")

# Unit tests
class TestRND(unittest.TestCase):
    def setUp(self):
        """Initialize the test environment"""
        self.input_dim = 10
        self.output_dim = 8
        self.rnd = RandomNetworkDistillation(self.input_dim, self.output_dim, distance_metric="mse", device='cpu')

    def test_rnd_network_forward(self):
        """Test the forward pass of RNDNetwork"""
        network = RNDNetwork(self.input_dim, self.output_dim).to(self.rnd.device)
        input_data = torch.randn(1, self.input_dim).to(self.rnd.device)
        output = network(input_data)
        self.assertEqual(output.shape, (1, self.output_dim))

    def test_rnd_initialization(self):
        """Test the initialization of the RND module"""
        self.assertIsInstance(self.rnd.target_network, RNDNetwork)
        self.assertIsInstance(self.rnd.predictor_network, RNDNetwork)
        self.assertEqual(self.rnd.distance_metric, "mse")

    def test_rnd_intrinsic_reward(self):
        """Test the intrinsic reward calculation of the RND module"""
        state = np.random.rand(self.input_dim)
        reward = self.rnd.get_intrinsic_reward(state)
        self.assertIsInstance(reward, float)

    def test_rnd_update_predictor(self):
        """Test the predictor network update of the RND module"""
        state = np.random.rand(self.input_dim)
        initial_output = self.rnd.predictor_network(torch.FloatTensor(state).unsqueeze(0).to(self.rnd.device)).detach().cpu().numpy()
        self.rnd.update_predictor(state)
        updated_output = self.rnd.predictor_network(torch.FloatTensor(state).unsqueeze(0).to(self.rnd.device)).detach().cpu().numpy()
        self.assertFalse(np.array_equal(initial_output, updated_output))

    def test_distance_metrics(self):
        """Test the distance metrics"""
        target = torch.tensor([[1.0, 5.0, 3.0]]).to(self.rnd.device)
        pred = torch.tensor([[3.0, 2.0, 1.0]]).to(self.rnd.device)  

        # MSE
        mse_loss = nn.MSELoss(reduction='none')
        mse_result = torch.sum(mse_loss(pred, target), dim=-1).item()
        self.assertAlmostEqual(mse_result, 17.0, places=5)  # (1.0-3.0)^2 + (5.0-2.0)^2 + (3.0-1.0)^2 = 0.75

        # KLD
        softmax = nn.Softmax(dim=-1)
        log_softmax = nn.LogSoftmax(dim=-1)
        kld_result = torch.sum(softmax(target) * (log_softmax(target) - log_softmax(pred)), dim=-1).item()
        self.assertGreater(kld_result, 0)  # KLD should be greater than 0

        # Cosine Similarity
        cos_sim = nn.CosineSimilarity(dim=-1)
        cos_sim_result = -(cos_sim(target, pred) - 1).item()
        self.assertGreater(cos_sim_result, 0)  # Cosine Similarity difference should be greater than 0

        # Cross-Entropy
        # Something Wrong Here
        ce_result = -torch.sum(softmax(target) * log_softmax(pred), dim=-1).item()
        self.assertGreater(ce_result, 0)  # Cross-Entropy should be greater than 0


if __name__ == "__main__":
    unittest.main()
