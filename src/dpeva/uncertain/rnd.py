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
import os

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
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
        :param distance_metric: Distance metric ("mse", "kld", "cossim",), default is "cossim"
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

    def _prepare_state_to_device(self, state):
        """
        Prepare state to torch.FloatTensor with device
        :param state: np.ndarray or torch.Tensor
        :return state: torch.Tensor in self.device
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(state)}. Expected np.ndarray or torch.Tensor.")
        return state

    def _get_loss_fn(self, distance_metric):
        """
        Select the loss function based on the distance metric.
        :param distance_metric: Distance metric
        :return: Corresponding loss function
        """
        if distance_metric == "kld":
            # the loss of kld is the cross entropy between the target and the predictor
            softmax = nn.Softmax(dim=-1)
            log_softmax = nn.LogSoftmax(dim=-1)
            kld_novelty_func = lambda target, pred: torch.sum(softmax(target) * (log_softmax(target) - log_softmax(pred)), dim=-1)
            return kld_novelty_func
        elif distance_metric == "mse":
            return nn.MSELoss(reduction='none')
        elif distance_metric == "cossim":
            return lambda target, pred: -(nn.CosineSimilarity(dim=-1)(target, pred) - 1) 
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")


    def get_intrinsic_reward(self, state):
        """
        Calculate the intrinsic reward.
        :param state: Input state
        :return: Intrinsic reward value
        """
        target_output = self.target_network(state).detach()  # Output of the target network
        predictor_output = self.predictor_network(state)  # Output of the predictor network
        if self.distance_metric == "mse":
            intrinsic_reward = torch.sum(self.loss_fn(predictor_output, target_output), dim=-1).item()
        else:
            intrinsic_reward = torch.mean(self.loss_fn(target_output, predictor_output)).item()
        return intrinsic_reward
    
    
    def eval_intrinsic_rewards(self, target_vector, batch_size=2048, disp_freq=1):
        """
        Calculate the intrinsic rewards for the target vector.
        :param target_vector: Target data vector, shape (num_samples, input_dim)
        :param batch_size: Batch size, default is 2048
        """
        logger.info(f"Calculating intrinsic rewards for size {len(target_vector)} with batch size {batch_size}")
        intrinsic_rewards = torch.zeros(len(target_vector), device=self.device)
        target_vector = self._prepare_state_to_device(target_vector)
        disped_flag = True
        num_batches = len(target_vector) // batch_size + 1
        for i in range(0, len(target_vector), batch_size):
            if disped_flag:
                batch_start_time = time.perf_counter()
                disped_flag = False
            batch_now = i // batch_size + 1
            logger.info(f"Calculating intrinsic rewards for batch {batch_now}/{num_batches}")
            batch = target_vector[i:i + batch_size]  
            batch_rewards = self.get_intrinsic_reward(batch) 
            intrinsic_rewards[i * batch_size:(i + 1) * batch_size] = batch_rewards
            if batch_now % disp_freq == 0:
                batch_time = time.perf_counter() - batch_start_time
                logger.info(
                    f"Batch {batch_now}/{num_batches} completed, "
                    f"Time: {batch_time:.2f}s, ")
                disped_flag = True
        logger.info("Intrinsic rewards calculation done")
        return intrinsic_rewards.detach().cpu().numpy()
    
    def eval_intrinsic_rewards_forloop(self, target_vector, batch_size=2048, disp_freq=1):
        """
        Calculate the intrinsic rewards for the target vector by for-loop format.
        :param target_vector: Target data vector, shape (num_samples, input_dim)
        :param batch_size: Batch size, default is 2048
        """
        # need more test, seems for-loop and batch-in have different output ?
        logger.info(f"Calculating intrinsic rewards for size {len(target_vector)} with batch size {batch_size}")
        intrinsic_rewards = []
        disped_flag = True
        for i in range(0, len(target_vector), batch_size):
            num_batches = len(target_vector) // batch_size + 1
            if disped_flag:
                batch_start_time = time.perf_counter()
                disped_flag = False
            batch_now = i // batch_size + 1
            logger.info(f"Calculating intrinsic rewards for batch {batch_now}/{num_batches}")
            batch = target_vector[i:i + batch_size]
            batch = self._prepare_state_to_device(batch)
            batch_rewards = [self.get_intrinsic_reward(state) for state in batch]  
            intrinsic_rewards.extend(batch_rewards)
            if batch_now % disp_freq == 0:
                batch_time = time.perf_counter() - batch_start_time
                logger.info(
                    f"Batch {batch_now}/{num_batches} completed, "
                    f"Time: {batch_time:.2f}s, ")
                disped_flag = True
        intrinsic_rewards = np.array(intrinsic_rewards)
        logger.info(f"Intrinsic rewards calculation done")
        return intrinsic_rewards


    def update_predictor(self, state):
        """
        Update the predictor network.
        :param state: Input state
        :return: Loss value of the current batch
        """
        state = self._prepare_state_to_device(state) # Convert input data to tensor and add batch dimension
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


    def train(self, train_data: np.ndarray, 
              num_batches=10000, 
              batch_size=4096, 
              initial_lr=1e-3, 
              gamma=0.95, 
              decay_steps=100, 
              disp_freq=500, 
              save_freq=2000, 
              save_path="./models"):
        """
        Train the RND model.
        :param train_data: Training data, shape (num_samples, input_dim)
        :param num_batches: Number of training batches, default is 1000
        :param batch_size: Batch size, default is 2048
        :param initial_lr: Initial learning rate, default is 1e-3
        :param gamma: Learning rate decay factor, default is 0.90
        :param decay_steps: Learning rate decay steps, default is 5
        :param disp_freq: Number of batches after which to display training results, default is 100
        :param save_freq: Number of batches after which to save the predictor network, default is 500
        :param save_path: Directory where the network will be saved, default is "./models"
        """
        # Create the save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        target_network_path = os.path.join(save_path, "target_network.pth")
        self.save_target_network(target_network_path)

        # Set the initial learning rate for the optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        
        train_data = self._prepare_state_to_device(train_data)
        # Start training
        # Log the start of training
        logger.info(f"Training started with {num_batches} batches and batch size {batch_size} ...")
        start_time = time.perf_counter()
        # Save the predictor network at the start of training
        predictor_network_path = os.path.join(save_path, "predictor_network.pth")
        self.save_predictor_network(predictor_network_path)

        # Start training
        total_loss = 0.0
        disped_flag = True
        for batch_idx in range(num_batches):
            if disped_flag:
                batch_start_time = time.perf_counter()  # Record the start time of the batch
                disped_flag = False
            # batch = self._prepare_state_to_device(batch)
            # Sample a batch from the training data
            batch_indices = np.random.choice(len(train_data), batch_size, replace=True)
            batch = train_data[batch_indices]
            
            # Update the predictor network and return the loss
            batch_loss = self.update_predictor(batch)
            total_loss += batch_loss

            # Display training results every `disp_freq` batches or at the 0th batch
            if batch_idx == 0:
                batch_time = time.perf_counter() - batch_start_time
                logger.info(f"Batch 0/{num_batches} trained, "
                f"Time: {batch_time:.2f}s, "
                f"Avg Loss: {batch_loss:.6f}")
            
            if (batch_idx + 1) % disp_freq == 0:
                avg_loss = total_loss / disp_freq
                batch_time = time.perf_counter()- batch_start_time
                logger.info(f"Batch {batch_idx + 1}/{num_batches} trained, "
                           f"Time: {batch_time:.2f}s, "
                           f"Avg Loss: {avg_loss:.6f}")
                total_loss = 0.0
                disped_flag = True

            # Save the predictor network every `save_freq` batches
            if (batch_idx + 1) % save_freq == 0:
                predictor_network_path = os.path.join(save_path, f"predictor_network.pth")
                self.save_predictor_network(predictor_network_path)

            # Update the learning rate
            if (batch_idx + 1) % decay_steps == 0:
                scheduler.step()
        
        # Log final state after training completes if the final batch is not displayed
        if not disped_flag:
            total_time = time.perf_counter() - start_time
            final_loss = total_loss / (num_batches % disp_freq)
            logger.info(f"Batch {num_batches} loss: {final_loss:.6f}")
            logger.info(f"Training completed. Total time: {total_time:.2f}s")

        # Save the predictor network after training
        predictor_network_path = os.path.join(save_path, "predictor_network.pth")
        self.save_predictor_network(predictor_network_path)
        logger.info(f"Training completed")

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

if __name__ == "__main__":
    unittest.main()
