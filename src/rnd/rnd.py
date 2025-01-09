import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import unittest
from rnd.rndmodels import RNDNetwork

# 定义RND模块
class RND:
    def __init__(self, input_dim, output_dim, distance_metric="mse", 
                 lr=1e-3, use_normalization=True):
        """
        初始化RND模块。
        :param input_dim: 输入数据的维度
        :param output_dim: 输出数据的维度
        :param distance_metric: 距离度量方案 ("mse", "kld", "cossim", "ce")
        :param lr: 学习率
        :param use_normalization: 是否开启动态归一化
        """
        self.target_network = RNDNetwork(input_dim, output_dim)  # 目标网络
        for param in self.target_network.parameters():
            param.requires_grad = False
            # 冻结目标网络的参数
        self.predictor_network = RNDNetwork(input_dim, output_dim)  # 预测网络
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=lr)  # 优化器
        self.distance_metric = distance_metric  # 距离度量方案
        self.loss_fn = self._get_loss_fn(distance_metric)  # 损失函数
        # 是否开启动态归一化
        '''
        动态归一化方法在强化学习中具有显著优势，尤其是在实时性、内存效率、计算效率和适应性方面。通过增量式更新奖励的均值和标准差，动态归一化能够实时提供准确的归一化奖励信号，确保代理的策略更新稳定且高效。相比之下，全局归一化方法在这些方面存在明显不足，因此在实际应用中通常选择动态归一化。
        '''
        self.use_normalization = use_normalization
        # 奖励归一化参数，采用动态归一化方法
        self.intrinsic_reward_mean = 0.0  # 内在奖励的均值
        self.intrinsic_reward_std = 1.0  # 内在奖励的标准差
        self.intrinsic_reward_count = 0  # 内在奖励的计数
        
    def _update_reward_normalization(self, intrinsic_reward):
        """
        更新内在奖励的归一化参数。
        :param intrinsic_reward: 当前批次的内在奖励
        """
        self.intrinsic_reward_count += 1
        delta = intrinsic_reward - self.intrinsic_reward_mean
        self.intrinsic_reward_mean += delta / self.intrinsic_reward_count
        delta2 = intrinsic_reward - self.intrinsic_reward_mean
        self.intrinsic_reward_std += delta * delta2
        
    def _normalize_reward(self, intrinsic_reward):
        """
        归一化内在奖励。
        :param intrinsic_reward: 原始内在奖励
        :return: 归一化后的内在奖励
        """
        if self.use_normalization:
            # 更新归一化参数
            self._update_reward_normalization(intrinsic_reward)
            # 归一化奖励
            normalized_intrinsic_reward = (intrinsic_reward - self.intrinsic_reward_mean) / (self.intrinsic_reward_std + 1e-8)
            return normalized_intrinsic_reward
        else:
            # 不进行归一化，直接返回原始奖励
            return intrinsic_reward

    def _get_loss_fn(self, distance_metric):
        """
        根据距离度量方案选择损失函数。
        :param distance_metric: 距离度量方案
        :return: 对应的损失函数
        """
        if distance_metric == "kld":
            softmax = nn.Softmax(dim=-1)
            log_softmax = nn.LogSoftmax(dim=-1)
            kld_novelty_func = lambda target, pred: torch.sum(softmax(target) * (log_softmax(target) - log_softmax(pred)), dim=-1)
            return kld_novelty_func
        elif distance_metric == "mse":
            return nn.MSELoss(reduction='none')
        elif distance_metric == "cossim":
            return lambda target, pred: -(nn.CosineSimilarity(dim=-1)(target, pred) - 1)  # 修复维度问题
        elif distance_metric == "ce":
            softmax = nn.Softmax(dim=-1)
            log_softmax = nn.LogSoftmax(dim=-1)
            return lambda target, pred: -torch.sum(softmax(target) * log_softmax(pred), dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    def get_intrinsic_reward(self, state):
        """
        计算内在奖励。
        :param state: 输入状态
        :return: 内在奖励值
        """
        state = torch.FloatTensor(state).unsqueeze(0)  # 将输入数据转换为张量并增加批次维度
        target_output = self.target_network(state).detach()  # 目标网络的输出
        predictor_output = self.predictor_network(state)  # 预测网络的输出
        if self.distance_metric == "mse":
            intrinsic_reward = torch.sum(self.loss_fn(predictor_output, target_output), dim=-1).item()
        else:
            intrinsic_reward = torch.mean(self.loss_fn(target_output, predictor_output)).item()
        # 返回归一化或原始奖励
        return self._normalize_reward(intrinsic_reward)

    def update_predictor(self, state):
        """
        更新预测网络。
        :param state: 输入状态
        :return: 当前批次的损失值
        """
        state = torch.FloatTensor(state).unsqueeze(0)  # 将输入数据转换为张量并增加批次维度
        target_output = self.target_network(state).detach()  # 目标网络的输出
        predictor_output = self.predictor_network(state)  # 预测网络的输出
        if self.distance_metric == "mse":
            loss = torch.mean(self.loss_fn(predictor_output, target_output))
        else:
            loss = torch.mean(self.loss_fn(target_output, predictor_output))
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        return loss.item()  # 返回当前批次的损失值

# 单元测试
class TestRND(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.input_dim = 10
        self.output_dim = 8
        self.rnd = RND(self.input_dim, self.output_dim, distance_metric="mse")

    def test_rnd_network_forward(self):
        """测试 RNDNetwork 的前向传播"""
        network = RNDNetwork(self.input_dim, self.output_dim)
        input_data = torch.randn(1, self.input_dim)
        output = network(input_data)
        self.assertEqual(output.shape, (1, self.output_dim))

    def test_rnd_initialization(self):
        """测试 RND 模块的初始化"""
        self.assertIsInstance(self.rnd.target_network, RNDNetwork)
        self.assertIsInstance(self.rnd.predictor_network, RNDNetwork)
        self.assertEqual(self.rnd.distance_metric, "mse")

    def test_rnd_intrinsic_reward(self):
        """测试 RND 模块的内在奖励计算"""
        state = np.random.rand(self.input_dim)
        reward = self.rnd.get_intrinsic_reward(state)
        self.assertIsInstance(reward, float)

    def test_rnd_update_predictor(self):
        """测试 RND 模块的预测网络更新"""
        state = np.random.rand(self.input_dim)
        initial_output = self.rnd.predictor_network(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
        self.rnd.update_predictor(state)
        updated_output = self.rnd.predictor_network(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
        self.assertFalse(np.array_equal(initial_output, updated_output))

    def test_distance_metrics(self):
        """测试距离度量方案"""
        target = torch.tensor([[1.0, 5.0, 3.0]])
        pred = torch.tensor([[3.0, 2.0, 1.0]])  # 调整 pred 的值，使其与 target 差异更大

        # MSE
        mse_loss = nn.MSELoss(reduction='none')
        mse_result = torch.sum(mse_loss(pred, target), dim=-1).item()
        self.assertAlmostEqual(mse_result, 17.0, places=5)  # (1.0-3.0)^2 + (5.0-2.0)^2 + (3.0-1.0)^2 = 0.75

        # KLD
        softmax = nn.Softmax(dim=-1)
        log_softmax = nn.LogSoftmax(dim=-1)
        kld_result = torch.sum(softmax(target) * (log_softmax(target) - log_softmax(pred)), dim=-1).item()
        self.assertGreater(kld_result, 0)  # KLD 应该大于 0

        # Cosine Similarity
        cos_sim = nn.CosineSimilarity(dim=-1)
        cos_sim_result = -(cos_sim(target, pred) - 1).item()
        self.assertGreater(cos_sim_result, 0)  # Cosine Similarity 差异应该大于 0

        # Cross-Entropy
        ce_result = -torch.sum(softmax(target) * log_softmax(pred), dim=-1).item()
        self.assertGreater(ce_result, 0)  # Cross-Entropy 应该大于 0



if __name__ == "__main__":
    unittest.main()