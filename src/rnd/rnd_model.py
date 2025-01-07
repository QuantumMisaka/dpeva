import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import unittest

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        """
        初始化残差块。
        :param hidden_dim: 隐藏层的维度
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第二层全连接层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        """
        前向传播。
        :param x: 输入数据
        :return: 输出数据
        """
        residual = x  # 保存输入作为残差
        out = self.relu(self.fc1(x))  # 通过第一层全连接层和激活函数
        out = self.fc2(out)  # 通过第二层全连接层
        out += residual  # 添加残差连接
        out = self.relu(out)  # 再次通过激活函数
        return out

# 定义带残差连接的目标网络和预测网络
class RNDNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=240):
        """
        初始化目标网络和预测网络的架构。
        :param input_dim: 输入数据的维度
        :param output_dim: 输出数据的维度
        :param hidden_dim: 隐藏层的维度, 默认为240
        """
        super(RNDNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层全连接层
        self.residual_block = ResidualBlock(hidden_dim)  # 残差块
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 第二层全连接层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        """
        前向传播。
        :param x: 输入数据
        :return: 输出数据
        """
        x = self.fc1(x) # 通过第一层全连接层
        x = self.relu(x) # 通过激活函数
        x = self.residual_block(x)  # 通过残差块（包括激活函数）
        x = self.fc2(x)  # 通过第二层全连接层
        return x

# 单元测试
class TestRNDNetwork(unittest.TestCase):
    def test_network_output_shape(self):
        input_dim = 64
        output_dim = 32
        batch_size = 10
        model = RNDNetwork(input_dim, output_dim)
        input_tensor = torch.randn(batch_size, input_dim)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([batch_size, output_dim]))

if __name__ == "__main__":
    unittest.main()