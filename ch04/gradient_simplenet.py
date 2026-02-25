# coding: utf-8
import sys, os

# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"ROOT_DIR = ", ROOT_DIR)  # /root/private/fishbook/deep-learning-from-scratch
sys.path.append(ROOT_DIR)

import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 用高斯分布进行初始化

    def predict(self, x):
        """
        计算输入数据x与权重矩阵 W 的点积，用于预测输出结果。

        参数:
            x (numpy.ndarray): 输入数据，形状为(n_samples, n_features)，其中 n_samples 是样本数量，
                               n_features是特征数量。

        返回:
            numpy.ndarray: 预测结果，形状为(n_samples, n_outputs)，其中 n_outputs 是输出维度。
        """
        return np.dot(x, self.W)

    def loss(self, x, t):
        """
        计算模型的损失值。

        参数:
            x: 输入数据，通常是一个批次的样本。
            t: 真实标签，通常是一个批次的标签。

        返回:
            loss: 模型的损失值，用于衡量预测结果与真实标签之间的差异。
        """
        # 预测输入数据的输出
        z = self.predict(x)
        # 对预测结果应用softmax函数，得到概率分布
        y = softmax(z)
        # 计算交叉熵误差，衡量预测概率分布与真实标签之间的差异
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])  # 输入数据，shape = (2,)
t = np.array([0, 0, 1])  # 正确解标签，shape = (3,)

net = simpleNet()
print(f"W: \n", net.W)  # shape = (2, 3)
# [
#     [-1.10562569 -0.39181726  0.8030943 ]
#     [-0.10365085 -1.46645678 -1.5295111 ]
# ]

p = net.predict(x)
print(f"p: \n", p)
# [-2.45940602  0.52066243  0.88623259]

print(np.argmax(p))  # 最大值的索引，0、1 和 2 中的一个

# f = lambda w: net.loss(x, t)  # ! 等价于下面的函数，尽量不要用！


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)

print(f"dW: \n", dW)  # shape = (2, 3)，同 W
# [
#     [ 0.2584889   0.11635111 -0.37484001]
#     [ 0.38773334  0.17452667 -0.56226001]
# ]
