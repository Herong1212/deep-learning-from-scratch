# coding: utf-8
import sys, os

# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"ROOT_DIR = ", ROOT_DIR)  # /root/private/fishbook/deep-learning-from-scratch
sys.path.append(ROOT_DIR)

from common.functions import *
from common.gradient import numerical_gradient
import numpy as np


class TwoLayerNet:

    # def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    #     """
    #     初始化神经网络的权重和偏置参数。

    #     参数:
    #         input_size (int): 输入层的大小（特征数量）。
    #         hidden_size (int): 隐藏层的大小（神经元数量）。
    #         output_size (int): 输出层的大小（类别数量或输出维度）。
    #         weight_init_std (float, optional): 权重初始化的标准差，默认值为 0.01。

    #     说明:
    #         该函数用于初始化一个两层神经网络的参数，包括输入层--隐藏层、隐藏层--输出层的权重矩阵和对应的偏置向量。
    #         权重使用高斯分布进行初始化，偏置初始化为零。
    #     """
    #     # 初始化参数字典
    #     self.params = {}
    #     # 输入层到隐藏层的权重矩阵初始化
    #     self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
    #     # 输入层到隐藏层的偏置向量初始化
    #     self.params["b1"] = np.zeros(hidden_size)
    #     # 隐藏层到输出层的权重矩阵初始化
    #     self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
    #     # 隐藏层到输出层的偏置向量初始化
    #     self.params["b2"] = np.zeros(output_size)

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化参数字典
        self.params = {}

        # 设定权重及偏置参数
        # ? weight_init_std：权重初始化的标准差。作用（打破对称性）———— 如果将所有权重统一初始化为相同的值（例如全部设为 0），在反向传播计算梯度时，隐藏层中所有神经元接收到的误差信号将完全一致。
        # ? 这将导致所有神经元更新出完全相同的权重，使得整个隐藏层等同于只有一个神经元，多层网络将失去提取复杂特征的意义。
        # np.random.randn 会生成标准正态分布（均值为 0，标准差为 1）的随机数。乘以 0.01 是将这些随机数按比例缩小，使其集中在 0 附近，但又不等于 0。
        # ! 是否必须：必须引入随机性。至于是否必须是 0.01，则不一定。
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        """
        对输入数据进行前向传播预测，返回输出层的概率分布。

        参数:
            x (numpy.ndarray): 输入数据，形状为 (样本数, 特征数)。

        返回:
            numpy.ndarray: 输出层的概率分布，形状为 (样本数, 类别数)。
        """
        # 获取网络参数：权重和偏置
        W1, W2 = self.params["W1"], self.params["W2"]
        # print(f"W1.shape = ", W1.shape)
        # print(f"W2.shape = ", W2.shape)
        b1, b2 = self.params["b1"], self.params["b2"]
        # print(f"b1.shape = ", b1.shape)
        # print(f"b2.shape = ", b2.shape)

        # print(f"x_predict.shape = ", x.shape)

        # 第一层线性变换和激活函数
        a1 = np.dot(x, W1) + b1
        # print(f"a1.shape = ", a1.shape)
        z1 = sigmoid(a1)
        # print(f"z1.shape = ", z1.shape)

        # 第二层线性变换和 softmax 输出
        # ! a2 是未经过 Softmax 处理的原始预测分数
        # 在深度学习的规范术语中，网络最后一层未经激活函数映射的线性输出结果，通常被称为 Logits（逻辑值）。它是一个实数数组，其元素可以为正数、负数，且整体求和没有任何约束（并不等于 1）。
        a2 = np.dot(z1, W2) + b2
        # print(f"a2 = ", a2)  # 此时输出结果中有正有负
        # print(f"a2.shape = ", a2.shape)
        # ! y 是经过 Softmax 严格处理后的数据，此时所有单个元素的值均严格处于 (0, 1) 区间内，且这 10 个元素的累加和严格等于 1.0。
        y = softmax(a2)
        # print(f"y.shape = ", y.shape)

        return y

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        """
        计算模型的损失值

        参数:
          x: 输入数据，通常是一个 numpy 数组或类似结构，表示模型的输入特征，如 (100, 784)
          t: 监督数据（真实标签），通常是一个 numpy 数组或类似结构，表示对应的真实输出，如 (100, 10)

        返回值:
          模型预测结果与真实标签之间的交叉熵误差值
        """
        # 使用模型对输入数据进行预测
        y = self.predict(x)

        # print(f"y.shape = ", y.shape)
        # print(f"t.shape = ", t.shape)

        # 计算并返回交叉熵误差
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """
        计算模型预测结果的准确率。

        参数:
            x (numpy.ndarray): 输入数据，形状为 (样本数, 特征数)。
            t (numpy.ndarray): 真实标签，通常为 one-hot 编码格式，形状为 (样本数, 类别数)。

        返回:
            float: 模型预测的准确率，表示正确预测的样本占总样本的比例。
        """
        # 获取模型对输入数据的预测结果
        y = self.predict(x)

        # 将预测结果和真实标签从 one-hot 编码转换为类别索引
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        # 计算预测正确的样本数量，并除以总样本数得到准确率
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        """
        计算神经网络参数的数值梯度。

        参数:
            x (numpy.ndarray): 输入数据，形状为 (batch_size, input_size)。
            t (numpy.ndarray): 教师数据（标签），形状为 (batch_size, output_size)。

        返回:
            dict: 包含各参数梯度的字典，键为参数名（如 "W1", "b1" 等），
                  值为对应的梯度矩阵或向量。
        """
        # 定义损失函数关于权重的闭包函数
        loss_W = lambda W: self.loss(x, t)

        # 初始化梯度字典
        grads = {}

        # 计算各参数的数值梯度并存储到字典中
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, x, t):
        """
        计算神经网络的梯度。

        参数:
        x : numpy.ndarray
            输入数据，形状为 (batch_size, input_size)。
        t : numpy.ndarray
            目标标签，形状为 (batch_size, output_size)。

        返回:
        grads : dict
            包含各参数梯度的字典，键为参数名，值为对应的梯度矩阵。
        """
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        grads = {}

        batch_num = x.shape[0]

        # 前向传播：计算每一层的输出
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # 反向传播：计算梯度
        dy = (y - t) / batch_num
        grads["W2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis=0)

        return grads


if __name__ == "__main__":
    network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

    # print(network.params)  # params 变量中保存了该神经网络所需的全部参数
    # print(f"W1: \n", network.params["W1"])  # 每一次都是不一样的，因为是随机初始化的
    # print(f"network.params['W1'].shape = ", network.params["W1"].shape)  # (784, 100)
    # print(f"b1: \n", network.params["b1"])
    # print(f"network.params['b1'].shape = ", network.params["b1"].shape)  # (100,)
    # print(f"W2: \n", network.params["W2"])
    # print(f"network.params['W2'].shape = ", network.params["W2"].shape)  # (100, 10)
    # print(f"b2: \n", network.params["b2"])
    # print(f"network.params['b2'].shape = ", network.params["b2"].shape)  #  (10,)

    # x_batch_input = np.random.rand(100, 784)
    # x_single_input = np.random.rand(784)
    # t_single_input = np.random.rand(10)

    # print(f"x_batch_input.shape = ", x_batch_input.shape)  # (100, 784)
    # print(f"x_single_input.shape = ", x_single_input.shape)  # (784,)

    # y = network.predict(x_single_input)
    # y = network.predict(x_batch_input)
    # print(f"y.shape = ", y.shape)  # y.shape =  (10,)
    # print(f"y = ", y)

    # loss = network.loss(x_single_input, t_single_input)
    # print(f"loss = ", loss)
