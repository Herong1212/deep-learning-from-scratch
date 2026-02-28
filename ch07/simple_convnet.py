# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class SimpleConvNet:
    """简单的卷积神经网络

    结构: conv - relu - pool - affine - relu - affine - softmax

    参数说明
    ----------
    input_size : 输入大小 (MNIST 数据集为 784)
    hidden_size_list : 隐藏层神经元数量列表 (例如 [100, 100, 100])
    output_size : 输出大小 (MNIST 数据集为 10)
    activation : 激活函数类型 'relu' 或 'sigmoid'
    weight_init_std : 权重的标准差 (例如 0.01)
        如果指定 'relu' 或 'he'，则使用 “He初始化”
        如果指定 'sigmoid' 或 'xavier'，则使用 “Xavier 初始化”
    """

    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        # 初始化网络结构和参数
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = (
            input_size - filter_size + 2 * filter_pad
        ) / filter_stride + 1
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        # 权重初始化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(
            pool_output_size, hidden_size
        )
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        # 创建网络层
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param["stride"],
            conv_param["pad"],
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # 对输入数据进行前向传播并返回输出结果
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        计算损失函数值
        参数 x 是输入数据, t 是真实标签
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # 计算模型准确率
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    # 使用数值微分计算梯度
    def numerical_gradient(self, x, t):
        """
        使用数值微分计算梯度

        参数说明
        ----------
        x : 输入数据
        t : 真实标签

        返回值
        -------
        包含每层梯度的字典变量
            grads['W1']、grads['W2'] 等表示各层权重
            grads['b1']、grads['b2'] 等表示各层偏置
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads["W" + str(idx)] = numerical_gradient(
                loss_w, self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = numerical_gradient(
                loss_w, self.params["b" + str(idx)]
            )

        return grads

    # 使用误差反向传播法计算梯度
    def gradient(self, x, t):
        """
        使用误差反向传播法计算梯度

        参数说明
        ----------
        x : 输入数据
        t : 真实标签

        返回值
        -------
        包含每层梯度的字典变量
            grads['W1']、grads['W2'] 等表示各层权重
            grads['b1']、grads['b2'] 等表示各层偏置
        """
        # 前向传播
        self.loss(x, t)

        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设置梯度
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

    # 将网络参数保存到文件中
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        # 从文件中加载网络参数
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params["W" + str(i + 1)]
            self.layers[key].b = self.params["b" + str(i + 1)]
