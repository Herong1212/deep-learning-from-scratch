# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    """
    计算输入值的Sigmoid函数值。

    Sigmoid 函数是一种常用的激活函数，将输入值映射到 (0, 1) 区间内，
    常用于神经网络中的输出层或二分类问题中。

    参数:
        x (float or array-like): 输入值，可以是单个数值或数组。

    返回:
        float or array-like: Sigmoid 函数的计算结果，与输入 x 的形状一致，但是数值映射到在 (0, 1) 范围内。
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """
    计算交叉熵误差，用于衡量模型预测结果与真实标签之间的差异。

    参数:
        y (numpy.ndarray): 模型的预测概率分布，形状为 (batch_size, num_classes)。
        t (numpy.ndarray): 真实标签，可以是 one-hot 编码形式或类别索引形式。

    返回:
        float: 交叉熵误差的平均值，用于评估模型性能。

    注意:
        - 如果输入为一维数组，则会自动调整为二维形式以适配批量处理。
        - 当标签为 one-hot 编码时，会将其转换为【类别索引】形式以简化计算。
        - 为了避免对数运算中出现无穷大问题，在计算中加入了一个极小值（1e-7）作为平滑项。
    """

    # print(f"y.shape = ", y.shape)  # y.shape =  (100, 10)
    # print(f"t.shape = ", t.shape)  # t.shape =  (100, 10)

    # print(f"y.ndim = ", y.ndim)  # y.ndim =  2
    # print(f"t.ndim = ", t.ndim)  # t.ndim =  2
    # 如果输入是一维数组，将其重塑为二维形式以便统一处理
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # print(f"y.size = ", y.size)  # y.size = 100 * 10 = 1000
    # print(f"t.size = ", t.size)  # t.size = 100 * 10 = 1000
    # 如果标签是 one-hot 向量形式，则将其转换为类别索引形式
    if t.size == y.size:
        # 两者都是 1000，这说明 t 目前是 One-Hot 格式（因为它和输出层一样大）。
        # 如果 t 已经是标签索引（如 [0, 1, 2]），它的大小应该是 3，这个条件就不成立了。
        t = t.argmax(axis=1)  # 它沿着横向（axis=1）寻找每一行的最大值 1 所在的索引。
        # print(f"t.shape = ", t.shape)  # t.shape =  (100,)

    # 获取批次大小
    batch_size = y.shape[0]
    # print(f"batch_size = ", batch_size)  # batch_size =  100

    # NOTE 监督数据是标签形式（非 one-hot 表示）时，计算交叉熵误差，并返回平均值
    # + 1e-7：防止 np.log(0)。数学上 log(0) = -inf。如果不加这个，只要有一个预测概率是 0，程序就会算出 nan 或崩溃。
    # print(np.arange(batch_size))  # 数组，0 1 2 3 4 5 ... 99
    # print(t)  # 数组，2 4 5 1 8 1 9 5 1 4 1 6 2 9 8 2 2 6 3 6 6 3 3 ... 1 6 5 4
    # print([np.arange(batch_size), t])  # 包含两个一维 NumPy 数组的列表（List）
    # print(type([np.arange(batch_size), t]))  # <class 'list'>
    # print(y[np.arange(batch_size), t])
    # print(type(y[np.arange(batch_size), t]))  # <class 'numpy.ndarray'>
    cross_entropy_error = (
        -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    )
    # print(f"cross_entropy_error = ", cross_entropy_error)

    return cross_entropy_error


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
