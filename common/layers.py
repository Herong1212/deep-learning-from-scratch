# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(
            *self.original_x_shape
        )  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax 的输出
        self.t = None  # 监督数据（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        """
        初始化卷积层的参数和相关变量。

        参数:
            W (array-like): 卷积核权重矩阵。
            b (array-like): 偏置项。
            stride (int, optional): 卷积步长，默认为 1。
            pad (int, optional): 填充大小，默认为 0。

        属性:
            self.W: 存储卷积核权重。
            self.b: 存储偏置项。
            self.stride: 存储卷积步长。
            self.pad: 存储填充大小。
            self.x: 存储输入数据，用于反向传播。
            self.col: 存储输入数据的列展开形式，用于反向传播。
            self.col_W: 存储权重的列展开形式，用于反向传播。
            self.dW: 存储权重的梯度，用于参数更新。
            self.db: 存储偏置的梯度，用于参数更新。
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（在反向传播时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        前向传播函数，用于计算卷积层的输出。

        参数:
            x (numpy.ndarray): 输入数据，形状为 (N, C, H, W)，其中 N 是批次大小，
                                C 是通道数, H 和 W 分别是输入的高度和宽度。

        返回:
            numpy.ndarray: 卷积层的输出，形状为 (N, FN, out_h, out_w)，其中 FN 是滤波器数量，
                            out_h 和 out_w 是输出的高度和宽度。
        """
        # 获取滤波器的形状：FN（滤波器数量）、C（通道数）、FH（滤波器高度）、FW（滤波器宽度）
        FN, C, FH, FW = self.W.shape
        # 获取输入数据的形状：N（批次大小）、C（通道数）、H（输入高度）、W（输入宽度）
        N, C, H, W = x.shape
        # 计算输出特征图的高度
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        # 计算输出特征图的宽度
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        # 将输入数据转换为列格式，便于与滤波器进行矩阵乘法
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 将滤波器权重 reshape 为二维矩阵，并转置以匹配列格式数据
        col_W = self.W.reshape(FN, -1).T  # 滤波器的展开

        # 执行卷积操作：通过矩阵乘法计算输出
        out = np.dot(col, col_W) + self.b
        # 将输出 reshape 为四维张量并调整轴顺序以符合标准格式 (N, FN, out_h, out_w)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # 保存中间变量供反向传播使用
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """
        反向传播函数，用于计算卷积层的梯度。

        参数:
            dout (numpy.ndarray): 上一层传递过来的梯度，形状为 (N, FN, OH, OW)，其中 N 是批次大小, FN 是滤波器数量, OH 和 OW 分别是输出特征图的高度和宽度。

        返回:
            dx (numpy.ndarray): 输入数据 x 的梯度，形状与输入数据 x 相同。
        """
        # 获取权重矩阵 W 的形状信息：滤波器数量 FN、通道数 C、滤波器高度 FH、滤波器宽度 FW
        FN, C, FH, FW = self.W.shape

        # 将 dout 转换为适合矩阵运算的形状，并展平为二维数组
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 计算偏置项 b 的梯度 db，通过对 dout 在样本维度求和得到
        self.db = np.sum(dout, axis=0)

        # 计算权重 W 的梯度 dW：
        # 1. 使用转置后的列展开矩阵 self.col 与 dout 相乘
        # 2. 将结果重新排列为权重矩阵的原始形状
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 计算输入数据 x 的梯度 dx：
        # 1. 通过 dout 与转置后的权重矩阵 self.col_W 相乘，得到列展开形式的梯度 dcol
        # 2. 使用 col2im 函数将 dcol 转换回图像形式的梯度 dx
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        # 返回输入数据 x 的梯度
        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        """
        初始化池化层参数。

        参数:
            pool_h (int): 池化窗口的高度。
            pool_w (int): 池化窗口的宽度。
            stride (int, optional): 池化操作的步长，默认为 2。
            pad (int, optional): 输入数据的填充大小，默认为 0。

        成员变量:
            self.pool_h: 存储池化窗口的高度。
            self.pool_w: 存储池化窗口的宽度。
            self.stride: 存储池化操作的步长。
            self.pad: 存储输入数据的填充大小。
            self.x: 用于存储输入数据，在前向传播时赋值。
            self.arg_max: 用于存储最大值的位置索引，在反向传播时使用。
        """
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        """
        前向传播函数，实现池化操作。

        参数:
            x (numpy.ndarray): 输入张量，形状为 (N, C, H, W)，其中 N 是批次大小，
                                C 是通道数，H 和 W 分别是输入的高度和宽度。

        返回:
            numpy.ndarray: 池化后的输出张量，形状为 (N, C, out_h, out_w)，
                            其中 out_h 和 out_w 是池化后的高度和宽度。
        """
        # 获取输入张量的形状信息
        N, C, H, W = x.shape
        
        # 计算池化后的输出高度和宽度
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 将输入张量转换为列形式，便于进行池化操作
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 对每一列找到最大值的位置和值
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        
        # 将输出重塑为目标形状并调整维度顺序
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # 保存输入和最大值位置，用于反向传播
        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        反向传播函数，用于计算池化层的梯度。

        参数:
            dout (numpy.ndarray): 上游传来的梯度，形状为 (N, C, H, W)，其中 N 是批次大小，
                                    C 是通道数，H 和 W 分别是高度和宽度。

        返回:
            dx (numpy.ndarray): 输入数据的梯度，形状与原始输入数据相同。
        """
        # 将 dout 的维度顺序调整为 (N, H, W, C)，以便后续处理
        dout = dout.transpose(0, 2, 3, 1)

        # 计算池化窗口的大小
        pool_size = self.pool_h * self.pool_w

        # 初始化一个零矩阵 dmax，用于存储最大值位置的梯度
        dmax = np.zeros((dout.size, pool_size))

        # 根据 arg_max 中记录的最大值位置，将 dout 的梯度分配到对应位置
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        # 将 dmax 重新 reshape 为与 dout 相同的形状，并附加池化窗口的维度
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # 将 dmax 转换为列形式，便于后续的反向操作
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

        # 使用 col2im 函数将列形式的梯度转换回图像形式，得到输入数据的梯度 dx
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
