# coding: utf-8


# 乘法层
class MulLayer:
    """
    初始化函数，用于设置实例变量。

    实例变量:
        self.x: 用于保存正向传播时的输入值。
        self.y: 用于保存正向传播时的输入值。
    """

    def __init__(self):
        """
        初始化函数，用于设置实例变量的初始值。

        实例变量:
            x: 用于存储 x 坐标或相关数据，默认值为 None。
            y: 用于存储 y 坐标或相关数据，默认值为 None。
        """
        self.x = None
        self.y = None

    def forward(self, x, y):
        """
        执行前向传播计算，将输入 x 和 y 相乘并返回结果。

        参数:
            x: 第一个输入值，通常为张量或数值。
            y: 第二个输入值，通常为张量或数值。

        返回:
            out: x 与 y 的乘积结果。
        """
        # 保存输入 x 和 y 到实例变量中
        self.x = x
        self.y = y

        # 计算 x 与 y 的乘积
        out = x * y

        return out

    def backward(self, dout):
        """
        计算反向传播的梯度。

        参数:
            dout: 上游传来的梯度，通常是一个张量或数值，表示损失函数对当前层输出的偏导数。

        返回:
            dx: 损失函数对 self.x 的偏导数。
            dy: 损失函数对 self.y 的偏导数。
        """
        # 计算损失函数对 self.x 和 self.y 的偏导数
        dx = dout * self.y  # 翻转 x 和 y
        dy = dout * self.x

        return dx, dy


# 加法层
class AddLayer:
    def __init__(self):
        """
        初始化函数。
        
        该函数用于初始化类的实例。当前实现为空，未执行任何操作。
        """
        pass

    def forward(self, x, y):
        """
        执行前向传播计算，将两个输入张量相加。

        参数:
            x (Tensor): 第一个输入张量。
            y (Tensor): 第二个输入张量。

        返回:
            Tensor: 两个输入张量相加的结果。
        """
        # 将输入张量 x 和 y 相加
        out = x + y

        return out

    def backward(self, dout):
        """
        计算反向传播的梯度。

        参数:
            dout: 上游传来的梯度，通常是一个张量或数值，表示损失函数对当前层输出的偏导数。

        返回:
            dx: 损失函数对输入 x 的偏导数。
            dy: 损失函数对输入 y 的偏导数。
        """
        # 将上游梯度直接传递给 dx 和 dy，因为当前层的计算是线性变换（乘以1）
        dx = dout * 1
        dy = dout * 1

        return dx, dy
