# coding: utf-8
import numpy as np


class SGD:
    """
    随机梯度下降法 (Stochastic Gradient Descent)

    该类实现了随机梯度下降优化算法，用于更新模型参数。

    参数:
        lr (float): 学习率，默认值为 0.01。学习率决定了每次参数更新的步长。
    """

    def __init__(self, lr=0.01):
        """
        初始化 SGD 优化器。

        参数:
            lr (float): 学习率，默认值为 0.01。
        """
        self.lr = lr

    def update(self, params, grads):
        """
        使用梯度更新参数。

        该方法通过减去学习率与梯度的乘积来更新参数字典中的每个参数。

        参数:
            params (dict): 包含模型参数的字典，键为参数名，值为参数值。
            grads (dict): 包含对应参数梯度的字典，键为参数名，值为梯度值。
        """
        # 遍历所有参数并根据梯度进行更新
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    实现动量随机梯度下降 (Momentum SGD) 优化算法。

    动量方法通过引入速度变量来加速梯度下降过程，减少震荡并加快收敛。

    参数:
        lr (float): 学习率，默认值为 0.01。控制每次更新的步长。
        momentum (float): 动量系数，默认值为 0.9。用于控制历史梯度对当前更新的影响程度。
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None  # 用于存储每个参数的历史速度

    def update(self, params, grads):
        """
        使用动量法更新参数。

        参数:
            params (dict): 包含模型参数的字典，键为参数名，值为对应的参数数组。
            grads (dict): 包含参数梯度的字典，键为参数名，值为对应的梯度数组。

        返回值:
            无返回值。直接修改传入的 params 字典中的参数值。
        """
        # 初始化速度变量 v，如果尚未初始化
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(
                    val
                )  # 为每个参数创建与之形状相同的零数组作为初始速度

        # 更新每个参数的速度和参数值
        for key in params.keys():
            # 计算新的速度：动量项 + 当前梯度项
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            # 更新参数值
            params[key] += self.v[key]

class Nesterov:
    """
    Nesterov's Accelerated Gradient 优化器实现。 (http://arxiv.org/abs/1212.0901)

    该类实现了 Nesterov 动量加速梯度下降算法，用于优化神经网络或其他机器学习模型的参数。

    参数:
        lr (float): 学习率，默认值为 0.01。控制每次参数更新的步长。
        momentum (float): 动量系数，默认值为 0.9。控制历史梯度对当前更新的影响程度。
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None  # 用于存储历史速度的字典

    def update(self, params, grads):
        """
        更新模型参数。
        
        使用 Nesterov 动量加速梯度下降算法更新参数。该方法首先初始化速度缓存（如果尚未初始化），
        然后根据当前梯度和历史速度计算新的参数值。
        
        参数:
            params (dict): 包含模型参数的字典，键为参数名，值为对应的参数数组。
            grads (dict): 包含参数梯度的字典，键为参数名，值为对应的梯度数组。
        """
        # 如果速度缓存未初始化，则为每个参数创建一个与之形状相同的零数组
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 遍历所有参数，使用 Nesterov 动量公式更新参数和速度
        for key in params.keys():
            # 根据Nesterov公式更新参数
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]

            # 更新速度缓存
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]


class AdaGrad:
    """
    AdaGrad 优化器类，用于自适应学习率衰减（即随着学习的进行，使学习率逐渐减小）的梯度下降算法。

    该类实现了 AdaGrad 优化算法，通过累积历史梯度的平方来调整每个参数的学习率，
    使得频繁更新的参数学习率减小，稀疏更新的参数学习率增大。

    参数:
        lr (float): 初始学习率，默认值为 0.01。
    """

    def __init__(self, lr=0.01):
        """
        初始化 AdaGrad 优化器。

        参数:
            lr (float): 初始学习率，默认值为 0.01。
        """
        self.lr = lr
        self.h = None  # 用于存储每个参数的历史梯度平方和

    def update(self, params, grads):
        """
        更新模型参数。

        根据当前梯度和历史梯度信息，使用 AdaGrad 算法更新参数。 首次调用时初始化历史梯度平方和字典。

        参数:
            params (dict): 包含模型参数的字典，键为参数名，值为参数数组。
            grads (dict): 包含梯度的字典，键为参数名，值为对应梯度数组。
        """
        # 如果是首次调用，初始化历史梯度平方和字典
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)  # 初始化为与参数相同形状的零数组

        # 对每个参数进行更新
        for key in params.keys():
            # 累积当前梯度的平方到历史梯度平方和中
            self.h[key] += grads[key] * grads[key]
            # 使用 AdaGrad 公式更新参数
            # ! 最后一行加上了微小值 1e-7。这是为了防止当 self.h[key] 中有 0 时，将 0 用作除数的情况
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# NOTE AdaGrad 的升级版，解决问题：AdaGrad 无止境地学习时，更新量就会变为 0，完全不再更新。
class RMSprop:
    """
    RMSprop 优化器类，用于更新神经网络参数。

    RMSprop 是一种自适应学习率优化算法，通过维护参数梯度的平方的指数加权移动平均来调整学习率，
    从而在训练过程中动态地适应不同的参数。

    参数:
        lr (float): 学习率，默认值为 0.01。
        decay_rate (float): 衰减率，用于计算梯度平方的指数加权移动平均，默认值为 0.99。
    """

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None  # 用于存储梯度平方的指数加权移动平均

    def update(self, params, grads):
        """
        更新参数。

        根据RMSprop算法更新参数，使用梯度和梯度平方的指数加权移动平均来调整每个参数的学习率。

        参数:
            params (dict): 包含模型参数的字典，键为参数名，值为参数数组。
            grads (dict): 包含参数梯度的字典，键为参数名，值为梯度数组。
        """
        # 初始化h字典，用于存储每个参数的梯度平方的指数加权移动平均
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 更新每个参数
        for key in params.keys():
            # 更新梯度平方的指数加权移动平均
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]

            # 使用RMSprop公式更新参数
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# NOTE 首选方法！！！Adam = Momentum + AdaGrad
# Momentum 参照小球在碗中滚动的物理规则进行移动，AdaGrad 为参数的每个元素适当地调整更新步伐。
class Adam:
    """
    Adam 优化器实现，基于论文 《Adam: A Method for Stochastic Optimization》(http://arxiv.org/abs/1412.6980v8)。

    参数:
        lr (float): 学习率，默认值为 0.001。
        beta1 (float): 一阶矩估计的指数衰减率，默认值为 0.9。
        beta2 (float): 二阶矩估计的指数衰减率，默认值为 0.999。
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr  # 参数一：学习率
        self.beta1 = beta1  # 参数二：一次 Momentum 系数
        self.beta2 = beta2  # 参数三：二次 Momentum 系数

        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        """
        更新模型参数。

        参数:
            params (dict): 模型参数字典，键为参数名，值为对应的参数数组。
            grads (dict): 梯度字典，键为参数名，值为对应的梯度数组。
        """
        # 初始化动量项和二阶矩项
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        # 迭代次数加 1
        self.iter += 1

        # 计算修正后的学习率
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        # 更新参数
        for key in params.keys():
            # 更新一阶矩估计（动量项）
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])

            # 更新二阶矩估计（平方梯度的指数移动平均）
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            # 使用修正后的一阶矩和二阶矩更新参数
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
