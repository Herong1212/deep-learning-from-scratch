# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    """
    计算函数f在点x处的数值梯度 (无批量处理版本，即一维数组的梯度计算)。

    参数:
        f (callable): 需要求梯度的目标函数，接受一个 numpy 数组作为输入并返回一个标量值。
        x (numpy.ndarray): 输入变量，表示函数 f 的自变量点。

    返回:
        numpy.ndarray: 函数 f(x) 在点 x 处的梯度向量，形状与 x 相同。
    """
    h = 1e-4  # 设置微小增量h用于数值微分
    # print(f"x.shape = ", x.shape)  # x.shape =  (2,)
    # print(f"x = ", x)  # x =  [-2. -2.]
    grad = np.zeros_like(x)  # 初始化梯度数组，形状与x一致

    # 遍历输入变量x的每一个维度值，计算对应的偏导数
    for idx in range(x.size):
        tmp_val = x[idx]  # 备份当前维度的原始值，类似 C++ 里的 temp 变量，-2.0

        # f(x+h)的计算

        # NOTE！只有第0个元素变了，后面7个没变！
        x[idx] = float(tmp_val) + h  # x 变成了 [1.0001, 2.0, 3.0, ..., 8.0]

        # 把整个新数组传进去算函数值。
        # fxh1 ≈ 1.0001^2 + 2.0^2 + ... = 7.99960001
        fxh1 = f(x)  # 计算 f(x + h)

        # f(x-h)的计算
        x[idx] = tmp_val - h  # 将当前维度减少h

        # fxh2 ≈ 0.9999^2 + 2.0^2 + ...
        fxh2 = f(x)  # 计算f(x - h)

        # 使用【中心差分】公式计算当前维度的偏导数
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # ! 如果不还原，x[0] 就永远留在 0.9999 了。等到下一次循环算 x[1] 的时候，x[0] 就不是原始值了，算出来的结果就是错的！这叫消除副作用。
        x[idx] = tmp_val  # 恢复当前维度的原始值

    return grad


def numerical_gradient(f, X):
    """
    计算函数f在点X处的数值梯度。

    参数:
        f (callable): 需要求梯度的目标函数，接受一个 numpy 数组作为输入并返回一个标量。
        X (numpy.ndarray): 输入点，可以是一维或二维数组。如果是一维数组，则计算单个点的梯度；
                          如果是二维数组，则对每一行（即每个样本）分别计算梯度。

    返回:
        numpy.ndarray: 与 X 形状相同的数组，表示 f 在 X 处的梯度。如果 X 是一维数组，返回一维梯度；
                       如果 X 是二维数组，返回二维梯度，每行对应一个样本的梯度。
    """

    # print(f"X.shape = ", X.shape)  # X.shape =  (324, 2)
    # print(f"X = ", X)

    # 如果 X 是一维数组，直接调用 _numerical_gradient_no_batch() 计算梯度
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        # 初始化与 X 形状相同的梯度数组
        grad = np.zeros_like(X)  # 生成一个形状和 x 相同、所有元素都为 0 的数组
        print(f"grad_n_grad.shape = ", grad.shape)  # grad_n_grad.shape =  (324, 2)

        # 对 X 中的每一行（样本）分别计算梯度
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    """
    计算输入数组的平方和。

    参数:
        x (numpy.ndarray): 输入的数组。可以是一维或二维数组。

    返回:
        numpy.ndarray 或 float:
            - 如果输入是一维数组，返回一个标量，表示所有元素的平方和。
            - 如果输入是二维数组，返回一个一维数组，表示每行元素的平方和。
    """
    # 检查输入数组的维度
    if x.ndim == 1:
        # 对于一维数组，计算所有元素的平方和并返回标量结果
        return np.sum(x**2)
    else:
        # 对于多维数组，沿轴1（行）计算每行元素的平方和并返回一维数组
        # return x[0] ** 2 + x[1] ** 2
        # axis=1 计算指令：沿水平方向向右计算，将每一行中的各列元素相加。
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    """
    计算函数f在点x处的切线方程，并返回该切线的函数表示。

    参数:
        f (function): 需要求切线的目标函数，接受一个数值输入并返回一个数值输出。
        x (float): 切点的横坐标值。

    返回:
        function: 表示切线方程的lambda函数，形式为 t -> d * t + y，
                  其中d是函数f在点x处的导数（斜率），y是切点的纵坐标调整值。
    """
    # 计算函数f在点x处的数值梯度（近似导数）
    d = numerical_gradient(f, x)
    print(d)

    # 计算切点的纵坐标调整值y，使得切线经过点(x, f(x))
    y = f(x) - d * x

    # 返回切线方程的函数表示
    return lambda t: d * t + y


if __name__ == "__main__":
    x0 = np.arange(-2, 2.5, 0.25)
    print(f"x0.shape = ", x0.shape)  # x0.shape =  (18,)
    print(f"x0 = ", x0)
    # x0 =  [-2. -1.75 -1.5 -1.25 -1. -0.75 -0.5 -0.25 0. 0.25 0.5 0.75 1. 1.25 1.5 1.75 2. 2.25]
    x1 = np.arange(-2, 2.5, 0.25)
    print(f"x1.shape = ", x1.shape)  # x1.shape =  (18,)
    print(f"x1 = ", x1)
    # x1 =  [-2. -1.75 -1.5 -1.25 -1. -0.75 -0.5 -0.25 0. 0.25 0.5 0.75 1. 1.25 1.5 1.75 2. 2.25]

    X, Y = np.meshgrid(x0, x1)  # ? 生成网格矩阵？
    print(f"X.shape = ", X.shape)  # X.shape =  (18, 18)
    print(f"Y.shape = ", Y.shape)  # Y.shape =  (18, 18)

    X = X.flatten()
    print(f"X.shape = ", X.shape)  # X.shape =  (324,)
    print(f"X = ", X)
    Y = Y.flatten()
    print(f"Y.shape = ", Y.shape)  # Y.shape =  (324,)
    print(f"Y = ", Y)

    grad = numerical_gradient(function_2, np.array([X, Y]).T).T
    print(f"grad_main.shape = ", grad.shape)  # grad_main.shape =  (2, 324)

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid()
    plt.draw()

    # 保存图片
    plt.savefig("ch04/gradient_2d.png")

    # plt.show()
