# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    """
    计算函数f在点x处的数值微分(中心差分法)。

    参数:
        f (function): 需要求导的函数，接受一个数值输入并返回一个数值输出。
        x (float): 计算导数的点。

    返回:
        float: 函数f在点x处的近似导数值。
    """
    h = 1e-4  # 0.0001, 设置微小增量h，用于计算差分
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    """
    计算一个二次函数的值。

    参数:
        x (float): 输入的自变量值。

    返回:
        float: 函数计算结果，即 0.01 * x^2 + 0.1 * x 的值。
    """
    return 0.01 * x**2 + 0.1 * x


def tangent_line(f, x):
    """
    计算函数f在点x处的切线方程，并返回该切线的函数表示。

    参数:
        f (function): 需要求切线的函数，接受一个数值输入并返回一个数值输出。
        x (float): 切点的横坐标。

    返回:
        function: 表示切线方程的函数，接受一个数值t作为输入，返回切线上对应点的纵坐标。
    """
    # 计算函数f在点x处的数值导数
    d = numerical_diff(f, x)
    print(d)  # x = 5时，中心差分出来的导数为：0.1999999999990898

    # 计算切线方程中的常数项y
    y = f(x) - d * x  # -0.24999999999544897

    # 返回切线方程的lambda函数形式
    return lambda t: d * t + y


x = np.arange(0.0, 20.0, 0.1)  # 以 0.1 为单位，生成从 0 到 20 的 20 个元素的数组 x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)

# 保存到本地
plt.savefig("ch04/gradient_1d.png")

# plt.show()
