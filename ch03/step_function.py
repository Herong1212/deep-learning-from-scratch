# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    """
    阶跃函数实现。

    参数:
        x (array-like): 输入值，可以是标量或数组。

    返回:
        numpy.ndarray: 输出值，形状与输入相同。对于输入中大于0的元素返回1，否则返回0。
    """
    return np.array(x > 0, dtype=int)


# 生成从-5.0到5.0，步长为0.1的数组作为输入数据
X = np.arange(-5.0, 5.0, 0.1)
# 对输入数据应用阶跃函数，得到对应的输出数据
Y = step_function(X)
# 绘制阶跃函数图像
plt.plot(X, Y)
# 设置y轴显示范围，确保图像清晰展示阶跃特性
plt.ylim(-0.1, 1.1)
# 显示绘制的图像
# plt.show()
# 保存图像
plt.title("step_function")
plt.savefig("step_function.png")
