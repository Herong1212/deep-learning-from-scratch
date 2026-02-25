# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    使用梯度下降法优化函数f的参数。

    参数:
        f (function): 需要优化的目标函数，接受一个 numpy 数组作为输入并返回一个标量值。
        init_x (numpy.ndarray): 初始参数值，表示优化的起始点。
        lr (float, optional): 学习率，控制每次更新的步长，默认值为 0.01。
        step_num (int, optional): 梯度下降的迭代次数，默认值为 100。

    返回:
        tuple: 包含两个元素：
            - x (numpy.ndarray): 优化后的参数值。
            - x_history (numpy.ndarray): 优化过程中每一步的参数值记录，形状为(step_num, len(init_x))。
    """
    x = init_x
    # 一个形状为 (step_num, x.shape) 的二维数组，包含了从起点到终点的所有坐标点序列。
    x_history = []

    # 迭代执行梯度下降步骤
    for i in range(step_num):
        # print(f"step {i+1}/{step_num}")

        # 记录当前参数值
        x_history.append(x.copy())
        # print(f"x_history: ", x_history)

        # 计算当前点的梯度
        grad = numerical_gradient(f, x)
        # 更新参数值
        x -= lr * grad
        # print(f"x_update = ", x)

    return x, np.array(x_history)


def function_2(x):
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)


init_x = np.array([-3.0, 4.0])  # shape= (2,)
# init_x = np.array([[1.0, -2.0, 3.0], [-3.0, 4.0, -5.0]]) # shape= (2,3)

lr = 0.1
step_num = 100
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
print(f"最终的 x = ", x)
print(f"最终的 x_history = ", x_history)

plt.plot([-5, 5], [0, 0], "--b")  # "--b" 表示蓝色虚线
plt.plot([0, 0], [-5, 5], "--b")

# x_history[:, 0]：提取二维数组中所有行的第 0 列元素，即构成了轨迹中所有点的 X_0 坐标集合。
# x_history[:, 1]：提取二维数组中所有行的第 1 列元素，即构成了轨迹中所有点的 X_1 坐标集合。
plt.plot(x_history[:, 0], x_history[:, 1], "o")

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")

# 图片保存到本地
plt.savefig("ch04/gradient_method.png")

# plt.show()
