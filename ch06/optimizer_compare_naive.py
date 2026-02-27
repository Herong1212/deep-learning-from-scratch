# coding: utf-8
import sys, os

# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    """
    计算并返回两个数值的变换结果。

    参数:
        x (float): 第一个输入数值，将被除以 10.0。
        y (float): 第二个输入数值，将被乘以 2.0。

    返回:
        tuple: 包含两个元素的元组，第一个元素是 x 除以 10.0 的结果，
               第二个元素是 y 乘以 2.0 的结果。
    """
    # 返回该位置关于 x 和 y 的偏导数值，作为后续参数更新的依据。
    return x / 10.0, 2.0 * y


init_pos = (-7.0, 2.0)  # tuple，设定了所有算法统一的优化起始坐标为 (-7.0, 2.0)
params = {}  # dict
params["x"], params["y"] = init_pos[0], init_pos[1]
print("init pos: ", params)  # 初始参数： {'x': -7.0, 'y': 2.0}
grads = {}  # dict
grads["x"], grads["y"] = 0, 0
print("init grad: ", grads)  # 初始梯度： {'x': 0, 'y': 0}

# OrderedDict 是 Python 标准库 collections 模块中的“有序字典”类。与普通字典的区别在于：它会严格记录元素被添加进去的顺序。
# 这样在后续 for key in optimizers: 循环遍历时，能保证严格按照 SGD -> Momentum -> AdaGrad -> Adam 的顺序执行和绘图。
optimizers = OrderedDict()  # dict
print("optimizers_origin: ", optimizers.keys())
# optimizers_origin: odict_keys([]) 此时未添加优化器，所以为空

# 优化器字典：实例化了四种优化器对象。这里设定的学习率（lr）数值各不相同，这是经过特意调配的超参数，目的是确保在相同的 30 次迭代限制下，每种算法都能充分展现其典型特质。
# 向这个字典中添加一个键值对（元素）。键（Key）是字符串 "SGD"；值（Value）是 SGD 这个类被实例化后的一个对象。
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)
print("optimizers_addkeys: ", optimizers.keys())
# optimizers_addkeys: odict_keys(["SGD", "Momentum", "AdaGrad", "Adam"]) 此时添加了4个优化器
for k, v in optimizers.items():
    print(k, v)
for i, v in enumerate(optimizers):
    print(i, v)

idx = 1

for key in optimizers:
    print("######################## START LOOP!!!########################")
    optimizer = optimizers[key]  # 取出每个优化器
    print("optimizer: ", key)
    x_history = []
    y_history = []

    # 每次切换优化器时，重置起点
    # ? 这里为什么要重置起点？而不需要重置梯度？答：
    params["x"], params["y"] = init_pos[0], init_pos[1]
    print("params_init: ", params["x"], params["y"])
    print("grads_init: ", grads["x"], grads["y"])

    # 执行 iter_num 次参数更新
    iter_num = 30
    for i in range(iter_num):
        print(f"----------------------step {i+1}/{iter_num}----------------------")

        x_history.append(params["x"])
        print(f"x_history: ", x_history)
        y_history.append(params["y"])
        print(f"y_history: ", y_history)

        # 计算当前坐标的梯度，并执行更新
        grads["x"], grads["y"] = df(params["x"], params["y"])
        print(f"grads_update: ", grads)  # 梯度在这里更新后，传入下面的 update() 函数
        optimizer.update(params, grads)  # 这里只有 params 更新了，而 grad 只是调用！
        print(f"params_update: ", params)

    x = np.arange(-10, 10, 0.01)  # shape = (2000,)
    y = np.arange(-5, 5, 0.01)  # shape = (1000,)

    # 在给定的区间内生成密集的坐标网格，并计算出每个网格点对应的函数值 Z
    # meshgrid 作用：接收两个一维数组，并生成一个用于绘制二维曲面或等高线图的“网格矩阵”。
    X, Y = np.meshgrid(x, y)  # X.shape=(1000, 2000), Y.shape=(1000, 2000)
    # Z 是一个矩阵，保存了网格上所有点的高度值（函数值）。
    Z = f(X, Y)  # Z.shape=(1000, 2000)

    # for simple contour line
    # mask = Z > 7 利用了 NumPy 的广播机制，生成了一个与 Z 形状完全相同的布尔型（Boolean）矩阵。在 Z 中值大于 7 的位置，mask 对应位置为 True，否则为 False。
    mask = Z > 7
    # Z[mask] = 0 是 NumPy 的布尔索引操作。它会将 Z 矩阵中所有大于 7 的数值强行修改为 0。
    Z[mask] = 0

    # plot
    # NOTE subplot 是 Matplotlib 中用于规划画布区域的函数。
    # 2, 2 表示将整个窗口划分为 2 行 2 列（共 4 个格子）的矩阵。
    # idx 代表当前选中第几个格子。当 idx=1 时，激活左上角的画板；idx=2 激活右上角。它仅仅是“分配画图区域”，并不负责画具体的线条。
    plt.subplot(2, 2, idx)  # 将这四幅图拼装成一个 2 x 2 的图像矩阵并保存
    idx += 1
    # * plot 是实际执行连线绘图的函数。
    # 它读取之前循环中保存的坐标轨迹列表（x_history, y_history），在当前 subplot 激活的格子里画线。"o-" 的意思是：在每个坐标点上画一个圆圈标记（o），并且用实线（-）把这些圆圈连起来。
    plt.plot(x_history, y_history, "o-", color="red")

    # 利用上述网格数据绘制出目标函数的地形等高线，作为背景
    # * contour 是 Matplotlib 中专门用于绘制二维等高线图的函数。
    # 它接收刚才由 meshgrid 生成的网格坐标矩阵 X 和 Y，以及对应的高度矩阵 Z，在平面上画出多条封闭的曲线。
    # 在同一条曲线上的所有点，其目标函数值 f(x,y) 均相等。这为刚才 plot 画出的红色折线路径提供了一个直观的地形背景，从而能清楚地看出优化器是如何沿着“坡度”滚落到中心的。
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)

    # 将代码第 3 步记录下来的 x_history 和 y_history 绘制为红色折线图，叠加在等高线图上，代表每种优化算法从起点向中心极小值点 (0, 0) 探索的具体路径。
    plt.plot(0, 0, "+")
    # colorbar()
    # spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")

plt.savefig("ch06/optimizer_compare_naive.png")

# plt.show()
